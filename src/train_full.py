"""
src/train_full.py
VLM Full Fine-Tuning — 24-Hour Production Run (Anirudh | May 9)

Launches the primary fine-tuning run on the full vlm_conversational_dataset.json
(1703 image-caption pairs) on the MIG A100 (1g.5gb) GPU slice.

Key improvements over the smoke-test (train.py):
  - Full dataset (all valid records, not a 10-sample subset)
  - 10 epochs with cosine LR schedule + linear warmup
  - Checkpoint saved every epoch to checkpoints/full_train/epoch_N/
  - Best checkpoint tracked by validation loss when validation is enabled
  - Resume-from-checkpoint support (restart from latest saved epoch)
  - Step-level train/validation loss logged to logs/full_train_loss.csv
  - Loss curve SVG written to logs/full_train_loss_curve.svg
  - Collapse detection with conservative automatic recovery/restart
  - Per-epoch summary written to results/full_training_manifest.json
  - ETA printed per epoch

Design constraints for 5 GB MIG slice:
  - fp16/bfloat16 via torch.amp.autocast
  - gradient checkpointing enabled
  - batch_size=1, gradient_accumulation_steps=8 (effective batch=8)
  - num_workers=0 (MIG slice shares host memory; avoid multiprocess overhead)
"""

import os
import json
import math
import time
import random
import csv
import shutil
from statistics import median

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from PIL import Image

# ─── Hyperparameters ──────────────────────────────────────────────────────────
SEED                  = 42
NUM_EPOCHS            = 10
BATCH_SIZE            = 1          # must stay 1 for 5 GB MIG slice
GRADIENT_ACCUMULATION = 8          # effective batch size = 8
LEARNING_RATE         = 2e-5       # lower than smoke-test for stable full-run
WARMUP_RATIO          = 0.05       # 5% of total steps used for LR warm-up
MAX_TEXT_LEN          = 128        # full captions; longer than smoke-test
MODEL_NAME            = "./models/blip/models--Salesforce--blip-image-captioning-base/snapshots/82a37760796d32b1411fe092ab5d4e227313294b"
CHECKPOINT_SUBDIR     = "checkpoints/full_train"
LOG_EVERY_N_STEPS     = 25         # write a CSV row every N global steps
VALIDATION_SPLIT      = 0.10       # deterministic holdout for validation loss
CHECKPOINT_EVERY_EPOCHS = 1
MAX_GRAD_NORM         = 1.0
WEIGHT_DECAY          = 0.01

# Collapse detection/recovery. Thresholds are conservative for BLIP caption loss:
# the historical run falls smoothly from ~8.38 to ~2.01, so only severe
# numerical instability or catastrophic regression triggers a restart.
LOSS_WINDOW_STEPS     = 50
LOSS_EXPLOSION_FACTOR = 4.0
LOSS_EXPLOSION_MIN    = 20.0
GRAD_NORM_COLLAPSE_THRESHOLD = 100.0
VAL_LOSS_COLLAPSE_FACTOR     = 2.5
VAL_LOSS_COLLAPSE_MARGIN     = 1.0
WORSENING_EPOCH_PATIENCE     = 3
WORSENING_EPOCH_MARGIN       = 0.02
MAX_RECOVERY_RESTARTS        = 2
RECOVERY_LR_FACTOR           = 0.5
RECOVERY_WARMUP_INCREMENT    = 0.05
RECOVERY_MAX_WARMUP_RATIO    = 0.20
RECOVERY_GRAD_CLIP_FACTOR    = 0.75
RECOVERY_MIN_GRAD_CLIP       = 0.25
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────

class VLMDataset(Dataset):
    """
    Full production dataset wrapping the LLaVA-style conversational JSON.

    Each record:
      {
        "id": "...",
        "image": "data/whole_multicare_dataset/vlm_mri_subset/images/...",
        "conversations": [
          {"from": "human", "value": "<image>\\nDescribe ..."},
          {"from": "gpt",   "value": "<caption text>"}
        ]
      }
    """

    def __init__(self, records: list, base_dir: str, processor: BlipProcessor):
        self.records   = records
        self.base_dir  = base_dir
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec      = self.records[idx]
        img_path = os.path.join(self.base_dir, rec["image"])
        caption  = rec["conversations"][1]["value"]

        image = Image.open(img_path).convert("RGB")

        encoding = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_TEXT_LEN,
            truncation=True,
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


# ── Helpers ───────────────────────────────────────────────────────────────────

def log_gpu(label: str):
    """Print GPU memory usage (no-op on CPU)."""
    if torch.cuda.is_available():
        alloc    = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved()  / 1e9
        print(f"    [GPU | {label}]  alloc: {alloc:.3f} GB  reserved: {reserved:.3f} GB")


def validate_records(records: list, base_dir: str) -> list:
    """Drop records whose image file is missing or unreadable."""
    valid   = []
    skipped = 0
    for rec in records:
        path = os.path.join(base_dir, rec["image"])
        try:
            with Image.open(path) as im:
                im.verify()
            valid.append(rec)
        except Exception:
            skipped += 1
    if skipped:
        print(f"  [WARN] Dropped {skipped} unreadable images.")
    return valid


def split_train_validation(records: list, validation_split: float, seed: int) -> tuple[list, list]:
    """Deterministically split records into train/validation subsets."""
    if validation_split <= 0 or len(records) < 2:
        return records, []

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * validation_split))
    n_val = min(n_val, len(shuffled) - 1)
    return shuffled[n_val:], shuffled[:n_val]


def find_latest_checkpoint(checkpoint_root: str):
    """
    Scan checkpoint_root for epoch_N subdirs and return the highest N.
    Returns (epoch_number, path) or (0, None) if nothing found.
    """
    if not os.path.isdir(checkpoint_root):
        return 0, None
    epochs = []
    for name in os.listdir(checkpoint_root):
        if name.startswith("epoch_"):
            try:
                n = int(name.split("_")[1])
                epochs.append(n)
            except ValueError:
                pass
    if not epochs:
        return 0, None
    latest = max(epochs)
    return latest, os.path.join(checkpoint_root, f"epoch_{latest}")


def get_training_config(current_lr: float, warmup_ratio: float, max_grad_norm: float) -> dict:
    """Snapshot the active hyperparameters that affect optimization."""
    return {
        "seed"                   : SEED,
        "num_epochs"             : NUM_EPOCHS,
        "batch_size"             : BATCH_SIZE,
        "gradient_accumulation"  : GRADIENT_ACCUMULATION,
        "learning_rate"          : current_lr,
        "base_learning_rate"     : LEARNING_RATE,
        "warmup_ratio"           : warmup_ratio,
        "max_text_len"           : MAX_TEXT_LEN,
        "weight_decay"           : WEIGHT_DECAY,
        "max_grad_norm"          : max_grad_norm,
        "validation_split"       : VALIDATION_SPLIT,
        "checkpoint_every_epochs": CHECKPOINT_EVERY_EPOCHS,
        "collapse_detection"     : {
            "loss_window_steps"       : LOSS_WINDOW_STEPS,
            "loss_explosion_factor"   : LOSS_EXPLOSION_FACTOR,
            "loss_explosion_min"      : LOSS_EXPLOSION_MIN,
            "grad_norm_threshold"     : GRAD_NORM_COLLAPSE_THRESHOLD,
            "val_loss_factor"         : VAL_LOSS_COLLAPSE_FACTOR,
            "val_loss_margin"         : VAL_LOSS_COLLAPSE_MARGIN,
            "worsening_epoch_patience": WORSENING_EPOCH_PATIENCE,
        },
    }


def get_rng_state() -> dict:
    """Capture RNG state so a checkpoint can resume reproducibly."""
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None):
    """Restore RNG state when present in older/newer checkpoints."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(model, processor, optimizer, scheduler, epoch: int,
                    checkpoint_root: str, global_step: int, train_loss: float,
                    val_loss: float | None, config: dict, recovery_state: dict):
    """Save model + training state after an epoch."""
    epoch_dir = os.path.join(checkpoint_root, f"epoch_{epoch}")
    if os.path.exists(epoch_dir) and os.listdir(epoch_dir):
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint: {epoch_dir}. "
            "Resume should continue from the next epoch."
        )
    os.makedirs(epoch_dir, exist_ok=True)

    model.save_pretrained(epoch_dir)
    processor.save_pretrained(epoch_dir)

    state = {
        "epoch"        : epoch,
        "global_step"  : global_step,
        "train_loss"   : train_loss,
        "val_loss"     : val_loss,
        "optimizer"    : optimizer.state_dict(),
        "scheduler"    : scheduler.state_dict(),
        "rng_state"    : get_rng_state(),
        "config"       : config,
        "recovery_state": recovery_state,
    }
    torch.save(state, os.path.join(epoch_dir, "trainer_state.pt"))

    latest_path = os.path.join(checkpoint_root, "latest_checkpoint.json")
    with open(latest_path, "w") as f:
        json.dump({
            "epoch": epoch,
            "global_step": global_step,
            "checkpoint_dir": os.path.relpath(epoch_dir, os.path.dirname(checkpoint_root)),
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
        }, f, indent=2)

    print(f"  Checkpoint saved → {epoch_dir}")
    return epoch_dir


def load_trainer_state(checkpoint_path: str) -> dict | None:
    state_path = os.path.join(checkpoint_path, "trainer_state.pt")
    if not os.path.exists(state_path):
        return None
    try:
        return torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(state_path, map_location="cpu")


def build_optimizer_and_scheduler(model, lr: float, warmup_ratio: float,
                                  total_steps: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler, warmup_steps


def load_manifest_state(manifest_path: str) -> tuple[list, int | None, float, dict]:
    """Carry best/checkpoint/recovery metadata forward during resume."""
    if not os.path.exists(manifest_path):
        return [], None, float("inf"), {"restart_count": 0, "events": []}
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return [], None, float("inf"), {"restart_count": 0, "events": []}

    best_loss = manifest.get("best_metric", manifest.get("best_avg_loss", float("inf")))
    if best_loss is None:
        best_loss = float("inf")
    recovery_state = manifest.get("recovery_state") or {"restart_count": 0, "events": []}
    recovery_state.setdefault("restart_count", 0)
    recovery_state.setdefault("events", [])
    return (
        manifest.get("epoch_summaries", []),
        manifest.get("best_epoch"),
        float(best_loss),
        recovery_state,
    )


def make_json_safe(value):
    """Convert non-finite floats before writing strict JSON artifacts."""
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    return value


def save_loss_curve_svg(epoch_summaries: list[dict], output_path: str):
    """Write a small dependency-free SVG plot for train/validation loss curves."""
    if not epoch_summaries:
        return

    train_points = [
        (item["epoch"], item["train_avg_loss"])
        for item in epoch_summaries
        if item.get("train_avg_loss") is not None
    ]
    val_points = [
        (item["epoch"], item["val_avg_loss"])
        for item in epoch_summaries
        if item.get("val_avg_loss") is not None
    ]
    all_points = train_points + val_points
    if not all_points:
        return

    width, height = 720, 420
    left, right, top, bottom = 70, 30, 30, 60
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x += 1
    if min_y == max_y:
        max_y += 1

    def scale(point):
        x, y = point
        px = left + (x - min_x) / (max_x - min_x) * (width - left - right)
        py = top + (max_y - y) / (max_y - min_y) * (height - top - bottom)
        return px, py

    def polyline(points, color):
        if not points:
            return ""
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(scale, points))
        circles = "\n".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" />'
            for x, y in map(scale, points)
        )
        return (
            f'<polyline points="{coords}" fill="none" stroke="{color}" '
            f'stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />\n'
            f'{circles}'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="22" font-family="Arial" font-size="16" font-weight="700">Full fine-tuning loss curves</text>
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333" />
  <text x="{width/2}" y="{height-18}" font-family="Arial" font-size="13" text-anchor="middle">epoch</text>
  <text x="18" y="{height/2}" font-family="Arial" font-size="13" transform="rotate(-90 18 {height/2})" text-anchor="middle">loss</text>
  <text x="{left}" y="{height-bottom+22}" font-family="Arial" font-size="12">{min_x}</text>
  <text x="{width-right}" y="{height-bottom+22}" font-family="Arial" font-size="12" text-anchor="end">{max_x}</text>
  <text x="{left-10}" y="{height-bottom}" font-family="Arial" font-size="12" text-anchor="end">{min_y:.3f}</text>
  <text x="{left-10}" y="{top+4}" font-family="Arial" font-size="12" text-anchor="end">{max_y:.3f}</text>
  {polyline(train_points, "#2563eb")}
  {polyline(val_points, "#dc2626")}
  <rect x="{width-180}" y="34" width="130" height="54" fill="#fff" stroke="#ddd" />
  <line x1="{width-166}" y1="54" x2="{width-136}" y2="54" stroke="#2563eb" stroke-width="3" />
  <text x="{width-126}" y="58" font-family="Arial" font-size="12">train</text>
  <line x1="{width-166}" y1="76" x2="{width-136}" y2="76" stroke="#dc2626" stroke-width="3" />
  <text x="{width-126}" y="80" font-family="Arial" font-size="12">validation</text>
</svg>
"""
    with open(output_path, "w") as f:
        f.write(svg)


class CollapseMonitor:
    """Detect numerical collapse and catastrophic loss divergence."""

    def __init__(self):
        self.recent_losses = []
        self.best_metric = float("inf")
        self.epoch_metrics = []

    def check_step_loss(self, loss_value: float, epoch: int, step: int,
                        global_step: int, lr: float) -> dict | None:
        event = {
            "epoch": epoch,
            "step_in_epoch": step,
            "global_step": global_step,
            "lr": lr,
            "loss": loss_value,
        }
        if not math.isfinite(loss_value):
            return {**event, "reason": "non_finite_train_loss"}

        if len(self.recent_losses) >= LOSS_WINDOW_STEPS:
            baseline = median(self.recent_losses[-LOSS_WINDOW_STEPS:])
            if (
                baseline > 0
                and loss_value >= LOSS_EXPLOSION_MIN
                and loss_value > baseline * LOSS_EXPLOSION_FACTOR
            ):
                return {
                    **event,
                    "reason": "train_loss_explosion",
                    "window_median_loss": baseline,
                    "threshold": baseline * LOSS_EXPLOSION_FACTOR,
                }

        self.recent_losses.append(loss_value)
        return None

    def check_grad_norm(self, grad_norm: float, epoch: int, step: int,
                        global_step: int) -> dict | None:
        if not math.isfinite(grad_norm):
            return {
                "reason": "non_finite_grad_norm",
                "epoch": epoch,
                "step_in_epoch": step,
                "global_step": global_step,
                "grad_norm": grad_norm,
            }
        if grad_norm > GRAD_NORM_COLLAPSE_THRESHOLD:
            return {
                "reason": "grad_norm_explosion",
                "epoch": epoch,
                "step_in_epoch": step,
                "global_step": global_step,
                "grad_norm": grad_norm,
                "threshold": GRAD_NORM_COLLAPSE_THRESHOLD,
            }
        return None

    def check_epoch(self, train_loss: float, val_loss: float | None,
                    epoch: int) -> dict | None:
        metric = val_loss if val_loss is not None else train_loss
        if not math.isfinite(metric):
            return {"reason": "non_finite_epoch_loss", "epoch": epoch, "metric": metric}

        if (
            val_loss is not None
            and self.best_metric < float("inf")
            and val_loss > self.best_metric * VAL_LOSS_COLLAPSE_FACTOR
            and val_loss > self.best_metric + VAL_LOSS_COLLAPSE_MARGIN
        ):
            return {
                "reason": "validation_loss_catastrophe",
                "epoch": epoch,
                "val_loss": val_loss,
                "best_metric": self.best_metric,
            }

        self.epoch_metrics.append(metric)
        if metric < self.best_metric:
            self.best_metric = metric

        if len(self.epoch_metrics) >= WORSENING_EPOCH_PATIENCE + 1:
            recent = self.epoch_metrics[-(WORSENING_EPOCH_PATIENCE + 1):]
            worsening = all(
                recent[i] > recent[i - 1] + WORSENING_EPOCH_MARGIN
                for i in range(1, len(recent))
            )
            if worsening and recent[-1] > self.best_metric * 1.25:
                return {
                    "reason": "persistent_epoch_loss_worsening",
                    "epoch": epoch,
                    "recent_epoch_metrics": recent,
                    "best_metric": self.best_metric,
                }
        return None


def write_collapse_diagnostics(event: dict, logs_dir: str, recovery_state: dict,
                               config: dict) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(logs_dir, f"collapse_event_{timestamp}.json")
    suffix = 1
    while os.path.exists(path):
        path = os.path.join(logs_dir, f"collapse_event_{timestamp}_{suffix}.json")
        suffix += 1
    payload = {
        "event": event,
        "recovery_state": recovery_state,
        "active_config": config,
    }
    with open(path, "w") as f:
        json.dump(make_json_safe(payload), f, indent=2, allow_nan=False)
    return path


def format_eta(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    base_dir        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path    = os.path.join(base_dir, "data", "vlm_conversational_dataset.json")
    checkpoint_root = os.path.join(base_dir, CHECKPOINT_SUBDIR)
    logs_dir        = os.path.join(base_dir, "logs")
    results_dir     = os.path.join(base_dir, "results")
    loss_csv_path   = os.path.join(logs_dir, "full_train_loss.csv")
    loss_curve_path = os.path.join(logs_dir, "full_train_loss_curve.svg")
    manifest_path   = os.path.join(results_dir, "full_training_manifest.json")

    os.makedirs(checkpoint_root, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ── 1. Validate dataset ───────────────────────────────────────────────────
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run prepare_vlm_dataset.py first."
        )

    print("=" * 65)
    print("  VLM Full Fine-Tuning Run — May 9 (Anirudh)")
    print("=" * 65)

    # ── 2. Load & validate records ────────────────────────────────────────────
    print("\n[Step 1] Loading dataset ...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_records = json.load(f)
    print(f"  Total records in JSON : {len(all_records)}")

    valid_records = validate_records(all_records, base_dir)
    print(f"  Valid records         : {len(valid_records)}")

    if not valid_records:
        raise RuntimeError("No valid image records found. Check dataset paths.")

    # ── 3. Resume detection ───────────────────────────────────────────────────
    start_epoch, resume_path = find_latest_checkpoint(checkpoint_root)
    resume_state = load_trainer_state(resume_path) if resume_path else None
    current_lr = LEARNING_RATE
    current_warmup_ratio = WARMUP_RATIO
    current_max_grad_norm = MAX_GRAD_NORM
    if resume_state and resume_state.get("config"):
        saved_config = resume_state["config"]
        current_lr = saved_config.get("learning_rate", current_lr)
        current_warmup_ratio = saved_config.get("warmup_ratio", current_warmup_ratio)
        current_max_grad_norm = saved_config.get("max_grad_norm", current_max_grad_norm)

    if resume_path:
        print(f"\n[Resume] Detected checkpoint at epoch {start_epoch}: {resume_path}")
        print(f"  Will resume from epoch {start_epoch + 1}")
    else:
        print("\n[Resume] No prior checkpoint found — starting fresh.")

    # ── 4. Load model & processor ─────────────────────────────────────────────
    print("\n[Step 2] Loading model ...")
    if resume_path:
        print(f"  Loading fine-tuned weights from: {resume_path}")
        processor = BlipProcessor.from_pretrained(resume_path)
        model = BlipForConditionalGeneration.from_pretrained(
            resume_path, torch_dtype=torch.bfloat16
        )
    else:
        processor = BlipProcessor.from_pretrained(MODEL_NAME)
        model = BlipForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16
        )

    model.gradient_checkpointing_enable()
    print("  gradient_checkpointing : ENABLED")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device                 : {device}")
    if device.type == "cuda":
        print(f"  GPU                    : {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Total GPU memory       : {total_mem:.2f} GB")

    model.to(device)
    log_gpu("after model.to(device)")

    # ── 5. DataLoader ─────────────────────────────────────────────────────────
    print(f"\n[Step 3] Building DataLoader ...")
    train_records, val_records = split_train_validation(valid_records, VALIDATION_SPLIT, SEED)

    def build_loaders(active_processor):
        train_dataset = VLMDataset(train_records, base_dir, active_processor)
        val_dataset = VLMDataset(val_records, base_dir, active_processor) if val_records else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,          # safer on MIG slices
            pin_memory=(device.type == "cuda"),
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )
        return train_dataset, val_dataset, train_loader, val_loader

    dataset, val_dataset, loader, val_loader = build_loaders(processor)
    steps_per_epoch = math.ceil(len(loader) / GRADIENT_ACCUMULATION)
    total_steps     = steps_per_epoch * NUM_EPOCHS

    print(f"  Train samples    : {len(dataset)}")
    print(f"  Val samples      : {len(val_dataset) if val_dataset is not None else 0}")
    print(f"  Train batches/epoch : {len(loader)}")
    if val_loader is not None:
        print(f"  Val batches/epoch   : {len(val_loader)}")
    print(f"  Optimizer steps/epoch : {steps_per_epoch}")
    print(f"  Total opt. steps : {total_steps}")

    # ── 6. Optimizer & Scheduler ──────────────────────────────────────────────
    optimizer, scheduler, warmup_steps = build_optimizer_and_scheduler(
        model, current_lr, current_warmup_ratio, total_steps
    )

    # Restore optimizer/scheduler state if resuming
    global_step = 0
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        restore_rng_state(resume_state.get("rng_state"))
        global_step = resume_state["global_step"]
        print(f"  Restored optimizer/scheduler state (global_step={global_step})")

    print(f"  LR schedule      : cosine with warmup")
    print(f"  Effective batch  : {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Learning rate    : {current_lr}")
    print(f"  Warmup ratio     : {current_warmup_ratio}  (warmup steps: {warmup_steps})")
    print(f"  Grad clip max    : {current_max_grad_norm}")

    # ── 7. Loss CSV setup ─────────────────────────────────────────────────────
    csv_mode = "a" if start_epoch > 0 else "w"
    csv_file   = open(loss_csv_path, csv_mode, newline="", buffering=1)
    csv_writer = csv.writer(csv_file)
    if csv_mode == "w":
        csv_writer.writerow(["phase", "epoch", "step_in_epoch", "global_step", "loss", "lr"])

    # ── 8. Training loop ──────────────────────────────────────────────────────
    print(f"\n[Step 4] Training ({NUM_EPOCHS} epochs, resuming from epoch {start_epoch + 1}) ...")

    epoch_summaries, best_epoch, best_metric, recovery_state = load_manifest_state(manifest_path)
    recovery_state.setdefault("restart_count", 0)
    recovery_state.setdefault("events", [])
    stable_checkpoint_path = resume_path
    monitor = CollapseMonitor()
    monitor.best_metric = best_metric
    for item in epoch_summaries:
        metric = item.get("val_avg_loss")
        if metric is None:
            metric = item.get("train_avg_loss", item.get("avg_loss"))
        if metric is not None:
            monitor.epoch_metrics.append(metric)
    run_start_time   = time.time()

    def run_validation(epoch: int) -> float | None:
        if val_loader is None:
            return None
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=(device.type == "cuda"),
                ):
                    outputs = model(
                        pixel_values   = batch["pixel_values"],
                        input_ids      = batch["input_ids"],
                        attention_mask = batch["attention_mask"],
                        labels         = batch["input_ids"],
                    )
                    val_loss = outputs.loss

                raw_val_loss = float(val_loss.item())
                val_loss_total += raw_val_loss
                if val_step % LOG_EVERY_N_STEPS == 0 or val_step == len(val_loader):
                    csv_writer.writerow([
                        "val", epoch, val_step, global_step,
                        f"{raw_val_loss:.6f}", f"{scheduler.get_last_lr()[0]:.8f}",
                    ])

        model.train()
        return val_loss_total / len(val_loader)

    epoch = start_epoch + 1
    while epoch <= NUM_EPOCHS:
        print(f"\n{'─'*65}")
        print(f"  EPOCH {epoch}/{NUM_EPOCHS}")
        print(f"{'─'*65}")

        model.train()
        epoch_loss        = 0.0
        optimizer_steps   = 0
        optimizer.zero_grad()
        epoch_start_time  = time.time()
        collapse_event    = None

        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(
                    pixel_values   = batch["pixel_values"],
                    input_ids      = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                    labels         = batch["input_ids"],
                )
                loss = outputs.loss / GRADIENT_ACCUMULATION

            raw_loss = float(loss.item()) * GRADIENT_ACCUMULATION
            current_lr_for_log = scheduler.get_last_lr()[0]
            collapse_event = monitor.check_step_loss(
                raw_loss, epoch, step + 1, global_step, current_lr_for_log
            )
            if collapse_event:
                optimizer.zero_grad()
                break

            loss.backward()

            if (step + 1) % GRADIENT_ACCUMULATION == 0 or (step + 1) == len(loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=current_max_grad_norm
                )
                collapse_event = monitor.check_grad_norm(
                    float(grad_norm), epoch, step + 1, global_step
                )
                if collapse_event:
                    optimizer.zero_grad()
                    break
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step   += 1
                optimizer_steps += 1

            epoch_loss += raw_loss

            # Console progress every LOG_EVERY_N_STEPS batches
            if (step + 1) % LOG_EVERY_N_STEPS == 0 or (step + 1) == len(loader):
                elapsed   = time.time() - epoch_start_time
                pct       = (step + 1) / len(loader)
                eta_epoch = (elapsed / pct) * (1 - pct) if pct > 0 else 0
                progress_lr = scheduler.get_last_lr()[0]
                print(
                    f"    e{epoch:02d} | batch {step+1:>4}/{len(loader)} "
                    f"| loss: {raw_loss:.4f}  lr: {progress_lr:.2e}  "
                    f"eta: {format_eta(eta_epoch)}"
                )

                # Write to CSV
                csv_writer.writerow([
                    "train", epoch, step + 1, global_step,
                    f"{raw_loss:.6f}", f"{current_lr_for_log:.8f}",
                ])

        if collapse_event:
            config = get_training_config(current_lr, current_warmup_ratio, current_max_grad_norm)
            diag_path = write_collapse_diagnostics(
                collapse_event, logs_dir, recovery_state, config
            )
            print(f"\n  [COLLAPSE] {collapse_event['reason']} detected.")
            print(f"    diagnostics → {diag_path}")

            if recovery_state["restart_count"] >= MAX_RECOVERY_RESTARTS:
                csv_file.close()
                raise RuntimeError(
                    "Collapse recovery limit reached. "
                    f"Last event: {collapse_event['reason']}"
                )

            recovery_state["restart_count"] += 1
            current_lr *= RECOVERY_LR_FACTOR
            current_warmup_ratio = min(
                current_warmup_ratio + RECOVERY_WARMUP_INCREMENT,
                RECOVERY_MAX_WARMUP_RATIO,
            )
            current_max_grad_norm = max(
                current_max_grad_norm * RECOVERY_GRAD_CLIP_FACTOR,
                RECOVERY_MIN_GRAD_CLIP,
            )
            recovery_state["events"].append({
                **collapse_event,
                "diagnostics": os.path.relpath(diag_path, base_dir),
                "restart_count": recovery_state["restart_count"],
                "new_learning_rate": current_lr,
                "new_warmup_ratio": current_warmup_ratio,
                "new_max_grad_norm": current_max_grad_norm,
                "reload_checkpoint": stable_checkpoint_path,
            })

            reload_path = stable_checkpoint_path if stable_checkpoint_path else MODEL_NAME
            print(f"    reloading stable weights from: {reload_path}")
            print(
                f"    adjusted hyperparameters: lr={current_lr:.2e}, "
                f"warmup={current_warmup_ratio:.2f}, grad_clip={current_max_grad_norm:.2f}"
            )
            processor = BlipProcessor.from_pretrained(reload_path)
            model = BlipForConditionalGeneration.from_pretrained(
                reload_path, torch_dtype=torch.bfloat16
            )
            model.gradient_checkpointing_enable()
            model.to(device)
            dataset, val_dataset, loader, val_loader = build_loaders(processor)
            optimizer, scheduler, warmup_steps = build_optimizer_and_scheduler(
                model, current_lr, current_warmup_ratio, total_steps
            )

            stable_state = load_trainer_state(stable_checkpoint_path) if stable_checkpoint_path else None
            if stable_state:
                optimizer.load_state_dict(stable_state["optimizer"])
                restore_rng_state(stable_state.get("rng_state"))
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
                    group["initial_lr"] = current_lr
                global_step = stable_state["global_step"]
                epoch = stable_state["epoch"] + 1
            else:
                global_step = 0
                epoch = 1

            epoch_summaries = [
                item for item in epoch_summaries
                if item.get("epoch", 0) < epoch
            ]
            monitor = CollapseMonitor()
            monitor.best_metric = best_metric
            for item in epoch_summaries:
                metric = item.get("val_avg_loss")
                if metric is None:
                    metric = item.get("train_avg_loss", item.get("avg_loss"))
                if metric is not None:
                    monitor.epoch_metrics.append(metric)
            continue

        # ── Epoch summary ─────────────────────────────────────────────────────
        avg_loss    = epoch_loss / len(loader)
        val_avg_loss = run_validation(epoch)
        collapse_event = monitor.check_epoch(avg_loss, val_avg_loss, epoch)
        if collapse_event:
            config = get_training_config(current_lr, current_warmup_ratio, current_max_grad_norm)
            diag_path = write_collapse_diagnostics(
                collapse_event, logs_dir, recovery_state, config
            )
            print(f"\n  [COLLAPSE] {collapse_event['reason']} detected after validation.")
            print(f"    diagnostics → {diag_path}")
            if recovery_state["restart_count"] >= MAX_RECOVERY_RESTARTS:
                csv_file.close()
                raise RuntimeError(
                    "Collapse recovery limit reached. "
                    f"Last event: {collapse_event['reason']}"
                )

            recovery_state["restart_count"] += 1
            current_lr *= RECOVERY_LR_FACTOR
            current_warmup_ratio = min(
                current_warmup_ratio + RECOVERY_WARMUP_INCREMENT,
                RECOVERY_MAX_WARMUP_RATIO,
            )
            current_max_grad_norm = max(
                current_max_grad_norm * RECOVERY_GRAD_CLIP_FACTOR,
                RECOVERY_MIN_GRAD_CLIP,
            )
            recovery_state["events"].append({
                **collapse_event,
                "diagnostics": os.path.relpath(diag_path, base_dir),
                "restart_count": recovery_state["restart_count"],
                "new_learning_rate": current_lr,
                "new_warmup_ratio": current_warmup_ratio,
                "new_max_grad_norm": current_max_grad_norm,
                "reload_checkpoint": stable_checkpoint_path,
            })

            reload_path = stable_checkpoint_path if stable_checkpoint_path else MODEL_NAME
            processor = BlipProcessor.from_pretrained(reload_path)
            model = BlipForConditionalGeneration.from_pretrained(
                reload_path, torch_dtype=torch.bfloat16
            )
            model.gradient_checkpointing_enable()
            model.to(device)
            dataset, val_dataset, loader, val_loader = build_loaders(processor)
            optimizer, scheduler, warmup_steps = build_optimizer_and_scheduler(
                model, current_lr, current_warmup_ratio, total_steps
            )
            stable_state = load_trainer_state(stable_checkpoint_path) if stable_checkpoint_path else None
            if stable_state:
                optimizer.load_state_dict(stable_state["optimizer"])
                restore_rng_state(stable_state.get("rng_state"))
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
                    group["initial_lr"] = current_lr
                global_step = stable_state["global_step"]
                epoch = stable_state["epoch"] + 1
            else:
                global_step = 0
                epoch = 1
            epoch_summaries = [
                item for item in epoch_summaries
                if item.get("epoch", 0) < epoch
            ]
            monitor = CollapseMonitor()
            monitor.best_metric = best_metric
            for item in epoch_summaries:
                metric = item.get("val_avg_loss")
                if metric is None:
                    metric = item.get("train_avg_loss", item.get("avg_loss"))
                if metric is not None:
                    monitor.epoch_metrics.append(metric)
            continue

        epoch_time  = time.time() - epoch_start_time
        total_elapsed = time.time() - run_start_time

        print(f"\n  ► Epoch {epoch} done")
        print(f"    train avg loss : {avg_loss:.4f}")
        if val_avg_loss is not None:
            print(f"    val avg loss   : {val_avg_loss:.4f}")
        print(f"    epoch duration : {format_eta(epoch_time)}")
        print(f"    total elapsed  : {format_eta(total_elapsed)}")
        log_gpu(f"end of epoch {epoch}")

        # ── Save epoch checkpoint ─────────────────────────────────────────────
        epoch_dir = None
        if epoch % CHECKPOINT_EVERY_EPOCHS == 0 or epoch == NUM_EPOCHS:
            epoch_dir = save_checkpoint(
                model, processor, optimizer, scheduler,
                epoch, checkpoint_root, global_step, avg_loss, val_avg_loss,
                get_training_config(current_lr, current_warmup_ratio, current_max_grad_norm),
                recovery_state,
            )
            stable_checkpoint_path = epoch_dir

        # ── Track best ────────────────────────────────────────────────────────
        metric_for_best = val_avg_loss if val_avg_loss is not None else avg_loss
        if epoch_dir and metric_for_best < best_metric:
            best_metric = metric_for_best
            best_epoch = epoch
            best_dir   = os.path.join(checkpoint_root, "best")
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            shutil.copytree(epoch_dir, best_dir)
            print(f"    ★ New best checkpoint (metric={best_metric:.4f}) → {best_dir}")

        epoch_summaries.append({
            "epoch"         : epoch,
            "train_avg_loss": round(avg_loss, 6),
            "val_avg_loss"  : round(val_avg_loss, 6) if val_avg_loss is not None else None,
            "duration_s"    : round(epoch_time, 1),
            "global_step"   : global_step,
        })
        save_loss_curve_svg(epoch_summaries, loss_curve_path)

        # ── Write manifest after every epoch (safe mid-run checkpoint) ────────
        manifest = {
            "model_name"        : MODEL_NAME,
            "total_samples"     : len(valid_records),
            "train_samples"     : len(dataset),
            "val_samples"       : len(val_dataset) if val_dataset is not None else 0,
            "num_epochs"        : NUM_EPOCHS,
            "epochs_completed"  : epoch,
            "batch_size"        : BATCH_SIZE,
            "gradient_accum"    : GRADIENT_ACCUMULATION,
            "effective_batch"   : BATCH_SIZE * GRADIENT_ACCUMULATION,
            "learning_rate"     : current_lr,
            "base_learning_rate": LEARNING_RATE,
            "warmup_ratio"      : current_warmup_ratio,
            "max_grad_norm"     : current_max_grad_norm,
            "max_text_len"      : MAX_TEXT_LEN,
            "seed"              : SEED,
            "best_epoch"        : best_epoch,
            "best_metric"       : round(best_metric, 6) if best_metric < float("inf") else None,
            "best_metric_source": "val_avg_loss" if val_loader is not None else "train_avg_loss",
            "checkpoint_root"   : CHECKPOINT_SUBDIR,
            "loss_csv"          : "logs/full_train_loss.csv",
            "loss_curve_svg"    : "logs/full_train_loss_curve.svg",
            "latest_checkpoint" : "checkpoints/full_train/latest_checkpoint.json",
            "recovery_state"    : recovery_state,
            "epoch_summaries"   : epoch_summaries,
            "status"            : "in_progress" if epoch < NUM_EPOCHS else "completed",
        }
        with open(manifest_path, "w") as mf:
            json.dump(make_json_safe(manifest), mf, indent=2, allow_nan=False)

        epoch += 1

    csv_file.close()

    # ── Done ──────────────────────────────────────────────────────────────────
    total_time = time.time() - run_start_time
    print("\n" + "=" * 65)
    print("  FULL FINE-TUNING COMPLETE")
    print(f"  Total training time : {format_eta(total_time)}")
    if best_metric < float("inf"):
        print(f"  Best epoch          : {best_epoch}  (metric: {best_metric:.4f})")
    print(f"  Best checkpoint     : {os.path.join(checkpoint_root, 'best')}")
    print(f"  Loss curve CSV      : {loss_csv_path}")
    print(f"  Loss curve SVG      : {loss_curve_path}")
    print(f"  Manifest            : {manifest_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
