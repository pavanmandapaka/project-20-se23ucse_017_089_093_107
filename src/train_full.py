"""
src/train_full.py
VLM Full Fine-Tuning — 24-Hour Production Run (Anirudh | May 9)

Launches the primary fine-tuning run on the full vlm_conversational_dataset.json
(1703 image-caption pairs) on the MIG A100 (1g.5gb) GPU slice.

Key improvements over the smoke-test (train.py):
  - Full dataset (all valid records, not a 10-sample subset)
  - 10 epochs with cosine LR schedule + linear warmup
  - Checkpoint saved every epoch to checkpoints/full_train/epoch_N/
  - Best checkpoint tracked by lowest avg epoch loss
  - Resume-from-checkpoint support (restart from latest saved epoch)
  - Step-level loss logged to logs/full_train_loss.csv
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

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from PIL import Image, UnidentifiedImageError

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


def save_checkpoint(model, processor, optimizer, scheduler, epoch: int,
                    checkpoint_root: str, global_step: int, avg_loss: float):
    """Save model + training state after an epoch."""
    epoch_dir = os.path.join(checkpoint_root, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    model.save_pretrained(epoch_dir)
    processor.save_pretrained(epoch_dir)

    state = {
        "epoch"       : epoch,
        "global_step" : global_step,
        "avg_loss"    : avg_loss,
        "optimizer"   : optimizer.state_dict(),
        "scheduler"   : scheduler.state_dict(),
    }
    torch.save(state, os.path.join(epoch_dir, "trainer_state.pt"))
    print(f"  Checkpoint saved → {epoch_dir}")


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
    if resume_path:
        print(f"\n[Resume] Detected checkpoint at epoch {start_epoch}: {resume_path}")
        print(f"  Will resume from epoch {start_epoch + 1}")
    else:
        print("\n[Resume] No prior checkpoint found — starting fresh.")

    # ── 4. Load model & processor ─────────────────────────────────────────────
    print(f"\n[Step 2] Loading model ...")
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
    dataset = VLMDataset(valid_records, base_dir, processor)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,          # safer on MIG slices
        pin_memory=(device.type == "cuda"),
    )
    steps_per_epoch = math.ceil(len(loader) / GRADIENT_ACCUMULATION)
    total_steps     = steps_per_epoch * NUM_EPOCHS
    warmup_steps    = int(total_steps * WARMUP_RATIO)

    print(f"  Samples          : {len(dataset)}")
    print(f"  Batches/epoch    : {len(loader)}")
    print(f"  Optimizer steps/epoch : {steps_per_epoch}")
    print(f"  Total opt. steps : {total_steps}  (warmup: {warmup_steps})")

    # ── 6. Optimizer & Scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Restore optimizer/scheduler state if resuming
    global_step = 0
    if resume_path:
        trainer_state_path = os.path.join(resume_path, "trainer_state.pt")
        if os.path.exists(trainer_state_path):
            state = torch.load(trainer_state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            global_step = state["global_step"]
            print(f"  Restored optimizer/scheduler state (global_step={global_step})")

    print(f"  LR schedule      : cosine with warmup")
    print(f"  Effective batch  : {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Learning rate    : {LEARNING_RATE}")

    # ── 7. Loss CSV setup ─────────────────────────────────────────────────────
    csv_file   = open(loss_csv_path, "a", newline="", buffering=1)
    csv_writer = csv.writer(csv_file)
    if start_epoch == 0:
        csv_writer.writerow(["epoch", "step_in_epoch", "global_step", "step_loss", "lr"])

    # ── 8. Training loop ──────────────────────────────────────────────────────
    print(f"\n[Step 4] Training ({NUM_EPOCHS} epochs, resuming from epoch {start_epoch + 1}) ...")

    epoch_summaries  = []
    best_epoch       = None
    best_loss        = float("inf")
    run_start_time   = time.time()

    for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
        print(f"\n{'─'*65}")
        print(f"  EPOCH {epoch}/{NUM_EPOCHS}")
        print(f"{'─'*65}")

        model.train()
        epoch_loss        = 0.0
        optimizer_steps   = 0
        optimizer.zero_grad()
        epoch_start_time  = time.time()

        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    pixel_values   = batch["pixel_values"],
                    input_ids      = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                    labels         = batch["input_ids"],
                )
                loss = outputs.loss / GRADIENT_ACCUMULATION

            loss.backward()

            if (step + 1) % GRADIENT_ACCUMULATION == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step   += 1
                optimizer_steps += 1

            raw_loss    = loss.item() * GRADIENT_ACCUMULATION
            epoch_loss += raw_loss

            # Console progress every LOG_EVERY_N_STEPS batches
            if (step + 1) % LOG_EVERY_N_STEPS == 0 or (step + 1) == len(loader):
                elapsed   = time.time() - epoch_start_time
                pct       = (step + 1) / len(loader)
                eta_epoch = (elapsed / pct) * (1 - pct) if pct > 0 else 0
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"    e{epoch:02d} | batch {step+1:>4}/{len(loader)} "
                    f"| loss: {raw_loss:.4f}  lr: {current_lr:.2e}  "
                    f"eta: {format_eta(eta_epoch)}"
                )

                # Write to CSV
                csv_writer.writerow([epoch, step + 1, global_step, f"{raw_loss:.6f}", f"{current_lr:.8f}"])

        # ── Epoch summary ─────────────────────────────────────────────────────
        avg_loss    = epoch_loss / len(loader)
        epoch_time  = time.time() - epoch_start_time
        total_elapsed = time.time() - run_start_time

        print(f"\n  ► Epoch {epoch} done")
        print(f"    avg loss       : {avg_loss:.4f}")
        print(f"    epoch duration : {format_eta(epoch_time)}")
        print(f"    total elapsed  : {format_eta(total_elapsed)}")
        log_gpu(f"end of epoch {epoch}")

        # ── Save epoch checkpoint ─────────────────────────────────────────────
        save_checkpoint(model, processor, optimizer, scheduler,
                        epoch, checkpoint_root, global_step, avg_loss)

        # ── Track best ────────────────────────────────────────────────────────
        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_epoch = epoch
            best_dir   = os.path.join(checkpoint_root, "best")
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            shutil.copytree(os.path.join(checkpoint_root, f"epoch_{epoch}"), best_dir)
            print(f"    ★ New best checkpoint (loss={best_loss:.4f}) → {best_dir}")

        epoch_summaries.append({
            "epoch"       : epoch,
            "avg_loss"    : round(avg_loss, 6),
            "duration_s"  : round(epoch_time, 1),
            "global_step" : global_step,
        })

        # ── Write manifest after every epoch (safe mid-run checkpoint) ────────
        manifest = {
            "model_name"        : MODEL_NAME,
            "total_samples"     : len(dataset),
            "num_epochs"        : NUM_EPOCHS,
            "epochs_completed"  : epoch,
            "batch_size"        : BATCH_SIZE,
            "gradient_accum"    : GRADIENT_ACCUMULATION,
            "effective_batch"   : BATCH_SIZE * GRADIENT_ACCUMULATION,
            "learning_rate"     : LEARNING_RATE,
            "max_text_len"      : MAX_TEXT_LEN,
            "seed"              : SEED,
            "best_epoch"        : best_epoch,
            "best_avg_loss"     : round(best_loss, 6),
            "checkpoint_root"   : CHECKPOINT_SUBDIR,
            "loss_csv"          : "logs/full_train_loss.csv",
            "epoch_summaries"   : epoch_summaries,
            "status"            : "in_progress" if epoch < NUM_EPOCHS else "completed",
        }
        with open(manifest_path, "w") as mf:
            json.dump(manifest, mf, indent=2)

    csv_file.close()

    # ── Done ──────────────────────────────────────────────────────────────────
    total_time = time.time() - run_start_time
    print("\n" + "=" * 65)
    print("  FULL FINE-TUNING COMPLETE")
    print(f"  Total training time : {format_eta(total_time)}")
    print(f"  Best epoch          : {best_epoch}  (avg loss: {best_loss:.4f})")
    print(f"  Best checkpoint     : {os.path.join(checkpoint_root, 'best')}")
    print(f"  Loss curve CSV      : {loss_csv_path}")
    print(f"  Manifest            : {manifest_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()