"""
src/train.py
VLM Fine-Tuning — Smoke Test (Pranav | May 8)

Initializes the BLIP training loop and runs it on a tiny batch of 10 images
from the prepared conversational dataset to verify no Out-Of-Memory (OOM)
errors on the MIG A100 (1g.5gb) GPU slice.

Design choices for 5 GB GPU:
  - fp16 mixed precision via torch.cuda.amp
  - gradient checkpointing (trades compute for ~30% less activation memory)
  - batch_size=1 with gradient_accumulation_steps=4 (effective batch = 4)
  - AdamW with standard lr; no 8-bit optimizer needed at this scale
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, UnidentifiedImageError

# ─── Hyperparameters ──────────────────────────────────────────────────────────
SEED                    = 42
SMOKE_TEST_SAMPLES      = 10       # tiny batch as per task spec
BATCH_SIZE              = 1        # must be 1 for 5 GB MIG slice
GRADIENT_ACCUMULATION   = 4        # effective batch size = 4
NUM_EPOCHS              = 2        # enough to confirm loss is decreasing
LEARNING_RATE           = 5e-5
MAX_TEXT_LEN            = 64       # short captions; saves memory vs 128
MODEL_NAME              = "Salesforce/blip-image-captioning-base"
CHECKPOINT_SUBDIR       = "checkpoints/smoke_test_checkpoint"
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
torch.manual_seed(SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────

class VLMSmokeDataset(Dataset):
    """
    Wraps the LLaVA-style conversational JSON (produced by prepare_vlm_dataset.py)
    into a PyTorch Dataset.

    Each record looks like:
      {
        "id": "...",
        "image": "data/whole_multicare_dataset/vlm_mri_subset/images/...",
        "conversations": [
          {"from": "human", "value": "<image>\nDescribe ..."},
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
        caption  = rec["conversations"][1]["value"]   # gpt / ground-truth turn

        image = Image.open(img_path).convert("RGB")

        # BlipProcessor returns pixel_values + tokenised text in one call
        encoding = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_TEXT_LEN,
            truncation=True,
        )
        # squeeze out the batch dim that processor adds
        return {k: v.squeeze(0) for k, v in encoding.items()}


# ── Helpers ───────────────────────────────────────────────────────────────────

def log_gpu(label: str):
    """Print allocated / reserved GPU memory (no-op on CPU)."""
    if torch.cuda.is_available():
        alloc    = torch.cuda.memory_allocated()  / 1e9
        reserved = torch.cuda.memory_reserved()   / 1e9
        print(f"    [GPU | {label}]  allocated: {alloc:.3f} GB  |  reserved: {reserved:.3f} GB")


def load_and_sample_dataset(dataset_path: str, n: int) -> list:
    """Load the conversational JSON and return n shuffled records."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  Total records in dataset : {len(records)}")
    random.shuffle(records)
    sampled = records[:n]
    print(f"  Smoke-test subset        : {len(sampled)} samples")
    return sampled


def validate_records(records: list, base_dir: str) -> list:
    """
    Drop any record whose image file is missing or unreadable.
    Keeps the run from crashing mid-loop on a corrupt file.
    """
    valid = []
    for rec in records:
        path = os.path.join(base_dir, rec["image"])
        try:
            with Image.open(path) as im:
                im.verify()
            valid.append(rec)
        except (FileNotFoundError, UnidentifiedImageError, Exception):
            print(f"  [WARN] Skipping unreadable image: {path}")
    return valid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    base_dir        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path    = os.path.join(base_dir, "data", "vlm_conversational_dataset.json")
    checkpoint_path = os.path.join(base_dir, CHECKPOINT_SUBDIR)
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)

    # ── 1. Sanity-check dataset file ─────────────────────────────────────────
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}\n"
            "Run prepare_vlm_dataset.py first (Anirudh's step)."
        )

    print("=" * 60)
    print("VLM Smoke-Test Training — May 8 (Pranav)")
    print("=" * 60)

    # ── 2. Load & sample dataset ─────────────────────────────────────────────
    print("\n[Step 1] Loading dataset ...")
    records = load_and_sample_dataset(dataset_path, SMOKE_TEST_SAMPLES)
    records = validate_records(records, base_dir)

    if not records:
        raise RuntimeError(
            "No valid image records found in the sample. "
            "Check that image files exist relative to the repo root."
        )

    print(f"  Valid records after image check : {len(records)}")

    # ── 3. Load model & processor ────────────────────────────────────────────
    print(f"\n[Step 2] Loading model: {MODEL_NAME} ...")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
)

    # Gradient checkpointing: ~30% less activation memory at cost of ~20% more compute
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

    # ── 4. DataLoader ────────────────────────────────────────────────────────
    print(f"\n[Step 3] Building DataLoader (batch_size={BATCH_SIZE}) ...")
    dataset = VLMSmokeDataset(records, base_dir, processor)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"  Batches per epoch : {len(loader)}")

    # ── 5. Optimizer & AMP scaler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    use_amp   = device.type == "cuda"
    scaler = None  # not needed with bfloat16
    print(f"  Mixed precision (AMP) : {'ENABLED (fp16)' if use_amp else 'DISABLED (CPU)'}")
    print(f"  Gradient accumulation : {GRADIENT_ACCUMULATION} steps  →  effective batch: {GRADIENT_ACCUMULATION}")

    # ── 6. Training loop ─────────────────────────────────────────────────────
    print(f"\n[Step 4] Training ({NUM_EPOCHS} epoch(s), {len(dataset)} samples) ...")
    model.train()

    for epoch in range(NUM_EPOCHS):
        print(f"\n  ── Epoch {epoch + 1}/{NUM_EPOCHS} ──")
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(
                    pixel_values   = batch["pixel_values"],
                    input_ids      = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                    labels         = batch["input_ids"],   # BLIP: labels = input_ids for LM loss
                )
                # Scale loss by accumulation steps before backward
                loss = outputs.loss / GRADIENT_ACCUMULATION

            loss.backward()

            # Step optimizer every GRADIENT_ACCUMULATION mini-batches (or at end of epoch)
            if (step + 1) % GRADIENT_ACCUMULATION == 0 or (step + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()

            raw_loss = loss.item() * GRADIENT_ACCUMULATION
            epoch_loss += raw_loss
            print(f"    step {step + 1:>2}/{len(loader)} | loss: {raw_loss:.4f}")
            log_gpu(f"e{epoch+1}s{step+1}")

        avg_loss = epoch_loss / len(loader)
        print(f"\n  Epoch {epoch + 1} done — avg loss: {avg_loss:.4f}")

    # ── 7. Save checkpoint ───────────────────────────────────────────────────
    print(f"\n[Step 5] Saving checkpoint to: {checkpoint_path}")
    model.save_pretrained(checkpoint_path)
    processor.save_pretrained(checkpoint_path)

    # Write a small training manifest so later steps can reference it
    manifest = {
        "model_name"        : MODEL_NAME,
        "smoke_test_samples": len(dataset),
        "epochs"            : NUM_EPOCHS,
        "batch_size"        : BATCH_SIZE,
        "gradient_accum"    : GRADIENT_ACCUMULATION,
        "learning_rate"     : LEARNING_RATE,
        "max_text_len"      : MAX_TEXT_LEN,
        "seed"              : SEED,
        "checkpoint_dir"    : CHECKPOINT_SUBDIR,
        "status"            : "smoke_test_passed",
    }
    import json as _json
    manifest_path = os.path.join(checkpoint_path, "training_manifest.json")
    with open(manifest_path, "w") as mf:
        _json.dump(manifest, mf, indent=2)

    print("\n" + "=" * 60)
    print("  SMOKE TEST PASSED — no OOM errors detected.")
    print("  Checkpoint and manifest written.")
    print("=" * 60)


if __name__ == "__main__":
    main()