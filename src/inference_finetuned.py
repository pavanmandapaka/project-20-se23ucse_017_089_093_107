"""
src/inference_finetuned.py
Fine-Tuned VLM Inference — Extract Clinical Text (Anirudh | May 11)

Runs the fine-tuned BLIP checkpoint against the full test dataset
and saves generated captions to results/fine_tuned_results.txt.
This output is used by Kolla's evaluate.py to complete the ablation table.

Key differences from inference.py (zero-shot):
  - Loads from checkpoints/full_run/best/ instead of base pretrained weights
  - Uses GPU (bfloat16) for speed — 1703 images in ~15-20 mins vs hours on CPU
  - Saves to fine_tuned_results.txt (never overwrites preliminary_results.txt)
  - Logs progress every 100 images so you can monitor via tail -f
  - Writes inference_manifest.json on completion
"""

import os
import glob
import json
import time
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# ─── Config ───────────────────────────────────────────────────────────────────
CHECKPOINT_DIR   = "checkpoints/full_run/epoch_10"   # best epoch per manifest
FALLBACK_CKPT    = "checkpoints/full_run/epoch_09"   # fallback if epoch_10 missing
OUTPUT_FILE      = "results/fine_tuned_results.txt"
MANIFEST_OUT     = "results/inference_manifest.json"
LOG_EVERY        = 100                                # print progress every N images
MAX_NEW_TOKENS   = 100                                # cap generation length
# ─────────────────────────────────────────────────────────────────────────────


def find_checkpoint(base_dir: str) -> str:
    """
    Resolve the checkpoint path. Priority:
      1. epoch_10  (best per full_training_manifest.json)
      2. epoch_09  (fallback)
      3. Any epoch_ dir sorted descending
    """
    for ckpt in [CHECKPOINT_DIR, FALLBACK_CKPT]:
        full = os.path.join(base_dir, ckpt)
        if os.path.isdir(full):
            return full

    # Last resort — find highest epoch dir
    ckpt_root = os.path.join(base_dir, "checkpoints", "full_run")
    if os.path.isdir(ckpt_root):
        epoch_dirs = sorted(
            [d for d in os.listdir(ckpt_root) if d.startswith("epoch_")],
            reverse=True
        )
        if epoch_dirs:
            return os.path.join(ckpt_root, epoch_dirs[0])

    raise FileNotFoundError(
        f"No fine-tuned checkpoint found under {base_dir}/checkpoints/full_run/\n"
        "Make sure train_full.py completed successfully."
    )


def main():
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir    = os.path.join(base_dir, "data", "whole_multicare_dataset",
                               "vlm_mri_subset", "images")
    results_dir = os.path.join(base_dir, "results")
    output_file = os.path.join(base_dir, OUTPUT_FILE)
    manifest_out= os.path.join(base_dir, MANIFEST_OUT)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)

    print("=" * 60)
    print("  Fine-Tuned VLM Inference — May 11 (Anirudh)")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── 1. Resolve checkpoint ─────────────────────────────────────────────────
    ckpt_path = find_checkpoint(base_dir)
    print(f"\n[Step 1] Checkpoint : {ckpt_path}")

    # ── 2. Load images ────────────────────────────────────────────────────────
    print(f"\n[Step 2] Scanning images in {data_dir} ...")
    images = glob.glob(os.path.join(data_dir, "**", "*.webp"), recursive=True)
    images.sort()
    print(f"  Found {len(images)} images")

    if not images:
        print(f"  [ERROR] No .webp images found. Check data path.")
        return

    # ── 3. Load fine-tuned model ──────────────────────────────────────────────
    print(f"\n[Step 3] Loading fine-tuned model ...")
    processor = BlipProcessor.from_pretrained(ckpt_path)
    model     = BlipForConditionalGeneration.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM   : {alloc:.2f} GB allocated after model load")

    # ── 4. Run inference ──────────────────────────────────────────────────────
    print(f"\n[Step 4] Running inference on {len(images)} images ...")
    print(f"  Output → {output_file}\n")

    run_start  = time.time()
    success    = 0
    failed     = 0
    captions   = []

    with open(output_file, "w") as f:
        for i, img_path in enumerate(images, start=1):
            try:
                raw_image = Image.open(img_path).convert("RGB")

                inputs = processor(
                    images=raw_image,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                            enabled=(device.type == "cuda")):
                        out = model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS,
                            num_beams=4,           # beam search for better quality
                            early_stopping=True,
                        )

                caption  = processor.decode(out[0], skip_special_tokens=True).strip()
                img_name = os.path.basename(img_path)

                line = f"File: {img_name} | Caption: {caption}\n"
                f.write(line)
                f.flush()

                captions.append({"image": img_name, "caption": caption})
                success += 1

            except Exception as e:
                failed += 1
                img_name = os.path.basename(img_path)
                f.write(f"File: {img_name} | Caption: [ERROR: {str(e)[:50]}]\n")
                f.flush()

            # ── Progress log ──────────────────────────────────────────────
            if i % LOG_EVERY == 0 or i == len(images):
                elapsed = (time.time() - run_start) / 60
                rate    = i / max(elapsed, 0.01)
                eta     = (len(images) - i) / max(rate, 0.01)
                print(f"  [{i:>5}/{len(images)}]  "
                      f"success: {success}  failed: {failed}  "
                      f"elapsed: {elapsed:.1f}m  ETA: {eta:.1f}m")

    # ── 5. Write inference manifest ───────────────────────────────────────────
    total_time = (time.time() - run_start) / 60
    manifest   = {
        "run_type"        : "fine_tuned_inference",
        "checkpoint_used" : ckpt_path,
        "total_images"    : len(images),
        "successful"      : success,
        "failed"          : failed,
        "output_file"     : OUTPUT_FILE,
        "max_new_tokens"  : MAX_NEW_TOKENS,
        "num_beams"       : 4,
        "device"          : str(device),
        "total_time_mins" : round(total_time, 2),
        "status"          : "completed",
        "completed_at"    : time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("  INFERENCE COMPLETE")
    print(f"  Total images  : {len(images)}")
    print(f"  Successful    : {success}")
    print(f"  Failed        : {failed}")
    print(f"  Time taken    : {total_time:.1f} mins")
    print(f"  Output file   : {output_file}")
    print(f"  Manifest      : {manifest_out}")
    print("=" * 60)
    print("\nNext step → Kolla runs:")
    print("  python3 src/evaluate.py --model zero_shot --model fine_tuned")


if __name__ == "__main__":
    main()