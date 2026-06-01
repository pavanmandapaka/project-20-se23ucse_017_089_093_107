import os, glob, json, time, torch, sys
from safetensors.torch import load_file
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipConfig
from PIL import Image

BASE_MODEL  = "models/blip/models--Salesforce--blip-image-captioning-base/snapshots/82a37760796d32b1411fe092ab5d4e227313294b"
CKPT_FILE   = "checkpoints/full_train/epoch_10/model.safetensors"
OUTPUT_FILE = "results/fine_tuned_results.txt"
MANIFEST    = "results/inference_manifest.json"
LOG_EVERY   = 100
MAX_TOKENS  = 60

def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()

def main():
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_file   = os.path.join(base_dir, CKPT_FILE)
    base_model  = os.path.join(base_dir, BASE_MODEL)
    data_dir    = os.path.join(base_dir, "data", "whole_multicare_dataset", "vlm_mri_subset", "images")
    output_file = os.path.join(base_dir, OUTPUT_FILE)
    manifest    = os.path.join(base_dir, MANIFEST)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

    log("=" * 60)
    log("  Fine-Tuned VLM Inference — May 11 (Anirudh)")
    log(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    log(f"\n[Step 1] Checkpoint : {ckpt_file}")
    log(f"\n[Step 2] Scanning images ...")
    images = sorted(glob.glob(os.path.join(data_dir, "**", "*.webp"), recursive=True))
    log(f"  Found {len(images)} images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"\n[Step 3] Device: {device}")
    if device.type == "cuda":
        log(f"  GPU: {torch.cuda.get_device_name(0)}")

    log(f"\n[Step 4] Loading processor ...")
    log("  calling BlipProcessor.from_pretrained ...")
    processor = BlipProcessor.from_pretrained(base_model)
    log("  Processor OK")

    log(f"\n[Step 5] Building model from config only ...")
    log("  calling BlipConfig.from_pretrained ...")
    config = BlipConfig.from_pretrained(base_model)
    log("  Config OK")
    log("  building model architecture ...")
    model = BlipForConditionalGeneration(config)
    log("  Model architecture built OK")

    log(f"\n[Step 6] Injecting fine-tuned weights ...")
    state_dict = load_file(ckpt_file, device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log(f"  Missing keys   : {len(missing)}")
    log(f"  Unexpected keys: {len(unexpected)}")
    model.to(device)
    model.eval()
    log("  Fine-tuned weights loaded OK")

    log(f"\n[Step 7] Running inference on {len(images)} images ...")
    run_start = time.time()
    success = failed = 0

    with open(output_file, "w") as f:
        for i, img_path in enumerate(images, start=1):
            try:
                raw_image = Image.open(img_path).convert("RGB")
                inputs = processor(images=raw_image, return_tensors="pt").to(device, torch.float16)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, num_beams=4, early_stopping=True, repetition_penalty=2.0, no_repeat_ngram_size=4, length_penalty=0.8, min_length=10)
                caption  = processor.decode(out[0], skip_special_tokens=True).strip()
                img_name = os.path.basename(img_path)
                f.write(f"File: {img_name} | Caption: {caption}\n")
                f.flush()
                success += 1
            except Exception as e:
                img_name = os.path.basename(img_path)
                f.write(f"File: {img_name} | Caption: [ERROR] {e}\n")
                f.flush()
                failed += 1

            if i % LOG_EVERY == 0 or i == len(images):
                elapsed = (time.time() - run_start) / 60
                rate    = i / max(elapsed, 0.01)
                eta     = (len(images) - i) / max(rate, 0.01)
                log(f"  [{i:>5}/{len(images)}]  success={success}  failed={failed}  elapsed={elapsed:.1f}m  ETA={eta:.1f}m")

    total_time = (time.time() - run_start) / 60
    result = {
        "checkpoint_used": CKPT_FILE,
        "total_images": len(images),
        "successful": success,
        "failed": failed,
        "total_time_mins": round(total_time, 2),
        "status": "completed",
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(manifest, "w") as f:
        json.dump(result, f, indent=2)

    log("\n" + "=" * 60)
    log(f"  INFERENCE COMPLETE  —  {success}/{len(images)} successful")
    log(f"  Time: {total_time:.1f} mins  |  Output: {output_file}")
    log("=" * 60)

if __name__ == "__main__":
    main()
