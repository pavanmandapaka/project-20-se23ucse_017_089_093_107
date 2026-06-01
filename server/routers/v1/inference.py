import os
import threading
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
from safetensors.torch import load_file
from transformers import BlipForConditionalGeneration, BlipProcessor, BlipConfig

from server.db import get_inference, list_inferences, log_inference, update_inference_text

router = APIRouter(prefix="/inference", tags=["inference"])

BASE_DIR        = Path(__file__).resolve().parents[3]
UPLOAD_DIR      = BASE_DIR / "data" / "uploads"
CHECKPOINT_ROOT = BASE_DIR / "checkpoints" / "full_train"
BASE_MODEL_ID   = "Salesforce/blip-image-captioning-base"

_MODEL_CACHE: dict[str, Tuple[BlipProcessor, BlipForConditionalGeneration, torch.device]] = {}
_MODEL_LOCK = threading.Lock()


def _load_model(model_version: str) -> Tuple[BlipProcessor, BlipForConditionalGeneration, torch.device]:
    with _MODEL_LOCK:
        if model_version in _MODEL_CACHE:
            return _MODEL_CACHE[model_version]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_version.lower() in {"fine-tuned", "finetuned", "fine_tuned"}:
            # Find best checkpoint
            ckpt_file = None
            for ckpt in ["epoch_10", "epoch_09", "epoch_08"]:
                candidate = CHECKPOINT_ROOT / ckpt / "model.safetensors"
                if candidate.is_file():
                    ckpt_file = candidate
                    break

            if ckpt_file is None:
                raise RuntimeError(f"No checkpoint found in {CHECKPOINT_ROOT}")

            print(f"[Genni] Loading fine-tuned model from {ckpt_file} ...")
            processor = BlipProcessor.from_pretrained(BASE_MODEL_ID)
            config    = BlipConfig.from_pretrained(BASE_MODEL_ID)
            model     = BlipForConditionalGeneration(config)
            state_dict = load_file(str(ckpt_file), device="cpu")
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()
            print("[Genni] Fine-tuned model loaded OK")
        else:
            print(f"[Genni] Loading zero-shot model: {BASE_MODEL_ID} ...")
            processor = BlipProcessor.from_pretrained(BASE_MODEL_ID)
            model     = BlipForConditionalGeneration.from_pretrained(BASE_MODEL_ID)
            model.to(device)
            model.eval()
            print("[Genni] Zero-shot model loaded OK")

        _MODEL_CACHE[model_version] = (processor, model, device)
        return processor, model, device


def _generate_caption(
    image_path: Path,
    prompt: Optional[str],
    model_version: str,
) -> str:
    processor, model, device = _load_model(model_version)
    raw_image = Image.open(image_path).convert("RGB")

    if prompt:
        inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(images=raw_image, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=60,
            num_beams=4,
            early_stopping=True,
            repetition_penalty=2.0,
            no_repeat_ngram_size=4,
            length_penalty=0.8,
            min_length=10,
        )
    return processor.decode(out[0], skip_special_tokens=True).strip()


@router.post("/run")
async def run_inference(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    model_version: str = Form("fine_tuned"),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    safe_name = Path(file.filename).name
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid4().hex}_{safe_name}"
    stored_path = UPLOAD_DIR / stored_name

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    with stored_path.open("wb") as out_f:
        out_f.write(content)

    relative_path = os.path.relpath(stored_path, BASE_DIR)
    inference_id = log_inference(
        model_version=model_version,
        image_path=relative_path,
        generated_text="pending",
        ground_truth=None,
    )

    try:
        caption = _generate_caption(stored_path, prompt, model_version)
        update_inference_text(inference_id, caption)
    except Exception as exc:
        error_text = f"error: {str(exc)[:200]}"
        update_inference_text(inference_id, error_text)
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    return {
        "status": "completed",
        "inference_id": inference_id,
        "image_path": relative_path,
        "caption": caption,
        "prompt": prompt,
        "model_version": model_version,
    }


@router.get("/history")
def get_history(limit: int = 50, offset: int = 0) -> dict:
    return {"inferences": list_inferences(limit=limit, offset=offset)}


@router.get("/{inference_id}")
def get_inference_by_id(inference_id: int) -> dict:
    record = get_inference(inference_id)
    if not record:
        raise HTTPException(status_code=404, detail="Inference not found")
    return record
