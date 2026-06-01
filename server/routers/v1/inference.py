import os
import threading
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from server.db import get_inference, list_inferences, log_inference, update_inference_text

router = APIRouter(prefix="/inference", tags=["inference"])


BASE_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CHECKPOINT_ROOT = BASE_DIR / "checkpoints" / "full_run"
_MODEL_CACHE: dict[str, Tuple[BlipProcessor, BlipForConditionalGeneration, torch.device]] = {}
_MODEL_LOCK = threading.Lock()


def _resolve_model_path(model_version: str) -> str:
    if model_version in {"blip-image-captioning-base", "Salesforce/blip-image-captioning-base"}:
        return "Salesforce/blip-image-captioning-base"

    if model_version.lower() in {"fine-tuned", "finetuned", "fine_tuned"}:
        for ckpt in ["epoch_10", "epoch_09"]:
            candidate = CHECKPOINT_ROOT / ckpt
            if candidate.is_dir():
                return str(candidate)

    candidate = (BASE_DIR / model_version).resolve()
    if candidate.is_dir():
        return str(candidate)

    return model_version


def _load_model(model_version: str) -> Tuple[BlipProcessor, BlipForConditionalGeneration, torch.device]:
    resolved = _resolve_model_path(model_version)
    cache_key = f"{resolved}"

    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached:
            return cached

        processor = BlipProcessor.from_pretrained(resolved)
        model = BlipForConditionalGeneration.from_pretrained(resolved)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        _MODEL_CACHE[cache_key] = (processor, model, device)
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
            max_new_tokens=100,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return processor.decode(out[0], skip_special_tokens=True).strip()


@router.post("/run")
async def run_inference(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    model_version: str = Form("blip-image-captioning-base"),
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
