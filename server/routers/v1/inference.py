import os
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from server.db import get_inference, list_inferences, log_inference

router = APIRouter(prefix="/inference", tags=["inference"])


BASE_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"


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

    return {
        "status": "received",
        "inference_id": inference_id,
        "image_path": relative_path,
        "caption": "Image received and queued for processing.",
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
