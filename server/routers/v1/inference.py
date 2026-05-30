from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/inference", tags=["inference"])


class InferenceRequest(BaseModel):
    prompt: str
    image_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@router.post("/run")
def run_inference(payload: InferenceRequest) -> dict:
    return {
        "status": "queued",
        "prompt": payload.prompt,
        "image_id": payload.image_id,
    }
