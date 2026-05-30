from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/training", tags=["training"])


class TrainingJobRequest(BaseModel):
    run_name: str
    epochs: int = 1
    notes: Optional[str] = None


@router.post("/jobs")
def create_job(payload: TrainingJobRequest) -> dict:
    return {
        "job_id": "job_0001",
        "status": "queued",
        "run_name": payload.run_name,
        "epochs": payload.epochs,
    }


@router.get("/jobs")
def list_jobs() -> dict:
    return {"jobs": []}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return {"job_id": job_id, "status": "running"}
