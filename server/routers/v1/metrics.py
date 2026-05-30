from fastapi import APIRouter

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/summary")
def metrics_summary() -> dict:
    return {"status": "ok", "metrics": {}}
