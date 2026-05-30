from fastapi import APIRouter

from . import datasets, inference, metrics, training

router = APIRouter(tags=["v1"])


@router.get("/version")
def version() -> dict:
    return {"api": "v1"}


router.include_router(inference.router)
router.include_router(datasets.router)
router.include_router(training.router)
router.include_router(metrics.router)
