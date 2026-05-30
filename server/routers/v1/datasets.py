from fastapi import APIRouter

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("")
def list_datasets() -> dict:
    return {"datasets": []}


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str) -> dict:
    return {"dataset_id": dataset_id, "status": "ready"}
