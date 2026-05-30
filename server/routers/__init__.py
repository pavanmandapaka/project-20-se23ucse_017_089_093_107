from .health import router as health_router
from .v1 import router as v1_router

__all__ = ["health_router", "v1_router"]
