"""Health check endpoints."""

import time
from fastapi import APIRouter
from app.models.schemas import HealthCheckResponse
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Check the health status of the application and its services.
    
    Returns service availability status for monitoring purposes.
    """
    services = {
        "assemblyai": "available" if settings.assemblyai_api_key else "unavailable",
        "gemini": "available" if settings.gemini_api_key else "unavailable",
        "murf": "available" if settings.murf_api_key else "unavailable"
    }
    
    # Determine overall status
    unavailable_services = [name for name, status in services.items() if status == "unavailable"]
    
    if not unavailable_services:
        overall_status = "healthy"
    elif len(unavailable_services) < len(services):
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    logger.info(f"Health check: {overall_status}, services: {services}")
    
    return HealthCheckResponse(
        status=overall_status,
        services=services,
        timestamp=time.time()
    )
