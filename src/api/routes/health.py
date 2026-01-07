"""
Health Routes - Health check and system status endpoints.

This module provides endpoints for monitoring the application's
health status and component availability.
"""

from typing import Any

from fastapi import APIRouter, Depends, status

from src.api.dependencies.services import get_settings, get_llm
from src.infrastructure.config.settings import Settings
from src.infrastructure.adapters.llm.circuit_breaker_llm import CircuitBreakerLLM


router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check if the API is running and healthy.",
)
async def health_check() -> dict[str, str]:
    """
    Basic health check endpoint.
    
    Returns:
        Health status message.
    """
    return {"status": "healthy"}


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness Check",
    description="Check if the API is ready to handle requests.",
)
async def readiness_check(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Readiness check with component status.
    
    Args:
        settings: Application settings.
        
    Returns:
        Readiness status with component details.
    """
    return {
        "status": "ready",
        "environment": settings.environment,
        "version": settings.app_version,
    }


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness Check",
    description="Check if the API process is alive.",
)
async def liveness_check() -> dict[str, str]:
    """
    Liveness check endpoint.
    
    Returns:
        Liveness status.
    """
    return {"status": "alive"}


@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="System Status",
    description="Get detailed system status including LLM provider info.",
)
async def system_status(
    settings: Settings = Depends(get_settings),
    llm: CircuitBreakerLLM = Depends(get_llm),
) -> dict[str, Any]:
    """
    Detailed system status.
    
    Args:
        settings: Application settings.
        llm: LLM adapter with circuit breaker.
        
    Returns:
        Comprehensive system status.
    """
    return {
        "status": "operational",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "llm": {
            "available": llm.is_available(),
            "current_provider": llm.get_current_provider(),
            "current_model": llm.get_model_name(),
            "circuit_state": llm.get_circuit_state(),
        },
        "features": {
            "llmguard_enabled": settings.llmguard_enabled,
            "session_ttl_hours": settings.session_ttl_hours,
        },
    }
