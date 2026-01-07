"""
Simplified Health Check Route

Single endpoint that verifies all critical services are operational.
"""

import structlog
from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorClient

from src.infrastructure.config.settings import Settings, get_settings
from src.api.dependencies.services import get_llm


logger = structlog.get_logger()
router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(
    settings: Settings = Depends(get_settings)
) -> dict:
    """
    Comprehensive health check for all services.
    
    Verifies:
    - API is running
    - MongoDB connection
    - OpenAI LLM availability
    
    Returns:
        Health status with service details
    """
    health_status = {
        "status": "healthy",
        "services": {}
    }
    
    # Check MongoDB
    try:
        client = AsyncIOMotorClient(settings.mongodb_uri)
        await client.admin.command('ping')
        health_status["services"]["mongodb"] = {
            "status": "healthy",
            "uri": settings.mongodb_uri.split("@")[-1] if "@" in settings.mongodb_uri else settings.mongodb_uri
        }
        client.close()
    except Exception as e:
        logger.error("mongodb_health_check_failed", error=str(e))
        health_status["status"] = "degraded"
        health_status["services"]["mongodb"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Check OpenAI LLM
    try:
        llm = get_llm()
        health_status["services"]["llm"] = {
            "status": "healthy",
            "provider": "OpenAI",
            "model": settings.openai_model
        }
    except Exception as e:
        logger.error("llm_health_check_failed", error=str(e))
        health_status["status"] = "degraded"
        health_status["services"]["llm"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # List available agents
    health_status["agents"] = [
        {
            "name": "Conversational Assistant",
            "endpoint": "/api/v1/agents/conversational/chat"
        },
        {
            "name": "PDF Analyzer",
            "endpoint": "/api/v1/agents/pdf-analyzer/chat"
        },
        {
            "name": "Cypher Query Optimizer",
            "endpoint": "/api/v1/agents/cypher-query/chat"
        },
        {
            "name": "RAG Research Agent",
            "endpoint": "/api/v1/agents/rag/chat"
        }
    ]
    
    return health_status
