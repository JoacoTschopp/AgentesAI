"""
Main Application Entry Point - FastAPI application setup with MCP support.

This module configures and creates the FastAPI application instance,
setting up routes, middleware, and application lifecycle events.
Optionally runs MCP server in parallel based on MCP_ACTIVE configuration.
"""

import asyncio
import structlog
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.infrastructure.config.settings import get_settings
from src.api.routes.agents_chat import router as agents_chat_router
from src.api.routes.health_simple import router as health_router
from src.api.routes.pdf_routes import router as pdf_router
from src.api.middleware.error_handler import setup_exception_handlers
from src.api.middleware.logging import LoggingMiddleware
from src.api.dependencies.services import (
    get_session_repository,
    get_document_repository,
    get_chat_service,
    get_pdf_ingestion_service,
    get_agent_service,
)
from src.infrastructure.mcp.mcp_server import MCPServer


structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager with MCP support.
    
    Handles startup and shutdown events for the application.
    Optionally starts MCP server based on MCP_ACTIVE configuration.
    
    Args:
        app: FastAPI application instance.
    """
    settings = get_settings()
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        mcp_active=settings.mcp_active,
    )
    
    # Initialize repositories
    try:
        repository = get_session_repository()
        await repository.initialize()
        logger.info("session_repository_initialized")
    except Exception as e:
        logger.warning("session_repository_init_failed", error=str(e))
    
    try:
        doc_repository = get_document_repository()
        await doc_repository.initialize()
        logger.info("document_repository_initialized")
    except Exception as e:
        logger.warning("document_repository_init_failed", error=str(e))
    
    # Initialize MCP server if enabled
    mcp_server = None
    mcp_task = None
    
    if settings.mcp_active:
        try:
            logger.info("mcp_server_initializing", transport=settings.mcp_transport)
            
            # Get services for MCP
            chat_service = get_chat_service()
            pdf_service = get_pdf_ingestion_service()
            agent_service = get_agent_service()
            
            # Create MCP server
            mcp_server = MCPServer(
                settings=settings,
                chat_service=chat_service,
                pdf_service=pdf_service,
                agent_service=agent_service,
            )
            
            # Start MCP server in background (stdio transport)
            if settings.mcp_transport in ["stdio", "both"]:
                # Note: stdio MCP runs via CLI, not in FastAPI lifecycle
                logger.info("mcp_stdio_available_via_cli")
            
            # Start SSE transport if configured
            if settings.mcp_transport in ["sse", "both"]:
                mcp_task = asyncio.create_task(mcp_server.run_sse())
                logger.info("mcp_sse_server_started", port=settings.mcp_sse_port)
            
            logger.info("mcp_server_ready")
        except Exception as e:
            logger.error("mcp_server_init_failed", error=str(e))
    
    yield
    
    logger.info("application_shutting_down")
    
    # Shutdown MCP server
    if mcp_server:
        try:
            await mcp_server.shutdown()
            if mcp_task:
                mcp_task.cancel()
        except Exception as e:
            logger.warning("mcp_shutdown_error", error=str(e))
    
    # Close repositories
    try:
        repository = get_session_repository()
        await repository.close()
    except Exception:
        pass
    
    try:
        doc_repository = get_document_repository()
        await doc_repository.close()
    except Exception:
        pass


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description="Production-ready AI Agents Platform with Hexagonal Architecture",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(LoggingMiddleware)

    setup_exception_handlers(app)

    # Simplified routes - health, agent-specific chat, and PDF processing
    app.include_router(health_router)
    app.include_router(agents_chat_router)
    app.include_router(pdf_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers,
    )
