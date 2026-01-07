"""
MCP CLI Entry Point.

This module provides the command-line interface for running the MCP server
with stdio transport, suitable for integration with MCP clients like Claude Desktop.
"""

import asyncio
import sys

import structlog

from src.infrastructure.config.settings import get_settings
from src.infrastructure.mcp.mcp_server import MCPServer
from src.application.services.chat_service import ChatService
from src.application.services.pdf_ingestion_service import PDFIngestionService
from src.application.services.agent_service import AgentService
from src.application.services.session_service import SessionService
from src.infrastructure.adapters.llm.circuit_breaker_llm import CircuitBreakerLLM
from src.infrastructure.adapters.persistence.mongodb_agent_repository import MongoDBAgentRepository
from src.infrastructure.adapters.persistence.mongodb_session_repository import MongoDBSessionRepository
from src.infrastructure.adapters.persistence.mongodb_document_repository import MongoDBDocumentRepository
from src.infrastructure.adapters.security.llmguard_sanitizer import LLMGuardSanitizer
from src.infrastructure.adapters.vector_store.chromadb_adapter import ChromaDBAdapter

logger = structlog.get_logger()


async def initialize_services():
    """Initialize all required services for MCP server."""
    settings = get_settings()
    
    # Initialize repositories
    agent_repo = MongoDBAgentRepository(settings)
    await agent_repo.initialize()
    
    session_repo = MongoDBSessionRepository(settings)
    await session_repo.initialize()
    
    document_repo = MongoDBDocumentRepository(settings)
    await document_repo.initialize()
    
    # Initialize LLM
    llm = CircuitBreakerLLM(settings)
    
    # Initialize sanitizer
    sanitizer = LLMGuardSanitizer(settings)
    
    # Initialize vector store
    vector_store = ChromaDBAdapter(settings)
    await vector_store.initialize()
    
    # Initialize services
    agent_service = AgentService(agent_repo)
    session_service = SessionService(session_repo)
    
    chat_service = ChatService(
        llm=llm,
        sanitizer=sanitizer,
        agent_service=agent_service,
        session_service=session_service
    )
    
    pdf_service = PDFIngestionService(
        llm=llm,
        vector_store=vector_store,
        document_repository=document_repo
    )
    
    return {
        "settings": settings,
        "chat_service": chat_service,
        "pdf_service": pdf_service,
        "agent_service": agent_service,
        "repos": {
            "agent": agent_repo,
            "session": session_repo,
            "document": document_repo,
            "vector": vector_store
        }
    }


async def cleanup_services(services: dict):
    """Cleanup all services."""
    repos = services.get("repos", {})
    
    if "agent" in repos:
        await repos["agent"].close()
    if "session" in repos:
        await repos["session"].close()
    if "document" in repos:
        await repos["document"].close()
    if "vector" in repos:
        await repos["vector"].close()


async def main():
    """Main entry point for MCP CLI."""
    logger.info("mcp_cli_starting")
    
    services = None
    try:
        # Initialize services
        services = await initialize_services()
        settings = services["settings"]
        
        # Create MCP server
        mcp_server = MCPServer(
            settings=settings,
            chat_service=services["chat_service"],
            pdf_service=services["pdf_service"],
            agent_service=services["agent_service"]
        )
        
        logger.info("mcp_server_ready", transport="stdio")
        
        # Run stdio server
        await mcp_server.run_stdio()
        
    except KeyboardInterrupt:
        logger.info("mcp_cli_interrupted")
    except Exception as e:
        logger.error("mcp_cli_error", error=str(e))
        sys.exit(1)
    finally:
        if services:
            await cleanup_services(services)
        logger.info("mcp_cli_stopped")


if __name__ == "__main__":
    asyncio.run(main())
