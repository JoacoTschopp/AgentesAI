"""
Service Dependencies - Dependency injection for application services.

This module provides FastAPI dependency functions for injecting
services and adapters into route handlers.
"""

from functools import lru_cache

from src.infrastructure.config.settings import Settings, get_settings as _get_settings
from src.infrastructure.adapters.llm.circuit_breaker_llm import CircuitBreakerLLM
from src.infrastructure.adapters.persistence.mongodb_session_repository import MongoDBSessionRepository
from src.infrastructure.adapters.persistence.mongodb_document_repository import MongoDBDocumentRepository
from src.infrastructure.adapters.vector_store.pinecone_adapter import PineconeAdapter
from src.infrastructure.adapters.vector_store.chromadb_adapter import ChromaDBAdapter
from src.infrastructure.adapters.security.llmguard_sanitizer import LLMGuardSanitizer
from src.application.services.agent_service import AgentService
from src.application.services.session_service import SessionService
from src.application.services.chat_service import ChatService
from src.application.services.pdf_ingestion_service import PDFIngestionService
from src.domain.ports.llm_port import LLMPort
from src.domain.ports.session_repository_port import SessionRepositoryPort
from src.domain.ports.document_repository_port import DocumentRepositoryPort
from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.ports.prompt_sanitizer_port import PromptSanitizerPort


def get_settings() -> Settings:
    """
    Get application settings.
    
    Returns:
        Cached Settings instance.
    """
    return _get_settings()


@lru_cache
def get_llm() -> LLMPort:
    """
    Get LLM adapter with circuit breaker.
    
    Returns:
        CircuitBreakerLLM instance.
    """
    settings = get_settings()
    return CircuitBreakerLLM(settings)


@lru_cache
def get_sanitizer() -> PromptSanitizerPort:
    """
    Get prompt sanitizer.
    
    Returns:
        LLMGuardSanitizer instance.
    """
    settings = get_settings()
    return LLMGuardSanitizer(settings)


@lru_cache
def get_session_repository() -> SessionRepositoryPort:
    """
    Get session repository.
    
    Returns:
        MongoDBSessionRepository instance.
    """
    settings = get_settings()
    return MongoDBSessionRepository(settings)


@lru_cache
def get_vector_store() -> VectorStorePort:
    """
    Get vector store adapter.
    
    Returns:
        PineconeAdapter instance.
    """
    settings = get_settings()
    return PineconeAdapter(settings)


@lru_cache
def get_agent_service() -> AgentService:
    """
    Get agent service.
    
    Returns:
        AgentService instance.
    """
    return AgentService()


@lru_cache
def get_session_service() -> SessionService:
    """
    Get session service.
    
    Returns:
        SessionService instance.
    """
    settings = get_settings()
    repository = get_session_repository()
    return SessionService(repository, settings)


@lru_cache
def get_chat_service() -> ChatService:
    """
    Get chat service.
    
    Returns:
        ChatService instance.
    """
    llm = get_llm()
    sanitizer = get_sanitizer()
    agent_service = get_agent_service()
    session_service = get_session_service()
    
    return ChatService(llm, sanitizer, agent_service, session_service)


@lru_cache
def get_chromadb_adapter() -> ChromaDBAdapter:
    """
    Get ChromaDB adapter.
    
    Returns:
        ChromaDBAdapter instance.
    """
    settings = get_settings()
    return ChromaDBAdapter(settings)


@lru_cache
def get_document_repository() -> DocumentRepositoryPort:
    """
    Get document repository.
    
    Returns:
        MongoDBDocumentRepository instance.
    """
    settings = get_settings()
    return MongoDBDocumentRepository(settings)


@lru_cache
def get_pdf_ingestion_service() -> PDFIngestionService:
    """
    Get PDF ingestion service.
    
    Returns:
        PDFIngestionService instance.
    """
    llm = get_llm()
    chromadb = get_chromadb_adapter()
    doc_repo = get_document_repository()
    return PDFIngestionService(llm, chromadb, doc_repo, chunk_size=1000, chunk_overlap=200)
