"""
Settings - Application configuration using Pydantic Settings.

This module centralizes all application configuration, loading values
from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All configuration is centralized here, following the 12-factor app methodology.
    Settings are loaded from environment variables with fallback to .env file.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="AI Agents Platform", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment"
    )
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # API Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of workers")

    # OpenAI Configuration (Primary LLM)
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_api_base: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")
    openai_embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")

    # Google Gemini Configuration (Fallback LLM)
    google_api_key: str = Field(default="", description="Google API key")
    gemini_model: str = Field(default="gemini-1.5-pro", description="Gemini model name")

    # MongoDB Configuration
    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI"
    )
    mongodb_database: str = Field(default="ai_agents", description="MongoDB database name")
    mongodb_sessions_collection: str = Field(default="sessions", description="Sessions collection")
    mongodb_checkpoints_collection: str = Field(default="checkpoints", description="LangGraph checkpoints")

    # Pinecone Configuration
    pinecone_api_key: str = Field(default="", description="Pinecone API key")
    pinecone_environment: str = Field(default="", description="Pinecone environment")
    pinecone_index_name: str = Field(default="ai-agents", description="Pinecone index name")

    # Circuit Breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    circuit_breaker_recovery_timeout: int = 30

    # MCP (Model Context Protocol) settings
    mcp_active: bool = False
    mcp_transport: str = "both"  # stdio, sse, or both
    mcp_sse_port: int = 8001
    mcp_server_name: str = "ai-agents-mcp"

    # LLMGuard Configuration
    llmguard_enabled: bool = Field(default=True, description="Enable LLMGuard sanitization")
    llmguard_risk_threshold: float = Field(default=0.7, description="Risk score threshold")

    # Session Configuration
    session_ttl_hours: int = Field(default=24, description="Session TTL in hours")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Uses lru_cache to ensure settings are only loaded once.
    
    Returns:
        Settings instance with loaded configuration.
    """
    return Settings()
