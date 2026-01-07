"""
Agent Model - Represents an AI agent configuration.

This module defines the core Agent entity used throughout the application.
The Agent class encapsulates the configuration and metadata for different
types of AI agents that can be instantiated and executed.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """
    Enumeration of available agent types.
    
    Attributes:
        CONVERSATIONAL: General-purpose conversational agent for Q&A.
        PDF_ANALYZER: Specialized agent for PDF document analysis and summarization.
        SQL_QUERY: Agent capable of querying SQL databases.
        RAG: Retrieval-Augmented Generation agent with vector store integration.
    """
    
    CONVERSATIONAL = "conversational"
    PDF_ANALYZER = "pdf_analyzer"
    SQL_QUERY = "sql_query"
    RAG = "rag"


class Agent(BaseModel):
    """
    Core Agent entity representing an AI agent configuration.
    
    This class defines the structure and behavior configuration for AI agents.
    Each agent has a unique identifier, type, and customizable parameters.
    
    Attributes:
        id: Unique identifier for the agent instance.
        name: Human-readable name for the agent.
        agent_type: The type/category of the agent.
        description: Detailed description of the agent's purpose.
        system_prompt: The system prompt that defines agent behavior.
        temperature: LLM temperature parameter for response generation.
        max_tokens: Maximum tokens for LLM response.
        metadata: Additional custom configuration parameters.
        created_at: Timestamp of agent creation.
        updated_at: Timestamp of last modification.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique agent identifier")
    name: str = Field(..., min_length=1, max_length=100, description="Agent name")
    agent_type: AgentType = Field(..., description="Type of the agent")
    description: str = Field(default="", max_length=500, description="Agent description")
    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        description="System prompt defining agent behavior"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=4096, ge=1, le=128000, description="Max response tokens")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Update timestamp")

    class Config:
        """Pydantic configuration."""
        
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "name": "Customer Support Agent",
                "agent_type": "conversational",
                "description": "Handles customer inquiries and support tickets",
                "system_prompt": "You are a helpful customer support agent...",
                "temperature": 0.7,
                "max_tokens": 4096,
            }
        }
