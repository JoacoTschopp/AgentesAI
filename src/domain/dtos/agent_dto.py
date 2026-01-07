"""
Agent DTOs - Data Transfer Objects for Agent API operations.

This module defines the request and response schemas for agent-related
API endpoints, ensuring proper validation and serialization.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from src.domain.models.agent import AgentType


class AgentCreateDTO(BaseModel):
    """
    DTO for creating a new agent.
    
    Attributes:
        name: Human-readable name for the agent.
        agent_type: The type/category of the agent.
        description: Optional description of the agent's purpose.
        system_prompt: Optional custom system prompt.
        temperature: LLM temperature parameter.
        max_tokens: Maximum tokens for responses.
        metadata: Additional custom configuration.
    """
    
    name: str = Field(..., min_length=1, max_length=100, description="Agent name")
    agent_type: AgentType = Field(..., description="Type of the agent")
    description: str = Field(default="", max_length=500, description="Agent description")
    system_prompt: str = Field(
        default="You are a helpful AI assistant.",
        max_length=10000,
        description="System prompt"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=4096, ge=1, le=128000, description="Max tokens")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom metadata")

    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "name": "Research Assistant",
                "agent_type": "conversational",
                "description": "Helps with research and analysis tasks",
                "temperature": 0.7,
                "max_tokens": 4096,
            }
        }


class AgentUpdateDTO(BaseModel):
    """
    DTO for updating an existing agent.
    
    All fields are optional to support partial updates.
    """
    
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    system_prompt: str | None = Field(default=None, max_length=10000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=128000)
    metadata: dict[str, Any] | None = Field(default=None)


class AgentResponseDTO(BaseModel):
    """
    DTO for agent API responses.
    
    Attributes:
        id: Unique agent identifier.
        name: Agent name.
        agent_type: Type of the agent.
        description: Agent description.
        system_prompt: Agent's system prompt.
        temperature: LLM temperature setting.
        max_tokens: Max tokens setting.
        metadata: Custom metadata.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """
    
    id: UUID = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent name")
    agent_type: str = Field(..., description="Type of the agent")
    description: str = Field(..., description="Agent description")
    system_prompt: str = Field(..., description="System prompt")
    temperature: float = Field(..., description="LLM temperature")
    max_tokens: int = Field(..., description="Max tokens")
    metadata: dict[str, Any] = Field(..., description="Custom metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")

    class Config:
        """Pydantic configuration."""
        
        from_attributes = True
