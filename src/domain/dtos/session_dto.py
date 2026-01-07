"""
Session DTOs - Data Transfer Objects for session API operations.

This module defines the request and response schemas for session-related
API endpoints.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from src.domain.models.session import SessionStatus


class SessionCreateDTO(BaseModel):
    """
    DTO for creating a new session.
    
    Attributes:
        user_id: Identifier of the user.
        agent_id: Optional agent to associate with the session.
        context: Initial session context data.
        metadata: Additional session metadata.
    """
    
    user_id: str = Field(..., min_length=1, max_length=100, description="User identifier")
    agent_id: UUID | None = Field(default=None, description="Agent to associate")
    context: dict[str, Any] = Field(default_factory=dict, description="Initial context")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Session metadata")

    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "context": {"preferences": {"language": "en"}},
            }
        }


class SessionResponseDTO(BaseModel):
    """
    DTO for session API responses.
    
    Attributes:
        id: Unique session identifier.
        user_id: User identifier.
        agent_id: Associated agent identifier.
        status: Current session status.
        context: Session context data.
        metadata: Session metadata.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        expires_at: Expiration timestamp.
    """
    
    id: UUID = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")
    agent_id: UUID | None = Field(default=None, description="Agent identifier")
    status: SessionStatus = Field(..., description="Session status")
    context: dict[str, Any] = Field(..., description="Session context")
    metadata: dict[str, Any] = Field(..., description="Session metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")
    expires_at: datetime | None = Field(default=None, description="Expiration timestamp")

    class Config:
        """Pydantic configuration."""
        
        from_attributes = True
        use_enum_values = True
