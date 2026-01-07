"""
Session Model - Represents a user session for long-term persistence.

This module defines the Session entity that maintains state across
multiple conversations and interactions with AI agents.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    """
    Enumeration of session statuses.
    
    Attributes:
        ACTIVE: Session is currently active and accepting interactions.
        PAUSED: Session is temporarily paused.
        EXPIRED: Session has expired due to inactivity.
        TERMINATED: Session was explicitly terminated.
    """
    
    ACTIVE = "active"
    PAUSED = "paused"
    EXPIRED = "expired"
    TERMINATED = "terminated"


class Session(BaseModel):
    """
    Session entity for managing user interaction state.
    
    Sessions are persisted in MongoDB and maintain state across multiple
    conversations. They are used by LangGraph for checkpoint management.
    
    Attributes:
        id: Unique identifier for the session.
        user_id: Identifier of the user owning the session.
        agent_id: Identifier of the agent associated with the session.
        status: Current status of the session.
        context: Session-level context and state data.
        metadata: Additional session metadata.
        created_at: Timestamp of session creation.
        updated_at: Timestamp of last activity.
        expires_at: Timestamp when the session will expire.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique session identifier")
    user_id: str = Field(..., min_length=1, description="User identifier")
    agent_id: UUID | None = Field(default=None, description="Associated agent identifier")
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="Session status")
    context: dict[str, Any] = Field(default_factory=dict, description="Session context data")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Session metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Update timestamp")
    expires_at: datetime | None = Field(default=None, description="Expiration timestamp")

    def is_active(self) -> bool:
        """
        Check if the session is currently active.
        
        Returns:
            True if session status is ACTIVE, False otherwise.
        """
        return self.status == SessionStatus.ACTIVE

    def update_activity(self) -> None:
        """Update the session's last activity timestamp."""
        self.updated_at = datetime.utcnow()

    def terminate(self) -> None:
        """Terminate the session."""
        self.status = SessionStatus.TERMINATED
        self.updated_at = datetime.utcnow()

    class Config:
        """Pydantic configuration."""
        
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "status": "active",
                "context": {"preferences": {"language": "en"}},
            }
        }
