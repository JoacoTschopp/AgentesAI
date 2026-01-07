"""
Chat DTOs - Data Transfer Objects for chat/conversation API operations.

This module defines the request and response schemas for chat-related
API endpoints, including streaming responses.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequestDTO(BaseModel):
    """
    DTO for chat request.
    
    Attributes:
        message: The user's input message.
        session_id: Optional session ID for conversation continuity.
        agent_id: Optional specific agent ID to use.
        stream: Whether to stream the response.
        metadata: Additional request metadata.
    """
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="User input message"
    )
    session_id: UUID | None = Field(default=None, description="Session ID for continuity")
    agent_id: UUID | None = Field(default=None, description="Specific agent to use")
    stream: bool = Field(default=False, description="Enable response streaming")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Request metadata")

    class Config:
        """Pydantic configuration."""
        
        json_schema_extra = {
            "example": {
                "message": "Hello, can you help me analyze this document?",
                "stream": False,
            }
        }


class ChatResponseDTO(BaseModel):
    """
    DTO for chat response.
    
    Attributes:
        id: Unique response identifier.
        session_id: Session ID for the conversation.
        agent_id: Agent that generated the response.
        message: The assistant's response message.
        tokens_used: Token usage information.
        model: The model used for generation.
        metadata: Additional response metadata.
        created_at: Response timestamp.
    """
    
    id: UUID = Field(..., description="Response identifier")
    session_id: UUID = Field(..., description="Session identifier")
    agent_id: UUID = Field(..., description="Agent identifier")
    message: str = Field(..., description="Assistant response")
    tokens_used: dict[str, int] = Field(
        default_factory=dict,
        description="Token usage stats"
    )
    model: str = Field(..., description="Model used")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")

    class Config:
        """Pydantic configuration."""
        
        from_attributes = True


class StreamChunkDTO(BaseModel):
    """
    DTO for streaming response chunks.
    
    Attributes:
        chunk: The text chunk content.
        session_id: Session identifier.
        is_final: Whether this is the final chunk.
        metadata: Chunk metadata.
    """
    
    chunk: str = Field(..., description="Text chunk content")
    session_id: UUID = Field(..., description="Session identifier")
    is_final: bool = Field(default=False, description="Is final chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
