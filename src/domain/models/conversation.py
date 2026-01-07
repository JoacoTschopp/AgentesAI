"""
Conversation Model - Represents a conversation thread with messages.

This module defines the Conversation and Message entities that represent
the communication history between users and AI agents.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """
    Enumeration of message roles in a conversation.
    
    Attributes:
        USER: Message from the user/human.
        ASSISTANT: Message from the AI assistant.
        SYSTEM: System-level message or instruction.
        TOOL: Message representing tool/function output.
    """
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    """
    Message entity representing a single message in a conversation.
    
    Attributes:
        id: Unique identifier for the message.
        role: The role of the message sender.
        content: The text content of the message.
        metadata: Additional message metadata (tokens, model, etc.).
        created_at: Timestamp when the message was created.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique message identifier")
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Message content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Message metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")

    class Config:
        """Pydantic configuration."""
        
        use_enum_values = True


class Conversation(BaseModel):
    """
    Conversation entity representing a complete conversation thread.
    
    A conversation contains an ordered list of messages exchanged between
    a user and an AI agent within a session context.
    
    Attributes:
        id: Unique identifier for the conversation.
        session_id: Reference to the parent session.
        agent_id: Reference to the agent handling the conversation.
        messages: Ordered list of messages in the conversation.
        metadata: Additional conversation metadata.
        created_at: Timestamp of conversation creation.
        updated_at: Timestamp of last message addition.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique conversation identifier")
    session_id: UUID = Field(..., description="Parent session identifier")
    agent_id: UUID = Field(..., description="Agent handling the conversation")
    messages: list[Message] = Field(default_factory=list, description="Conversation messages")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Conversation metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Update timestamp")

    def add_message(self, role: MessageRole, content: str, metadata: dict[str, Any] | None = None) -> Message:
        """
        Add a new message to the conversation.
        
        Args:
            role: The role of the message sender.
            content: The text content of the message.
            metadata: Optional additional metadata.
            
        Returns:
            The newly created Message instance.
        """
        message = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message

    def get_message_history(self) -> list[dict[str, str]]:
        """
        Get the message history in a format suitable for LLM consumption.
        
        Returns:
            List of dictionaries with 'role' and 'content' keys.
        """
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    class Config:
        """Pydantic configuration."""
        
        use_enum_values = True
