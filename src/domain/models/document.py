"""
Document Entity - Domain model for PDF documents.

This module defines the document entity representing uploaded PDFs
with their metadata and tracking information.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Document processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class Document(BaseModel):
    """
    Document entity representing an uploaded PDF.
    
    Attributes:
        id: Unique document identifier
        document_id: ChromaDB document identifier (hash)
        filename: Original filename
        user_id: User who uploaded the document
        file_size: Size in bytes
        total_chunks: Number of chunks created
        total_characters: Total characters extracted
        status: Processing status
        uploaded_at: Upload timestamp
        processed_at: Processing completion timestamp
        deleted_at: Soft delete timestamp
        metadata: Additional metadata
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique document ID")
    document_id: str = Field(..., description="ChromaDB document identifier")
    filename: str = Field(..., description="Original filename")
    user_id: str = Field(..., description="User who uploaded the document")
    file_size: int = Field(..., description="File size in bytes", ge=0)
    total_chunks: int = Field(..., description="Number of chunks", ge=0)
    total_characters: int = Field(..., description="Total characters", ge=0)
    status: DocumentStatus = Field(default=DocumentStatus.UPLOADED)
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: datetime | None = Field(default=None)
    deleted_at: datetime | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
    
    def mark_as_processing(self) -> None:
        """Mark document as being processed."""
        self.status = DocumentStatus.PROCESSING
    
    def mark_as_completed(self) -> None:
        """Mark document as successfully processed."""
        self.status = DocumentStatus.COMPLETED
        self.processed_at = datetime.utcnow()
    
    def mark_as_failed(self) -> None:
        """Mark document processing as failed."""
        self.status = DocumentStatus.FAILED
        self.processed_at = datetime.utcnow()
    
    def soft_delete(self) -> None:
        """Soft delete the document."""
        self.status = DocumentStatus.DELETED
        self.deleted_at = datetime.utcnow()
    
    def is_deleted(self) -> bool:
        """Check if document is soft deleted."""
        return self.status == DocumentStatus.DELETED
    
    def is_completed(self) -> bool:
        """Check if document processing is completed."""
        return self.status == DocumentStatus.COMPLETED
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        data = self.model_dump()
        data['id'] = str(data['id'])
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Create from MongoDB dictionary."""
        if 'id' in data and isinstance(data['id'], str):
            data['id'] = UUID(data['id'])
        if '_id' in data:
            del data['_id']
        return cls(**data)
