"""
Document DTOs - Data Transfer Objects for document operations.

This module defines the request and response models for document management endpoints.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from src.domain.models.document import DocumentStatus


class DocumentResponse(BaseModel):
    """Response model for document information."""
    
    id: UUID = Field(..., description="Unique document ID")
    document_id: str = Field(..., description="ChromaDB document identifier")
    filename: str = Field(..., description="Original filename")
    user_id: str = Field(..., description="User who uploaded the document")
    file_size: int = Field(..., description="File size in bytes")
    total_chunks: int = Field(..., description="Number of chunks")
    total_characters: int = Field(..., description="Total characters")
    status: str = Field(..., description="Processing status")
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    processed_at: datetime | None = Field(default=None, description="Processing completion timestamp")
    deleted_at: datetime | None = Field(default=None, description="Soft delete timestamp")


class DocumentListResponse(BaseModel):
    """Response model for document list."""
    
    documents: list[DocumentResponse] = Field(default_factory=list, description="List of documents")
    total: int = Field(..., description="Total number of documents")
    skip: int = Field(..., description="Number of documents skipped")
    limit: int = Field(..., description="Maximum documents returned")


class DocumentSummaryResponse(BaseModel):
    """Response model for PDF summary generation."""
    
    document_id: str = Field(..., description="ChromaDB document identifier")
    filename: str = Field(..., description="Original filename")
    summary: str = Field(..., description="Generated summary")
    total_chunks: int = Field(..., description="Number of chunks processed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
