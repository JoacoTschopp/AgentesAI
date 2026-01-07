"""
PDF DTOs - Data Transfer Objects for PDF operations.

This module defines the request and response models for PDF processing endpoints.
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class PDFUploadResponse(BaseModel):
    """Response model for PDF upload and ingestion."""
    
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    total_chunks: int = Field(..., description="Number of chunks created")
    total_characters: int = Field(..., description="Total characters in document")
    user_id: str = Field(..., description="User who uploaded the document")
    status: str = Field(..., description="Ingestion status")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PDFQueryRequest(BaseModel):
    """Request model for querying PDF documents."""
    
    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(default=5, description="Number of results to return", ge=1, le=20)
    filter_user_id: str | None = Field(default=None, description="Filter by user ID")


class PDFQueryResult(BaseModel):
    """Individual query result from PDF search."""
    
    chunk_id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Source document ID")
    filename: str = Field(..., description="Source filename")
    text: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="Relevance score")
    chunk_index: int = Field(..., description="Chunk position in document")
    total_chunks: int = Field(..., description="Total chunks in document")


class PDFQueryResponse(BaseModel):
    """Response model for PDF query."""
    
    query: str = Field(..., description="Original query")
    results: list[PDFQueryResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(..., description="Number of results returned")


class PDFStatsResponse(BaseModel):
    """Response model for PDF statistics."""
    
    total_documents: int = Field(..., description="Total number of document chunks")
    collection_name: str = Field(..., description="ChromaDB collection name")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Collection metadata")
