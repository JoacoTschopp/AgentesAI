"""
PDF Routes - Endpoints for PDF upload, ingestion, and retrieval.

This module provides API endpoints for:
- Uploading and processing PDF files
- Querying stored PDF documents
- Getting statistics about stored documents
"""

import os
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile

from src.domain.dtos.pdf_dto import (
    PDFStatsResponse,
    PDFUploadResponse,
)
from src.domain.dtos.document_dto import (
    DocumentListResponse,
    DocumentResponse,
)
from src.api.dependencies.services import get_pdf_ingestion_service, get_document_repository
from src.application.services.pdf_ingestion_service import PDFIngestionService
from src.domain.ports.document_repository_port import DocumentRepositoryPort


logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/pdf", tags=["PDF Processing"])


@router.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    x_user_id: str = Header(..., description="User identifier"),
    pdf_service: PDFIngestionService = Depends(get_pdf_ingestion_service),
) -> PDFUploadResponse:
    """
    Upload and process a PDF file.
    
    This endpoint:
    1. Receives a PDF file
    2. Extracts text content
    3. Chunks the text with fixed size
    4. Generates embeddings
    5. Stores in ChromaDB for later retrieval
    
    Args:
        file: PDF file upload
        x_user_id: User identifier from header
        pdf_service: PDF ingestion service
        
    Returns:
        Upload response with document details
        
    Raises:
        HTTPException: If upload or processing fails
    """
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    logger.info(
        "pdf_upload_started",
        filename=file.filename,
        user_id=x_user_id,
        content_type=file.content_type
    )
    
    # Create temporary directory for uploads
    upload_dir = Path("./temp_uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Save uploaded file temporarily
    temp_file_path = upload_dir / file.filename
    
    try:
        # Write uploaded file to disk
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info("pdf_saved_temporarily", path=str(temp_file_path))
        
        # Process the PDF
        result = await pdf_service.ingest_pdf(
            pdf_path=str(temp_file_path),
            user_id=x_user_id,
            additional_metadata={
                "original_filename": file.filename,
                "content_type": file.content_type or "application/pdf"
            }
        )
        
        return PDFUploadResponse(**result)
        
    except ValueError as e:
        logger.error("pdf_processing_failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("pdf_upload_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process PDF")
    finally:
        # Clean up temporary file
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
                logger.info("temp_file_cleaned", path=str(temp_file_path))
            except Exception as e:
                logger.warning("temp_file_cleanup_failed", error=str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    x_user_id: str = Header(..., description="User identifier"),
    skip: int = 0,
    limit: int = 100,
    include_deleted: bool = False,
    doc_repo: DocumentRepositoryPort = Depends(get_document_repository),
) -> DocumentListResponse:
    """
    List all documents uploaded by the user.
    
    Args:
        x_user_id: User identifier
        skip: Number of documents to skip (pagination)
        limit: Maximum number of documents to return
        include_deleted: Whether to include soft-deleted documents
        doc_repo: Document repository
        
    Returns:
        List of documents with metadata
    """
    try:
        logger.info(
            "listing_documents",
            user_id=x_user_id,
            skip=skip,
            limit=limit
        )
        
        # Get documents for user
        documents = await doc_repo.list_by_user(
            user_id=x_user_id,
            include_deleted=include_deleted,
            skip=skip,
            limit=limit
        )
        
        # Get total count
        total = await doc_repo.count(user_id=x_user_id, include_deleted=include_deleted)
        
        # Convert to response DTOs
        document_responses = [
            DocumentResponse(
                id=doc.id,
                document_id=doc.document_id,
                filename=doc.filename,
                user_id=doc.user_id,
                file_size=doc.file_size,
                total_chunks=doc.total_chunks,
                total_characters=doc.total_characters,
                status=doc.status.value if hasattr(doc.status, 'value') else doc.status,
                uploaded_at=doc.uploaded_at,
                processed_at=doc.processed_at,
                deleted_at=doc.deleted_at
            )
            for doc in documents
        ]
        
        logger.info(
            "documents_listed",
            user_id=x_user_id,
            count=len(document_responses),
            total=total
        )
        
        return DocumentListResponse(
            documents=document_responses,
            total=total,
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        logger.error("list_documents_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list documents")


@router.get("/stats", response_model=PDFStatsResponse)
async def get_pdf_stats(
    pdf_service: PDFIngestionService = Depends(get_pdf_ingestion_service),
) -> PDFStatsResponse:
    """
    Get statistics about stored PDF documents.
    
    Returns:
        Statistics about the ChromaDB collection
    """
    try:
        stats = await pdf_service.get_document_stats()
        
        return PDFStatsResponse(
            total_documents=stats.get('total_vectors', 0),
            collection_name=stats.get('collection_name', 'pdf_documents'),
            metadata=stats.get('metadata', {})
        )
        
    except Exception as e:
        logger.error("get_stats_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")
