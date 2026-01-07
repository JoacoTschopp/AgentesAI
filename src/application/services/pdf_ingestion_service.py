"""
PDF Ingestion Service - Business logic for PDF processing and vector storage.

This module provides the service layer for handling PDF uploads,
text extraction, chunking, embedding generation, and storage in ChromaDB.
"""

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog
from pypdf import PdfReader

from src.domain.ports.llm_port import LLMPort
from src.domain.ports.document_repository_port import DocumentRepositoryPort
from src.domain.models.document import Document, DocumentStatus
from src.infrastructure.adapters.vector_store.chromadb_adapter import ChromaDBAdapter


logger = structlog.get_logger()


class PDFIngestionService:
    """
    Service for handling PDF ingestion and vector storage.
    
    Orchestrates the complete PDF processing flow:
    - PDF text extraction
    - Text chunking with fixed size
    - Embedding generation
    - Storage in ChromaDB
    
    Attributes:
        llm: Language model adapter for embeddings
        vector_store: ChromaDB adapter for storage
        chunk_size: Fixed size for text chunks
        chunk_overlap: Overlap between chunks
    """

    def __init__(
        self,
        llm: LLMPort,
        vector_store: ChromaDBAdapter,
        document_repository: DocumentRepositoryPort,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize PDF ingestion service.
        
        Args:
            llm: LLM adapter for generating embeddings
            vector_store: ChromaDB adapter
            document_repository: Document repository for MongoDB
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.llm = llm
        self.vector_store = vector_store
        self.document_repository = document_repository
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If PDF cannot be read
        """
        try:
            reader = PdfReader(pdf_path)
            text_content = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_content.append(text)
            
            full_text = "\n\n".join(text_content)
            
            logger.info(
                "pdf_text_extracted",
                pdf_path=pdf_path,
                pages=len(reader.pages),
                text_length=len(full_text)
            )
            
            return full_text
            
        except Exception as e:
            logger.error("pdf_extraction_failed", pdf_path=pdf_path, error=str(e))
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary if possible
            if end < text_length:
                # Look for sentence endings near the chunk boundary
                last_period = chunk.rfind('. ')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.7:  # Only break if we're past 70% of chunk
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        
        logger.info(
            "text_chunked",
            total_chunks=len(chunks),
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap
        )
        
        return chunks

    def generate_document_id(self, pdf_path: str) -> str:
        """
        Generate a unique document ID based on file path.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Unique document ID
        """
        # Use hash of filename for consistent IDs
        filename = Path(pdf_path).name
        return hashlib.md5(filename.encode()).hexdigest()

    async def ingest_pdf(
        self,
        pdf_path: str,
        user_id: str,
        additional_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Process a PDF file and store it in ChromaDB.
        
        Args:
            pdf_path: Path to the PDF file
            user_id: User who uploaded the PDF
            additional_metadata: Optional additional metadata
            
        Returns:
            Dictionary with ingestion results
            
        Raises:
            ValueError: If ingestion fails
        """
        try:
            # Generate document ID
            doc_id = self.generate_document_id(pdf_path)
            filename = Path(pdf_path).name
            
            # Check if document already exists in MongoDB
            existing_doc = await self.document_repository.get_by_document_id(doc_id)
            if existing_doc and not existing_doc.is_deleted():
                logger.info(
                    "document_already_exists",
                    document_id=doc_id,
                    filename=filename
                )
                return {
                    "document_id": doc_id,
                    "filename": filename,
                    "total_chunks": existing_doc.total_chunks,
                    "total_characters": existing_doc.total_characters,
                    "user_id": user_id,
                    "status": "already_exists",
                    "message": "Document already processed"
                }
            
            # Get file size
            file_size = Path(pdf_path).stat().st_size
            
            # Create document entity
            document = Document(
                document_id=doc_id,
                filename=filename,
                user_id=user_id,
                file_size=file_size,
                total_chunks=0,
                total_characters=0,
                status=DocumentStatus.PROCESSING,
                metadata=additional_metadata or {}
            )
            
            # Save to MongoDB
            document = await self.document_repository.add(document)
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            if not text or len(text.strip()) < 10:
                document.mark_as_failed()
                await self.document_repository.update(document)
                raise ValueError("PDF contains no extractable text")
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            if not chunks:
                document.mark_as_failed()
                await self.document_repository.update(document)
                raise ValueError("Failed to create chunks from PDF text")
            
            # Generate embeddings for all chunks
            logger.info("generating_embeddings", chunk_count=len(chunks))
            embeddings = await self.llm.get_embeddings(chunks)
            
            # Prepare metadata and IDs
            
            chunk_ids = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_ids.append(chunk_id)
                
                metadata = {
                    "document_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "user_id": user_id,
                    "text": chunk,  # Store the actual text in metadata
                }
                
                if additional_metadata:
                    metadata.update(additional_metadata)
                
                metadatas.append(metadata)
            
            # Store in ChromaDB
            success = await self.vector_store.add_documents(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=chunk_ids,
                namespace="pdf_documents"
            )
            
            if not success:
                document.mark_as_failed()
                await self.document_repository.update(document)
                raise ValueError("Failed to store documents in ChromaDB")
            
            # Update document status
            document.total_chunks = len(chunks)
            document.total_characters = len(text)
            document.mark_as_completed()
            await self.document_repository.update(document)
            
            result = {
                "document_id": doc_id,
                "filename": filename,
                "total_chunks": len(chunks),
                "total_characters": len(text),
                "user_id": user_id,
                "status": "success"
            }
            
            logger.info(
                "pdf_ingestion_complete",
                document_id=doc_id,
                chunks=len(chunks)
            )
            
            return result
            
        except Exception as e:
            logger.error("pdf_ingestion_failed", pdf_path=pdf_path, error=str(e))
            raise ValueError(f"PDF ingestion failed: {str(e)}")

    async def query_documents(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query the vector store for relevant document chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            # Generate embedding for the query
            query_embeddings = await self.llm.get_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Query ChromaDB
            results = await self.vector_store.query(
                vector=query_embedding,
                top_k=top_k,
                namespace="pdf_documents",
                filter=filter_metadata,
                include_metadata=True
            )
            
            logger.info(
                "document_query_complete",
                query_length=len(query),
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            logger.error("document_query_failed", error=str(e))
            return []

    async def get_document_stats(self) -> dict[str, Any]:
        """
        Get statistics about stored documents.
        
        Returns:
            Dictionary with document statistics
        """
        return await self.vector_store.get_stats(namespace="pdf_documents")
