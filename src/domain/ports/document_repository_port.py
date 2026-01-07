"""
Document Repository Port - Abstract interface for document persistence.

This module defines the contract for document storage operations.
"""

from abc import ABC, abstractmethod
from uuid import UUID

from src.domain.models.document import Document


class DocumentRepositoryPort(ABC):
    """
    Abstract interface for document repository operations.
    
    All document repository adapters must implement this interface.
    """

    @abstractmethod
    async def add(self, document: Document) -> Document:
        """
        Add a new document to the repository.
        
        Args:
            document: Document to add
            
        Returns:
            The added document
        """
        pass

    @abstractmethod
    async def get_by_id(self, document_id: UUID) -> Document | None:
        """
        Get a document by its ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_by_document_id(self, document_id: str) -> Document | None:
        """
        Get a document by its ChromaDB document ID.
        
        Args:
            document_id: ChromaDB document identifier
            
        Returns:
            Document if found, None otherwise
        """
        pass

    @abstractmethod
    async def exists(self, document_id: str) -> bool:
        """
        Check if a document exists by ChromaDB document ID.
        
        Args:
            document_id: ChromaDB document identifier
            
        Returns:
            True if document exists and is not deleted
        """
        pass

    @abstractmethod
    async def list_by_user(
        self,
        user_id: str,
        include_deleted: bool = False,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Document]:
        """
        List documents for a specific user.
        
        Args:
            user_id: User identifier
            include_deleted: Whether to include soft-deleted documents
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            
        Returns:
            List of documents
        """
        pass

    @abstractmethod
    async def list_all(
        self,
        include_deleted: bool = False,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Document]:
        """
        List all documents.
        
        Args:
            include_deleted: Whether to include soft-deleted documents
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            
        Returns:
            List of documents
        """
        pass

    @abstractmethod
    async def update(self, document: Document) -> Document:
        """
        Update a document.
        
        Args:
            document: Document with updated data
            
        Returns:
            Updated document
        """
        pass

    @abstractmethod
    async def soft_delete(self, document_id: UUID) -> bool:
        """
        Soft delete a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if deletion succeeded
        """
        pass

    @abstractmethod
    async def count(self, user_id: str | None = None, include_deleted: bool = False) -> int:
        """
        Count documents.
        
        Args:
            user_id: Optional user filter
            include_deleted: Whether to include soft-deleted documents
            
        Returns:
            Number of documents
        """
        pass
