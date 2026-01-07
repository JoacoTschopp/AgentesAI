"""
Vector Store Port - Abstract interface for vector database operations.

This module defines the contract for vector storage operations,
enabling different vector databases (Pinecone, Milvus, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID


class VectorStorePort(ABC):
    """
    Abstract interface for vector store operations.
    
    All vector store adapters must implement this interface
    to ensure consistent vector storage and retrieval behavior.
    """

    @abstractmethod
    async def upsert(
        self,
        vectors: list[list[float]],
        ids: list[str],
        metadata: list[dict[str, Any]] | None = None,
        namespace: str | None = None,
    ) -> bool:
        """
        Insert or update vectors in the store.
        
        Args:
            vectors: List of embedding vectors.
            ids: Unique identifiers for each vector.
            metadata: Optional metadata for each vector.
            namespace: Optional namespace/partition.
            
        Returns:
            True if operation succeeded.
            
        Raises:
            VectorStoreError: If operation fails.
        """
        pass

    @abstractmethod
    async def query(
        self,
        vector: list[float],
        top_k: int = 10,
        namespace: str | None = None,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Query for similar vectors.
        
        Args:
            vector: The query vector.
            top_k: Number of results to return.
            namespace: Optional namespace to search within.
            filter: Optional metadata filter.
            include_metadata: Whether to include metadata in results.
            
        Returns:
            List of matching results with scores and metadata.
            
        Raises:
            VectorStoreError: If query fails.
        """
        pass

    @abstractmethod
    async def delete(
        self,
        ids: list[str] | None = None,
        namespace: str | None = None,
        filter: dict[str, Any] | None = None,
    ) -> bool:
        """
        Delete vectors from the store.
        
        Args:
            ids: Optional list of IDs to delete.
            namespace: Optional namespace.
            filter: Optional metadata filter for deletion.
            
        Returns:
            True if operation succeeded.
            
        Raises:
            VectorStoreError: If deletion fails.
        """
        pass

    @abstractmethod
    async def get_stats(self, namespace: str | None = None) -> dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Args:
            namespace: Optional namespace to get stats for.
            
        Returns:
            Dictionary with store statistics.
            
        Raises:
            VectorStoreError: If retrieval fails.
        """
        pass
