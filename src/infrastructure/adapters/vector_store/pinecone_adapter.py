"""
Pinecone Adapter - Pinecone implementation of VectorStorePort.

This module provides Pinecone-based vector storage for RAG operations,
supporting efficient similarity search and document retrieval.
"""

from typing import Any

import structlog
from pinecone import Pinecone, ServerlessSpec

from src.domain.ports.vector_store_port import VectorStorePort
from src.infrastructure.config.settings import Settings


logger = structlog.get_logger()


class PineconeAdapter(VectorStorePort):
    """
    Pinecone implementation of the vector store port.
    
    Provides vector storage and similarity search capabilities
    using Pinecone's serverless infrastructure.
    
    Attributes:
        client: Pinecone client instance.
        index: Pinecone index for operations.
    """

    def __init__(self, settings: Settings):
        """
        Initialize Pinecone adapter.
        
        Args:
            settings: Application settings with Pinecone configuration.
        """
        self._settings = settings
        self._index_name = settings.pinecone_index_name
        self._client = None
        self._index = None
        
        if settings.pinecone_api_key:
            self._client = Pinecone(api_key=settings.pinecone_api_key)
            self._initialize_index()

    def _initialize_index(self) -> None:
        """Initialize or connect to the Pinecone index."""
        if not self._client:
            return

        existing_indexes = [idx.name for idx in self._client.list_indexes()]
        
        if self._index_name not in existing_indexes:
            self._client.create_index(
                name=self._index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
            )
            logger.info("pinecone_index_created", index=self._index_name)
        
        self._index = self._client.Index(self._index_name)
        logger.info("pinecone_adapter_initialized", index=self._index_name)

    async def upsert(
        self,
        vectors: list[list[float]],
        ids: list[str],
        metadata: list[dict[str, Any]] | None = None,
        namespace: str | None = None,
    ) -> bool:
        """
        Insert or update vectors in Pinecone.
        
        Args:
            vectors: List of embedding vectors.
            ids: Unique identifiers for each vector.
            metadata: Optional metadata for each vector.
            namespace: Optional namespace/partition.
            
        Returns:
            True if operation succeeded.
        """
        if not self._index:
            logger.warning("pinecone_not_initialized")
            return False

        try:
            records = []
            for i, (vector, id_) in enumerate(zip(vectors, ids)):
                record = {
                    "id": id_,
                    "values": vector,
                }
                if metadata and i < len(metadata):
                    record["metadata"] = metadata[i]
                records.append(record)

            self._index.upsert(
                vectors=records,
                namespace=namespace or "",
            )
            
            logger.info("pinecone_upsert_complete", count=len(records))
            return True

        except Exception as e:
            logger.error("pinecone_upsert_failed", error=str(e))
            raise

    async def query(
        self,
        vector: list[float],
        top_k: int = 10,
        namespace: str | None = None,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Query for similar vectors in Pinecone.
        
        Args:
            vector: The query vector.
            top_k: Number of results to return.
            namespace: Optional namespace to search within.
            filter: Optional metadata filter.
            include_metadata: Whether to include metadata.
            
        Returns:
            List of matching results with scores.
        """
        if not self._index:
            logger.warning("pinecone_not_initialized")
            return []

        try:
            results = self._index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace or "",
                filter=filter,
                include_metadata=include_metadata,
            )

            matches = []
            for match in results.matches:
                matches.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata if include_metadata else {},
                })

            logger.info("pinecone_query_complete", matches=len(matches))
            return matches

        except Exception as e:
            logger.error("pinecone_query_failed", error=str(e))
            raise

    async def delete(
        self,
        ids: list[str] | None = None,
        namespace: str | None = None,
        filter: dict[str, Any] | None = None,
    ) -> bool:
        """
        Delete vectors from Pinecone.
        
        Args:
            ids: Optional list of IDs to delete.
            namespace: Optional namespace.
            filter: Optional metadata filter.
            
        Returns:
            True if operation succeeded.
        """
        if not self._index:
            logger.warning("pinecone_not_initialized")
            return False

        try:
            if ids:
                self._index.delete(
                    ids=ids,
                    namespace=namespace or "",
                )
            elif filter:
                self._index.delete(
                    filter=filter,
                    namespace=namespace or "",
                )
            else:
                self._index.delete(
                    delete_all=True,
                    namespace=namespace or "",
                )

            logger.info("pinecone_delete_complete")
            return True

        except Exception as e:
            logger.error("pinecone_delete_failed", error=str(e))
            raise

    async def get_stats(self, namespace: str | None = None) -> dict[str, Any]:
        """
        Get statistics about the Pinecone index.
        
        Args:
            namespace: Optional namespace to get stats for.
            
        Returns:
            Dictionary with index statistics.
        """
        if not self._index:
            return {"error": "not_initialized"}

        try:
            stats = self._index.describe_index_stats()
            
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "namespaces": {
                    ns: {"vector_count": data.vector_count}
                    for ns, data in stats.namespaces.items()
                },
            }

        except Exception as e:
            logger.error("pinecone_stats_failed", error=str(e))
            return {"error": str(e)}
