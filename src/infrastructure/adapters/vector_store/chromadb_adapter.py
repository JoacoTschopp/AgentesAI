"""
ChromaDB Adapter - Implementation of VectorStorePort for ChromaDB.

This module provides the ChromaDB implementation of the vector store interface,
enabling document storage and retrieval with embeddings.
"""

from typing import Any
import structlog
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.domain.ports.vector_store_port import VectorStorePort
from src.infrastructure.config.settings import Settings


logger = structlog.get_logger()


class ChromaDBAdapter(VectorStorePort):
    """
    ChromaDB implementation of the vector store port.
    
    Provides document storage and similarity search using ChromaDB.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the ChromaDB adapter.

        Args:
            settings: Application settings
        """
        self.settings = settings
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        logger.info("chromadb_adapter_initialized", path="./chroma_db")

    def get_or_create_collection(self, collection_name: str = "pdf_documents"):
        """
        Get or create a ChromaDB collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            ChromaDB collection
        """
        try:
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "PDF document embeddings"}
            )
            logger.info("collection_ready", name=collection_name)
            return collection
        except Exception as e:
            logger.error("collection_creation_failed", error=str(e))
            raise

    async def upsert(
        self,
        vectors: list[list[float]],
        ids: list[str],
        metadata: list[dict[str, Any]] | None = None,
        namespace: str | None = None,
    ) -> bool:
        """
        Insert or update vectors in ChromaDB.
        
        Args:
            vectors: List of embedding vectors
            ids: Unique identifiers for each vector
            metadata: Optional metadata for each vector
            namespace: Collection name (optional)
            
        Returns:
            True if operation succeeded
        """
        try:
            collection_name = namespace or "pdf_documents"
            collection = self.get_or_create_collection(collection_name)
            
            # ChromaDB expects embeddings, ids, and metadatas
            collection.upsert(
                embeddings=vectors,
                ids=ids,
                metadatas=metadata or [{}] * len(ids)
            )
            
            logger.info(
                "vectors_upserted",
                collection=collection_name,
                count=len(ids)
            )
            return True
            
        except Exception as e:
            logger.error("upsert_failed", error=str(e), exc_info=True)
            return False

    async def query(
        self,
        vector: list[float],
        top_k: int = 10,
        namespace: str | None = None,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Query for similar vectors in ChromaDB.
        
        Args:
            vector: The query vector
            top_k: Number of results to return
            namespace: Collection name (optional)
            filter: Optional metadata filter
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of matching results with scores and metadata
        """
        try:
            collection_name = namespace or "pdf_documents"
            collection = self.get_or_create_collection(collection_name)
            
            results = collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                where=filter,
                include=["metadatas", "documents", "distances"] if include_metadata else ["distances"]
            )
            
            # Format results
            formatted_results = []
            if results and results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    }
                    
                    if include_metadata and results.get('metadatas'):
                        result['metadata'] = results['metadatas'][0][i]
                    
                    if results.get('documents'):
                        result['document'] = results['documents'][0][i]
                    
                    formatted_results.append(result)
            
            logger.info(
                "query_complete",
                collection=collection_name,
                results_count=len(formatted_results)
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error("query_failed", error=str(e), exc_info=True)
            return []

    async def delete(
        self,
        ids: list[str] | None = None,
        namespace: str | None = None,
        filter: dict[str, Any] | None = None,
    ) -> bool:
        """
        Delete vectors from ChromaDB.
        
        Args:
            ids: Optional list of IDs to delete
            namespace: Collection name (optional)
            filter: Optional metadata filter for deletion
            
        Returns:
            True if operation succeeded
        """
        try:
            collection_name = namespace or "pdf_documents"
            collection = self.get_or_create_collection(collection_name)
            
            if ids:
                collection.delete(ids=ids)
                logger.info("vectors_deleted", collection=collection_name, count=len(ids))
            elif filter:
                collection.delete(where=filter)
                logger.info("vectors_deleted_by_filter", collection=collection_name)
            else:
                logger.warning("delete_called_without_ids_or_filter")
                return False
            
            return True
            
        except Exception as e:
            logger.error("delete_failed", error=str(e), exc_info=True)
            return False

    async def get_stats(self, namespace: str | None = None) -> dict[str, Any]:
        """
        Get statistics about the ChromaDB collection.
        
        Args:
            namespace: Collection name (optional)
            
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_name = namespace or "pdf_documents"
            collection = self.get_or_create_collection(collection_name)
            
            count = collection.count()
            
            stats = {
                "collection_name": collection_name,
                "total_vectors": count,
                "metadata": collection.metadata
            }
            
            logger.info("stats_retrieved", stats=stats)
            return stats
            
        except Exception as e:
            logger.error("get_stats_failed", error=str(e), exc_info=True)
            return {}

    async def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        ids: list[str],
        namespace: str | None = None,
    ) -> bool:
        """
        Add documents with their embeddings and metadata to ChromaDB.
        
        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: List of unique identifiers
            namespace: Collection name (optional)
            
        Returns:
            True if operation succeeded
        """
        try:
            collection_name = namespace or "pdf_documents"
            collection = self.get_or_create_collection(collection_name)
            
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(
                "documents_added",
                collection=collection_name,
                count=len(documents)
            )
            return True
            
        except Exception as e:
            logger.error("add_documents_failed", error=str(e), exc_info=True)
            return False
