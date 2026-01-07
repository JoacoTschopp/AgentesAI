"""
MongoDB Document Repository - Implementation of DocumentRepositoryPort.

This module provides MongoDB persistence for document entities.
"""

from datetime import datetime
from uuid import UUID

import structlog
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from src.domain.models.document import Document, DocumentStatus
from src.domain.ports.document_repository_port import DocumentRepositoryPort
from src.infrastructure.config.settings import Settings


logger = structlog.get_logger()


class MongoDBDocumentRepository(DocumentRepositoryPort):
    """
    MongoDB implementation of document repository.
    
    Stores document metadata and provides CRUD operations.
    """

    def __init__(self, settings: Settings):
        """
        Initialize MongoDB document repository.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.client: AsyncIOMotorClient | None = None
        self.collection: AsyncIOMotorCollection | None = None

    async def initialize(self) -> None:
        """Initialize MongoDB connection and collection."""
        try:
            self.client = AsyncIOMotorClient(self.settings.mongodb_uri)
            db = self.client[self.settings.mongodb_database]
            self.collection = db["documents"]
            
            # Create indexes
            await self.collection.create_index("document_id", unique=True)
            await self.collection.create_index("user_id")
            await self.collection.create_index("status")
            await self.collection.create_index("uploaded_at")
            await self.collection.create_index([("user_id", 1), ("status", 1)])
            
            logger.info("mongodb_document_repository_initialized")
        except Exception as e:
            logger.error("mongodb_document_repository_init_failed", error=str(e))
            raise

    async def close(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("mongodb_document_repository_closed")

    async def add(self, document: Document) -> Document:
        """
        Add a new document to the repository.
        
        Args:
            document: Document to add
            
        Returns:
            The added document
        """
        try:
            doc_dict = document.to_dict()
            await self.collection.insert_one(doc_dict)
            
            logger.info(
                "document_added",
                document_id=document.document_id,
                filename=document.filename
            )
            
            return document
        except Exception as e:
            logger.error("document_add_failed", error=str(e), exc_info=True)
            raise

    async def get_by_id(self, document_id: UUID) -> Document | None:
        """
        Get a document by its ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        try:
            doc_dict = await self.collection.find_one({"id": str(document_id)})
            
            if doc_dict:
                return Document.from_dict(doc_dict)
            
            return None
        except Exception as e:
            logger.error("document_get_by_id_failed", error=str(e))
            return None

    async def get_by_document_id(self, document_id: str) -> Document | None:
        """
        Get a document by its ChromaDB document ID.
        
        Args:
            document_id: ChromaDB document identifier
            
        Returns:
            Document if found, None otherwise
        """
        try:
            doc_dict = await self.collection.find_one({"document_id": document_id})
            
            if doc_dict:
                return Document.from_dict(doc_dict)
            
            return None
        except Exception as e:
            logger.error("document_get_by_document_id_failed", error=str(e))
            return None

    async def exists(self, document_id: str) -> bool:
        """
        Check if a document exists by ChromaDB document ID.
        
        Args:
            document_id: ChromaDB document identifier
            
        Returns:
            True if document exists and is not deleted
        """
        try:
            count = await self.collection.count_documents({
                "document_id": document_id,
                "status": {"$ne": DocumentStatus.DELETED.value}
            })
            return count > 0
        except Exception as e:
            logger.error("document_exists_check_failed", error=str(e))
            return False

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
        try:
            query = {"user_id": user_id}
            
            if not include_deleted:
                query["status"] = {"$ne": DocumentStatus.DELETED.value}
            
            cursor = self.collection.find(query).sort("uploaded_at", -1).skip(skip).limit(limit)
            
            documents = []
            async for doc_dict in cursor:
                documents.append(Document.from_dict(doc_dict))
            
            logger.info(
                "documents_listed_by_user",
                user_id=user_id,
                count=len(documents)
            )
            
            return documents
        except Exception as e:
            logger.error("list_by_user_failed", error=str(e))
            return []

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
        try:
            query = {}
            
            if not include_deleted:
                query["status"] = {"$ne": DocumentStatus.DELETED.value}
            
            cursor = self.collection.find(query).sort("uploaded_at", -1).skip(skip).limit(limit)
            
            documents = []
            async for doc_dict in cursor:
                documents.append(Document.from_dict(doc_dict))
            
            logger.info("documents_listed", count=len(documents))
            
            return documents
        except Exception as e:
            logger.error("list_all_failed", error=str(e))
            return []

    async def update(self, document: Document) -> Document:
        """
        Update a document.
        
        Args:
            document: Document with updated data
            
        Returns:
            Updated document
        """
        try:
            doc_dict = document.to_dict()
            
            await self.collection.update_one(
                {"id": str(document.id)},
                {"$set": doc_dict}
            )
            
            logger.info("document_updated", document_id=document.document_id)
            
            return document
        except Exception as e:
            logger.error("document_update_failed", error=str(e))
            raise

    async def soft_delete(self, document_id: UUID) -> bool:
        """
        Soft delete a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if deletion succeeded
        """
        try:
            result = await self.collection.update_one(
                {"id": str(document_id)},
                {
                    "$set": {
                        "status": DocumentStatus.DELETED.value,
                        "deleted_at": datetime.utcnow()
                    }
                }
            )
            
            success = result.modified_count > 0
            
            if success:
                logger.info("document_soft_deleted", document_id=str(document_id))
            
            return success
        except Exception as e:
            logger.error("soft_delete_failed", error=str(e))
            return False

    async def count(self, user_id: str | None = None, include_deleted: bool = False) -> int:
        """
        Count documents.
        
        Args:
            user_id: Optional user filter
            include_deleted: Whether to include soft-deleted documents
            
        Returns:
            Number of documents
        """
        try:
            query = {}
            
            if user_id:
                query["user_id"] = user_id
            
            if not include_deleted:
                query["status"] = {"$ne": DocumentStatus.DELETED.value}
            
            count = await self.collection.count_documents(query)
            
            return count
        except Exception as e:
            logger.error("count_failed", error=str(e))
            return 0
