"""
MongoDB Session Repository - MongoDB implementation of SessionRepositoryPort.

This module provides MongoDB-based persistence for user sessions,
supporting long-term session storage for LangGraph workflows.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

import structlog
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from src.domain.models.session import Session, SessionStatus
from src.domain.ports.session_repository_port import SessionRepositoryPort
from src.infrastructure.config.settings import Settings


logger = structlog.get_logger()


class MongoDBSessionRepository(SessionRepositoryPort):
    """
    MongoDB implementation of the session repository.
    
    Uses Motor (async MongoDB driver) for non-blocking database operations.
    Sessions are stored in a dedicated collection with TTL indexing support.
    
    Attributes:
        client: MongoDB async client.
        db: Database instance.
        collection: Sessions collection.
    """

    def __init__(self, settings: Settings):
        """
        Initialize MongoDB session repository.
        
        Args:
            settings: Application settings with MongoDB configuration.
        """
        self._settings = settings
        self._client: AsyncIOMotorClient = AsyncIOMotorClient(settings.mongodb_uri)
        self._db: AsyncIOMotorDatabase = self._client[settings.mongodb_database]
        self._collection = self._db[settings.mongodb_sessions_collection]

    async def initialize(self) -> None:
        """
        Initialize the repository with indexes.
        
        Creates necessary indexes for efficient querying.
        """
        await self._collection.create_index("user_id")
        await self._collection.create_index("status")
        await self._collection.create_index("created_at")
        await self._collection.create_index(
            "expires_at",
            expireAfterSeconds=0,
        )
        logger.info("mongodb_session_repository_initialized")

    async def create(self, session: Session) -> Session:
        """
        Create a new session in MongoDB.
        
        Args:
            session: The session entity to persist.
            
        Returns:
            The persisted session.
        """
        document = self._to_document(session)
        
        await self._collection.insert_one(document)
        
        logger.info("session_created", session_id=str(session.id))
        return session

    async def get_by_id(self, session_id: UUID) -> Session | None:
        """
        Retrieve a session by its ID.
        
        Args:
            session_id: The unique session identifier.
            
        Returns:
            The session if found, None otherwise.
        """
        document = await self._collection.find_one({"_id": str(session_id)})
        
        if document:
            return self._from_document(document)
        return None

    async def get_by_user_id(self, user_id: str, limit: int = 10) -> list[Session]:
        """
        Retrieve sessions for a specific user.
        
        Args:
            user_id: The user identifier.
            limit: Maximum number of sessions to return.
            
        Returns:
            List of sessions for the user.
        """
        cursor = self._collection.find(
            {"user_id": user_id}
        ).sort("created_at", -1).limit(limit)
        
        sessions = []
        async for document in cursor:
            sessions.append(self._from_document(document))
        
        return sessions

    async def update(self, session: Session) -> Session:
        """
        Update an existing session.
        
        Args:
            session: The session entity with updated data.
            
        Returns:
            The updated session.
        """
        session.updated_at = datetime.utcnow()
        document = self._to_document(session)
        
        await self._collection.replace_one(
            {"_id": str(session.id)},
            document,
        )
        
        logger.info("session_updated", session_id=str(session.id))
        return session

    async def delete(self, session_id: UUID) -> bool:
        """
        Delete a session by its ID.
        
        Args:
            session_id: The unique session identifier.
            
        Returns:
            True if deleted, False if not found.
        """
        result = await self._collection.delete_one({"_id": str(session_id)})
        
        if result.deleted_count > 0:
            logger.info("session_deleted", session_id=str(session_id))
            return True
        return False

    async def update_context(
        self,
        session_id: UUID,
        context: dict[str, Any],
    ) -> Session | None:
        """
        Update the context of a session.
        
        Args:
            session_id: The unique session identifier.
            context: The new context data.
            
        Returns:
            The updated session if found, None otherwise.
        """
        result = await self._collection.find_one_and_update(
            {"_id": str(session_id)},
            {
                "$set": {
                    "context": context,
                    "updated_at": datetime.utcnow(),
                }
            },
            return_document=True,
        )
        
        if result:
            return self._from_document(result)
        return None

    def _to_document(self, session: Session) -> dict[str, Any]:
        """Convert Session entity to MongoDB document."""
        return {
            "_id": str(session.id),
            "user_id": session.user_id,
            "agent_id": str(session.agent_id) if session.agent_id else None,
            "status": session.status.value if isinstance(session.status, SessionStatus) else session.status,
            "context": session.context,
            "metadata": session.metadata,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "expires_at": session.expires_at,
        }

    def _from_document(self, document: dict[str, Any]) -> Session:
        """Convert MongoDB document to Session entity."""
        return Session(
            id=UUID(document["_id"]),
            user_id=document["user_id"],
            agent_id=UUID(document["agent_id"]) if document.get("agent_id") else None,
            status=SessionStatus(document["status"]),
            context=document.get("context", {}),
            metadata=document.get("metadata", {}),
            created_at=document["created_at"],
            updated_at=document["updated_at"],
            expires_at=document.get("expires_at"),
        )

    async def close(self) -> None:
        """Close the MongoDB connection."""
        self._client.close()
