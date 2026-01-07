"""
Session Service - Business logic for session management.

This module provides the service layer for managing user sessions,
including creation, retrieval, and lifecycle management.
"""

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import structlog

from src.domain.models.session import Session, SessionStatus
from src.domain.dtos.session_dto import SessionCreateDTO, SessionResponseDTO
from src.domain.ports.session_repository_port import SessionRepositoryPort
from src.infrastructure.config.settings import Settings


logger = structlog.get_logger()


class SessionService:
    """
    Service for managing user sessions.
    
    Provides business logic for session creation, retrieval, updating,
    and termination. Sessions are persisted in MongoDB for long-term storage.
    
    Attributes:
        repository: Session repository for persistence.
        settings: Application settings.
    """

    def __init__(
        self,
        repository: SessionRepositoryPort,
        settings: Settings,
    ):
        """
        Initialize session service.
        
        Args:
            repository: Session repository implementation.
            settings: Application settings.
        """
        self._repository = repository
        self._settings = settings

    async def create(self, dto: SessionCreateDTO) -> SessionResponseDTO:
        """
        Create a new session.
        
        Args:
            dto: Session creation data.
            
        Returns:
            The created session as a response DTO.
        """
        expires_at = datetime.utcnow() + timedelta(hours=self._settings.session_ttl_hours)
        
        session = Session(
            user_id=dto.user_id,
            agent_id=dto.agent_id,
            context=dto.context,
            metadata=dto.metadata,
            expires_at=expires_at,
        )
        
        created_session = await self._repository.create(session)
        
        logger.info(
            "session_created",
            session_id=str(created_session.id),
            user_id=dto.user_id,
        )
        
        return self._to_response_dto(created_session)

    async def get_by_id(self, session_id: UUID) -> SessionResponseDTO | None:
        """
        Retrieve a session by ID.
        
        Args:
            session_id: The session's unique identifier.
            
        Returns:
            The session if found, None otherwise.
        """
        session = await self._repository.get_by_id(session_id)
        
        if session:
            return self._to_response_dto(session)
        return None

    async def get_by_user(self, user_id: str, limit: int = 10) -> list[SessionResponseDTO]:
        """
        Retrieve sessions for a user.
        
        Args:
            user_id: The user's identifier.
            limit: Maximum number of sessions to return.
            
        Returns:
            List of sessions for the user.
        """
        sessions = await self._repository.get_by_user_id(user_id, limit)
        return [self._to_response_dto(s) for s in sessions]

    async def update_context(
        self,
        session_id: UUID,
        context: dict[str, Any],
    ) -> SessionResponseDTO | None:
        """
        Update session context.
        
        Args:
            session_id: The session's unique identifier.
            context: New context data.
            
        Returns:
            The updated session if found, None otherwise.
        """
        session = await self._repository.update_context(session_id, context)
        
        if session:
            logger.info("session_context_updated", session_id=str(session_id))
            return self._to_response_dto(session)
        return None

    async def terminate(self, session_id: UUID) -> bool:
        """
        Terminate a session.
        
        Args:
            session_id: The session's unique identifier.
            
        Returns:
            True if terminated, False if not found.
        """
        session = await self._repository.get_by_id(session_id)
        
        if not session:
            return False

        session.terminate()
        await self._repository.update(session)
        
        logger.info("session_terminated", session_id=str(session_id))
        return True

    async def delete(self, session_id: UUID) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: The session's unique identifier.
            
        Returns:
            True if deleted, False if not found.
        """
        return await self._repository.delete(session_id)

    async def get_or_create(
        self,
        session_id: UUID | None,
        user_id: str,
        agent_id: UUID | None = None,
    ) -> Session:
        """
        Get existing session or create a new one.
        
        Args:
            session_id: Optional existing session ID.
            user_id: User identifier.
            agent_id: Optional agent ID.
            
        Returns:
            Existing or new session.
        """
        if session_id:
            session = await self._repository.get_by_id(session_id)
            if session and session.is_active():
                session.update_activity()
                await self._repository.update(session)
                return session

        dto = SessionCreateDTO(user_id=user_id, agent_id=agent_id)
        response = await self.create(dto)
        
        return await self._repository.get_by_id(response.id)

    def _to_response_dto(self, session: Session) -> SessionResponseDTO:
        """Convert Session model to response DTO."""
        return SessionResponseDTO(
            id=session.id,
            user_id=session.user_id,
            agent_id=session.agent_id,
            status=session.status,
            context=session.context,
            metadata=session.metadata,
            created_at=session.created_at,
            updated_at=session.updated_at,
            expires_at=session.expires_at,
        )
