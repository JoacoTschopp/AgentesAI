"""
Session Repository Port - Abstract interface for session persistence.

This module defines the contract for session storage operations,
enabling different storage backends (MongoDB, Redis, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from src.domain.models.session import Session


class SessionRepositoryPort(ABC):
    """
    Abstract interface for session repository operations.
    
    All session storage adapters must implement this interface
    to ensure consistent persistence behavior.
    """

    @abstractmethod
    async def create(self, session: Session) -> Session:
        """
        Create a new session in the repository.
        
        Args:
            session: The session entity to persist.
            
        Returns:
            The persisted session with any generated fields.
            
        Raises:
            RepositoryError: If creation fails.
        """
        pass

    @abstractmethod
    async def get_by_id(self, session_id: UUID) -> Session | None:
        """
        Retrieve a session by its ID.
        
        Args:
            session_id: The unique session identifier.
            
        Returns:
            The session if found, None otherwise.
            
        Raises:
            RepositoryError: If retrieval fails.
        """
        pass

    @abstractmethod
    async def get_by_user_id(self, user_id: str, limit: int = 10) -> list[Session]:
        """
        Retrieve sessions for a specific user.
        
        Args:
            user_id: The user identifier.
            limit: Maximum number of sessions to return.
            
        Returns:
            List of sessions for the user.
            
        Raises:
            RepositoryError: If retrieval fails.
        """
        pass

    @abstractmethod
    async def update(self, session: Session) -> Session:
        """
        Update an existing session.
        
        Args:
            session: The session entity with updated data.
            
        Returns:
            The updated session.
            
        Raises:
            RepositoryError: If update fails.
        """
        pass

    @abstractmethod
    async def delete(self, session_id: UUID) -> bool:
        """
        Delete a session by its ID.
        
        Args:
            session_id: The unique session identifier.
            
        Returns:
            True if deleted, False if not found.
            
        Raises:
            RepositoryError: If deletion fails.
        """
        pass

    @abstractmethod
    async def update_context(self, session_id: UUID, context: dict[str, Any]) -> Session | None:
        """
        Update the context of a session.
        
        Args:
            session_id: The unique session identifier.
            context: The new context data.
            
        Returns:
            The updated session if found, None otherwise.
            
        Raises:
            RepositoryError: If update fails.
        """
        pass
