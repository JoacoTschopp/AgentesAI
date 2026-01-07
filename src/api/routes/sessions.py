"""
Session Routes - Endpoints for session management.

This module provides RESTful endpoints for managing user sessions,
including creation, retrieval, and lifecycle operations.
"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Header, status

from src.api.dependencies.services import get_session_service
from src.application.services.session_service import SessionService
from src.domain.dtos.session_dto import SessionCreateDTO, SessionResponseDTO


router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.post(
    "",
    response_model=SessionResponseDTO,
    status_code=status.HTTP_201_CREATED,
    summary="Create Session",
    description="Create a new user session for conversation continuity.",
)
async def create_session(
    dto: SessionCreateDTO,
    service: SessionService = Depends(get_session_service),
) -> SessionResponseDTO:
    """
    Create a new session.
    
    Args:
        dto: Session creation data.
        service: Session service instance.
        
    Returns:
        The created session.
    """
    return await service.create(dto)


@router.get(
    "/{session_id}",
    response_model=SessionResponseDTO,
    status_code=status.HTTP_200_OK,
    summary="Get Session",
    description="Retrieve a specific session by its ID.",
)
async def get_session(
    session_id: UUID,
    service: SessionService = Depends(get_session_service),
) -> SessionResponseDTO:
    """
    Get a specific session by ID.
    
    Args:
        session_id: The session's unique identifier.
        service: Session service instance.
        
    Returns:
        The requested session.
        
    Raises:
        HTTPException: If session not found.
    """
    session = await service.get_by_id(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    
    return session


@router.get(
    "/user/{user_id}",
    response_model=list[SessionResponseDTO],
    status_code=status.HTTP_200_OK,
    summary="List User Sessions",
    description="Retrieve all sessions for a specific user.",
)
async def list_user_sessions(
    user_id: str,
    limit: int = 10,
    service: SessionService = Depends(get_session_service),
) -> list[SessionResponseDTO]:
    """
    List sessions for a user.
    
    Args:
        user_id: The user's identifier.
        limit: Maximum number of sessions to return.
        service: Session service instance.
        
    Returns:
        List of sessions for the user.
    """
    return await service.get_by_user(user_id, limit)


@router.patch(
    "/{session_id}/context",
    response_model=SessionResponseDTO,
    status_code=status.HTTP_200_OK,
    summary="Update Session Context",
    description="Update the context data for a session.",
)
async def update_session_context(
    session_id: UUID,
    context: dict[str, Any],
    service: SessionService = Depends(get_session_service),
) -> SessionResponseDTO:
    """
    Update session context.
    
    Args:
        session_id: The session's unique identifier.
        context: New context data.
        service: Session service instance.
        
    Returns:
        The updated session.
        
    Raises:
        HTTPException: If session not found.
    """
    session = await service.update_context(session_id, context)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    
    return session


@router.post(
    "/{session_id}/terminate",
    status_code=status.HTTP_200_OK,
    summary="Terminate Session",
    description="Terminate an active session.",
)
async def terminate_session(
    session_id: UUID,
    service: SessionService = Depends(get_session_service),
) -> dict[str, str]:
    """
    Terminate a session.
    
    Args:
        session_id: The session's unique identifier.
        service: Session service instance.
        
    Returns:
        Termination confirmation.
        
    Raises:
        HTTPException: If session not found.
    """
    terminated = await service.terminate(session_id)
    
    if not terminated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    
    return {"status": "terminated", "session_id": str(session_id)}


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Session",
    description="Permanently delete a session.",
)
async def delete_session(
    session_id: UUID,
    service: SessionService = Depends(get_session_service),
) -> None:
    """
    Delete a session.
    
    Args:
        session_id: The session's unique identifier.
        service: Session service instance.
        
    Raises:
        HTTPException: If session not found.
    """
    deleted = await service.delete(session_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
