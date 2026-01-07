"""
Chat Routes - Endpoints for chat interactions with AI agents.

This module provides endpoints for sending messages to AI agents
and receiving responses, with support for streaming.
"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from src.api.dependencies.services import get_chat_service
from src.application.services.chat_service import ChatService
from src.domain.dtos.chat_dto import ChatRequestDTO, ChatResponseDTO


router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponseDTO,
    status_code=status.HTTP_200_OK,
    summary="Send Chat Message",
    description="Send a message to an AI agent and receive a response.",
)
async def chat(
    request: ChatRequestDTO,
    x_user_id: str = Header(default="anonymous", alias="X-User-ID"),
    service: ChatService = Depends(get_chat_service),
) -> ChatResponseDTO:
    """
    Send a chat message and get a response.
    
    Args:
        request: Chat request with user message.
        x_user_id: User identifier from header.
        service: Chat service instance.
        
    Returns:
        Chat response with assistant message.
    """
    return await service.chat(request, x_user_id)


@router.post(
    "/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream Chat Response",
    description="Send a message and receive a streaming response.",
)
async def chat_stream(
    request: ChatRequestDTO,
    x_user_id: str = Header(default="anonymous", alias="X-User-ID"),
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    Send a chat message and stream the response.
    
    Args:
        request: Chat request with user message.
        x_user_id: User identifier from header.
        service: Chat service instance.
        
    Returns:
        Streaming response with text chunks.
    """
    async def generate():
        async for chunk in service.chat_stream(request, x_user_id):
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get(
    "/history/{session_id}",
    response_model=list[dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="Get Conversation History",
    description="Retrieve the conversation history for a session.",
)
async def get_conversation_history(
    session_id: UUID,
    service: ChatService = Depends(get_chat_service),
) -> list[dict[str, Any]]:
    """
    Get conversation history for a session.
    
    Args:
        session_id: Session identifier.
        service: Chat service instance.
        
    Returns:
        List of messages in the conversation.
    """
    return await service.get_conversation_history(session_id)


@router.delete(
    "/history/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear Conversation History",
    description="Clear the conversation history for a session.",
)
async def clear_conversation_history(
    session_id: UUID,
    service: ChatService = Depends(get_chat_service),
) -> None:
    """
    Clear conversation history for a session.
    
    Args:
        session_id: Session identifier.
        service: Chat service instance.
    """
    await service.clear_conversation(session_id)
