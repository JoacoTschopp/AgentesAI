"""
Chat Service - Business logic for chat interactions.

This module provides the service layer for handling chat messages,
orchestrating the flow between sanitization, LLM generation, and session management.
"""

from datetime import datetime
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

import structlog

from src.domain.models.agent import Agent
from src.domain.models.conversation import Conversation, Message, MessageRole
from src.domain.models.session import Session
from src.domain.models.prompts import get_agent_prompt
from src.domain.dtos.chat_dto import ChatRequestDTO, ChatResponseDTO, StreamChunkDTO
from src.domain.ports.llm_port import LLMPort
from src.domain.ports.prompt_sanitizer_port import PromptSanitizerPort
from src.application.services.agent_service import AgentService
from src.application.services.session_service import SessionService


logger = structlog.get_logger()


class ChatService:
    """
    Service for handling chat interactions.
    
    Orchestrates the complete chat flow including:
    - Input sanitization
    - Session management
    - LLM generation
    - Output sanitization
    - Response formatting
    
    Attributes:
        llm: Language model adapter.
        sanitizer: Prompt sanitizer adapter.
        agent_service: Agent management service.
        session_service: Session management service.
    """

    def __init__(
        self,
        llm: LLMPort,
        sanitizer: PromptSanitizerPort,
        agent_service: AgentService,
        session_service: SessionService,
    ):
        """
        Initialize chat service.
        
        Args:
            llm: LLM adapter (with circuit breaker).
            sanitizer: Prompt sanitizer.
            agent_service: Agent service.
            session_service: Session service.
        """
        self._llm = llm
        self._sanitizer = sanitizer
        self._agent_service = agent_service
        self._session_service = session_service
        self._conversations: dict[UUID, Conversation] = {}

    async def chat(
        self,
        request: ChatRequestDTO,
        user_id: str,
    ) -> ChatResponseDTO:
        """
        Process a chat message and generate a response.
        
        Args:
            request: Chat request with user message.
            user_id: Identifier of the user.
            
        Returns:
            Chat response with assistant message.
            
        Raises:
            ValueError: If input fails sanitization.
        """
        input_result = await self._sanitizer.sanitize_input(request.message)
        
        if not input_result.is_safe:
            logger.warning(
                "input_rejected",
                risk_score=input_result.risk_score,
                issues=input_result.detected_issues,
            )
            raise ValueError(f"Input rejected: {', '.join(input_result.detected_issues)}")

        sanitized_message = input_result.sanitized_text

        session = await self._session_service.get_or_create(
            session_id=request.session_id,
            user_id=user_id,
            agent_id=request.agent_id,
        )

        agent = None
        if request.agent_id:
            agent = self._agent_service.get_agent_model(request.agent_id)
        
        if not agent:
            agents = await self._agent_service.get_all()
            if agents:
                agent = self._agent_service.get_agent_model(agents[0].id)

        conversation = self._get_or_create_conversation(session.id, agent.id if agent else None)
        conversation.add_message(MessageRole.USER, sanitized_message)

        messages = conversation.get_message_history()
        
        # Use centralized prompts from prompts.py
        if agent and agent.id:
            system_prompt = get_agent_prompt(str(agent.id))
        else:
            system_prompt = "You are a helpful AI assistant."
        
        temperature = agent.temperature if agent else 0.7
        max_tokens = agent.max_tokens if agent else 4096

        response_text = await self._llm.generate(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        output_result = await self._sanitizer.sanitize_output(response_text)
        final_response = output_result.sanitized_text

        conversation.add_message(MessageRole.ASSISTANT, final_response)

        logger.info(
            "chat_complete",
            session_id=str(session.id),
            model=self._llm.get_model_name(),
        )

        return ChatResponseDTO(
            id=uuid4(),
            session_id=session.id,
            agent_id=agent.id if agent else uuid4(),
            message=final_response,
            tokens_used={},
            model=self._llm.get_model_name(),
            metadata={
                "input_sanitized": input_result.sanitized_text != request.message,
                "output_sanitized": output_result.sanitized_text != response_text,
            },
        )

    async def chat_stream(
        self,
        request: ChatRequestDTO,
        user_id: str,
    ) -> AsyncIterator[StreamChunkDTO]:
        """
        Process a chat message and stream the response.
        
        Args:
            request: Chat request with user message.
            user_id: Identifier of the user.
            
        Yields:
            Stream chunks with response fragments.
        """
        input_result = await self._sanitizer.sanitize_input(request.message)
        
        if not input_result.is_safe:
            yield StreamChunkDTO(
                chunk=f"Error: {', '.join(input_result.detected_issues)}",
                session_id=request.session_id or uuid4(),
                is_final=True,
            )
            return

        session = await self._session_service.get_or_create(
            session_id=request.session_id,
            user_id=user_id,
            agent_id=request.agent_id,
        )

        agent = None
        if request.agent_id:
            agent = self._agent_service.get_agent_model(request.agent_id)

        conversation = self._get_or_create_conversation(session.id, agent.id if agent else None)
        conversation.add_message(MessageRole.USER, input_result.sanitized_text)

        messages = conversation.get_message_history()
        system_prompt = agent.system_prompt if agent else "You are a helpful AI assistant."

        full_response = ""
        async for chunk in self._llm.generate_stream(
            messages=messages,
            system_prompt=system_prompt,
            temperature=agent.temperature if agent else 0.7,
            max_tokens=agent.max_tokens if agent else 4096,
        ):
            full_response += chunk
            yield StreamChunkDTO(
                chunk=chunk,
                session_id=session.id,
                is_final=False,
            )

        conversation.add_message(MessageRole.ASSISTANT, full_response)

        yield StreamChunkDTO(
            chunk="",
            session_id=session.id,
            is_final=True,
            metadata={"model": self._llm.get_model_name()},
        )

    def _get_or_create_conversation(
        self,
        session_id: UUID,
        agent_id: UUID | None,
    ) -> Conversation:
        """Get existing conversation or create a new one."""
        if session_id not in self._conversations:
            self._conversations[session_id] = Conversation(
                session_id=session_id,
                agent_id=agent_id or uuid4(),
            )
        return self._conversations[session_id]

    async def get_conversation_history(
        self,
        session_id: UUID,
    ) -> list[dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            List of messages in the conversation.
        """
        conversation = self._conversations.get(session_id)
        
        if conversation:
            return [
                {
                    "id": str(msg.id),
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat(),
                }
                for msg in conversation.messages
            ]
        return []

    async def clear_conversation(self, session_id: UUID) -> bool:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            True if cleared, False if not found.
        """
        if session_id in self._conversations:
            del self._conversations[session_id]
            logger.info("conversation_cleared", session_id=str(session_id))
            return True
        return False
