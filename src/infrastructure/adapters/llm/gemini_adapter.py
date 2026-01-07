"""
Gemini Adapter - Implementation of LLMPort for Google Gemini API using LangChain.

This module provides the Google Gemini implementation of the LLM interface,
serving as the fallback provider in the circuit breaker pattern.
"""

import logging
from typing import Any, AsyncIterator, List

import structlog
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

from src.domain.ports.llm_port import LLMPort
from src.infrastructure.config.settings import Settings


logger = structlog.get_logger()


class GeminiAdapter(LLMPort):
    """
    Google Gemini implementation of the LLM port using LangChain.
    
    Uses langchain-google-genai for chat completions and embeddings.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the Gemini adapter.

        Args:
            settings: Application settings containing Gemini configuration
        """
        self.settings = settings
        self.model_name = settings.gemini_model
        
        self.chat_model = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=settings.google_api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.google_api_key
        )
        
        logger.info(
            "gemini_adapter_initialized",
            model=self.model_name
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """
        Generate text completion using Gemini.

        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        try:
            langchain_messages = []
            if system_prompt:
                langchain_messages.append(SystemMessage(content=system_prompt))
            
            for msg in messages:
                langchain_messages.append(HumanMessage(content=msg.get("content", "")))

            logger.debug(
                "gemini_generate_request",
                model=self.model_name,
                num_messages=len(messages),
                temperature=temperature
            )

            response = await self.chat_model.ainvoke(langchain_messages)
            
            logger.info(
                "gemini_generate_success",
                model=self.model_name,
                response_length=len(response.content)
            )

            return response.content

        except Exception as e:
            logger.error(
                "gemini_generate_error",
                model=self.model_name,
                error=str(e),
                exc_info=True
            )
            raise

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate streaming text completion using Gemini.

        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Yields:
            Text chunks as they are generated
        """
        try:
            langchain_messages = []
            if system_prompt:
                langchain_messages.append(SystemMessage(content=system_prompt))
            
            for msg in messages:
                langchain_messages.append(HumanMessage(content=msg.get("content", "")))

            logger.debug(
                "gemini_stream_request",
                model=self.model_name,
                num_messages=len(messages),
                temperature=temperature
            )

            async for chunk in self.chat_model.astream(langchain_messages):
                if chunk.content:
                    yield chunk.content

            logger.info(
                "gemini_stream_complete",
                model=self.model_name
            )

        except Exception as e:
            logger.error(
                "gemini_stream_error",
                model=self.model_name,
                error=str(e),
                exc_info=True
            )
            raise

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts using Gemini.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            logger.debug(
                "gemini_embeddings_request",
                num_texts=len(texts)
            )

            embeddings = await self.embeddings.aembed_documents(texts)

            logger.info(
                "gemini_embeddings_success",
                num_texts=len(texts),
                embedding_dim=len(embeddings[0]) if embeddings else 0
            )

            return embeddings

        except Exception as e:
            logger.error(
                "gemini_embeddings_error",
                error=str(e),
                exc_info=True
            )
            raise

    def get_model_name(self) -> str:
        """
        Get the name of the Gemini model being used.

        Returns:
            Model name
        """
        return self.model_name

    async def is_available(self) -> bool:
        """
        Check if the Gemini API is available.

        Returns:
            True if available, False otherwise
        """
        try:
            test_response = await self.generate(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return bool(test_response)
        except Exception as e:
            logger.warning(
                "gemini_availability_check_failed",
                error=str(e)
            )
            return False
