"""
OpenAI Adapter - Implementation of LLMPort for OpenAI API.

This module provides the OpenAI implementation of the LLM interface,
supporting both chat completions and embeddings.
"""

import structlog
from typing import Any, AsyncIterator

from openai import AsyncOpenAI, OpenAIError

from src.domain.ports.llm_port import LLMPort
from src.infrastructure.config.settings import Settings


logger = structlog.get_logger()


class OpenAIAdapter(LLMPort):
    """
    OpenAI implementation of the LLM port.
    
    Provides access to OpenAI's chat completion and embedding APIs
    with proper error handling and logging.
    
    Attributes:
        client: AsyncOpenAI client instance.
        model: Model name for chat completions.
        embedding_model: Model name for embeddings.
    """

    def __init__(self, settings: Settings):
        """
        Initialize OpenAI adapter.
        
        Args:
            settings: Application settings with OpenAI configuration.
        """
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base if settings.openai_api_base else None,
        )
        self._model = settings.openai_model
        self._embedding_model = settings.openai_embedding_model
        self._available = bool(settings.openai_api_key)

    async def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response using OpenAI chat completion.
        
        Args:
            messages: List of conversation messages.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional OpenAI-specific parameters.
            
        Returns:
            Generated text response.
            
        Raises:
            OpenAIError: If API call fails.
        """
        try:
            formatted_messages = []
            
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            
            formatted_messages.extend(messages)

            response = await self._client.chat.completions.create(
                model=self._model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            content = response.choices[0].message.content or ""
            
            logger.info(
                "openai_generation_complete",
                model=self._model,
                tokens_used=response.usage.total_tokens if response.usage else 0,
            )
            
            return content

        except OpenAIError as e:
            logger.error("openai_generation_failed", error=str(e))
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
        Generate a streaming response using OpenAI.
        
        Args:
            messages: List of conversation messages.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters.
            
        Yields:
            Text chunks as they are generated.
        """
        try:
            formatted_messages = []
            
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            
            formatted_messages.extend(messages)

            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except OpenAIError as e:
            logger.error("openai_stream_failed", error=str(e))
            raise

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings using OpenAI.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        try:
            response = await self._client.embeddings.create(
                model=self._embedding_model,
                input=texts,
            )
            
            return [data.embedding for data in response.data]

        except OpenAIError as e:
            logger.error("openai_embedding_failed", error=str(e))
            raise

    def get_model_name(self) -> str:
        """Get the current model name."""
        return self._model

    def is_available(self) -> bool:
        """Check if OpenAI service is configured and available."""
        return self._available
