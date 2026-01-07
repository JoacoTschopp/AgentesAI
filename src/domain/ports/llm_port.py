"""
LLM Port - Abstract interface for Language Model operations.

This module defines the contract that all LLM adapters must implement,
enabling the use of different LLM providers (OpenAI, Gemini, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from src.domain.models.conversation import Message


class LLMPort(ABC):
    """
    Abstract interface for Language Model operations.
    
    All LLM adapters (OpenAI, Gemini, etc.) must implement this interface
    to ensure consistent behavior across different providers.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of conversation messages.
            system_prompt: Optional system prompt override.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            Generated text response.
            
        Raises:
            LLMError: If generation fails.
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            messages: List of conversation messages.
            system_prompt: Optional system prompt override.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific parameters.
            
        Yields:
            Text chunks as they are generated.
            
        Raises:
            LLMError: If generation fails.
        """
        pass

    @abstractmethod
    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
            
        Raises:
            LLMError: If embedding generation fails.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the current model.
        
        Returns:
            Model name string.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM service is available.
        
        Returns:
            True if service is available, False otherwise.
        """
        pass
