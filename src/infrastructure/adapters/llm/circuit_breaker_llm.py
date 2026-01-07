"""
Circuit Breaker LLM - Resilient LLM adapter with automatic failover.

This module implements the Circuit Breaker pattern for LLM operations,
providing automatic failover between OpenAI (primary) and Gemini (fallback).
Gracefully handles invalid API keys by disabling unavailable providers.
"""

import logging
from typing import Any, AsyncIterator

import pybreaker
import structlog

from src.domain.ports.llm_port import LLMPort
from src.infrastructure.adapters.llm.openai_adapter import OpenAIAdapter
from src.infrastructure.adapters.llm.gemini_adapter import GeminiAdapter
from src.infrastructure.config.settings import Settings


logger = structlog.get_logger()


class CircuitBreakerLLM(LLMPort):
    """
    Circuit Breaker implementation for LLM operations.
    
    Provides automatic failover between primary (OpenAI) and fallback (Gemini)
    providers. Gracefully handles missing or invalid API keys by only using
    available providers.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the circuit breaker LLM with available providers.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._primary = None
        self._fallback = None
        self._current_provider = None
        
        # Try to initialize OpenAI (primary)
        try:
            if settings.openai_api_key and settings.openai_api_key.strip():
                self._primary = OpenAIAdapter(settings)
                self._current_provider = "openai"
                logger.info("openai_adapter_initialized")
            else:
                logger.warning("openai_api_key_not_configured")
        except Exception as e:
            logger.warning("openai_adapter_initialization_failed", error=str(e))
        
        # Try to initialize Gemini (fallback)
        try:
            if settings.google_api_key and settings.google_api_key.strip():
                self._fallback = GeminiAdapter(settings)
                if not self._current_provider:
                    self._current_provider = "gemini"
                logger.info("gemini_adapter_initialized")
            else:
                logger.warning("gemini_api_key_not_configured")
        except Exception as e:
            logger.warning("gemini_adapter_initialization_failed", error=str(e))
        
        # Ensure at least one provider is available
        if not self._primary and not self._fallback:
            raise RuntimeError(
                "No LLM providers available. Please configure at least one API key "
                "(OPENAI_API_KEY or GOOGLE_API_KEY)"
            )
        
        # Initialize circuit breaker only if we have both providers
        if self._primary and self._fallback:
            self._breaker = pybreaker.CircuitBreaker(
                fail_max=settings.circuit_breaker_fail_max,
                reset_timeout=settings.circuit_breaker_reset_timeout,
                name="llm_circuit_breaker",
            )
            logger.info(
                "circuit_breaker_initialized",
                primary="openai",
                fallback="gemini"
            )
        else:
            self._breaker = None
            provider = "openai" if self._primary else "gemini"
            logger.info(
                "single_provider_mode",
                provider=provider
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
        Generate text with automatic failover.
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        # Single provider mode (no circuit breaker)
        if not self._breaker:
            provider = self._primary if self._primary else self._fallback
            return await provider.generate(
                messages=messages,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        
        # Circuit breaker mode with failover
        try:
            if self._breaker.current_state == "open":
                logger.warning("circuit_open_using_fallback", provider="gemini")
                return await self._fallback.generate(
                    messages=messages,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )

            try:
                response = await self._breaker.call_async(
                    self._primary.generate,
                    messages=messages,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                self._current_provider = "openai"
                return response
            except Exception as e:
                logger.warning(
                    "primary_llm_failed_switching_to_fallback",
                    error=str(e),
                    provider="openai"
                )
                self._current_provider = "gemini"
                return await self._fallback.generate(
                    messages=messages,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
        except pybreaker.CircuitBreakerError:
            logger.warning("circuit_breaker_tripped", provider="gemini")
            return await self._fallback.generate(
                messages=messages,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate streaming response with automatic failover.
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Yields:
            Text chunks as they are generated
        """
        # Single provider mode
        if not self._breaker:
            provider = self._primary if self._primary else self._fallback
            async for chunk in provider.generate_stream(
                messages=messages,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ):
                yield chunk
            return
        
        # Circuit breaker mode with failover
        try:
            if self._breaker.current_state == "open":
                logger.warning("circuit_open_streaming_fallback", provider="gemini")
                async for chunk in self._fallback.generate_stream(
                    messages, system_prompt, temperature, max_tokens, **kwargs
                ):
                    yield chunk
                return

            async for chunk in self._primary.generate_stream(
                messages, system_prompt, temperature, max_tokens, **kwargs
            ):
                yield chunk
            self._current_provider = "openai"

        except Exception as e:
            logger.error("primary_stream_failed", error=str(e))
            self._breaker.fail()
            async for chunk in self._fallback.generate_stream(
                messages, system_prompt, temperature, max_tokens, **kwargs
            ):
                yield chunk

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings with failover support.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Single provider mode
        if not self._breaker:
            provider = self._primary if self._primary else self._fallback
            return await provider.get_embeddings(texts)
        
        # Circuit breaker mode with failover
        try:
            return await self._primary.get_embeddings(texts)
        except Exception as e:
            logger.warning(
                "primary_embeddings_failed_using_fallback",
                error=str(e)
            )
            return await self._fallback.get_embeddings(texts)

    def get_model_name(self) -> str:
        """
        Get the name of the currently active model.
        
        Returns:
            Model name
        """
        if self._current_provider == "openai" and self._primary:
            return self._primary.get_model_name()
        elif self._fallback:
            return self._fallback.get_model_name()
        return "unknown"

    async def is_available(self) -> bool:
        """
        Check if at least one LLM provider is available.
        
        Returns:
            True if any provider is available
        """
        if self._primary:
            try:
                return await self._primary.is_available()
            except:
                pass
        
        if self._fallback:
            try:
                return await self._fallback.is_available()
            except:
                pass
        
        return False

    def get_current_provider(self) -> str:
        """
        Get the name of the currently active provider.
        
        Returns:
            Provider name ('openai', 'gemini', or 'none')
        """
        return self._current_provider or "none"

    def get_available_providers(self) -> list[str]:
        """
        Get list of available providers.
        
        Returns:
            List of provider names
        """
        providers = []
        if self._primary:
            providers.append("openai")
        if self._fallback:
            providers.append("gemini")
        return providers
