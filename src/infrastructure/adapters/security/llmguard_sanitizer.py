"""
Mock Prompt Sanitizer Implementation

LLMGuard is temporarily disabled due to dependency conflicts.
This is a pass-through implementation that allows the application to run.
"""

import logging
from typing import Optional

import structlog

from src.domain.ports.prompt_sanitizer_port import (
    PromptSanitizerPort,
    SanitizationResult,
)
from src.infrastructure.config.settings import Settings


logger = structlog.get_logger()


class LLMGuardSanitizer(PromptSanitizerPort):
    """
    Mock implementation of the prompt sanitizer port.
    
    This is a temporary pass-through implementation while LLMGuard
    dependency conflicts are resolved.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the mock sanitizer.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.enabled = settings.llmguard_enabled
        logger.warning(
            "llmguard_mock_initialized",
            enabled=self.enabled,
            message="Using mock sanitizer - no actual sanitization performed"
        )

    async def sanitize_input(self, prompt: str, user_id: Optional[str] = None) -> SanitizationResult:
        """
        Mock input sanitization - passes through without changes.

        Args:
            prompt: The input prompt to sanitize
            user_id: Optional user identifier

        Returns:
            SanitizationResult with is_safe=True and original prompt
        """
        if not self.enabled:
            return SanitizationResult(
                is_safe=True,
                sanitized_text=prompt,
                risk_score=0.0,
                detected_issues=[]
            )

        logger.debug(
            "mock_sanitize_input",
            user_id=user_id,
            prompt_length=len(prompt)
        )

        return SanitizationResult(
            is_safe=True,
            sanitized_text=prompt,
            risk_score=0.0,
            detected_issues=[]
        )

    async def sanitize_output(self, output: str, user_id: Optional[str] = None) -> SanitizationResult:
        """
        Mock output sanitization - passes through without changes.

        Args:
            output: The LLM output to sanitize
            user_id: Optional user identifier

        Returns:
            SanitizationResult with is_safe=True and original output
        """
        if not self.enabled:
            return SanitizationResult(
                is_safe=True,
                sanitized_text=output,
                risk_score=0.0,
                detected_issues=[]
            )

        logger.debug(
            "mock_sanitize_output",
            user_id=user_id,
            output_length=len(output)
        )

        return SanitizationResult(
            is_safe=True,
            sanitized_text=output,
            risk_score=0.0,
            detected_issues=[]
        )

    async def detect_prompt_injection(self, prompt: str) -> bool:
        """
        Mock prompt injection detection - always returns False.

        Args:
            prompt: The prompt to check

        Returns:
            False (no injection detected in mock)
        """
        logger.debug("mock_detect_prompt_injection", prompt_length=len(prompt))
        return False

    async def detect_sensitive_data(self, text: str) -> bool:
        """
        Mock sensitive data detection - always returns False.

        Args:
            text: The text to check

        Returns:
            False (no sensitive data detected in mock)
        """
        logger.debug("mock_detect_sensitive_data", text_length=len(text))
        return False
