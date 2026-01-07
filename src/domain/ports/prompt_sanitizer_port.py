"""
Prompt Sanitizer Port - Abstract interface for prompt security.

This module defines the contract for prompt sanitization operations,
enabling different security implementations (LLMGuard, custom, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any


class SanitizationResult:
    """
    Result of a prompt sanitization operation.
    
    Attributes:
        is_safe: Whether the prompt passed all security checks.
        sanitized_text: The sanitized version of the text.
        risk_score: Overall risk score (0.0 to 1.0).
        detected_issues: List of detected security issues.
        metadata: Additional result metadata.
    """
    
    def __init__(
        self,
        is_safe: bool,
        sanitized_text: str,
        risk_score: float = 0.0,
        detected_issues: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.is_safe = is_safe
        self.sanitized_text = sanitized_text
        self.risk_score = risk_score
        self.detected_issues = detected_issues or []
        self.metadata = metadata or {}


class PromptSanitizerPort(ABC):
    """
    Abstract interface for prompt sanitization operations.
    
    All sanitization adapters must implement this interface
    to ensure consistent security behavior across the application.
    """

    @abstractmethod
    async def sanitize_input(self, text: str) -> SanitizationResult:
        """
        Sanitize user input before sending to LLM.
        
        Args:
            text: The user input text to sanitize.
            
        Returns:
            SanitizationResult with sanitization details.
            
        Raises:
            SanitizationError: If sanitization fails.
        """
        pass

    @abstractmethod
    async def sanitize_output(self, text: str) -> SanitizationResult:
        """
        Sanitize LLM output before returning to user.
        
        Args:
            text: The LLM output text to sanitize.
            
        Returns:
            SanitizationResult with sanitization details.
            
        Raises:
            SanitizationError: If sanitization fails.
        """
        pass

    @abstractmethod
    async def detect_prompt_injection(self, text: str) -> tuple[bool, float]:
        """
        Detect potential prompt injection attempts.
        
        Args:
            text: The text to analyze.
            
        Returns:
            Tuple of (is_injection_detected, confidence_score).
            
        Raises:
            SanitizationError: If detection fails.
        """
        pass

    @abstractmethod
    async def detect_sensitive_data(self, text: str) -> tuple[bool, list[str]]:
        """
        Detect sensitive data (PII, secrets, etc.) in text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            Tuple of (sensitive_data_found, list_of_types_found).
            
        Raises:
            SanitizationError: If detection fails.
        """
        pass
