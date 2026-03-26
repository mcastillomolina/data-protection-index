"""
Abstract base class for LLM clients.

This module defines the interface that all LLM client implementations must follow.
It includes methods for text generation, structured JSON responses, and usage tracking.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time
from loguru import logger


@dataclass
class LLMUsage:
    """
    Track token usage and costs for LLM API calls.

    Attributes:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used (prompt + completion)
        estimated_cost_usd: Estimated cost in USD
        model: Model name used
        timestamp: Unix timestamp when the call was made
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model: str = ""
    timestamp: float = field(default_factory=time.time)


class LLMClient(ABC):
    """
    Abstract base class for LLM client implementations.

    All LLM clients (OpenAI, Anthropic, etc.) must implement this interface
    to ensure consistent behavior across different providers.
    """

    def __init__(self, api_key: str, model: str, timeout: int = 30):
        """
        Initialize the LLM client.

        Args:
            api_key: API key for the LLM provider
            model: Model name to use (e.g., "gpt-4-turbo-preview", "claude-3-opus-20240229")
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.total_usage = LLMUsage(model=model)

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Generate a text completion from the LLM.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system prompt for instruction
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response

        Raises:
            Exception: If the API call fails after retries
        """
        pass

    @abstractmethod
    def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response from the LLM.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system prompt for instruction
            schema: Optional JSON schema to validate against
            temperature: Sampling temperature (lower for more consistent JSON)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Parsed JSON response as a dictionary

        Raises:
            ValueError: If the response is not valid JSON
            Exception: If the API call fails after retries
        """
        pass

    @abstractmethod
    def _estimate_cost(self, usage: LLMUsage) -> float:
        """
        Estimate the cost of an API call based on token usage.

        Args:
            usage: LLMUsage object with token counts

        Returns:
            Estimated cost in USD
        """
        pass

    def get_total_usage(self) -> LLMUsage:
        """
        Get cumulative usage statistics for this client instance.

        Returns:
            LLMUsage object with total tokens and cost
        """
        return self.total_usage

    def reset_usage(self) -> None:
        """Reset the usage statistics to zero."""
        self.total_usage = LLMUsage(model=self.model)
        logger.info(f"Reset usage statistics for {self.model}")

    def log_usage(self, usage: LLMUsage) -> None:
        """
        Log and accumulate usage statistics.

        Args:
            usage: LLMUsage from the latest API call
        """
        self.total_usage.prompt_tokens += usage.prompt_tokens
        self.total_usage.completion_tokens += usage.completion_tokens
        self.total_usage.total_tokens += usage.total_tokens
        self.total_usage.estimated_cost_usd += usage.estimated_cost_usd

        logger.debug(
            f"LLM call completed: {usage.total_tokens} tokens, "
            f"${usage.estimated_cost_usd:.4f} (cumulative: ${self.total_usage.estimated_cost_usd:.4f})"
        )
