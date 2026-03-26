"""
OpenAI client implementation for GPT models.

This module provides an implementation of the LLMClient interface for OpenAI's
GPT models, including support for JSON mode and structured outputs.
"""

import json
import time
from typing import Any, Dict, Optional

from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from loguru import logger

from .llm_client import LLMClient, LLMUsage


class OpenAIClient(LLMClient):
    """
    OpenAI implementation of LLMClient using GPT models.

    Supports GPT-4, GPT-4 Turbo, and other OpenAI models with JSON mode
    for structured outputs.
    """

    # Pricing per 1K tokens (as of 2024)
    # Update these values based on current OpenAI pricing
    PRICING = {
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
        "gpt-4o": {"prompt": 0.005, "completion": 0.015},
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., "gpt-4-turbo-preview", "gpt-4o")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        super().__init__(api_key, model, timeout)
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"Initialized OpenAI client with model: {model}")

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Generate a text completion using OpenAI's chat completion API.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system prompt for instruction
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            Generated text response

        Raises:
            Exception: If the API call fails after retries
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )

                # Extract usage and log it
                usage = LLMUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    model=self.model,
                    timestamp=time.time()
                )
                usage.estimated_cost_usd = self._estimate_cost(usage)
                self.log_usage(usage)

                return response.choices[0].message.content

            except RateLimitError as e:
                logger.warning(f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise

            except (APIError, APIConnectionError) as e:
                logger.error(f"API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise

            except Exception as e:
                logger.error(f"Unexpected error calling OpenAI API: {e}")
                raise

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
        Generate a structured JSON response using OpenAI's JSON mode.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system prompt (should mention JSON format)
            schema: Optional JSON schema (for documentation, not enforced)
            temperature: Sampling temperature (lower for more consistent JSON)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            Parsed JSON response as a dictionary

        Raises:
            ValueError: If the response is not valid JSON
            Exception: If the API call fails after retries
        """
        # Ensure system prompt mentions JSON format
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that responds in JSON format."
        elif "json" not in system_prompt.lower():
            system_prompt += "\n\nRespond in valid JSON format."

        # Add schema to prompt if provided
        if schema:
            prompt += f"\n\nPlease respond with JSON matching this schema:\n{json.dumps(schema, indent=2)}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},  # Enable JSON mode
                    **kwargs
                )

                # Extract usage and log it
                usage = LLMUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    model=self.model,
                    timestamp=time.time()
                )
                usage.estimated_cost_usd = self._estimate_cost(usage)
                self.log_usage(usage)

                # Parse JSON response
                content = response.choices[0].message.content
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.debug(f"Raw response: {content}")
                    raise ValueError(f"Invalid JSON response from OpenAI: {e}")

            except RateLimitError as e:
                logger.warning(f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise

            except (APIError, APIConnectionError) as e:
                logger.error(f"API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise

            except ValueError:
                # Don't retry JSON parsing errors
                raise

            except Exception as e:
                logger.error(f"Unexpected error calling OpenAI API: {e}")
                raise

    def _estimate_cost(self, usage: LLMUsage) -> float:
        """
        Estimate the cost of an API call based on token usage.

        Args:
            usage: LLMUsage object with token counts

        Returns:
            Estimated cost in USD
        """
        # Find the pricing tier for this model
        pricing = None
        for model_key, prices in self.PRICING.items():
            if model_key in self.model.lower():
                pricing = prices
                break

        if pricing is None:
            logger.warning(f"No pricing info for model {self.model}, using gpt-4-turbo defaults")
            pricing = self.PRICING["gpt-4-turbo"]

        # Calculate cost (prices are per 1K tokens)
        prompt_cost = (usage.prompt_tokens / 1000) * pricing["prompt"]
        completion_cost = (usage.completion_tokens / 1000) * pricing["completion"]

        return prompt_cost + completion_cost
