"""
Anthropic client implementation for Claude models.

This module provides an implementation of the LLMClient interface for Anthropic's
Claude models, including support for structured JSON outputs.
"""

import json
import time
from typing import Any, Dict, Optional

from anthropic import Anthropic, APIError, RateLimitError, APIConnectionError
from loguru import logger

from .llm_client import LLMClient, LLMUsage


class AnthropicClient(LLMClient):
    """
    Anthropic implementation of LLMClient using Claude models.

    Supports Claude 3 models (Opus, Sonnet, Haiku) with JSON parsing
    for structured outputs.
    """

    # Pricing per 1M tokens (as of 2024)
    # Update these values based on current Anthropic pricing
    PRICING = {
        "claude-3-opus": {"prompt": 15.0, "completion": 75.0},
        "claude-3-sonnet": {"prompt": 3.0, "completion": 15.0},
        "claude-3-haiku": {"prompt": 0.25, "completion": 1.25},
        "claude-3-5-sonnet": {"prompt": 3.0, "completion": 15.0},
    }

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key
            model: Model name (e.g., "claude-3-opus-20240229", "claude-3-5-sonnet-20241022")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        super().__init__(api_key, model, timeout)
        self.client = Anthropic(api_key=api_key, timeout=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"Initialized Anthropic client with model: {model}")

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Generate a text completion using Anthropic's messages API.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system prompt for instruction
            temperature: Sampling temperature (0.0 to 1.0 for Claude)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            Generated text response

        Raises:
            Exception: If the API call fails after retries
        """
        for attempt in range(self.max_retries):
            try:
                # Build the request
                request_params = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                    **kwargs
                }

                # Add system prompt if provided
                if system_prompt:
                    request_params["system"] = system_prompt

                response = self.client.messages.create(**request_params)

                # Extract usage and log it
                usage = LLMUsage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                    model=self.model,
                    timestamp=time.time()
                )
                usage.estimated_cost_usd = self._estimate_cost(usage)
                self.log_usage(usage)

                # Extract text from response
                return response.content[0].text

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
                logger.error(f"Unexpected error calling Anthropic API: {e}")
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
        Generate a structured JSON response using Anthropic's messages API.

        Note: Claude doesn't have a native JSON mode, so we instruct it via
        the system prompt and parse the response.

        Args:
            prompt: The user prompt/message
            system_prompt: Optional system prompt (we'll add JSON instructions)
            schema: Optional JSON schema to include in the prompt
            temperature: Sampling temperature (lower for more consistent JSON)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            Parsed JSON response as a dictionary

        Raises:
            ValueError: If the response is not valid JSON
            Exception: If the API call fails after retries
        """
        # Build system prompt with JSON instructions
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that responds in JSON format."
        else:
            system_prompt += "\n\nYou must respond with valid JSON only, no additional text."

        # Add schema to prompt if provided
        if schema:
            prompt += f"\n\nPlease respond with JSON matching this schema:\n{json.dumps(schema, indent=2)}"

        prompt += "\n\nIMPORTANT: Respond with only valid JSON, no markdown formatting or code blocks."

        for attempt in range(self.max_retries):
            try:
                # Build the request
                request_params = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                    "system": system_prompt,
                    **kwargs
                }

                response = self.client.messages.create(**request_params)

                # Extract usage and log it
                usage = LLMUsage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                    model=self.model,
                    timestamp=time.time()
                )
                usage.estimated_cost_usd = self._estimate_cost(usage)
                self.log_usage(usage)

                # Extract and parse text
                content = response.content[0].text.strip()

                # Remove markdown code blocks if present
                if content.startswith("```"):
                    # Remove opening ```json or ``` and closing ```
                    lines = content.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].strip() == "```":
                        lines = lines[:-1]
                    content = "\n".join(lines)

                # Parse JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.debug(f"Raw response: {content}")
                    raise ValueError(f"Invalid JSON response from Anthropic: {e}")

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
                logger.error(f"Unexpected error calling Anthropic API: {e}")
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
            logger.warning(f"No pricing info for model {self.model}, using claude-3-sonnet defaults")
            pricing = self.PRICING["claude-3-sonnet"]

        # Calculate cost (prices are per 1M tokens)
        prompt_cost = (usage.prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (usage.completion_tokens / 1_000_000) * pricing["completion"]

        return prompt_cost + completion_cost
