"""
DeepSeek client implementation using the OpenAI-compatible API.

DeepSeek provides an OpenAI-compatible API for its models:
- deepseek-chat      (DeepSeek-V3, general purpose)
- deepseek-reasoner  (DeepSeek-R1, chain-of-thought reasoning)
"""

import json
import time
from typing import Any, Dict, Optional

from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from loguru import logger

from .llm_client import LLMClient, LLMUsage

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekClient(LLMClient):
    """
    DeepSeek implementation of LLMClient via OpenAI-compatible API.

    Supported models:
    - deepseek-chat      (DeepSeek-V3, fast and cost-efficient)
    - deepseek-reasoner  (DeepSeek-R1, extended reasoning)
    """

    # Pricing per 1M tokens (USD)
    PRICING = {
        "deepseek-v4-flash": {"prompt": 0.14, "completion": 0.28},
        "deepseek-v4-pro": {"prompt": 0.435, "completion": 0.87},
        "deepseek-chat": {"prompt": 0.27, "completion": 1.10},
        "deepseek-reasoner": {"prompt": 0.55, "completion": 2.19},
    }

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key (platform.deepseek.com)
            model: Model name (e.g., "deepseek-chat")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        super().__init__(api_key, model, timeout)
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"Initialized DeepSeek client with model: {model}")

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """Generate a text completion using DeepSeek's chat completions API."""
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
                logger.error(f"Unexpected error calling DeepSeek API: {e}")
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
        """Generate a structured JSON response using DeepSeek's JSON mode."""
        if system_prompt is None:
            system_prompt = "You are a helpful assistant that responds in JSON format."
        else:
            system_prompt += "\n\nYou must respond with valid JSON only, no additional text."

        if schema:
            prompt += f"\n\nPlease respond with JSON matching this schema:\n{json.dumps(schema, indent=2)}"

        prompt += "\n\nIMPORTANT: Respond with only valid JSON, no markdown formatting or code blocks."

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
                    response_format={"type": "json_object"},
                    **kwargs
                )

                usage = LLMUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    model=self.model,
                    timestamp=time.time()
                )
                usage.estimated_cost_usd = self._estimate_cost(usage)
                self.log_usage(usage)

                content = response.choices[0].message.content.strip()

                if content.startswith("```"):
                    lines = content.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].strip() == "```":
                        lines = lines[:-1]
                    content = "\n".join(lines)

                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.debug(f"Raw response: {content}")
                    raise ValueError(f"Invalid JSON response from DeepSeek: {e}")

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
                raise

            except Exception as e:
                logger.error(f"Unexpected error calling DeepSeek API: {e}")
                raise

    def _estimate_cost(self, usage: LLMUsage) -> float:
        """Estimate cost based on token usage."""
        pricing = self.PRICING.get(self.model)
        if pricing is None:
            logger.debug(f"No pricing info for model {self.model}, defaulting to deepseek-chat rates")
            pricing = self.PRICING["deepseek-chat"]

        prompt_cost = (usage.prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (usage.completion_tokens / 1_000_000) * pricing["completion"]
        return prompt_cost + completion_cost
