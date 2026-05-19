"""
Mistral AI client implementation using the official mistralai SDK.

Mistral provides a range of open and proprietary models:
- mistral-large-latest    (most capable, best for complex tasks)
- mistral-small-latest    (fast, cost-efficient)
- open-mistral-7b         (open-source 7B)
- open-mixtral-8x7b       (open-source MoE 8x7B)
- open-mixtral-8x22b      (open-source MoE 8x22B)
"""

import json
import time
from typing import Any, Dict, Optional

from mistralai import Mistral
from mistralai.models import SDKError
from loguru import logger

from .llm_client import LLMClient, LLMUsage


class MistralClient(LLMClient):
    """
    Mistral AI implementation of LLMClient.

    Supported models:
    - mistral-large-latest   (frontier, best reasoning)
    - mistral-small-latest   (fast and affordable)
    - open-mistral-7b        (open-weight, low cost)
    - open-mixtral-8x7b      (open-weight MoE)
    - open-mixtral-8x22b     (open-weight MoE, high capacity)
    """

    # Pricing per 1M tokens (USD)
    PRICING = {
        "mistral-large-latest": {"prompt": 2.00, "completion": 6.00},
        "mistral-medium-latest": {"prompt": 2.70, "completion": 8.10},
        "mistral-small-latest": {"prompt": 0.20, "completion": 0.60},
        "open-mistral-7b": {"prompt": 0.25, "completion": 0.25},
        "open-mixtral-8x7b": {"prompt": 0.70, "completion": 0.70},
        "open-mixtral-8x22b": {"prompt": 2.00, "completion": 6.00},
        "codestral-latest": {"prompt": 0.20, "completion": 0.60},
    }

    # Models that support JSON mode
    JSON_MODE_MODELS = {
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest",
        "open-mixtral-8x7b",
        "open-mixtral-8x22b",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-small-latest",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Mistral client.

        Args:
            api_key: Mistral API key (console.mistral.ai)
            model: Model name (e.g., "mistral-small-latest")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        super().__init__(api_key, model, timeout)
        self.client = Mistral(api_key=api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"Initialized Mistral client with model: {model}")

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """Generate a text completion using Mistral's chat completions API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.complete(
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

            except SDKError as e:
                if e.status_code == 429:
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        raise
                else:
                    logger.error(f"API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        raise

            except Exception as e:
                logger.error(f"Unexpected error calling Mistral API: {e}")
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
        """Generate a structured JSON response using Mistral's JSON mode when available."""
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

        # Use JSON mode for supported models, fall back to prompt-based for others
        use_json_mode = self.model in self.JSON_MODE_MODELS
        extra = {"response_format": {"type": "json_object"}} if use_json_mode else {}

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra,
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
                    raise ValueError(f"Invalid JSON response from Mistral: {e}")

            except SDKError as e:
                if e.status_code == 429:
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        raise
                else:
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
                logger.error(f"Unexpected error calling Mistral API: {e}")
                raise

    def _estimate_cost(self, usage: LLMUsage) -> float:
        """Estimate cost based on token usage."""
        pricing = None
        for model_key, prices in self.PRICING.items():
            if model_key in self.model.lower():
                pricing = prices
                break

        if pricing is None:
            logger.debug(f"No pricing info for model {self.model}, assuming mistral-small rates")
            pricing = self.PRICING["mistral-small-latest"]

        prompt_cost = (usage.prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (usage.completion_tokens / 1_000_000) * pricing["completion"]
        return prompt_cost + completion_cost
