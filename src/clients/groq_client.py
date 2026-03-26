"""
Groq client implementation for fast, free-tier LLM inference.

Groq provides OpenAI-compatible API access to open-source models
(Llama, Mixtral, Gemma) with a generous free tier.
"""

import json
import time
from typing import Any, Dict, Optional

from groq import Groq, APIError, RateLimitError, APIConnectionError
from loguru import logger

from .llm_client import LLMClient, LLMUsage


class GroqClient(LLMClient):
    """
    Groq implementation of LLMClient.

    Supports fast inference on open-source models via Groq's free tier:
    - llama-3.3-70b-versatile  (best quality, recommended)
    - llama-3.1-8b-instant     (fastest)
    - mixtral-8x7b-32768       (large context window)
    - gemma2-9b-it             (Google's Gemma 2)
    """

    # Groq free tier is $0.00 — pricing shown for pay-as-you-go reference
    PRICING = {
        "llama-3.3-70b-versatile": {"prompt": 0.59, "completion": 0.79},
        "llama-3.1-70b-versatile": {"prompt": 0.59, "completion": 0.79},
        "llama-3.1-8b-instant": {"prompt": 0.05, "completion": 0.08},
        "mixtral-8x7b-32768": {"prompt": 0.24, "completion": 0.24},
        "gemma2-9b-it": {"prompt": 0.20, "completion": 0.20},
    }

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Groq client.

        Args:
            api_key: Groq API key (get one free at console.groq.com)
            model: Model name (e.g., "llama-3.3-70b-versatile")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        super().__init__(api_key, model, timeout)
        self.client = Groq(api_key=api_key, timeout=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"Initialized Groq client with model: {model}")

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """Generate a text completion using Groq's chat completions API."""
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
                logger.error(f"Unexpected error calling Groq API: {e}")
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
        """Generate a structured JSON response using Groq's JSON mode."""
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

                # Remove markdown code blocks if present
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
                    raise ValueError(f"Invalid JSON response from Groq: {e}")

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
                logger.error(f"Unexpected error calling Groq API: {e}")
                raise

    def _estimate_cost(self, usage: LLMUsage) -> float:
        """Estimate cost based on token usage (free tier = $0.00)."""
        pricing = None
        for model_key, prices in self.PRICING.items():
            if model_key in self.model.lower():
                pricing = prices
                break

        if pricing is None:
            logger.debug(f"No pricing info for model {self.model}, assuming free tier")
            return 0.0

        prompt_cost = (usage.prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (usage.completion_tokens / 1_000_000) * pricing["completion"]
        return prompt_cost + completion_cost
