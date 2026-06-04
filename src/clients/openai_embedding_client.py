"""OpenAI embedding client using text-embedding-3-small."""

import time
from typing import List

from loguru import logger
from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from src.clients.embedding_client import EmbeddingClient


class OpenAIEmbeddingClient(EmbeddingClient):
    """Embeds text via OpenAI's embeddings API (text-embedding-3-small by default)."""

    PRICING: dict = {
        "text-embedding-3-small": 0.00002,  # USD per 1K tokens
        "text-embedding-3-large": 0.00013,
        "text-embedding-ada-002": 0.00010,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        super().__init__(api_key, model, dims=1536)
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"Initialized OpenAIEmbeddingClient with model: {model}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts; returns one 1536-dim vector per text."""
        if not texts:
            return []

        last_error: Exception = RuntimeError("No attempts made")
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model,
                )
                token_count = response.usage.total_tokens
                cost = self._estimate_cost(token_count)
                self._log_usage(token_count, cost)
                return [item.embedding for item in response.data]
            except RateLimitError as e:
                last_error = e
                wait = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(f"Rate limit hit, retrying in {wait}s (attempt {attempt})")
                time.sleep(wait)
            except (APIError, APIConnectionError) as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = self.retry_delay * attempt
                    logger.warning(f"API error, retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"Embedding failed after {self.max_retries} attempts: {e}")

        raise last_error

    def _estimate_cost(self, token_count: int) -> float:
        rate = self.PRICING.get(self.model, 0.00002)
        return (token_count / 1000) * rate
