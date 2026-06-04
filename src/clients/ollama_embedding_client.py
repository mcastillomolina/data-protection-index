"""Ollama embedding client using the /api/embed endpoint (nomic-embed-text by default)."""

import time
from typing import List

import httpx
from loguru import logger

from src.clients.embedding_client import EmbeddingClient


class OllamaEmbeddingClient(EmbeddingClient):
    """
    Embeds text via a local Ollama instance.

    Uses the /api/embed endpoint (Ollama ≥ 0.1.30) which accepts a batch of inputs.
    No API key required.

    Start Ollama and pull the model before use:
      docker compose up -d
      docker exec dpi_ollama ollama pull nomic-embed-text
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        # No API key for Ollama — pass empty string to satisfy base class
        super().__init__(api_key="", model=model, dims=768)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(
            f"Initialized OllamaEmbeddingClient with model: {model} at {base_url}"
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts; returns one 768-dim vector per text."""
        if not texts:
            return []

        url = f"{self.base_url}/api/embed"
        payload = {"model": self.model, "input": texts}

        last_error: Exception = RuntimeError("No attempts made")
        for attempt in range(1, self.max_retries + 1):
            try:
                response = httpx.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                vectors: List[List[float]] = data["embeddings"]
                # Ollama doesn't report token counts; approximate from char length
                approx_tokens = sum(len(t) // 4 for t in texts)
                self._log_usage(approx_tokens, 0.0)  # free
                return vectors
            except httpx.HTTPStatusError as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = self.retry_delay * attempt
                    logger.warning(
                        f"Ollama HTTP {e.response.status_code}, retrying in {wait}s "
                        f"(attempt {attempt})"
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"Ollama embedding failed after {self.max_retries} attempts: {e}"
                    )
            except httpx.ConnectError as e:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Ensure the container is running: docker compose up -d"
                ) from e

        raise last_error

    def _estimate_cost(self, token_count: int) -> float:
        return 0.0  # local — no cost
