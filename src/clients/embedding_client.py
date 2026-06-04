"""Abstract base class for embedding clients."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List
import time

from loguru import logger


@dataclass
class EmbeddingUsage:
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model: str = ""
    timestamp: float = field(default_factory=time.time)


class EmbeddingClient(ABC):
    """Abstract base for embedding providers (OpenAI, Ollama, etc.)."""

    def __init__(self, api_key: str, model: str, dims: int) -> None:
        self.api_key = api_key
        self.model = model
        self.dims = dims
        self.total_usage = EmbeddingUsage(model=model)

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts; returns one vector per text."""
        pass

    @abstractmethod
    def _estimate_cost(self, token_count: int) -> float:
        """Return estimated cost in USD for the given token count."""
        pass

    def get_total_usage(self) -> Dict:
        return {
            "model": self.total_usage.model,
            "total_tokens": self.total_usage.total_tokens,
            "estimated_cost_usd": round(self.total_usage.estimated_cost_usd, 6),
        }

    def reset_usage(self) -> None:
        self.total_usage = EmbeddingUsage(model=self.model)

    def _log_usage(self, token_count: int, cost: float) -> None:
        self.total_usage.total_tokens += token_count
        self.total_usage.estimated_cost_usd += cost
        logger.debug(
            f"Embedding call: {token_count} tokens, ${cost:.6f} "
            f"(cumulative: ${self.total_usage.estimated_cost_usd:.4f})"
        )
