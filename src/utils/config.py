"""Configuration management for Data Protection Index."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv


@dataclass
class LLMConfig:
    """LLM configuration."""

    provider: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    max_retries: int


@dataclass
class SearchConfig:
    """Search configuration."""

    provider: str
    max_results_per_query: int
    timeout: int
    rate_limit_delay: float


@dataclass
class PipelineConfig:
    """Pipeline configuration."""

    max_documents_per_country: int
    top_urls_per_document: int
    min_relevance_score: float
    enable_deduplication: bool
    enable_caching: bool
    cache_dir: str


@dataclass
class OutputConfig:
    """Output configuration."""

    directory: str
    save_intermediate: bool
    intermediate_dir: str
    format: str
    pretty_print: bool
    include_metadata: bool


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str
    format: str
    file: str
    rotation: str
    retention: str


class Config:
    """Main configuration class."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config YAML file. Defaults to config/config.yaml
        """
        self.config_path = config_path or Path("config/config.yaml")
        self._load_env_vars()
        self._load_config()
        self._load_document_types()
        self._load_countries()

    def _load_env_vars(self) -> None:
        """Load environment variables from .env file."""
        load_dotenv()

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.serpapi_key = os.getenv("SERPAPI_KEY")

        # Allow environment to override config path
        env_config_path = os.getenv("CONFIG_PATH")
        if env_config_path:
            self.config_path = Path(env_config_path)

    def _load_config(self) -> None:
        """Load main configuration from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        self.project = config["project"]
        self.llm = LLMConfig(**config["llm"])
        self.search = SearchConfig(**config["search"])
        self.pipeline = PipelineConfig(**config["pipeline"])
        self.output = OutputConfig(**config["output"])
        self.logging = LoggingConfig(**config["logging"])

        # Allow environment variable to override log level
        log_level = os.getenv("LOG_LEVEL")
        if log_level:
            self.logging.level = log_level

    def _load_document_types(self) -> None:
        """Load document type definitions."""
        doc_types_path = Path("config/document_types.yaml")
        if doc_types_path.exists():
            with open(doc_types_path) as f:
                data = yaml.safe_load(f)
                self.document_types = data["document_types"]
        else:
            self.document_types = []

    def _load_countries(self) -> None:
        """Load country metadata."""
        countries_path = Path("config/countries.yaml")
        if countries_path.exists():
            with open(countries_path) as f:
                data = yaml.safe_load(f)
                self._countries_data = {c["name"]: c for c in data["countries"]}
        else:
            self._countries_data = {}

    def get_country_metadata(self, country_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific country.

        Args:
            country_name: Name of the country

        Returns:
            Dictionary with country metadata, or None if not found
        """
        return self._countries_data.get(country_name)

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Check API keys based on provider
        if self.llm.provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        if self.llm.provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        if self.search.provider == "serpapi" and not self.serpapi_key:
            raise ValueError("SERPAPI_KEY not set in environment")

        # Validate numeric values
        if self.llm.temperature < 0 or self.llm.temperature > 1:
            raise ValueError("LLM temperature must be between 0 and 1")
        if self.pipeline.min_relevance_score < 0 or self.pipeline.min_relevance_score > 10:
            raise ValueError("Minimum relevance score must be between 0 and 10")

        return True

    def get_llm_client(self):
        """
        Factory method to create configured LLM client.

        Returns:
            LLMClient instance (imported lazily to avoid circular deps)
        """
        from src.clients.openai_client import OpenAIClient
        from src.clients.anthropic_client import AnthropicClient

        if self.llm.provider == "openai":
            return OpenAIClient(self.openai_api_key, self.llm.model, self)
        elif self.llm.provider == "anthropic":
            return AnthropicClient(self.anthropic_api_key, self.llm.model, self)
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm.provider}")

    def get_search_client(self):
        """
        Factory method to create configured search client.

        Returns:
            SearchClient instance
        """
        from src.clients.search_client import SearchClient

        return SearchClient(self.search.provider, self.serpapi_key, self)
