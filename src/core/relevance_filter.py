"""
Relevance filter using LLM.

This module uses an LLM to score and filter search results by relevance
to a specific document.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from loguru import logger

from src.clients.llm_client import LLMClient
from src.models.document import DocumentMetadata, SearchResult, ScoredResult
from src.prompts.relevance_scoring import (
    SYSTEM_PROMPT,
    RELEVANCE_SCORING_SCHEMA,
    create_relevance_scoring_prompt,
)


class RelevanceFilter:
    """
    Uses LLM to score and filter search results by relevance.

    This class leverages a language model to evaluate how relevant each
    search result is to finding a specific legal document.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        temperature: float = 0.2,
        max_tokens: int = 4000,
        min_relevance_score: float = 6.0,
        cache_dir: Optional[Path] = None,
    ):
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_relevance_score = min_relevance_score
        self._cache_dir: Optional[Path] = None

        if cache_dir is not None:
            self._cache_dir = Path(cache_dir) / "relevance"
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Relevance cache enabled — dir: {self._cache_dir}")

        logger.info(
            f"Initialized RelevanceFilter "
            f"(min_score={min_relevance_score}, temp={temperature})"
        )

    def filter_results(
        self,
        document: DocumentMetadata,
        results: List[SearchResult],
        country_name: str,
        top_n: int = 5,
        criterion_number: int = None,
        criterion_core_question: str = None,
    ) -> List[ScoredResult]:
        """
        Score and filter search results by relevance.

        Args:
            document: Document being searched for
            results: List of SearchResult objects to score
            country_name: Name of the country (for context)
            top_n: Number of top results to return

        Returns:
            List of ScoredResult objects, sorted by relevance (highest first)

        Raises:
            ValueError: If LLM response is invalid
            Exception: If LLM call fails
        """
        if not results:
            logger.warning("No results to filter")
            return []

        logger.info(
            f"Filtering {len(results)} results for '{document.official_name}'"
        )

        # Convert SearchResult objects to dicts for prompt
        results_dicts = [
            {
                "url": r.url,
                "title": r.title,
                "snippet": r.snippet,
                "domain": r.source_domain
            }
            for r in results
        ]

        # Create prompt
        prompt = create_relevance_scoring_prompt(
            document_name=document.official_name,
            document_type=document.document_type,
            country_name=country_name,
            search_results=results_dicts,
            min_score=self.min_relevance_score,
            criterion_number=criterion_number,
            criterion_core_question=criterion_core_question,
        )

        try:
            # Call LLM
            logger.debug(f"Calling LLM to score {len(results)} results")
            response = self.llm_client.complete_json(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                schema=RELEVANCE_SCORING_SCHEMA,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Validate response structure
            if "scored_results" not in response:
                raise ValueError("LLM response missing 'scored_results' field")

            scored_data = response["scored_results"]
            logger.info(f"LLM scored {len(scored_data)} results")

            # Convert to ScoredResult objects
            scored_results = []
            for i, score_data in enumerate(scored_data):
                try:
                    # Find matching SearchResult by URL
                    url = score_data["url"]
                    matching_result = next(
                        (r for r in results if r.url == url),
                        None
                    )

                    if not matching_result:
                        logger.warning(f"No matching result for URL: {url}")
                        continue

                    scored = ScoredResult(
                        search_result=matching_result,
                        relevance_score=float(score_data["relevance_score"]),
                        reasoning=score_data["reasoning"],
                        is_likely_official=score_data["is_likely_official"],
                        confidence=score_data["confidence"]
                    )

                    scored_results.append(scored)

                    logger.debug(
                        f"Result {i+1}: score={scored.relevance_score:.1f}, "
                        f"official={scored.is_likely_official}, "
                        f"confidence={scored.confidence}"
                    )

                except Exception as e:
                    logger.warning(f"Failed to create ScoredResult: {e}")
                    logger.debug(f"Score data: {score_data}")
                    continue

            # Sort by relevance score (highest first)
            scored_results.sort(key=lambda x: x.relevance_score, reverse=True)

            # Filter by minimum score
            relevant_results = [
                r for r in scored_results
                if r.relevance_score >= self.min_relevance_score
            ]

            logger.info(
                f"Found {len(relevant_results)}/{len(scored_results)} results "
                f"with score >= {self.min_relevance_score}"
            )

            # Limit to top N
            top_results = relevant_results[:top_n]

            # Log summary if available
            if "summary" in response:
                summary = response["summary"]
                logger.info(
                    f"Summary - Total: {summary.get('total_results', 0)}, "
                    f"Highly relevant: {summary.get('highly_relevant_count', 0)}, "
                    f"Official: {summary.get('official_source_count', 0)}"
                )
                if "recommended_top_result" in summary:
                    logger.info(f"Top result: {summary['recommended_top_result']}")

            return top_results

        except ValueError as e:
            logger.error(f"Invalid LLM response: {e}")
            raise

        except Exception as e:
            logger.error(f"Error filtering results: {e}")
            raise

    def filter_results_batch(
        self,
        document: DocumentMetadata,
        results: List[SearchResult],
        country_name: str,
        batch_size: int = 20,
        top_n: int = 5,
        criterion_number: int = None,
        criterion_core_question: str = None,
    ) -> List[ScoredResult]:
        """
        Filter results in batches (for large result sets).

        Args:
            document: Document being searched for
            results: List of SearchResult objects to score
            country_name: Name of the country
            batch_size: Number of results to score per LLM call
            top_n: Number of top results to return overall

        Returns:
            List of top N ScoredResult objects
        """
        logger.info(
            f"Filtering {len(results)} results in batches of {batch_size}"
        )

        if self._cache_dir is not None:
            key = self._relevance_cache_key(document.official_name, country_name, results)
            cached = self._load_relevance_cache(key)
            if cached is not None:
                logger.info(
                    f"[CACHE HIT] Relevance scores for '{document.official_name}' "
                    f"({len(cached)} results)"
                )
                return cached[:top_n]

        all_scored = []

        # Process in batches
        for i in range(0, len(results), batch_size):
            batch = results[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch)} results")

            try:
                scored = self.filter_results(
                    document=document,
                    results=batch,
                    country_name=country_name,
                    top_n=batch_size,
                    criterion_number=criterion_number,
                    criterion_core_question=criterion_core_question,
                )
                all_scored.extend(scored)

            except Exception as e:
                logger.error(f"Failed to process batch: {e}")
                continue

        # Sort all scored results
        all_scored.sort(key=lambda x: x.relevance_score, reverse=True)

        # Return top N overall
        top_results = all_scored[:top_n]

        logger.info(f"Returning top {len(top_results)} results from {len(all_scored)} scored")

        if self._cache_dir is not None and top_results:
            self._save_relevance_cache(key, document.official_name, country_name, results, top_results)

        return top_results

    def _relevance_cache_key(
        self, doc_name: str, country_name: str, results: List[SearchResult]
    ) -> str:
        url_fingerprint = ",".join(sorted(r.url for r in results))
        raw = f"{doc_name}|{country_name}|{url_fingerprint}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _load_relevance_cache(self, key: str) -> Optional[List[ScoredResult]]:
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [ScoredResult(**r) for r in data["results"]]
        except Exception as e:
            logger.warning(f"Failed to read relevance cache {path}: {e}")
            return None

    def _save_relevance_cache(
        self,
        key: str,
        doc_name: str,
        country_name: str,
        input_results: List[SearchResult],
        scored: List[ScoredResult],
    ) -> None:
        path = self._cache_dir / f"{key}.json"
        payload = {
            "doc_name": doc_name,
            "country_name": country_name,
            "url_fingerprint": ",".join(sorted(r.url for r in input_results)),
            "cached_at": datetime.now().isoformat(),
            "results": [r.model_dump() for r in scored],
        }
        try:
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.debug(f"Cached {len(scored)} relevance results → {path.name}")
        except Exception as e:
            logger.warning(f"Failed to write relevance cache {path}: {e}")

    def get_official_results(
        self,
        scored_results: List[ScoredResult],
        min_score: float = None
    ) -> List[ScoredResult]:
        """
        Filter to only results from likely official sources.

        Args:
            scored_results: List of ScoredResult objects
            min_score: Optional minimum score (uses class default if None)

        Returns:
            List of ScoredResult objects from official sources
        """
        min_score = min_score if min_score is not None else self.min_relevance_score

        official = [
            r for r in scored_results
            if r.is_likely_official and r.relevance_score >= min_score
        ]

        logger.info(
            f"Found {len(official)}/{len(scored_results)} official results "
            f"with score >= {min_score}"
        )

        return official

    def get_high_confidence_results(
        self,
        scored_results: List[ScoredResult],
        min_score: float = None
    ) -> List[ScoredResult]:
        """
        Filter to only high-confidence results.

        Args:
            scored_results: List of ScoredResult objects
            min_score: Optional minimum score (uses class default if None)

        Returns:
            List of high-confidence ScoredResult objects
        """
        min_score = min_score if min_score is not None else self.min_relevance_score

        high_conf = [
            r for r in scored_results
            if r.confidence == "high" and r.relevance_score >= min_score
        ]

        logger.info(
            f"Found {len(high_conf)}/{len(scored_results)} high-confidence results "
            f"with score >= {min_score}"
        )

        return high_conf

    def get_scoring_summary(self, scored_results: List[ScoredResult]) -> dict:
        """
        Get a summary of scoring results.

        Args:
            scored_results: List of ScoredResult objects

        Returns:
            Dictionary with summary statistics
        """
        if not scored_results:
            return {
                "total_results": 0,
                "avg_score": 0.0,
                "official_count": 0,
                "high_confidence_count": 0,
                "score_distribution": {}
            }

        scores = [r.relevance_score for r in scored_results]
        official_count = sum(1 for r in scored_results if r.is_likely_official)
        high_conf_count = sum(1 for r in scored_results if r.confidence == "high")

        # Score distribution by ranges
        distribution = {
            "9-10": sum(1 for s in scores if 9 <= s <= 10),
            "7-8.9": sum(1 for s in scores if 7 <= s < 9),
            "5-6.9": sum(1 for s in scores if 5 <= s < 7),
            "3-4.9": sum(1 for s in scores if 3 <= s < 5),
            "0-2.9": sum(1 for s in scores if 0 <= s < 3),
        }

        summary = {
            "total_results": len(scored_results),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "official_count": official_count,
            "high_confidence_count": high_conf_count,
            "score_distribution": distribution
        }

        return summary
