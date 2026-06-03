"""Language detection for Phase 3 — deterministic, no LLM."""

from loguru import logger


class LanguageDetector:
    """Detects the primary language of a text using langdetect."""

    # Use only the first N chars — language is stable across a document
    _SAMPLE_CHARS = 2000

    def detect(self, text: str) -> str:
        """Return ISO 639-1 language code, or 'unknown' on failure."""
        lang, _ = self.detect_with_confidence(text)
        return lang

    def detect_with_confidence(self, text: str) -> tuple[str, float]:
        """Return (ISO 639-1 code, probability) or ('unknown', 0.0) on failure."""
        try:
            from langdetect import detect_langs, LangDetectException  # type: ignore[import]
        except ImportError:
            logger.warning("langdetect not installed — returning 'unknown'")
            return "unknown", 0.0

        sample = text[: self._SAMPLE_CHARS].strip()
        if not sample:
            return "unknown", 0.0

        try:
            from langdetect import LangDetectException  # noqa: F811

            results = detect_langs(sample)
            if not results:
                return "unknown", 0.0
            top = results[0]
            lang: str = top.lang
            prob: float = top.prob
            logger.debug(f"Language detected: {lang} (prob={prob:.2f})")
            return lang, prob
        except Exception as exc:  # catches LangDetectException + any other
            logger.debug(f"Language detection failed: {exc}")
            return "unknown", 0.0
