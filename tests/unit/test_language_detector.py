"""Unit tests for LanguageDetector."""

from unittest.mock import patch, MagicMock

import pytest

from src.core.language_detector import LanguageDetector


class TestLanguageDetector:
    def setup_method(self):
        self.detector = LanguageDetector()

    def test_english_text(self):
        text = (
            "The right to data protection is a fundamental right in democratic societies. "
            "Personal data must be processed lawfully and transparently."
        )
        lang = self.detector.detect(text)
        assert lang == "en"

    def test_spanish_text(self):
        text = (
            "El derecho a la protección de datos es un derecho fundamental. "
            "Los datos personales deben ser tratados de manera lícita y transparente."
        )
        lang = self.detector.detect(text)
        assert lang == "es"

    def test_german_text(self):
        text = (
            "Das Recht auf Datenschutz ist ein Grundrecht in demokratischen Gesellschaften. "
            "Personenbezogene Daten müssen rechtmäßig und transparent verarbeitet werden."
        )
        lang = self.detector.detect(text)
        assert lang == "de"

    def test_empty_text_returns_unknown(self):
        assert self.detector.detect("") == "unknown"
        assert self.detector.detect("   ") == "unknown"

    def test_lang_detect_exception_returns_unknown(self):
        with patch("src.core.language_detector.LanguageDetector.detect_with_confidence") as mock:
            mock.return_value = ("unknown", 0.0)
            result = self.detector.detect("x")
        assert result == "unknown"

    def test_detect_with_confidence_returns_tuple(self):
        text = "Personal data protection law enforcement authority penalties."
        lang, prob = self.detector.detect_with_confidence(text)
        assert isinstance(lang, str)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_short_text_does_not_raise(self):
        # Very short texts may cause LangDetectException — must not propagate
        result = self.detector.detect("ok")
        assert isinstance(result, str)

    def test_uses_only_sample_chars(self):
        # Prepend 2000+ chars of English then add Spanish — should detect English
        english_lead = "The right to data protection. " * 70  # ~2100 chars
        spanish_tail = "protección de datos personales " * 100
        text = english_lead + spanish_tail
        lang = self.detector.detect(text)
        assert lang == "en"

    def test_langdetect_import_error_returns_unknown(self):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "langdetect":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            detector = LanguageDetector()
            lang, prob = detector.detect_with_confidence("hello world")
        assert lang == "unknown"
        assert prob == 0.0
