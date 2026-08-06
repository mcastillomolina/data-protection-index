"""Unit tests for IndexAggregator — mocks psycopg2, no DB or LLM involved."""

from unittest.mock import MagicMock, patch

import pytest

from src.core.index_aggregator import IndexAggregator


def _make_config(
    legal_weight: float = 0.40,
    enforcement_weight: float = 0.60,
    missing_strategy: str = "exclude",
    confidence_weighting: bool = True,
    min_criteria_for_ranking: int = 12,
):
    """Lightweight stand-in for ScoringConfig — IndexAggregator only reads these attrs."""
    cfg = MagicMock()
    cfg.legal_weight = legal_weight
    cfg.enforcement_weight = enforcement_weight
    cfg.missing_strategy = missing_strategy
    cfg.confidence_weighting = confidence_weighting
    cfg.min_criteria_for_ranking = min_criteria_for_ranking
    return cfg


def _make_aggregator(**config_overrides) -> IndexAggregator:
    return IndexAggregator("postgresql://fake:fake@localhost/fake", _make_config(**config_overrides))


def _score_row(
    criterion_number: int,
    dimension: str,
    criterion_score: float,
    confidence: str,
    legal_subscore: float | None = None,
    enforcement_subscore: float | None = None,
    information_opacity: bool = False,
) -> dict:
    return {
        "criterion_number": criterion_number,
        "dimension": dimension,
        "criterion_score": criterion_score,
        "legal_subscore": legal_subscore,
        "enforcement_subscore": enforcement_subscore,
        "confidence": confidence,
        "information_opacity": information_opacity,
    }


# Germany, country_id=18, reference_year=2026, model_used='deepseek-chat' — real rows
# reproduced by hand in claude/AGGREGATION_FORMULA.md. Expected DB output:
# legal_score=4.2857, enforcement_score=3.2955, final_score=3.6916.
GERMANY_ROWS = [
    _score_row(1,  "legal",       5.0, "high",   legal_subscore=5.0),
    _score_row(2,  "legal",       5.0, "high",   legal_subscore=5.0),
    _score_row(3,  "enforcement", 4.0, "medium", enforcement_subscore=4.0),
    _score_row(4,  "mixed",       3.4, "medium", legal_subscore=4.0, enforcement_subscore=3.0),
    _score_row(5,  "mixed",       3.4, "medium", legal_subscore=4.0, enforcement_subscore=3.0),
    _score_row(6,  "enforcement", 3.0, "medium", legal_subscore=4.0, enforcement_subscore=2.0),
    _score_row(7,  "mixed",       3.4, "medium", legal_subscore=4.0, enforcement_subscore=3.0),
    _score_row(8,  "enforcement", 4.0, "medium", legal_subscore=5.0, enforcement_subscore=3.0),
    _score_row(9,  "mixed",       2.8, "low",    legal_subscore=4.0, enforcement_subscore=2.0),
    _score_row(10, "legal",       3.0, "low",    legal_subscore=3.0),
    _score_row(11, "enforcement", 3.0, "medium", legal_subscore=4.0, enforcement_subscore=2.0),
    _score_row(12, "enforcement", 3.0, "medium", legal_subscore=3.0, enforcement_subscore=2.0),
    _score_row(13, "legal",       4.0, "medium", legal_subscore=4.0),
    _score_row(14, "enforcement", 4.4, "medium", legal_subscore=5.0, enforcement_subscore=4.0),
]


@pytest.fixture
def mock_conn():
    """A mock psycopg2 connection whose cursor supports the `with conn.cursor(...) as cur` pattern."""
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)

    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


class TestWeightedMean:
    """Confidence weighting: high=1.0, medium=0.7, low=0.4."""

    def test_confidence_weights_applied(self):
        agg = _make_aggregator(confidence_weighting=True)
        pairs = [(5.0, "high"), (3.0, "medium"), (1.0, "low")]
        expected = (5.0 * 1.0 + 3.0 * 0.7 + 1.0 * 0.4) / (1.0 + 0.7 + 0.4)
        assert agg._weighted_mean(pairs) == pytest.approx(expected)

    def test_all_high_confidence_equals_simple_mean(self):
        agg = _make_aggregator(confidence_weighting=True)
        pairs = [(5.0, "high"), (3.0, "high")]
        assert agg._weighted_mean(pairs) == pytest.approx(4.0)

    def test_unknown_confidence_defaults_to_low_weight(self):
        agg = _make_aggregator(confidence_weighting=True)
        pairs = [(5.0, "bogus")]
        # CONFIDENCE_WEIGHTS.get(c, 0.4) -> weight 0.4, single value -> mean is just the value
        assert agg._weighted_mean(pairs) == pytest.approx(5.0)

    def test_confidence_weighting_disabled_uses_simple_mean(self):
        agg = _make_aggregator(confidence_weighting=False)
        pairs = [(5.0, "high"), (1.0, "low")]
        assert agg._weighted_mean(pairs) == pytest.approx(3.0)

    def test_empty_pairs_returns_none(self):
        agg = _make_aggregator()
        assert agg._weighted_mean([]) is None


class TestExtractPairs:
    """Legal criteria feed the legal pool, enforcement feed the enforcement pool,
    mixed criteria split their subscore across both."""

    def test_pure_legal_only_in_legal_pool(self):
        agg = _make_aggregator()
        scores = [_score_row(1, "legal", 5.0, "high", legal_subscore=5.0)]
        assert agg._extract_pairs(scores, "legal") == [(5.0, "high")]
        assert agg._extract_pairs(scores, "enforcement") == []

    def test_pure_enforcement_only_in_enforcement_pool(self):
        agg = _make_aggregator()
        scores = [_score_row(3, "enforcement", 4.0, "medium", enforcement_subscore=4.0)]
        assert agg._extract_pairs(scores, "enforcement") == [(4.0, "medium")]
        assert agg._extract_pairs(scores, "legal") == []

    def test_mixed_criterion_splits_subscores_across_both_pools(self):
        agg = _make_aggregator()
        scores = [
            _score_row(4, "mixed", 3.4, "medium", legal_subscore=4.0, enforcement_subscore=3.0)
        ]
        assert agg._extract_pairs(scores, "legal") == [(4.0, "medium")]
        assert agg._extract_pairs(scores, "enforcement") == [(3.0, "medium")]

    def test_mixed_criterion_uses_legal_subscore_not_criterion_score(self):
        """The pool gets the split subscore, not the blended criterion_score."""
        agg = _make_aggregator()
        scores = [
            _score_row(5, "mixed", 3.4, "medium", legal_subscore=4.0, enforcement_subscore=3.0)
        ]
        legal_pairs = agg._extract_pairs(scores, "legal")
        assert legal_pairs[0][0] != 3.4
        assert legal_pairs[0][0] == 4.0

    def test_mixed_criterion_falls_back_to_criterion_score_when_subscore_missing(self):
        agg = _make_aggregator()
        scores = [_score_row(4, "mixed", 3.4, "medium", legal_subscore=None, enforcement_subscore=3.0)]
        assert agg._extract_pairs(scores, "legal") == [(3.4, "medium")]


class TestMapCategory:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (5.0, "Consistently upholds human rights standards"),
            (4.1, "Consistently upholds human rights standards"),
            (4.09, "Significant protections and safeguards"),
            (3.6916, "Significant protections and safeguards"),
            (3.1, "Adequate safeguards against abuse"),
            (2.6, "Some safeguards but weakened protections"),
            (2.1, "Systemic failure to uphold safeguards"),
            (1.6, "Extensive surveillance societies"),
            (1.1, "Endemic surveillance societies"),
            (1.0, "Endemic surveillance societies"),
        ],
    )
    def test_thresholds(self, score, expected):
        agg = _make_aggregator()
        assert agg._map_category(score) == expected


class TestComputeCountryScore:
    @patch("psycopg2.connect")
    def test_germany_reproduces_documented_scores(self, mock_connect, mock_conn):
        """Reproduces the Germany (country_id=18) figures verified by hand in
        claude/AGGREGATION_FORMULA.md: legal=4.2857, enforcement=3.2955, final=3.6916."""
        conn, cursor = mock_conn
        mock_connect.return_value = conn
        cursor.fetchall.return_value = GERMANY_ROWS

        agg = _make_aggregator()  # defaults: legal_weight=0.40, enforcement_weight=0.60
        result = agg.compute_country_score(country_id=18, reference_year=2026, model_used="deepseek-chat")

        assert result.legal_score == pytest.approx(4.2857, abs=1e-4)
        assert result.enforcement_score == pytest.approx(3.2955, abs=1e-4)
        assert result.final_score == pytest.approx(3.6916, abs=1e-4)
        assert result.pi_category == "Significant protections and safeguards"
        assert result.criteria_count == 14
        assert result.missing_criteria == []
        assert result.partial_coverage is False

    @patch("psycopg2.connect")
    def test_simple_fixed_set_known_result(self, mock_connect, mock_conn):
        """A small, hand-computable set of criterion_scores with known confidences."""
        conn, cursor = mock_conn
        mock_connect.return_value = conn
        rows = [
            _score_row(1, "legal", 4.0, "high"),        # weight 1.0
            _score_row(2, "legal", 2.0, "low"),          # weight 0.4
            _score_row(3, "enforcement", 5.0, "high"),   # weight 1.0
            _score_row(6, "enforcement", 3.0, "medium"), # weight 0.7
        ]
        cursor.fetchall.return_value = rows

        agg = _make_aggregator(legal_weight=0.5, enforcement_weight=0.5)
        result = agg.compute_country_score(country_id=1, reference_year=2024)

        expected_legal = (4.0 * 1.0 + 2.0 * 0.4) / (1.0 + 0.4)
        expected_enforcement = (5.0 * 1.0 + 3.0 * 0.7) / (1.0 + 0.7)
        expected_final = expected_legal * 0.5 + expected_enforcement * 0.5

        assert result.legal_score == pytest.approx(round(expected_legal, 4), abs=1e-4)
        assert result.enforcement_score == pytest.approx(round(expected_enforcement, 4), abs=1e-4)
        assert result.final_score == pytest.approx(round(expected_final, 4), abs=1e-4)
        # Only 4/14 criteria present -> below default min_criteria_for_ranking=12
        assert result.criteria_count == 4
        assert result.partial_coverage is True
        assert len(result.missing_criteria) == 10

    @patch("psycopg2.connect")
    def test_writes_result_to_db(self, mock_connect, mock_conn):
        conn, cursor = mock_conn
        mock_connect.return_value = conn
        cursor.fetchall.return_value = [_score_row(1, "legal", 5.0, "high")]

        agg = _make_aggregator()
        agg.compute_country_score(country_id=1, reference_year=2024)

        insert_calls = [c for c in cursor.execute.call_args_list if "INSERT INTO country_index_scores" in c[0][0]]
        assert len(insert_calls) == 1
        assert conn.commit.called

    @patch("psycopg2.connect")
    def test_missing_strategy_exclude_leaves_criteria_out(self, mock_connect, mock_conn):
        conn, cursor = mock_conn
        mock_connect.return_value = conn
        cursor.fetchall.return_value = [_score_row(1, "legal", 5.0, "high")]

        agg = _make_aggregator(missing_strategy="exclude")
        result = agg.compute_country_score(country_id=1, reference_year=2024)

        assert result.criteria_count == 1
        assert result.legal_score == pytest.approx(5.0)
        assert result.enforcement_score is None
        # enforcement pool empty -> final_score falls back to legal_mean alone
        assert result.final_score == pytest.approx(5.0)

    @patch("psycopg2.connect")
    def test_missing_strategy_penalise_as_1_fills_gaps_with_low_confidence_ones(
        self, mock_connect, mock_conn
    ):
        conn, cursor = mock_conn
        mock_connect.return_value = conn
        cursor.fetchall.return_value = [_score_row(1, "legal", 5.0, "high")]

        agg = _make_aggregator(missing_strategy="penalise_as_1")
        result = agg.compute_country_score(country_id=1, reference_year=2024)

        # criteria_count only reflects criteria actually returned by the DB fetch,
        # but the missing criteria are synthesised into the score with value=1.0/low confidence.
        assert result.legal_score < 5.0
        assert result.enforcement_score == pytest.approx(1.0)
        assert result.missing_criteria != []

    @patch("psycopg2.connect")
    def test_no_scores_at_all_defaults_final_score_to_one(self, mock_connect, mock_conn):
        conn, cursor = mock_conn
        mock_connect.return_value = conn
        cursor.fetchall.return_value = []

        agg = _make_aggregator(missing_strategy="exclude")
        result = agg.compute_country_score(country_id=1, reference_year=2024)

        assert result.legal_score is None
        assert result.enforcement_score is None
        assert result.final_score == pytest.approx(1.0)
        assert result.criteria_count == 0
        assert result.partial_coverage is True

    @patch("psycopg2.connect")
    def test_coverage_threshold_partial_coverage_flag(self, mock_connect, mock_conn):
        conn, cursor = mock_conn
        mock_connect.return_value = conn
        # Exactly min_criteria_for_ranking criteria present -> not partial.
        rows = [_score_row(n, "legal", 3.0, "medium") for n in range(1, 13)]
        cursor.fetchall.return_value = rows

        agg = _make_aggregator(min_criteria_for_ranking=12)
        result = agg.compute_country_score(country_id=1, reference_year=2024)

        assert result.criteria_count == 12
        assert result.partial_coverage is False

        # One fewer -> partial.
        cursor.fetchall.return_value = rows[:-1]
        result2 = agg.compute_country_score(country_id=1, reference_year=2024)
        assert result2.criteria_count == 11
        assert result2.partial_coverage is True
