"""Unit tests for EmbeddingPopulator — mocks psycopg2 and EmbeddingClient, no network calls."""

from unittest.mock import MagicMock, patch

import pytest

from src.core.embedding_populator import EmbeddingPopulator, _MAX_EMBED_CHARS


def _make_embedding_client(vector_dims: int = 4):
    client = MagicMock()
    client.model = "nomic-embed-text"

    def _embed(texts):
        return [[0.1] * vector_dims for _ in texts]

    client.embed.side_effect = _embed
    client.get_total_usage.return_value = {
        "model": "nomic-embed-text",
        "total_tokens": 100,
        "estimated_cost_usd": 0.0,
    }
    return client


@pytest.fixture
def mock_conn():
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    # execute_batch() mogrifies each param set to bytes before joining them —
    # a bare MagicMock cursor returns a MagicMock, which b";".join() rejects.
    cursor.mogrify.return_value = b"UPDATE ..."

    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


class TestPopulateNoPendingRows:
    @patch("psycopg2.connect")
    def test_returns_zero_and_never_calls_embed(self, mock_connect, mock_conn):
        conn, cursor = mock_conn
        mock_connect.return_value = conn
        cursor.fetchall.return_value = []

        embedding_client = _make_embedding_client()
        populator = EmbeddingPopulator("postgresql://fake", embedding_client)

        result = populator.populate(country_id=1)

        assert result == 0
        embedding_client.embed.assert_not_called()


class TestPopulateBatching:
    @patch("psycopg2.connect")
    def test_batches_respect_batch_size(self, mock_connect, mock_conn):
        conn, cursor = mock_conn
        mock_connect.return_value = conn
        rows = [(i, f"section text {i}") for i in range(5)]
        cursor.fetchall.return_value = rows

        embedding_client = _make_embedding_client()
        populator = EmbeddingPopulator("postgresql://fake", embedding_client, batch_size=2)

        total = populator.populate(country_id=1)

        assert total == 5
        # 5 rows / batch_size=2 -> 3 embed() calls (2, 2, 1)
        assert embedding_client.embed.call_count == 3
        call_sizes = [len(c.args[0]) for c in embedding_client.embed.call_args_list]
        assert call_sizes == [2, 2, 1]

    @patch("psycopg2.connect")
    def test_commits_once_per_batch(self, mock_connect, mock_conn):
        conn, cursor = mock_conn
        mock_connect.return_value = conn
        rows = [(i, f"text {i}") for i in range(4)]
        cursor.fetchall.return_value = rows

        embedding_client = _make_embedding_client()
        populator = EmbeddingPopulator("postgresql://fake", embedding_client, batch_size=2)
        populator.populate(country_id=1)

        assert conn.commit.call_count == 2  # 4 rows / batch_size 2 -> 2 batches

    @patch("psycopg2.connect")
    def test_single_batch_when_batch_size_exceeds_row_count(self, mock_connect, mock_conn):
        conn, cursor = mock_conn
        mock_connect.return_value = conn
        rows = [(1, "text one"), (2, "text two")]
        cursor.fetchall.return_value = rows

        embedding_client = _make_embedding_client()
        populator = EmbeddingPopulator("postgresql://fake", embedding_client, batch_size=100)
        total = populator.populate(country_id=1)

        assert total == 2
        embedding_client.embed.assert_called_once()


class TestPopulateTextTruncation:
    @patch("psycopg2.connect")
    def test_text_truncated_to_max_embed_chars(self, mock_connect, mock_conn):
        conn, cursor = mock_conn
        mock_connect.return_value = conn
        long_text = "x" * (_MAX_EMBED_CHARS + 500)
        cursor.fetchall.return_value = [(1, long_text)]

        embedding_client = _make_embedding_client()
        populator = EmbeddingPopulator("postgresql://fake", embedding_client)
        populator.populate(country_id=1)

        embedded_texts = embedding_client.embed.call_args.args[0]
        assert len(embedded_texts[0]) == _MAX_EMBED_CHARS

    @patch("psycopg2.connect")
    def test_short_text_not_truncated(self, mock_connect, mock_conn):
        conn, cursor = mock_conn
        mock_connect.return_value = conn
        short_text = "short section text"
        cursor.fetchall.return_value = [(1, short_text)]

        embedding_client = _make_embedding_client()
        populator = EmbeddingPopulator("postgresql://fake", embedding_client)
        populator.populate(country_id=1)

        embedded_texts = embedding_client.embed.call_args.args[0]
        assert embedded_texts[0] == short_text


class TestPopulateWriteBack:
    @patch("psycopg2.extras.execute_batch")
    @patch("psycopg2.connect")
    def test_writes_vector_and_model_for_each_row(self, mock_connect, mock_execute_batch, mock_conn):
        conn, cursor = mock_conn
        mock_connect.return_value = conn
        cursor.fetchall.return_value = [(10, "a"), (20, "b")]

        embedding_client = _make_embedding_client(vector_dims=3)
        populator = EmbeddingPopulator("postgresql://fake", embedding_client, batch_size=10)
        populator.populate(country_id=1)

        mock_execute_batch.assert_called_once()
        _, args, kwargs = mock_execute_batch.mock_calls[0]
        sql = args[1]
        params_list = args[2]
        assert "UPDATE section_extractions" in sql
        assert "embedding" in sql
        ids_written = [p[2] for p in params_list]
        assert ids_written == [10, 20]
        models_written = {p[1] for p in params_list}
        assert models_written == {"nomic-embed-text"}

    @patch("psycopg2.connect")
    def test_fetch_query_filters_null_and_unembedded(self, mock_connect, mock_conn):
        """_fetch_pending's SQL must restrict to all_null=false AND embedding IS NULL."""
        conn, cursor = mock_conn
        mock_connect.return_value = conn
        cursor.fetchall.return_value = []

        embedding_client = _make_embedding_client()
        populator = EmbeddingPopulator("postgresql://fake", embedding_client)
        populator.populate(country_id=7)

        sql, params = cursor.execute.call_args.args
        assert "all_null" in sql and "false" in sql
        assert "embedding IS NULL" in sql
        assert params == (7,)
