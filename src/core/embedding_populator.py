"""Populate section_extractions.embedding for all non-null sections post-Phase 3."""

from typing import List, Tuple

import psycopg2  # type: ignore[import]
import psycopg2.extras  # type: ignore[import]
from loguru import logger

from src.clients.embedding_client import EmbeddingClient


class EmbeddingPopulator:
    """
    Fetches sections where all_null=false AND embedding IS NULL, embeds them in
    batches using the provided EmbeddingClient, and writes vectors back to DB.

    Idempotent: already-embedded sections are skipped automatically.
    Embeds section_text_original (source language) — not the English extraction.
    """

    def __init__(
        self,
        dsn: str,
        embedding_client: EmbeddingClient,
        batch_size: int = 100,
    ) -> None:
        self._dsn = dsn
        self.embedding_client = embedding_client
        self.batch_size = batch_size

    def populate(self, country_id: int) -> int:
        """
        Embed all pending sections for a country.

        Returns:
            Number of sections embedded in this run.
        """
        conn = psycopg2.connect(self._dsn)
        conn.autocommit = False
        try:
            rows = self._fetch_pending(conn, country_id)
            if not rows:
                logger.info(f"No pending sections to embed for country_id={country_id}")
                return 0

            logger.info(
                f"Embedding {len(rows)} sections for country_id={country_id} "
                f"using {self.embedding_client.model}"
            )
            total_embedded = 0

            for batch_start in range(0, len(rows), self.batch_size):
                batch = rows[batch_start : batch_start + self.batch_size]
                ids = [r[0] for r in batch]
                texts = [r[1] for r in batch]

                vectors = self.embedding_client.embed(texts)

                self._write_batch(conn, ids, vectors, self.embedding_client.model)
                conn.commit()
                total_embedded += len(batch)
                logger.debug(
                    f"Embedded batch {batch_start // self.batch_size + 1}: "
                    f"{len(batch)} sections (total so far: {total_embedded})"
                )

            usage = self.embedding_client.get_total_usage()
            logger.info(
                f"Embedding complete: {total_embedded} sections embedded, "
                f"{usage['total_tokens']} tokens, ${usage['estimated_cost_usd']:.4f}"
            )
            return total_embedded

        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_pending(
        self, conn: psycopg2.extensions.connection, country_id: int
    ) -> List[Tuple[int, str]]:
        """Return (id, section_text_original) for un-embedded non-null sections."""
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT se.id, se.section_text_original
                FROM section_extractions se
                JOIN documents d ON d.id = se.document_id
                WHERE d.country_id = %s
                  AND se.all_null  = false
                  AND se.embedding IS NULL
                ORDER BY se.id
                """,
                (country_id,),
            )
            return cur.fetchall()

    def _write_batch(
        self,
        conn: psycopg2.extensions.connection,
        ids: List[int],
        vectors: List[List[float]],
        model: str,
    ) -> None:
        """UPDATE section_extractions for each (id, vector) pair."""
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                UPDATE section_extractions
                   SET embedding       = %s::vector,
                       embedding_model = %s
                 WHERE id = %s
                """,
                [(str(vec), model, sid) for sid, vec in zip(ids, vectors)],
                page_size=self.batch_size,
            )
