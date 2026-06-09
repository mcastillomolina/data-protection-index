"""Phase 4 Step E — ExternalSourceFetcher: ingests institutional index data."""

import json
import re
from pathlib import Path
from typing import Any

import httpx
import psycopg2
import psycopg2.extras
from loguru import logger

# RSF: year-specific CSV, semicolon-delimited, ISO3 country codes
_RSF_CSV_URL = "https://rsf.org/sites/default/files/import_classement/{year}.csv"

# Freedom House: per-country page scrape (no public bulk download exists)
_FH_COUNTRY_URL = "https://freedomhouse.org/country/{slug}/freedom-world/{year}"

# Enforcement tracker: replaces defunct GDPRhub tracker page
_ET_URL = "https://www.enforcementtracker.com/"


def _rsf_score_col(year: int) -> str:
    """RSF column name is 'Score {year}' (e.g. 'Score 2026')."""
    return f"Score {year}"


def _fh_slug(country_name: str) -> str:
    """Convert country name to Freedom House URL slug (lowercase, hyphens)."""
    slug = country_name.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    return slug


class ExternalSourceFetcher:
    """
    Fetches and normalises data from four institutional sources into external_indicators.

    All fetch_* methods are fault-tolerant: a single source failure logs a warning
    and returns 0 rather than raising.
    """

    def __init__(self, dsn: str, cache_dir: Path = Path("data/cache/external")) -> None:
        self._dsn = dsn
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all(
        self,
        country_iso: str,
        country_id: int,
        year: int,
        country_name: str = "",
    ) -> dict[str, int]:
        """Run all four fetchers. Returns {source_name: rows_written}. Never raises."""
        results: dict[str, int] = {}
        for name, fn in [
            ("freedom_house",      lambda: self.fetch_freedom_house(country_iso, country_id, year, country_name)),
            ("vdem",               lambda: self.fetch_vdem(country_iso, country_id, year)),
            ("rsf",                lambda: self.fetch_rsf(country_iso, country_id, year)),
            ("enforcement_tracker", lambda: self.fetch_enforcement_tracker(country_id, country_name)),
        ]:
            try:
                results[name] = fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"ExternalSourceFetcher: {name} failed — {exc}")
                results[name] = 0
        return results

    # ------------------------------------------------------------------
    # Freedom House — criterion 14, per-country page scrape
    # No public bulk CSV/XLSX download exists; scores are on HTML country pages.
    # ------------------------------------------------------------------

    def fetch_freedom_house(
        self,
        country_iso: str,
        country_id: int,
        year: int,
        country_name: str = "",
    ) -> int:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("Freedom House fetch requires beautifulsoup4 — skipping")
            return 0

        slug = _fh_slug(country_name or country_iso)
        url = _FH_COUNTRY_URL.format(slug=slug, year=year)

        try:
            with httpx.Client(timeout=20, follow_redirects=True) as client:
                resp = client.get(url)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                # Try previous year as fallback
                fallback_year = year - 1
                url = _FH_COUNTRY_URL.format(slug=slug, year=fallback_year)
                logger.info(
                    f"Freedom House {year} not found for '{slug}' — trying {fallback_year}"
                )
                try:
                    with httpx.Client(timeout=20, follow_redirects=True) as client:
                        resp = client.get(url)
                    resp.raise_for_status()
                    year = fallback_year
                except Exception as exc2:
                    logger.warning(f"Freedom House: could not fetch '{slug}': {exc2}")
                    return 0
            else:
                logger.warning(f"Freedom House HTTP error for '{slug}': {exc}")
                return 0

        soup = BeautifulSoup(resp.text, "lxml")

        # Aggregate score is in <span class="country-score-actual"> inside div.country-score
        score_el = soup.select_one("div.country-score > span.country-score-actual")
        if score_el is None:
            logger.warning(f"Freedom House: score element not found for '{slug}' ({year})")
            return 0

        try:
            raw_score = float(score_el.get_text(strip=True))
        except (ValueError, TypeError) as exc:
            logger.warning(f"Freedom House: could not parse score for '{slug}': {exc}")
            return 0

        # Status class on the parent scorecard container (status-free / status-partly-free / status-not-free)
        status_map = {"free": "Free", "partly-free": "Partly Free", "not-free": "Not Free"}
        status = ""
        scorecard = soup.find(class_=re.compile(r"\bstatus-(free|partly-free|not-free)\b"))
        if scorecard:
            for cls in scorecard.get("class", []):
                m = re.match(r"status-(free|partly-free|not-free)", cls)
                if m:
                    status = status_map.get(m.group(1), "")
                    break

        # FH score: 0–100 (higher = freer) → normalise to 1–5
        normalised = max(1.0, min(5.0, raw_score / 100 * 4 + 1))

        indicators: list[dict[str, Any]] = [{
            "country_id": country_id,
            "pi_criterion_number": 14,
            "dimension": "enforcement",
            "source_name": "freedom_house",
            "source_year": year,
            "indicator_name": "aggregate_score",
            "indicator_value": raw_score,
            "indicator_normalised": round(normalised, 4),
            "raw_data": json.dumps({"score": raw_score, "status": status, "slug": slug}),
            "notes": f"Freedom House Freedom in the World {year} — {slug}",
        }]

        if status:
            status_val = {"Free": 5.0, "Partly Free": 3.0, "Not Free": 1.0}.get(status, 3.0)
            status_norm = {"Free": 1.0, "Partly Free": 0.6, "Not Free": 0.2}.get(status, 0.6)
            indicators.append({
                "country_id": country_id,
                "pi_criterion_number": 14,
                "dimension": "enforcement",
                "source_name": "freedom_house",
                "source_year": year,
                "indicator_name": "status",
                "indicator_value": status_val,
                "indicator_normalised": status_norm,
                "raw_data": json.dumps({"status": status}),
                "notes": f"Freedom House status category {year}",
            })

        return self._write_indicators(indicators)

    # ------------------------------------------------------------------
    # V-Dem — no public API exists; api.v-dem.net does not resolve.
    # Bulk download requires registration at v-dem.net.
    # ------------------------------------------------------------------

    def fetch_vdem(self, country_iso: str, country_id: int, year: int) -> int:
        logger.info(
            "V-Dem skipped — no public API available. "
            "Bulk data requires registration at https://v-dem.net/data/"
        )
        return 0

    # ------------------------------------------------------------------
    # RSF Press Freedom Index — criterion 14
    # Real URL: rsf.org/sites/default/files/import_classement/{year}.csv
    # Delimiter: semicolon. ISO: ISO3 (e.g. CAN). Score col: 'Score {year}'.
    # Decimals: European format (78,76 → 78.76).
    # ------------------------------------------------------------------

    def fetch_rsf(self, country_iso: str, country_id: int, year: int) -> int:
        import csv, io

        url = _RSF_CSV_URL.format(year=year)
        cache_key = f"rsf_{year}.csv"
        raw = self._download(url, cache_key)

        text = raw.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(text), delimiter=";")
        score_col = _rsf_score_col(year)

        row: dict[str, str] | None = None
        for r in reader:
            iso_val = (r.get("ISO") or "").strip().upper()
            # RSF uses ISO3; accept both ISO2 and ISO3 input
            if iso_val == country_iso.upper() or iso_val[:2] == country_iso.upper()[:2]:
                row = r
                break

        if row is None:
            logger.warning(f"RSF {year}: no row found for iso={country_iso}")
            return 0

        raw_str = (row.get(score_col) or row.get("Score") or "").strip()
        if not raw_str:
            logger.warning(f"RSF: score column '{score_col}' missing for {country_iso}")
            return 0

        try:
            raw_score = float(raw_str.replace(",", "."))
        except (ValueError, TypeError):
            logger.warning(f"RSF: could not parse score '{raw_str}' for {country_iso}")
            return 0

        # RSF: 0 (worst) → 100 (best). Normalise to 1–5.
        normalised = max(1.0, min(5.0, raw_score / 100 * 4 + 1))

        rows = [{
            "country_id": country_id,
            "pi_criterion_number": 14,
            "dimension": "enforcement",
            "source_name": "rsf",
            "source_year": year,
            "indicator_name": "press_freedom_score",
            "indicator_value": raw_score,
            "indicator_normalised": round(normalised, 4),
            "raw_data": json.dumps({k: v for k, v in row.items() if k}),
            "notes": f"RSF Press Freedom Index {year}",
        }]
        return self._write_indicators(rows)

    # ------------------------------------------------------------------
    # Enforcement Tracker (enforcementtracker.com) — criterion 3
    # Replaces defunct GDPRhub enforcement tracker page.
    # Data is embedded as JSON in <script id="et-cases"> in the HTML.
    # Country field 'c' is uppercase full country name (e.g. 'CANADA').
    # ------------------------------------------------------------------

    def fetch_enforcement_tracker(self, country_id: int, country_name: str) -> int:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("Enforcement tracker fetch requires beautifulsoup4 — skipping")
            return 0

        # Use cache: enforcement tracker data is the full dataset, so cache it
        cache_key = "enforcement_tracker.json"
        cache_path = self._cache_dir / cache_key
        if cache_path.exists():
            cases = json.loads(cache_path.read_text())
            logger.debug("ExternalSourceFetcher: enforcement_tracker cache hit")
        else:
            try:
                with httpx.Client(timeout=30, follow_redirects=True) as client:
                    resp = client.get(_ET_URL)
                resp.raise_for_status()
            except Exception as exc:
                logger.warning(f"Enforcement tracker HTTP error: {exc}")
                return 0

            soup = BeautifulSoup(resp.text, "lxml")
            el = soup.find("script", id="et-cases")
            if el is None:
                logger.warning("Enforcement tracker: <script id='et-cases'> not found in page")
                return 0

            cases = json.loads(el.string)
            cache_path.write_text(json.dumps(cases))
            logger.debug(f"Enforcement tracker: cached {len(cases)} cases")

        # 'c' field is uppercase full country name (e.g. 'AUSTRIA', 'CANADA')
        target = country_name.upper()
        matched = [c for c in cases if (c.get("c") or "").upper() == target]

        if not matched:
            logger.info(
                f"Enforcement tracker: no decisions found for '{country_name}' "
                f"(country may not be subject to GDPR)"
            )
            return 0

        rows: list[dict[str, Any]] = []
        for case in matched:
            # Extract year from date field 'd' (e.g. '2018-12-09')
            src_year: int | None = case.get("y")
            if src_year is None:
                date_str = case.get("d") or ""
                try:
                    src_year = int(date_str[:4])
                except (ValueError, TypeError):
                    src_year = None

            rows.append({
                "country_id": country_id,
                "pi_criterion_number": 3,
                "dimension": "enforcement",
                "source_name": "enforcement_tracker",
                "source_year": src_year,
                "indicator_name": "enforcement_decision",
                "indicator_value": float(case.get("f") or 0) or 1.0,
                "indicator_normalised": 1.0,
                "raw_data": json.dumps({
                    "authority": case.get("a"),
                    "date": case.get("d"),
                    "fine_eur": case.get("f"),
                    "sector": case.get("s"),
                    "article": case.get("r"),
                    "type": case.get("t"),
                    "url": case.get("u"),
                }),
                "notes": f"GDPR enforcement decision — {case.get('a', '')} ({case.get('d', '')})",
            })

        if rows:
            return self._write_indicators(rows)
        return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_indicators(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        conn = psycopg2.connect(self._dsn)
        try:
            with conn.cursor() as cur:
                written = 0
                for row in rows:
                    cur.execute(
                        """
                        INSERT INTO external_indicators (
                            country_id, pi_criterion_number, dimension,
                            source_name, source_year, indicator_name,
                            indicator_value, indicator_normalised, raw_data, notes
                        ) VALUES (
                            %(country_id)s, %(pi_criterion_number)s, %(dimension)s,
                            %(source_name)s, %(source_year)s, %(indicator_name)s,
                            %(indicator_value)s, %(indicator_normalised)s,
                            %(raw_data)s::jsonb, %(notes)s
                        )
                        ON CONFLICT (country_id, pi_criterion_number, source_name,
                                     indicator_name, source_year)
                        DO UPDATE SET
                            indicator_value      = EXCLUDED.indicator_value,
                            indicator_normalised = EXCLUDED.indicator_normalised,
                            raw_data             = EXCLUDED.raw_data,
                            notes                = EXCLUDED.notes,
                            created_at           = NOW()
                        """,
                        row,
                    )
                    written += 1
            conn.commit()
        finally:
            conn.close()
        return written

    def _download(self, url: str, cache_key: str) -> bytes:
        """Download URL, cache to disk, return raw bytes."""
        cache_path = self._cache_dir / cache_key
        if cache_path.exists():
            logger.debug(f"ExternalSourceFetcher: cache hit — {cache_key}")
            return cache_path.read_bytes()
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(url)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        logger.debug(f"ExternalSourceFetcher: downloaded and cached {cache_key}")
        return resp.content
