"""Country resolution: DB lookup → YAML seed → pycountry → LLM enrichment."""

from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import yaml
from loguru import logger

from src.models.country import Country

if TYPE_CHECKING:
    from src.db.writer import DatabaseWriter

COUNTRIES_YAML_PATH = Path("config/countries.yaml")

ENRICHMENT_SCHEMA = {
    "type": "object",
    "required": ["iso_code", "region", "language", "dpa_name", "primary_law", "search_keywords"],
    "properties": {
        "iso_code": {
            "type": "string",
            "minLength": 2,
            "maxLength": 2,
            "description": "ISO 3166-1 alpha-2 country code"
        },
        "region": {
            "type": "string",
            "description": "Geographic region (e.g. Europe, Latin America, Asia)"
        },
        "language": {
            "type": "string",
            "description": "Primary official language as ISO 639-1 code"
        },
        "dpa_name": {
            "type": "string",
            "description": "Full name of the national data protection authority"
        },
        "primary_law": {
            "type": "string",
            "description": "Full name of the primary data protection law"
        },
        "search_keywords": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 6,
            "description": "Seed search keywords relevant to data protection for this country"
        }
    }
}

_SYSTEM_PROMPT = (
    "You are a legal and regulatory expert specializing in data protection frameworks "
    "worldwide. Return only valid JSON matching the requested schema — no prose, no markdown."
)


def resolve_country(
    country_name: str,
    config,
    db_writer: Optional["DatabaseWriter"] = None,
) -> Country:
    """
    Return a Country for country_name using a four-step lookup.

    1. DB name/alias lookup (when db_writer is provided).
    2. YAML/in-memory seed lookup; opportunistically upserts to DB.
    3. pycountry fuzzy lookup → ISO dedup against DB or YAML.
    4. LLM enrichment as last resort; persists to DB (or YAML when --skip-db).
    """
    # Step 1: DB lookup by name (ILIKE) or alias
    if db_writer is not None:
        db_meta = db_writer.find_country(country_name)
        if db_meta:
            logger.debug(f"Country '{country_name}' found in DB")
            return _build_country(db_meta)

    # Step 2: YAML / in-memory seed lookup
    yaml_meta = _find_country(country_name, config._countries_data)
    if yaml_meta:
        logger.debug(f"Country '{country_name}' found in YAML seed")
        country = _build_country(yaml_meta)
        if db_writer is not None:
            db_writer.upsert_country(country)
            logger.debug(f"Seeded '{country_name}' to DB from YAML")
        return country

    # Step 3: pycountry fuzzy lookup → ISO code → DB or YAML dedup
    iso = _resolve_iso_via_pycountry(country_name)
    if iso:
        if db_writer is not None:
            db_meta = db_writer.find_country_by_iso(iso)
            if db_meta:
                logger.info(
                    f"'{country_name}' resolved to '{db_meta['name']}' "
                    f"via ISO {iso} — adding alias"
                )
                db_writer.add_alias(iso, country_name)
                return _build_country(db_meta)
        yaml_by_iso = _find_by_iso_code(iso, config._countries_data)
        if yaml_by_iso:
            logger.info(
                f"'{country_name}' resolved to '{yaml_by_iso['name']}' "
                f"via ISO {iso} in YAML"
            )
            return _build_country(yaml_by_iso)

    # Step 4: LLM enrichment
    logger.info(f"Country '{country_name}' not found anywhere — enriching via LLM")
    metadata = _enrich_via_llm(country_name, config)

    if db_writer is not None:
        existing_db = db_writer.find_country_by_iso(metadata["iso_code"])
        if existing_db:
            logger.info(
                f"'{country_name}' resolved to existing DB entry '{existing_db['name']}' "
                f"via ISO {metadata['iso_code']} — adding alias"
            )
            db_writer.add_alias(metadata["iso_code"], country_name)
            return _build_country(existing_db)
        country = _build_country(metadata)
        db_writer.upsert_country(country)
        logger.info(f"Upserted new country '{country_name}' to DB")
        return country

    # --skip-db path: YAML ISO dedup + YAML cache (original behaviour)
    existing_yaml = _find_by_iso_code(metadata["iso_code"], config._countries_data)
    if existing_yaml:
        logger.info(
            f"'{country_name}' resolved to existing entry '{existing_yaml['name']}' "
            f"via ISO code {metadata['iso_code']} — skipping cache"
        )
        return _build_country(existing_yaml)

    logger.info(f"Caching new country entry for '{country_name}'")
    _cache_to_yaml(metadata)
    config._countries_data[metadata["name"]] = metadata
    return _build_country(metadata)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_country(country_name: str, countries_data: Dict[str, Any]) -> Optional[Dict]:
    """Case-insensitive name lookup in the loaded countries dict."""
    lower = country_name.lower()
    for name, metadata in countries_data.items():
        if name.lower() == lower:
            return metadata
    return None


def _find_by_iso_code(iso_code: str, countries_data: Dict[str, Any]) -> Optional[Dict]:
    """Look up a country by ISO 3166-1 alpha-2 code."""
    upper = iso_code.upper()
    for metadata in countries_data.values():
        if metadata.get("iso_code", "").upper() == upper:
            return metadata
    return None


def _resolve_iso_via_pycountry(country_name: str) -> Optional[str]:
    """Return ISO alpha-2 code via pycountry fuzzy search, or None on miss."""
    try:
        import pycountry
        matches = pycountry.countries.search_fuzzy(country_name)
        return matches[0].alpha_2
    except LookupError:
        return None


def _enrich_via_llm(country_name: str, config) -> Dict[str, Any]:
    """Ask the LLM for basic data-protection metadata about a country."""
    llm_client = config.get_llm_client()

    prompt = (
        f"Provide data protection regulatory information for {country_name}. "
        "Return JSON with these fields: iso_code (2-letter ISO 3166-1 alpha-2), "
        "region (geographic region), language (primary official language as ISO 639-1), "
        "dpa_name (national data protection authority name), "
        "primary_law (primary data protection law name), "
        "search_keywords (4-5 seed search terms relevant to data protection in this country)."
    )

    response = llm_client.complete_json(
        prompt=prompt,
        system_prompt=_SYSTEM_PROMPT,
        schema=ENRICHMENT_SCHEMA,
        temperature=0.1,
        max_tokens=512,
    )

    iso_code = response.get("iso_code", "XX").upper().strip()[:2]

    return {
        "name": country_name,
        "iso_code": iso_code,
        "official_languages": [response.get("language", "en")],
        "government_domains": [],
        "region": response.get("region", ""),
        "known_documents": {
            "dpa": response.get("dpa_name", ""),
            "data_protection_law": response.get("primary_law", ""),
        },
        "search_hints": response.get("search_keywords", []),
        "aliases": [],
    }


def _build_country(metadata: Dict[str, Any]) -> Country:
    """Construct a Country model from a metadata dict."""
    return Country(
        name=metadata["name"],
        iso_code=metadata["iso_code"],
        official_languages=metadata.get("official_languages", ["en"]),
        government_domains=metadata.get("government_domains", []),
        region=metadata.get("region", ""),
        known_documents=metadata.get("known_documents", {}),
        search_hints=metadata.get("search_hints", []),
        aliases=metadata.get("aliases", []),
        metadata=metadata,
    )


def _cache_to_yaml(metadata: Dict[str, Any]) -> None:
    """Append a new country entry to countries.yaml (only used in --skip-db mode)."""
    entry_yaml = yaml.dump(
        [metadata],
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    indented = "\n".join("  " + line for line in entry_yaml.rstrip("\n").split("\n"))

    with open(COUNTRIES_YAML_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + indented + "\n")
