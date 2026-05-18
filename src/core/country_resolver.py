"""Country resolution: config lookup with LLM enrichment fallback."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger

from src.models.country import Country

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


def resolve_country(country_name: str, config) -> Country:
    """
    Return a Country for country_name.

    Checks config/countries.yaml first (case-insensitive). If not found,
    calls the configured LLM to enrich the entry, caches it back to the
    YAML file, and returns the result.
    """
    metadata = _find_country(country_name, config._countries_data)
    if metadata:
        return _build_country(metadata)

    logger.info(f"Country '{country_name}' not found in config — enriching via LLM")
    metadata = _enrich_via_llm(country_name, config)

    # Before caching, check if a country with the same ISO code already exists
    # (handles alternate names / localised spellings, e.g. "España" vs "Spain")
    existing = _find_by_iso_code(metadata["iso_code"], config._countries_data)
    if existing:
        logger.info(
            f"'{country_name}' resolved to existing entry '{existing['name']}' "
            f"via ISO code {metadata['iso_code']} — skipping cache"
        )
        return _build_country(existing)

    logger.info(f"Caching new country entry for '{country_name}'")
    _cache_to_yaml(metadata)
    config._countries_data[metadata["name"]] = metadata
    return _build_country(metadata)


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
        metadata=metadata,
    )


def _cache_to_yaml(metadata: Dict[str, Any]) -> None:
    """Append a new country entry to countries.yaml."""
    entry_yaml = yaml.dump(
        [metadata],
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    # yaml.dump([item]) produces "- key: val\n  key2: val2\n..."
    # Indent by 2 spaces to nest under the top-level `countries:` list.
    indented = "\n".join("  " + line for line in entry_yaml.rstrip("\n").split("\n"))

    with open(COUNTRIES_YAML_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + indented + "\n")
