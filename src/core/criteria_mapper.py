"""Static mapping from document_type → PI criterion IDs (1–14)."""

_DOCUMENT_TYPE_CRITERIA: dict[str, list[int]] = {
    "constitution":            [1],
    "court_decision":          [1, 14],
    "data_protection_law":     [2, 5, 9, 10],
    "regulation":              [2],
    "dpa_regulation":          [2, 3],
    "enforcement_report":      [3],
    "dpa_annual_report":       [3],
    "biometrics_id_law":       [4],
    "international_treaty":    [5, 13],
    "surveillance_law":        [6, 7, 11],
    "data_retention_law":      [7, 10],
    "workplace_privacy_law":   [8],
    "government_access_law":   [9],
    "border_surveillance_law": [12],
}


def get_criteria_ids(document_type: str) -> list[int]:
    """Return the PI criterion IDs (1–14) covered by *document_type*."""
    return list(_DOCUMENT_TYPE_CRITERIA.get(document_type, []))
