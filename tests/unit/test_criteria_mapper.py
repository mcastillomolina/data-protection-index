"""Unit tests for criteria_mapper."""

import pytest

from src.core.criteria_mapper import get_criteria_ids


class TestCriteriaMapper:

    def test_constitution_maps_to_criterion_1(self):
        assert get_criteria_ids("constitution") == [1]

    def test_court_decision_maps_to_criteria_1_and_14(self):
        result = get_criteria_ids("court_decision")
        assert 1 in result
        assert 14 in result

    def test_data_protection_law_maps_to_multiple(self):
        result = get_criteria_ids("data_protection_law")
        for cid in [2, 5, 9, 10]:
            assert cid in result

    def test_enforcement_report_maps_to_criterion_3(self):
        assert 3 in get_criteria_ids("enforcement_report")

    def test_dpa_annual_report_maps_to_criterion_3(self):
        assert 3 in get_criteria_ids("dpa_annual_report")

    def test_surveillance_law_maps_to_criteria_6_7_11(self):
        result = get_criteria_ids("surveillance_law")
        for cid in [6, 7, 11]:
            assert cid in result

    def test_international_treaty_maps_to_criteria_5_and_13(self):
        result = get_criteria_ids("international_treaty")
        assert 5 in result
        assert 13 in result

    def test_border_surveillance_law_maps_to_criterion_12(self):
        assert 12 in get_criteria_ids("border_surveillance_law")

    def test_workplace_privacy_law_maps_to_criterion_8(self):
        assert 8 in get_criteria_ids("workplace_privacy_law")

    def test_unknown_type_returns_empty(self):
        assert get_criteria_ids("unknown_document_type") == []

    def test_returns_list_not_reference(self):
        """Each call returns an independent list copy."""
        r1 = get_criteria_ids("constitution")
        r2 = get_criteria_ids("constitution")
        r1.append(99)
        assert 99 not in r2

    def test_all_criteria_ids_are_1_to_14(self):
        """Every mapped criterion ID must be in the valid 1–14 range."""
        all_types = [
            "constitution", "court_decision", "data_protection_law", "regulation",
            "dpa_regulation", "enforcement_report", "dpa_annual_report", "biometrics_id_law",
            "international_treaty", "surveillance_law", "data_retention_law",
            "workplace_privacy_law", "government_access_law", "border_surveillance_law",
        ]
        for doc_type in all_types:
            for cid in get_criteria_ids(doc_type):
                assert 1 <= cid <= 14, f"{doc_type} produced out-of-range id {cid}"
