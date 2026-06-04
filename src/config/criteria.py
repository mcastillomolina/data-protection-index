"""Privacy International 14-criteria reference — used by CriterionExtractor and CriterionScorer."""

CRITERIA: dict[int, dict] = {
    1:  {"name": "Constitutional Protection",                    "dimension": "legal"},
    2:  {"name": "Statutory Protection",                         "dimension": "legal"},
    3:  {"name": "Privacy Enforcement",                          "dimension": "enforcement"},
    4:  {"name": "Identity Cards and Biometrics",                "dimension": "mixed"},
    5:  {"name": "Data Sharing",                                 "dimension": "mixed"},
    6:  {"name": "Visual Surveillance",                          "dimension": "enforcement"},
    7:  {"name": "Communication Interception",                   "dimension": "mixed"},
    8:  {"name": "Workplace Monitoring",                         "dimension": "enforcement"},
    9:  {"name": "Government Access to Data",                    "dimension": "mixed"},
    10: {"name": "Communications Data Retention",                "dimension": "legal"},
    11: {"name": "Surveillance of Medical/Financial/Movement",   "dimension": "enforcement"},
    12: {"name": "Border and Trans-border Issues",               "dimension": "enforcement"},
    13: {"name": "Leadership",                                   "dimension": "legal"},
    14: {"name": "Democratic Safeguards",                        "dimension": "enforcement"},
}
