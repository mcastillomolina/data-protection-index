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

CRITERION_CORE_QUESTIONS: dict[int, str] = {
    1:  "Does the constitution explicitly protect privacy, and is there jurisprudence enforcing it?",
    2:  "What comprehensive data protection laws exist and what rights do they grant?",
    3:  "Does the DPA have real independence and does it actually impose sanctions?",
    4:  "Are national ID cards and biometric databases implemented, and how?",
    5:  "Are there laws against secondary data use, and do government agencies share data in practice?",
    6:  "How widespread is CCTV deployment and is it regulated effectively?",
    7:  "What authorisation is required for communications interception — judicial or ministerial?",
    8:  "Are there enforceable protections for employees against workplace surveillance?",
    9:  "What legal process do law enforcement agencies need to access personal data?",
    10: "Is there a law mandating retention of telecommunications data, and for how long?",
    11: "Are medical, financial, and movement data protected from surveillance in practice?",
    12: "Are biometrics collected at borders and is passenger data shared internationally?",
    13: "Has the country signed international treaties that expand or restrict surveillance?",
    14: "Do courts and parliament effectively limit executive surveillance overreach?",
}

TRUSTED_DOMAINS_BY_CRITERION: dict[int, list[str]] = {
    1:  ["constituteproject.org", "hudoc.echr.coe.int", "venice.coe.int"],
    2:  ["eur-lex.europa.eu", "legislation.gov.uk", "boe.es"],
    3:  ["gdprhub.eu", "edpb.europa.eu", "ico.org.uk", "cnil.fr", "aepd.es"],
    4:  [],
    5:  [],
    6:  ["privacyinternational.org", "eff.org"],
    7:  ["privacyinternational.org", "accessnow.org", "eff.org"],
    8:  [],
    9:  [],
    10: ["eur-lex.europa.eu"],
    11: ["privacyinternational.org"],
    12: ["privacyinternational.org", "statewatch.org"],
    13: ["treaty.un.org", "coe.int"],
    14: ["freedomhouse.org", "v-dem.net", "rsf.org"],
}
