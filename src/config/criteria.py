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

CONFIDENCE_WEIGHTS: dict[str, float] = {
    "high":   1.0,
    "medium": 0.7,
    "low":    0.4,
}

# Embedded once per criterion; used as the query vector for Gate 2 cosine search.
CRITERION_QUERY_SENTENCES: dict[int, str] = {
    1:  "Constitutional protection of privacy rights and court jurisprudence on data protection",
    2:  "Data protection law rights of individuals obligations on data controllers",
    3:  "Data protection authority enforcement actions sanctions fines independence",
    4:  "National identity card biometric database fingerprint collection",
    5:  "Secondary use of personal data inter-agency data sharing government programs",
    6:  "CCTV surveillance cameras public spaces facial recognition regulation",
    7:  "Communications interception wiretapping warrant judicial authorisation",
    8:  "Workplace monitoring employee surveillance privacy protection",
    9:  "Law enforcement access to personal data warrant requirement",
    10: "Mandatory data retention telecommunications traffic data period",
    11: "Medical records financial data location tracking surveillance protection",
    12: "Border control biometrics passenger name record international data sharing",
    13: "International treaty data sharing surveillance Budapest Convention Prum",
    14: "Parliamentary oversight judicial review executive surveillance democratic safeguards",
}

CRITERION_RUBRICS: dict[int, str] = {

    1: """
5 — Constitution explicitly protects privacy; strong jurisprudence; court
    actively enforces against government intrusion.
4 — Constitutional protection exists; some jurisprudence; enforcement
    generally effective with occasional gaps.
3 — Protection implied or indirect; limited jurisprudence; inconsistent
    enforcement by constitutional courts.
2 — Weak or qualified constitutional protection; courts rarely enforce;
    significant exceptions erode the right.
1 — No constitutional privacy protection; or exists only on paper with
    no enforcement history.
""",

    2: """
5 — Comprehensive data protection law; broad sectoral laws; rights fully
    operational; law meets or exceeds international standards.
4 — Comprehensive law exists; most rights operational; minor gaps in
    sectoral coverage.
3 — Law exists but with significant exemptions; some sectoral laws;
    rights partially functional.
2 — Partial or fragmented laws; broad exemptions; rights difficult to
    exercise in practice.
1 — No comprehensive data protection law; only minimal or no sectoral
    protections.
""",

    3: """
5 — DPA fully independent; proactively investigates; regularly imposes
    meaningful sanctions; demonstrably deters violations.
4 — DPA mostly independent and active; has imposed notable sanctions;
    some gaps in proactive enforcement.
3 — DPA exists and active but powers limited; enforcement reactive,
    sporadic, or inconsistent.
2 — DPA weak, underfunded, or lacking independence; few meaningful
    enforcement actions.
1 — No functional DPA; or exists purely on paper with no real activity.
""",

    4: """
LEGAL sub-dimension:
5 — No mandatory national ID; biometrics strictly regulated by law with
    strong privacy safeguards; no central biometric database mandated.
3 — ID card exists; biometrics collected but regulated; some safeguards.
1 — Mandatory ID with extensive biometrics; no meaningful legal limits;
    central database without privacy protections.

ENFORCEMENT sub-dimension:
5 — Biometric systems minimally deployed; opt-out available; no documented
    misuse; regulatory oversight effective.
3 — Systems deployed broadly; limited oversight; some documented concerns.
1 — Extensive biometric surveillance operational; documented abuses;
    no effective oversight.
""",

    5: """
LEGAL sub-dimension:
5 — Strong prohibition on secondary use; data minimisation enforced;
    strict limits on inter-agency sharing.
3 — Some restrictions on secondary use; limited sharing framework.
1 — No restrictions; broad government data sharing permitted by law.

ENFORCEMENT sub-dimension:
5 — No active inter-agency sharing programs beyond legal mandate;
    violations prosecuted.
3 — Some programs exist; mixed compliance.
1 — Extensive data sharing programs; no accountability mechanism.
""",

    6: """
5 — Minimal CCTV in public spaces; strict regulation; independent
    oversight; no facial recognition deployment.
4 — CCTV present but regulated; oversight generally effective.
3 — Widespread CCTV; regulation exists but inconsistently enforced.
2 — Very widespread CCTV; weak regulation; facial recognition deployed.
1 — Pervasive surveillance infrastructure; no effective regulation;
    extensive facial recognition in public spaces.
""",

    7: """
LEGAL sub-dimension:
5 — Judicial warrant required; strict crime threshold; time limits;
    independent oversight body; service providers not required to
    build surveillance backdoors.
3 — Judicial authorisation with exceptions; some oversight.
1 — Ministerial or self-authorisation; no meaningful oversight; broad
    backdoor requirements.

ENFORCEMENT sub-dimension:
5 — Low interception volume; no documented illegal wiretapping; oversight
    demonstrably effective.
3 — Moderate volume; some documented abuses; oversight partially effective.
1 — High volume; systematic illegal interception documented; oversight
    ineffective or absent.
""",

    8: """
5 — Specific workplace privacy law; clear limits on monitoring; DPA
    guidelines enforced; employees can challenge violations.
4 — Protections in general law; guidelines issued; some enforcement.
3 — Limited protections; guidelines exist but rarely enforced.
2 — No specific protections; employer surveillance largely unregulated.
1 — No protections; surveillance of employees pervasive and legally
    unchallenged.
""",

    9: """
LEGAL sub-dimension:
5 — Judicial warrant required for all access types; strict standards;
    notification to subjects after surveillance.
3 — Warrant required for most access; some administrative exceptions.
1 — No warrant requirement; broad administrative access powers.

ENFORCEMENT sub-dimension:
5 — Warrantless access not documented; legal challenges succeed;
    access volumes reported publicly.
3 — Some warrantless access documented; mixed accountability.
1 — Systematic warrantless access; no accountability; no public reporting.
""",

    10: """
5 — No mandatory retention law; or very short period (< 6 months) with
    strong judicial oversight and narrow access rules.
4 — Short retention period (6-12 months); access restricted to serious
    crimes; judicial oversight.
3 — 12-24 month retention; moderate access restrictions.
2 — Long retention (2+ years); broad access; limited oversight.
1 — Very long retention (5+ years); or blanket retention with no
    meaningful access controls.
""",

    11: """
5 — Medical, financial, and movement data strongly protected; sensitive
    data processed only with consent or strict legal basis; no mass
    surveillance programs documented.
4 — Strong protections; minor gaps; isolated incidents.
3 — Some protections; surveillance programs exist but with oversight.
2 — Weak protections; documented programs without adequate oversight.
1 — Extensive surveillance of sensitive data; no meaningful protection;
    mass programs operational.
""",

    12: """
5 — Biometrics not collected at borders beyond travel documents;
    no unnecessary passenger data sharing; proportionate border measures.
4 — Some biometric collection; data sharing with privacy safeguards.
3 — Biometric collection at borders; passenger data shared under formal
    agreements with some oversight.
2 — Extensive border biometrics; broad data sharing without adequate
    safeguards.
1 — Maximum biometric collection; blanket international data sharing;
    no independent oversight.
""",

    13: """
5 — Has not signed treaties that expand surveillance; actively promotes
    privacy-protective international standards; DPA acts as privacy
    ambassador internationally.
4 — Mostly positive international stance; minor treaty commitments
    that marginally expand surveillance.
3 — Mixed: some good treaty commitments, some expansive surveillance
    agreements.
2 — Has signed multiple treaties expanding surveillance (e.g. Prum,
    extensive bilateral sharing); limited push-back.
1 — Active promoter of surveillance-expanding international frameworks;
    signed all major expansive treaties; no privacy advocacy.
""",

    14: """
5 — Courts regularly rule against executive surveillance overreach;
    parliament exercises effective oversight; free press; strong
    protections for journalists and lawyers.
4 — Courts and parliament generally effective; occasional overreach
    corrected; press mostly free.
3 — Oversight exists but inconsistent; some overreach goes unchecked;
    press freedom concerns.
2 — Weak oversight; executive surveillance largely unchecked; press
    freedom significantly restricted.
1 — No effective judicial or parliamentary oversight; press controlled;
    systematic targeting of civil society and journalists.
""",

}
