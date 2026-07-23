"""Two-gate pre-filter for document sections before LLM extraction."""

import re

from src.core.section_splitter import DocumentSection

# ---------------------------------------------------------------------------
# Gate 1 — Structural noise patterns
# ---------------------------------------------------------------------------

_STRUCTURAL_NOISE_PATTERNS: list[str] = [
    r'\.{4,}',                                                              # dot-leaders
    r'^\s*[\d\s\.\-–—]+\s*$',                                              # digits/punctuation only
    r'archived?\s+(content|notice|page)',                                   # archival banners
    r'this\s+(page|content|document)\s+(has\s+been|is)\s+archived',
    r'archiv[eé][e]?\s*[-–—]',
    r'©\s*\d{4}',                                                           # copyright lines
    r'all\s+rights\s+reserved',
    r'^(table\s+of\s+contents|contents|index|índice|sommaire|inhaltsverzeichnis)\s*$',
    r'(signed|signature)\s+by',
    r'in\s+witness\s+whereof',
    r'page\s+\d+\s+of\s+\d+',
    r'^\s*\d+\s*$',                                                         # lone page number
    r'^[\s\-–—_=\*]{5,}$',                                                 # visual separators
]

_COMPILED_NOISE: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _STRUCTURAL_NOISE_PATTERNS
]

_SHORT_LINE_MIN_LINES: int = 4
_SHORT_LINE_MAX_CHARS: int = 60
_SHORT_LINE_RATIO: float = 0.65

# ---------------------------------------------------------------------------
# Gate 2 — Privacy/legal signal terms
#
# TERMS_BOUNDARY: languages where \b word boundaries work reliably
#   (Latin-script + Cyrillic: English, Spanish, French, German, Portuguese, Russian)
# TERMS_SUBSTRING: CJK and Arabic, where \b doesn't work reliably —
#   checked with direct string containment instead.
# ---------------------------------------------------------------------------

TERMS_BOUNDARY: list[str] = [
    # ── English ───────────────────────────────────────────────
    "personal data", "data protection", "privacy", "surveillance",
    "enforcement", "sanction", "fine", "penalty", "consent",
    "data subject", "controller", "processor", "retention",
    "interception", "biometric", "warrant", "judicial",
    "supervisory authority", "data breach", "transfer",
    "encryption", "anonymi", "pseudonymi", "lawful basis",

    # ── Spanish ───────────────────────────────────────────────
    "datos personales", "protección de datos", "privacidad",
    "vigilancia", "sanción", "multa", "consentimiento",
    "titular", "responsable", "encargado", "retención",
    "interceptación", "biométrico", "autorización judicial",
    "autoridad de control", "brecha de datos", "transferencia",
    "cifrado", "anonimización", "seudonimización",

    # ── French ────────────────────────────────────────────────
    "données personnelles", "protection des données", "vie privée",
    "surveillance", "sanction", "amende", "consentement",
    "personne concernée", "responsable du traitement", "sous-traitant",
    "conservation", "interception", "biométrique", "mandat judiciaire",
    "autorité de contrôle", "violation de données", "transfert",
    "chiffrement", "anonymisation", "pseudonymisation",

    # ── German ────────────────────────────────────────────────
    "personenbezogene daten", "datenschutz", "privatsphäre",
    "überwachung", "sanktion", "bußgeld", "einwilligung",
    "betroffene person", "verantwortlicher", "auftragsverarbeiter",
    "speicherung", "abhören", "biometrisch", "richtervorbehalt",
    "aufsichtsbehörde", "datenpanne", "übermittlung",
    "verschlüsselung", "anonymisierung", "pseudonymisierung",

    # ── Portuguese ────────────────────────────────────────────
    "dados pessoais", "proteção de dados", "privacidade",
    "vigilância", "sanção", "multa", "consentimento",
    "titular dos dados", "controlador", "operador", "retenção",
    "interceptação", "biométrico", "autorização judicial",
    "autoridade supervisora", "violação de dados", "transferência",
    "criptografia", "anonimização", "pseudonimização",

    # ── Russian ───────────────────────────────────────────────
    "персональные данные", "защита данных", "конфиденциальность",
    "слежка", "санкция", "штраф", "согласие",
    "субъект данных", "оператор", "хранение", "перехват",
    "биометрический", "судебный ордер", "нарушение данных",

    # ── Vietnamese ────────────────────────────────────────────
    "dữ liệu cá nhân", "bảo vệ dữ liệu", "quyền riêng tư",
    "giám sát", "xử phạt", "phạt tiền", "đồng ý",
    "chủ thể dữ liệu", "bên kiểm soát", "xử lý dữ liệu",
    "lưu trữ dữ liệu", "chặn thu", "sinh trắc học",
    "cơ quan giám sát", "vi phạm dữ liệu", "chuyển dữ liệu",
    "mã hóa", "ẩn danh hóa", "an ninh mạng", "thông tin cá nhân",
]

TERMS_SUBSTRING: list[str] = [
    # ── Chinese (Simplified) ──────────────────────────────────
    "个人数据", "数据保护", "隐私", "监控", "制裁",
    "罚款", "同意", "数据主体", "数据控制者", "保留",
    "拦截", "生物特征", "司法令状", "数据泄露", "传输",

    # ── Chinese (Traditional — Taiwan/HK) ─────────────────────
    "個人資料", "資料保護", "隱私", "監控", "制裁",
    "罰款", "同意", "資料主體", "資料控制者", "保存",
    "攔截", "生物特徵", "司法令狀", "資料外洩", "傳輸",
    "個人資料保護", "資訊安全", "監察",

    # ── Arabic ────────────────────────────────────────────────
    "البيانات الشخصية", "حماية البيانات", "الخصوصية",
    "المراقبة", "العقوبة", "الغرامة", "الموافقة",
    "سلطة الإشراف", "انتهاك البيانات",

    # ── Thai ──────────────────────────────────────────────────
    "ข้อมูลส่วนบุคคล", "การคุ้มครองข้อมูล", "ความเป็นส่วนตัว",
    "การเฝ้าระวัง", "บทลงโทษ", "ค่าปรับ", "ความยินยอม",
    "เจ้าของข้อมูล", "ผู้ควบคุมข้อมูล", "ผู้ประมวลผลข้อมูล",
    "การเก็บรักษา", "การดักฟัง", "ข้อมูลชีวภาพ",
    "หน่วยงานกำกับดูแล", "การละเมิดข้อมูล", "การโอนข้อมูล",
    "การเข้ารหัส", "การทำให้ไม่ระบุตัวตน", "ความมั่นคงปลอดภัย",

    # ── Korean ────────────────────────────────────────────────
    "개인정보", "개인정보 보호", "사생활", "프라이버시",
    "감시", "제재", "과태료", "벌금", "동의",
    "정보주체", "개인정보처리자", "수탁자", "보유기간",
    "감청", "생체인식", "법원 명령", "감독기관",
    "개인정보 침해", "개인정보 이전", "암호화", "익명처리",
    "개인정보보호위원회", "정보통신망",
]

# "anonymi" / "pseudonymi" are English partial stems: match the prefix only
# (covers anonymise/anonymize/anonymisation/anonymization and their equivalents)
_PARTIAL_STEMS: frozenset[str] = frozenset({"anonymi", "pseudonymi"})


def _build_boundary_patterns() -> list[re.Pattern]:
    patterns = []
    for term in TERMS_BOUNDARY:
        if term in _PARTIAL_STEMS:
            patterns.append(re.compile(r"\b" + re.escape(term), re.IGNORECASE))
        else:
            patterns.append(
                re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            )
    return patterns


_COMPILED_BOUNDARY: list[re.Pattern] = _build_boundary_patterns()

# ---------------------------------------------------------------------------
# Public gate functions
# ---------------------------------------------------------------------------


def is_structural_noise(text: str) -> bool:
    """Return True if the section is structural noise (ToC, archival banner, separator, etc.)."""
    if not text:
        return True
    for pattern in _COMPILED_NOISE:
        if pattern.search(text):
            return True
    lines = text.splitlines()
    if len(lines) >= _SHORT_LINE_MIN_LINES:
        short = sum(1 for line in lines if len(line) < _SHORT_LINE_MAX_CHARS)
        if short / len(lines) > _SHORT_LINE_RATIO:
            return True
    return False


def has_signal_terms(text: str) -> bool:
    """Return True if the section contains at least one substantive privacy/legal signal term.

    Boundary languages (Latin/Cyrillic): word-boundary regex.
    CJK/Arabic: direct string containment.
    """
    if any(p.search(text) for p in _COMPILED_BOUNDARY):
        return True
    return any(term in text for term in TERMS_SUBSTRING)


# ---------------------------------------------------------------------------
# Filter class
# ---------------------------------------------------------------------------


class SectionPreFilter:
    """
    Two-gate pre-filter for document sections before LLM extraction.

    Gate 1 (is_structural_noise): blocks archival banners, ToC pages,
    dot-leaders, visual separators, and short-line-ratio pages.
    Gate 2 (has_signal_terms): blocks sections with no substantive
    privacy/legal vocabulary. Uses word-boundary regex for Latin/Cyrillic
    languages and string containment for CJK/Arabic.
    Both gates must pass for a section to proceed to the LLM extractor.

    Gate 2 can be disabled via disable_gate2=True for languages not yet
    covered by the vocabulary list (e.g. during demos with unexpected countries).
    """

    def __init__(self, disable_gate2: bool = False) -> None:
        self._disable_gate2 = disable_gate2

    def passes(self, text: str) -> bool:
        """
        Returns True if the section should proceed to LLM extraction.
        Filter 1: not structural noise (regex + short-line ratio)
        Filter 2: contains at least one substantive privacy/legal signal term
        Both gates must pass (Gate 2 skipped if disable_gate2=True).
        """
        if is_structural_noise(text):
            return False
        if not self._disable_gate2 and not has_signal_terms(text):
            return False
        return True

    def filter(
        self, sections: list[DocumentSection]
    ) -> tuple[list[DocumentSection], list[DocumentSection]]:
        """
        Partition *sections* into (passing, blocked).

        Blocked sections are not sent to the LLM but are still recorded
        with error_message='pre-filter:no-signal'.
        """
        passing: list[DocumentSection] = []
        blocked: list[DocumentSection] = []
        for section in sections:
            (passing if self.passes(section.text) else blocked).append(section)
        return passing, blocked
