import os
import psycopg2
import psycopg2.extras
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Data Protection Index",
    page_icon="🛡️",
    layout="wide",
)

DB_HOST = os.environ.get("DPI_DB_HOST", "localhost")
DB_PORT = os.environ.get("DPI_DB_PORT", "5433")
DB_NAME = os.environ.get("DPI_DB_NAME", "dpi")
DB_USER = os.environ.get("DPI_DB_USER", "dpi")
DB_PASS = os.environ.get("DPI_DB_PASS", "dpi")


def get_conn():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )


@st.cache_data(ttl=300)
def load_scored_countries():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT c.id, c.name, c.iso_code, c.region
                FROM countries c
                JOIN criterion_scores cs ON c.id = cs.country_id
                ORDER BY c.name
            """)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_criterion_scores(country_id: int):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    criterion_number,
                    criterion_name,
                    dimension,
                    legal_subscore,
                    enforcement_subscore,
                    criterion_score,
                    confidence,
                    evidence_count,
                    information_opacity,
                    rationale,
                    evidence_gaps
                FROM criterion_scores
                WHERE country_id = %s
                ORDER BY criterion_number
            """, (country_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_documents(country_id: int):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    official_name,
                    document_type,
                    source_url,
                    criteria_ids
                FROM documents
                WHERE country_id = %s
                  AND source_url IS NOT NULL
                  AND source_url != ''
                ORDER BY document_type, official_name
            """, (country_id,))
            return cur.fetchall()


def compute_aggregate(scores):
    """Compute a simple mean of criterion_score across all criteria."""
    vals = [r["criterion_score"] for r in scores if r["criterion_score"] is not None]
    return round(sum(vals) / len(vals) * 20, 1) if vals else None  # scale 0-5 → 0-100


def score_badge(score):
    if score is None:
        return "⬜ N/A", "#555"
    if score >= 70:
        return f"🟢 {score:.1f} / 100", "#2ecc71"
    if score >= 40:
        return f"🟡 {score:.1f} / 100", "#f39c12"
    return f"🔴 {score:.1f} / 100", "#e74c3c"


def pi_category(score):
    if score is None:
        return "Unknown"
    if score >= 80:
        return "Adequate protection"
    if score >= 60:
        return "Significant protections"
    if score >= 40:
        return "Systemic problems"
    if score >= 20:
        return "Extensive surveillance"
    return "Endemic surveillance"


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("🛡️ Data Protection Index")
st.sidebar.markdown("---")

countries = load_scored_countries()
if not countries:
    st.sidebar.error("No scored countries found in the database.")
    st.stop()

country_options = {r["name"]: r for r in countries}
selected_name = st.sidebar.selectbox(
    "Select country",
    options=list(country_options.keys()),
)
country = country_options[selected_name]

scores = load_criterion_scores(country["id"])
agg_score = compute_aggregate(scores)
badge_text, badge_color = score_badge(agg_score)
category = pi_category(agg_score)

st.sidebar.markdown("### Overall DPI Score")
st.sidebar.markdown(
    f"""
    <div style="
        background:{badge_color}22;
        border:2px solid {badge_color};
        border-radius:8px;
        padding:12px 16px;
        text-align:center;
    ">
        <span style="font-size:1.6rem;font-weight:700;color:{badge_color}">
            {badge_text}
        </span><br>
        <span style="font-size:0.85rem;color:#aaa">{category}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown(f"**Region:** {country.get('region') or '—'}")
st.sidebar.markdown(f"**ISO:** {country['iso_code'].strip()}")

# ── Main ─────────────────────────────────────────────────────────────────────

st.title(f"{selected_name} — Data Protection Profile")
st.markdown("---")

if not scores:
    st.warning("No criterion scores found for this country.")
    st.stop()

df = pd.DataFrame(scores)

# ── Section 1: Score Overview ─────────────────────────────────────────────────

st.subheader("Score Overview")


def score_color(v):
    if v is None:
        return "#555555"
    if v >= 4:
        return "#2ecc71"
    if v >= 3:
        return "#1abc9c"
    if v >= 2:
        return "#e67e22"
    return "#e74c3c"


CONF_COLOR = {"high": "#2ecc71", "medium": "#e67e22", "low": "#e74c3c"}

header_col, score_col = st.columns([5, 3])
header_col.markdown(
    f"<div style='background:#1c3a4a;padding:8px 12px;border-radius:4px 0 0 4px;"
    f"font-weight:600;color:#7ec8e3;font-size:0.9rem;letter-spacing:0.05em'>CRITERION</div>",
    unsafe_allow_html=True,
)
score_col.markdown(
    f"<div style='background:#1c3a4a;padding:8px 12px;border-radius:0 4px 4px 0;"
    f"font-weight:600;color:#fff;font-size:0.9rem;text-align:center'>{selected_name.upper()}</div>",
    unsafe_allow_html=True,
)

for i, row in enumerate(scores):
    bg = "#16202e" if i % 2 == 0 else "#1a2535"
    composite = row["criterion_score"]
    conf = (row.get("confidence") or "").lower()
    ccolor = score_color(composite)
    conf_color = CONF_COLOR.get(conf, "#aaa")
    score_str = f"{composite:.1f}" if composite is not None else "—"

    crit_col, val_col = st.columns([5, 3])
    crit_col.markdown(
        f"<div style='background:{bg};padding:8px 12px;font-size:0.92rem'>"
        f"{row['criterion_number']}. {row['criterion_name']}</div>",
        unsafe_allow_html=True,
    )
    val_col.markdown(
        f"<div style='background:{bg};padding:8px 12px;display:flex;align-items:center;gap:10px'>"
        f"<span style='background:{ccolor};color:#000;font-weight:700;font-size:0.95rem;"
        f"padding:2px 10px;border-radius:4px;min-width:38px;text-align:center'>{score_str}</span>"
        f"<span style='color:{conf_color};font-size:0.85rem'>{conf}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Section 2: Criterion Detail Cards ─────────────────────────────────────────

st.subheader("Criterion Details")

for row in scores:
    composite = row["criterion_score"]
    conf = (row.get("confidence") or "").lower()

    with st.expander(f"{row['criterion_number']}. {row['criterion_name']}", expanded=False):
        col_comp, col_legal, col_enf = st.columns(3)
        with col_comp:
            st.metric("Composite Score", f"{composite:.1f}/5" if composite is not None else "—")
        with col_legal:
            legal = row.get("legal_subscore")
            st.metric("Lex Scripta (Legal)", f"{legal:.1f}/5" if legal is not None else "—")
        with col_enf:
            enf = row.get("enforcement_subscore")
            st.metric("Lex Viva (Enforcement)", f"{enf:.1f}/5" if enf is not None else "—")

        dim = row.get("dimension", "")
        evcount = row.get("evidence_count")
        meta_parts = []
        if dim:
            meta_parts.append(f"Dimension: **{dim}**")
        if conf:
            meta_parts.append(f"Confidence: **{conf}**")
        if evcount is not None:
            meta_parts.append(f"Evidence docs: **{evcount}**")
        if row.get("information_opacity"):
            meta_parts.append("⚠️ **Information opacity flag**")
        if meta_parts:
            st.caption("  ·  ".join(meta_parts))

        rationale = row.get("rationale") or ""
        if rationale:
            st.markdown("**Rationale**")
            st.markdown(rationale)

        gaps = row.get("evidence_gaps") or ""
        if gaps and gaps.strip().lower() not in ("none", "none significant", "none.", "n/a"):
            st.warning(f"**Evidence gaps:** {gaps}")

st.markdown("---")

# ── Section 3: Key Documents ──────────────────────────────────────────────────

st.subheader("Key Documents")

docs = load_documents(country["id"])
if not docs:
    st.info("No source documents with URLs found for this country.")
else:
    # Build criteria_number → name lookup
    crit_map = {int(r["criterion_number"]): r["criterion_name"] for r in scores}

    rows = []
    for d in docs:
        crit_ids = d.get("criteria_ids") or []
        crit_names = ", ".join(
            crit_map.get(c, f"#{c}") for c in sorted(crit_ids) if c in crit_map
        ) or "—"
        url = d["source_url"]
        link = f"[{d['official_name']}]({url})"
        rows.append({
            "Document": link,
            "Type": d["document_type"].replace("_", " ").title(),
            "Criteria covered": crit_names,
        })

    hdr_doc, hdr_type, hdr_crit = st.columns([3, 2, 3])
    hdr_doc.markdown("**Document**")
    hdr_type.markdown("**Type**")
    hdr_crit.markdown("**Criteria covered**")
    st.markdown("<hr style='margin:4px 0 8px'>", unsafe_allow_html=True)

    for r in rows:
        col_doc, col_type, col_crit = st.columns([3, 2, 3])
        col_doc.markdown(r["Document"])
        col_type.markdown(r["Type"])
        col_crit.markdown(r["Criteria covered"])

