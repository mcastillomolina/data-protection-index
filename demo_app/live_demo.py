import os
import sys
import subprocess
from pathlib import Path

import psycopg2
import psycopg2.extras
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent

DB_HOST = os.environ.get("DPI_DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DPI_DB_PORT", "5433"))
DB_NAME = os.environ.get("DPI_DB_NAME", "dpi")
DB_USER = os.environ.get("DPI_DB_USER", "dpi")
DB_PASS = os.environ.get("DPI_DB_PASS", "dpi")

DEMO_CRITERIA = {
    1: "Constitutional Protection",
    2: "Statutory Protection",
    3: "Privacy Enforcement",
}

st.set_page_config(
    page_title="Data Protection Index — Demo",
    page_icon="🛡️",
    layout="wide",
)


def db_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )


def fetch_scores(country_name: str):
    try:
        conn = db_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT cs.criterion_number, cs.criterion_name, cs.criterion_score,
                       cs.legal_subscore, cs.enforcement_subscore,
                       cs.confidence, cs.rationale, cs.evidence_gaps
                FROM criterion_scores cs
                JOIN countries c ON c.id = cs.country_id
                WHERE c.name ILIKE %s AND cs.criterion_number = ANY(%s)
                ORDER BY cs.criterion_number
                """,
                (country_name, list(DEMO_CRITERIA.keys())),
            )
            rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def run_pipeline(country: str):
    """Generator: yields stdout+stderr lines from the demo pipeline."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.main", country, "--demo", "--no-save"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ},
        cwd=str(PROJECT_ROOT),
    )
    for line in proc.stdout:
        yield line
    proc.wait()
    if proc.returncode != 0:
        yield f"\n[exit code {proc.returncode}]\n"


def _score_color(score):
    if score is None:
        return "#555"
    if score >= 3.5:
        return "#2ecc71"
    if score >= 2.0:
        return "#e67e22"
    return "#e74c3c"


def show_results(scores, country_name: str):
    if not scores:
        st.warning("No scores found in the database for this country.")
        return

    st.markdown(f"### Results — **{country_name}**")

    cols = st.columns(len(scores))
    for col, row in zip(cols, scores):
        score = row["criterion_score"]
        color = _score_color(score)
        score_str = f"{score:.1f}" if score is not None else "—"
        legal = row.get("legal_subscore")
        enf = row.get("enforcement_subscore")
        with col:
            st.markdown(
                f"""
                <div style="background:#1a2535;border-left:4px solid {color};
                    padding:16px 18px;border-radius:6px;height:100%">
                  <div style="font-size:0.75rem;color:#888;text-transform:uppercase;
                      letter-spacing:0.08em;margin-bottom:4px">
                    Criterion {row['criterion_number']}</div>
                  <div style="font-size:1rem;font-weight:600;color:#e0e0e0;margin-bottom:14px">
                    {row['criterion_name']}</div>
                  <div style="font-size:2.4rem;font-weight:800;color:{color};line-height:1">
                    {score_str}
                    <span style="font-size:1rem;color:#888">/5</span>
                  </div>
                  <div style="margin-top:10px;font-size:0.8rem;color:#aaa">
                    Legal: {f"{legal:.1f}" if legal else "—"} &nbsp;·&nbsp;
                    Enforcement: {f"{enf:.1f}" if enf else "—"}
                  </div>
                  <div style="margin-top:4px;font-size:0.78rem;color:#888">
                    confidence: {row.get('confidence') or '—'}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Rationale & evidence gaps", expanded=False):
        for row in scores:
            st.markdown(f"**{row['criterion_number']}. {row['criterion_name']}**")
            rationale = row.get("rationale") or "_No rationale available._"
            st.markdown(rationale)
            gaps = row.get("evidence_gaps") or ""
            if gaps and gaps.strip().lower() not in ("none", "none.", "n/a", "none significant"):
                st.warning(f"Evidence gaps: {gaps}")
            st.markdown("---")


# ── Page layout ──────────────────────────────────────────────────────────────

st.markdown(
    """
    <h1 style='margin-bottom:0'>🛡️ Data Protection Index</h1>
    <p style='color:#aaa;margin-top:4px;font-size:1.05rem'>
      Live pipeline demo — type any country to analyse its data protection landscape
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

st.markdown(
    "**Demo mode** runs 3 criteria: *Constitutional Protection*, "
    "*Statutory Protection*, and *Privacy Enforcement*. Takes roughly 3–5 minutes."
)

country_input = st.text_input(
    "Country name",
    placeholder="e.g. Colombia, Germany, Kenya, Japan, México",
    label_visibility="collapsed",
)

col_run, col_check, col_spacer = st.columns([2, 2, 8])
run_btn = col_run.button(
    "▶ Run Analysis", type="primary", disabled=not bool(country_input.strip())
)
check_btn = col_check.button(
    "🔍 Show cached results", disabled=not bool(country_input.strip())
)

st.markdown("---")

if check_btn and country_input.strip():
    with st.spinner("Fetching results from database..."):
        scores = fetch_scores(country_input.strip())
    if scores:
        show_results(scores, country_input.strip())
    else:
        st.info(
            f"No results yet for **{country_input.strip()}**. "
            "Click **Run Analysis** to start the pipeline."
        )

if run_btn and country_input.strip():
    country_clean = country_input.strip()

    st.info(
        f"Starting pipeline for **{country_clean}** — this takes 3–5 minutes. "
        "Leave this tab open."
    )

    log_placeholder = st.empty()
    log_lines: list[str] = []

    for line in run_pipeline(country_clean):
        log_lines.append(line)
        visible = "".join(log_lines[-70:])
        log_placeholder.code(visible, language="bash")

    st.success("✅ Pipeline complete!")

    with st.spinner("Loading results..."):
        scores = fetch_scores(country_clean)

    show_results(scores, country_clean)
