#!/usr/bin/env python
"""
Sensitivity analysis of the dual-framework legal/enforcement weighting.

Recomputes each country's final_score under a range of legal weights
(w_legal ∈ {1.0, 0.6, 0.5, 0.4, 0.3, 0.0}) and reports how ranking, position,
and PI category respond relative to the 40/60 baseline.

READ-ONLY: reads criterion_scores from the DB, writes CSV + markdown to
data/outputs/. Never writes to the database.

The weight is varied in BOTH places it can appear:
  1. Inside mixed criteria — the reweighted combined score
     (w_legal·legal_subscore + w_enforcement·enforcement_subscore), used only as
     a fallback when a subscore is missing.
  2. In the final aggregation — final = w_legal·legal_mean + w_enforcement·enf_mean.

Aggregation math mirrors src/core/index_aggregator.py exactly (same dimension
sets, same confidence weighting, same missing_strategy="exclude" behaviour of
simply omitting absent criteria).

Usage:
    python -m scripts.sensitivity_analysis [--year YEAR]
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

# Reuse the real aggregator's constants so this analysis can never drift from it.
from src.config.criteria import CONFIDENCE_WEIGHTS
from src.core.index_aggregator import (
    _ENFORCEMENT_DIMS,
    _LEGAL_DIMENSIONS,
    _MIXED_CRITERIA,
)

# w_legal scenarios; w_enforcement is always (1 - w_legal). 0.4 is the 40/60 baseline.
SCENARIOS = [1.0, 0.6, 0.5, 0.4, 0.3, 0.0]
BASELINE_W = 0.4

# PI category bands (upper-inclusive ranges as specified for the analysis).
PI_BANDS = [
    (4.1, 5.01, "4.1-5.0  Consistently upholds standards"),
    (3.6, 4.1,  "3.6-4.0  Significant protections"),
    (3.1, 3.6,  "3.1-3.5  Adequate safeguards"),
    (2.6, 3.1,  "2.6-3.0  Some safeguards (weakened)"),
    (2.1, 2.6,  "2.1-2.5  Systemic failure"),
    (1.6, 2.1,  "1.6-2.0  Extensive surveillance"),
    (1.0, 1.6,  "1.1-1.5  Endemic surveillance"),
]


def band_for(score: float) -> str:
    for lo, hi, label in PI_BANDS:
        if lo <= score < hi:
            return label
    return PI_BANDS[-1][2] if score < 1.6 else PI_BANDS[0][2]


# ---------------------------------------------------------------------------
# Aggregation (faithful port of IndexAggregator._extract_pairs / _weighted_mean)
# ---------------------------------------------------------------------------

def _extract_pairs(rows: list[dict], group: str, w_legal: float) -> list[tuple[float, str]]:
    w_enf = 1.0 - w_legal
    pairs: list[tuple[float, str]] = []
    for s in rows:
        dim = s["dimension"] or ""
        if group == "legal" and dim not in _LEGAL_DIMENSIONS:
            continue
        if group == "enforcement" and dim not in _ENFORCEMENT_DIMS:
            continue

        if dim == "mixed" and s["criterion_number"] in _MIXED_CRITERIA:
            leg = s["legal_subscore"]
            enf = s["enforcement_subscore"]
            # Reweighted combined score — the "inside mixed" weight. Only used as a
            # fallback when the dimension-specific subscore is missing.
            combined = (
                leg * w_legal + enf * w_enf
                if (leg is not None and enf is not None)
                else None
            )
            if group == "legal":
                val = leg if leg is not None else (
                    combined if combined is not None else s["criterion_score"]
                )
            else:
                val = enf if enf is not None else (
                    combined if combined is not None else s["criterion_score"]
                )
        else:
            val = s["criterion_score"]

        pairs.append((float(val), s.get("confidence") or "low"))
    return pairs


def _weighted_mean(pairs: list[tuple[float, str]]) -> float | None:
    if not pairs:
        return None
    weights = [CONFIDENCE_WEIGHTS.get(c, 0.4) for _, c in pairs]
    values = [v for v, _ in pairs]
    total_w = sum(weights)
    if total_w == 0:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / total_w


def compute_final(rows: list[dict], w_legal: float) -> tuple[float, float | None, float | None]:
    w_enf = 1.0 - w_legal
    legal_mean = _weighted_mean(_extract_pairs(rows, "legal", w_legal))
    enf_mean = _weighted_mean(_extract_pairs(rows, "enforcement", w_legal))
    if legal_mean is not None and enf_mean is not None:
        final = legal_mean * w_legal + enf_mean * w_enf
    elif legal_mean is not None:
        final = legal_mean
    elif enf_mean is not None:
        final = enf_mean
    else:
        final = 1.0
    return round(final, 4), legal_mean, enf_mean


# ---------------------------------------------------------------------------
# Ranking + Spearman (manual; scipy not available in the env)
# ---------------------------------------------------------------------------

def average_ranks(scores: dict[str, float]) -> dict[str, float]:
    """Rank 1 = highest score. Ties share the average rank."""
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # 1-indexed average of the tied block
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = avg
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n == 0:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman(a: dict[str, float], b: dict[str, float]) -> float:
    """Spearman ρ = Pearson correlation of average ranks."""
    ra = average_ranks(a)
    rb = average_ranks(b)
    keys = sorted(ra.keys())
    return pearson([ra[k] for k in keys], [rb[k] for k in keys])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=None,
                        help="Reference year (default: the year present in the index)")
    args = parser.parse_args()

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Resolve reference year from the ranked (non-partial) index if not given.
            if args.year is None:
                cur.execute(
                    "SELECT MAX(reference_year) AS y FROM country_index_scores "
                    "WHERE partial_coverage = false"
                )
                row = cur.fetchone()
                year = row["y"] if row and row["y"] else None
                if year is None:
                    print("ERROR: no non-partial index rows found", file=sys.stderr)
                    sys.exit(1)
            else:
                year = args.year

            # The 9 ranked countries (partial_coverage = false).
            cur.execute(
                """
                SELECT c.id, c.name, c.iso_code
                FROM country_index_scores cis
                JOIN countries c ON c.id = cis.country_id
                WHERE cis.reference_year = %s AND cis.partial_coverage = false
                ORDER BY c.name
                """,
                (year,),
            )
            countries = [
                {"id": r["id"], "name": r["name"], "iso": (r["iso_code"] or "").strip()}
                for r in cur.fetchall()
            ]

            # Criterion scores per country.
            crit_by_country: dict[int, list[dict]] = {}
            for c in countries:
                cur.execute(
                    """
                    SELECT criterion_number, dimension, criterion_score,
                           legal_subscore, enforcement_subscore, confidence
                    FROM criterion_scores
                    WHERE country_id = %s AND reference_year = %s
                    """,
                    (c["id"], year),
                )
                crit_by_country[c["id"]] = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    if not countries:
        print("ERROR: no ranked countries to analyse", file=sys.stderr)
        sys.exit(1)

    # Compute per scenario.
    # results[w] = {name: {"final", "legal", "enf", "rank", "band"}}
    results: dict[float, dict[str, dict]] = {}
    for w in SCENARIOS:
        finals: dict[str, float] = {}
        detail: dict[str, dict] = {}
        for c in countries:
            final, legal, enf = compute_final(crit_by_country[c["id"]], w)
            finals[c["name"]] = final
            detail[c["name"]] = {
                "iso": c["iso"], "final": final, "legal": legal, "enf": enf,
                "band": band_for(final),
            }
        ranks = average_ranks(finals)
        for name in detail:
            detail[name]["rank"] = ranks[name]
        results[w] = detail

    baseline = results[BASELINE_W]
    baseline_finals = {n: baseline[n]["final"] for n in baseline}

    # Per-scenario summary metrics vs baseline.
    summary_rows = []
    for w in SCENARIOS:
        scen = results[w]
        scen_finals = {n: scen[n]["final"] for n in scen}
        rho = spearman(scen_finals, baseline_finals)
        pos_changes = sum(
            1 for n in scen if scen[n]["rank"] != baseline[n]["rank"]
        )
        cat_changes = sum(
            1 for n in scen if scen[n]["band"] != baseline[n]["band"]
        )
        max_disp = max(
            (abs(scen[n]["rank"] - baseline[n]["rank"]) for n in scen), default=0.0
        )
        summary_rows.append({
            "w_legal": w, "w_enforcement": round(1 - w, 4),
            "spearman_vs_40_60": round(rho, 4),
            "position_changes": pos_changes, "category_changes": cat_changes,
            "max_rank_displacement": max_disp,
        })

    # ---- Write tidy CSV ----
    out_dir = Path("data/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sensitivity_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "w_legal", "w_enforcement", "country_name", "iso_code",
            "legal_mean", "enforcement_mean", "final_score", "rank", "pi_band",
            "rank_delta_vs_40_60", "band_changed_vs_40_60",
        ])
        for w in SCENARIOS:
            for name in sorted(results[w], key=lambda n: results[w][n]["rank"]):
                d = results[w][name]
                b = baseline[name]
                writer.writerow([
                    w, round(1 - w, 4), name, d["iso"],
                    round(d["legal"], 4) if d["legal"] is not None else "",
                    round(d["enf"], 4) if d["enf"] is not None else "",
                    d["final"], d["rank"], d["band"],
                    round(d["rank"] - b["rank"], 1),
                    "yes" if d["band"] != b["band"] else "no",
                ])

    # ---- Write markdown summary ----
    md_path = out_dir / "sensitivity_analysis.md"
    lines: list[str] = []
    lines.append(f"# Sensitivity analysis — legal/enforcement weighting ({year})\n")
    lines.append(
        f"Countries analysed: **{len(countries)}** (partial_coverage = false). "
        f"Baseline: **w_legal = {BASELINE_W}** (40/60). "
        "Aggregation math ported verbatim from `IndexAggregator`.\n"
    )

    lines.append("## Summary vs 40/60 baseline\n")
    lines.append("| w_legal | w_enf | Spearman ρ vs 40/60 | Position changes | Category changes | Max rank shift |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in summary_rows:
        lines.append(
            f"| {r['w_legal']:.1f} | {r['w_enforcement']:.1f} | "
            f"{r['spearman_vs_40_60']:.4f} | {r['position_changes']} | "
            f"{r['category_changes']} | {r['max_rank_displacement']:.1f} |"
        )
    lines.append("")

    # Rank matrix (country × scenario).
    lines.append("## Rank by scenario (1 = best)\n")
    header = "| Country | " + " | ".join(f"w={w:.1f}" for w in SCENARIOS) + " |"
    sep = "|" + "---|" * (len(SCENARIOS) + 1)
    lines.append(header)
    lines.append(sep)
    for name in sorted(baseline, key=lambda n: baseline[n]["rank"]):
        cells = " | ".join(
            f"{results[w][name]['rank']:.0f} ({results[w][name]['final']:.2f})"
            for w in SCENARIOS
        )
        lines.append(f"| {name} | {cells} |")
    lines.append("")

    # Final-score matrix.
    lines.append("## Final score by scenario\n")
    lines.append(header)
    lines.append(sep)
    for name in sorted(baseline, key=lambda n: baseline[n]["rank"]):
        cells = " | ".join(f"{results[w][name]['final']:.4f}" for w in SCENARIOS)
        lines.append(f"| {name} | {cells} |")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    # ---- Console summary ----
    print(f"\nSensitivity analysis ({year}) — {len(countries)} countries")
    print("=" * 64)
    print(f"{'w_legal':>8} {'ρ vs 40/60':>12} {'pos Δ':>7} {'cat Δ':>7} {'max shift':>10}")
    for r in summary_rows:
        print(f"{r['w_legal']:>8.1f} {r['spearman_vs_40_60']:>12.4f} "
              f"{r['position_changes']:>7} {r['category_changes']:>7} "
              f"{r['max_rank_displacement']:>10.1f}")
    print("=" * 64)
    print(f"CSV:      {csv_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
