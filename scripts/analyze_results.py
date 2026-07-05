#!/usr/bin/env python3
"""
analyze_results.py — Post-hoc analysis of directed-evolution results.

Reads scores.csv files from a completed evolution run and produces:
  - Fitness trajectory plots (best score per round)
  - Best-so-far trajectory
  - Mutation enrichment analysis (which substitutions dominate top hits)
  - Per-round summary CSV

Usage
-----
    python scripts/analyze_results.py \\
        --root ./lmna_evolution \\
        --out  ./lmna_analysis \\
        --top-n 10

    # Compare two methods (baseline vs MSA-engineered):
    python scripts/analyze_results.py \\
        --root /mnt/Data/.../serial_mutagenesis_results \\
        --out  ./comparison_analysis
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script without installing the package
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE / "src"))

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phosphoswitch_evolution.io_utils import (
    discover_rounds,
    load_scores_csv,
    parse_mut_tokens,
)

# ---------------------------------------------------------------------------
# Experiment discovery (mirrors analyse_serial_mutagenesis.py logic)
# ---------------------------------------------------------------------------

_ENGINEERED_HINTS = [
    "msa_modified", "msa-engineered", "msa_engineered", "_msa_modified",
    "msa_modified_2", "msa_force", "msa_forced", "engineered",
]
_BASELINE_HINTS = ["baseline", "no_msa", "no-msa", "nomsa", "plain", "vanilla"]


def _infer_family(name: str) -> str:
    n = name.lower()
    for h in _ENGINEERED_HINTS + _BASELINE_HINTS:
        n = n.replace(h, "")
    n = re.sub(r"[\W_]+", " ", n).strip()
    return n.split()[0] if n else "unknown"


def _infer_method(name: str) -> str:
    n = name.lower()
    if any(h in n for h in _ENGINEERED_HINTS):
        return "msa_engineered"
    if any(h in n for h in _BASELINE_HINTS):
        return "baseline"
    return "unknown"


def load_all_scores(root: Path) -> pd.DataFrame:
    """Discover and load all scores.csv files under *root*."""
    round_infos = discover_rounds(root)
    if not round_infos:
        raise SystemExit(f"No round_*/scores.csv found under {root}")

    frames = []
    for ri in round_infos:
        exp_name = ri.exp_dir.name if ri.exp_dir != root else "ROOT"
        family   = _infer_family(exp_name)
        method   = _infer_method(exp_name)
        try:
            df = load_scores_csv(
                ri.scores_csv,
                experiment=exp_name,
                family=family,
                method=method,
                round_num=ri.round_num,
            )
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] {ri.scores_csv}: {exc}")

    if not frames:
        raise SystemExit("No scores loaded (all failed).")

    all_scores = pd.concat(frames, ignore_index=True)

    # Normalise unknown methods: if a family has an engineered experiment,
    # label the unknown one as baseline
    fam_methods: dict = {}
    for _, row in all_scores.drop_duplicates(["experiment", "method"]).iterrows():
        fam_methods.setdefault(row["family"], set()).add(row["method"])
    for fam, methods in fam_methods.items():
        if "msa_engineered" in methods:
            mask = (all_scores["family"] == fam) & (all_scores["method"] == "unknown")
            all_scores.loc[mask, "method"] = "baseline"

    return all_scores


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def per_round_summary(scores: pd.DataFrame) -> pd.DataFrame:
    ok = scores[scores["ok"].astype(bool)].copy()
    if ok.empty:
        return pd.DataFrame()

    def agg(g):
        x = g["score"].to_numpy(dtype=float)
        return pd.Series({
            "n_ok":   len(x),
            "best":   float(np.nanmax(x)),
            "mean":   float(np.nanmean(x)),
            "median": float(np.nanmedian(x)),
            "q10":    float(np.nanquantile(x, 0.10)),
            "q90":    float(np.nanquantile(x, 0.90)),
        })

    return ok.groupby(["family", "method", "round"], dropna=False).apply(agg).reset_index()


def mutation_enrichment(scores: pd.DataFrame, top_frac: float = 0.05) -> pd.DataFrame:
    """Identify mutations enriched among the top *top_frac* of all hits."""
    ok = scores[scores["ok"].astype(bool)].copy()
    rows = []
    for (family, method), sub in ok.groupby(["family", "method"]):
        k = max(5, int(len(sub) * top_frac))
        top = sub.nlargest(k, "score")

        def extract_muts(df):
            muts = []
            for mp in df["mut_part"].astype(str):
                for t in parse_mut_tokens(mp):
                    muts.append((t.pos, t.aa))
            return muts

        all_muts  = extract_muts(sub)
        top_muts  = extract_muts(top)
        if not all_muts or not top_muts:
            continue

        from collections import Counter
        ac = Counter(all_muts)
        tc = Counter(top_muts)

        for (pos, aa), ctop in tc.items():
            call = ac.get((pos, aa), 0)
            ftop = ctop / sum(tc.values())
            fall = call / sum(ac.values()) if sum(ac.values()) else 0
            rows.append({
                "family":      family,
                "method":      method,
                "pos":         pos,
                "to_aa":       aa,
                "count_top":   ctop,
                "count_all":   call,
                "freq_top":    ftop,
                "freq_all":    fall,
                "enrich_log2": float(np.log2((ftop + 1e-9) / (fall + 1e-9))),
            })

    return pd.DataFrame(rows)


def top_mutants(scores: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the top *top_n* mutants per (family, method), pooled across rounds."""
    ok = scores[scores["ok"].astype(bool)].copy()
    return (
        ok.sort_values("score", ascending=False)
          .groupby(["family", "method"], dropna=False)
          .head(top_n)
          .copy()
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_fitness_trajectory(per_round: pd.DataFrame, out_path: Path, title: str):
    """Plot best-so-far score per round, grouped by method."""
    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for method, sub in per_round.groupby("method"):
        sub = sub.sort_values("round")
        best = -np.inf
        bsf  = []
        for v in sub["best"]:
            best = max(best, float(v))
            bsf.append(best)
        ax.plot(sub["round"], bsf, marker="o", linewidth=2, label=method)
    ax.set_xlabel("Round")
    ax.set_ylabel("Best-so-far contact_delta_score")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _savefig(out_path)


def plot_score_distribution(scores: pd.DataFrame, fam: str, out_path: Path):
    """Box-plots of score distribution per round (all methods overlaid)."""
    d = scores[(scores["family"] == fam) & (scores["ok"].astype(bool))].copy()
    if d.empty:
        return
    rounds  = sorted(d["round"].unique())
    methods = sorted(d["method"].unique())

    fig, ax = plt.subplots(figsize=(max(8, len(rounds) * 1.5), 5))
    positions = []
    data      = []
    labels    = []
    offset    = 0
    width     = 0.8 / max(len(methods), 1)

    for r in rounds:
        for i, m in enumerate(methods):
            sub = d[(d["round"] == r) & (d["method"] == m)]["score"].dropna().to_numpy()
            if len(sub) == 0:
                continue
            pos = offset + i * width
            positions.append(pos)
            data.append(sub)
            labels.append(f"R{r:02d}\n{m[:4]}")
        offset += 1 + len(methods) * width

    ax.boxplot(data, positions=positions, widths=width * 0.8, showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("contact_delta_score")
    ax.set_title(f"{fam}: score distributions per round")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _savefig(out_path)


def plot_enrichment(enrich_df: pd.DataFrame, fam: str, out_path: Path, top_n: int = 20):
    """Horizontal bar chart of mutation enrichment in top hits."""
    d = enrich_df[enrich_df["family"] == fam].copy()
    if d.empty:
        return
    d = d.sort_values("enrich_log2", ascending=False).head(top_n)

    plt.figure(figsize=(10, max(4, top_n * 0.3)))
    ax = plt.gca()
    y  = np.arange(len(d))
    ax.barh(y, d["enrich_log2"].to_numpy())
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{int(r.pos)}→{r.to_aa} ({r.method[:4]})" for _, r in d.iterrows()],
        fontsize=9,
    )
    ax.axvline(0, linewidth=1, linestyle="--", color="grey")
    ax.set_xlabel("log2 enrichment among top hits")
    ax.set_title(f"{fam}: most enriched mutations")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _savefig(out_path)


def print_best_lineages(scores: pd.DataFrame, top_n: int = 5):
    """Print the top *top_n* lineages for each family and method."""
    ok = scores[scores["ok"].astype(bool)].copy()
    print("\n=== Best lineages ===")
    for (fam, method), sub in ok.groupby(["family", "method"]):
        best = sub.nlargest(top_n, "score")
        print(f"\n{fam} | {method}")
        keep = [c for c in ("round", "score", "mutset", "n_mut", "rmsd_ca", "plddt_ph") if c in best.columns]
        print(best[keep].to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyse directed-evolution results from round_*/scores.csv files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--root",      required=True, type=Path, help="Root directory containing round_*/scores.csv (or experiment subdirs)")
    p.add_argument("--out",       type=Path, default=None,  help="Output directory (default: <root>/analysis)")
    p.add_argument("--top-n",     type=int, default=10,     help="Top-N mutants to report (default: 10)")
    p.add_argument("--top-frac",  type=float, default=0.05, help="Fraction of hits used for enrichment analysis (default: 0.05)")
    p.add_argument("--no-plots",  action="store_true",      help="Skip plotting (tables only)")
    return p


def main(argv=None):
    parser = build_parser()
    args   = parser.parse_args(argv)

    root = args.root.resolve()
    out  = (args.out or root / "analysis").resolve()
    tabs = out / "tables"
    figs = out / "figures"
    tabs.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        figs.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading scores from: {root}")
    all_scores = load_all_scores(root)
    all_scores.to_csv(tabs / "all_scores_long.csv", index=False)
    print(f"[INFO] Loaded {len(all_scores):,} rows from {all_scores['experiment'].nunique()} experiments")

    # --- Per-round summary ---
    pr = per_round_summary(all_scores)
    pr.to_csv(tabs / "per_round_summary.csv", index=False)

    # --- Mutation enrichment ---
    enrich = mutation_enrichment(all_scores, top_frac=args.top_frac)
    if not enrich.empty:
        enrich.to_csv(tabs / "mutation_enrichment.csv", index=False)

    # --- Top mutants ---
    tops = top_mutants(all_scores, top_n=args.top_n)
    keep = [c for c in ("family", "method", "round", "score", "mutset", "n_mut", "rmsd_ca", "plddt_ph") if c in tops.columns]
    tops[keep].to_csv(tabs / "top_mutants.csv", index=False)

    # --- Console output ---
    print_best_lineages(all_scores, top_n=args.top_n)

    # --- Plots ---
    if not args.no_plots:
        for fam in sorted(all_scores["family"].unique()):
            fam_pr = pr[pr["family"] == fam]
            if fam_pr.empty:
                continue
            plot_fitness_trajectory(
                fam_pr,
                figs / f"{fam}_fitness_trajectory",
                title=f"{fam}: fitness trajectory",
            )
            plot_score_distribution(all_scores, fam, figs / f"{fam}_score_distributions")
            if not enrich.empty:
                plot_enrichment(enrich, fam, figs / f"{fam}_mutation_enrichment", top_n=20)

    print(f"\n[DONE] Tables: {tabs}")
    if not args.no_plots:
        print(f"       Figures: {figs}")


if __name__ == "__main__":
    main()
