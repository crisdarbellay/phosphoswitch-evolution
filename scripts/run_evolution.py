#!/usr/bin/env python3
"""
run_evolution.py — CLI driver for in silico directed evolution of phosphoswitches.

Usage
-----
    python scripts/run_evolution.py \\
        --starting-seq ASSTPLSPTRITRLQEKEDLQELNRRLAVYIDRVRSEETENAGLRLRITESEEVVSREV \\
        --target lmna \\
        --rounds 5 \\
        --n-top 4 \\
        --out-dir ./lmna_evolution \\
        --project-name lmna_phosphoswitch

Workflow
--------
Round preparation
    For each round the script writes a FASTA of all single-mutant candidates
    to ``<out-dir>/round_NN/candidates.fasta``.

External ColabFold step
    Run ColabFold on the FASTA and score each pair (WT + mutant) to produce
    ``<out-dir>/round_NN/scores.csv``.  The scoring pipeline writes the
    ``contact_delta_score`` column (see scoring.py).

Selection
    Once ``scores.csv`` is present the script reads it, selects the top
    ``--n-top`` sequences, writes ``top_selections.csv``, and generates the
    FASTA for the next round.

Running fully automated (custom scorer)
    Pass ``--colabfold-bin`` to supply the ColabFold executable path.  When
    provided, the script shells out to ColabFold automatically after writing
    each FASTA.  The scoring step still requires a separate ``score_contacts.py``
    script to convert ColabFold PDBs into ``scores.csv``.

Re-analysis mode
    If all ``scores.csv`` files already exist (e.g., re-running after a
    completed experiment), the script reads them directly and reproduces all
    selections without re-running ColabFold.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Allow running as a script without installing the package
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE / "src"))

from phosphoswitch_evolution.evolution import (
    EvolutionConfig,
    lmna_config,
    mapre1_config,
    run_evolution,
    summarise_evolution,
)
from phosphoswitch_evolution.io_utils import load_scores_csv
from phosphoswitch_evolution.evolution import select_top  # noqa: F401


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="In silico directed evolution of phosphoswitches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    seq_group = p.add_mutually_exclusive_group()
    seq_group.add_argument(
        "--starting-seq",
        metavar="SEQ",
        help="Starting protein sequence (one-letter codes).",
    )
    seq_group.add_argument(
        "--target",
        choices=["lmna", "mapre1"],
        help="Use a pre-built config for LMNA or MAPRE1 (sets template seq + fixed positions).",
    )

    p.add_argument("--rounds",       type=int,  default=5,    help="Number of evolution rounds (default: 5)")
    p.add_argument("--n-top",        type=int,  default=4,    help="Top sequences kept per round (default: 4)")
    p.add_argument("--phosphosite",  type=int,  default=None, help="1-indexed phosphosite position (overrides target preset)")
    p.add_argument("--fixed-pos",    type=str,  default=None, metavar="1,2,3", help="Comma-separated list of fixed positions (never mutated)")
    p.add_argument("--allowed-aas",  type=str,  default=None, metavar="ACDEF...", help="Allowed amino acids for scanning (default: all 20 standard AAs)")
    p.add_argument("--out-dir",      type=Path, default=Path("evolution_output"), help="Root output directory")
    p.add_argument("--project-name", type=str,  default="phosphoswitch", help="Short label for FASTA headers and output files")
    p.add_argument("--score-col",    type=str,  default="score", help="Column name used for selection (default: 'score')")
    p.add_argument(
        "--colabfold-bin",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the colabfold_batch executable.  If given, ColabFold is "
             "run automatically after each FASTA is written.",
    )
    p.add_argument("--dry-run", action="store_true", help="Only write FASTAs; do not attempt to load scores or run ColabFold.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


# ---------------------------------------------------------------------------
# ColabFold runner
# ---------------------------------------------------------------------------

def run_colabfold(
    fasta_path: Path,
    out_dir: Path,
    colabfold_bin: str,
    verbose: bool = False,
) -> bool:
    """Shell out to ``colabfold_batch`` for a single round FASTA.

    Returns True on success, False on failure.
    """
    cmd = [
        colabfold_bin,
        str(fasta_path),
        str(out_dir / "colabfold_output"),
        "--num-recycle", "3",
        "--model-type", "auto",
    ]
    if verbose:
        print(f"[colabfold] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=not verbose)
    if result.returncode != 0:
        print(f"[ERROR] ColabFold failed for {fasta_path}: exit code {result.returncode}")
        if not verbose:
            print(result.stderr.decode(errors="replace")[-2000:])
        return False
    return True


# ---------------------------------------------------------------------------
# Scores loader factory
# ---------------------------------------------------------------------------

def make_scores_loader(colabfold_bin: str | None, verbose: bool):
    """Return a scores_loader callable for :func:`run_evolution`, or None."""
    if colabfold_bin is None:
        return None

    def loader(round_dir: Path):
        fasta_path = round_dir / "candidates.fasta"
        scores_csv = round_dir / "scores.csv"

        if not scores_csv.exists():
            ok = run_colabfold(fasta_path, round_dir, colabfold_bin, verbose)
            if not ok:
                raise RuntimeError(f"ColabFold failed for {round_dir}")
            if not scores_csv.exists():
                raise FileNotFoundError(
                    f"{scores_csv} not found after ColabFold.  "
                    "Make sure your scoring script writes scores.csv."
                )
        return load_scores_csv(scores_csv, round_num=int(round_dir.name.split("_")[-1]))

    return loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # --- Build config -------------------------------------------------------
    if args.target == "lmna":
        cfg_kwargs: dict = {}
        if args.starting_seq:
            cfg_kwargs["template_seq"] = args.starting_seq
        if args.phosphosite:
            cfg_kwargs["phosphosite_pos"] = args.phosphosite
        cfg = lmna_config(**cfg_kwargs)
    elif args.target == "mapre1":
        cfg_kwargs = {}
        if args.starting_seq:
            cfg_kwargs["template_seq"] = args.starting_seq
        if args.phosphosite:
            cfg_kwargs["phosphosite_pos"] = args.phosphosite
        cfg = mapre1_config(**cfg_kwargs)
    else:
        if not args.starting_seq:
            parser.error("Either --starting-seq or --target {lmna,mapre1} is required.")
        fixed = frozenset(int(x) for x in args.fixed_pos.split(",")) if args.fixed_pos else frozenset()
        cfg = EvolutionConfig(
            template_seq=args.starting_seq,
            phosphosite_pos=args.phosphosite or 1,
            fixed_positions=fixed,
            n_rounds=args.rounds,
            n_top=args.n_top,
            score_column=args.score_col,
            out_dir=args.out_dir,
            project_name=args.project_name,
        )

    # Apply CLI overrides
    cfg.n_rounds     = args.rounds
    cfg.n_top        = args.n_top
    cfg.out_dir      = args.out_dir
    cfg.project_name = args.project_name
    if args.allowed_aas:
        cfg.allowed_aas = args.allowed_aas

    if args.verbose:
        print(f"[config] project      : {cfg.project_name}")
        print(f"[config] template_len : {len(cfg.template_seq)}")
        print(f"[config] phosphosite  : {cfg.phosphosite_pos}")
        print(f"[config] fixed_pos    : {sorted(cfg.fixed_positions)}")
        print(f"[config] designable   : {len(cfg.designable_positions)} positions")
        print(f"[config] rounds       : {cfg.n_rounds}")
        print(f"[config] n_top        : {cfg.n_top}")
        print(f"[config] out_dir      : {cfg.out_dir}")

    if args.dry_run:
        print("[dry-run] Writing candidate FASTAs only ...")
        from phosphoswitch_evolution.evolution import run_round
        seeds = [(cfg.template_seq, [])]
        for rnum in range(1, cfg.n_rounds + 1):
            fasta = run_round(rnum, seeds, cfg)
            print(f"  round {rnum}: {fasta}")
        print("[dry-run] Done.  No ColabFold was run.")
        return

    # --- Run evolution ------------------------------------------------------
    loader = make_scores_loader(args.colabfold_bin, args.verbose)
    all_rounds = run_evolution(cfg, scores_loader=loader)

    if not all_rounds:
        print("[INFO] No rounds completed (scores.csv files may be missing).")
        return

    summary = summarise_evolution(all_rounds, score_column=cfg.score_column)
    summary_path = cfg.out_dir / "evolution_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== Evolution summary ===")
    print(summary.to_string(index=False))
    print(f"\nSummary written to: {summary_path}")
    print(f"Best overall: {summary.iloc[summary['best_score'].argmax()].to_dict()}")


if __name__ == "__main__":
    main()
