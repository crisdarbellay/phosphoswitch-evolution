"""
evolution.py — Core logic for in silico directed evolution of phosphoswitches.

Implements the greedy single-mutant scan used in the serial mutagenesis
experiments on LMNA (Y45, Src kinase) and MAPRE1 (Y247, Src kinase).

Algorithm overview
------------------
Each round of evolution:
  1. Start from the best sequence(s) of the previous round (or the WT).
  2. Enumerate all single-point substitutions at designable positions.
  3. Write a FASTA with all candidate sequences.
  4. (External step) Run ColabFold in paired WT/phospho mode.
  5. Load ``scores.csv`` produced by the scoring pipeline.
  6. Select the top *n_top* sequences by ``contact_delta_score``.
  7. Carry selected sequences forward as seeds for the next round.

Positions labelled as *fixed* are never mutated (e.g., the phosphosite
residue itself and structurally essential prolines in the LMNA template).

Key result
----------
LMNA best hit after 5 rounds:
    L20P_D25R_L37E_A42P_T49P
    contact_delta_score = 18.6, Cα-RMSD = 17.27 Å (vs WT)

Note: evolved sequences contain multiple prolines that disrupt the straight
helix.  They were not expressed directly but established the sequence
logic for the multi-state design module.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import pandas as pd

from .io_utils import (
    write_fasta,
    load_scores_csv,
    parse_mut_tokens,
    canonical_mutset,
    MutToken,
)

# ---------------------------------------------------------------------------
# Standard amino acids (excludes Cys by default to avoid disulphide artefacts)
# ---------------------------------------------------------------------------

ALL_AAS: str = "ACDEFGHIKLMNPQRSTVWY"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvolutionConfig:
    """Hyperparameters for one directed-evolution experiment.

    Attributes
    ----------
    template_seq:
        Wild-type sequence (one-letter codes, 1-indexed positions).
    phosphosite_pos:
        1-indexed position of the tyrosine phosphosite (e.g., 45 for LMNA Y45).
    fixed_positions:
        Residue positions that are never mutated.  Defaults to the positions
        used in the original LMNA experiments.
    allowed_aas:
        Amino acids considered during single-mutant scan.
    n_rounds:
        Number of evolution rounds (default 5).
    n_top:
        Number of top sequences kept per round (default 4).
    score_column:
        Column in ``scores.csv`` used for selection (default ``"score"``).
    min_score_delta:
        Minimum score improvement required to accept a mutation.  Rounds with
        no improvement above this threshold terminate early.
    out_dir:
        Root output directory.  Round subdirectories are created automatically.
    project_name:
        Short label used to prefix mutant IDs in FASTA headers.
    """

    template_seq:     str
    phosphosite_pos:  int                       = 45
    fixed_positions:  FrozenSet[int]            = field(default_factory=frozenset)
    allowed_aas:      str                       = ALL_AAS
    n_rounds:         int                       = 5
    n_top:            int                       = 4
    score_column:     str                       = "score"
    min_score_delta:  float                     = 0.0
    out_dir:          Path                      = Path("evolution_output")
    project_name:     str                       = "phosphoswitch"

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        # Always keep the phosphosite itself fixed
        if self.fixed_positions:
            object.__setattr__(
                self,
                "fixed_positions",
                self.fixed_positions | frozenset({self.phosphosite_pos}),
            )
        else:
            object.__setattr__(
                self,
                "fixed_positions",
                frozenset({self.phosphosite_pos}),
            )

    @property
    def designable_positions(self) -> List[int]:
        """Positions that may be mutated, in ascending order."""
        return sorted(
            p for p in range(1, len(self.template_seq) + 1)
            if p not in self.fixed_positions
        )


# ---------------------------------------------------------------------------
# LMNA and MAPRE1 preset configs
# ---------------------------------------------------------------------------

def lmna_config(**kwargs) -> EvolutionConfig:
    """Return an :class:`EvolutionConfig` pre-loaded for the LMNA experiment.

    The template is the 59-residue LMNA coiled-coil fragment used in the
    original serial mutagenesis study.  Fixed positions match those used
    in ``mech_iter_bidirectional.py``.

    Extra keyword arguments override the defaults.
    """
    fixed = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20, 30, 42, 49, 59})
    defaults: Dict = dict(
        template_seq=(
            "ASSTPLSPTRITRLQEKEDLQELNRRLAVYIDRVRSEETENAGLRLRITESEEVVSREV"
        ),
        phosphosite_pos=45,   # Y45 (Src kinase site)
        fixed_positions=fixed,
        n_rounds=5,
        n_top=4,
        project_name="lmna_phosphoswitch",
    )
    defaults.update(kwargs)
    return EvolutionConfig(**defaults)


def mapre1_config(**kwargs) -> EvolutionConfig:
    """Return an :class:`EvolutionConfig` pre-loaded for the MAPRE1 experiment.

    MAPRE1 Y247 is phosphorylated by Src kinase.  Update ``template_seq``
    with the actual MAPRE1 sequence before use.

    Extra keyword arguments override the defaults.
    """
    defaults: Dict = dict(
        template_seq="",          # user must supply
        phosphosite_pos=247,
        n_rounds=5,
        n_top=4,
        project_name="mapre1_phosphoswitch",
    )
    defaults.update(kwargs)
    return EvolutionConfig(**defaults)


# ---------------------------------------------------------------------------
# Single-mutant generation
# ---------------------------------------------------------------------------


def generate_single_mutants(
    sequence: str,
    config: EvolutionConfig,
    accumulated_muts: List[MutToken] | None = None,
    lineage_label: str = "POOL",
) -> Dict[str, str]:
    """Generate all single-point substitutions at designable positions.

    Parameters
    ----------
    sequence:
        Current sequence to scan (may already carry prior-round mutations).
    config:
        :class:`EvolutionConfig` specifying designable positions and allowed AAs.
    accumulated_muts:
        Mutation tokens already present in *sequence* (for FASTA header labels).
    lineage_label:
        Pool/lineage identifier embedded in the FASTA headers.

    Returns
    -------
    Dict mapping FASTA header to sequence string.
    """
    accumulated_muts = accumulated_muts or []
    records: Dict[str, str] = {}

    for pos in config.designable_positions:
        if pos > len(sequence):
            continue
        current_aa = sequence[pos - 1]
        for new_aa in config.allowed_aas:
            if new_aa == current_aa:
                continue
            new_seq = sequence[: pos - 1] + new_aa + sequence[pos:]
            # Build mutation token list
            new_tok = MutToken(wt=current_aa, pos=pos, aa=new_aa)
            all_toks = sorted(
                accumulated_muts + [new_tok], key=lambda t: t.pos
            )
            # Remove any duplicate positions (keep the new one)
            seen_pos: Set[int] = set()
            deduped: List[MutToken] = []
            for tok in reversed(all_toks):
                if tok.pos not in seen_pos:
                    deduped.insert(0, tok)
                    seen_pos.add(tok.pos)

            mut_str = "__".join(str(t) for t in deduped) if deduped else "WT0"
            header = f"{config.project_name}__{mut_str}"
            records[header] = new_seq

    return records


def generate_wt_fasta(config: EvolutionConfig) -> Dict[str, str]:
    """Return a FASTA dict containing only the wild-type sequence."""
    return {f"{config.project_name}__WT0": config.template_seq}


# ---------------------------------------------------------------------------
# Round management
# ---------------------------------------------------------------------------


def select_top(
    scores_df: pd.DataFrame,
    n_top: int,
    score_column: str = "score",
) -> pd.DataFrame:
    """Return the top *n_top* rows by *score_column*.

    Only rows with ``ok == True`` are considered.  In case of ties the first
    occurrence (as ordered in the CSV) is kept.
    """
    ok = scores_df[scores_df["ok"].astype(bool)].copy()
    return ok.nlargest(n_top, score_column, keep="first")


def run_round(
    round_num: int,
    seeds: List[Tuple[str, List[MutToken]]],
    config: EvolutionConfig,
) -> Path:
    """Prepare the FASTA input for one evolution round.

    This function generates single-mutant FASTAs for every seed sequence and
    writes them to ``<out_dir>/round_{round_num:02d}/candidates.fasta``.

    After running ColabFold externally on this FASTA, call
    :func:`load_scores_csv` on the resulting ``scores.csv`` and then
    :func:`select_top` to choose the seeds for the next round.

    Parameters
    ----------
    round_num:
        Current round index (1-based).
    seeds:
        List of ``(sequence, accumulated_mutations)`` tuples carried over from
        the previous round's selection.  Round 1 receives the WT.
    config:
        :class:`EvolutionConfig`.

    Returns
    -------
    Path to the written FASTA file.
    """
    round_dir = config.out_dir / f"round_{round_num:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    all_records: Dict[str, str] = {}
    for seq, muts in seeds:
        lineage = f"L{len(muts)}" if muts else "POOL"
        records = generate_single_mutants(
            sequence=seq,
            config=config,
            accumulated_muts=muts,
            lineage_label=lineage,
        )
        all_records.update(records)

    fasta_path = round_dir / "candidates.fasta"
    write_fasta(all_records, fasta_path)
    return fasta_path


# ---------------------------------------------------------------------------
# Full evolution loop
# ---------------------------------------------------------------------------


def run_evolution(
    config: EvolutionConfig,
    scores_loader=None,
) -> List[Tuple[int, pd.DataFrame]]:
    """Run the full directed-evolution loop.

    This function orchestrates the round-by-round FASTA generation and score
    loading.  ColabFold must be run externally between rounds; by default this
    function expects a ``scores.csv`` to already exist in each round directory
    (useful for re-analysis).  Pass a custom *scores_loader* callable for
    integration with an automated ColabFold runner.

    Parameters
    ----------
    config:
        :class:`EvolutionConfig` controlling all evolution hyperparameters.
    scores_loader:
        Optional callable ``(round_dir: Path) -> pd.DataFrame``.  If None,
        :func:`~phosphoswitch_evolution.io_utils.load_scores_csv` is used on
        ``round_dir/scores.csv``.

    Returns
    -------
    List of ``(round_num, top_df)`` tuples — the top selections for each round.
    """
    config.out_dir.mkdir(parents=True, exist_ok=True)

    # Write round 0 WT FASTA for baseline reference
    wt_fasta = config.out_dir / "round_00" / "wt.fasta"
    wt_fasta.parent.mkdir(parents=True, exist_ok=True)
    write_fasta(generate_wt_fasta(config), wt_fasta)

    # Seeds: start with the WT sequence, no accumulated mutations
    seeds: List[Tuple[str, List[MutToken]]] = [
        (config.template_seq, [])
    ]

    all_rounds: List[Tuple[int, pd.DataFrame]] = []

    for rnum in range(1, config.n_rounds + 1):
        # --- Write candidate FASTA ---
        fasta_path = run_round(rnum, seeds, config)
        round_dir  = fasta_path.parent
        scores_csv = round_dir / "scores.csv"

        # --- Load scores (real or pre-existing) ---
        if scores_loader is not None:
            scores_df = scores_loader(round_dir)
        elif scores_csv.exists():
            scores_df = load_scores_csv(
                scores_csv,
                experiment=config.project_name,
                round_num=rnum,
            )
        else:
            print(
                f"[round {rnum}] Waiting for scores: run ColabFold on "
                f"{fasta_path} then place scores.csv in {round_dir}"
            )
            break

        # --- Select top sequences ---
        top_df = select_top(scores_df, config.n_top, config.score_column)
        all_rounds.append((rnum, top_df))

        if top_df.empty:
            print(f"[round {rnum}] No valid scores. Stopping.")
            break

        # Save top selections for inspection
        top_df.to_csv(round_dir / "top_selections.csv", index=False)

        # --- Prepare seeds for next round ---
        new_seeds: List[Tuple[str, List[MutToken]]] = []
        for _, row in top_df.iterrows():
            mut_part = str(row.get("mut_part", "WT0"))
            muts = parse_mut_tokens(mut_part)
            # Reconstruct the sequence by applying all mutations to template
            seq = list(config.template_seq)
            for tok in muts:
                if 1 <= tok.pos <= len(seq):
                    seq[tok.pos - 1] = tok.aa
            new_seeds.append(("".join(seq), muts))

        seeds = new_seeds

        # --- Early stopping ---
        if rnum > 1:
            prev_best = max(
                r[1][config.score_column].max()
                for r in all_rounds[:-1]
                if not r[1].empty
            )
            curr_best = top_df[config.score_column].max()
            if float(curr_best) - float(prev_best) <= config.min_score_delta:
                print(
                    f"[round {rnum}] No improvement (Δ ≤ {config.min_score_delta}). Stopping."
                )
                break

    return all_rounds


# ---------------------------------------------------------------------------
# Convenience: summarise an evolution result
# ---------------------------------------------------------------------------


def summarise_evolution(
    all_rounds: List[Tuple[int, pd.DataFrame]],
    score_column: str = "score",
) -> pd.DataFrame:
    """Build a summary dataframe of best score per round.

    Parameters
    ----------
    all_rounds:
        Output of :func:`run_evolution`.
    score_column:
        Name of the score column.

    Returns
    -------
    DataFrame with columns: round, best_score, best_mutset, n_selected.
    """
    rows = []
    for rnum, top_df in all_rounds:
        if top_df.empty:
            continue
        best_row = top_df.loc[top_df[score_column].idxmax()]
        rows.append(
            {
                "round":       rnum,
                "best_score":  float(best_row[score_column]),
                "best_mutset": str(best_row.get("mutset", "")),
                "n_selected":  len(top_df),
            }
        )
    return pd.DataFrame(rows)
