"""
phosphoswitch_evolution
=======================

Computational directed evolution of phosphoswitches via iterative
in silico mutagenesis scored by ColabFold structure prediction.

Modules
-------
evolution   — core greedy single-mutant scan and round management
scoring     — contact-delta score from paired ColabFold PDB predictions
mechanism   — analytical phosphate-binding model (backbone-level scoring)
analysis    — trajectory summaries, fitness curves, mutation enrichment
io_utils    — FASTA / scores.csv I/O, round discovery, lineage tracking
"""

from .evolution import EvolutionConfig, run_evolution, run_round, generate_single_mutants
from .mechanism import (
    parse_pdb,
    bidirectional_score,
    directional_score,
    count_interactions,
)
from .scoring import contact_delta_score, parse_structure, find_contacts
from .io_utils import load_scores_csv, write_fasta, read_fasta, discover_rounds

__version__ = "1.0.0"
__all__ = [
    "EvolutionConfig",
    "run_evolution",
    "run_round",
    "generate_single_mutants",
    "parse_pdb",
    "bidirectional_score",
    "directional_score",
    "count_interactions",
    "contact_delta_score",
    "parse_structure",
    "find_contacts",
    "load_scores_csv",
    "write_fasta",
    "read_fasta",
    "discover_rounds",
]
