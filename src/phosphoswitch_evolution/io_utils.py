"""
io_utils.py — I/O helpers for the phosphoswitch directed-evolution pipeline.

Covers:
  - FASTA reading and writing (multi-record)
  - scores.csv loading with column normalisation matching the pipeline schema
  - Round-directory discovery (``round_01/``, ``round_1/``, etc.)
  - Lineage tracking helpers

scores.csv column contract
--------------------------
Required:
    id                    — unique mutant identifier, format ``project__mutset``
    score                 — composite contact-delta score (float)

Optional (enriched by this module):
    contact_delta_score   — alias for score when the primary column is missing
    contact_delta_count   — number of new Cβ contacts
    rmsd_ca               — Cα RMSD between WT and phospho structures
    plddt_wt / plddt_ph   — ColabFold pLDDT for WT and phospho models
    dssp_delta            — change in α-helical content
    lineage               — label of the parent pool (POOL, L1, L2, …)
    error                 — error message if ColabFold failed for this variant
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Regex / constants
# ---------------------------------------------------------------------------

_ROUND_RE  = re.compile(r"round[_\- ]?(\d+)", re.IGNORECASE)
_MUT_RE    = re.compile(r"^([A-Z])(\d+)([A-Z])$")


# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------

def write_fasta(
    sequences: Dict[str, str],
    path: str | Path,
    line_width: int = 60,
) -> None:
    """Write a dict of {name: sequence} to a FASTA file.

    Parameters
    ----------
    sequences:
        Ordered mapping of sequence name → one-letter sequence.
    path:
        Output file path (parent directories are created if needed).
    line_width:
        Characters per sequence line (default 60).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for name, seq in sequences.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), line_width):
                fh.write(seq[i : i + line_width] + "\n")


def read_fasta(path: str | Path) -> Dict[str, str]:
    """Read a FASTA file and return {name: sequence}.

    Handles multi-line FASTA records.  The returned dict preserves insertion
    order (Python 3.7+).

    Parameters
    ----------
    path:
        Path to the FASTA file.
    """
    records: Dict[str, str] = {}
    current_name: Optional[str] = None
    chunks: List[str] = []

    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    records[current_name] = "".join(chunks)
                current_name = line[1:].split()[0]  # strip description
                chunks = []
            else:
                chunks.append(line.strip())

    if current_name is not None:
        records[current_name] = "".join(chunks)

    return records


def iter_fasta(path: str | Path) -> Iterator[Tuple[str, str]]:
    """Iterate over (name, sequence) pairs in a FASTA file (memory-efficient)."""
    current_name: Optional[str] = None
    chunks: List[str] = []

    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    yield current_name, "".join(chunks)
                current_name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())

    if current_name is not None:
        yield current_name, "".join(chunks)


# ---------------------------------------------------------------------------
# Mutation parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MutToken:
    """A single point mutation: wild-type residue, position, new residue."""
    wt:  str
    pos: int
    aa:  str

    def __str__(self) -> str:
        return f"{self.wt}{self.pos}{self.aa}"


def parse_mut_tokens(mut_part: str) -> List[MutToken]:
    """Parse a mutation string like ``E30D__A9I`` into a list of :class:`MutToken`.

    The special string ``"WT0"`` (or empty input) returns an empty list.
    Tokens are sorted by position.
    """
    if not mut_part or mut_part == "WT0":
        return []
    tokens: List[MutToken] = []
    for chunk in str(mut_part).split("__"):
        chunk = chunk.strip()
        m = _MUT_RE.match(chunk)
        if not m:
            continue
        tokens.append(MutToken(wt=m.group(1), pos=int(m.group(2)), aa=m.group(3)))
    tokens.sort(key=lambda t: t.pos)
    return tokens


def canonical_mutset(mut_part: str) -> str:
    """Return a canonical, position-sorted representation of a mutation set.

    Example: ``"A9I__E30D"`` → ``"A9I__E30D"`` (already sorted by position).
    """
    tokens = parse_mut_tokens(mut_part)
    return "WT0" if not tokens else "__".join(str(t) for t in tokens)


def parse_id(mutant_id: str) -> Tuple[str, str]:
    """Split a mutant ID of the form ``project__mutset`` into its two parts.

    Returns ``("unknown", mutant_id)`` if no ``__`` separator is found.
    """
    if "__" not in mutant_id:
        return "unknown", mutant_id
    project, mut_part = mutant_id.split("__", 1)
    return project, mut_part


# ---------------------------------------------------------------------------
# scores.csv loading
# ---------------------------------------------------------------------------

def load_scores_csv(
    path: str | Path,
    experiment: str = "",
    family: str = "",
    method: str = "",
    round_num: Optional[int] = None,
) -> pd.DataFrame:
    """Load a ``scores.csv`` file and normalise column names.

    Enriches the dataframe with:
    - ``experiment``, ``family``, ``method``, ``round`` metadata columns
    - ``project`` and ``mut_part`` parsed from the ``id`` column
    - ``mutset`` — canonical, position-sorted mutation set string
    - ``n_mut``  — number of point mutations in this variant
    - ``ok``     — boolean; False when an ``error`` column has a non-null value

    Parameters
    ----------
    path:
        Path to the scores CSV file.
    experiment:
        Label of the experiment (e.g., ``"lmna_msa_modified"``).
    family:
        Protein family (e.g., ``"lmna"``).
    method:
        Scoring method label (``"baseline"`` or ``"msa_engineered"``).
    round_num:
        Evolution round number.  Inferred from the directory name if None.
    """
    path = Path(path)

    # Infer round number from path if not provided
    if round_num is None:
        m = _ROUND_RE.search(str(path))
        round_num = int(m.group(1)) if m else 0

    df = pd.read_csv(path)

    # Normalise the primary key column to "id"
    if "id" not in df.columns:
        for candidate in ("stem", "name", "mutant", "mut_id"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "id"})
                break

    if "id" not in df.columns:
        raise ValueError(f"{path}: no 'id' column found (tried stem/name/mutant/mut_id)")

    # Normalise the primary score column to "score"
    if "score" not in df.columns:
        if "contact_delta_score" in df.columns:
            df["score"] = df["contact_delta_score"]
        else:
            raise ValueError(f"{path}: no 'score' or 'contact_delta_score' column")

    # Attach metadata
    df["experiment"] = experiment
    df["family"]     = family
    df["method"]     = method
    df["round"]      = int(round_num)

    # Parse mutant ID into project + mut_part
    parsed = df["id"].astype(str).map(parse_id)
    df["project"]  = [p for p, _ in parsed]
    df["mut_part"] = [m for _, m in parsed]
    df["mutset"]   = df["mut_part"].map(canonical_mutset)
    df["n_mut"]    = df["mut_part"].map(lambda x: len(parse_mut_tokens(x)))

    # Valid-row flag
    df["ok"] = True
    if "error" in df.columns:
        df["ok"] = df["error"].isna()

    return df


# ---------------------------------------------------------------------------
# Round / experiment discovery
# ---------------------------------------------------------------------------

@dataclass
class RoundInfo:
    """Metadata about a single evolution round's scores file."""
    round_num:  int
    scores_csv: Path
    exp_dir:    Path


def discover_rounds(root: str | Path) -> List[RoundInfo]:
    """Recursively find all ``round_*/scores.csv`` files under *root*.

    Handles both zero-padded (``round_01``) and unpadded (``round_1``) names.

    Parameters
    ----------
    root:
        Directory to search.  Both the root itself and immediate subdirectories
        are scanned (matching the layout of the serial-mutagenesis results).

    Returns
    -------
    List of :class:`RoundInfo` sorted by (experiment directory, round number).
    """
    root = Path(root)
    results: List[RoundInfo] = []
    seen: set = set()  # deduplicate

    # Candidates: the root itself and its immediate subdirectories
    candidates = [root] + [p for p in root.iterdir() if p.is_dir()]

    for exp_dir in candidates:
        for rdir in sorted(exp_dir.glob("round*")):
            if not rdir.is_dir():
                continue
            scores_csv = rdir / "scores.csv"
            if not scores_csv.exists():
                continue
            m = _ROUND_RE.search(rdir.name)
            if not m:
                continue
            rnum = int(m.group(1))
            key  = (exp_dir, rnum)
            if key in seen:
                continue
            seen.add(key)
            results.append(RoundInfo(round_num=rnum, scores_csv=scores_csv, exp_dir=exp_dir))

    results.sort(key=lambda r: (r.exp_dir, r.round_num))
    return results


# ---------------------------------------------------------------------------
# Lineage tracking
# ---------------------------------------------------------------------------

@dataclass
class LineageEntry:
    """One entry in an evolution lineage: a selected sequence and its score."""
    round_num: int
    mutset:    str   # canonical mutation string, e.g. "L20P__D25R"
    score:     float
    lineage:   str   # pool label, e.g. "L1"
    parent:    Optional[str] = None  # parent mutset from previous round


def build_lineages(
    per_round_selections: Dict[int, List[Tuple[str, float, str]]],
) -> List[LineageEntry]:
    """Convert per-round top selections into lineage entries.

    Parameters
    ----------
    per_round_selections:
        ``{round_num: [(mutset, score, lineage_label), ...]}``

    Returns
    -------
    Flat list of :class:`LineageEntry` objects in round order.
    """
    entries: List[LineageEntry] = []
    prev_mutsets: Dict[str, str] = {}  # lineage_label → mutset

    for rnum in sorted(per_round_selections):
        for mutset, score, lineage in per_round_selections[rnum]:
            # Best-effort parent lookup: strip the last mutation
            toks = parse_mut_tokens(mutset)
            parent_mutset = canonical_mutset("__".join(str(t) for t in toks[:-1])) if len(toks) > 1 else None
            entries.append(
                LineageEntry(
                    round_num=rnum,
                    mutset=mutset,
                    score=score,
                    lineage=lineage,
                    parent=parent_mutset,
                )
            )
            prev_mutsets[lineage] = mutset

    return entries
