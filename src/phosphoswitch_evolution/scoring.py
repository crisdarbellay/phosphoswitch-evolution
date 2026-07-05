"""
scoring.py — Contact-delta scoring from ColabFold PDB predictions.

contact_delta_score quantifies how phosphorylation reshapes the protein's
contact network.  Two ColabFold PDB files are compared:

  - WT structure:      unphosphorylated sequence folded by ColabFold
  - Phospho structure: same sequence with the phosphosite modelled

New Cβ contacts that appear in the phospho structure (within ``cutoff`` Å)
but are absent in the WT structure are counted and classified.  Each new
contact contributes a type-specific weight to the final score:

    contact type    weight
    ----------      ------
    vdw             0.2   (van-der-Waals / generic)
    sb              2.0   (salt bridge: K/R/H ↔ D/E)
    pc              1.0   (pi–cation: aromatic ↔ K/R)
    ps              0.7   (pi–pi stacking)
    ts              0.7   (T-shaped pi)

These weights reproduce the empirical scores in the serial-mutagenesis CSV
files (verified against the lmna_msa_modified and mapre1_baseline datasets).

Usage
-----
    from phosphoswitch_evolution.scoring import contact_delta_score, parse_structure

    wt_atoms   = parse_structure("wt.pdb")
    ph_atoms   = parse_structure("phospho.pdb")
    result     = contact_delta_score(wt_atoms, ph_atoms)
    print(result["score"], result["contact_delta_count"])
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Contact-type weight table (empirically calibrated against serial mutagenesis
# CSV files)
# ---------------------------------------------------------------------------

CONTACT_WEIGHTS: Dict[str, float] = {
    "vdw": 0.2,   # generic van-der-Waals
    "sb":  2.0,   # salt bridge
    "pc":  1.0,   # pi–cation
    "ps":  0.7,   # pi–pi stacking
    "ts":  0.7,   # T-shaped pi interaction
}

# Default Cβ–Cβ contact cutoff (Å)
DEFAULT_CUTOFF: float = 8.0

# Amino acid classification sets (single-letter codes)
_POSITIVE = frozenset("KRH")
_NEGATIVE = frozenset("DE")
_AROMATIC = frozenset("FYWH")
_CHARGED  = _POSITIVE | _NEGATIVE

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Coord      = Tuple[float, float, float]
AtomRecord = Dict[str, Coord]     # residue_number (int) -> {"CA": xyz, "CB": xyz}
StructureData = Tuple[str, AtomRecord]  # (sequence, atoms)


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

def parse_structure(path: str | Path) -> StructureData:
    """Parse a ColabFold PDB file and return (sequence, atom_positions).

    Extracts the one-letter sequence from SEQRES or ATOM records and the
    Cα/Cβ positions for each residue.

    Parameters
    ----------
    path:
        Path to the PDB file (output of ColabFold, model rank 1 is preferred).

    Returns
    -------
    (sequence, residues)
        sequence — one-letter AA sequence inferred from ATOM records
        residues — dict mapping 1-indexed residue number to
                   ``{"CA": (x,y,z), "CB": (x,y,z)}``
    """
    _aa3to1 = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        "MSE": "M",  # selenomethionine
    }

    residues: AtomRecord = {}
    resnum_to_aa: Dict[int, str] = {}

    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom = line[12:16].strip()
            resname = line[17:20].strip()
            if atom.startswith("H") or atom[0].isdigit():
                continue
            try:
                resnum = int(line[22:26].strip())
                xyz: Coord = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except (ValueError, IndexError):
                continue

            if resname in _aa3to1:
                resnum_to_aa[resnum] = _aa3to1[resname]

            if atom in ("CA", "CB"):
                if resnum not in residues:
                    residues[resnum] = {}
                residues[resnum][atom] = xyz

    # Build sequence from ordered residue numbers
    sorted_resnums = sorted(resnum_to_aa)
    sequence = "".join(resnum_to_aa[r] for r in sorted_resnums)

    return sequence, residues


# ---------------------------------------------------------------------------
# Contact detection and classification
# ---------------------------------------------------------------------------

def _dist(a: Coord, b: Coord) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _classify_contact(aa_i: str, aa_j: str) -> str:
    """Return the contact type for a pair of amino acids.

    Classification priority: salt bridge > pi-cation > pi-pi > generic vdw.
    """
    # Salt bridge: one positive (K/R/H) and one negative (D/E)
    if (aa_i in _POSITIVE and aa_j in _NEGATIVE) or \
       (aa_i in _NEGATIVE and aa_j in _POSITIVE):
        return "sb"

    # Pi–cation: aromatic ↔ K/R (H is both aromatic and cationic; treat as pi)
    if (aa_i in frozenset("FYW") and aa_j in frozenset("KR")) or \
       (aa_j in frozenset("FYW") and aa_i in frozenset("KR")):
        return "pc"

    # Pi–pi stacking (both aromatic, not H)
    _pi_aro = frozenset("FYWH")
    if aa_i in _pi_aro and aa_j in _pi_aro:
        # H-H treated as T-shaped, aromatic-aromatic as stacking
        if aa_i == "H" or aa_j == "H":
            return "ts"
        return "ps"

    # Generic van-der-Waals
    return "vdw"


def find_contacts(
    residues: AtomRecord,
    sequence: str,
    cutoff: float = DEFAULT_CUTOFF,
    min_seq_sep: int = 4,
) -> Dict[Tuple[int, int], str]:
    """Find all Cβ–Cβ contacts within *cutoff* Å.

    Parameters
    ----------
    residues:
        Atom positions from :func:`parse_structure`.
    sequence:
        One-letter sequence (position i = sequence[i-1]).
    cutoff:
        Distance threshold in Å.
    min_seq_sep:
        Minimum sequence separation to avoid trivial backbone contacts.

    Returns
    -------
    dict mapping ``(i, j)`` pairs (i < j, 1-indexed) to contact type string.
    """
    contacts: Dict[Tuple[int, int], str] = {}
    resnums = sorted(residues)

    for idx_i, i in enumerate(resnums):
        if i < 1 or i > len(sequence):
            continue
        aa_i = sequence[i - 1]
        # Use CB if available (Cα fallback for Gly)
        ref_i = residues[i].get("CB") or residues[i].get("CA")
        if ref_i is None:
            continue

        for j in resnums[idx_i + 1 :]:
            if j - i < min_seq_sep:
                continue
            if j < 1 or j > len(sequence):
                continue
            aa_j = sequence[j - 1]
            ref_j = residues[j].get("CB") or residues[j].get("CA")
            if ref_j is None:
                continue
            if _dist(ref_i, ref_j) <= cutoff:
                contacts[(i, j)] = _classify_contact(aa_i, aa_j)

    return contacts


# ---------------------------------------------------------------------------
# Delta-score computation
# ---------------------------------------------------------------------------

def contact_delta_score(
    wt_data: StructureData,
    phospho_data: StructureData,
    cutoff: float = DEFAULT_CUTOFF,
    min_seq_sep: int = 4,
) -> Dict:
    """Compute the contact-delta score between WT and phospho structures.

    The score reflects new Cβ contacts that appear in the phospho structure
    but are absent in the WT.  Each new contact contributes a type-specific
    weight (see :data:`CONTACT_WEIGHTS`).

    Parameters
    ----------
    wt_data:
        Parsed WT structure from :func:`parse_structure`.
    phospho_data:
        Parsed phospho structure from :func:`parse_structure`.
    cutoff:
        Cβ–Cβ distance threshold in Å (default 8.0).
    min_seq_sep:
        Minimum sequence separation between residues (default 4).

    Returns
    -------
    dict with keys:
        score               — weighted sum of new contacts
        contact_delta_score — alias for score
        contact_delta_count — number of new contacts
        per_type_counts     — dict of {contact_type: count}
        new_contacts        — list of (i, j, type) tuples
        lost_contacts       — list of (i, j, type) for contacts lost on phospho
    """
    wt_seq, wt_res     = wt_data
    ph_seq, ph_res     = phospho_data

    # Use the WT sequence as reference (phospho may carry modified residue)
    seq = wt_seq if wt_seq else ph_seq

    wt_contacts = find_contacts(wt_res, seq, cutoff=cutoff, min_seq_sep=min_seq_sep)
    ph_contacts = find_contacts(ph_res, seq, cutoff=cutoff, min_seq_sep=min_seq_sep)

    new_contact_pairs   = set(ph_contacts) - set(wt_contacts)
    lost_contact_pairs  = set(wt_contacts) - set(ph_contacts)

    per_type: Dict[str, int] = {}
    total_score = 0.0

    new_contact_list: List[Tuple[int, int, str]] = []
    for pair in sorted(new_contact_pairs):
        ctype = ph_contacts[pair]
        per_type[ctype] = per_type.get(ctype, 0) + 1
        total_score += CONTACT_WEIGHTS.get(ctype, CONTACT_WEIGHTS["vdw"])
        new_contact_list.append((pair[0], pair[1], ctype))

    lost_contact_list: List[Tuple[int, int, str]] = [
        (p[0], p[1], wt_contacts[p]) for p in sorted(lost_contact_pairs)
    ]

    score = round(total_score, 6)

    return {
        "score":               score,
        "contact_delta_score": score,
        "contact_delta_count": len(new_contact_list),
        "per_type_counts":     per_type,
        "new_contacts":        new_contact_list,
        "lost_contacts":       lost_contact_list,
    }
