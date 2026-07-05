"""
mechanism.py — Analytical phosphate-binding model for phosphoswitch scoring.

Computes how well a given protein sequence can coordinate a phosphate group
when threaded onto two alternative backbone conformations:

  - Hairpin backbone  (phospho_pulled.pdb): the pulled/folded conformation
  - Straight backbone (phospho.pdb):        the extended helix conformation

The two scoring modes are:

Bidirectional
    Selects whichever direction (hairpin vs straight) yields the larger
    phosphate-binding differential.  Switch magnitude = |HP_score - ST_score|.
    Used when the target direction is not pre-specified.

Directional
    Explicitly optimises for a single direction:
      - 'hairpin'  (H2): phosphorylation favours the hairpin state
      - 'straight' (H1): phosphorylation favours the straight/helix state
    Uses a quality score that rewards target-state binding and penalises
    decoy-state binding together with composition constraints.

Amino-acid scoring table
    PHOS_BINDERS  — residues that coordinate phosphate (K, R, H, S, T, Y, N, Q)
                    scored by distance-dependent weights
    PHOS_REPELLERS — residues that electrostatically repel phosphate (D, E)
                    apply a negative score within their cutoff distance

Usage
-----
    from phosphoswitch_evolution.mechanism import parse_pdb, bidirectional_score

    parsed_hp = parse_pdb("backbone/stateA_phospho_pulled.pdb")
    parsed_st = parse_pdb("backbone/stateA_phospho.pdb")
    result = bidirectional_score(sequence, parsed_hp, parsed_st)
    print(result["switch_magnitude"], result["direction"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Amino-acid interaction parameters
# ---------------------------------------------------------------------------

#: Residues that can donate to or H-bond with a phosphate group.
#: Values: (cutoff_Angstrom, per-contact_weight)
PHOS_BINDERS: Dict[str, Tuple[float, float]] = {
    "K": (10.0, 1.5),
    "R": (11.0, 2.0),
    "H": (9.0, 0.8),
    "S": (6.0, 0.5),
    "T": (6.0, 0.5),
    "Y": (10.0, 0.5),
    "N": (8.0, 0.3),
    "Q": (9.0, 0.3),
}

#: Residues that electrostatically repel the phosphate dianion.
#: Values: (cutoff_Angstrom, per-contact_weight)  — weights are negative.
PHOS_REPELLERS: Dict[str, Tuple[float, float]] = {
    "D": (8.0, -2.0),
    "E": (9.0, -2.0),
}

# Composition guardrails used by the directional scorer
MAX_TOTAL_CHARGED: int = 18
MAX_TOTAL_KR: int = 12

# Region definitions (1-indexed residue numbers matching the backbone PDBs)
HAIRPIN_TAIL: List[int] = list(range(44, 60))
PHOS_REGION: List[int] = list(range(22, 39))


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

Coord = Tuple[float, float, float]
ResidueAtoms = Dict[int, Dict[str, Coord]]
ParsedBackbone = Tuple[Optional[Coord], ResidueAtoms]


def parse_pdb(path: str | Path) -> ParsedBackbone:
    """Parse a backbone PDB file and extract the phosphate centroid and Cα/Cβ positions.

    Parameters
    ----------
    path:
        Path to a PDB file containing the protein backbone with an explicit
        phosphate group (HETATM or ATOM records for P, OP1, OP2, OP3 atoms).

    Returns
    -------
    (p_centroid, residues)
        p_centroid — centroid of all phosphorus/oxygen-of-phosphate atoms, or
                     None if the file contains no phosphate atoms.
        residues   — dict mapping residue number to {"CA": xyz, "CB": xyz}.
    """
    phos_atoms: List[Coord] = []
    residues: ResidueAtoms = {}

    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom = line[12:16].strip()
            try:
                resnum = int(line[22:26].strip())
                xyz: Coord = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except (ValueError, IndexError):
                continue

            # Collect phosphate atoms to build the centroid
            if atom in ("P", "P1", "O1P", "O2P", "O3P", "OP1", "OP2", "OP3"):
                phos_atoms.append(xyz)
                continue

            # Skip hydrogen and atom names starting with a digit
            if atom.startswith("H") or atom[0].isdigit():
                continue

            # Retain only backbone Cα and Cβ
            if atom in ("CA", "CB"):
                if resnum not in residues:
                    residues[resnum] = {}
                residues[resnum][atom] = xyz

    if phos_atoms:
        n = len(phos_atoms)
        p_centroid: Optional[Coord] = (
            sum(c[0] for c in phos_atoms) / n,
            sum(c[1] for c in phos_atoms) / n,
            sum(c[2] for c in phos_atoms) / n,
        )
    else:
        p_centroid = None

    return p_centroid, residues


# ---------------------------------------------------------------------------
# Distance and per-residue scoring
# ---------------------------------------------------------------------------


def _dist(a: Coord, b: Coord) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _aa_score(aa: str, d: float) -> Tuple[float, str]:
    """Return (score_contribution, category) for amino acid *aa* at distance *d*
    from the phosphate centroid.

    Distance-dependent weighting for binders:
        d ≤ 4 Å  → weight × 1.5
        d ≤ 6 Å  → weight × 1.2
        d ≤ cutoff → weight × 1.0
    """
    if aa in PHOS_BINDERS:
        cutoff, weight = PHOS_BINDERS[aa]
        if d <= cutoff:
            category = "donor" if aa in "KR" else "h_bond"
            if d <= 4.0:
                return weight * 1.5, category
            elif d <= 6.0:
                return weight * 1.2, category
            else:
                return weight * 1.0, category
    elif aa in PHOS_REPELLERS:
        cutoff, weight = PHOS_REPELLERS[aa]
        if d <= cutoff:
            return weight, "repeller"
    return 0.0, "none"


def count_interactions(
    seq: str,
    p_centroid: Optional[Coord],
    residues: ResidueAtoms,
) -> Dict:
    """Compute phosphate-binding interactions for *seq* threaded onto a backbone.

    Parameters
    ----------
    seq:
        Protein sequence (1-indexed; residue *i* = seq[i-1]).
    p_centroid:
        3-D centroid of the phosphate group from :func:`parse_pdb`.
    residues:
        Cα/Cβ atom positions per residue from :func:`parse_pdb`.

    Returns
    -------
    dict with keys: score, donors, h_bonds, repellers,
                    tail_contacts, phos_region_contacts, contacts
    """
    if p_centroid is None:
        return {
            "score": 0.0,
            "donors": 0,
            "h_bonds": 0,
            "repellers": 0,
            "tail_contacts": 0,
            "phos_region_contacts": 0,
            "contacts": [],
        }

    score = donors = h_bonds = repellers = tail = phosreg = 0
    contacts: List[Tuple] = []

    for pos, atoms in residues.items():
        if pos < 1 or pos > len(seq) or pos == 30:
            # Position 30 is fixed/proline in the LMNA template; skip
            continue
        seq_aa = seq[pos - 1]
        ref = atoms.get("CB") or atoms.get("CA")
        if ref is None:
            continue
        d = _dist(ref, p_centroid)
        s, cat = _aa_score(seq_aa, d)
        if cat == "none":
            continue
        score += s
        contacts.append((pos, seq_aa, round(d, 1), cat))
        if cat == "donor":
            donors += 1
        elif cat == "h_bond":
            h_bonds += 1
        elif cat == "repeller":
            repellers += 1
        if pos in HAIRPIN_TAIL:
            tail += 1
        if pos in PHOS_REGION:
            phosreg += 1

    return {
        "score": round(score, 2),
        "donors": donors,
        "h_bonds": h_bonds,
        "repellers": repellers,
        "tail_contacts": tail,
        "phos_region_contacts": phosreg,
        "contacts": contacts,
    }


# ---------------------------------------------------------------------------
# Bidirectional scorer
# ---------------------------------------------------------------------------


def bidirectional_score(
    seq: str,
    parsed_hairpin: ParsedBackbone,
    parsed_straight: ParsedBackbone,
) -> Dict:
    """Score a sequence for phosphate-induced conformational switching in *either* direction.

    The algorithm evaluates both backbone conformations and selects whichever
    shows the larger phosphate-binding differential.  The score is:

        switch_magnitude = |HP_score − ST_score|
        effective_score  = switch_magnitude
                           − 3.0  (if no K/R donors in the favoured state)
                           − 0.3 × total_repellers

    Parameters
    ----------
    seq:
        Protein sequence string (one-letter codes).
    parsed_hairpin:
        Output of :func:`parse_pdb` for the hairpin/pulled backbone.
    parsed_straight:
        Output of :func:`parse_pdb` for the straight/helix backbone.

    Returns
    -------
    dict with keys:
        switch_magnitude, direction, effective_score,
        hairpin_score, hairpin_donors, hairpin_h_bonds, hairpin_repellers,
        hairpin_tail_contacts,
        straight_score, straight_donors, straight_h_bonds, straight_repellers,
        straight_phos_contacts,
        hairpin_contacts_str, straight_contacts_str
    """
    hp_p, hp_r = parsed_hairpin
    st_p, st_r = parsed_straight

    hp = count_interactions(seq, hp_p, hp_r)
    st = count_interactions(seq, st_p, st_r)

    diff = hp["score"] - st["score"]

    if diff > 0:
        direction = "favors_HAIRPIN"
    elif diff < 0:
        direction = "favors_STRAIGHT"
    else:
        direction = "neutral"

    abs_mag = abs(diff)
    score = abs_mag

    # Penalise if no strong ionic donors in the favoured state
    favoured_donors = max(hp["donors"], st["donors"])
    if favoured_donors == 0:
        score -= 3.0

    # Penalise all repellers (guards against stacking D/E for cheap magnitude)
    score -= 0.3 * (hp["repellers"] + st["repellers"])

    def contacts_str(contacts: List) -> str:
        return " ".join(f"{p}{a}({d})" for p, a, d, _c in contacts)

    return {
        "switch_magnitude": round(abs_mag, 2),
        "direction": direction,
        "effective_score": round(score, 2),
        "hairpin_score": hp["score"],
        "hairpin_donors": hp["donors"],
        "hairpin_h_bonds": hp["h_bonds"],
        "hairpin_repellers": hp["repellers"],
        "hairpin_tail_contacts": hp["tail_contacts"],
        "straight_score": st["score"],
        "straight_donors": st["donors"],
        "straight_h_bonds": st["h_bonds"],
        "straight_repellers": st["repellers"],
        "straight_phos_contacts": st["phos_region_contacts"],
        "hairpin_contacts_str": contacts_str(hp["contacts"]),
        "straight_contacts_str": contacts_str(st["contacts"]),
    }


# ---------------------------------------------------------------------------
# Directional scorer
# ---------------------------------------------------------------------------


def directional_score(
    seq: str,
    parsed_target: ParsedBackbone,
    parsed_decoy: ParsedBackbone,
    target_name: str,
) -> Dict:
    """Score a sequence for switching in a *specified* direction.

    Quality score:
        quality = 2.5 × target_donors
                  − 2.0 × decoy_donors
                  + 0.4 × target_h_bonds
                  − 0.5 × (target_repellers + decoy_repellers)

    Composition constraints (hard penalties):
        total charged (K+R+D+E) > MAX_TOTAL_CHARGED → −5 per excess
        total K+R                > MAX_TOTAL_KR      → −5 per excess
        no donors in target state                     → −5

    Parameters
    ----------
    seq:
        Protein sequence string.
    parsed_target:
        Backbone PDB data for the *desired* phosphorylated state.
    parsed_decoy:
        Backbone PDB data for the state that should NOT be stabilised.
    target_name:
        Either ``"HAIRPIN"`` (H2 switch) or ``"STRAIGHT"`` (H1 switch).

    Returns
    -------
    dict with keys:
        quality_score, target_state, donor_diff,
        target_donors, target_h_bonds, target_repellers, target_score,
        target_tail_contacts, target_phos_region_contacts,
        decoy_donors, decoy_h_bonds, decoy_repellers, decoy_score,
        n_total_charged, n_KR, n_DE,
        target_contacts_str, decoy_contacts_str
    """
    tgt_p, tgt_r = parsed_target
    dec_p, dec_r = parsed_decoy

    target = count_interactions(seq, tgt_p, tgt_r)
    decoy = count_interactions(seq, dec_p, dec_r)

    donor_diff = target["donors"] - decoy["donors"]

    quality = (
        2.5 * target["donors"]
        - 2.0 * decoy["donors"]
        + 0.4 * target["h_bonds"]
        - 0.5 * (target["repellers"] + decoy["repellers"])
    )

    n_K = seq.count("K")
    n_R = seq.count("R")
    n_D = seq.count("D")
    n_E = seq.count("E")
    n_charged = n_K + n_R + n_D + n_E
    n_KR = n_K + n_R

    if n_charged > MAX_TOTAL_CHARGED:
        quality -= 5.0 * (n_charged - MAX_TOTAL_CHARGED)
    if n_KR > MAX_TOTAL_KR:
        quality -= 5.0 * (n_KR - MAX_TOTAL_KR)
    if target["donors"] == 0:
        quality -= 5.0

    def contacts_str(contacts: List) -> str:
        return " ".join(f"{p}{a}({d})" for p, a, d, _c in contacts)

    return {
        "quality_score": round(quality, 2),
        "target_state": target_name,
        "donor_diff": donor_diff,
        "target_donors": target["donors"],
        "target_h_bonds": target["h_bonds"],
        "target_repellers": target["repellers"],
        "target_score": target["score"],
        "target_tail_contacts": target["tail_contacts"],
        "target_phos_region_contacts": target["phos_region_contacts"],
        "decoy_donors": decoy["donors"],
        "decoy_h_bonds": decoy["h_bonds"],
        "decoy_repellers": decoy["repellers"],
        "decoy_score": decoy["score"],
        "n_total_charged": n_charged,
        "n_KR": n_KR,
        "n_DE": n_D + n_E,
        "target_contacts_str": contacts_str(target["contacts"]),
        "decoy_contacts_str": contacts_str(decoy["contacts"]),
    }
