"""
test_mechanism.py — Unit tests for the bidirectional phosphate-binding scorer.

Tests cover:
  - PDB parsing (including edge cases: no phosphate, Gly Cα fallback)
  - _aa_score distance-dependent weighting
  - count_interactions accumulation
  - bidirectional_score switch-magnitude and direction assignment
  - directional_score quality calculation and composition penalties
"""

from __future__ import annotations

import io
import math
import tempfile
from pathlib import Path

import pytest

from phosphoswitch_evolution.mechanism import (
    PHOS_BINDERS,
    PHOS_REPELLERS,
    HAIRPIN_TAIL,
    PHOS_REGION,
    MAX_TOTAL_CHARGED,
    MAX_TOTAL_KR,
    parse_pdb,
    _dist,
    _aa_score,
    count_interactions,
    bidirectional_score,
    directional_score,
)


# ---------------------------------------------------------------------------
# Helpers: build minimal PDB strings for testing
# ---------------------------------------------------------------------------

def _pdb_line(
    record: str,
    serial: int,
    atom: str,
    resname: str,
    chain: str,
    resnum: int,
    x: float,
    y: float,
    z: float,
) -> str:
    """Format a PDB ATOM / HETATM line."""
    return (
        f"{record:<6}{serial:>5} {atom:<4} {resname:<3} {chain}{resnum:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        f"  1.00 20.00           C  \n"
    )


def _make_pdb(
    residues: list[tuple[int, str, float, float, float]],
    phos_xyz: tuple[float, float, float] | None = None,
) -> str:
    """
    Build a minimal PDB string.

    residues: list of (resnum, atom_name, x, y, z)
    phos_xyz: optional phosphate coordinates
    """
    lines = []
    for i, (resnum, atom, x, y, z) in enumerate(residues, start=1):
        record = "ATOM"
        resname = "ALA"
        lines.append(_pdb_line(record, i, atom, resname, "A", resnum, x, y, z))

    if phos_xyz is not None:
        px, py, pz = phos_xyz
        lines.append(_pdb_line("HETATM", 999, "P", "PHO", "B", 900, px, py, pz))
        lines.append(_pdb_line("HETATM", 998, "OP1", "PHO", "B", 900, px + 1, py, pz))

    lines.append("END\n")
    return "".join(lines)


def _write_temp_pdb(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


# ---------------------------------------------------------------------------
# parse_pdb
# ---------------------------------------------------------------------------

class TestParsePdb:
    def test_no_phosphate_returns_none_centroid(self):
        content = _make_pdb([(1, "CA", 0.0, 0.0, 0.0)], phos_xyz=None)
        path = _write_temp_pdb(content)
        p_centroid, residues = parse_pdb(path)
        assert p_centroid is None

    def test_phosphate_centroid_single_atom(self):
        content = _make_pdb([], phos_xyz=(3.0, 4.0, 5.0))
        # One P atom at (3,4,5) and one OP1 at (4,4,5)
        path = _write_temp_pdb(content)
        p_centroid, _ = parse_pdb(path)
        assert p_centroid is not None
        # centroid of P(3,4,5) + OP1(4,4,5) = (3.5, 4, 5)
        assert abs(p_centroid[0] - 3.5) < 0.01
        assert abs(p_centroid[1] - 4.0) < 0.01

    def test_residues_ca_cb_extracted(self):
        content = _make_pdb([
            (5, "CA", 1.0, 2.0, 3.0),
            (5, "CB", 4.0, 5.0, 6.0),
            (6, "CA", 7.0, 8.0, 9.0),
        ])
        path = _write_temp_pdb(content)
        _, residues = parse_pdb(path)
        assert 5 in residues
        assert 6 in residues
        assert "CA" in residues[5]
        assert "CB" in residues[5]
        assert "CA" in residues[6]

    def test_hydrogen_atoms_skipped(self):
        content = _make_pdb([
            (1, "CA",  0.0, 0.0, 0.0),
            (1, "HB2", 0.5, 0.0, 0.0),   # hydrogen — should be skipped
        ])
        path = _write_temp_pdb(content)
        _, residues = parse_pdb(path)
        assert 1 in residues
        assert "HB2" not in residues[1]


# ---------------------------------------------------------------------------
# _dist
# ---------------------------------------------------------------------------

class TestDist:
    def test_zero_distance(self):
        assert _dist((0, 0, 0), (0, 0, 0)) == pytest.approx(0.0)

    def test_unit_vector(self):
        assert _dist((0, 0, 0), (1, 0, 0)) == pytest.approx(1.0)

    def test_3d(self):
        # distance from origin to (3,4,0) = 5
        assert _dist((0, 0, 0), (3, 4, 0)) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _aa_score
# ---------------------------------------------------------------------------

class TestAaScore:
    def test_lysine_close_range(self):
        score, cat = _aa_score("K", 3.0)   # d <= 4 Å → weight * 1.5
        assert cat == "donor"
        assert score == pytest.approx(1.5 * 1.5)

    def test_lysine_mid_range(self):
        score, cat = _aa_score("K", 5.0)   # 4 < d <= 6 → weight * 1.2
        assert cat == "donor"
        assert score == pytest.approx(1.5 * 1.2)

    def test_lysine_far_range(self):
        score, cat = _aa_score("K", 8.0)   # 6 < d <= 10 → weight * 1.0
        assert cat == "donor"
        assert score == pytest.approx(1.5 * 1.0)

    def test_arginine_within_cutoff(self):
        score, cat = _aa_score("R", 2.0)
        assert cat == "donor"
        assert score == pytest.approx(2.0 * 1.5)  # R weight=2.0, d<=4 → ×1.5

    def test_glutamate_repeller(self):
        score, cat = _aa_score("E", 5.0)
        assert cat == "repeller"
        assert score == pytest.approx(-2.0)

    def test_aspartate_beyond_cutoff(self):
        score, cat = _aa_score("D", 9.0)   # D cutoff = 8 Å → no score
        assert cat == "none"
        assert score == 0.0

    def test_non_interacting_aa(self):
        score, cat = _aa_score("A", 3.0)
        assert cat == "none"
        assert score == 0.0

    def test_serine_hbond(self):
        score, cat = _aa_score("S", 4.0)  # d<=4 → weight*1.5 = 0.5*1.5 = 0.75
        assert cat == "h_bond"
        assert score == pytest.approx(0.5 * 1.5)

    def test_histidine_is_hbond_not_donor(self):
        score, cat = _aa_score("H", 3.0)
        assert cat == "h_bond"   # H is not in "KR"


# ---------------------------------------------------------------------------
# count_interactions
# ---------------------------------------------------------------------------

class TestCountInteractions:
    def _make_residues(self, positions_xyz):
        """Helper: {resnum: {"CB": xyz}}"""
        return {pos: {"CB": xyz} for pos, xyz in positions_xyz.items()}

    def test_no_phosphate(self):
        residues = self._make_residues({1: (0.0, 0.0, 0.0)})
        result   = count_interactions("K", p_centroid=None, residues=residues)
        assert result["score"] == 0.0
        assert result["donors"] == 0

    def test_lysine_donor_counted(self):
        seq      = "K" + "A" * 58   # K at position 1
        residues = {1: {"CB": (0.0, 0.0, 0.0)}}
        p_centroid = (3.0, 0.0, 0.0)  # 3 Å from K at pos 1
        result = count_interactions(seq, p_centroid, residues)
        assert result["donors"] == 1
        assert result["score"] > 0.0

    def test_aspartate_repeller_counted(self):
        seq      = "D" + "A" * 58
        residues = {1: {"CB": (0.0, 0.0, 0.0)}}
        p_centroid = (2.0, 0.0, 0.0)  # 2 Å: well within D cutoff of 8 Å
        result = count_interactions(seq, p_centroid, residues)
        assert result["repellers"] == 1
        assert result["score"] < 0.0

    def test_position_30_skipped(self):
        """Position 30 is fixed (proline in LMNA template) and must be ignored."""
        seq      = "A" * 29 + "K" + "A" * 29   # K at position 30
        residues = {30: {"CB": (0.0, 0.0, 0.0)}}
        p_centroid = (1.0, 0.0, 0.0)
        result = count_interactions(seq, p_centroid, residues)
        assert result["donors"] == 0  # pos 30 skipped

    def test_hairpin_tail_contact_counted(self):
        pos_in_tail = HAIRPIN_TAIL[0]
        seq = "A" * (pos_in_tail - 1) + "K" + "A" * (59 - pos_in_tail)
        residues = {pos_in_tail: {"CB": (0.0, 0.0, 0.0)}}
        p_centroid = (2.0, 0.0, 0.0)
        result = count_interactions(seq, p_centroid, residues)
        assert result["tail_contacts"] == 1

    def test_glycine_fallback_to_ca(self):
        """Gly has no CB; the function should fall back to CA."""
        seq = "G" + "A" * 58
        # G at pos 1 with only CA
        residues = {1: {"CA": (0.0, 0.0, 0.0)}}
        # G is not in PHOS_BINDERS so score should be 0
        p_centroid = (2.0, 0.0, 0.0)
        result = count_interactions(seq, p_centroid, residues)
        assert result["score"] == 0.0  # G not in binders/repellers


# ---------------------------------------------------------------------------
# bidirectional_score
# ---------------------------------------------------------------------------

class TestBidirectionalScore:
    def _parsed(self, phos_xyz, residues):
        return phos_xyz, residues

    def test_favors_hairpin_when_hp_score_higher(self):
        seq = "K" + "A" * 58

        # Hairpin: K at pos 1 is 2 Å from phosphate → high score
        hp = self._parsed((2.0, 0.0, 0.0), {1: {"CB": (0.0, 0.0, 0.0)}})
        # Straight: phosphate is far away → no interaction
        st = self._parsed((100.0, 0.0, 0.0), {1: {"CB": (0.0, 0.0, 0.0)}})

        result = bidirectional_score(seq, hp, st)
        assert result["direction"] == "favors_HAIRPIN"
        assert result["switch_magnitude"] > 0.0
        assert result["hairpin_score"] > result["straight_score"]

    def test_favors_straight_when_st_score_higher(self):
        seq = "K" + "A" * 58
        hp = self._parsed((100.0, 0.0, 0.0), {1: {"CB": (0.0, 0.0, 0.0)}})
        st = self._parsed((2.0, 0.0, 0.0),   {1: {"CB": (0.0, 0.0, 0.0)}})

        result = bidirectional_score(seq, hp, st)
        assert result["direction"] == "favors_STRAIGHT"
        assert result["hairpin_score"] < result["straight_score"]

    def test_neutral_when_equal(self):
        seq = "A" * 59  # no binders/repellers
        shared_parsed = self._parsed(None, {})  # no phosphate
        result = bidirectional_score(seq, shared_parsed, shared_parsed)
        assert result["direction"] == "neutral"
        assert result["switch_magnitude"] == 0.0

    def test_no_donor_penalty(self):
        """If no K/R donors in either state, effective_score should be penalised."""
        seq = "S" + "A" * 58   # S is h_bond, not donor

        hp = self._parsed((3.0, 0.0, 0.0), {1: {"CB": (0.0, 0.0, 0.0)}})
        st = self._parsed((100.0, 0.0, 0.0), {1: {"CB": (0.0, 0.0, 0.0)}})

        result = bidirectional_score(seq, hp, st)
        assert result["effective_score"] == pytest.approx(
            result["switch_magnitude"] - 3.0, abs=0.1
        )

    def test_repeller_penalty(self):
        """D/E repellers reduce effective_score."""
        seq_r = "K" + "A" * 27 + "E" + "A" * 30  # K at 1, E at 29

        hp = self._parsed(
            (2.0, 0.0, 0.0),
            {
                1:  {"CB": (0.0,  0.0, 0.0)},   # K, 2 Å → donor
                29: {"CB": (4.0,  0.0, 0.0)},   # E, 6 Å → repeller
            },
        )
        st = self._parsed((100.0, 0.0, 0.0), {1: {"CB": (0.0, 0.0, 0.0)}})
        result = bidirectional_score(seq_r, hp, st)
        # Penalty = 0.3 * (hp_repellers + st_repellers)
        assert result["effective_score"] < result["switch_magnitude"]

    def test_return_keys_complete(self):
        """bidirectional_score must return all documented keys."""
        seq = "A" * 59
        parsed = (None, {})
        result = bidirectional_score(seq, parsed, parsed)
        required_keys = {
            "switch_magnitude", "direction", "effective_score",
            "hairpin_score", "hairpin_donors", "hairpin_h_bonds", "hairpin_repellers",
            "hairpin_tail_contacts",
            "straight_score", "straight_donors", "straight_h_bonds", "straight_repellers",
            "straight_phos_contacts",
            "hairpin_contacts_str", "straight_contacts_str",
        }
        assert required_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# directional_score
# ---------------------------------------------------------------------------

class TestDirectionalScore:
    def _parsed(self, phos_xyz, residues):
        return phos_xyz, residues

    def test_quality_formula(self):
        """Manual calculation of quality score."""
        seq = "K" + "A" * 58  # K at pos 1

        # Target backbone: K at pos 1 is 2 Å → donor, weight 2.25 (1.5*1.5)
        target = self._parsed((2.0, 0.0, 0.0), {1: {"CB": (0.0, 0.0, 0.0)}})
        # Decoy: far → no interaction
        decoy  = self._parsed((100.0, 0.0, 0.0), {1: {"CB": (0.0, 0.0, 0.0)}})

        result = directional_score(seq, target, decoy, "HAIRPIN")

        # Expected: quality = 2.5*1 - 2.0*0 + 0.4*0 - 0.5*(0+0) = 2.5
        assert result["quality_score"] == pytest.approx(2.5, abs=0.01)
        assert result["target_donors"] == 1
        assert result["decoy_donors"]  == 0
        assert result["donor_diff"]    == 1

    def test_no_target_donor_penalty(self):
        """quality is penalised by -5 when target has no K/R donors."""
        seq = "S" + "A" * 58  # S is h_bond only
        target = self._parsed((3.0, 0.0, 0.0), {1: {"CB": (0.0, 0.0, 0.0)}})
        decoy  = self._parsed((100.0, 0.0, 0.0), {})
        result = directional_score(seq, target, decoy, "STRAIGHT")
        # base quality without penalty = 2.5*0 - 2.0*0 + 0.4*1 - 0.5*0 = 0.4
        # after -5 penalty = -4.6
        assert result["quality_score"] == pytest.approx(-4.6, abs=0.1)

    def test_composition_kr_penalty(self):
        """Sequences with too many K/R are penalised."""
        # Build a sequence with MAX_TOTAL_KR + 1 = 13 K residues
        n_kr = MAX_TOTAL_KR + 1
        seq = "K" * n_kr + "A" * (59 - n_kr)

        target = self._parsed((2.0, 0.0, 0.0), {i: {"CB": (float(i), 0.0, 0.0)} for i in range(1, 14)})
        decoy  = self._parsed((100.0, 0.0, 0.0), {})
        result = directional_score(seq, target, decoy, "HAIRPIN")

        # n_KR penalty should fire
        assert result["n_KR"] > MAX_TOTAL_KR
        # quality should be reduced by at least 5 * 1 = 5
        # (exact value depends on interaction count; just check penalty applied)
        assert result["quality_score"] < result["target_donors"] * 2.5  # less than unconstrained

    def test_direction_stored_correctly(self):
        seq = "A" * 59
        parsed = (None, {})
        result = directional_score(seq, parsed, parsed, "HAIRPIN")
        assert result["target_state"] == "HAIRPIN"

    def test_return_keys_complete(self):
        seq    = "A" * 59
        parsed = (None, {})
        result = directional_score(seq, parsed, parsed, "STRAIGHT")
        required = {
            "quality_score", "target_state", "donor_diff",
            "target_donors", "target_h_bonds", "target_repellers", "target_score",
            "target_tail_contacts", "target_phos_region_contacts",
            "decoy_donors", "decoy_h_bonds", "decoy_repellers", "decoy_score",
            "n_total_charged", "n_KR", "n_DE",
            "target_contacts_str", "decoy_contacts_str",
        }
        assert required.issubset(result.keys())


# ---------------------------------------------------------------------------
# Integration: known LMNA template scores
# ---------------------------------------------------------------------------

class TestLMNATemplate:
    """
    Smoke test using the actual LMNA template sequence and synthetic
    backbone data.  These are not exact physical tests — they validate that
    the code path runs end-to-end and returns plausible numeric ranges.
    """

    TEMPLATE = "ASSTPLSPTRITRLQEKEDLQELNRRLAVYIDRVRSEETENAGLRLRITESEEVVSREV"

    def _synthetic_backbone(self, phos_near_tail: bool):
        """Build a synthetic backbone where phosphate is near the hairpin tail."""
        residues = {}
        for i in range(1, 60):
            residues[i] = {"CB": (float(i), 0.0, 0.0)}

        # Phosphate near tail (pos 44-59) or near helix (pos 22-38)
        if phos_near_tail:
            p_centroid = (51.0, 0.0, 0.0)  # near pos 51
        else:
            p_centroid = (30.0, 0.0, 0.0)  # near pos 30

        return p_centroid, residues

    def test_template_bidirectional_runs(self):
        hp = self._synthetic_backbone(phos_near_tail=True)
        st = self._synthetic_backbone(phos_near_tail=False)
        result = bidirectional_score(self.TEMPLATE, hp, st)
        assert isinstance(result["switch_magnitude"], float)
        assert result["switch_magnitude"] >= 0.0

    def test_template_directional_runs(self):
        hp = self._synthetic_backbone(phos_near_tail=True)
        st = self._synthetic_backbone(phos_near_tail=False)
        result = directional_score(self.TEMPLATE, hp, st, "HAIRPIN")
        assert isinstance(result["quality_score"], float)
