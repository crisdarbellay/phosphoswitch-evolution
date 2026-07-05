# phosphoswitch-evolution

> **Part of the [Proteus Design](https://github.com/cdarbellay) framework** — backbone-explicit,
> thermodynamically grounded multi-state protein design. Named after Proteus, the Greek sea god
> who could change his shape at will.

Computational *in silico* directed evolution of phosphoswitches.

This module implements the iterative single-mutant scan used to explore which
mutations could amplify phosphorylation-driven conformational switching in
**LMNA** (Y45, Src kinase) and **MAPRE1** (Y247, Src kinase).

> **Note on module boundaries.** This package is a standalone analysis tool.
> It is separate from the multi-state protein design pipeline
> (`phospho_switch_pipeline`). The two modules address complementary
> questions: directed evolution asks *which sequence changes amplify the
> switch?*; multi-state design asks *how do we engineer a stable two-state
> protein?* The evolved sequences produced here informed the design logic of
> the multi-state module even though they were not directly expressed.

---

## Biology background

### Why directed evolution?

A phosphoswitch flips between two conformations when a tyrosine residue is
phosphorylated. The size of the conformational change — the **switch
magnitude** — depends on the sequence context surrounding the phosphosite.
We wanted to know: what is the theoretical ceiling for switch magnitude, and
which positions in the sequence matter most?

To answer this we performed *in silico* directed evolution: an iterative
greedy search over single-point mutations, evaluating each candidate
computationally rather than experimentally.

### What does `contact_delta_score` measure?

Each candidate sequence is folded by ColabFold in two states:

| State   | Description |
|---------|-------------|
| WT      | Unphosphorylated sequence |
| Phospho | Same sequence, phosphosite modelled with phospho-Tyr |

The contact-delta score counts **new Cβ contacts** that appear in the
phospho structure but are absent in the WT, weighted by contact type:

| Contact type | Weight | Meaning |
|---|---|---|
| `vdw`  | 0.20 | Generic van-der-Waals |
| `sb`   | 2.00 | Salt bridge (K/R/H ↔ D/E) |
| `pc`   | 1.00 | Pi–cation (aromatic ↔ K/R) |
| `ps`   | 0.70 | Pi–pi stacking |
| `ts`   | 0.70 | T-shaped pi interaction |

A higher score means phosphorylation creates more new contacts — a larger
structural rearrangement.

### Mechanism scoring: bidirectional phosphate binding

In parallel to the contact-delta scan we used an analytical backbone model
(`mechanism.py`) that scores a sequence directly against two pre-computed
backbone conformations without ColabFold:

- **Hairpin backbone** (`stateA_phospho_pulled.pdb`): the pulled/folded state
- **Straight backbone** (`stateA_phospho.pdb`): the extended helix state

```
switch_magnitude = |HP_score − ST_score|
HP_score / ST_score = Σ aa_weight(d) for each residue within cutoff of phosphate
```

The bidirectional scorer selects whichever direction (hairpin or straight)
yields the larger differential. This is implemented in `mechanism.py` and
was used in the `mech_iter_bidirectional.py` greedy-scan loop.

---

## Key results

### LMNA best hit (5 rounds, n_top=4)

```
Sequence mutations : L20P | D25R | L37E | A42P | T49P
contact_delta_score: 18.6
Cα-RMSD vs WT     : 17.27 Å
```

**Why prolines?** L20P and T49P introduce helix-breaking prolines that
destabilise the straight helix. This is what drives the large contact delta —
the phospho structure is forced to adopt an entirely different fold. However,
prolines also make the sequence biophysically unattractive for expression
(helix-breakers at every turn), so these evolved sequences were **not
directly cloned or expressed**.

**What they taught us:** The evolution revealed that positions 20, 25, 37,
42, and 49 are the key levers for amplifying switching in LMNA. The
multi-state design module used this spatial information — rather than the
actual mutations — to guide its constrained sequence search.

### MAPRE1 round-1 scan

MAPRE1 Y247 showed modest contact-delta scores in round-1 single-mutant
scans (score ≈ 1.6–1.8 for most single substitutions), consistent with
the smaller conformational change accessible from a less flexible scaffold.

---

## Package structure

```
phosphoswitch-evolution/
├── src/
│   └── phosphoswitch_evolution/
│       ├── __init__.py      — public API
│       ├── evolution.py     — greedy single-mutant scan, round management
│       ├── scoring.py       — contact_delta_score from ColabFold PDB pairs
│       ├── mechanism.py     — analytical phosphate-binding model (bidirectional
│       │                      and directional scorers)
│       └── io_utils.py      — FASTA I/O, scores.csv loading, round discovery
├── scripts/
│   ├── run_evolution.py     — CLI driver for a full evolution run
│   └── analyze_results.py  — CLI for post-hoc analysis and plotting
├── examples/
│   └── lmna_results.csv    — Summary of LMNA directed-evolution results
└── tests/
    └── test_mechanism.py   — Unit tests for the mechanism scorer
```

---

## Installation

```bash
git clone https://github.com/<you>/phosphoswitch-evolution.git
cd phosphoswitch-evolution
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9, NumPy, Pandas, Matplotlib, SciPy, tqdm.

---

## Quick start

### 1. Mechanism scoring (no ColabFold required)

```python
from phosphoswitch_evolution.mechanism import parse_pdb, bidirectional_score

# Parse backbone PDB files (stateA_phospho_pulled.pdb and stateA_phospho.pdb)
parsed_hp = parse_pdb("backbone/stateA_phospho_pulled.pdb")
parsed_st = parse_pdb("backbone/stateA_phospho.pdb")

# LMNA template sequence
seq = "ASSTPLSPTRITRLQEKEDLQELNRRLAVYIDRVRSEETENAGLRLRITESEEVVSREV"

result = bidirectional_score(seq, parsed_hp, parsed_st)
print(f"switch magnitude: {result['switch_magnitude']}")
print(f"direction:        {result['direction']}")
print(f"effective score:  {result['effective_score']}")
```

### 2. Prepare a directed-evolution run (LMNA, 5 rounds)

```bash
# Generate candidate FASTAs for round 1
python scripts/run_evolution.py \
    --target lmna \
    --rounds 5 \
    --n-top 4 \
    --out-dir ./lmna_evo \
    --dry-run
```

Then run ColabFold on `lmna_evo/round_01/candidates.fasta`, produce
`scores.csv`, and re-run without `--dry-run` to load the scores and proceed
to round 2.

### 3. Analyse existing results

```bash
python scripts/analyze_results.py \
    --root /path/to/serial_mutagenesis_results \
    --out  ./analysis \
    --top-n 10
```

Produces per-round summary tables, fitness-trajectory plots, and mutation
enrichment charts.

---

## Scoring pipeline details

### `contact_delta_score` (scoring.py)

```python
from phosphoswitch_evolution.scoring import parse_structure, contact_delta_score

wt_data    = parse_structure("wt_model.pdb")
phospho_data = parse_structure("phospho_model.pdb")
result = contact_delta_score(wt_data, phospho_data)

print(result["score"])               # weighted sum of new contacts
print(result["contact_delta_count"]) # count of new contacts
print(result["per_type_counts"])     # {"vdw": 9, "sb": 3, ...}
```

### `bidirectional_score` (mechanism.py)

Scores a sequence against two pre-computed backbone PDB files.
Selects whichever backbone shows the larger phosphate-binding differential.
Penalises sequences with no K/R donors or too many D/E repellers.

### `directional_score` (mechanism.py)

Forces the algorithm toward a specific direction (`"HAIRPIN"` or
`"STRAIGHT"`). Applies composition constraints (max 12 K+R, max 18 charged
residues) to produce physically plausible designs.

---

## Running the tests

```bash
pytest tests/ -v
```

The tests use synthetic PDB data so no external files are needed.

---

## Relationship to the multi-state design module

```
phosphoswitch_evolution (this repo)
  └─ explored: which mutations MAXIMISE switch magnitude?
  └─ produced: best sequences (L20P D25R L37E A42P T49P, score=18.6)
  └─ finding:  positions 20, 25, 37, 42, 49 are key levers in LMNA

phospho_switch_pipeline (separate repo)
  └─ uses:    spatial information from evolution to seed constrained design
  └─ asks:    how to engineer a STABLE two-state protein at those positions?
  └─ avoids:  the helix-breaking prolines found by evolution
```

The evolved sequences in this module are **not the final designed proteins**.
They are computational waypoints that revealed the sequence landscape.

---

## Citation

If you use this code please cite:

> Darby C. *et al.* (in preparation). Mechanism-guided directed evolution of
> phosphoswitch proteins using iterative in silico mutagenesis.

---

## License

MIT
