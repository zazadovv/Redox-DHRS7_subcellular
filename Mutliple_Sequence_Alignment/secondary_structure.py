#!/usr/bin/env python3
"""
secondary_structure.py
======================

Secondary-structure assignment from a model's backbone hydrogen bonding, using
the Kabsch-Sander definition (the rule set DSSP implements).

Why this rather than backbone torsion angles: a torsion-angle classifier decides
each residue on its own phi/psi pair, so any extended stretch of backbone falls
into the beta window -- including disordered tails, which adopt extended and
polyproline-II conformations but pair with nothing. Assigning those as strand is
wrong, and it shows up exactly where it does most damage, at flexible termini.
A strand is defined by its bridge partner, so the assignment here only calls a
strand when the backbone hydrogen bonds that make a beta bridge are present.

Helices come from repeated i -> i+n turns; strands come from bridges between two
segments. Everything else is loop.

Model confidence (pLDDT, stored in the B-factor column of an AlphaFold model) is
read alongside the assignment so low-confidence regions can be reported or
masked by the caller.

    from secondary_structure import assign_secondary_structure
    result = assign_secondary_structure(open("model.pdb").read())
    result["kinds"]      # 'helix' | 'sheet' | 'loop' per residue
    result["sequence"]   # one-letter sequence
    result["plddt"]      # per-residue model confidence
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

HBOND_ENERGY_CUTOFF = -0.5      # kcal/mol, Kabsch & Sander
_COUPLING = 0.084 * 332.0       # q1*q2*f in the same units
MIN_HELIX_TURNS = 2             # a helix needs two consecutive n-turns

# Edge extension. Hydrogen bonding decides whether an element exists; the
# backbone torsions decide how far it runs. The terminal residue of a strand or
# helix often frays, so its own hydrogen bond is missing even though the backbone
# is still clearly in the right conformation and the residue is plainly part of
# the element. Those residues are added back, but only next to an element that
# hydrogen bonding already established -- an isolated stretch of extended
# backbone never becomes a strand this way.
MAX_EDGE_EXTENSION = 2          # residues added per end, at most
MIN_EDGE_PLDDT = 70.0           # do not extend into low-confidence regions
# Extension needs something real to grow from. A single hydrogen-bonded residue
# pair is an isolated beta bridge: a genuine structural contact, but not a strand
# on its own. Extending from one shows it in the track; the pairing still has to
# exist first, so a stretch of extended backbone with no partner -- a disordered
# tail, for instance -- is never drawn as a strand either way.
MIN_RUN_TO_EXTEND = {"helix": 4, "sheet": 1}
PHI_PSI_WINDOWS = {
    "helix": lambda phi, psi: -100.0 <= phi <= -30.0 and -90.0 <= psi <= 0.0,
    "sheet": lambda phi, psi: -180.0 <= phi <= -45.0 and (45.0 <= psi <= 180.0
                                                          or -180.0 <= psi <= -145.0),
}


def parse_backbone(pdb_text: str) -> List[Dict[str, Any]]:
    """Backbone N/CA/C/O coordinates, residue identity and pLDDT, first chain only."""
    residues: Dict[int, Dict[str, Any]] = {}
    order: List[int] = []
    chain = None
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            if line.startswith("ENDMDL"):
                break
            continue
        atom = line[12:16].strip()
        if atom not in ("N", "CA", "C", "O"):
            continue
        this_chain = line[21]
        if chain is None:
            chain = this_chain
        elif this_chain != chain:
            continue
        number = int(line[22:26])
        entry = residues.get(number)
        if entry is None:
            entry = {"position": number,
                     "residue_name": line[17:20].strip().upper(),
                     "atoms": {}, "plddt": None}
            residues[number] = entry
            order.append(number)
        entry["atoms"][atom] = np.array(
            (float(line[30:38]), float(line[38:46]), float(line[46:54])), dtype=float)
        if atom == "CA":
            try:
                entry["plddt"] = float(line[60:66])
            except ValueError:
                entry["plddt"] = None
    return [residues[n] for n in order if {"N", "CA", "C", "O"} <= set(residues[n]["atoms"])]


def _amide_hydrogens(residues: List[Dict[str, Any]]) -> List[Any]:
    """Amide H placed 1 A from N, opposite the preceding carbonyl. Proline and the
    first residue have no donor hydrogen."""
    hydrogens: List[Any] = [None] * len(residues)
    for i in range(1, len(residues)):
        if residues[i]["residue_name"] == "PRO":
            continue
        previous, current = residues[i - 1]["atoms"], residues[i]["atoms"]
        direction = previous["C"] - previous["O"]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        hydrogens[i] = current["N"] + direction / norm
    return hydrogens


def hydrogen_bond_map(residues: List[Dict[str, Any]]) -> np.ndarray:
    """bonded[donor, acceptor] -- N-H of `donor` hydrogen bonds to C=O of `acceptor`."""
    n = len(residues)
    bonded = np.zeros((n, n), dtype=bool)
    if n < 2:
        return bonded
    hydrogens = _amide_hydrogens(residues)
    ca = np.array([r["atoms"]["CA"] for r in residues])
    for donor in range(n):
        h = hydrogens[donor]
        if h is None:
            continue
        nitrogen = residues[donor]["atoms"]["N"]
        # Only residues whose CA is within reach can hydrogen bond.
        near = np.where(np.linalg.norm(ca - ca[donor], axis=1) < 9.0)[0]
        for acceptor in near:
            if abs(acceptor - donor) < 2:
                continue
            carbon = residues[acceptor]["atoms"]["C"]
            oxygen = residues[acceptor]["atoms"]["O"]
            r_on = np.linalg.norm(oxygen - nitrogen)
            r_ch = np.linalg.norm(carbon - h)
            r_oh = np.linalg.norm(oxygen - h)
            r_cn = np.linalg.norm(carbon - nitrogen)
            if min(r_on, r_ch, r_oh, r_cn) < 0.5:
                continue
            energy = _COUPLING * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn)
            if energy < HBOND_ENERGY_CUTOFF:
                bonded[donor, acceptor] = True
    return bonded


def _dihedral(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    n1 = float(np.linalg.norm(b1))
    if n1 < 1e-8:
        return float("nan")
    b1u = b1 / n1
    v = b0 - np.dot(b0, b1u) * b1u
    w = b2 - np.dot(b2, b1u) * b1u
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1u, v), w), np.dot(v, w))))


def backbone_torsions(residues: List[Dict[str, Any]]):
    """(phi, psi) per residue; None where the neighbouring atoms are missing."""
    n = len(residues)
    phi: List[Any] = [None] * n
    psi: List[Any] = [None] * n
    for i in range(n):
        atoms = residues[i]["atoms"]
        if i > 0:
            phi[i] = _dihedral(residues[i - 1]["atoms"]["C"], atoms["N"], atoms["CA"], atoms["C"])
        if i + 1 < n:
            psi[i] = _dihedral(atoms["N"], atoms["CA"], atoms["C"], residues[i + 1]["atoms"]["N"])
    return phi, psi


def _extend_edges(kinds, phi, psi, plddt) -> int:
    """Grow established helices and strands into frayed terminal residues.
    Returns how many residues were added."""
    n = len(kinds)
    added = 0
    starts = [i for i in range(n) if kinds[i] in PHI_PSI_WINDOWS and (i == 0 or kinds[i - 1] != kinds[i])]
    for start in starts:
        end = start
        while end + 1 < n and kinds[end + 1] == kinds[start]:
            end += 1
        kind = kinds[start]
        if (end - start + 1) < MIN_RUN_TO_EXTEND.get(kind, 1):
            continue
        matches = PHI_PSI_WINDOWS[kind]
        for step, edge in ((-1, start), (1, end)):
            position = edge
            for _ in range(MAX_EDGE_EXTENSION):
                position += step
                if not (0 <= position < n) or kinds[position] != "loop":
                    break
                a, b, c = phi[position], psi[position], plddt[position]
                if a is None or b is None or (c is not None and c < MIN_EDGE_PLDDT):
                    break
                if not matches(a, b):
                    break
                kinds[position] = kind
                added += 1
    return added


def assign_secondary_structure(pdb_text: str, extend_edges: bool = True) -> Dict[str, Any]:
    """Kabsch-Sander assignment -> 'helix' / 'sheet' / 'loop' per residue."""
    residues = parse_backbone(pdb_text)
    n = len(residues)
    result = {
        "method": "kabsch_sander_backbone_hydrogen_bonds",
        "sequence": "".join(THREE_TO_ONE.get(r["residue_name"], "X") for r in residues),
        "positions": [r["position"] for r in residues],
        "plddt": [r["plddt"] for r in residues],
        "kinds": ["loop"] * n,
    }
    if n < 4:
        return result

    bonded = hydrogen_bond_map(residues)

    def bond(i: int, j: int) -> bool:
        """C=O of i hydrogen bonds to N-H of j."""
        return 0 <= i < n and 0 <= j < n and bool(bonded[j, i])

    kinds = ["loop"] * n

    def apply_helix(turn: int) -> None:
        """Two consecutive n-turns mark residues i .. i+n-1 as helical."""
        turns = [bond(i, i + turn) for i in range(n)]
        for i in range(1, n - turn):
            if all(turns[i - 1 + k] for k in range(MIN_HELIX_TURNS)):
                for offset in range(turn):
                    if i + offset < n:
                        kinds[i + offset] = "helix"

    # Applied in order of increasing precedence, matching the DSSP hierarchy:
    # 3-10 and pi helices are weakest, a beta bridge outranks them, and a
    # 4-turn alpha helix outranks everything.
    apply_helix(5)
    apply_helix(3)

    # Strands: a residue is in a strand when it forms a beta bridge with a
    # partner. Parallel and antiparallel bridges have distinct bond patterns.
    for i in range(1, n - 1):
        for j in range(i + 3, n - 1):
            parallel = ((bond(i - 1, j) and bond(j, i + 1)) or
                        (bond(j - 1, i) and bond(i, j + 1)))
            antiparallel = ((bond(i, j) and bond(j, i)) or
                            (bond(i - 1, j + 1) and bond(j - 1, i + 1)))
            if parallel or antiparallel:
                kinds[i] = "sheet"
                kinds[j] = "sheet"

    apply_helix(4)

    if extend_edges:
        phi, psi = backbone_torsions(residues)
        result["edge_extended"] = _extend_edges(kinds, phi, psi, result["plddt"])
    result["bridged"] = _bridge_short_breaks(kinds)

    result["kinds"] = kinds
    return result


MIN_TORSION_HELIX_RUN = 4
MIN_TORSION_STRAND_RUN = 3
# Interruptions are left as they are. Closing a one-residue break would join two
# elements that the torsions genuinely separate -- a glycine swinging positive,
# or a residue sitting in the helical window in the middle of a strand -- and the
# result reads as one long element the backbone does not support. The assignment
# is meant to report the torsions, so a broken run is kept broken.
MAX_BRIDGED_BREAK = 0


def _bridge_short_breaks(kinds: List[str], max_break: int = MAX_BRIDGED_BREAK) -> int:
    """Close breaks of at most `max_break` residues inside one element. With the
    default allowance of zero this does nothing and the runs are left as the
    torsions describe them."""
    if max_break < 1:
        return 0
    bridged = 0
    i = 0
    while i < len(kinds):
        if kinds[i] != "loop":
            i += 1
            continue
        end = i
        while end + 1 < len(kinds) and kinds[end + 1] == "loop":
            end += 1
        length = end - i + 1
        before = kinds[i - 1] if i > 0 else None
        after = kinds[end + 1] if end + 1 < len(kinds) else None
        if length <= max_break and before is not None and before == after and before != "loop":
            for j in range(i, end + 1):
                kinds[j] = before
            bridged += length
        i = end + 1
    return bridged


def _suppress_short_runs(kinds: List[str], kind: str, minimum: int) -> None:
    start = 0
    while start < len(kinds):
        if kinds[start] != kind:
            start += 1
            continue
        end = start
        while end + 1 < len(kinds) and kinds[end + 1] == kind:
            end += 1
        if (end - start + 1) < minimum:
            for i in range(start, end + 1):
                kinds[i] = "loop"
        start = end + 1


def assign_from_torsion(pdb_text: str) -> Dict[str, Any]:
    """Secondary structure from backbone phi/psi alone.

    Each residue is classified on its own torsion pair, with runs shorter than
    four (helix) or three (strand) residues suppressed. This is the classical
    Ramachandran-window assignment. It needs no hydrogen positions and makes no
    assumption about hydrogen bonding, but because it judges residues in
    isolation it reports any extended backbone as strand, including tails that
    have no strand partner. Provided alongside the hydrogen-bond assignment so
    the two can be compared directly.
    """
    residues = parse_backbone(pdb_text)
    n = len(residues)
    result = {
        "method": "backbone_phi_psi_windows",
        "sequence": "".join(THREE_TO_ONE.get(r["residue_name"], "X") for r in residues),
        "positions": [r["position"] for r in residues],
        "plddt": [r["plddt"] for r in residues],
        "kinds": ["loop"] * n,
    }
    if n < 3:
        return result
    phi, psi = backbone_torsions(residues)
    kinds = []
    for i in range(n):
        a, b = phi[i], psi[i]
        if a is None or b is None:
            kinds.append("loop")
        elif PHI_PSI_WINDOWS["helix"](a, b):
            kinds.append("helix")
        elif PHI_PSI_WINDOWS["sheet"](a, b):
            kinds.append("sheet")
        else:
            kinds.append("loop")
    result["bridged"] = _bridge_short_breaks(kinds)
    _suppress_short_runs(kinds, "helix", MIN_TORSION_HELIX_RUN)
    _suppress_short_runs(kinds, "sheet", MIN_TORSION_STRAND_RUN)
    result["kinds"] = kinds
    return result


def kinds_by_position(pdb_text: str) -> Dict[int, str]:
    """Residue number -> assignment, for callers that index by model numbering."""
    data = assign_secondary_structure(pdb_text)
    return dict(zip(data["positions"], data["kinds"]))
