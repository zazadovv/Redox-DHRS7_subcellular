#!/usr/bin/env python3
"""
dhrs7_snapshot_figure.py
============================

Pure-Python reproduction of the DHRS7 **species-snapshot** publication figure.

This module is a faithful, line-for-line port of the alignment browser's
JavaScript renderer ``buildSpeciesSnapshotSvg`` (and every helper it calls) from
``gene_phylo_conservation_archive.py``. It reads the alignment *payload*
(the JSON the pipeline embeds in ``alignment_browser.html``) and draws the exact
same SVG the interactive "Species snapshot export" produces — with **no browser
and no third-party dependencies** (Python standard library only). This makes the
figure a transparent, reviewer-auditable artefact for the paper: every element
(coloured residue cells, secondary-structure ribbons, tick axis, conservation
row, the black human+zebrafish-vs-rodent frames, and the red active-trio
asterisks) is drawn by readable Python here.

What the figure shows (all definitions ported verbatim from the browser):

* **Rows** — the 5 species compared for the paper, in fixed order:
  mus_musculus, rattus_norvegicus, bos_taurus, danio_rerio, then
  homo_sapiens (the human reference row, highlighted). The full ortholog set is
  retrieved and aligned upstream; only these 5 are drawn here for the residue
  comparison.
* **Columns** — the "raw MUSCLE alignment (all residues)" scope (``selected_raw``)
  with all-gap columns removed; numbering uses human reference positions.
* **Secondary structure** — grey (#6b7280) cursive-helix / sheet-arrow / loop
  glyphs above each row, sourced from the per-species AlphaFold models
  (``alphafold_structure.comparative_secondary_structure.records[].mapped_ranges``,
  already mapped onto human reference positions) and, for the human row, from
  ``alphafold_structure.secondary_structure.ranges``.
* **Conservation row** — Clustal-style ``* : .`` symbols across the 5 rows.
* **Black frames** — columns where homo_sapiens and danio_rerio carry the
  identical residue but every selected rodent (mouse, rat) differs.
* **Red asterisk** — the subset of framed columns where bos_taurus ALSO shares
  that residue: identical across all three *active* species (human, zebrafish,
  bovine) while both inactive rodents diverge (candidate activity-linked sites).

Run standalone:  ``python dhrs7_snapshot_figure.py <alignment_browser.html> [out.svg]``
or import ``render_dhrs7_snapshot(payload)`` from the reproduction driver.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Constants (ported from the browser)
# --------------------------------------------------------------------------- #
GAP_CHARS = {"-", "."}
NON_CONSENSUS_RESIDUES = GAP_CHARS | {"X", "?"}

# Clustal-style conservation groups (JS SNAPSHOT_STRONG/WEAK_CONSERVATION_GROUPS)
STRONG_GROUPS = [set(g) for g in ("STA", "NEQK", "NHQK", "NDEQ", "QHRK",
                                  "MILV", "MILF", "HY", "FYW")]
WEAK_GROUPS = [set(g) for g in ("CSA", "ATV", "SAG", "STNK", "STPA", "SGND",
                                "SNDEQK", "NDEQHK", "NEQHRK", "FVLIM", "HFY")]

RODENT_CLADE_RE = re.compile(
    r"rodent|glires|muroidea|myomorph|murin|hystricomorph|sciuromorph|castorimorph", re.I)
RODENT_SPECIES_RE = re.compile(
    r"^(mus_|mus$|rattus|cricetulus|mesocricetus|peromyscus|microtus|cavia|"
    r"ictidomys|urocitellus|marmota|castor|jaculus|nannospalax|fukomys|"
    r"heterocephalus|chinchilla|octodon|dipodomys|perognathus|meriones|"
    r"psammomys|spermophilus|myodes|ondatra|sigmodon|apodemus|acomys|"
    r"grammomys|arvicanthis|mastomys)", re.I)

# Class -> inlined SVG presentation attributes (JS SNAPSHOT_SVG_STYLE_ATTRS).
# Illustrator / Inkscape ignore <style>, so the exported figure inlines these.
SNAPSHOT_SVG_STYLE_ATTRS: Dict[str, str] = {
    "title": 'font-family="Segoe UI, Tahoma, sans-serif" font-size="36" font-weight="700" fill="#18202a"',
    "subtitle": 'font-family="Segoe UI, Tahoma, sans-serif" font-size="21" font-weight="600" fill="#617083"',
    "legend-range": 'font-family="Segoe UI, Tahoma, sans-serif" font-size="28" font-weight="700" fill="#405774"',
    "legend-name": 'font-family="Segoe UI, Tahoma, sans-serif" font-size="28" font-weight="700" fill="#18202a"',
    "legend-ref": 'font-family="Segoe UI, Tahoma, sans-serif" font-size="28" font-weight="700" fill="#0f766e"',
    "legend-ref-bg": 'fill="#eefaf8"',
    "axis": 'stroke="#7b8188" stroke-width="3"',
    "tick": 'stroke="#7b8188" stroke-width="2.2"',
    "tick-label": 'font-family="Segoe UI, Tahoma, sans-serif" font-size="22" font-weight="500" fill="#6b7280"',
    "row-number": 'font-family="Segoe UI, Tahoma, sans-serif" font-size="22" font-weight="500" fill="#18202a"',
    "aa": 'font-family="Consolas, \'Courier New\', monospace" font-size="28" font-weight="700" fill="#18202a"',
    "snapshot-ss-loop": 'fill="none" stroke="#6b7280" stroke-width="2.2" stroke-linecap="round" opacity="0.5"',
    "snapshot-ss-helix": 'stroke="#6b7280" stroke-width="2.8"',
    "snapshot-ss-sheet": 'fill="#6b7280" opacity="0.86"',
    "snapshot-conservation-label": 'font-family="Segoe UI, Tahoma, sans-serif" font-size="24" font-weight="700" fill="#405774"',
    "snapshot-conservation-symbol": 'font-family="Consolas, \'Courier New\', monospace" font-size="30" font-weight="800" fill="#111827"',
    "snapshot-conservation-key": 'font-family="Segoe UI, Tahoma, sans-serif" font-size="22" font-weight="600" fill="#617083"',
    "snapshot-hd-frame": 'fill="none" stroke="#000000" stroke-width="3.6"',
    "snapshot-hd-tab": 'fill="#000000"',
    "snapshot-hd-trio-star": 'fill="none" stroke="#dc2626" stroke-width="2.8" stroke-linecap="round"',
}

# The 5 species compared in the paper figure, in row order (reference last).
DHRS7_SELECTED_ORDER = ["mus_musculus", "rattus_norvegicus", "bos_taurus", "danio_rerio"]
DHRS7_REFERENCE_SPECIES = "homo_sapiens"

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _f1(x: float) -> str:
    return f"{x:.1f}"


def _f2(x: float) -> str:
    return f"{x:.2f}"


def escape_xml(value: Any) -> str:
    return (str("" if value is None else value)
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))


def format_label(value: Any) -> str:
    return str(value if value else "Unassigned").replace("_", " ")


def is_informative(aa: Any) -> bool:
    return bool(aa) and str(aa).upper() not in NON_CONSENSUS_RESIDUES


def residue_at(row: Dict[str, Any], idx: int) -> str:
    seq = str(row.get("aligned_sequence") or "").upper()
    return seq[idx] if 0 <= idx < len(seq) else ""


def ungapped_length(row: Dict[str, Any]) -> Optional[int]:
    numeric = row.get("ungapped_length")
    try:
        n = int(numeric)
        if n > 0:
            return n
    except (TypeError, ValueError):
        pass
    seq = str(row.get("aligned_sequence") or "")
    if not seq:
        return None
    return len(seq.replace("-", "").replace(".", ""))


def is_rodent_row(row: Dict[str, Any]) -> bool:
    if not row:
        return False
    fields = " ".join(str(row.get(k) or "") for k in ("clade", "broad_clade", "taxonomy_level"))
    if RODENT_CLADE_RE.search(fields):
        return True
    return bool(RODENT_SPECIES_RE.search(str(row.get("species") or "").lower()))


# --------------------------------------------------------------------------- #
# Conservation row
# --------------------------------------------------------------------------- #

def _all_in_any_group(residues: List[str], groups: List[set]) -> bool:
    return any(all(aa in group for aa in residues) for group in groups)


def conservation_symbol(rows: List[Dict[str, Any]], idx: int) -> str:
    residues = [residue_at(r, idx) for r in rows]
    residues = [aa for aa in residues if is_informative(aa)]
    if len(residues) < 2:
        return " "
    if len(set(residues)) == 1:
        return "*"
    if _all_in_any_group(residues, STRONG_GROUPS):
        return ":"
    if _all_in_any_group(residues, WEAK_GROUPS):
        return "."
    return " "


def conservation_row_svg(rows, chunk, legend_x, matrix_x, grid_x, y, cell_width) -> str:
    if not isinstance(rows, list) or len(rows) < 2:
        return ""
    parts = [
        f'<text class="snapshot-conservation-label" x="{legend_x}" y="{y}">conservation</text>',
        f'<text class="row-number" x="{matrix_x + 20}" y="{y}" text-anchor="end"></text>',
    ]
    for ordinal, idx in enumerate(chunk):
        symbol = conservation_symbol(rows, idx)
        if not symbol.strip():
            continue
        x = grid_x + ordinal * cell_width + cell_width / 2
        title = ("identical residue across snapshot rows" if symbol == "*"
                 else "strongly similar residue properties across snapshot rows" if symbol == ":"
                 else "weakly similar residue properties across snapshot rows")
        parts.append(f'<text class="snapshot-conservation-symbol" x="{_f1(x)}" y="{y}" '
                     f'text-anchor="middle"><title>{escape_xml(title)}</title>{escape_xml(symbol)}</text>')
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Highlight: human+zebrafish shared, rodent-divergent (+ bovine trio subset)
# --------------------------------------------------------------------------- #

def human_danio_shared_rodent_divergent_columns(rows, columns) -> Dict[str, Any]:
    result: Dict[str, Any] = {"columns": set(), "bovineColumns": set(), "human": None,
                              "danio": None, "bovine": None, "rodents": [],
                              "active": False, "reason": ""}
    lst = rows if isinstance(rows, list) else []
    result["human"] = next((r for r in lst if r and (r.get("is_reference")
                            or str(r.get("species") or "").lower() == "homo_sapiens")), None)
    result["danio"] = next((r for r in lst if r and str(r.get("species") or "").lower() == "danio_rerio"), None)
    result["bovine"] = next((r for r in lst if r and str(r.get("species") or "").lower() == "bos_taurus"), None)
    result["rodents"] = [r for r in lst if is_rodent_row(r)]
    if not result["human"] or not result["danio"] or not result["rodents"]:
        result["reason"] = "needs homo_sapiens + danio_rerio + >=1 rodent selected"
        return result
    result["active"] = True
    for idx in (columns or []):
        h = residue_at(result["human"], idx).upper()
        d = residue_at(result["danio"], idx).upper()
        if not is_informative(h) or not is_informative(d) or h != d:
            continue
        diverged = True
        for rod in result["rodents"]:
            r = residue_at(rod, idx).upper()
            if not is_informative(r) or r == h:
                diverged = False
                break
        if not diverged:
            continue
        result["columns"].add(idx)
        if result["bovine"]:
            b = residue_at(result["bovine"], idx).upper()
            if is_informative(b) and b == h:
                result["bovineColumns"].add(idx)
    return result


# --------------------------------------------------------------------------- #
# Secondary structure ribbons
# --------------------------------------------------------------------------- #

def range_start(rng: Dict[str, Any]) -> float:
    for k in ("start_reference_position", "startRef", "reference_start", "start"):
        if rng.get(k) is not None:
            try:
                return float(rng[k])
            except (TypeError, ValueError):
                pass
    return math.nan


def range_end(rng: Dict[str, Any]) -> float:
    for k in ("end_reference_position", "endRef", "reference_end", "end"):
        if rng.get(k) is not None:
            try:
                return float(rng[k])
            except (TypeError, ValueError):
                pass
    return math.nan


def range_kind(rng: Dict[str, Any]) -> str:
    kind = str(rng.get("kind") or rng.get("secondary_structure") or rng.get("ss") or "loop").lower()
    if kind.startswith("h"):
        return "helix"
    if kind.startswith("s") or kind.startswith("e") or kind.startswith("b"):
        return "sheet"
    return "loop"


def helix_element(x1: float, x2: float, y: float, title: str) -> str:
    """Snapshot cursive-teardrop helix (cubic Bezier per loop). Port of the
    ``snapshot-ss-helix`` branch of alphaFoldHelixElement."""
    width = max(0.001, x2 - x1)
    out_sw = 2.80
    pitch = 14.0
    height = 20.0
    overshoot = 1.85
    num_loops = max(1, round(width / pitch))
    step = width / num_loops
    chunks = [f"M {_f2(x1)} {_f2(y)}"]
    for i in range(num_loops):
        x_start = x1 + step * i
        x_end = x1 + step * (i + 1)
        cp1x = x_start + overshoot * step
        cp2x = x_start + (1.0 - overshoot) * step
        cp_y = y - height
        chunks.append(f"C {_f2(cp1x)} {_f2(cp_y)} {_f2(cp2x)} {_f2(cp_y)} {_f2(x_end)} {_f2(y)}")
    g = ('<g fill="none" stroke-linecap="round" stroke-linejoin="round" class="snapshot-ss-helix">'
         f'<path d="{" ".join(chunks)}" stroke-width="{out_sw:.2f}"/>')
    if title:
        g += f"<title>{title}</title>"
    return g + "</g>"


def _clamp_range(rng, max_residue):
    start = math.floor(range_start(rng))
    end = math.floor(range_end(rng))
    if not (math.isfinite(start) and math.isfinite(end)):
        return None
    left = max(1, min(max_residue, min(start, end)))
    right = max(1, min(max_residue, max(start, end)))
    return {"kind": range_kind(rng), "start": left, "end": right}


def architecture_ranges(payload, track_length) -> List[Dict[str, Any]]:
    ss = (payload.get("alphafold_structure") or {}).get("secondary_structure") or {}
    residue_count = int((payload.get("alphafold_structure") or {}).get("residue_count") or 1)
    max_residue = max(1, int(track_length or residue_count or 1))
    out = []
    for rng in ss.get("ranges") or []:
        clamped = _clamp_range(rng, max_residue)
        if clamped:
            if clamped["kind"] not in ("helix", "sheet", "loop"):
                clamped["kind"] = "loop"
            out.append(clamped)
    return out


def comparative_lookup(payload) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    css = (payload.get("alphafold_structure") or {}).get("comparative_secondary_structure") or {}
    for entry in css.get("records") or []:
        for key in (entry.get("record_id"), entry.get("protein_record_id"), entry.get("species"),
                    entry.get("alphafold_entry_id"), entry.get("uniprot_accession")):
            if key:
                lookup[str(key)] = entry
    return lookup


def entry_for_row(row, lookup) -> Optional[Dict[str, Any]]:
    for key in (row.get("record_id"), row.get("protein_record_id"), row.get("species"),
                row.get("alphafold_entry_id"), row.get("uniprot_accession")):
        if key and str(key) in lookup:
            return lookup[str(key)]
    return None


def entry_display_ranges(entry, track_length) -> List[Dict[str, Any]]:
    max_residue = max(1, int(track_length or 1))
    ranges = (entry.get("mapped_ranges") if isinstance(entry.get("mapped_ranges"), list)
              else entry.get("reference_mapped_ranges") if isinstance(entry.get("reference_mapped_ranges"), list)
              else entry.get("ranges") if isinstance(entry.get("ranges"), list) else [])
    out = []
    for rng in ranges:
        s = range_start(rng)
        e = range_end(rng)
        if not (math.isfinite(s) and math.isfinite(e)):
            continue
        left = max(1, min(max_residue, math.floor(s)))
        right = max(1, min(max_residue, math.floor(e)))
        out.append({"kind": range_kind(rng), "start": min(left, right), "end": max(left, right)})
    out.sort(key=lambda r: (r["start"], r["end"]))
    return out


def secondary_ranges_for_row(row, scope, payload, lookup) -> List[Dict[str, Any]]:
    max_ref = 0
    for value in scope.get("reference_positions") or []:
        try:
            max_ref = max(max_ref, int(value))
        except (TypeError, ValueError):
            pass
    residue_count = int((payload.get("alphafold_structure") or {}).get("residue_count") or 0)
    track_length = max(1, max_ref, int(scope.get("alignment_length") or 0), residue_count)
    entry = entry_for_row(row, lookup)
    if entry:
        return entry_display_ranges(entry, track_length)
    if row.get("is_reference"):
        return architecture_ranges(payload, track_length)
    return []


def reference_spans(scope, chunk, start, end) -> List[Dict[str, int]]:
    left, right = min(start, end), max(start, end)
    spans: List[Dict[str, int]] = []
    active: Optional[Dict[str, int]] = None
    ref_positions = scope.get("reference_positions") or []
    for ordinal, idx in enumerate(chunk):
        try:
            ref_pos = float(ref_positions[idx])
        except (TypeError, ValueError, IndexError):
            ref_pos = math.nan
        in_range = math.isfinite(ref_pos) and left <= ref_pos <= right
        if not in_range:
            if active:
                spans.append(active)
                active = None
            continue
        if active and ordinal == active["endOrdinal"] + 1:
            active["endOrdinal"] = ordinal
        else:
            if active:
                spans.append(active)
            active = {"startOrdinal": ordinal, "endOrdinal": ordinal}
    if active:
        spans.append(active)
    return spans


def species_label(row, reference_species) -> str:
    species = format_label(row.get("species") or row.get("record_id") or "unknown")
    if not row.get("is_reference"):
        return species
    ref_tag = ("human ref" if str(reference_species or "").lower() == "homo_sapiens"
               else f"{format_label(reference_species or 'reference')} ref")
    return f"{species} ({ref_tag})"


def secondary_trace_svg(row, scope, chunk, grid_x, row_top, cell_width, ranges) -> str:
    if not isinstance(ranges, list) or not ranges:
        return ""
    y = row_top + 18
    parts = []
    for rng in ranges:
        kind = range_kind(rng)
        start = math.floor(range_start(rng))
        end = math.floor(range_end(rng))
        for span in reference_spans(scope, chunk, start, end):
            x1 = grid_x + span["startOrdinal"] * cell_width
            x2 = grid_x + (span["endOrdinal"] + 1) * cell_width
            if not (x2 > x1):
                continue
            label = species_label(row, scope.get("reference_species") or "reference")
            title = escape_xml(rng.get("title") or f"{label}; {kind}; reference {min(start, end)}-{max(start, end)}")
            if kind == "sheet":
                tip = max(7, min(17, (x2 - x1) * 0.45))
                pts = (f"{_f2(x1)},{_f2(y - 6)} {_f2(x2 - tip)},{_f2(y - 6)} {_f2(x2 - tip)},{_f2(y - 13)} "
                       f"{_f2(x2)},{_f2(y)} {_f2(x2 - tip)},{_f2(y + 13)} {_f2(x2 - tip)},{_f2(y + 6)} "
                       f"{_f2(x1)},{_f2(y + 6)}")
                parts.append(f'<polygon class="snapshot-ss-sheet" points="{pts}"><title>{title}</title></polygon>')
            elif kind == "helix":
                parts.append(helix_element(x1, x2, y, title))
            else:
                parts.append(f'<line class="snapshot-ss-loop" x1="{_f2(x1)}" y1="{_f2(y)}" '
                             f'x2="{_f2(x2)}" y2="{_f2(y)}"><title>{title}</title></line>')
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Scope / row selection (raw MUSCLE alignment, all residues)
# --------------------------------------------------------------------------- #

def reference_row(scope) -> Optional[Dict[str, Any]]:
    records = scope.get("records") or []
    return next((r for r in records if r.get("is_reference")), records[0] if records else None)


def select_dhrs7_records(scope) -> List[str]:
    """Return the ordered record_ids for the 4 non-reference compared species."""
    by_species: Dict[str, str] = {}
    for r in scope.get("records") or []:
        sp = str(r.get("species") or "").lower()
        if sp in DHRS7_SELECTED_ORDER and sp not in by_species:
            by_species[sp] = str(r.get("record_id") or "")
    return [by_species[sp] for sp in DHRS7_SELECTED_ORDER if sp in by_species]


def export_rows(scope, selected_ids) -> List[Dict[str, Any]]:
    record_map = {str(r.get("record_id") or ""): r for r in scope.get("records") or []}
    rows = [record_map[i] for i in selected_ids if i in record_map]
    ref = reference_row(scope)
    if ref and not any(str(r.get("record_id") or "") == str(ref.get("record_id") or "") for r in rows):
        rows.append(ref)
    return rows


def resolve_raw_scope_and_columns(payload, selected_ids):
    scopes = payload.get("scopes") or {}
    scope = scopes.get("selected_raw") or scopes.get("aligned_full")
    rows = export_rows(scope, selected_ids)
    total = (len(scope.get("reference_positions") or [])
             or int(scope.get("alignment_length") or 0)
             or (len(str(rows[0].get("aligned_sequence") or "")) if rows else 0))
    cols = []
    for i in range(total):
        if any(residue_at(r, i) not in ("", "-", ".") for r in rows):
            cols.append(i)
    return scope, cols


def range_label(scope, chunk) -> str:
    ref_positions = scope.get("reference_positions") or []
    values = []
    for idx in chunk:
        rp = ref_positions[idx] if idx < len(ref_positions) else None
        values.append(int(rp) if rp is not None else idx + 1)
    if not values:
        return "0-0"
    return f"{values[0]}-{values[-1]}"


# --------------------------------------------------------------------------- #
# Main renderer (port of buildSpeciesSnapshotSvg)
# --------------------------------------------------------------------------- #

def build_species_snapshot_svg(payload, selected_ids, residues_per_line=70,
                               highlight=True) -> str:
    scope, columns = resolve_raw_scope_and_columns(payload, selected_ids)
    rows = export_rows(scope, selected_ids)
    gene = str((payload.get("meta") or {}).get("gene_symbol") or "Alignment")
    if not columns or not rows:
        return ""

    chunks = [columns[i:i + residues_per_line] for i in range(0, len(columns), residues_per_line)]

    margin, legend_width, legend_gap, row_number_width = 36, 480, 44, 44
    cell_width, cell_height, cell_top_offset, row_height = 32, 46, 32, 80
    conservation_row_height = 52 if len(rows) > 1 else 0
    axis_height, block_gap, title_block_height = 70, 16, 100
    footer_key_height = 30 if len(rows) > 1 else 0
    max_columns = max([len(c) for c in chunks] + [1])
    grid_width = max_columns * cell_width
    block_height = axis_height + len(rows) * row_height + conservation_row_height
    width = margin * 2 + legend_width + legend_gap + row_number_width + grid_width + 8
    height = (margin + title_block_height + len(chunks) * block_height
              + max(0, len(chunks) - 1) * block_gap + footer_key_height + margin)

    ref = reference_row(scope)
    selected_count = max(0, len(rows) - (1 if ref else 0))
    reference_species = scope.get("reference_species") or (ref and ref.get("species")) or "reference"
    compare_mode, min_run = "exact", 6

    hl = (human_danio_shared_rodent_divergent_columns(rows, columns) if highlight
          else {"columns": set(), "bovineColumns": set(), "active": False, "reason": ""})
    hl_cols = hl["columns"]

    subtitle = (f"{selected_count} requested species + {format_label(reference_species)} | "
                f"{compare_mode} | min run {min_run} | {len(columns)} columns | "
                f"raw MUSCLE alignment (all residues)")
    if highlight:
        if hl["active"]:
            subtitle += (f" | {len(hl_cols)} human+zebrafish-shared rodent-divergent "
                         f"site{'' if len(hl_cols) == 1 else 's'}")
            trio_n = len(hl["bovineColumns"])
            if trio_n:
                subtitle += f" ({trio_n} also in bos_taurus, red asterisk)"
        else:
            subtitle += f" | highlight off ({hl['reason']})"

    lookup = comparative_lookup(payload)
    aa_colors = payload.get("aa_colors") or {}
    style_block = (
        '<style>.title{font:700 36px Segoe UI,Tahoma,sans-serif;fill:#18202a}'
        '.subtitle{font:600 21px Segoe UI,Tahoma,sans-serif;fill:#617083}'
        '.legend-range{font:700 28px Segoe UI,Tahoma,sans-serif;fill:#405774}'
        '.legend-name{font:700 28px Segoe UI,Tahoma,sans-serif;fill:#18202a}'
        '.legend-ref{font:700 28px Segoe UI,Tahoma,sans-serif;fill:#0f766e}'
        '.legend-ref-bg{fill:#eefaf8}.axis{stroke:#7b8188;stroke-width:3}'
        '.tick{stroke:#7b8188;stroke-width:2.2}'
        '.tick-label{font:500 20px Segoe UI,Tahoma,sans-serif;fill:#6b7280}'
        '.row-number{font:500 22px Segoe UI,Tahoma,sans-serif;fill:#18202a}'
        '.aa{font:700 28px Consolas,Courier New,monospace;fill:#18202a}'
        '.snapshot-ss-loop{stroke:#6b7280;stroke-width:2.2;stroke-linecap:round;opacity:.5}'
        '.snapshot-ss-helix{fill:none;stroke:#6b7280;stroke-width:2.8;stroke-linecap:round;stroke-linejoin:round}'
        '.snapshot-ss-sheet{fill:#6b7280;opacity:.86}'
        '.snapshot-conservation-label{font:700 24px Segoe UI,Tahoma,sans-serif;fill:#405774}'
        '.snapshot-conservation-symbol{font:800 30px Consolas,Courier New,monospace;fill:#111827}'
        '.snapshot-conservation-key{font:600 22px Segoe UI,Tahoma,sans-serif;fill:#617083}'
        '.snapshot-hd-frame{fill:none;stroke:#000000;stroke-width:3.6}'
        '.snapshot-hd-tab{fill:#000000}'
        '.snapshot-hd-trio-star{fill:none;stroke:#dc2626;stroke-width:2.8;stroke-linecap:round}</style>')
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        style_block,
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="{margin}" y="{margin + 20}">{escape_xml(gene + " species snapshot")}</text>',
        f'<text class="subtitle" x="{margin}" y="{margin + 40}">{escape_xml(subtitle)}</text>',
    ]

    y = margin + title_block_height
    legend_x = margin
    matrix_x = margin + legend_width + legend_gap
    grid_x = matrix_x + row_number_width
    ref_positions = scope.get("reference_positions") or []
    ref_residues = scope.get("reference_residues") or []

    for chunk in chunks:
        axis_y = y + 18
        tick_label_y = y + 54
        grid_top = y + axis_height
        parts.append(f'<text class="legend-range" x="{legend_x}" y="{y + 34}">{escape_xml(range_label(scope, chunk))}</text>')
        parts.append(f'<line class="axis" x1="{grid_x - 2}" y1="{axis_y}" x2="{grid_x + len(chunk) * cell_width + 2}" y2="{axis_y}"/>')

        # tick candidates: block ends + every reference position % 3 == 0
        labeled = []
        for ordinal, idx in enumerate(chunk):
            ref_pos = ref_positions[idx] if idx < len(ref_positions) else None
            if ref_pos is None:
                continue
            is_end = ordinal == 0 or ordinal == len(chunk) - 1
            if not (is_end or int(ref_pos) % 3 == 0):
                continue
            labeled.append({"refPos": ref_pos, "x": grid_x + ordinal * cell_width + cell_width / 2, "isEnd": is_end})
        for t in labeled:
            parts.append(f'<line class="tick" x1="{_f1(t["x"])}" y1="{axis_y}" x2="{_f1(t["x"])}" y2="{_f1(axis_y + 13)}"/>')
        # declutter: boundary labels win; drop colliding interior labels
        char_w, gap = 12.5, 4
        def half(t):
            return len(str(t["refPos"])) * char_w / 2
        kept = []
        def try_keep(t):
            if not any(abs(k["x"] - t["x"]) < half(k) + half(t) + gap for k in kept):
                kept.append(t)
        for t in labeled:
            if t["isEnd"]:
                try_keep(t)
        for t in labeled:
            if not t["isEnd"]:
                try_keep(t)
        for t in kept:
            parts.append(f'<text class="tick-label" x="{_f1(t["x"])}" y="{tick_label_y}" '
                         f'text-anchor="middle">{escape_xml(t["refPos"])}</text>')

        for row_index, row in enumerate(rows):
            row_top = grid_top + row_index * row_height
            row_mid = row_top + cell_top_offset + 29
            label = species_label(row, reference_species)
            if row.get("is_reference"):
                parts.append(f'<rect class="legend-ref-bg" x="{legend_x}" y="{row_top + 1}" '
                             f'width="{legend_width - 12}" height="{row_height - 3}" rx="0"/>')
            parts.append(f'<text class="{"legend-ref" if row.get("is_reference") else "legend-name"}" '
                         f'x="{legend_x}" y="{row_mid}">{escape_xml(label)}</text>')
            parts.append(f'<text class="row-number" x="{matrix_x + row_number_width - 8}" y="{row_mid}" '
                         f'text-anchor="end">{row_index + 1}</text>')
            ranges = secondary_ranges_for_row(row, scope, payload, lookup)
            parts.append(secondary_trace_svg(row, scope, chunk, grid_x, row_top, cell_width, ranges))
            for ordinal, idx in enumerate(chunk):
                aa = residue_at(row, idx) or "-"
                ref_res = ref_residues[idx] if idx < len(ref_residues) else ""
                ref_pos = ref_positions[idx] if idx < len(ref_positions) else None
                color = aa_colors.get(aa) or aa_colors.get("X") or "#f2f2f2"
                x = grid_x + ordinal * cell_width
                pos_txt = "" if ref_pos is None else f" | reference {ref_pos}"
                title = f"{label} | alignment {idx + 1}{pos_txt} | {aa or ' '} vs {ref_res or ' '}"
                parts.append(f'<rect x="{x}" y="{row_top + cell_top_offset}" width="{cell_width}" '
                             f'height="{cell_height}" fill="{escape_xml(color)}"><title>{escape_xml(title)}</title></rect>')
                parts.append(f'<text class="aa" x="{_f1(x + cell_width / 2)}" y="{row_top + cell_top_offset + 34}" '
                             f'text-anchor="middle">{escape_xml(aa)}</text>')

        if conservation_row_height:
            conservation_y = grid_top + len(rows) * row_height + 34
            parts.append(conservation_row_svg(rows, chunk, legend_x, matrix_x, grid_x, conservation_y, cell_width))

        if highlight and hl["active"] and hl_cols:
            frame_top = grid_top + cell_top_offset
            frame_height = (len(rows) - 1) * row_height + cell_height
            trio_cols = hl["bovineColumns"]
            for ordinal, idx in enumerate(chunk):
                if idx not in hl_cols:
                    continue
                x = grid_x + ordinal * cell_width
                ref_pos = ref_positions[idx] if idx < len(ref_positions) else None
                trio = idx in trio_cols
                pos_txt = "" if ref_pos is None else f" | reference {ref_pos}"
                tip = ((f"Shared in homo_sapiens + danio_rerio + bos_taurus (all active species), "
                        f"divergent in every selected rodent | alignment {idx + 1}{pos_txt}") if trio
                       else (f"Shared in homo_sapiens + danio_rerio, divergent in every selected rodent "
                             f"| alignment {idx + 1}{pos_txt}"))
                parts.append(f'<rect class="snapshot-hd-frame" x="{x}" y="{frame_top}" width="{cell_width}" '
                             f'height="{frame_height}"><title>{escape_xml(tip)}</title></rect>')
                parts.append(f'<rect class="snapshot-hd-tab" x="{x}" y="{_f1(frame_top + frame_height + 5)}" '
                             f'width="{cell_width}" height="9"><title>{escape_xml(tip)}</title></rect>')
                if trio:
                    cx = x + cell_width / 2
                    cy = frame_top + frame_height + 2
                    r = 10
                    arm = 0.866 * r
                    segs = [(0, -r, 0, r), (-arm, -r / 2, arm, r / 2), (-arm, r / 2, arm, -r / 2)]
                    for s in segs:
                        parts.append(f'<line class="snapshot-hd-trio-star" x1="{_f1(cx + s[0])}" '
                                     f'y1="{_f1(cy + s[1])}" x2="{_f1(cx + s[2])}" y2="{_f1(cy + s[3])}">'
                                     f'<title>{escape_xml(tip)}</title></line>')
        y += block_height + block_gap

    if footer_key_height:
        trio_n = len(hl["bovineColumns"]) if (highlight and hl["active"]) else 0
        legend = ""
        if highlight and hl["active"] and hl_cols:
            legend = "  |  Black frames: identical in homo_sapiens + danio_rerio, different in every selected rodent."
            if trio_n:
                legend += "  Red asterisk: also identical in bos_taurus (shared by all active species: human + zebrafish + bovine)."
        parts.append(f'<text class="snapshot-conservation-key" x="{legend_x}" y="{height - margin - 8}">'
                     f'Conservation key: * identical across snapshot rows; : strongly similar amino-acid '
                     f'properties; . weakly similar amino-acid properties.{escape_xml(legend)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Inlining (Illustrator/Inkscape safe) + payload loading + CLI
# --------------------------------------------------------------------------- #

def inline_snapshot_svg_styles(svg: str) -> str:
    out = re.sub(r"<style>.*?</style>", "", svg, count=1, flags=re.S)

    def repl(m):
        attrs = SNAPSHOT_SVG_STYLE_ATTRS.get(m.group(1))
        return " " + attrs if attrs else ""

    return re.sub(r' class="([^"]+)"', repl, out)


def load_payload(source: str) -> Dict[str, Any]:
    """Load the alignment payload from an alignment_browser.html or a JSON file."""
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    m = re.search(r'<script id="alignment-payload" type="application/json">(.*?)</script>', text, re.S)
    if not m:
        raise ValueError(f"No alignment-payload found in {source}")
    return json.loads(m.group(1))


def render_dhrs7_snapshot(payload: Dict[str, Any], residues_per_line: int = 70,
                          inline: bool = True) -> str:
    scopes = payload.get("scopes") or {}
    scope = scopes.get("selected_raw") or scopes.get("aligned_full")
    if scope is None:
        raise ValueError("payload has no selected_raw / aligned_full scope")
    selected_ids = select_dhrs7_records(scope)
    svg = build_species_snapshot_svg(payload, selected_ids, residues_per_line=residues_per_line)
    return inline_snapshot_svg_styles(svg) if inline else svg


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        print("usage: python dhrs7_snapshot_figure.py <alignment_browser.html|payload.json> [out.svg]")
        return 2
    src = argv[0]
    out = argv[1] if len(argv) > 1 else "dhrs7_species_snapshot.svg"
    payload = load_payload(src)
    svg = render_dhrs7_snapshot(payload)
    Path(out).write_text(svg, encoding="utf-8")
    print(f"wrote {out}  ({len(svg):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
