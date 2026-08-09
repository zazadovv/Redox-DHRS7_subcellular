"""Add a 'selected_raw' scope to a run's alignment_browser.html: a clean MUSCLE
re-alignment of just the chosen species (all residues, natural gaps, UGENE-style),
so the species-snapshot "Raw alignment (all residues)" toggle renders that clean
alignment in our styling instead of the human reference-projected view.

Usage:
    python add_raw_alignment_scope.py <output_dir> [record_substr ...]

<output_dir>      a run output dir containing alignment_browser.html (e.g. DHRS7_Output)
record_substr...  optional; substrings that identify the records to include, in the
                  desired row order. Defaults to the 5 canonical DHRS7 orthologs
                  in figure order: mouse, rat, cattle, zebrafish, human.

Without this step the Raw toggle falls back to the full multi-species alignment
subset (all residues shown, but with extra gaps inherited from other species).
Requires MUSCLE on PATH (or MUSCLE_EXE pointing at the executable).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Bio import SeqIO
from gene_phylo_conservation_archive import build_alignment_browser_html


def _find_muscle():
    """MUSCLE_EXE, then PATH, then a tools folder here or in the folders above.

    The same search the entry point uses, so running this step on its own finds
    the same executable the full build would."""
    cand = os.environ.get("MUSCLE_EXE")
    if cand and os.path.exists(cand):
        return cand
    on_path = shutil.which("muscle") or shutil.which("muscle.exe")
    if on_path:
        return on_path
    base = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        for name in ("muscle.exe", "muscle"):
            for c in (os.path.join(base, "tools", name), os.path.join(base, name)):
                if os.path.exists(c):
                    return c
        parent = os.path.dirname(base)
        if parent == base:
            break
        base = parent
    return None


MUSCLE = _find_muscle()
if MUSCLE is None:
    raise SystemExit(
        "MUSCLE was not found. It is needed to align the species drawn in the "
        "figure. Set MUSCLE_EXE to its full path, put it on PATH, or place it in "
        "a 'tools' folder beside these scripts, then try again. "
        "Run  python dhrs7_alignment.py --check  for the full report.")
# The canonical DHRS7 ortholog per species, in figure row order (mouse, rat,
# cattle, zebrafish, then the human reference row last). Mouse, rat and zebrafish
# each return more than one ortholog from Ensembl, so the record used for the
# comparison is pinned here rather than being resolved on the fly: these are the
# canonical translations, not the additional paralogous records.
CANONICAL_RECORD_BY_SPECIES = {
    "mus_musculus": "ENSMUSP00000021512",
    "rattus_norvegicus": "ENSRNOP00000007645",
    "bos_taurus": "ENSBTAP00000086519",
    "danio_rerio": "ENSDARP00000004163",
    "homo_sapiens": "ENSP00000216500",
}
DEFAULT_RECORDS = list(CANONICAL_RECORD_BY_SPECIES.values())

OUT = (sys.argv[1] if len(sys.argv) > 1 else "DHRS7_Output").replace("\\", "/").rstrip("/")
WANT = sys.argv[2:] if len(sys.argv) > 2 else DEFAULT_RECORDS
BROWSER = OUT + "/alignment_browser.html"

html = open(BROWSER, encoding="utf-8").read()
payload = json.loads(re.search(r'<script id="alignment-payload" type="application/json">(.*?)</script>', html, re.S).group(1))
vm = re.search(r'<title>.*?</title>\s*<script>(.*?)</script>', html, re.S)
if vm:
    payload["alphafold_viewer_js"] = vm.group(1).replace("<\\/script>", "</script>")

full = payload["scopes"]["aligned_full"]


def pick_record(records, token):
    """Resolve one requested row. A species name selects that species' canonical
    record when one is pinned above, otherwise the ortholog closest to the human
    reference; anything else is matched as a record-id substring."""
    key = str(token).strip().lower()
    pinned = CANONICAL_RECORD_BY_SPECIES.get(key)
    if pinned:
        match = next((r for r in records if pinned in str(r.get("record_id", ""))), None)
        if match is not None:
            return match
    same_species = [r for r in records if str(r.get("species") or "").lower() == key]
    if same_species:
        same_species.sort(key=lambda r: (bool(r.get("is_reference")),
                                         float(r.get("identity_to_reference") or 0.0)),
                          reverse=True)
        return same_species[0]
    return next((r for r in records if token in str(r.get("record_id", ""))), None)


chosen = [pick_record(full["records"], s) for s in WANT]
missing = [s for s, r in zip(WANT, chosen) if r is None]
if missing:
    raise SystemExit("records not found: " + ", ".join(missing))

ungap = lambda s: "".join(c for c in str(s).upper() if c not in "-.")
tmp = tempfile.mkdtemp()
infa, outfa = tmp + "/in.fa", tmp + "/out.fa"
with open(infa, "w") as f:
    for i, r in enumerate(chosen):
        f.write(f">r{i}\n{ungap(r['aligned_sequence'])}\n")
try:
    res = subprocess.run([MUSCLE, "-align", infa, "-output", outfa],
                         capture_output=True, text=True)
except OSError as exc:
    raise SystemExit(f"could not run MUSCLE at {MUSCLE!r}: {exc}")
if not os.path.exists(outfa):
    raise SystemExit("MUSCLE failed:\n" + (res.stderr or "")[-2000:])
aln = {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(outfa, "fasta")}

raw_records = []
for i, r in enumerate(chosen):
    a = aln[f"r{i}"]
    rec = dict(r)
    ug = ungap(a)
    rec.update(aligned_sequence=a, alignment_scope="selected_raw",
               scope_label="Selected species (raw MUSCLE)", aligned_length=len(a),
               ungapped_length=len(ug), gap_count=len(a) - len(ug),
               gap_fraction=((len(a) - len(ug)) / len(a)) if a else 0)
    raw_records.append(rec)

human = next((r for r in raw_records if r.get("is_reference")), None) or raw_records[-1]
href = human["aligned_sequence"]
rp, rr, p = [], [], 0
for c in href:
    rr.append(c.upper())
    if c not in "-.":
        p += 1
        rp.append(p)
    else:
        rp.append(None)

payload["scopes"]["selected_raw"] = {
    "label": "Selected species (raw MUSCLE)",
    "source_fasta": "selected_raw_muscle.fasta",
    "alignment_length": len(href),
    "reference_species": human.get("species"),
    "reference_record_id": human.get("record_id"),
    "reference_sequence": href,
    "reference_positions": rp,
    "reference_residues": rr,
    "reference_landmarks": full.get("reference_landmarks"),
    "records": raw_records,
    "evolutionary_divergence": full.get("evolutionary_divergence"),
}


# --------------------------------------------------------------------------- #
# Structure track: assign secondary structure on THIS alignment
# --------------------------------------------------------------------------- #
# Each species' secondary structure is taken from its own AlphaFold model and
# placed on the alignment built above, so the ribbon follows the residues that
# are drawn rather than being inherited from the full multi-species alignment
# (where a residue sitting in an insertion column carries no assignment at all).
#
# Assignment uses backbone hydrogen bonding (Kabsch-Sander), so a strand is only
# called where a beta bridge partner exists. A torsion-angle classifier reports
# any extended backbone as strand, which mislabels the disordered, extended
# C-terminal tails of these proteins; those regions are loop here, in agreement
# with their low model confidence.

MIN_HELIX_RUN = 4      # shorter helical turns are not drawn as helices
MIN_STRAND_RUN = 3     # isolated bridges are not drawn as strands


def _drop_short_runs(kinds):
    """Suppress runs too short to be worth drawing (they become loop)."""
    out = list(kinds)
    start = 0
    while start < len(out):
        end = start
        while end + 1 < len(out) and out[end + 1] == out[start]:
            end += 1
        length = end - start + 1
        minimum = MIN_HELIX_RUN if out[start] == "helix" else MIN_STRAND_RUN if out[start] == "sheet" else 0
        if minimum and length < minimum:
            for i in range(start, end + 1):
                out[i] = "loop"
        start = end + 1
    return out


def _residue_index_map(model_seq, species_seq):
    """species residue index (1-based) -> model residue index (1-based)."""
    if len(model_seq) == len(species_seq):
        return {i + 1: i + 1 for i in range(len(species_seq))}
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    try:
        alignment = next(iter(aligner.align(model_seq, species_seq)))
    except Exception:  # noqa: BLE001
        return {}
    mapping, mi, si = {}, 0, 0
    for a, b in zip(str(alignment[0]), str(alignment[1])):
        if a != "-":
            mi += 1
        if b != "-":
            si += 1
        if a != "-" and b != "-":
            mapping[si] = mi
    return mapping


def _ranges_from_column_kinds(column_kinds, reference_positions):
    """Contiguous same-kind column runs -> ranges on the reference ruler. Loop
    runs are kept so the backbone is drawn as a continuous trace."""
    ranges, active = [], None
    for column in sorted(column_kinds):
        ref = reference_positions[column] if column < len(reference_positions) else None
        if ref is None:
            continue
        kind = column_kinds[column]
        if active and active["kind"] == kind and int(ref) == active["end_reference_position"] + 1:
            active["end_reference_position"] = int(ref)
            continue
        if active:
            ranges.append(active)
        active = {"kind": kind, "start_reference_position": int(ref), "end_reference_position": int(ref)}
    if active:
        ranges.append(active)
    return ranges


def _model_path_for(entry, outdir):
    """Locate a record's model file, falling back to the stored reference model."""
    name = str((entry or {}).get("model_filename") or "")
    if name:
        candidate = os.path.join(outdir, name.replace("/", os.sep))
        if os.path.exists(candidate):
            return candidate
    fallback = os.path.join(outdir, "human_reference_alphafold_model.pdb")
    return fallback if os.path.exists(fallback) else ""


def build_structure_tracks(payload, scope, outdir):
    """Structure tracks for every row drawn in `scope`, under both assignment
    methods. Returns ({method: {record_id: ranges}}, report lines)."""
    import dhrs7_snapshot_figure as figure
    from secondary_structure import assign_secondary_structure, assign_from_torsion

    methods = {"torsion": assign_from_torsion, "hbond": assign_secondary_structure}
    reference_positions = scope.get("reference_positions") or []
    lookup = figure.comparative_lookup(payload)
    track_length = max((int(v) for v in reference_positions if v is not None), default=0)
    tracks = {name: {} for name in methods}
    notes = []

    for record in scope.get("records") or []:
        species = str(record.get("species") or "?")
        record_id = str(record.get("record_id") or "")
        sequence = str(record.get("aligned_sequence") or "")
        entry = figure.entry_for_row(record, lookup)
        model_path = _model_path_for(entry if entry is not None else
                                     ({} if record.get("is_reference") else None), outdir)

        summary = []
        for name, assign in methods.items():
            ranges = None
            if model_path:
                try:
                    assigned = assign(open(model_path, encoding="utf-8", errors="ignore").read())
                    index_map = _residue_index_map(
                        assigned["sequence"], "".join(c for c in sequence if c not in "-."))
                    column_kinds, seen = {}, 0
                    for column, char in enumerate(sequence):
                        if char in "-.":
                            continue
                        seen += 1
                        model_index = index_map.get(seen)
                        if model_index and 1 <= model_index <= len(assigned["kinds"]):
                            column_kinds[column] = assigned["kinds"][model_index - 1]
                    columns = sorted(column_kinds)
                    cleaned = _drop_short_runs([column_kinds[c] for c in columns])
                    ranges = _ranges_from_column_kinds(dict(zip(columns, cleaned)),
                                                       reference_positions)
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"{species}/{name}: could not assign from model ({exc})")
            if ranges is None:
                if entry is not None:
                    base = figure.entry_display_ranges(entry, track_length)
                elif record.get("is_reference"):
                    base = figure.architecture_ranges(payload, track_length)
                else:
                    base = []
                ranges = [{"kind": r["kind"], "start_reference_position": r["start"],
                           "end_reference_position": r["end"]} for r in base]
            tracks[name][record_id] = ranges
            structured = sum(1 for r in ranges if r["kind"] != "loop")
            summary.append(f"{name}={structured}")
        notes.append(f"{species}: " + ", ".join(summary) +
                     f" elements ({os.path.basename(model_path) if model_path else 'stored'})")

    return tracks, notes


# Both assignments are stored so either can be drawn without recomputing, and so
# the two can be compared directly on the same alignment.
DEFAULT_SS_METHOD = os.environ.get("DHRS7_SS_METHOD", "torsion").strip().lower()

try:
    tracks, track_notes = build_structure_tracks(payload, payload["scopes"]["selected_raw"], OUT)
    scope_out = payload["scopes"]["selected_raw"]
    scope_out["secondary_structure_methods"] = tracks
    scope_out["secondary_structure_method"] = DEFAULT_SS_METHOD
    scope_out["secondary_structure_by_record"] = tracks.get(DEFAULT_SS_METHOD) or tracks["torsion"]
    for note in track_notes:
        print("  structure: " + note)
    print(f"  structure: drawing '{DEFAULT_SS_METHOD}'; both assignments stored in the payload")
except Exception as exc:  # noqa: BLE001
    print(f"  structure track assignment skipped ({exc})")

open(BROWSER, "w", encoding="utf-8").write(build_alignment_browser_html(payload))
print(f"selected_raw scope added: {len(raw_records)} species, {len(href)} columns.")
