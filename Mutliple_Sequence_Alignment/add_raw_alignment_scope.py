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
    """MUSCLE_EXE env -> muscle/muscle.exe on PATH -> local tools/ dir -> 'muscle'."""
    cand = os.environ.get("MUSCLE_EXE")
    if cand and os.path.exists(cand):
        return cand
    on_path = shutil.which("muscle") or shutil.which("muscle.exe")
    if on_path:
        return on_path
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "tools", "muscle.exe"), os.path.join(here, "tools", "muscle"),
              os.path.join(here, "muscle.exe"), os.path.join(here, "muscle")):
        if os.path.exists(c):
            return c
    return "muscle"  # last resort: assume it is on PATH


MUSCLE = _find_muscle()
# The 5 canonical DHRS7 orthologs, in figure row order (mouse, rat, cattle,
# zebrafish, then the human reference row last).
DEFAULT_RECORDS = ["ENSMUSP00000021512", "ENSRNOP00000007645",
                   "ENSBTAP00000086519", "ENSDARP00000004163", "ENSP00000216500"]

OUT = (sys.argv[1] if len(sys.argv) > 1 else "DHRS7_Output").replace("\\", "/").rstrip("/")
WANT = sys.argv[2:] if len(sys.argv) > 2 else DEFAULT_RECORDS
BROWSER = OUT + "/alignment_browser.html"

html = open(BROWSER, encoding="utf-8").read()
payload = json.loads(re.search(r'<script id="alignment-payload" type="application/json">(.*?)</script>', html, re.S).group(1))
vm = re.search(r'<title>.*?</title>\s*<script>(.*?)</script>', html, re.S)
if vm:
    payload["alphafold_viewer_js"] = vm.group(1).replace("<\\/script>", "</script>")

full = payload["scopes"]["aligned_full"]
chosen = [next((r for r in full["records"] if s in str(r.get("record_id", ""))), None) for s in WANT]
missing = [s for s, r in zip(WANT, chosen) if r is None]
if missing:
    raise SystemExit("records not found: " + ", ".join(missing))

ungap = lambda s: "".join(c for c in str(s).upper() if c not in "-.")
tmp = tempfile.mkdtemp()
infa, outfa = tmp + "/in.fa", tmp + "/out.fa"
with open(infa, "w") as f:
    for i, r in enumerate(chosen):
        f.write(f">r{i}\n{ungap(r['aligned_sequence'])}\n")
res = subprocess.run([MUSCLE, "-align", infa, "-output", outfa], capture_output=True, text=True)
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

open(BROWSER, "w", encoding="utf-8").write(build_alignment_browser_html(payload))
print(f"selected_raw scope added: {len(raw_records)} species, {len(href)} columns. "
      f"Turn on the snapshot 'Raw alignment (all residues)' toggle to use it.")
