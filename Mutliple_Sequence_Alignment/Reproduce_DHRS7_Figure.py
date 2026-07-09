#!/usr/bin/env python3
r"""
Reproduce_DHRS7_Figure.py
=========================

ONE-COMMAND, TRANSPARENT REPRODUCTION of the DHRS7 species-snapshot figure used
in the publication panel. Run this single file (``python Reproduce_DHRS7_Figure.py``,
or open it in an editor and press Run) and it will, end to end:

  1. Retrieve **all** DHRS7 orthologues from Ensembl and fetch one representative
     protein sequence per species (with the documented Bos taurus UniProt
     Q24K14 substitution) -- the exact retrieval used for the paper.
  2. Length-filter (+/- 30 aa vs the human reference) and align every retrieved
     ortholog with **MUSCLE**, then project the alignment onto human reference
     positions.
  3. Retrieve the **AlphaFold** model for each species and extract its ribbon
     secondary structure (helix / strand / loop), mapping it onto human
     reference positions (sequence-aware).
  4. Build a clean 5-species MUSCLE re-alignment (mouse, rat, cattle, zebrafish,
     human) -- the residues actually compared in the figure.
  5. Render the publication **species-snapshot figure** (SVG) in pure Python
     (see ``dhrs7_snapshot_figure.py``) and collect it into ``plots/``.
  6. Emit the interactive **alignment_browser.html** (alignment + snapshot
     export only; the phylogenetic tree, 3D structure viewer, and other analysis
     panels are hidden) and a paper-ready ``METHODS_DHRS7.md``.

Every retrieved ortholog is aligned and stored, so the output lets you retrieve
any species you like; only the five listed above are drawn in the figure, in the
fixed order mouse, rat, cattle, zebrafish, then human (the reference row).

Requirements
------------
* A Python environment with Biopython, pandas, numpy, matplotlib and requests
  (``conda env create -f phylo.yml`` then ``conda activate phylo``).
* **MUSCLE** on your PATH (``muscle``/``muscle.exe``), or point ``MUSCLE_EXE`` at
  the executable. On macOS/Linux ``conda install -c bioconda muscle`` provides it;
  on Windows download ``muscle.exe`` and place it on PATH or in a ``tools/`` folder
  next to this script.
* Internet access to Ensembl, UniProt and AlphaFold.

Behind a TLS-inspecting corporate proxy? Set ``REQUESTS_CA_BUNDLE`` to your CA
bundle (or drop a ``ca_bundle.pem`` next to this script) and it will be used.

Outputs (under ``DHRS7_Output/``):
  * ``alignment_browser.html``            interactive alignment + snapshot export
  * ``plots/dhrs7_species_snapshot.svg``  the publication figure (+ .png if a Chromium/Edge browser is available)
  * ``METHODS_DHRS7.md``                  transparent, paper-ready methods text
  * aligned FASTAs, comparative AlphaFold SS json, conservation tables (provenance)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GENE = "DHRS7"
OUTPUT_DIR = SCRIPT_DIR / f"{GENE}_Output"
PLOTS_DIR = OUTPUT_DIR / "plots"

PIPELINE = SCRIPT_DIR / "gene_phylo_conservation_pipeline.py"
ADD_RAW_SCOPE = SCRIPT_DIR / "add_raw_alignment_scope.py"
BROWSER_HTML = OUTPUT_DIR / "alignment_browser.html"

# Optional overrides (all auto-detected if unset):
#   PHYLO_ENV  -- a conda-env prefix whose python/DLLs should be used for the
#                 pipeline subprocess. Leave unset to use the active environment.
#   MUSCLE_EXE -- path to the MUSCLE executable if it is not on PATH.
PHYLO_ENV_STR = os.environ.get("PHYLO_ENV")
PHYLO_ENV = Path(PHYLO_ENV_STR) if PHYLO_ENV_STR else None


def find_muscle():
    """Locate the MUSCLE executable: MUSCLE_EXE env -> PATH -> local tools/ dir."""
    cand = os.environ.get("MUSCLE_EXE")
    if cand and Path(cand).exists():
        return Path(cand)
    on_path = shutil.which("muscle") or shutil.which("muscle.exe")
    if on_path:
        return Path(on_path)
    for c in (SCRIPT_DIR / "tools" / "muscle.exe", SCRIPT_DIR / "tools" / "muscle",
              SCRIPT_DIR / "muscle.exe", SCRIPT_DIR / "muscle"):
        if c.exists():
            return c
    return None


def find_ca_bundle():
    """Optional CA bundle for TLS-inspecting proxies: env var -> local file."""
    for cand in (os.environ.get("REQUESTS_CA_BUNDLE"), os.environ.get("SSL_CERT_FILE")):
        if cand and Path(cand).exists():
            return Path(cand)
    local = SCRIPT_DIR / "ca_bundle.pem"
    return local if local.exists() else None


MUSCLE_EXE = find_muscle()
CA_BUNDLE = find_ca_bundle()

# Interactive-HTML panels hidden for the paper: the phylogenetic tree, the 3D
# AlphaFold structure viewer, the clade overlay, the run metrics, the
# domain-architecture ruler, and the evolutionary-divergence panel. Only the
# alignment grid + species-snapshot export remain. These panels are rendered by
# JS into their own <section>/<div> containers; we neutralise them by ID/class
# (safe: no JS references break, the alignment + snapshot are untouched).
STRIP_PANEL_SELECTORS = [
    "#metrics", "#node-conservation-panel", ".tree-panel", "#tree-panel",
    "#run-architecture-panel", "#alphafold-structure-panel",
    "#v11-clade-overlay-panel", "#evolutionary-divergence-panel",
]
ISOLATION_MARKER = "dhrs7-paper-isolation"

# The output kept by this reproduction: the DHRS7 alignment + species-snapshot
# figure deliverable, plus the per-species data needed to retrieve any ortholog.
# The general pipeline additionally emits clade / Fourier / evolutionary-segment /
# phylogeny / large-matrix analyses that are out of scope for this figure;
# prune_output() removes those so a fresh run yields exactly this set. Set the
# environment variable DHRS7_KEEP_ALL=1 to disable pruning and keep everything.
KEEP_OUTPUT_FILES = {
    "alignment_browser.html", "METHODS_DHRS7.md", "run_summary.txt",
    "aligned.fasta", "aligned_reference_projected.fasta",
    "alignment_reference_projected_pretty.txt", "proteins.fasta",
    "orthologs.tsv", "sequence_retrieval.tsv", "protein_metadata.tsv",
    "protein_xrefs.tsv", "protein_features.tsv", "domains.tsv",
    "length_filter_report.csv",
    "comparative_alphafold_secondary_structure.json",
    "human_reference_alphafold_model.pdb", "human_reference_alphafold_metadata.json",
    "conservation_per_position.csv", "conservation_scan.csv", "conservation_scan.svg",
    "conserved_regions.csv", "property_conservation.csv", "property_conservation.svg",
    "reference_domain_architecture.svg",
}
KEEP_OUTPUT_DIRS = {"plots", "comparative_alphafold_models",
                    "pairwise_human_reference_alignments"}

sys.path.insert(0, str(SCRIPT_DIR))
import dhrs7_snapshot_figure as figure  # noqa: E402  (pure-stdlib renderer)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def env_python() -> str:
    """Python for the pipeline subprocess: the PHYLO_ENV console python if that
    override is set, otherwise whatever interpreter is running us."""
    if PHYLO_ENV:
        for rel in ("python.exe", "python", "bin/python.exe", "bin/python"):
            cand = PHYLO_ENV / rel
            if cand.exists():
                return str(cand)
    return sys.executable


def build_subprocess_env() -> dict:
    """Environment for the pipeline / raw-scope subprocesses. If PHYLO_ENV is set
    we prepend its directories to PATH so native DLLs (matplotlib, numpy) load;
    otherwise we trust the already-activated environment. A CA bundle is applied
    only if one was found (default: the environment's certifi bundle)."""
    env = dict(os.environ)
    if PHYLO_ENV and PHYLO_ENV.exists():
        env_dirs = [str(PHYLO_ENV),
                    str(PHYLO_ENV / "Library" / "mingw-w64" / "bin"),
                    str(PHYLO_ENV / "Library" / "usr" / "bin"),
                    str(PHYLO_ENV / "Library" / "bin"),
                    str(PHYLO_ENV / "Scripts"), str(PHYLO_ENV / "bin")]
        env["PATH"] = os.pathsep.join(env_dirs) + os.pathsep + env.get("PATH", "")
        env["CONDA_PREFIX"] = str(PHYLO_ENV)
        env["CONDA_DEFAULT_ENV"] = PHYLO_ENV.name
    env["PYTHONIOENCODING"] = "utf-8"
    if CA_BUNDLE:
        env["SSL_CERT_FILE"] = str(CA_BUNDLE)
        env["REQUESTS_CA_BUNDLE"] = str(CA_BUNDLE)
    return env


def run_reduced_pipeline(env: dict) -> None:
    """Step 1-3: Ensembl retrieval -> MUSCLE alignment -> AlphaFold SS. No tree,
    no functional-divergence pilot (only the parts that build this figure)."""
    cmd = [env_python(), str(PIPELINE), GENE,
           "--source_species", "homo_sapiens",
           "--outdir", str(OUTPUT_DIR),
           "--alignment_method", "muscle",
           "--skip_v11_pilot"]    # NB: no --run_phylogeny  => no IQ-TREE
    if not os.environ.get("DHRS7_KEEP_ALL"):
        cmd.append("--figure_only")  # skip out-of-scope heavy exports in the pipeline
    if MUSCLE_EXE:
        cmd += ["--muscle_exe", str(MUSCLE_EXE)]
    log("Retrieving DHRS7 orthologs, aligning (MUSCLE), fetching AlphaFold SS ...")
    log("  " + " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=env).returncode
    if rc != 0:
        raise SystemExit(f"pipeline failed (exit {rc})")
    if not BROWSER_HTML.exists():
        raise SystemExit(f"pipeline did not produce {BROWSER_HTML}")


def add_selected_raw_scope(env: dict) -> None:
    """Step 4: build the clean 5-species MUSCLE re-alignment (selected_raw scope)."""
    log("Building the 5-species raw MUSCLE re-alignment (mouse, rat, cattle, fish, human) ...")
    rc = subprocess.run([env_python(), str(ADD_RAW_SCOPE), str(OUTPUT_DIR)],
                        cwd=str(SCRIPT_DIR), env=env).returncode
    if rc != 0:
        raise SystemExit(f"add_raw_alignment_scope failed (exit {rc})")


def render_figure() -> Path:
    """Step 5: render the publication species-snapshot SVG in pure Python."""
    log("Rendering the species-snapshot figure (pure-Python renderer) ...")
    payload = figure.load_payload(str(BROWSER_HTML))
    svg = figure.render_dhrs7_snapshot(payload, residues_per_line=70)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = PLOTS_DIR / "dhrs7_species_snapshot.svg"
    svg_path.write_text(svg, encoding="utf-8")
    log(f"  wrote {svg_path}  ({len(svg):,} bytes)")
    return svg_path


def rasterize_png(svg_path: Path) -> None:
    """Best-effort PNG via an installed Chromium/Edge headless browser. SVG is the
    primary deliverable; PNG is a convenience and skipped if no browser is found."""
    candidates = [
        shutil.which("chromium"), shutil.which("chromium-browser"),
        shutil.which("google-chrome"), shutil.which("chrome"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]
    for exe in candidates:
        if exe and Path(exe).exists():
            png = svg_path.with_suffix(".png")
            m = re.search(r'width="(\d+)" height="(\d+)"', svg_path.read_text(encoding="utf-8"))
            size = f"{m.group(1)},{m.group(2)}" if m else "2888,2876"
            try:
                subprocess.run([exe, "--headless=new", "--disable-gpu",
                                f"--screenshot={png}", f"--window-size={size}",
                                "--default-background-color=00FFFFFF", svg_path.as_uri()],
                               timeout=120, capture_output=True)
                if png.exists():
                    log(f"  wrote {png} (via {Path(exe).name})")
                    return
            except Exception as exc:  # noqa: BLE001
                log(f"  PNG rasterization skipped: {exc}")
            return
    log("  PNG rasterization skipped (no Chromium/Edge/Chrome found); SVG is the deliverable.")


def strip_interactive_html() -> None:
    """Step 6a: neutralise the tree, 3D structure, clade-overlay, metrics,
    architecture and divergence panels so the interactive browser shows only the
    alignment grid + species-snapshot export. Implemented as an injected style
    rule (removes them from view + layout) -- safe against the JS that renders
    them, so the alignment and snapshot keep working."""
    log("Hiding tree / 3D-structure / divergence / metrics panels in alignment_browser.html ...")
    html = BROWSER_HTML.read_text(encoding="utf-8")
    if ISOLATION_MARKER in html:
        return
    css = (f"\n<style id=\"{ISOLATION_MARKER}\">/* Isolate the paper alignment "
           "browser + species-snapshot export; hide all other analysis panels. */\n"
           + ", ".join(STRIP_PANEL_SELECTORS)
           + "{display:none !important}</style>\n")
    if "</head>" in html:
        html = html.replace("</head>", css + "</head>", 1)
    else:
        html = css + html
    BROWSER_HTML.write_text(html, encoding="utf-8")


def write_methods(env_payload: dict) -> None:
    """Step 6b: paper-ready methods text with the actual run numbers."""
    meta = env_payload.get("meta") or {}
    scopes = env_payload.get("scopes") or {}
    full = scopes.get("aligned_full") or {}
    proj = scopes.get("aligned_reference_projected") or {}
    raw = scopes.get("selected_raw") or {}
    n_orthologs = meta.get("recovered_sequence_count") or len(full.get("records") or [])
    afs = (env_payload.get("alphafold_structure") or {})
    css = (afs.get("comparative_secondary_structure") or {})
    muscle_name = MUSCLE_EXE.name if MUSCLE_EXE else "muscle"
    text = f"""# DHRS7 species-snapshot figure -- Methods (auto-generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})

## Ortholog retrieval
DHRS7 orthologues were enumerated from **Ensembl** (REST homology endpoint,
`type=orthologues`) starting from the human gene (*homo_sapiens*), which returns
both one2one and one2many orthology relationships across sampled vertebrates
(**{n_orthologs}** orthologs retrieved). For each species a single representative
protein was taken as the translation of the Ensembl **canonical transcript**;
where a species carried more than one ortholog, the candidate closest to the
human reference (length and percent identity) was used as the representative.
UniProt was queried to resolve each protein's reviewed/preferred accession and
cross-references. For *Bos taurus*, whose Ensembl canonical translation
(ENSBTAP00000086519, 373 aa) runs longer than the reference, the reference-length
bovine DHRS7 ortholog was taken from UniProt (**Q24K14**, 339 aa; AlphaFold model
AF-Q24K14-F1). The five orthologs compared in the figure are mouse
ENSMUSP00000021512, rat ENSRNOP00000007645, cattle ENSBTAP00000086519 (UniProt
Q24K14 sequence), zebrafish ENSDARP00000004163, and human ENSP00000216500.

## Sequence processing and alignment
Sequences were filtered to +/- 30 aa of the human reference length and aligned
with **MUSCLE** (`{muscle_name}`, `muscle -align in.fa -output out.fa`). The
multiple alignment ({full.get('alignment_length', '?')} columns) was projected
onto human reference positions ({proj.get('alignment_length', '?')} reference
columns) for numbering.

## AlphaFold structures and ribbon secondary structure
The AlphaFold model for each species was retrieved and its ribbon secondary
structure (helix / strand / loop) extracted, then mapped onto human reference
positions through a sequence-aware alignment of the model sequence to the
aligned species sequence (so substituted or length-mismatched models map their
helices/strands correctly). Comparative SS was computed for **{css.get('record_count', len(css.get('records') or []))}**
records; the human reference SS is taken from its AlphaFold model
({afs.get('model_filename', 'human AlphaFold model')}).

## Species compared in the figure
All retrieved orthologs were aligned, but the residue comparison shown in the
species-snapshot figure uses **five** species only -- **mouse (Mus musculus),
rat (Rattus norvegicus), cattle (Bos taurus), zebrafish (Danio rerio), and human
(Homo sapiens)** -- re-aligned cleanly with MUSCLE ({raw.get('alignment_length', '?')}
columns, all residues, natural gaps). Human is the reference row.

## Figure annotations
* **Grey glyphs** above each row are that species' AlphaFold secondary structure
  (cursive helix, arrow = strand, line = loop).
* **Black frames** mark columns where *homo_sapiens* and *danio_rerio* carry the
  identical residue while every rodent (mouse, rat) differs.
* **Red asterisks** mark the subset of those columns where *bos_taurus* also
  shares the residue -- i.e. identical across all three catalytically-active
  species (human, zebrafish, bovine) while both inactive rodents diverge;
  candidate activity-linked residues.
* Conservation row: `*` identical, `:` strongly similar, `.` weakly similar
  across the five rows (Clustal convention).

## Reproducibility
`python Reproduce_DHRS7_Figure.py` reproduces this figure and this file from
scratch (Ensembl + UniProt + AlphaFold retrieval, MUSCLE alignment, pure-Python
figure rendering). Figure: `plots/dhrs7_species_snapshot.svg`.
"""
    (OUTPUT_DIR / "METHODS_DHRS7.md").write_text(text, encoding="utf-8")
    log(f"  wrote {OUTPUT_DIR / 'METHODS_DHRS7.md'}")


def collect_plots(svg_path: Path) -> None:
    """Copy any other final paper figures the pipeline produced into plots/."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("reference_domain_architecture.svg", "conservation_scan.svg",
                 "property_conservation.svg"):
        src = OUTPUT_DIR / name
        if src.exists():
            shutil.copy2(src, PLOTS_DIR / name)
    log(f"  final figures collected in {PLOTS_DIR}")


def prune_output() -> None:
    """Remove any pipeline output outside the alignment + figure deliverable so the
    reproduction yields exactly the documented set (see KEEP_OUTPUT_* above). This
    is an allow-list: anything not explicitly kept is deleted. Disable with the
    environment variable DHRS7_KEEP_ALL=1."""
    if os.environ.get("DHRS7_KEEP_ALL"):
        log("  pruning skipped (DHRS7_KEEP_ALL set); keeping all pipeline outputs")
        return
    removed = 0
    for entry in sorted(OUTPUT_DIR.iterdir()):
        keep = (entry.name in KEEP_OUTPUT_DIRS) if entry.is_dir() else (entry.name in KEEP_OUTPUT_FILES)
        if keep:
            continue
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            removed += 1
        except OSError as exc:  # noqa: PERF203
            log(f"  prune: could not remove {entry.name}: {exc}")
    log(f"  pruned {removed} out-of-scope item(s); output is the alignment + figure deliverable")


def main() -> int:
    log(f"=== DHRS7 figure reproduction -> {OUTPUT_DIR} ===")
    if MUSCLE_EXE:
        log(f"MUSCLE: {MUSCLE_EXE}")
    else:
        log("WARNING: MUSCLE not found on PATH or via MUSCLE_EXE; the pipeline will "
            "try 'muscle' on PATH and fail if it is not installed.")
    env = build_subprocess_env()
    run_reduced_pipeline(env)
    add_selected_raw_scope(env)
    payload = figure.load_payload(str(BROWSER_HTML))
    svg_path = render_figure()
    rasterize_png(svg_path)
    strip_interactive_html()
    write_methods(payload)
    collect_plots(svg_path)
    prune_output()
    log("=== DONE ===")
    log(f"Figure : {svg_path}")
    log(f"Browser: {BROWSER_HTML}")
    log(f"Methods: {OUTPUT_DIR / 'METHODS_DHRS7.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
