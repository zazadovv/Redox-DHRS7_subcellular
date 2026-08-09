#!/usr/bin/env python3
r"""
dhrs7_alignment.py
==================

DHRS7 cross-species alignment and species-snapshot figure.

This is the single command that builds the DHRS7 comparison from scratch. It
collects every DHRS7 ortholog Ensembl knows about, keeps one representative
protein per species, aligns them, attaches each species' AlphaFold model, and
draws the annotated alignment panel used for the residue-level comparison.

    python dhrs7_alignment.py --check              # verify the setup first
    python dhrs7_alignment.py                      # full build
    python dhrs7_alignment.py --figure-only        # redraw from an existing build
    python dhrs7_alignment.py --species mus_musculus,danio_rerio
    python dhrs7_alignment.py --residues-per-line 60

What it does
------------
1. Enumerate DHRS7 orthologs from Ensembl and take one representative protein per
   species (the canonical transcript translation; for species carrying several
   orthologs, the candidate closest to the human reference). *Bos taurus* is the
   one documented exception: its Ensembl canonical translation is a longer
   isoform, so the reference-length UniProt entry Q24K14 is used instead, which
   keeps cattle in the comparison with a matching AlphaFold model.
2. Restrict to sequences within +/- 30 aa of the human reference length. This
   window was settled on after trialling wider and narrower cut-offs: it removes
   fragment and read-through predictions that otherwise open long spurious gaps,
   while retaining the full vertebrate span.
3. Align with MUSCLE and project onto human reference numbering.
4. Retrieve each species' AlphaFold model and derive its secondary structure from
   the model backbone.
5. Re-align the species selected for the figure on their own, so the panel shows
   their true pairwise gaps rather than gaps inherited from the full alignment,
   and place each species' secondary structure onto that alignment.
6. Draw the figure, write the interactive alignment browser and a methods
   summary filled in with the numbers from the run.

Requirements
------------
* Python 3.10 with biopython, pandas, numpy, matplotlib and requests
  (``conda env create -f phylo.yml`` then ``conda activate phylo``).
* MUSCLE v5 on PATH, in a ``tools`` folder beside this script, or at
  ``MUSCLE_EXE``. Run with ``--check`` to confirm before starting.
* Network access to Ensembl, UniProt and AlphaFold (not needed for
  ``--figure-only``).

Behind a TLS-inspecting proxy, set ``REQUESTS_CA_BUNDLE`` or drop a
``ca_bundle.pem`` next to this file and it will be picked up automatically.

Output (``DHRS7_Output/`` by default)
-------------------------------------
* ``plots/dhrs7_species_snapshot.svg``  the figure (plus ``.png`` when a
  Chromium-based browser is available to rasterise it)
* ``alignment_browser.html``            interactive alignment, all species
* ``METHODS_DHRS7.md``                  methods summary for this run
* ortholog tables, aligned FASTAs, per-species AlphaFold models and secondary
  structure, and conservation tables
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GENE = "DHRS7"          # default; --gene overrides it
REFERENCE_SPECIES = "homo_sapiens"

PIPELINE = SCRIPT_DIR / "gene_phylo_conservation_pipeline.py"
ADD_RAW_SCOPE = SCRIPT_DIR / "add_raw_alignment_scope.py"

# Set from the command line in main(); see resolve_paths().
OUTPUT_DIR = SCRIPT_DIR / f"{GENE}_Output"
PLOTS_DIR = OUTPUT_DIR / "plots"
BROWSER_HTML = OUTPUT_DIR / "alignment_browser.html"

# Optional overrides, all auto-detected when unset:
#   PHYLO_ENV  -- an environment prefix whose interpreter should run the pipeline
#                 step. Leave unset to use the active environment.
#   MUSCLE_EXE -- MUSCLE executable, if it is not on PATH.
PHYLO_ENV_STR = os.environ.get("PHYLO_ENV")
PHYLO_ENV = Path(PHYLO_ENV_STR) if PHYLO_ENV_STR else None

# The interactive browser is written by the shared pipeline, which also renders
# a phylogeny, a 3D structure viewer and several summary panels. This build is
# about the alignment, so those panels are hidden; the JavaScript that draws them
# is left untouched, which keeps the alignment and its export working.
HIDDEN_PANEL_SELECTORS = [
    "#metrics", "#node-conservation-panel", ".tree-panel", "#tree-panel",
    "#run-architecture-panel", "#alphafold-structure-panel",
    "#clade-overlay-panel", "#evolutionary-divergence-panel",
]
HIDDEN_PANEL_MARKER = "dhrs7-alignment-view"

# Files this build keeps. The shared pipeline can also emit clade, spectral and
# phylogeny analyses that are not part of this comparison; prune_output() drops
# them so a run always leaves the same, predictable set of files behind.
KEEP_OUTPUT_FILES = {
    "alignment_browser.html", "run_summary.txt",
    "environment.txt",
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
import dhrs7_snapshot_figure as figure  # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def methods_filename() -> str:
    return f"METHODS_{GENE}.md"


def figure_filename() -> str:
    return f"{GENE.lower()}_species_snapshot.svg"


def resolve_paths(outdir: Path) -> None:
    global OUTPUT_DIR, PLOTS_DIR, BROWSER_HTML
    OUTPUT_DIR = outdir
    PLOTS_DIR = outdir / "plots"
    BROWSER_HTML = outdir / "alignment_browser.html"


MUSCLE_SEARCH_LEVELS = 4  # this folder, then three above it

# Non-executable extensions that a "muscle*" glob might otherwise pick up
# (the release page also ships archives and notes next to the binary).
_NON_BINARY_SUFFIXES = {".txt", ".md", ".zip", ".gz", ".tar", ".tgz", ".bz2",
                        ".xz", ".pdf", ".html", ".htm", ".sha256", ".asc"}


def _env_roots() -> list[Path]:
    """Environment prefixes whose bin folders may hold a conda-installed MUSCLE.

    Includes the interpreter's own prefix. When the GUI launches ``python.exe``
    straight from a conda environment the environment is not activated, so its
    Scripts/Library\\bin never reach PATH -- looking here finds MUSCLE anyway."""
    roots: list[Path] = []
    for root in (PHYLO_ENV, Path(sys.prefix), Path(sys.exec_prefix)):
        if root and root not in roots:
            roots.append(root)
    return roots


def muscle_search_dirs() -> list[Path]:
    """Directories scanned for a MUSCLE executable, most specific first.

    Two families: the bin folders of the environment the sub-steps run in, and
    ``tools`` folders beside the project (a shared copy is often kept next to the
    project rather than inside it, so the search climbs a few levels)."""
    dirs: list[Path] = []
    for root in _env_roots():
        for rel in (".", "Library/bin", "Scripts", "bin",
                    "Library/mingw-w64/bin", "Library/usr/bin"):
            dirs.append(root / rel)
    base = SCRIPT_DIR
    for _ in range(MUSCLE_SEARCH_LEVELS):
        dirs.append(base / "tools")
        dirs.append(base)
        if base.parent == base:
            break
        base = base.parent
    seen: set = set()
    unique: list[Path] = []
    for directory in dirs:
        try:
            key = directory.resolve()
        except OSError:
            key = directory
        if key not in seen:
            seen.add(key)
            unique.append(directory)
    return unique


def _fixed_drive_roots() -> list[Path]:
    """Root of every drive that exists (C:\\, D:\\, ... on Windows; / elsewhere)."""
    roots: list[Path] = []
    if os.name == "nt":
        import string
        for letter in string.ascii_uppercase:
            root = Path(f"{letter}:\\")
            try:
                if root.exists():
                    roots.append(root)
            except OSError:
                continue
    else:
        roots = [Path("/")]
    return roots


def _conda_env_bin_dirs(conda_root: Path) -> list[Path]:
    """The bin folders of every environment under a conda installation."""
    dirs: list[Path] = []
    envs = conda_root / "envs"
    try:
        entries = list(envs.iterdir()) if envs.is_dir() else []
    except OSError:
        entries = []
    for env in [conda_root] + entries:
        try:
            if env.is_dir():
                for rel in (".", "Library/bin", "Scripts", "bin"):
                    dirs.append(env / rel)
        except OSError:
            continue
    return dirs


def common_muscle_dirs() -> list[Path]:
    """Likely install spots on every drive, plus every conda environment found.

    This is the "look across all drives" search: it visits a curated set of
    folders (drive roots, tools/muscle/bin, Program Files, Downloads) and the bin
    folders of any conda installation on each drive and in the home directory,
    rather than walking whole disks. It finds a MUSCLE that lives in another
    environment even when a different interpreter is launched."""
    dirs: list[Path] = []
    conda_names = ("Conda", "Anaconda3", "anaconda3", "Miniconda3", "miniconda3",
                   "ProgramData/Anaconda3", "ProgramData/Miniconda3")
    for root in _fixed_drive_roots():
        for rel in ("", "tools", "muscle", "bin",
                    "Program Files/muscle", "Program Files (x86)/muscle"):
            dirs.append(root / rel if rel else root)
        for name in conda_names:
            dirs += _conda_env_bin_dirs(root / name)
    home = Path.home()
    for rel in ("Downloads", "Desktop", "tools"):
        dirs.append(home / rel)
    for name in (".conda", "Anaconda3", "anaconda3", "Miniconda3", "miniconda3",
                 "AppData/Local/miniconda3", "AppData/Local/Continuum/anaconda3"):
        dirs += _conda_env_bin_dirs(home / name)
    return dirs


def deep_scan_for_muscle(time_budget_s: float = 25.0, max_depth: int = 7,
                         announce=None) -> Path | None:
    """Last resort: a bounded recursive walk of every drive.

    Prunes system, recycle and package trees, follows no reparse points, caps
    depth and total time, and returns the first MUSCLE executable found. Slow by
    nature, so it is opt-in (``--scan-drives``) rather than part of every run."""
    import time
    skip = {"windows", "$recycle.bin", "system volume information", "$sysreset",
            "node_modules", ".git", "__pycache__", "appdata", "winsxs",
            "recovery", "perflogs", "msocache", "$windows.~ws", "$windows.~bt"}
    start = time.monotonic()
    for root in _fixed_drive_roots():
        if announce:
            announce(str(root))
        root_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            if time.monotonic() - start > time_budget_s:
                if announce:
                    announce(f"(stopped after {time_budget_s:.0f}s)")
                return None
            here = Path(dirpath)
            if len(here.parts) - root_depth >= max_depth:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames
                           if d.lower() not in skip and not d.startswith("$")
                           and not os.path.islink(os.path.join(dirpath, d))]
            for name in filenames:
                low = name.lower()
                if not low.startswith("muscle"):
                    continue
                candidate = here / name
                if _looks_like_binary(candidate):
                    return candidate
    return None


def _looks_like_binary(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() in _NON_BINARY_SUFFIXES:
        return False
    if os.name != "nt" and not os.access(path, os.X_OK):
        return False
    return True


def scan_dir_for_muscle(directory: Path) -> Path | None:
    """A MUSCLE executable in one directory, exact name preferred.

    Matches release binaries that keep their downloaded name too, e.g.
    ``muscle-win64.v5.3.exe`` on Windows or ``muscle5.1.linux_intel64``."""
    if not directory.is_dir():
        return None
    exact = directory / ("muscle.exe" if os.name == "nt" else "muscle")
    if _looks_like_binary(exact):
        return exact
    pattern = "muscle*.exe" if os.name == "nt" else "muscle*"
    matches = sorted(p for p in directory.glob(pattern) if _looks_like_binary(p))
    return matches[0] if matches else None


def _path_dirs() -> list[Path]:
    """Directories on PATH, so a release binary that kept its own name
    (e.g. muscle-win64.v5.3.exe) is found even though ``which`` cannot name it."""
    dirs: list[Path] = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        entry = entry.strip().strip('"')
        if entry:
            dirs.append(Path(entry))
    return dirs


MUSCLE_REQUIRED_MAJOR = 5  # the pipeline uses v5's -align / -super5 options


def muscle_version_info(path: Path | str) -> tuple[int | None, str | None]:
    """(major version, first line of ``-version`` output), or (None, None).

    MUSCLE v5 prints ``muscle 5.3.win64 [d9725ac]``; the old v3 prints
    ``MUSCLE v3.8.31 by Robert C. Edgar``. The two use different, incompatible
    command lines, so knowing the major version up front avoids a late failure."""
    try:
        finished = subprocess.run([str(path), "-version"],
                                  capture_output=True, text=True, timeout=15)
    except Exception:  # noqa: BLE001
        return None, None
    text = (finished.stdout or "") + "\n" + (finished.stderr or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    match = re.search(r"muscle\s+v?(\d+)\.", text, re.IGNORECASE)
    return (int(match.group(1)) if match else None), (lines[0] if lines else None)


def muscle_major_version(path: Path | str) -> int | None:
    return muscle_version_info(path)[0]


def find_muscle(explicit: str | None = None, deep: bool = False,
                announce=None) -> Path | None:
    """--muscle, then MUSCLE_EXE, then PATH, then the interpreter's environment,
    a tools folder nearby, and common install spots on every drive -- matching
    release binaries by name variant and preferring a v5 build. With ``deep``,
    fall back to a bounded recursive walk of all drives."""
    # An explicit choice (flag or env var) is honoured exactly, version aside.
    for candidate in (explicit, os.environ.get("MUSCLE_EXE")):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    # Collect auto-discovered candidates in priority order, de-duplicated.
    candidates: list[Path] = []
    for name in ("muscle", "muscle.exe", "muscle5", "muscle5.exe"):
        on_path = shutil.which(name)
        if on_path:
            candidates.append(Path(on_path))
    for directory in _path_dirs() + muscle_search_dirs() + common_muscle_dirs():
        hit = scan_dir_for_muscle(directory)
        if hit:
            candidates.append(hit)
    seen: set = set()
    ordered: list[Path] = []
    for candidate in candidates:
        key = str(candidate).lower()
        if key not in seen:
            seen.add(key)
            ordered.append(candidate)
    # Prefer a v5 build; otherwise return the first found so the caller can
    # report the version mismatch rather than silently using the wrong MUSCLE.
    fallback: Path | None = None
    for candidate in ordered:
        if muscle_major_version(candidate) == MUSCLE_REQUIRED_MAJOR:
            return candidate
        if fallback is None:
            fallback = candidate
    if fallback is not None:
        return fallback
    if deep:
        return deep_scan_for_muscle(announce=announce)
    return None


def find_ca_bundle() -> Path | None:
    """CA bundle for TLS-inspecting proxies: environment first, then a local file."""
    for candidate in (os.environ.get("REQUESTS_CA_BUNDLE"), os.environ.get("SSL_CERT_FILE")):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    local = SCRIPT_DIR / "ca_bundle.pem"
    return local if local.exists() else None


MUSCLE_EXE: Path | None = None
CA_BUNDLE: Path | None = None
SS_METHOD: str = "torsion"


def env_python() -> str:
    """Interpreter for the pipeline step: PHYLO_ENV's if set, else this one."""
    if PHYLO_ENV:
        for rel in ("python.exe", "python", "bin/python.exe", "bin/python"):
            candidate = PHYLO_ENV / rel
            if candidate.exists():
                return str(candidate)
    return sys.executable


def interpreter_prefix() -> Path:
    """Root of the environment whose interpreter runs the sub-steps."""
    if PHYLO_ENV and PHYLO_ENV.exists():
        return PHYLO_ENV
    return Path(sys.prefix)


def build_subprocess_env() -> dict:
    """Environment for the pipeline and alignment sub-steps.

    The interpreter's own library directories are always prepended to PATH. A
    conda environment keeps its compiled libraries in Library\\bin and similar
    folders, and activating the environment is what normally puts those on PATH.
    Starting python.exe directly -- which is what happens when the window is
    opened from a shortcut rather than from an activated shell -- leaves them
    off, and the first native library to be needed brings the process down with
    an unhandled DLL error rather than a Python traceback. Setting PATH here
    means the sub-steps behave the same whether or not anything was activated."""
    env = dict(os.environ)
    prefix = interpreter_prefix()
    if prefix.exists():
        candidates = [prefix,
                      prefix / "Library" / "mingw-w64" / "bin",
                      prefix / "Library" / "usr" / "bin",
                      prefix / "Library" / "bin",
                      prefix / "Scripts",
                      prefix / "bin",
                      prefix / "DLLs"]
        existing = [str(path) for path in candidates if path.exists()]
        if existing:
            env["PATH"] = os.pathsep.join(existing) + os.pathsep + env.get("PATH", "")
        env["CONDA_PREFIX"] = str(prefix)
        env["CONDA_DEFAULT_ENV"] = prefix.name
    env["PYTHONIOENCODING"] = "utf-8"
    if MUSCLE_EXE:
        # Every sub-step runs as its own process and looks MUSCLE up for itself.
        # Passing the resolved path along means they all use the one that was
        # found here, wherever it happened to be.
        env["MUSCLE_EXE"] = str(MUSCLE_EXE)
    if CA_BUNDLE:
        env["SSL_CERT_FILE"] = str(CA_BUNDLE)
        env["REQUESTS_CA_BUNDLE"] = str(CA_BUNDLE)
    return env


def run_retrieval_and_alignment(env: dict, keep_all: bool) -> None:
    """Steps 1-4: Ensembl retrieval, length filter, MUSCLE alignment, AlphaFold
    models and secondary structure. No phylogeny is built."""
    cmd = [env_python(), str(PIPELINE), GENE,
           "--source_species", REFERENCE_SPECIES,
           "--outdir", str(OUTPUT_DIR),
           "--alignment_method", "muscle",
           "--skip_pilot"]
    if not keep_all:
        cmd.append("--figure_only")
    if MUSCLE_EXE:
        cmd += ["--muscle_exe", str(MUSCLE_EXE)]
    log("Retrieving DHRS7 orthologs, aligning (MUSCLE), fetching AlphaFold models ...")
    log("  " + " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=env).returncode
    if rc != 0:
        raise SystemExit(f"retrieval/alignment step failed (exit {rc})")
    if not BROWSER_HTML.exists():
        raise SystemExit(f"expected {BROWSER_HTML} to be written")


def build_figure_alignment(env: dict, species: list[str]) -> None:
    """Step 5: re-align the species drawn in the figure on their own and place
    each one's AlphaFold secondary structure onto that alignment."""
    rows = ", ".join(species + [REFERENCE_SPECIES])
    log(f"Re-aligning the compared species ({rows}) ...")
    cmd = [env_python(), str(ADD_RAW_SCOPE), str(OUTPUT_DIR)] + species + [REFERENCE_SPECIES]
    rc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=dict(env, DHRS7_SS_METHOD=SS_METHOD)).returncode
    if rc != 0:
        raise SystemExit(f"figure alignment step failed (exit {rc})")


def render_figure(residues_per_line: int, species: list[str]) -> Path:
    """Step 6: draw the figure."""
    log("Drawing the species snapshot ...")
    payload = figure.load_payload(str(BROWSER_HTML))
    svg = figure.render_dhrs7_snapshot(payload, residues_per_line=residues_per_line,
                                       species_order=species)
    if not svg:
        raise SystemExit("the renderer produced no figure; check the alignment output")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = PLOTS_DIR / figure_filename()
    svg_path.write_text(svg, encoding="utf-8")
    log(f"  wrote {svg_path}  ({len(svg):,} bytes)")
    return svg_path


def rasterize_png(svg_path: Path) -> None:
    """Optional PNG copy via a headless Chromium/Edge/Chrome. The SVG is the
    primary output and is complete on its own."""
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
            match = re.search(r'width="(\d+)" height="(\d+)"', svg_path.read_text(encoding="utf-8"))
            size = f"{match.group(1)},{match.group(2)}" if match else "2888,2876"
            try:
                subprocess.run([exe, "--headless=new", "--disable-gpu",
                                f"--screenshot={png}", f"--window-size={size}",
                                "--default-background-color=00FFFFFF", svg_path.as_uri()],
                               timeout=120, capture_output=True)
                if png.exists():
                    log(f"  wrote {png} (via {Path(exe).name})")
                    return
            except Exception as exc:  # noqa: BLE001
                log(f"  PNG step skipped: {exc}")
            return
    log("  PNG step skipped (no Chromium/Edge/Chrome found); the SVG is complete.")


def hide_unused_panels() -> None:
    """Show only the alignment and its export in the interactive browser. This is
    a stylesheet rule rather than surgery on the markup, so nothing the shared
    pipeline's JavaScript expects to find goes missing."""
    log("Restricting alignment_browser.html to the alignment view ...")
    html = BROWSER_HTML.read_text(encoding="utf-8")
    if HIDDEN_PANEL_MARKER in html:
        return
    css = (f"\n<style id=\"{HIDDEN_PANEL_MARKER}\">/* Show the alignment and its "
           "export only. */\n" + ", ".join(HIDDEN_PANEL_SELECTORS)
           + "{display:none !important}</style>\n")
    html = html.replace("</head>", css + "</head>", 1) if "</head>" in html else css + html
    BROWSER_HTML.write_text(html, encoding="utf-8")


STRUCTURE_TOGGLE_MARKER = "dhrs7-ss-toggle"

STRUCTURE_TOGGLE_SCRIPT = """
<script id="dhrs7-ss-toggle">
/* Chooses which secondary-structure assignment the browser draws. Both are
   stored in the payload, so switching only re-reads what is already there.
   This runs before the page's own script parses the payload. */
(function () {
  var node = document.getElementById('alignment-payload');
  if (!node) { return; }
  var payload;
  try { payload = JSON.parse(node.textContent); } catch (err) { return; }
  var scope = (payload.scopes || {}).selected_raw || {};
  var methods = scope.secondary_structure_methods;
  if (!methods) { return; }
  var available = Object.keys(methods);
  if (!available.length) { return; }

  var chosen = null;
  var hash = /(?:^|[#&])ss=([a-z]+)/.exec(window.location.hash || '');
  if (hash) { chosen = hash[1]; }
  if (available.indexOf(chosen) === -1) {
    try { chosen = window.localStorage.getItem('dhrs7-ss-method'); } catch (err) { chosen = null; }
  }
  if (available.indexOf(chosen) === -1) {
    chosen = scope.secondary_structure_method || available[0];
  }

  var tracks = methods[chosen] || {};
  var structure = payload.alphafold_structure || {};
  var records = (structure.comparative_secondary_structure || {}).records || [];
  records.forEach(function (record) {
    var ranges = tracks[String(record.record_id || '')];
    if (ranges) { record.mapped_ranges = ranges; record.reference_mapped_ranges = ranges; }
  });
  (scope.records || []).forEach(function (row) {
    var ranges = tracks[String(row.record_id || '')];
    if (ranges && row.is_reference && structure.secondary_structure) {
      structure.secondary_structure.ranges = ranges;
    }
  });
  scope.secondary_structure_by_record = tracks;
  scope.secondary_structure_method = chosen;
  node.textContent = JSON.stringify(payload);

  var LABELS = {torsion: 'Torsion (phi/psi)', hbond: 'Torsion + H-bonding'};

  /* The page redraws the snapshot with renderSpeciesSnapshotPanel(currentScope()). */
  function refresh() {
    try {
      if (typeof window.renderSpeciesSnapshotPanel === 'function' &&
          typeof window.currentScope === 'function') {
        window.renderSpeciesSnapshotPanel(window.currentScope());
      }
    } catch (err) { /* leave the page's own state alone if it is not ready */ }
  }
  function build() {
    if (!document.body) { return; }
    /* Sits with the snapshot export controls, so the assignment is chosen right
       where the SVG and PNG are produced. Falls back to a floating panel if the
       export row is not present. The panel is rebuilt whenever the species list
       changes, which detaches this control, so build() is idempotent and is
       called again after every render. */
    var existing = document.getElementById('dhrs7-ss-control');
    if (existing && existing.isConnected) { return; }
    var host = document.querySelector('.snapshot-actions');
    var inline = !!host;
    var box = document.createElement(inline ? 'span' : 'div');
    box.id = 'dhrs7-ss-control';
    box.style.cssText = inline
      ? 'display:inline-flex;align-items:center;gap:12px;margin-left:14px;' +
        'padding:4px 10px;border:1px solid #cbd5e1;border-radius:5px;background:#f8fafc;' +
        'font:13px Segoe UI,Tahoma,sans-serif;color:#0f172a;text-transform:none;vertical-align:middle'
      : 'position:fixed;top:12px;right:12px;z-index:99999;background:#fff;' +
        'border:1px solid #cbd5e1;border-radius:6px;padding:8px 11px;color:#0f172a;' +
        'font:13px Segoe UI,Tahoma,sans-serif;text-transform:none;box-shadow:0 2px 10px rgba(15,23,42,.15)';
    if (!inline) { host = document.body; }
    var heading = document.createElement('span');
    heading.textContent = 'Secondary structure:';
    heading.style.cssText = inline
      ? 'font-weight:700;color:#334155;text-transform:none'
      : 'display:block;font-weight:700;margin-bottom:5px;color:#334155;text-transform:none';
    box.appendChild(heading);
    available.forEach(function (name) {
      var label = document.createElement('label');
      label.style.cssText = inline
        ? 'cursor:pointer;white-space:nowrap;text-transform:none'
        : 'display:block;cursor:pointer;line-height:1.6;text-transform:none';
      var input = document.createElement('input');
      input.type = 'radio';
      input.name = 'dhrs7-ss-method';
      input.checked = (name === chosen);
      input.style.marginRight = '6px';
      input.addEventListener('change', function () {
        chosen = name;
        scope.secondary_structure_method = name;
        scope.secondary_structure_by_record = methods[name] || {};
        try { window.localStorage.setItem('dhrs7-ss-method', name); } catch (err) { /* file:// */ }
        try { window.location.hash = 'ss=' + name; } catch (err) { /* ignore */ }
        refresh();
      });
      label.appendChild(input);
      label.appendChild(document.createTextNode(LABELS[name] || name));
      box.appendChild(label);
    });
    host.appendChild(box);
  }
  /* The page resolves each row's structure track through this function, both for
     the panel on screen and for the SVG and PNG it exports. Routing it through
     the stored assignments means switching needs no reload and the exports
     always match what is displayed. */
  function install() {
    if (typeof window.snapshotSecondaryRangesForRow !== 'function') { return; }
    if (window.snapshotSecondaryRangesForRow.__dhrs7Wrapped) { return; }
    var original = window.snapshotSecondaryRangesForRow;
    var wrapped = function (row, scopeArg, lookup) {
      var ranges = (methods[chosen] || {})[String((row && row.record_id) || '')];
      if (ranges && ranges.length) {
        return ranges.map(function (range) {
          return {
            kind: range.kind,
            start: range.start_reference_position,
            end: range.end_reference_position,
            start_reference_position: range.start_reference_position,
            end_reference_position: range.end_reference_position
          };
        });
      }
      return original.call(this, row, scopeArg, lookup);
    };
    wrapped.__dhrs7Wrapped = true;
    window.snapshotSecondaryRangesForRow = wrapped;
  }

  /* Adding or removing a species rebuilds the export panel, which throws the
     control away with it. Re-attach after every render. */
  function keepAttached() {
    var render = window.renderSpeciesSnapshotPanel;
    if (typeof render !== 'function' || render.__dhrs7Wrapped) { return; }
    var wrapped = function () {
      var result = render.apply(this, arguments);
      try { build(); } catch (err) { /* never block the page's own render */ }
      return result;
    };
    wrapped.__dhrs7Wrapped = true;
    window.renderSpeciesSnapshotPanel = wrapped;
  }

  function start() {
    install();
    keepAttached();
    build();
    refresh();
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
</script>
"""


def strip_local_paths() -> None:
    """Replace absolute paths in the browser with the names they refer to.

    The underlying pipeline records where a run happened -- the output directory,
    the archive database -- and those strings end up inside the browser's data
    block. They are provenance, not something the page needs, and they carry
    whatever the directory layout happened to be on the machine that produced
    the run, so they are reduced to plain names."""
    html = BROWSER_HTML.read_text(encoding="utf-8")

    def shorten(match: "re.Match[str]") -> str:
        parts = [p for p in re.split(r"[\\/]+", match.group(0)) if p]
        return parts[-1] if parts else match.group(0)

    # A drive-lettered path, reduced to the name it points at. The drive letter
    # must not follow another letter or digit, which is what separates "E:\..."
    # from the "p:" inside "http://" -- matching a scheme here would rewrite
    # every URL in the page, including the SVG namespace.
    pattern = r'(?<![A-Za-z0-9])[A-Za-z]:[\\/][^"\s<>]*'
    updated, count = re.subn(pattern, shorten, html)
    if not count:
        return
    # Refuse to write if a URL moved; a scrub is never worth a broken page.
    for token in ("http", "https://", "www.w3.org/2000/svg"):
        if updated.count(token) != html.count(token):
            log(f"  path scrub skipped: it would have altered {token!r}")
            return
    BROWSER_HTML.write_text(updated, encoding="utf-8")
    log(f"  reduced {count} absolute path reference(s) in alignment_browser.html")


def add_structure_toggle() -> None:
    """Let the interactive browser switch between the stored assignments, and
    make it draw the same one as the figure."""
    html = BROWSER_HTML.read_text(encoding="utf-8")
    if STRUCTURE_TOGGLE_MARKER in html:
        return
    marker = '<script id="alignment-payload"'
    start = html.find(marker)
    if start == -1:
        log("  structure toggle skipped (no alignment payload found)")
        return
    end = html.find("</script>", start)
    if end == -1:
        log("  structure toggle skipped (payload script not terminated)")
        return
    end += len("</script>")
    log("Adding the secondary-structure toggle to alignment_browser.html ...")
    BROWSER_HTML.write_text(html[:end] + STRUCTURE_TOGGLE_SCRIPT + html[end:], encoding="utf-8")


def write_methods(payload: dict, species: list[str], env_info: dict | None = None) -> None:
    """Methods summary, filled in with this run's numbers."""
    meta = payload.get("meta") or {}
    scopes = payload.get("scopes") or {}
    full = scopes.get("aligned_full") or {}
    projected = scopes.get("aligned_reference_projected") or {}
    raw = scopes.get("selected_raw") or {}
    n_orthologs = meta.get("recovered_sequence_count") or len(full.get("records") or [])
    structure = payload.get("alphafold_structure") or {}
    comparative = structure.get("comparative_secondary_structure") or {}
    muscle_name = MUSCLE_EXE.name if MUSCLE_EXE else "muscle"
    drawn = ", ".join(species + [REFERENCE_SPECIES])
    torsion_text = (
        "Each residue was classified from its backbone phi/psi torsion angles using\n"
        "the standard Ramachandran windows, with helices shorter than four residues\n"
        "and strands shorter than three shown as loop. Runs are reported as the\n"
        "torsions describe them: a residue falling outside a window breaks the run\n"
        "and the break is retained rather than smoothed over.")
    hbond_text = (
        "Elements were identified from backbone hydrogen bonding using the\n"
        "Kabsch-Sander definition, so a strand is called only where a beta bridge\n"
        "partner is present; terminal residues were extended by up to two positions\n"
        "where the torsions remain in the corresponding region of Ramachandran space\n"
        "and pLDDT is at least 70. The hydrogen-bond criterion is geometric rather\n"
        "than energetic: fixed partial charges, no solvent, pH or ionic strength, and\n"
        "the amide hydrogen placed geometrically because the models carry none.")
    environment_line = environment_summary(env_info) if env_info else "recorded in environment.txt"
    structure_method_text = (torsion_text if SS_METHOD == "torsion" else hbond_text)
    alternative = "hydrogen bonding (Kabsch-Sander)" if SS_METHOD == "torsion" else "backbone torsion angles"
    structure_method_text += (
        f"\nThe alternative assignment from {alternative} is computed in the same run\n"
        "and can be displayed instead; both are stored with the alignment.")
    text = f"""# DHRS7 cross-species alignment -- Methods ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})

## Ortholog retrieval
DHRS7 orthologs were enumerated from Ensembl through the REST homology endpoint
(`type=orthologues`) starting from the human gene, returning both one2one and
one2many relationships across sampled vertebrates ({n_orthologs} orthologs). One
representative protein was taken per species as the translation of the Ensembl
canonical transcript; where a species carried more than one ortholog, the
candidate closest to the human reference in length and percent identity was
used. UniProt was queried for each protein's reviewed accession and
cross-references. For *Bos taurus* the Ensembl canonical translation
(ENSBTAP00000086519, 373 aa) is a longer isoform than the reference, so the
reference-length UniProt entry Q24K14 (339 aa, AlphaFold model AF-Q24K14-F1) was
used so that cattle is represented by a comparable sequence and structure.

## Sequence selection and alignment
Sequences were restricted to within +/- 30 aa of the human reference length and
aligned with MUSCLE v5 (`{muscle_name}`, default settings; a single deterministic
alignment, no replicate or ensemble runs). The resulting alignment
({full.get('alignment_length', '?')} columns) was projected onto human reference
positions ({projected.get('alignment_length', '?')} columns) for numbering.
Percent identities to the human reference were computed from global
(Needleman-Wunsch) pairwise alignments with the BLOSUM62 matrix (gap opening
-10, gap extension -0.5).

## Structure and secondary structure
The AlphaFold model of each species was retrieved from the AlphaFold Protein
Structure Database. These models provide predicted coordinates and a per-residue
confidence estimate (pLDDT); they carry no secondary-structure annotation, so
secondary structure was assigned from the model coordinates.
{structure_method_text}
Assignments were taken from each species' own model and placed on the alignment
shown in the figure, so the structure track follows the residues that are drawn.

## Species compared in the figure
All retrieved orthologs are aligned and stored, and the residue-level comparison
shown in the figure uses: {drawn}. These were re-aligned on their own with MUSCLE
({raw.get('alignment_length', '?')} columns, all residues, natural gaps) so the
panel shows the gaps that exist between these sequences rather than gaps
inherited from the full alignment; human is the reference row. Each species'
secondary structure was placed on this alignment directly from its own model, so
the structure track follows the residues that are drawn. Secondary structure was
available for {comparative.get('record_count', len(comparative.get('records') or []))} records overall.

## Figure annotations
* Grey glyphs above each row are that species' AlphaFold secondary structure
  (coil = helix, arrow = strand, line = loop).
* Frames mark columns where *Homo sapiens* and *Danio rerio* carry the identical
  residue while every rodent in the panel differs.
* Asterisks mark the subset of those columns where *Bos taurus* also shares the
  residue, i.e. positions identical in human, zebrafish and cattle while both
  rodents diverge.
* The conservation line follows the Clustal convention: `*` identical, `:`
  strongly similar, `.` weakly similar across the drawn rows.

## Software
{environment_line}
The exact versions above were recorded from the environment that produced these
files and are repeated in `environment.txt` beside them.

## Availability
`python dhrs7_alignment.py` rebuilds the alignment, this file and the figure from
the source databases. Figure: `plots/dhrs7_species_snapshot.svg`.
"""
    (OUTPUT_DIR / methods_filename()).write_text(text, encoding="utf-8")
    log(f"  wrote {OUTPUT_DIR / methods_filename()}")


def collect_plots() -> None:
    """Gather the remaining figures alongside the main panel."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("reference_domain_architecture.svg", "conservation_scan.svg",
                 "property_conservation.svg"):
        source = OUTPUT_DIR / name
        if source.exists():
            shutil.copy2(source, PLOTS_DIR / name)
    log(f"  figures collected in {PLOTS_DIR}")


def prune_unused_models() -> None:
    """Drop model files no species in this run refers to.

    Models are fetched into a cache shared with other genes, and the copy step
    can bring across neighbours that belong to a different protein entirely.
    Anything the run's own secondary-structure record does not name is not part
    of this analysis and should not travel with it."""
    models = OUTPUT_DIR / "comparative_alphafold_models"
    catalogue = OUTPUT_DIR / "comparative_alphafold_secondary_structure.json"
    if not models.is_dir() or not catalogue.exists():
        return
    try:
        records = json.loads(catalogue.read_text(encoding="utf-8")).get("records") or []
    except (OSError, ValueError) as exc:
        log(f"  model check skipped ({exc})")
        return
    referenced = set()
    for record in records:
        for key in ("alphafold_entry_id", "model_filename"):
            name = str(record.get(key) or "").strip()
            if name:
                referenced.add(Path(name).name.replace("-model_v6.pdb", "").replace(".pdb", ""))
    removed = 0
    for model in models.glob("*.pdb"):
        if model.name.replace("-model_v6.pdb", "") not in referenced:
            try:
                model.unlink()
                removed += 1
            except OSError:
                pass
    if removed:
        log(f"  removed {removed} model(s) belonging to other genes")


def discard_run_archive() -> None:
    """Remove the cross-run SQLite archive the shared pipeline writes.

    The pipeline appends every run it has ever done to one database next to
    whatever it was started from. It is a working file of that pipeline, it
    accumulates other genes, and it grows into hundreds of megabytes; nothing in
    this analysis reads it back."""
    archive = SCRIPT_DIR / "phylo_runs.sqlite"
    if not archive.exists():
        return
    size = archive.stat().st_size / 1024 ** 2
    try:
        archive.unlink()
        log(f"  removed the shared run archive ({size:.0f} MB); it is not part of the results")
    except OSError as exc:
        log(f"  could not remove {archive.name}: {exc}")


def tidy_run(keep_all: bool) -> None:
    """Leave the documented output set behind, whatever happened during the run.

    This runs even when a step fails. A half-finished run used to leave every
    intermediate the shared pipeline emits sitting in the output folder, which
    made it impossible to tell what belonged to the analysis."""
    if not OUTPUT_DIR.exists():
        return
    prune_unused_models()
    prune_output(keep_all)
    if not keep_all:
        discard_run_archive()


def prune_output(keep_all: bool) -> None:
    """Drop anything outside the documented output set (see KEEP_OUTPUT_*)."""
    if keep_all:
        log("  keeping every file the shared pipeline produced (--keep-all)")
        return
    removed = 0
    for entry in sorted(OUTPUT_DIR.iterdir()):
        keepable = KEEP_OUTPUT_FILES | {methods_filename()}
        keep = (entry.name in KEEP_OUTPUT_DIRS) if entry.is_dir() else (entry.name in keepable)
        if keep:
            continue
        try:
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
            removed += 1
        except OSError as exc:
            log(f"  could not remove {entry.name}: {exc}")
    log(f"  removed {removed} file(s) outside the documented output set")


REQUIRED_PACKAGES = (
    ("Bio", "biopython", "reading and writing sequences, alignment handling"),
    ("pandas", "pandas", "ortholog and conservation tables"),
    ("numpy", "numpy", "coordinate geometry and conservation arithmetic"),
    ("matplotlib", "matplotlib", "the supporting plots"),
    ("requests", "requests", "Ensembl, UniProt and AlphaFold requests"),
)

MUSCLE_HELP = """MUSCLE v5 was not found. It is a separate program, not a Python
  package, and the alignment cannot run without it.

    macOS / Linux :  conda install -c bioconda muscle
    Windows       :  download the executable from
                     https://github.com/rcedgar/muscle/releases

  The download keeps its own name (e.g. muscle-win64.v5.3.exe); that is fine --
  just drop it in a "tools" folder beside this script, or anywhere on PATH, and
  it is picked up automatically. A conda-installed MUSCLE in this environment is
  found even without activating it. To point straight at the file instead, set
  MUSCLE_EXE to its full path, pass --muscle PATH, or use the Browse button in
  the window (MSA_GUI.py).

  Apple Silicon: if conda has no osx-arm64 build, take the macOS binary from the
  same page, then  chmod +x muscle  (and  xattr -d com.apple.quarantine muscle
  if Gatekeeper blocks it)."""


def collect_environment() -> dict:
    """Exact versions of everything this run depends on."""
    import platform

    info: dict = {
        "recorded": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "packages": {},
        "muscle": None,
        "muscle_path": str(MUSCLE_EXE) if MUSCLE_EXE else None,
    }
    for module, package, _ in REQUIRED_PACKAGES:
        try:
            info["packages"][package] = getattr(__import__(module), "__version__", "unknown")
        except ImportError:
            info["packages"][package] = "not installed"
    if MUSCLE_EXE:
        try:
            finished = subprocess.run([str(MUSCLE_EXE), "-version"],
                                      capture_output=True, text=True, timeout=30)
            lines = (finished.stdout or finished.stderr or "").strip().splitlines()
            if lines:
                info["muscle"] = lines[0].strip()
        except Exception:  # noqa: BLE001
            pass
    return info


def environment_summary(info: dict) -> str:
    """One-line-per-item rendering used in the methods text."""
    parts = [f"Python {info['python']}"]
    parts += [f"{name} {version}" for name, version in info["packages"].items()]
    if info.get("muscle"):
        parts.append(info["muscle"])
    return ", ".join(parts)


def write_environment_record(info: dict) -> None:
    """Record the exact environment beside the results it produced.

    The environment file installs a working set of packages but does not pin
    them, so the versions that actually ran are written out here. That makes the
    link between a set of outputs and the software that produced them checkable
    after the fact rather than assumed."""
    lines = [
        "Environment used for this run",
        "=" * 29,
        "",
        f"recorded   {info['recorded']}",
        f"platform   {info['platform']}",
        f"python     {info['python']}",
        "",
        "packages",
    ]
    for name, version in info["packages"].items():
        lines.append(f"  {name:<12} {version}")
    lines += ["", "external"]
    lines.append(f"  muscle       {info.get('muscle') or 'not recorded'}")
    if info.get("muscle_path"):
        lines.append(f"  muscle path  {Path(info['muscle_path']).name}")
    lines.append("")
    (OUTPUT_DIR / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"  wrote {OUTPUT_DIR / 'environment.txt'}")


def check_environment(verbose: bool = True) -> list[str]:
    """Confirm everything needed is in place. Returns a list of problems."""
    problems: list[str] = []
    report: list[str] = []

    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info < (3, 9):
        problems.append(f"Python {version} is too old; this needs 3.9 or newer (3.10 is tested).")
    report.append(f"  Python {version}")

    for module, package, purpose in REQUIRED_PACKAGES:
        try:
            imported = __import__(module)
            release = getattr(imported, "__version__", "?")
            report.append(f"  {package:<12} {release:<10} {purpose}")
        except ImportError:
            problems.append(f"The {package} package is missing ({purpose}).")

    if MUSCLE_EXE:
        major, version_line = muscle_version_info(MUSCLE_EXE)
        report.append(f"  MUSCLE       {version_line or 'found'}")
        report.append(f"               {MUSCLE_EXE}")
        if major is not None and major < MUSCLE_REQUIRED_MAJOR:
            problems.append(
                f"MUSCLE at {MUSCLE_EXE} is v{major}, but this pipeline needs "
                f"MUSCLE v{MUSCLE_REQUIRED_MAJOR}. Versions 3 and 5 take different, "
                f"incompatible options (v3 has no -align), and the reference figure "
                f"was built with v5 -- v3 would produce a different alignment. "
                f"Install v5, and if the v{major} copy stays on PATH point past it "
                f"with --muscle PATH or MUSCLE_EXE.\n\n" + MUSCLE_HELP)
        elif major is None:
            report.append(f"               (could not read version; "
                          f"v{MUSCLE_REQUIRED_MAJOR} is required)")
    else:
        name_hint = "muscle.exe or muscle*.exe" if os.name == "nt" else "muscle or muscle*"
        drive_count = len(_fixed_drive_roots())
        looked = ["  Looked for it here:",
                  f"    MUSCLE_EXE      {os.environ.get('MUSCLE_EXE') or 'not set'}",
                  "    on PATH         not found",
                  f"    in these folders (as {name_hint}):"]
        looked += [f"      {directory}" for directory in muscle_search_dirs()]
        looked.append(f"    plus common install spots and every conda environment "
                      f"on {drive_count} drive(s).")
        looked.append("    Add --scan-drives to search every drive top to bottom.")
        problems.append(MUSCLE_HELP + "\n\n" + "\n".join(looked))

    if verbose:
        print("Environment")
        for line in report:
            print(line)
        if problems:
            print("\nNot ready to run:")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print("\nEverything needed is present.")
            print("A full build also needs internet access to Ensembl, UniProt and "
                  "AlphaFold;\n  --figure-only redraws without any network access.")
    return problems


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dhrs7_alignment.py",
        description="Build the DHRS7 cross-species alignment and species-snapshot figure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Set MUSCLE_EXE if MUSCLE is not on PATH; set REQUESTS_CA_BUNDLE "
               "behind a TLS-inspecting proxy.")
    parser.add_argument("--gene", default=GENE, metavar="SYMBOL",
                        help="gene symbol to analyse (default: %(default)s). This package is "
                             "built around DHRS7; other genes run the same pipeline, but the "
                             "pinned records and the panel annotations are DHRS7-specific.")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="output directory (default: <GENE>_Output beside this script)")
    parser.add_argument("--species", default=",".join(figure.DHRS7_SELECTED_ORDER),
                        help="comma-separated species drawn above the human reference row, "
                             "in row order (default: %(default)s)")
    parser.add_argument("--residues-per-line", type=int, default=70, metavar="N",
                        help="residues per alignment block in the figure (default: %(default)s)")
    parser.add_argument("--figure-only", action="store_true",
                        help="redraw from an existing output directory; no network access")
    parser.add_argument("--keep-all", action="store_true",
                        help="keep every file the shared pipeline can produce")
    parser.add_argument("--muscle", default=None, metavar="PATH",
                        help="path to the MUSCLE executable")
    parser.add_argument("--scan-drives", action="store_true",
                        help="if MUSCLE is not found the quick way, walk every "
                             "drive to locate it (slower; use with --check)")
    parser.add_argument("--check", action="store_true",
                        help="report whether the environment is ready and exit")
    parser.add_argument("--ss", choices=("torsion", "hbond"), default="torsion",
                        help="secondary-structure assignment drawn in the figure: "
                             "'torsion' uses backbone phi/psi windows (default); 'hbond' "
                             "requires a backbone hydrogen-bond partner and extends frayed "
                             "ends by torsion. Both are computed and stored either way.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global MUSCLE_EXE, CA_BUNDLE, SS_METHOD, GENE
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    GENE = (args.gene or GENE).strip().upper()
    outdir = args.outdir if args.outdir is not None else SCRIPT_DIR / f"{GENE}_Output"
    resolve_paths(Path(outdir).resolve())
    species = [s.strip().lower() for s in args.species.split(",") if s.strip()
               and s.strip().lower() != REFERENCE_SPECIES]
    if not species:
        raise SystemExit("--species must name at least one species besides the human reference")
    SS_METHOD = args.ss
    MUSCLE_EXE = find_muscle(args.muscle)
    if MUSCLE_EXE is None and args.scan_drives:
        print("Scanning all drives for MUSCLE (this can take a moment) ...")
        MUSCLE_EXE = find_muscle(args.muscle, deep=True,
                                 announce=lambda where: print(f"  searching {where}"))
        if MUSCLE_EXE:
            print(f"  found: {MUSCLE_EXE}")
        else:
            print("  not found on any drive.")
    CA_BUNDLE = find_ca_bundle()
    keep_all = args.keep_all or bool(os.environ.get("DHRS7_KEEP_ALL"))

    if args.check:
        return 0 if not check_environment() else 1

    problems = check_environment(verbose=False)
    if problems:
        print("Cannot start; the environment is not ready.")
        print()
        for problem in problems:
            print(f"  - {problem}")
            print()
        print("Run  python dhrs7_alignment.py --check  for the full report.")
        return 1

    log(f"=== {GENE} alignment -> {OUTPUT_DIR} ===")
    log(f"Secondary structure drawn from: {SS_METHOD}")
    env = build_subprocess_env()

    if args.figure_only:
        if not BROWSER_HTML.exists():
            raise SystemExit(f"--figure-only needs an existing build; {BROWSER_HTML} not found")
        log("Redrawing from the existing build (no network access).")
    else:
        if MUSCLE_EXE:
            log(f"MUSCLE: {MUSCLE_EXE}")
        else:
            log("MUSCLE was not found on PATH or via MUSCLE_EXE; the alignment step "
                "will fail unless MUSCLE is installed.")
        run_retrieval_and_alignment(env, keep_all)

    try:
        build_figure_alignment(env, species)
        svg_path = render_figure(args.residues_per_line, species)
        rasterize_png(svg_path)
        hide_unused_panels()
        strip_local_paths()
        add_structure_toggle()
        environment_info = collect_environment()
        write_environment_record(environment_info)
        write_methods(figure.load_payload(str(BROWSER_HTML)), species, environment_info)
        if not args.figure_only:
            collect_plots()
    finally:
        # Tidy up whether or not the steps above succeeded, so the output folder
        # never keeps the shared pipeline's intermediates.
        if not args.figure_only:
            tidy_run(keep_all)

    log("=== done ===")
    log(f"Figure : {svg_path}")
    log(f"Browser: {BROWSER_HTML}")
    log(f"Methods: {OUTPUT_DIR / methods_filename()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
