# DHRS7 ortholog alignment & species-snapshot figure — reproducibility package

This repository lets anyone reproduce, from scratch, the DHRS7 cross-species
protein alignment and the **species-snapshot figure** used in the paper. It
retrieves every DHRS7 ortholog from Ensembl, fetches one representative protein
per species, aligns them with MUSCLE, maps each species' AlphaFold secondary
structure onto human reference positions, and renders the publication figure —
with a single command.

Every retrieved ortholog is aligned and stored, so you can pull out **any**
species you like from the outputs. The paper figure itself is fixed to five
species, drawn in this order:

1. **mouse** — *Mus musculus* (ENSMUSP00000021512)
2. **rat** — *Rattus norvegicus* (ENSRNOP00000007645)
3. **cattle** — *Bos taurus* (ENSBTAP00000086519 / UniProt **Q24K14**, 339 aa)
4. **zebrafish** — *Danio rerio* (ENSDARP00000004163)
5. **human** — *Homo sapiens* (ENSP00000216500) — the reference row

---

## Quick start

```bash
# 1. Create and activate the environment (Python 3.10 + the 5 packages)
conda env create -f phylo.yml
conda activate phylo

# 2. Install MUSCLE (the aligner)
#    macOS / Linux:
conda install -c bioconda muscle
#    Windows: download muscle.exe from https://github.com/rcedgar/muscle/releases
#             and put it on PATH, or in a tools\ folder next to these scripts,
#             or set MUSCLE_EXE to its full path.

# 3. Run the reproduction
python Reproduce_DHRS7_Figure.py
#    (Windows: double-click Run_DHRS7.bat  |  macOS/Linux: chmod +x run_dhrs7.sh && ./run_dhrs7.sh)
```

Requires internet access to Ensembl, UniProt and AlphaFold. The run takes a few
minutes. Results land in **`DHRS7_Output/`**.

> Behind a TLS-inspecting corporate proxy? Set `REQUESTS_CA_BUNDLE` to your CA
> bundle, or drop a `ca_bundle.pem` next to the scripts — it will be picked up
> automatically.

---

## Installing dependencies (step by step)

### 1. Conda
If you don't already have it, install **Miniconda** (or Miniforge) for your OS from
<https://docs.conda.io/en/latest/miniconda.html>. Then open an **Anaconda Prompt**
(Windows) or a normal terminal (macOS/Linux) in the folder containing these files.

### 2. Python packages
```bash
conda env create -f phylo.yml      # creates an environment named "phylo"
conda activate phylo
```
This installs Python 3.10 plus the only five packages the reproduction needs —
**biopython, pandas, numpy, matplotlib, requests**. Everything else the scripts use
is in the Python standard library.

> Prefer pip? Into a Python ≥3.10 virtual environment:
> `pip install biopython pandas numpy matplotlib requests` (you still need MUSCLE, below).

### 3. MUSCLE (the aligner) — required
The pipeline calls **MUSCLE v5** to align sequences. It is installed separately
because it has no Windows conda build:

- **macOS / Linux** — install it into the environment:
  ```bash
  conda install -n phylo -c bioconda muscle
  ```
  **Apple Silicon (M1/M2/M3):** if conda can't find a build for `osx-arm64`, download
  the macOS `muscle` binary from the releases page
  (<https://github.com/rcedgar/muscle/releases>) instead, then:
  ```bash
  chmod +x muscle                       # make it executable
  xattr -d com.apple.quarantine muscle  # only if Gatekeeper blocks it
  export MUSCLE_EXE="$PWD/muscle"        # or move it onto your PATH
  ```
- **Windows** — download `muscle.exe` from the official releases
  (<https://github.com/rcedgar/muscle/releases>) and do **one** of:
  - put it on your `PATH`, **or**
  - drop it in a `tools\` folder next to these scripts, **or**
  - set the `MUSCLE_EXE` environment variable to its full path.

The script locates MUSCLE automatically in this order: `MUSCLE_EXE` → your `PATH` →
a local `tools/` folder.

### 4. (Optional) a Chromium browser
If Chrome/Edge/Chromium is installed, the script also saves a **PNG** copy of the
figure (rasterized from the SVG headlessly). This is optional — the **SVG** is the
primary deliverable and needs no browser.

### 5. Verify the setup
```bash
muscle -version
python -c "import Bio, pandas, numpy, matplotlib, requests; print('Python deps OK')"
```
You also need internet access to **Ensembl**, **UniProt** and **AlphaFold**.

---

## Packages the reproduction uses (final)

The reproduction is deliberately lightweight — **five** third-party packages plus the
Python standard library:

| Package | Version tested | Used for |
|---|---|---|
| **Python** | 3.10 | interpreter |
| **biopython** (`Bio`) | 1.87 | reading/writing sequences, pairwise & multiple-alignment handling |
| **pandas** | 2.3.3 | tabular data (orthologs, conservation, metadata tables) |
| **numpy** | 1.24.4 | numeric arrays and conservation math |
| **matplotlib** | 3.10.8 | SVG/PNG plots (Agg backend; `matplotlib-base` is sufficient — no GUI) |
| **requests** | 2.33.1 | Ensembl / UniProt / AlphaFold REST calls |

**External (non-Python):** **MUSCLE v5** (alignment). **Optional:** a Chromium/Edge/Chrome
browser (PNG rasterization only).

Everything else the scripts import — `json`, `re`, `pathlib`, `subprocess`, `sqlite3`,
`concurrent.futures`, `xml`, `html`, `math`, `csv`, … — ships with Python. No
scipy / scikit-learn / torch / prody / etc. are required (those belong to a separate
functional-divergence pilot that is **not** part of this package). The pure-Python
figure renderer (`dhrs7_snapshot_figure.py`) uses the standard library only.

---

## What the run does

1. **Ortholog retrieval** — enumerates DHRS7 orthologs from the Ensembl REST
   homology endpoint (253 orthology relationships across sampled vertebrates),
   taking one representative protein per species (the canonical transcript
   translation; for species with several orthologs, the candidate closest to the
   human reference). For *Bos taurus*, whose Ensembl canonical translation runs
   longer than the reference, the reference-length UniProt ortholog **Q24K14**
   (339 aa) is used.
2. **Alignment** — filters to ±30 aa of the human reference length (187
   representative sequences) and aligns them with **MUSCLE**, then projects the
   alignment onto human reference positions for numbering.
3. **AlphaFold secondary structure** — retrieves each species' AlphaFold model
   (145 models), extracts its ribbon secondary structure (helix / strand / loop),
   and maps it onto human reference positions with a sequence-aware alignment.
4. **Five-species figure** — re-aligns just the five figure species cleanly with
   MUSCLE and renders the species-snapshot SVG in **pure Python** (no browser, no
   third-party libraries — see `dhrs7_snapshot_figure.py`).

### The figure's annotations

- **Grey glyphs** above each row — that species' AlphaFold secondary structure
  (cursive helix, arrow = strand, line = loop).
- **Black frames** — columns where human and zebrafish carry the identical
  residue but both rodents (mouse, rat) differ (20 sites).
- **Red asterisks** — the subset where cattle *also* shares that residue:
  identical across all three catalytically-active species (human, zebrafish,
  cattle) while both inactive rodents diverge — candidate activity-linked sites
  (15 sites).
- **Conservation row** — Clustal-style `*` identical / `:` strongly similar /
  `.` weakly similar across the five rows.

---

## Outputs (`DHRS7_Output/`)

| File | What it is |
|---|---|
| `plots/dhrs7_species_snapshot.svg` (+ `.png`) | **the publication figure** |
| `alignment_browser.html` | interactive alignment browser + species-snapshot export |
| `METHODS_DHRS7.md` | paper-ready methods text, auto-filled with the run's numbers |
| `orthologs.tsv`, `sequence_retrieval.tsv` | every ortholog retrieved, per species |
| `proteins.fasta` | one representative protein sequence per species |
| `aligned.fasta`, `aligned_reference_projected.fasta` | the full multiple alignments |
| `comparative_alphafold_secondary_structure.json`, `comparative_alphafold_models/` | per-species AlphaFold SS + models |
| `conservation_*.csv`, `property_conservation.*` | per-position conservation tables and plots |

A reference run is included in `DHRS7_Output/` so you can browse all the data
without re-running. Re-running `Reproduce_DHRS7_Figure.py` regenerates the
alignment and figure fresh from Ensembl / UniProt / AlphaFold and yields exactly
this set — the run keeps the alignment + species-snapshot deliverable (and the
per-species data below) and prunes the general pipeline's out-of-scope extras
(phylogenetic tree, clade/Fourier/evolutionary-segment analyses, large alignment
matrices). To keep the general pipeline's extra analyses instead of pruning them,
set the environment variable `DHRS7_KEEP_ALL=1` before running. (The V11
functional-divergence pilot is always skipped — its code is not part of this
package, so no `v11_*` files are ever produced.)

### Retrieving any species you want

Because all orthologs are aligned and stored, you are not limited to the five
figure species:

- Open `DHRS7_Output/alignment_browser.html` in any web browser to explore the
  full alignment interactively and export any subset.
- `aligned.fasta` / `aligned_reference_projected.fasta` contain every aligned
  species; `orthologs.tsv` and `sequence_retrieval.tsv` map species → accession.
- To change which species appear in the snapshot figure, pass the record IDs (in
  your desired row order) to the raw-alignment step, e.g.:
  ```bash
  python add_raw_alignment_scope.py DHRS7_Output ENSMUSP00000021512 ENSP00000216500
  python dhrs7_snapshot_figure.py DHRS7_Output/alignment_browser.html out.svg
  ```

### Reproducing only the figure (offline)

The renderer is pure standard-library Python and works directly on an existing
`alignment_browser.html`, with no network:

```bash
python dhrs7_snapshot_figure.py DHRS7_Output/alignment_browser.html figure.svg
```

---

## Files

| File | Role |
|---|---|
| `Reproduce_DHRS7_Figure.py` | **single entry point** — orchestrates the whole reproduction |
| `dhrs7_snapshot_figure.py` | pure-Python species-snapshot SVG renderer |
| `gene_phylo_conservation_pipeline.py` | ortholog retrieval + MUSCLE alignment + AlphaFold SS |
| `gene_phylo_conservation_archive.py` | helpers + interactive alignment-browser builder |
| `add_raw_alignment_scope.py` | builds the clean five-species raw MUSCLE re-alignment |
| `phylo.yml` | conda environment specification (creates the `phylo` env) |
| `Run_DHRS7.bat` / `run_dhrs7.sh` | thin launchers for Windows / macOS-Linux |

## Requirements

See **[Installing dependencies](#installing-dependencies-step-by-step)** and
**[Packages the reproduction uses](#packages-the-reproduction-uses-final)** above:
Python 3.10 + biopython / pandas / numpy / matplotlib / requests (via `phylo.yml`),
**MUSCLE v5** (on `PATH` or via `MUSCLE_EXE`), and internet access to Ensembl,
UniProt and AlphaFold. A Chromium/Edge/Chrome browser is optional (PNG
rasterization only; the SVG is the primary deliverable).

## Citation

If you use this analysis, please cite the accompanying paper. <!-- Add the full
citation / DOI here once available. -->
