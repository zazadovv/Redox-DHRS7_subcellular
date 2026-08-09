# DHRS7 cross-species alignment

Builds the DHRS7 ortholog alignment and the annotated species panel we use for
residue-level comparison across vertebrates. A single command does the whole
thing: ortholog retrieval from Ensembl, sequence selection, MUSCLE alignment,
AlphaFold models and secondary structure, and the figure.

```
python dhrs7_alignment.py
```

The panel compares five species, drawn in this order:

| Row | Species | Record |
|-----|---------|--------|
| 1 | mouse — *Mus musculus* | ENSMUSP00000021512 |
| 2 | rat — *Rattus norvegicus* | ENSRNOP00000007645 |
| 3 | cattle — *Bos taurus* | ENSBTAP00000086519  |
| 4 | zebrafish — *Danio rerio* | ENSDARP00000004163 |
| 5 | human — *Homo sapiens* (reference) | ENSP00000216500 |

Every ortholog retrieved is aligned and kept, so any species can be pulled out of
the output or swapped into the panel. The five above are the default comparison.

`DHRS7_Output/` in this repository is a completed run, so the alignment, the
figure and the per-species structures can be inspected without installing or
running anything. Running the analysis yourself overwrites it in place.


## Setting it up, step by step

Nothing here is configured for a particular machine. Download the folder, work
inside it, and the outputs appear beside the scripts.

**1. Get the code.** Download or clone the repository and open a terminal in the
folder that contains `dhrs7_alignment.py`. On Windows use an Anaconda Prompt.

**2. Create the environment.** This installs Python and the five packages the
analysis uses.

```
conda env create -f phylo.yml
conda activate phylo
```

Every later command assumes `phylo` is the active environment. If you prefer
pip, any Python 3.9+ environment with `biopython pandas numpy matplotlib
requests` works just as well.

**3. Install MUSCLE.** This is a standalone program, not a Python package, and
the alignment cannot run without it. It must be **MUSCLE v5** — the older v3.8
uses different, incompatible options (`-in`/`-out` instead of `-align`) and would
produce a different alignment, so `--check` refuses it and asks for v5.

- macOS and Linux: `conda install -c bioconda muscle`
- Windows: download the executable from
  <https://github.com/rcedgar/muscle/releases>. It keeps its own name (for
  example `muscle-win64.v5.3.exe`) — that is fine; drop it in a folder called
  `tools` beside the scripts, or anywhere on `PATH`, and it is picked up
  automatically. There is no Windows build on bioconda, so use the release page.
- Apple Silicon: if conda offers no `osx-arm64` build, take the macOS binary
  from the same page, then `chmod +x muscle`, and if Gatekeeper blocks it,
  `xattr -d com.apple.quarantine muscle`.

The scripts look for MUSCLE in this order: the `--muscle` option, then
`MUSCLE_EXE`, then `PATH`, then the bin folders of the Python environment you are
running in (a `conda install`-ed MUSCLE is found even without activating the
environment), then a `tools` folder — checked beside the scripts and in the next
few directories above them, so a shared `tools` folder kept alongside the project
is found too. A file whose name simply starts with `muscle` counts, so the
release binary works under its downloaded name. `--check` prints every folder it
looked in when it cannot find one; the window (`MSA_GUI.py`) also has a **Browse**
button to point straight at the file.

**4. Check the setup before running anything.**

```
python dhrs7_alignment.py --check
```

This prints the Python version, each package and its version, and the MUSCLE
build it found. If something is missing it says exactly what and how to install
it, and exits non-zero. A normal run performs the same check first and stops
early rather than failing halfway through.

**5. Run it.** Either double-click **`Run_DHRS7.bat`** (Windows), which opens a
window with everything pre-set for DHRS7 — species, output folder, MUSCLE
location, block width and the structure method — and a Run button, or use the
command line:

```
python MSA_GUI.py            # the same window, from any platform
python dhrs7_alignment.py    # no window, straight to the build
```

The window and the command line do exactly the same work and write exactly the
same output; the window simply runs the command for you and streams its log.
`Run_DHRS7.bat` finds a Python that has the required packages on its own, so
nothing has to be activated first.

The gene, the species and every parameter are already set for DHRS7, so there is
nothing to configure. The run takes roughly fifteen minutes, most of it waiting
on Ensembl, UniProt and AlphaFold, and prints each step as it goes. Results land
in `DHRS7_Output/` beside the script; `--outdir somewhere/else` puts them
anywhere you like.

**6. Look at the results.** The figure is
`DHRS7_Output/plots/dhrs7_species_snapshot.svg` (with a `.png` alongside it if a
Chromium-based browser is installed). Open
`DHRS7_Output/alignment_browser.html` in any web browser to explore the full
alignment, choose species and export the panel yourself.

Redrawing afterwards does not need the network:

```
python dhrs7_alignment.py --figure-only --residues-per-line 60
```

Behind a TLS-inspecting proxy, set `REQUESTS_CA_BUNDLE` or leave a
`ca_bundle.pem` beside the scripts and it is picked up automatically.

### How the pieces fit together

`dhrs7_alignment.py` is the only script that needs running. It drives the
others: `gene_phylo_conservation_pipeline.py` does the retrieval, alignment and
structure work and writes the interactive browser through
`gene_phylo_conservation_archive.py`; `add_raw_alignment_scope.py` then
re-aligns the panel species and builds their structure tracks using
`secondary_structure.py`; and `dhrs7_snapshot_figure.py` draws the figure. None
of them has to be called by hand.


## What the run does

1. Enumerates DHRS7 orthologs from the Ensembl REST homology endpoint and keeps
   one representative protein per species — the canonical transcript
   translation, or for species with several orthologs the candidate closest to
   the human reference.
2. Restricts to sequences within ±30 aa of the human reference length.
3. Aligns with MUSCLE and projects onto human reference numbering.
4. Retrieves each species' AlphaFold model and assigns secondary structure from
   the model backbone.
5. Re-aligns the five panel species on their own and places each one's secondary
   structure on that alignment.
6. Draws the figure and writes the interactive browser and a methods summary
   carrying the numbers from the run.


## Options

```
python dhrs7_alignment.py --check                     # is everything installed?
python dhrs7_alignment.py --figure-only               # redraw, no network
python dhrs7_alignment.py --species danio_rerio       # zebrafish vs human only
python dhrs7_alignment.py --residues-per-line 60      # narrower blocks
python dhrs7_alignment.py --ss hbond                  # alternative structure assignment
python dhrs7_alignment.py --outdir run2 --keep-all    # keep every intermediate
python dhrs7_alignment.py --gene <SYMBOL>             # another gene, same pipeline
```

The gene defaults to DHRS7 and the window keeps it fixed until the preset
checkbox is cleared. Another gene runs through the same steps and writes
`<GENE>_Output`, but the pinned ortholog records and the frames and asterisks on
the panel are specific to the DHRS7 comparison and do not carry over, so treat
other genes as exploratory.

`--figure-only` reuses an existing output directory, so changes to the panel
redraw in seconds without touching the databases. Windows users can double-click
`Run_DHRS7.bat`; on macOS and Linux use `chmod +x run_dhrs7.sh && ./run_dhrs7.sh`.
Both pass arguments through.

To redraw from an alignment that already exists, without the driver:

```
python dhrs7_snapshot_figure.py DHRS7_Output/alignment_browser.html out.svg
```


## Output

Written to `DHRS7_Output/`:

| File | Contents |
|------|----------|
| `plots/dhrs7_species_snapshot.svg` (and `.png`) | the panel |
| `alignment_browser.html` | interactive alignment over all retrieved species |
| `METHODS_DHRS7.md` | methods summary for that run |
| `orthologs.tsv`, `sequence_retrieval.tsv` | every ortholog retrieved |
| `proteins.fasta` | one representative protein per species |
| `aligned.fasta`, `aligned_reference_projected.fasta` | full alignments |
| `comparative_alphafold_secondary_structure.json`, `comparative_alphafold_models/` | per-species structures |
| `conservation_*.csv`, `property_conservation.*` | per-position conservation |
| `environment.txt` | exact software versions that produced this run |

`phylo.yml` deliberately leaves versions unpinned so the environment resolves on
any platform, which means it records what to install rather than what ran. Each
run therefore writes `environment.txt` next to its results — Python, every
package and the MUSCLE build, with a timestamp — and the same list is repeated in
`METHODS_DHRS7.md`. The software behind a given figure can be read off the output
itself instead of assumed. The released run used Python 3.10.20, biopython 1.87,
pandas 2.3.3, numpy 1.24.4, matplotlib 3.10.8, requests 2.33.1 and MUSCLE 5.3
(build d9725ac); `phylo.yml` lists those same versions in a comment for anyone
who wants to pin them exactly.

A run always leaves the same set of files; `--keep-all` additionally keeps the
broader analyses the underlying pipeline can produce.

Reading the panel: grey glyphs above each row are that species' secondary
structure (spiral = helix, arrow = strand, line = loop). Frames mark columns
where human and zebrafish share a residue that differs in both rodents;
asterisks mark the subset where cattle shares it as well. The conservation line
follows the Clustal convention (`*` identical, `:` strongly similar, `.` weakly
similar).


## Method notes

**Species in the panel.** Two rodents rather than one, so a rodent-specific
substitution has to appear in both before it is marked. Zebrafish gives the deep
outgroup and cattle separates "lost in rodents" from "changed in mammals".
Adding further species made the panel harder to read at full sequence length
without changing which positions were marked.

**Length window.** ±30 aa against the human reference. Wider windows let in
fragment and read-through predictions that open long spurious gaps; narrower
ones start discarding real orthologs.

**Cattle sequence.** The Ensembl canonical translation for *Bos taurus*
(ENSBTAP00000086519, 373 aa) is a long isoform that the length window rejects,
which would drop cattle from the comparison. The reference-length UniProt entry
Q24K14 (339 aa) is used instead so cattle is represented by a comparable
sequence and model.

**Panel alignment.** The five panel species are re-aligned on their own rather
than sliced out of the full alignment, so the gaps shown are the ones that exist
between these sequences and not gaps inherited from distant orthologs.

**Secondary structure.** Assigned per species from its own AlphaFold model and
placed on the alignment being drawn. Two assignments are computed and stored:

- `torsion` (default) — backbone φ/ψ windows, with runs shorter than four
  (helix) or three (strand) residues dropped. Interruptions are left alone: a
  residue whose torsions fall outside the window genuinely breaks the run, and
  closing such a break would join two elements the backbone keeps apart. The
  track is meant to report the torsions, so it is left jagged where they are.
- `hbond` — requires a backbone hydrogen-bond partner (Kabsch–Sander), with
  frayed terminal residues extended by torsion where model confidence is at
  least 70 pLDDT.

The two differ mainly at disordered termini, which torsion windows report as
strand because they judge each residue in isolation. Note that the hydrogen-bond
criterion is geometric, not energetic: fixed partial charges, no solvent, no pH
or ionic strength, and the amide hydrogen placed geometrically because AlphaFold
models carry no hydrogens. Both are stored in every run, and the alignment
browser has a switch beside the export buttons so the panel and the exported
SVG/PNG can be produced either way.


## Files

| File | Role |
|------|------|
| `dhrs7_alignment.py` | entry point and command line |
| `dhrs7_snapshot_figure.py` | figure renderer, standard library only |
| `secondary_structure.py` | secondary-structure assignment |
| `add_raw_alignment_scope.py` | panel alignment and structure tracks |
| `gene_phylo_conservation_pipeline.py` | retrieval, alignment, structures |
| `gene_phylo_conservation_archive.py` | shared helpers and alignment browser |
| `phylo.yml` | conda environment |
| `MSA_GUI.py` | the window, if you prefer not to use a terminal |
| `Run_DHRS7.bat`, `run_dhrs7.sh` | launchers |
