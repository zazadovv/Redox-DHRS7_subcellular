# DHRS7 species-snapshot figure — Methods

## Sequence retrieval and alignment

Analyses used Python v.3.10.20. Human *DHRS7* orthologues were retrieved from
Ensembl (release 116) via the REST homology endpoint (`type=orthologues`) using
*requests* v.2.33.1, yielding 253 one2one and one2many orthologues; the
representative protein per species was the Ensembl canonical-transcript
translation. Proteins within ±30 residues of the 339-residue human reference were
retained (n = 187) and cross-referenced to UniProt. The *Bos taurus* sequence was
taken from UniProt Q24K14 (339 aa) rather than the 373-aa Ensembl translation
(ENSBTAP00000086519). Sequences were aligned with MUSCLE v.5.3 (build d9725ac;
`muscle -align`). Parsing and pairwise identity (global alignment, standard
substitution matrices) used Biopython v.1.87; tables and numerics used pandas
v.2.3.3 and NumPy v.1.24.4.

## Structure, secondary structure and figure

Per-species AlphaFold models were obtained from the AlphaFold Protein Structure
Database (model v6) by UniProt accession. Secondary structure was assigned from
backbone φ/ψ torsion angles (helix/strand/loop, by Ramachandran region) and
mapped to human positions via pairwise alignment of model to aligned sequence.
Conservation followed the Clustal `* : .` convention and columns were scored by
exact residue identity across the five species. Auxiliary panels used
Matplotlib v.3.10.8 (Agg). The workflow — retrieval, alignment, structural
annotation, figure and interactive HTML browser — runs from a single script; the
environment (Python v.3.10.20 and the packages above) is provided as
`environment.yml` and a pip requirements file. Orthology, alignment, structure
prediction and structural annotation used established tools (Ensembl, UniProt,
MUSCLE, the AlphaFold Protein Structure Database and Biopython); no new algorithm
was introduced.

## Species and identifiers (compared in the figure)

| Species | Ensembl gene | Ensembl protein | UniProt | AlphaFold model | Length (aa) |
|---|---|---|---|---|---|
| *Homo sapiens* (human) | ENSG00000100612 | ENSP00000216500 | Q9Y394 | AF-Q9Y394-F1 | 339 |
| *Mus musculus* (mouse) | ENSMUSG00000021094 | ENSMUSP00000021512 | Q9CXR1 | AF-Q9CXR1-F1 | 338 |
| *Rattus norvegicus* (rat) | ENSRNOG00000005589 | ENSRNOP00000007645 | D4A0T8 | AF-D4A0T8-F1 | 338 |
| *Bos taurus* (cattle) | ENSBTAG00000020729 | ENSBTAP00000086519 | Q24K14 | AF-Q24K14-F1 | 339 |
| *Danio rerio* (zebrafish) | ENSDARG00000003444 | ENSDARP00000004163 | Q0P3U1 | AF-Q0P3U1-F1 | 338 |

For *Bos taurus*, the protein sequence and AlphaFold model correspond to UniProt
Q24K14 (the Ensembl protein ENSBTAP00000086519 cross-references UniProt
A0AAA9SJP4, a 373-residue translation). AlphaFold model files: `AF-<accession>-F1-model_v6.pdb`.

## Figure display and comparative highlighting

Each row shows one species' reference-projected sequence with residues coloured
by amino acid; the human reference is the lowest row. Above each row the species'
AlphaFold secondary structure is drawn in grey (α-helices as coils, β-strands as
arrows and loops as lines). A conservation row beneath the alignment marks
positions identical (`*`), strongly similar (`:`) or weakly similar (`.`) across
the five species (Clustal convention). Positions of interest are highlighted by
two nested criteria applied to informative residues (gaps and undetermined
positions excluded): columns at which *Homo sapiens* and *Danio rerio* carry the
identical residue while both rodents (*Mus musculus* and *Rattus norvegicus*)
differ are outlined with a black frame; the subset of these at which *Bos taurus*
also carries that residue — identity across all three catalytically active
species with divergence in both rodents — is additionally marked with a red
asterisk.

## Software versions

Python v.3.10.20; requests v.2.33.1; Biopython v.1.87; pandas v.2.3.3; NumPy
v.1.24.4; Matplotlib v.3.10.8; MUSCLE v.5.3. Data sources: Ensembl release 116;
UniProt; AlphaFold Protein Structure Database model version v6. Insert the
corresponding citations (Ensembl, UniProt, MUSCLE, AlphaFold and the AlphaFold DB,
Biopython) in the reference list.

## Reproducibility

`python Reproduce_DHRS7_Figure.py` reproduces the figure, this methods file
and the interactive browser from Ensembl / UniProt / AlphaFold retrieval and
MUSCLE alignment. Figure: `plots/dhrs7_species_snapshot.svg`.
