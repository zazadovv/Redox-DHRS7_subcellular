# DHRS7 cross-species alignment -- Methods (2026-08-09 19:42 UTC)

## Ortholog retrieval
DHRS7 orthologs were enumerated from Ensembl through the REST homology endpoint
(`type=orthologues`) starting from the human gene, returning both one2one and
one2many relationships across sampled vertebrates (253 orthologs). One
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
aligned with MUSCLE v5 (`muscle.exe`, default settings; a single deterministic
alignment, no replicate or ensemble runs). The resulting alignment
(835 columns) was projected onto human reference
positions (339 columns) for numbering.
Percent identities to the human reference were computed from global
(Needleman-Wunsch) pairwise alignments with the BLOSUM62 matrix (gap opening
-10, gap extension -0.5).

## Structure and secondary structure
The AlphaFold model of each species was retrieved from the AlphaFold Protein
Structure Database. These models provide predicted coordinates and a per-residue
confidence estimate (pLDDT); they carry no secondary-structure annotation, so
secondary structure was assigned from the model coordinates.
Each residue was classified from its backbone phi/psi torsion angles using
the standard Ramachandran windows, with helices shorter than four residues
and strands shorter than three shown as loop. Runs are reported as the
torsions describe them: a residue falling outside a window breaks the run
and the break is retained rather than smoothed over.
The alternative assignment from hydrogen bonding (Kabsch-Sander) is computed in the same run
and can be displayed instead; both are stored with the alignment.
Assignments were taken from each species' own model and placed on the alignment
shown in the figure, so the structure track follows the residues that are drawn.

## Species compared in the figure
All retrieved orthologs are aligned and stored, and the residue-level comparison
shown in the figure uses: mus_musculus, rattus_norvegicus, bos_taurus, danio_rerio, homo_sapiens. These were re-aligned on their own with MUSCLE
(339 columns, all residues, natural gaps) so the
panel shows the gaps that exist between these sequences rather than gaps
inherited from the full alignment; human is the reference row. Each species'
secondary structure was placed on this alignment directly from its own model, so
the structure track follows the residues that are drawn. Secondary structure was
available for 187 records overall.

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
Python 3.10.20, biopython 1.87, pandas 2.3.3, numpy 1.24.4, matplotlib 3.10.8, requests 2.33.1, muscle 5.3.win64 [d9725ac]
The exact versions above were recorded from the environment that produced these
files and are repeated in `environment.txt` beside them.

## Availability
`python dhrs7_alignment.py` rebuilds the alignment, this file and the figure from
the source databases. Figure: `plots/dhrs7_species_snapshot.svg`.
