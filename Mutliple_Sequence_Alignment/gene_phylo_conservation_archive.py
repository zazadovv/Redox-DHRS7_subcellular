#!/usr/bin/env python3
"""
SQLite archive and offline HTML report export for the gene phylogeny & conservation pipeline outputs.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import sqlite3
import time
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio import Phylo, SeqIO
from matplotlib.patches import Polygon, Rectangle


DEFAULT_SQLITE_ARCHIVE_PATH = Path(__file__).resolve().with_name("phylo_runs.sqlite")
INTERACTIVE_REPORT_FILENAME = "interactive_report.html"

# ---------------------------------------------------------------------------#
# V11: gene-agnostic labelling. V9.9 hard-coded "PLA2G4A" into several figure
# titles and HTML strings. V11 must work for ANY input gene, so we keep a
# single process-level "active gene symbol" that figure/report builders read
# via v11_gene_label(). The pipeline sets it from args.gene_symbol; the
# archive's export_output_archive sets it from run_meta. Domain-coordinate
# annotations that are genuinely PLA2G4A-specific (the PL/C2/PLLLLTP/DELD
# ruler, the P47712 reference accession) are gated behind v11_is_pla2g4a().
# ---------------------------------------------------------------------------#
_V11_ACTIVE_GENE_SYMBOL: str = "the target gene"


def v11_set_active_gene(symbol: Optional[str]) -> None:
    """Set the process-level active gene symbol used for figure/report labels."""
    global _V11_ACTIVE_GENE_SYMBOL
    if symbol and str(symbol).strip():
        _V11_ACTIVE_GENE_SYMBOL = str(symbol).strip()


def v11_gene_label() -> str:
    """Return the active gene symbol for labelling (falls back to a neutral
    phrase if never set)."""
    return _V11_ACTIVE_GENE_SYMBOL


def v11_is_pla2g4a() -> bool:
    """True when the active gene is PLA2G4A — gates PLA2G4A-specific domain
    annotations (PL/C2/PLLLLTP/DELD ruler, P47712 reference) that are
    meaningless for other genes."""
    return str(_V11_ACTIVE_GENE_SYMBOL).strip().upper() == "PLA2G4A"
ALIGNMENT_BROWSER_FILENAME = "alignment_browser.html"
PAIRWISE_DIRNAME = "pairwise_human_reference_alignments"
ALPHAFOLD_METADATA_FILENAME = "human_reference_alphafold_metadata.json"
ALPHAFOLD_MODEL_FILENAME = "human_reference_alphafold_model.pdb"
ALPHAFOLD_OVERLAY_FILENAME = "human_reference_alphafold_overlay.pml"
ALPHAFOLD_VIEWER_JS_FILENAME = "3Dmol-min.js"
COMPARATIVE_ALPHAFOLD_SS_FILENAME = "comparative_alphafold_secondary_structure.json"
COMPARATIVE_ALPHAFOLD_MODEL_DIRNAME = "comparative_alphafold_models"
NODE_CONSERVATION_EXTREMES_FILENAME = "node_conservation_extremes.csv"
NODE_CONSERVATION_TREE_SVG_FILENAME = "node_conservation_extremes_tree.svg"
NODE_CONSERVATION_PAPER_TREE_SVG_FILENAME = "node_conservation_extremes_paper_tree.svg"
NODE_CONSERVATION_REJECTED_SEQUENCE_FILENAME = "node_conservation_rejected_sequences.fasta"
BLOSUM62 = substitution_matrices.load("BLOSUM62")
COMPARATIVE_ALPHAFOLD_ACCESSION_OVERRIDES: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("danio_rerio", "ENSDARP00000090686"): {
        "uniprot_accession": "P50392",
        "alphafold_entry_id": "AF-P50392-F1",
        "alphafold_source_label": "AlphaFold DB P50392 Danio rerio PLA2G4A",
        "model_filename": "AF-P50392-F1-model_v6.pdb",
        "sequence_length": 741,
    },
    ("danio_rerio", "danio_rerio__ENSDARP00000090686"): {
        "uniprot_accession": "P50392",
        "alphafold_entry_id": "AF-P50392-F1",
        "alphafold_source_label": "AlphaFold DB P50392 Danio rerio PLA2G4A",
        "model_filename": "AF-P50392-F1-model_v6.pdb",
        "sequence_length": 741,
    },
    ("danio_rerio", "ENSDARG00000024546"): {
        "uniprot_accession": "P50392",
        "alphafold_entry_id": "AF-P50392-F1",
        "alphafold_source_label": "AlphaFold DB P50392 Danio rerio PLA2G4A",
        "model_filename": "AF-P50392-F1-model_v6.pdb",
        "sequence_length": 741,
    },
}

# V12: DHRS7-specific real per-species structure substitutions. These re-point
# the comparative-AlphaFold lookup at genuine species (or, for bovine, the
# Q24K14 Bos taurus) models so the secondary-structure track stops falling back
# to the human reference projection. Keyed by protein-record id, bare Ensembl
# protein id, and gene id so the override matches however the header is parsed.
# See V12_README.md. The accessions were validated by pairwise identity vs the
# aligned species record (rattus D4A0T8 100%, danio Q0P3U1 99.4%, bos Q24K14
# 100% over its 344 aligned residues).
_V12_DHRS7_STRUCTURE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "bos_taurus": {
        "keys": ["bos_taurus__ENSBTAP00000086519", "ENSBTAP00000086519", "ENSBTAG00000020729"],
        "payload": {
            "uniprot_accession": "Q24K14",
            "alphafold_entry_id": "AF-Q24K14-F1",
            "alphafold_source_label": "AlphaFold DB Q24K14 Bos taurus DHRS7 (339 aa isoform)",
            "model_filename": "AF-Q24K14-F1-model_v6.pdb",
            "sequence_length": 339,
        },
    },
    "rattus_norvegicus": {
        # 338-aa DHRS7 ortholog ONLY; the two 324-aa records are a DHRS7-like
        # paralog (F7EQE1) with no AlphaFold model and stay projected.
        "keys": ["rattus_norvegicus__ENSRNOP00000007645", "ENSRNOP00000007645", "ENSRNOG00000005589"],
        "payload": {
            "uniprot_accession": "D4A0T8",
            "alphafold_entry_id": "AF-D4A0T8-F1",
            "alphafold_source_label": "AlphaFold DB D4A0T8 Rattus norvegicus DHRS7",
            "model_filename": "AF-D4A0T8-F1-model_v6.pdb",
            "sequence_length": 338,
        },
    },
    "danio_rerio": {
        "keys": ["danio_rerio__ENSDARP00000004163", "ENSDARP00000004163", "ENSDARG00000003444"],
        "payload": {
            "uniprot_accession": "Q0P3U1",
            "alphafold_entry_id": "AF-Q0P3U1-F1",
            "alphafold_source_label": "AlphaFold DB Q0P3U1 Danio rerio DHRS7",
            "model_filename": "AF-Q0P3U1-F1-model_v6.pdb",
            "sequence_length": 338,
        },
    },
}
for _v12_species, _v12_cfg in _V12_DHRS7_STRUCTURE_OVERRIDES.items():
    for _v12_key in _v12_cfg["keys"]:
        COMPARATIVE_ALPHAFOLD_ACCESSION_OVERRIDES[(_v12_species, _v12_key)] = dict(_v12_cfg["payload"])

GAP_CHARS = {"-", "."}
AA_COLORS: Dict[str, str] = {
    "A": "#C8C8C8", "C": "#FFF35A", "D": "#FF5E5B", "E": "#FF8B68",
    "F": "#00BFD3", "G": "#FFAD33", "H": "#B58CFF", "I": "#00A6E8",
    "K": "#FF4FC3", "L": "#00D2C7", "M": "#DCE600", "N": "#4BEA77",
    "P": "#FFB5A3", "Q": "#37E6C4", "R": "#C85CFF", "S": "#86FF62",
    "T": "#52DAFF", "V": "#40E0A0", "W": "#0087C8", "Y": "#67D13F",
    "-": "#FFFFFF", ".": "#FFFFFF", "X": "#F2F2F2", "?": "#F2F2F2",
}
NON_CONSENSUS_RESIDUES = GAP_CHARS | {"X", "?"}
PROPERTY_GROUPS: Dict[str, Dict[str, set[str]]] = {
    "hydrophobicity": {
        "hydrophobic": set("AVILMFWYC"),
        "non_hydrophobic": set("RNDQEKHSTPG"),
    },
    "charge": {
        "positive": set("KRH"),
        "negative": set("DE"),
        "neutral": set("AVILMFWYCNQSTPG"),
    },
    "polarity": {
        "polar": set("RNDQEKHSTYC"),
        "nonpolar": set("AVILMFWPG"),
    },
    "size": {
        "small": set("AGSCTPDN"),
        "medium": set("VILQEH"),
        "large": set("MFKRWY"),
    },
    "aromaticity": {
        "aromatic": set("FWYH"),
        "non_aromatic": set("AVILMCGPSTNQDEKR"),
    },
}
ALIGNMENT_BROWSER_DEFAULT_STATE: Dict[str, Any] = {
    "scope": "aligned_reference_projected",
    "species_search": "",
    "clade": "all",
    "taxonomy": "all",
    "sort": "tree_order",
    "direction": "asc",
    "group": "taxonomy_level",
    "view": "compressed",
    "compare": "exact",
    "min_similar_run": 6,
    "offset": "flag",
    "residue": "",
    "gaps": "all",
    "reference_match": "all",
    "reference_start": "",
    "reference_end": "",
    "alignment_start": "",
    "alignment_end": "",
}
ALIGNMENT_BROWSER_DEFAULT_EXPORT_PREFIX = "alignment_browser_default_taxa_exact"
ALPHAFOLD_STRUCTURE_COLORS = {
    "helix": "#f97316",
    "sheet": "#7c3aed",
    "loop": "#9ca3af",
    "selected": "#fde047",
    "conserved": "#16a34a",
    "divergent": "#dc2626",
    "manual": "#0ea5e9",
    "charge_positive": "#2563eb",
    "charge_negative": "#dc2626",
    "charge_neutral": "#f8fafc",
    "calcium": "#0891b2",
    "calcium_ligand": "#0f172a",
}
ALIGNMENT_BROWSER_REFERENCE_LANDMARKS = (
    ("Phospholipid binding", "PL (Phospholipid binding)", "#0f766e"),
    ("C2", "C2", "#f59e0b"),
    ("PLA2c", "PLA2c", "#7c3aed"),
)
ALIGNMENT_BROWSER_REFERENCE_MOTIF_SEARCHES: Dict[str, Sequence[Dict[str, Any]]] = {
    "PLA2G4A": (
        {
            "label": "Import motif (PLLLLTP)",
            "motifs": ("PLLLLTP",),
            "color": "#a16207",
            "description": "Potential import motif",
        },
        {
            "label": "Caspase site",
            "motifs": ("DELD", "DEVD"),
            "color": "#b91c1c",
            "description": "Candidate caspase cleavage motif",
        },
    ),
}

ALIGNMENT_BROWSER_REFERENCE_POSITION_LANDMARKS: Dict[str, Sequence[Dict[str, Any]]] = {
    "PLA2G4A": (
        {
            "label": "S228 nucleophile",
            "row_label": "Catalytic residues",
            "position": 228,
            "color": "#ef4444",
            "description": "Catalytic nucleophile",
        },
        {
            "label": "D549 proton acceptor",
            "row_label": "Catalytic residues",
            "position": 549,
            "color": "#2563eb",
            "description": "Catalytic proton acceptor",
        },
    ),
}

PLA2G4A_CALCIUM_BINDING_ANNOTATION: Dict[str, Any] = {
    "available": True,
    "label": "Ca2+ binding sites",
    "note": (
        "Human cPLA2-alpha C2 domain calcium binding is mediated by three calcium-binding regions "
        "(CBR1-CBR3). Lipid-free cPLA2-alpha C2-domain structures report two canonical Ca2+ ions; "
        "the later DHPC-bound Patel/Brown/Chalfant structure reports a third Ca2+ (CaPC) that bridges "
        "the C2 domain to the phosphatidylcholine phosphate."
    ),
    "sources": [
        {
            "label": "Hirano et al. eLife 2019;8:e44760 / PDB 6IEJ",
            "url": "https://elifesciences.org/articles/44760",
            "note": "Dinshaw Patel coauthor study of cPLA2-alpha C2 domain bound to DHPC and Ca2+; reports third CaPC bridge and PC recognition by N65/Y96.",
        },
        {
            "label": "Xu et al. J Mol Biol 1998;280:485-500",
            "url": "https://pubmed.ncbi.nlm.nih.gov/9665851/",
            "note": "Human cPLA2 C2-domain solution study reporting two Ca2+ binding events.",
        },
        {
            "label": "Bittova et al. J Biol Chem 1999;274:9665-9672",
            "url": "https://www.sciencedirect.com/science/article/pii/S0021925819873027",
            "note": "Mutational study of essential Ca2+ ligands D40, D43, N65, D93, and N95.",
        },
        {
            "label": "NCBI CDD C2_cPLA2",
            "url": "https://www.ncbi.nlm.nih.gov/Structure/cdd/cd04036",
            "note": "C2_cPLA2 domains contain Ca2+-binding regions with acidic Ca2+ ligands.",
        },
    ],
    "loops": [
        {
            "label": "CBR1",
            "start": 31,
            "end": 43,
            "color": "#06b6d4",
            "description": "Ca2+/membrane loop I; includes D40, T41 backbone carbonyl, and D43 ligands.",
        },
        {
            "label": "CBR2",
            "start": 61,
            "end": 67,
            "color": "#0891b2",
            "description": "Ca2+/membrane loop II; includes N65 ligand.",
        },
        {
            "label": "CBR3",
            "start": 93,
            "end": 101,
            "color": "#0e7490",
            "description": "Ca2+/membrane loop III; includes D93, A94 backbone carbonyl, N95, and the site-III-disrupting N95 position.",
        },
    ],
    "ligands": [
        {"position": 40, "label": "D40", "sites": ["site I", "site II"]},
        {"position": 41, "label": "T41 carbonyl", "sites": ["site I"]},
        {"position": 43, "label": "D43", "sites": ["site I", "site II"]},
        {"position": 62, "label": "H62 PC contact", "sites": ["DHPC headgroup contact"], "color": "#f59e0b"},
        {"position": 64, "label": "N64 PC contact", "sites": ["DHPC headgroup contact"], "color": "#f59e0b"},
        {"position": 65, "label": "N65", "sites": ["site I", "CaPC/DHPC bridge", "PC selectivity"]},
        {"position": 93, "label": "D93", "sites": ["site II"]},
        {"position": 94, "label": "A94 carbonyl", "sites": ["site II", "DHPC headgroup contact"]},
        {"position": 95, "label": "N95", "sites": ["site II", "site III analog attenuated"]},
        {"position": 96, "label": "Y96 PC cation-pi", "sites": ["PC trimethylammonium recognition"], "color": "#f59e0b"},
    ],
    "sites": [
        {
            "label": "Ca2+ site I",
            "start": 40,
            "end": 65,
            "residues": [40, 41, 43, 65],
            "description": "Reported ligands: D40, T41 backbone carbonyl, D43, N65, plus waters.",
        },
        {
            "label": "Ca2+ site II",
            "start": 40,
            "end": 95,
            "residues": [40, 43, 93, 94, 95],
            "description": "Reported ligands: D40, D43, D93 bidentate, A94 backbone carbonyl, N95, plus water.",
        },
        {
            "label": "Ca2+ site III analog",
            "start": 93,
            "end": 99,
            "residues": [95],
            "description": "The analogous third site is not occupied in cPLA2; N95 replaces the Asp found in other C2 domains.",
        },
        {
            "label": "CaPC/DHPC bridge",
            "start": 62,
            "end": 96,
            "residues": [62, 64, 65, 94, 96],
            "description": "DHPC-bound 6IEJ contains a third Ca2+ that bridges DHPC phosphate/sn-1 carbonyl contacts and interacts with N65; Y96 recognizes the PC trimethylammonium group.",
        },
    ],
}

ALIGNMENT_BROWSER_SCOPES: Dict[str, Dict[str, str]] = {
    "aligned_reference_projected": {
        "filename": "aligned_reference_projected.fasta",
        "label": "Reference-projected",
    },
    "aligned_full": {
        "filename": "aligned.fasta",
        "label": "Full alignment",
    },
}

SITE_CLADE_MYA_DEFAULTS: Dict[str, float] = {
    "tetrapods": 420.0,
    "dipnoi": 430.0,
    "actinistia": 450.0,
    "holostei": 320.0,
    "teleosts": 320.0,
    "other_fish": 430.0,
    "other_vertebrates": float("nan"),
}

DIPNOI_SPECIES = {
    "protopterus_annectens", "protopterus_aethiopicus", "lepidosiren_paradoxa",
    "neoceratodus_forsteri", "protopterus_dolloi",
}
ACTINISTIA_SPECIES = {"latimeria_chalumnae", "latimeria_menadoensis"}
HOLOSTEI_SPECIES = {
    "lepisosteus_oculatus", "amia_calva", "lepisosteus_platostomus", "atractosteus_spatula",
}
TETRAPOD_TAXON_KEYWORDS = (
    "mammalia", "eutheria", "theria", "metatheria", "monotremata", "primates", "rodentia",
    "lagomorpha", "euarchontoglires", "laurasiatheria", "afrotheria", "xenarthra", "glires",
    "boreoeutheria", "catarrhini", "platyrrhini", "simiiformes", "hominoidea", "hominidae",
    "homininae", "aves", "bird", "reptilia", "sauropsida", "amniota", "amphibia", "anura",
    "xenopus", "leptobrachium", "gallus", "anas", "taeniopygia", "anolis", "pelodiscus",
    "crocodylus", "sphenodon", "chelonia", "testudines", "serpentes", "homo_sapiens",
    "pan_troglodytes", "mus_musculus",
)
OTHER_FISH_TAXON_KEYWORDS = (
    "chondrichthyes", "gnathostomata", "vertebrata", "cyclostomata", "petromyzontiformes",
    "myxini", "hagfish", "lamprey", "callorhinchus", "eptatretus",
)
CHORDATE_FALLBACK_KEYWORDS = ("chordata", "urochordata", "cephalochordata", "tunicata", "ciona")
CHORDATE_TREE_TAXON_KEYWORDS = (
    CHORDATE_FALLBACK_KEYWORDS
    + TETRAPOD_TAXON_KEYWORDS
    + OTHER_FISH_TAXON_KEYWORDS
    + tuple(sorted(DIPNOI_SPECIES))
    + tuple(sorted(ACTINISTIA_SPECIES))
    + tuple(sorted(HOLOSTEI_SPECIES))
    + (
        "euteleostomi", "actinopterygii", "tetrapoda", "sarcopterygii", "teleostei",
        "teleost", "holostei", "dipnoi", "actinistia",
    )
)
CLADE_ORDER = ["tetrapods", "dipnoi", "actinistia", "holostei", "teleosts", "other_fish", "other_vertebrates"]

CSV_TABLE_SPECS: Dict[str, Tuple[str, str]] = {
    "orthologs": ("orthologs.tsv", "\t"),
    "sequence_retrieval": ("sequence_retrieval.tsv", "\t"),
    "protein_metadata": ("protein_metadata.tsv", "\t"),
    "protein_features": ("protein_features.tsv", "\t"),
    "protein_xrefs": ("protein_xrefs.tsv", "\t"),
    "length_filter": ("length_filter_report.csv", ","),
    "conservation_scan": ("conservation_scan.csv", ","),
    "conservation_per_position": ("conservation_per_position.csv", ","),
    "annotated_functional_sites": ("annotated_functional_sites.csv", ","),
    "annotated_site_clade_comparison": ("annotated_site_clade_comparison.csv", ","),
    "domains": ("domains.tsv", "\t"),
    "selected_consensus_chunks": ("selected_consensus_chunks.tsv", "\t"),
    "selected_consensus_chunks_structure_map": ("selected_consensus_chunks_structure_map.tsv", "\t"),
    "conserved_regions": ("conserved_regions.csv", ","),
    "domain_clade_conservation_summary": ("domain_clade_conservation_summary.csv", ","),
    "evolutionary_segments": ("evolutionary_segments.csv", ","),
    "evolutionary_segment_metrics": ("evolutionary_segment_metrics_by_clade.csv", ","),
    "evolutionary_alignment_windows_manifest": ("evolutionary_alignment_windows_manifest.csv", ","),
    "clade_identity_profiles_wide": ("clade_identity_profiles.csv", ","),
    "clade_fourier_lowpass_profiles_wide": ("clade_fourier_lowpass_profiles.csv", ","),
    "clade_fourier_spectrum": ("clade_fourier_spectrum.csv", ","),
    "clade_difference_from_global_wide": ("clade_difference_from_global.csv", ","),
    "clade_fourier_regions": ("clade_fourier_conserved_and_divergent_regions.csv", ","),
    "node_conservation_extremes": (NODE_CONSERVATION_EXTREMES_FILENAME, ","),
}

FASTA_SPECS: Dict[str, str] = {
    "proteins.fasta": "proteins_raw",
    "aligned.fasta": "aligned_full",
    "aligned_reference_projected.fasta": "aligned_reference_projected",
}

TABLE_DISPLAY_SPECS: Dict[str, Dict[str, Any]] = {
    "v11_representatives": {
        "columns": [
            "clade", "species", "protein_label", "taxonomy_level",
            "mean_identity", "overlapping_positions", "identical_positions",
            "is_reference", "is_mandatory", "selection_rank", "record_id",
        ],
    },
    "v11_motifs_master": {
        "columns": [
            "motif_id", "label", "source", "start", "end",
            "motif_name", "matched_seq", "description", "citation", "regex",
        ],
    },
    "v11_motif_evolution_per_clade": {
        "columns": [
            "motif_id", "label", "clade", "n_species",
            "reference_motif", "consensus_motif",
            "fraction_matching_reference", "dominant_alternative",
            "consensus_count", "start", "end",
        ],
    },
    "v11_lineage_stabilization": {
        "columns": [
            "reference_ungapped_position", "reference_residue",
            "ancestral_entropy", "ancestral_n",
            "derived_entropy", "derived_n",
            "stabilization_score",
            "ancestral_clades", "derived_clades",
        ],
    },
    "orthologs": {
        "columns": [
            "species", "symbol", "ensembl_gene_id", "ensembl_protein_id",
            "perc_id_to_query", "orthology_type", "taxonomy_level", "is_query",
        ],
    },
    "sequence_retrieval": {
        "columns": [
            "species", "symbol", "protein_record_id", "translation_id", "length_aa", "status",
            "preferred_public_label", "uniprot_accession", "reviewed_status", "alphafold_entry_id",
            "metadata_pub_gene_id", "metadata_description",
        ],
    },
    "protein_metadata": {
        "columns": [
            "species", "protein_record_id", "symbol", "preferred_public_label",
            "preferred_protein_name", "taxonomy_level", "clade", "phylum", "broad_clade",
            "uniprot_accession", "reviewed_status", "alphafold_entry_id", "alphafold_source_label",
        ],
    },
    "protein_features": {
        "columns": [
            "protein_record_id", "species", "symbol", "uniprot_accession", "source_database",
            "feature_type", "description", "start", "end", "source_feature_id",
        ],
        "rangeStartField": "start",
        "rangeEndField": "end",
    },
    "protein_xrefs": {
        "columns": [
            "protein_record_id", "species", "symbol", "database", "external_id", "category", "label",
        ],
    },
    "length_filter": {
        "columns": [
            "species", "symbol", "length_aa", "reference_length_for_filter",
            "length_delta_from_reference", "max_allowed_abs_delta", "length_filter_status",
        ],
    },
    "conservation_scan": {
        "columns": [
            "reference_ungapped_position", "reference_residue", "identity_max",
            "reference_identity_fraction", "hydrophobicity_conservation",
            "charge_conservation", "polarity_conservation", "size_conservation",
            "aromaticity_conservation", "occupancy", "reference_domain_labels",
        ],
        "positionField": "reference_ungapped_position",
    },
    "conserved_regions": {
        "columns": [
            "score_column", "start_reference_position", "end_reference_position",
            "length_alignment", "mean_score", "mean_occupancy",
        ],
        "rangeStartField": "start_reference_position",
        "rangeEndField": "end_reference_position",
    },
    "annotated_functional_sites": {
        "columns": [
            "position", "label", "reference_residue", "identity_max",
            "reference_identity_fraction", "occupancy", "reference_domain_labels",
        ],
        "positionField": "position",
    },
    "annotated_site_clade_comparison": {
        "columns": [
            "site_position", "site_label", "clade", "mya_from_human_lineage",
            "reference_residue", "major_residue", "major_residue_fraction",
            "reference_residue_fraction", "classification", "consensus_motif",
            "reference_motif_fraction",
        ],
        "positionField": "site_position",
    },
    "domains": {
        "columns": [
            "species", "symbol", "protein_record_id", "uniprot_accession", "feature_type",
            "description", "start", "end", "interpro_ids",
        ],
        "rangeStartField": "start",
        "rangeEndField": "end",
    },
    "selected_consensus_chunks": {
        "columns": [
            "chunk_id", "label", "start_reference_position", "end_reference_position", "score", "color_hex", "notes",
        ],
        "rangeStartField": "start_reference_position",
        "rangeEndField": "end_reference_position",
    },
    "selected_consensus_chunks_structure_map": {
        "columns": [
            "chunk_id", "label", "start_reference_position", "end_reference_position",
            "start_structure_residue", "end_structure_residue", "score", "color_hex", "mapping_status", "notes",
        ],
        "rangeStartField": "start_reference_position",
        "rangeEndField": "end_reference_position",
    },
    "clade_identity_profiles": {
        "columns": [
            "reference_position", "reference_residue", "global_identity_to_human",
            "tetrapods_identity_to_human", "actinistia_identity_to_human",
            "holostei_identity_to_human", "teleosts_identity_to_human",
            "other_vertebrates_identity_to_human",
        ],
        "positionField": "reference_position",
    },
    "clade_difference_from_global": {
        "columns": [
            "reference_position", "reference_residue",
            "tetrapods_minus_global_identity", "actinistia_minus_global_identity",
            "holostei_minus_global_identity", "teleosts_minus_global_identity",
            "other_vertebrates_minus_global_identity",
        ],
        "positionField": "reference_position",
    },
    "clade_fourier_regions": {
        "columns": [
            "clade", "region_type", "start_reference_position",
            "end_reference_position", "length", "mean_identity_to_human",
            "mean_global_identity_to_human", "mean_delta_vs_global",
        ],
        "rangeStartField": "start_reference_position",
        "rangeEndField": "end_reference_position",
    },
    "domain_clade_conservation_summary": {
        "columns": [
            "domain_label", "feature_type", "start", "end", "length", "clade",
            "mean_identity_to_human", "min_identity_to_human",
            "fraction_positions_identity_ge_0_85",
        ],
        "rangeStartField": "start",
        "rangeEndField": "end",
    },
    "node_conservation_extremes": {
        "columns": [
            "node_label", "parent_label", "record_count", "species_count",
            "identity_range_percent", "min_identity_percent", "max_identity_percent",
            "least_conserved_label", "least_conserved_protein_id", "least_conserved_length_aa",
            "least_conserved_input_status", "least_conserved_metric_source",
            "most_conserved_label", "most_conserved_protein_id", "most_conserved_length_aa",
            "most_conserved_input_status", "most_conserved_metric_source",
            "membership_rule",
        ],
    },
    "evolutionary_segments": {
        "columns": [
            "segment_label", "segment_source", "feature_type", "region_type", "clade",
            "start", "end", "length",
            "first_divergent_clade_identity", "first_divergent_species_identity",
            "first_divergent_clade_similarity", "first_divergent_species_similarity",
            "first_divergent_clade_polarity", "first_divergent_species_polarity",
            "tree_order_source",
        ],
        "rangeStartField": "start",
        "rangeEndField": "end",
    },
    "evolutionary_segment_metrics": {
        "columns": [
            "segment_label", "segment_source", "clade", "start", "end", "length",
            "species_count", "informative_species_count",
            "identity_to_human_mean", "similarity_to_human_mean", "polarity_agreement_to_human_mean",
            "identity_is_conserved", "identity_is_diverged",
            "similarity_is_conserved", "similarity_is_diverged",
            "polarity_is_conserved", "polarity_is_diverged",
        ],
        "rangeStartField": "start",
        "rangeEndField": "end",
    },
    "evolutionary_alignment_windows_manifest": {
        "columns": [
            "segment_label", "segment_source", "window_export_group", "clade",
            "start", "end", "length", "text_report_path",
            "first_divergent_species_identity", "first_divergent_species_similarity",
            "first_divergent_species_polarity", "tree_order_source",
        ],
        "rangeStartField": "start",
        "rangeEndField": "end",
        "linkColumns": ["text_report_path"],
    },
    "pairwise_reports": {
        "columns": [
            "species", "symbol", "aligned_length", "shared_non_gap_sites",
            "exact_match_fraction", "exact_match_count", "mismatch_count",
            "target_gap_count", "target_gap_fraction",
            "text_report_path", "pdf_report_path", "png_page_paths",
        ],
        "linkColumns": ["text_report_path", "pdf_report_path", "png_page_paths"],
    },
    "downloads": {
        "columns": ["artifact_group", "file_type", "relative_path", "size_bytes"],
        "linkColumns": ["relative_path"],
    },
}


def sanitize_filename(text: str) -> str:
    keep: List[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        elif ch in (" ", "|", "/", "\\", ":"):
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "output"


def parse_header_field(record_id: str, field_name: str) -> Optional[str]:
    prefix = f"{field_name}="
    for part in record_id.split("|"):
        if part.startswith(prefix):
            return part.split("=", 1)[1]
    return None


def parse_header_species_symbol(record_id: str) -> Tuple[str, str]:
    species = record_id.split("|")[0].strip()
    symbol = parse_header_field(record_id, "Gene") or "unknown"
    return species, symbol


def normalize_run_key(outdir: Path) -> str:
    resolved = outdir.resolve()
    return os.path.normcase(os.path.normpath(str(resolved)))


def load_output_table(outdir: Path, filename: str, sep: str) -> pd.DataFrame:
    path = outdir / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep=sep)
    except pd.errors.EmptyDataError:
        # V11: tolerate header-less / empty CSVs (e.g. when no annotated sites match)
        # so the archive step doesn't abort the whole interactive-report build.
        return pd.DataFrame()


def parse_run_summary_metadata(summary_path: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if not summary_path.exists():
        return meta

    for raw_line in summary_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()

    seq_text = meta.get("Sequences recovered")
    if seq_text and "/" in seq_text:
        recovered_text, total_text = [chunk.strip() for chunk in seq_text.split("/", 1)]
        try:
            meta["recovered_sequence_count"] = int(recovered_text)
            meta["candidate_sequence_count"] = int(total_text)
        except ValueError:
            pass

    for key in ("Full alignment length", "Reference-projected alignment length"):
        if key in meta:
            try:
                meta[key] = int(str(meta[key]).strip())
            except ValueError:
                pass

    if "Phylogeny built" in meta:
        value = str(meta["Phylogeny built"]).strip().lower()
        meta["Phylogeny built"] = value in {"yes", "true", "1"}

    window_text = meta.get("Annotated-site comparison window")
    if window_text:
        digits = "".join(ch for ch in str(window_text) if ch.isdigit())
        if digits:
            meta["Annotated-site comparison window residues"] = int(digits)

    return meta


def classify_artifact(relative_path: str) -> str:
    path = Path(relative_path)
    suffix = path.suffix.lower()
    posix_path = relative_path.replace("\\", "/")

    if posix_path.startswith(f"{PAIRWISE_DIRNAME}/"):
        if suffix == ".txt":
            return "pairwise_text"
        if suffix == ".pdf":
            return "pairwise_pdf"
        if suffix == ".png":
            return "pairwise_png"
        return "pairwise_other"
    if suffix in {".csv", ".tsv"}:
        return "table"
    if suffix == ".txt":
        return "text"
    if suffix in {".svg"}:
        return "vector"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "image"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix in {".fasta", ".fa", ".faa", ".phy"}:
        return "sequence"
    if suffix in {".treefile", ".contree", ".nex", ".bionj", ".mldist", ".iqtree", ".log", ".gz"}:
        return "tree"
    if suffix == ".pml":
        return "pymol"
    return "other"


def collect_artifacts(outdir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(p for p in outdir.rglob("*") if p.is_file()):
        if path.name == INTERACTIVE_REPORT_FILENAME:
            continue
        rel = path.relative_to(outdir).as_posix()
        rows.append({
            "relative_path": rel,
            "file_name": path.name,
            "file_type": path.suffix.lower().lstrip("."),
            "artifact_group": classify_artifact(rel),
            "size_bytes": int(path.stat().st_size),
            "is_pairwise": rel.startswith(f"{PAIRWISE_DIRNAME}/"),
        })
    return pd.DataFrame(rows)


def collect_msa_sequences(outdir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for filename, alignment_scope in FASTA_SPECS.items():
        fasta_path = outdir / filename
        if not fasta_path.exists():
            continue
        for record_index, record in enumerate(SeqIO.parse(str(fasta_path), "fasta"), start=1):
            sequence = str(record.seq)
            species, symbol = parse_header_species_symbol(record.id)
            rows.append({
                "alignment_scope": alignment_scope,
                "source_fasta": filename,
                "record_index": record_index,
                "record_id": record.id,
                "species": species,
                "symbol": symbol,
                "description": record.description,
                "sequence": sequence,
                "aligned_length": len(sequence),
                "ungapped_length": len(sequence.replace("-", "").replace(".", "")),
                "gap_count": sequence.count("-") + sequence.count("."),
            })
    return pd.DataFrame(rows)


def infer_reference_species(tables: Dict[str, pd.DataFrame], msa_sequences: pd.DataFrame) -> Optional[str]:
    conservation_df = tables.get("conservation_per_position", pd.DataFrame())
    if not conservation_df.empty and "reference_species" in conservation_df.columns:
        value = conservation_df["reference_species"].dropna()
        if not value.empty:
            return str(value.iloc[0])

    projected = msa_sequences[msa_sequences["alignment_scope"] == "aligned_reference_projected"] if not msa_sequences.empty else pd.DataFrame()
    if not projected.empty and "species" in projected.columns:
        return str(projected.iloc[0]["species"])
    return None


def finite_age_mya(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def default_clade_age_mya(clade: Optional[str]) -> Optional[float]:
    return finite_age_mya(SITE_CLADE_MYA_DEFAULTS.get(str(clade or "").strip().lower()))


def classify_alignment_clade(species: str, taxonomy_level: Optional[str] = None) -> str:
    species_text = (species or "").strip().lower()
    taxonomy_text = (taxonomy_level or "").strip().lower()
    joined = f"{species_text} {taxonomy_text}"
    if species_text in DIPNOI_SPECIES or any(k in joined for k in ("dipnoi", "lungfish", "neoceratodus", "protopterus", "lepidosiren")):
        return "dipnoi"
    if species_text in ACTINISTIA_SPECIES or any(k in joined for k in ("actinistia", "coelacanth", "latimeria")):
        return "actinistia"
    if species_text in HOLOSTEI_SPECIES or any(k in joined for k in ("holostei", "lepisosteus", "amia_calva", "gar", "bowfin")):
        return "holostei"
    if any(k in joined for k in (
        "actinopterygii", "teleost", "clupeocephala", "cypriniformes", "danio", "oryzias", "gasterosteus",
        "takifugu", "tetraodon", "xiphophorus", "salmo", "oncorhynchus", "poecilia", "oreochromis",
        "astyanax", "ictalurus", "gadus", "fundulus", "amphilophus", "seriola", "betta", "carassius",
        "cyprinus", "labrus", "lates", "larimichthys", "hippocampus", "sander", "nothobranchius",
    )):
        return "teleosts"
    if any(k in joined for k in TETRAPOD_TAXON_KEYWORDS):
        return "tetrapods"
    if any(k in joined for k in OTHER_FISH_TAXON_KEYWORDS):
        return "other_fish"
    if any(k in joined for k in ("sarcopterygii",) + CHORDATE_FALLBACK_KEYWORDS):
        return "other_vertebrates"
    return "other_vertebrates"


TREE_PHYLA_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Chordata", CHORDATE_TREE_TAXON_KEYWORDS),
    ("Arthropoda", ("arthropoda", "insecta", "arachnida", "crustacea", "hexapoda", "drosophila", "bombyx")),
    ("Nematoda", ("nematoda", "caenorhabditis")),
    ("Mollusca", ("mollusca", "gastropoda", "cephalopoda", "bivalvia", "octopus", "lottia")),
    ("Annelida", ("annelida",)),
    ("Echinodermata", ("echinodermata", "echinoidea", "asteroidea", "strongylocentrotus")),
    ("Cnidaria", ("cnidaria", "hydrozoa", "anthozoa", "nematostella", "hydra")),
    ("Platyhelminthes", ("platyhelminthes", "schistosoma")),
    ("Porifera", ("porifera", "amphimedon")),
    ("Fungi", ("fungi", "ascomycota", "basidiomycota", "dikarya", "saccharomycetes", "saccharomyces")),
)


def classify_alignment_phylum(species: str, taxonomy_level: Optional[str] = None) -> str:
    species_text = (species or "").strip().lower()
    taxonomy_text = (taxonomy_level or "").strip().lower()
    joined = f"{species_text} {taxonomy_text}"
    for label, keywords in TREE_PHYLA_KEYWORDS:
        if any(keyword in joined for keyword in keywords):
            return label
    fallback = (taxonomy_level or "").strip()
    return fallback or "Unassigned"


def classify_alignment_broad_clade(clade: Optional[str],
                                   phylum: Optional[str],
                                   taxonomy_level: Optional[str] = None) -> str:
    clade_text = (clade or "").strip()
    if clade_text:
        return clade_text
    phylum_text = (phylum or "").strip()
    if phylum_text:
        return phylum_text
    fallback = (taxonomy_level or "").strip()
    return fallback or "Unassigned"


def derive_alignment_age_mya(species: str,
                             taxonomy_level: Optional[str],
                             clade: Optional[str],
                             phylum: Optional[str],
                             phylum_anchor_ages: Optional[Dict[str, float]] = None) -> Tuple[Optional[float], Optional[str]]:
    direct_age = default_clade_age_mya(clade)
    if direct_age is not None:
        return direct_age, "clade_default"

    joined = f"{(species or '').strip().lower()} {(taxonomy_level or '').strip().lower()}"
    if any(keyword in joined for keyword in TETRAPOD_TAXON_KEYWORDS):
        return default_clade_age_mya("tetrapods"), "taxonomy_inferred"
    if any(keyword in joined for keyword in OTHER_FISH_TAXON_KEYWORDS):
        return default_clade_age_mya("other_fish"), "taxonomy_inferred"

    phylum_text = str(phylum or "").strip()
    if phylum_text and phylum_anchor_ages and phylum_text in phylum_anchor_ages:
        return phylum_anchor_ages[phylum_text], "phylum_anchor"
    return None, None


def _tree_viewer_mya_legend(alignment_species_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if alignment_species_df is None or alignment_species_df.empty:
        return []
    present_clades = {
        str(clean_json_value(row.get("clade")))
        for _, row in alignment_species_df.iterrows()
        if clean_json_value(row.get("clade"))
    }
    ordered: List[Dict[str, Any]] = []
    for clade_name in CLADE_ORDER:
        age_value = default_clade_age_mya(clade_name)
        if clade_name in present_clades and age_value is not None:
            ordered.append({"clade": clade_name, "mya": age_value})
    return ordered


def load_tree_nomenclature_payload(outdir: Path) -> Dict[str, Any]:
    path = outdir / "tree_nomenclature.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def tree_species_signature(species_values: Sequence[str]) -> str:
    members = sorted({str(species).strip() for species in species_values if str(species).strip()})
    return "|".join(members)


def normalize_tree_record_key(record_id: str) -> str:
    text = str(record_id or "").strip().strip("'\"")
    for field_name in ("Gene", "EnsemblGene", "Protein"):
        text = text.replace(f"|{field_name}_", f"|{field_name}=")
    return text


def build_taxonomy_lookup(ortholog_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    if ortholog_df is None or ortholog_df.empty or "species" not in ortholog_df.columns:
        return lookup
    for _, row in ortholog_df.iterrows():
        species = clean_json_value(row.get("species"))
        if not species:
            continue
        species_text = str(species)
        if species_text in lookup:
            continue
        lookup[species_text] = {
            "taxonomy_level": clean_json_value(row.get("taxonomy_level")),
            "orthology_type": clean_json_value(row.get("orthology_type")),
            "is_query": bool(row.get("is_query")) if pd.notna(row.get("is_query")) else False,
            "ensembl_gene_id": clean_json_value(row.get("ensembl_gene_id")),
            "ensembl_protein_id": clean_json_value(row.get("ensembl_protein_id")),
            "perc_id_to_query": clean_json_value(row.get("perc_id_to_query")),
        }
    return lookup


def parse_tree_leaf_order(outdir: Path) -> Tuple[Dict[str, int], Dict[str, int], str]:
    tree_path = choose_existing_artifact(outdir, ["phylo.treefile", "phylo.contree", "phylo.bionj"])
    if not tree_path:
        return {}, {}, "record_order"
    try:
        tree = Phylo.read(str(outdir / tree_path), "newick")
    except Exception:
        return {}, {}, "record_order"

    record_order: Dict[str, int] = {}
    species_order: Dict[str, int] = {}
    for order, terminal in enumerate(tree.get_terminals(), start=1):
        raw_name = terminal.name or ""
        normalized_name = normalize_tree_record_key(raw_name)
        species, _ = parse_header_species_symbol(normalized_name)
        for key in {raw_name, normalized_name, normalize_tree_record_key(normalized_name)}:
            if key:
                record_order.setdefault(key, order)
        if species:
            species_order.setdefault(species, order)
    return record_order, species_order, tree_path


def _tree_viewer_metadata_maps(alignment_species_df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    by_record: Dict[str, Dict[str, Any]] = {}
    by_species: Dict[str, Dict[str, Any]] = {}
    if alignment_species_df is None or alignment_species_df.empty:
        return by_record, by_species

    for _, row in alignment_species_df.iterrows():
        record_id = clean_json_value(row.get("record_id"))
        species = clean_json_value(row.get("species"))
        if not record_id and not species:
            continue
        taxonomy_level = clean_json_value(row.get("taxonomy_level"))
        clade = clean_json_value(row.get("clade"))
        phylum = clean_json_value(row.get("phylum") or classify_alignment_phylum(str(species or ""), str(taxonomy_level or "")))
        broad_clade = clean_json_value(
            row.get("broad_clade") or classify_alignment_broad_clade(str(clade or ""), str(phylum or ""), str(taxonomy_level or ""))
        )
        metadata = {
            "record_id": record_id,
            "protein_record_id": clean_json_value(row.get("protein_record_id")),
            "species": species,
            "symbol": clean_json_value(row.get("symbol")),
            "clade": clade,
            "taxonomy_level": taxonomy_level,
            "phylum": phylum,
            "broad_clade": broad_clade,
            "clade_age_mya": clean_json_value(row.get("clade_age_mya")),
            "clade_age_source": clean_json_value(row.get("clade_age_source")),
            "species_display_label": clean_json_value(row.get("species_display_label")),
            "scientific_name": clean_json_value(row.get("scientific_name")),
            "common_name": clean_json_value(row.get("common_name")),
            "preferred_public_label": clean_json_value(row.get("preferred_public_label")),
            "preferred_public_gene_label": clean_json_value(row.get("preferred_public_gene_label")),
            "preferred_protein_name": clean_json_value(row.get("preferred_protein_name")),
            "nomenclature_leaf_label": clean_json_value(row.get("nomenclature_leaf_label")),
            "uniprot_accession": clean_json_value(row.get("uniprot_accession")),
            "reviewed_status": clean_json_value(row.get("reviewed_status")),
            "alphafold_entry_id": clean_json_value(row.get("alphafold_entry_id")),
        }
        if record_id:
            record_text = str(record_id)
            for key in {record_text, normalize_tree_record_key(record_text)}:
                if key:
                    by_record.setdefault(key, metadata)
        if species:
            by_species.setdefault(str(species), metadata)
    return by_record, by_species


def _tree_viewer_node_to_json(clade: Any,
                              metadata_by_record: Dict[str, Dict[str, Any]],
                              metadata_by_species: Dict[str, Dict[str, Any]],
                              tree_nomenclature: Optional[Dict[str, Any]] = None,
                              node_id: str = "0") -> Dict[str, Any]:
    tree_nomenclature = tree_nomenclature or {}
    group_lookup = {
        str(group.get("signature") or ""): group
        for group in tree_nomenclature.get("groups", []) or []
        if str(group.get("signature") or "").strip()
    }
    leaf_by_species = tree_nomenclature.get("leaf_by_species") or {}
    leaf_by_record = tree_nomenclature.get("leaf_by_protein_record_id") or {}

    def walk(current_clade: Any, current_node_id: str) -> Dict[str, Any]:
        raw_name = str(current_clade.name or "")
        normalized_name = normalize_tree_record_key(raw_name)
        species, symbol = parse_header_species_symbol(normalized_name)
        species_value = clean_json_value(species)
        metadata = (
            metadata_by_record.get(raw_name)
            or metadata_by_record.get(normalized_name)
            or metadata_by_species.get(species)
            or {}
        )
        protein_record_id = clean_json_value(
            metadata.get("protein_record_id") or parse_header_field(normalized_name, "ProteinRecordID")
        )
        children = [
            walk(child, f"{current_node_id}.{idx}")
            for idx, child in enumerate(getattr(current_clade, "clades", []) or [])
        ]
        branch_length = getattr(current_clade, "branch_length", None)
        if branch_length is not None:
            try:
                branch_length = float(branch_length)
            except (TypeError, ValueError):
                branch_length = None
        taxonomy_level = clean_json_value(metadata.get("taxonomy_level"))
        clade_name = clean_json_value(metadata.get("clade"))
        phylum = clean_json_value(metadata.get("phylum") or classify_alignment_phylum(str(species_value or ""), str(taxonomy_level or "")))
        broad_clade = clean_json_value(
            metadata.get("broad_clade")
            or classify_alignment_broad_clade(str(clade_name or ""), str(phylum or ""), str(taxonomy_level or ""))
        )
        descendant_species = sorted({
            str(member)
            for child in children
            for member in (child.get("descendant_species") or [])
            if str(member).strip()
        })
        if not children and species_value:
            descendant_species = [str(species_value)]
        signature = tree_species_signature(descendant_species)
        group = group_lookup.get(signature) or {}
        leaf_meta = (
            leaf_by_record.get(str(protein_record_id or "").strip())
            or leaf_by_species.get(str(species_value or "").strip())
            or {}
        )
        tip_records = [
            str(record_id)
            for child in children
            for record_id in (child.get("tip_records") or [])
            if str(record_id).strip()
        ]
        if not children:
            record_value = clean_json_value(metadata.get("record_id") or normalized_name)
            tip_records = [str(record_value)] if record_value else []
        return {
            "id": current_node_id,
            "name": clean_json_value(raw_name),
            "branch_length": branch_length,
            "record_id": clean_json_value(metadata.get("record_id") or (normalized_name if not children else None)),
            "protein_record_id": protein_record_id,
            "tip_records": tip_records,
            "descendant_species": descendant_species,
            "species": clean_json_value(metadata.get("species") or species_value),
            "symbol": clean_json_value(metadata.get("symbol") or symbol),
            "clade": clade_name,
            "taxonomy_level": taxonomy_level,
            "phylum": phylum,
            "broad_clade": broad_clade,
            "clade_age_mya": clean_json_value(metadata.get("clade_age_mya")),
            "clade_age_source": clean_json_value(metadata.get("clade_age_source")),
            "species_display_label": clean_json_value(metadata.get("species_display_label") or leaf_meta.get("species_display_label")),
            "scientific_name": clean_json_value(metadata.get("scientific_name") or leaf_meta.get("scientific_name")),
            "common_name": clean_json_value(metadata.get("common_name") or leaf_meta.get("common_name")),
            "preferred_public_label": clean_json_value(metadata.get("preferred_public_label") or leaf_meta.get("preferred_label")),
            "preferred_public_gene_label": clean_json_value(metadata.get("preferred_public_gene_label") or leaf_meta.get("preferred_gene_label")),
            "preferred_protein_name": clean_json_value(metadata.get("preferred_protein_name") or leaf_meta.get("preferred_protein_label")),
            "nomenclature_leaf_label": clean_json_value(metadata.get("nomenclature_leaf_label") or leaf_meta.get("preferred_label")),
            "nomenclature_group_label": clean_json_value(group.get("label")),
            "nomenclature_event_type": clean_json_value(group.get("event_type")),
            "nomenclature_group_source": clean_json_value(group.get("source")),
            "nomenclature_group_count": clean_json_value(group.get("homolog_count")),
            "uniprot_accession": clean_json_value(metadata.get("uniprot_accession")),
            "reviewed_status": clean_json_value(metadata.get("reviewed_status")),
            "alphafold_entry_id": clean_json_value(metadata.get("alphafold_entry_id")),
            "children": children,
        }

    return walk(clade, node_id)


def build_tree_viewer_data(outdir: Path | str, alignment_species_df: pd.DataFrame) -> Dict[str, Any]:
    outdir_path = Path(outdir)
    tree_path = choose_existing_artifact(outdir_path, ["phylo.treefile", "phylo.contree", "phylo.bionj"])
    if not tree_path:
        return {
            "available": False,
            "message": "No phylogeny tree file was found for this run.",
        }
    try:
        tree = Phylo.read(str(outdir_path / tree_path), "newick")
    except Exception as exc:
        return {
            "available": False,
            "source_tree": tree_path,
            "message": f"Could not parse {tree_path}: {exc}",
        }

    metadata_by_record, metadata_by_species = _tree_viewer_metadata_maps(alignment_species_df)
    tree_nomenclature = load_tree_nomenclature_payload(outdir_path)
    return {
        "available": True,
        "source_tree": tree_path,
        "nomenclature_source": clean_json_value(tree_nomenclature.get("source")),
        "event_types": clean_json_value(tree_nomenclature.get("event_types") or []),
        "tip_count": len(tree.get_terminals()),
        "mya_legend": _tree_viewer_mya_legend(alignment_species_df),
        "root": _tree_viewer_node_to_json(tree.root, metadata_by_record, metadata_by_species, tree_nomenclature=tree_nomenclature),
    }


def load_alignment_browser_records(outdir: Path) -> Dict[str, List[Dict[str, Any]]]:
    records_by_scope: Dict[str, List[Dict[str, Any]]] = {}
    for scope, spec in ALIGNMENT_BROWSER_SCOPES.items():
        fasta_path = outdir / spec["filename"]
        if not fasta_path.exists():
            records_by_scope[scope] = []
            continue
        rows: List[Dict[str, Any]] = []
        for record_index, record in enumerate(SeqIO.parse(str(fasta_path), "fasta"), start=1):
            sequence = str(record.seq)
            species, symbol = parse_header_species_symbol(record.id)
            rows.append({
                "alignment_scope": scope,
                "scope_label": spec["label"],
                "source_fasta": spec["filename"],
                "record_index": record_index,
                "record_id": record.id,
                "species": species,
                "symbol": symbol,
                "description": record.description,
                "sequence": sequence,
            })
        records_by_scope[scope] = rows
    return records_by_scope


def build_reference_maps(records: Sequence[Dict[str, Any]], reference_species: Optional[str]) -> Tuple[int, List[Optional[int]], List[str]]:
    if not records:
        return 0, [], []
    ref_index = 0
    if reference_species:
        for idx, record in enumerate(records):
            if record["species"] == reference_species:
                ref_index = idx
                break
    ref_seq = str(records[ref_index]["sequence"])
    reference_positions: List[Optional[int]] = []
    reference_residues: List[str] = []
    ref_ungapped_pos = 0
    for aa in ref_seq:
        aa_upper = aa.upper()
        reference_residues.append(aa_upper)
        if aa_upper not in GAP_CHARS:
            ref_ungapped_pos += 1
            reference_positions.append(ref_ungapped_pos)
        else:
            reference_positions.append(None)
    return ref_index, reference_positions, reference_residues


def build_alignment_archive_tables(outdir: Path,
                                   tables: Dict[str, pd.DataFrame],
                                   reference_species: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    taxonomy_lookup = build_taxonomy_lookup(tables.get("orthologs", pd.DataFrame()))
    protein_metadata_df = tables.get("protein_metadata", pd.DataFrame())
    protein_lookup_by_id: Dict[str, Dict[str, Any]] = {}
    protein_lookup_by_species: Dict[str, Dict[str, Any]] = {}
    if protein_metadata_df is not None and not protein_metadata_df.empty:
        for _, row in protein_metadata_df.iterrows():
            row_dict = {
                str(key): clean_json_value(value)
                for key, value in row.items()
            }
            protein_record_id = str(row_dict.get("protein_record_id") or "").strip()
            species = str(row_dict.get("species") or "").strip()
            if protein_record_id and protein_record_id not in protein_lookup_by_id:
                protein_lookup_by_id[protein_record_id] = row_dict
            if species and species not in protein_lookup_by_species:
                protein_lookup_by_species[species] = row_dict
    tree_record_order, tree_species_order, tree_order_source = parse_tree_leaf_order(outdir)
    records_by_scope = load_alignment_browser_records(outdir)
    species_rows: List[Dict[str, Any]] = []
    cell_rows: List[Dict[str, Any]] = []

    for scope, records in records_by_scope.items():
        if not records:
            continue
        ref_index, reference_positions, reference_residues = build_reference_maps(records, reference_species)
        reference_record = records[ref_index]
        alignment_length = len(reference_residues)

        for record in records:
            sequence = str(record["sequence"])
            species = str(record["species"])
            protein_record_id = clean_json_value(parse_header_field(record["record_id"], "ProteinRecordID"))
            protein_meta = (
                protein_lookup_by_id.get(str(protein_record_id or "").strip())
                or protein_lookup_by_species.get(species)
                or {}
            )
            taxonomy = taxonomy_lookup.get(species, {})
            taxonomy_level = taxonomy.get("taxonomy_level")
            clade = classify_alignment_clade(species, taxonomy_level)
            phylum = classify_alignment_phylum(species, taxonomy_level)
            broad_clade = classify_alignment_broad_clade(clade, phylum, taxonomy_level)
            clade_age_mya = default_clade_age_mya(clade)
            record_key = normalize_tree_record_key(record["record_id"])
            tree_order = (
                tree_record_order.get(record["record_id"])
                or tree_record_order.get(record_key)
                or tree_species_order.get(species)
                or int(record["record_index"])
            )
            tree_order_source_value = tree_order_source if tree_record_order or tree_species_order else "record_order"

            species_position = 0
            comparable_count = 0
            match_count = 0
            mismatch_count = 0
            gap_count = 0
            for pos0, aa in enumerate(sequence):
                aa_upper = aa.upper()
                is_gap = aa_upper in GAP_CHARS
                if is_gap:
                    gap_count += 1
                else:
                    species_position += 1
                ref_residue = reference_residues[pos0] if pos0 < len(reference_residues) else None
                ref_position = reference_positions[pos0] if pos0 < len(reference_positions) else None
                comparable = bool((not is_gap) and ref_residue not in GAP_CHARS and ref_residue is not None)
                match = bool(comparable and aa_upper == ref_residue)
                if comparable:
                    comparable_count += 1
                    if match:
                        match_count += 1
                    else:
                        mismatch_count += 1
                cell_rows.append({
                    "alignment_scope": scope,
                    "species": species,
                    "taxonomy_level": taxonomy_level,
                    "clade": clade,
                    "tree_order": int(tree_order),
                    "alignment_position": pos0 + 1,
                    "reference_position": ref_position,
                    "ungapped_species_position": None if is_gap else species_position,
                    "residue": aa_upper,
                    "is_gap": is_gap,
                    "reference_residue": ref_residue,
                    "match_to_reference": match,
                    "aa_color": AA_COLORS.get(aa_upper, AA_COLORS.get("X", "#F2F2F2")),
                })

            ungapped_length = len(sequence) - gap_count
            species_rows.append({
                "alignment_scope": scope,
                "scope_label": record["scope_label"],
                "source_fasta": record["source_fasta"],
                "record_index": record["record_index"],
                "record_id": record["record_id"],
                "species": species,
                "symbol": record["symbol"],
                "protein_record_id": protein_record_id or clean_json_value(protein_meta.get("protein_record_id")),
                "description": record["description"],
                "taxonomy_level": taxonomy_level,
                "clade": clade,
                "phylum": phylum,
                "broad_clade": broad_clade,
                "clade_age_mya": clade_age_mya,
                "clade_age_source": "clade_default" if clade_age_mya is not None else None,
                "clade_order_rank": CLADE_ORDER.index(clade) + 1 if clade in CLADE_ORDER else len(CLADE_ORDER) + 1,
                "tree_order": int(tree_order),
                "tree_order_source": tree_order_source_value,
                "is_reference": species == reference_record["species"],
                "aligned_length": alignment_length,
                "ungapped_length": ungapped_length,
                "gap_count": gap_count,
                "gap_fraction": (gap_count / alignment_length) if alignment_length else None,
                "comparable_to_reference_count": comparable_count,
                "match_to_reference_count": match_count,
                "mismatch_to_reference_count": mismatch_count,
                "identity_to_reference": (match_count / comparable_count) if comparable_count else None,
                "species_display_label": clean_json_value(protein_meta.get("species_display_label")),
                "scientific_name": clean_json_value(protein_meta.get("scientific_name")),
                "common_name": clean_json_value(protein_meta.get("common_name")),
                "preferred_public_label": clean_json_value(protein_meta.get("preferred_public_label")),
                "preferred_public_gene_label": clean_json_value(protein_meta.get("preferred_public_gene_label")),
                "preferred_protein_name": clean_json_value(protein_meta.get("preferred_protein_name")),
                "nomenclature_leaf_label": clean_json_value(protein_meta.get("nomenclature_leaf_label")),
                "uniprot_accession": clean_json_value(protein_meta.get("uniprot_accession")),
                "reviewed_status": clean_json_value(protein_meta.get("reviewed_status")),
                "alphafold_entry_id": clean_json_value(protein_meta.get("alphafold_entry_id")),
                "alphafold_source_label": clean_json_value(protein_meta.get("alphafold_source_label")),
                "aligned_sequence": sequence,
                "reference_species": reference_record["species"],
                "reference_record_id": reference_record["record_id"],
            })

    phylum_anchor_ages: Dict[str, float] = {}
    for row in species_rows:
        phylum = str(row.get("phylum") or "").strip()
        age_value = finite_age_mya(row.get("clade_age_mya"))
        if not phylum or age_value is None:
            continue
        phylum_anchor_ages[phylum] = max(phylum_anchor_ages.get(phylum, age_value), age_value)

    for row in species_rows:
        if finite_age_mya(row.get("clade_age_mya")) is not None:
            continue
        derived_age, derived_source = derive_alignment_age_mya(
            species=str(row.get("species") or ""),
            taxonomy_level=row.get("taxonomy_level"),
            clade=row.get("clade"),
            phylum=row.get("phylum"),
            phylum_anchor_ages=phylum_anchor_ages,
        )
        row["clade_age_mya"] = derived_age
        row["clade_age_source"] = derived_source

    return pd.DataFrame(species_rows), pd.DataFrame(cell_rows)


def node_conservation_species_label(row: Dict[str, Any]) -> str:
    label = clean_json_value(row.get("common_name") or row.get("species_display_label"))
    if not label:
        label = clean_json_value(row.get("scientific_name"))
    if not label:
        label = str(row.get("species") or "").replace("_", " ")
    text = str(label or "Unknown").strip()
    return text[:1].upper() + text[1:] if text else "Unknown"


def node_conservation_protein_id(row: Dict[str, Any]) -> str:
    record_id = str(row.get("record_id") or "")
    protein = parse_header_field(record_id, "Protein")
    if protein:
        return protein
    protein_record_id = str(row.get("protein_record_id") or "").strip()
    if "__" in protein_record_id:
        return protein_record_id.split("__", 1)[1]
    return protein_record_id


def node_conservation_record_label(row: Dict[str, Any]) -> str:
    label = node_conservation_species_label(row)
    protein_id = node_conservation_protein_id(row)
    return f"{label} ({protein_id})" if protein_id else label


def node_conservation_clean_sequence(sequence: Any) -> str:
    return "".join(ch for ch in str(sequence or "").upper() if ch.isalpha())


def node_conservation_pairwise_identity(reference_seq: str, target_seq: str) -> Tuple[Optional[float], int, int, int, Optional[float]]:
    ref = node_conservation_clean_sequence(reference_seq)
    target = node_conservation_clean_sequence(target_seq)
    if not ref or not target:
        return None, 0, 0, 0, None
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = BLOSUM62
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    alignments = aligner.align(ref, target)
    if len(alignments) < 1:
        return None, 0, 0, 0, None
    alignment = alignments[0]
    ref_blocks, target_blocks = alignment.aligned
    comparable = 0
    matches = 0
    for (ref_start, ref_end), (target_start, target_end) in zip(ref_blocks, target_blocks):
        ref_start = int(ref_start)
        ref_end = int(ref_end)
        target_start = int(target_start)
        target_end = int(target_end)
        block_len = min(ref_end - ref_start, target_end - target_start)
        if block_len <= 0:
            continue
        comparable += block_len
        matches += sum(
            1 for offset in range(block_len)
            if ref[ref_start + offset] == target[target_start + offset]
        )
    if comparable <= 0:
        return None, 0, 0, 0, None
    gap_count = abs(len(ref) - len(target))
    return matches / comparable, comparable, matches, comparable - matches, gap_count / max(1, len(ref))


def node_conservation_load_extra_sequences(outdir: Path) -> Dict[str, str]:
    path = outdir / NODE_CONSERVATION_REJECTED_SEQUENCE_FILENAME
    if not path.exists():
        return {}
    sequences: Dict[str, str] = {}
    for record in SeqIO.parse(str(path), "fasta"):
        seq = node_conservation_clean_sequence(str(record.seq))
        keys = {
            str(record.id),
            parse_header_field(record.id, "ProteinRecordID") or "",
            parse_header_field(record.id, "Protein") or "",
        }
        species, _symbol = parse_header_species_symbol(record.id)
        protein_id = parse_header_field(record.id, "Protein") or ""
        if species and protein_id:
            keys.add(f"{species}__{protein_id}")
        for key in keys:
            if key:
                sequences[key] = seq
    return sequences


def node_conservation_first_value(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        text = str(value)
        if text.strip() != "":
            return value
    return None


def node_conservation_metadata_maps(metadata_df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    by_protein: Dict[str, Dict[str, Any]] = {}
    by_species: Dict[str, Dict[str, Any]] = {}
    if metadata_df is None or metadata_df.empty:
        return by_protein, by_species
    for row in dataframe_to_json_records(metadata_df):
        protein_keys = {
            str(row.get("protein_record_id") or ""),
            str(row.get("translation_id") or ""),
            str(row.get("ensembl_protein_id") or ""),
            parse_header_field(str(row.get("sequence_header") or ""), "Protein") or "",
            parse_header_field(str(row.get("sequence_header") or ""), "ProteinRecordID") or "",
        }
        for key in protein_keys:
            if key:
                by_protein[key] = row
        species_key = str(row.get("species") or "")
        if species_key and species_key not in by_species:
            by_species[species_key] = row
    return by_protein, by_species


def build_rejected_node_conservation_records(outdir: Path,
                                             reference_sequence: str,
                                             reference_species: Optional[str] = None) -> List[Dict[str, Any]]:
    length_df = load_output_table(outdir, "length_filter_report.csv", ",")
    if length_df.empty:
        return []
    metadata_df = load_output_table(outdir, "protein_metadata.tsv", "\t")
    meta_by_protein, meta_by_species = node_conservation_metadata_maps(metadata_df)
    extra_sequences = node_conservation_load_extra_sequences(outdir)
    rows: List[Dict[str, Any]] = []
    for row in dataframe_to_json_records(length_df):
        status = str(node_conservation_first_value(row.get("length_filter_status"), row.get("record_length_filter_status")) or "")
        if not status.startswith("rejected"):
            continue
        species = str(row.get("species") or row.get("record_species") or "")
        if reference_species and species == str(reference_species):
            continue
        protein_record_id = str(row.get("protein_record_id") or "")
        protein_id = str(row.get("translation_id") or parse_header_field(str(row.get("sequence_header") or ""), "Protein") or "")
        sequence_header = str(row.get("sequence_header") or "")
        meta = (
            meta_by_protein.get(protein_record_id)
            or meta_by_protein.get(protein_id)
            or meta_by_species.get(species)
            or {}
        )
        target_sequence = (
            extra_sequences.get(sequence_header)
            or extra_sequences.get(protein_record_id)
            or extra_sequences.get(protein_id)
            or (extra_sequences.get(f"{species}__{protein_id}") if protein_id else None)
        )
        identity, comparable, matches, mismatches, gap_fraction = node_conservation_pairwise_identity(reference_sequence, target_sequence or "")
        if identity is None:
            continue
        length_aa = node_conservation_first_value(row.get("length_aa"), row.get("record_length_aa_from_record"), meta.get("length_aa"))
        rows.append({
            "alignment_scope": "node_extrema_rejected_pairwise",
            "record_id": sequence_header or protein_record_id or f"{species}|Protein={protein_id}",
            "species": species,
            "symbol": node_conservation_first_value(row.get("symbol"), meta.get("symbol")),
            "protein_record_id": protein_record_id or meta.get("protein_record_id"),
            "taxonomy_level": node_conservation_first_value(meta.get("taxonomy_level"), row.get("taxonomy_level")),
            "clade": node_conservation_first_value(meta.get("clade"), row.get("clade")),
            "phylum": node_conservation_first_value(meta.get("phylum"), row.get("phylum")),
            "broad_clade": node_conservation_first_value(meta.get("broad_clade"), row.get("broad_clade")),
            "species_display_label": node_conservation_first_value(row.get("species_display_label"), meta.get("species_display_label")),
            "scientific_name": node_conservation_first_value(row.get("scientific_name"), meta.get("scientific_name")),
            "common_name": node_conservation_first_value(row.get("common_name"), meta.get("common_name")),
            "preferred_public_label": node_conservation_first_value(row.get("preferred_public_label"), meta.get("preferred_public_label")),
            "preferred_public_gene_label": node_conservation_first_value(row.get("preferred_public_gene_label"), meta.get("preferred_public_gene_label")),
            "identity_to_reference": identity,
            "ungapped_length": int(float(length_aa)) if length_aa is not None else len(node_conservation_clean_sequence(target_sequence)),
            "comparable_to_reference_count": comparable,
            "match_to_reference_count": matches,
            "mismatch_to_reference_count": mismatches,
            "gap_fraction": gap_fraction,
            "is_reference": False,
            "length_filter_status": status,
            "node_extrema_input_status": status,
            "node_extrema_metric_source": "rejected_raw_sequence_pairwise_global",
        })
    return rows


def node_conservation_definitions() -> List[Dict[str, Any]]:
    jawless = {"petromyzon_marinus", "eptatretus_burgeri"}
    cartilaginous = {"callorhinchus_milii"}
    holosteans = {"lepisosteus_oculatus"}
    coelacanth = {"latimeria_chalumnae"}
    non_teleost_ray_finned = {
        "erpetoichthys_calabaricus", "polypterus_senegalus", "polypterus_bichir",
        "polypterus_endlicherii",
    }
    amphibians = {"xenopus_tropicalis", "leptobrachium_leishanense"}
    monotremes = {"ornithorhynchus_anatinus"}
    marsupials = {
        "sarcophilus_harrisii", "notamacropus_eugenii", "vombatus_ursinus",
        "phascolarctos_cinereus", "monodelphis_domestica",
    }
    birds = {
        "anas_platyrhynchos_platyrhynchos", "gallus_gallus", "serinus_canaria",
        "meleagris_gallopavo", "parus_major", "ficedula_albicollis",
        "taeniopygia_guttata", "geospiza_fortis", "aquila_chrysaetos_chrysaetos",
        "anser_brachyrhynchus", "struthio_camelus_australis", "coturnix_japonica",
        "strigops_habroptila",
    }
    reptiles = {
        "chrysemys_picta_bellii", "crocodylus_porosus", "sphenodon_punctatus",
        "laticauda_laticaudata", "notechis_scutatus", "pseudonaja_textilis",
        "anolis_carolinensis", "naja_naja", "pelodiscus_sinensis",
        "gopherus_evgoodei", "chelonoidis_abingdonii", "salvator_merianae",
        "terrapene_carolina_triunguis", "podarcis_muralis",
    }
    primates = {
        "nomascus_leucogenys", "pan_troglodytes", "pongo_abelii", "pan_paniscus",
        "gorilla_gorilla", "callithrix_jacchus", "cercocebus_atys",
        "macaca_fascicularis", "macaca_mulatta", "macaca_nemestrina",
        "papio_anubis", "mandrillus_leucophaeus", "rhinopithecus_roxellana",
        "rhinopithecus_bieti", "chlorocebus_sabaeus", "cebus_imitator",
        "saimiri_boliviensis_boliviensis", "aotus_nancymaae", "carlito_syrichta",
        "microcebus_murinus", "prolemur_simus", "propithecus_coquereli",
        "otolemur_garnettii",
    }
    old_world_monkeys = {
        "cercocebus_atys", "macaca_fascicularis", "macaca_mulatta",
        "macaca_nemestrina", "papio_anubis", "mandrillus_leucophaeus",
        "rhinopithecus_roxellana", "rhinopithecus_bieti", "chlorocebus_sabaeus",
    }
    apes = {
        "nomascus_leucogenys", "pan_troglodytes", "pongo_abelii",
        "pan_paniscus", "gorilla_gorilla",
    }
    placental_taxa = {
        "eutheria", "boreoeutheria", "euarchontoglires", "catarrhini",
        "simiiformes", "hominoidea", "hominidae", "homininae",
    }

    def species(row: Dict[str, Any]) -> str:
        return str(row.get("species") or "").strip()

    def taxonomy(row: Dict[str, Any]) -> str:
        return str(row.get("taxonomy_level") or "").strip().lower()

    def clade(row: Dict[str, Any]) -> str:
        return str(row.get("clade") or "").strip().lower()

    def phylum(row: Dict[str, Any]) -> str:
        return str(row.get("phylum") or "").strip().lower()

    def is_bony_fish(row: Dict[str, Any]) -> bool:
        return (
            (
                taxonomy(row) in {"euteleostomi", "actinopterygii", "sarcopterygii"}
                and not is_tetrapod(row)
            )
            or clade(row) in {"dipnoi", "actinistia"}
            or species(row) in DIPNOI_SPECIES | holosteans | coelacanth | non_teleost_ray_finned
        )

    def is_teleost(row: Dict[str, Any]) -> bool:
        return (
            taxonomy(row) in {"euteleostomi", "teleostei"}
            and species(row) not in holosteans | non_teleost_ray_finned
            and not is_tetrapod(row)
        )

    def is_placental(row: Dict[str, Any]) -> bool:
        if species(row) in primates:
            return True
        return taxonomy(row) in placental_taxa and species(row) not in marsupials | monotremes

    def is_mammal(row: Dict[str, Any]) -> bool:
        return species(row) in marsupials | monotremes or is_placental(row)

    def is_tetrapod(row: Dict[str, Any]) -> bool:
        return species(row) in amphibians | reptiles | birds or is_mammal(row)

    def is_vertebrate(row: Dict[str, Any]) -> bool:
        if species(row).startswith("ciona_"):
            return False
        return (
            phylum(row) == "chordata"
            and (species(row) in jawless | cartilaginous or is_bony_fish(row) or is_tetrapod(row))
        )

    return [
        {"key": "vertebrates", "label": "Vertebrates", "parent": "", "rule": "Chordata vertebrate set; human query excluded", "predicate": is_vertebrate},
        {"key": "jawless_vertebrates", "label": "Jawless vertebrates", "parent": "vertebrates", "rule": "Lamprey and hagfish", "predicate": lambda row: species(row) in jawless},
        {"key": "jawed_vertebrates", "label": "Jawed vertebrates", "parent": "vertebrates", "rule": "Vertebrates excluding lamprey/hagfish", "predicate": lambda row: is_vertebrate(row) and species(row) not in jawless},
        {"key": "cartilaginous_fish", "label": "Cartilaginous fish", "parent": "jawed_vertebrates", "rule": "Elephant shark in this run", "predicate": lambda row: species(row) in cartilaginous},
        {"key": "bony_vertebrates", "label": "Bony vertebrates", "parent": "jawed_vertebrates", "rule": "Jawed vertebrates excluding cartilaginous fish", "predicate": lambda row: is_vertebrate(row) and species(row) not in jawless | cartilaginous},
        {"key": "bony_fishes", "label": "Bony fishes", "parent": "bony_vertebrates", "rule": "Taxonomy/species-supported bony fish records; tetrapods excluded", "predicate": is_bony_fish},
        {"key": "teleosts", "label": "Teleosts", "parent": "bony_fishes", "rule": "Teleost-like Euteleostomi records; gar and reedfish excluded", "predicate": is_teleost},
        {"key": "holosteans", "label": "Holosteans", "parent": "bony_fishes", "rule": "Spotted gar in this run", "predicate": lambda row: species(row) in holosteans},
        {"key": "coelacanth", "label": "Coelacanth", "parent": "bony_fishes", "rule": "Latimeria chalumnae", "predicate": lambda row: species(row) in coelacanth},
        {"key": "tetrapods", "label": "Tetrapods", "parent": "bony_vertebrates", "rule": "Amphibians, reptiles/birds, mammals", "predicate": is_tetrapod},
        {"key": "amphibians", "label": "Amphibians", "parent": "tetrapods", "rule": "Xenopus and Leptobrachium records", "predicate": lambda row: species(row) in amphibians},
        {"key": "amniotes", "label": "Amniotes", "parent": "tetrapods", "rule": "Reptiles/birds and mammals", "predicate": lambda row: species(row) in reptiles | birds or is_mammal(row)},
        {"key": "reptiles_birds", "label": "Reptiles and birds", "parent": "amniotes", "rule": "Sauropsid records", "predicate": lambda row: species(row) in reptiles | birds},
        {"key": "reptiles", "label": "Reptiles", "parent": "reptiles_birds", "rule": "Turtle, crocodile, lizard, snake, tuatara records", "predicate": lambda row: species(row) in reptiles},
        {"key": "birds", "label": "Birds", "parent": "reptiles_birds", "rule": "Avian records", "predicate": lambda row: species(row) in birds},
        {"key": "mammals", "label": "Mammals", "parent": "amniotes", "rule": "Monotreme, marsupial, and placental records", "predicate": is_mammal},
        {"key": "monotremes", "label": "Monotremes", "parent": "mammals", "rule": "Platypus in this run", "predicate": lambda row: species(row) in monotremes},
        {"key": "marsupials", "label": "Marsupials", "parent": "mammals", "rule": "Marsupial species records", "predicate": lambda row: species(row) in marsupials},
        {"key": "placentals", "label": "Placentals", "parent": "mammals", "rule": "Eutherian records; human query excluded", "predicate": is_placental},
        {"key": "non_human_primates", "label": "Non-human primates", "parent": "placentals", "rule": "Primate species excluding Homo sapiens", "predicate": lambda row: species(row) in primates},
        {"key": "apes", "label": "Apes (non-human)", "parent": "non_human_primates", "rule": "Gibbon, chimpanzee/bonobo, gorilla, orangutan", "predicate": lambda row: species(row) in apes},
        {"key": "old_world_monkeys", "label": "Old World monkeys", "parent": "non_human_primates", "rule": "Catarrhine monkey records excluding apes", "predicate": lambda row: species(row) in old_world_monkeys},
        {"key": "non_primate_placentals", "label": "Non-primate placentals", "parent": "placentals", "rule": "Placental mammals excluding primates", "predicate": lambda row: is_placental(row) and species(row) not in primates},
    ]


def build_node_conservation_extremes(alignment_species_df: pd.DataFrame,
                                     reference_species: Optional[str] = None,
                                     outdir: Optional[Path] = None,
                                     include_rejected: bool = True) -> pd.DataFrame:
    columns = [
        "node_key", "node_label", "parent_key", "parent_label", "tree_depth",
        "record_count", "species_count", "identity_range_percent",
        "min_identity_fraction", "min_identity_percent", "max_identity_fraction", "max_identity_percent",
        "least_conserved_species", "least_conserved_label", "least_conserved_protein_id", "least_conserved_record_id",
        "least_conserved_length_aa", "least_comparable_to_reference_count", "least_gap_fraction",
        "least_conserved_input_status", "least_conserved_metric_source",
        "most_conserved_species", "most_conserved_label", "most_conserved_protein_id", "most_conserved_record_id",
        "most_conserved_length_aa", "most_comparable_to_reference_count", "most_gap_fraction",
        "most_conserved_input_status", "most_conserved_metric_source",
        "membership_rule",
    ]
    if alignment_species_df is None or alignment_species_df.empty:
        return pd.DataFrame(columns=columns)

    df = alignment_species_df.copy()
    df = df[df.get("alignment_scope", "") == "aligned_reference_projected"].copy()
    reference_sequence = ""
    if not df.empty:
        ref_mask = df["species"].astype(str).eq(str(reference_species or "")) if reference_species else pd.Series(False, index=df.index)
        if "is_reference" in df.columns:
            ref_mask = ref_mask | df["is_reference"].astype(bool)
        if ref_mask.any() and "aligned_sequence" in df.columns:
            reference_sequence = node_conservation_clean_sequence(str(df.loc[ref_mask, "aligned_sequence"].iloc[0]))
    if reference_species:
        df = df[df["species"].astype(str) != str(reference_species)]
    if "is_reference" in df.columns:
        df = df[~df["is_reference"].astype(bool)]
    df["identity_to_reference"] = pd.to_numeric(df.get("identity_to_reference"), errors="coerce")
    df = df[df["identity_to_reference"].notna()].copy()
    if "length_filter_status" not in df.columns:
        df["length_filter_status"] = "kept"
    df["node_extrema_input_status"] = df["length_filter_status"].fillna("kept").replace("", "kept")
    df["node_extrema_metric_source"] = "reference_projected_alignment"
    records = dataframe_to_json_records(df)
    if include_rejected and outdir is not None and reference_sequence:
        records.extend(build_rejected_node_conservation_records(outdir, reference_sequence, reference_species=reference_species))
    if not records:
        return pd.DataFrame(columns=columns)
    definitions = node_conservation_definitions()
    label_by_key = {str(defn["key"]): str(defn["label"]) for defn in definitions}
    depth_by_key: Dict[str, int] = {}

    def depth_for(key: str) -> int:
        if not key:
            return 0
        if key in depth_by_key:
            return depth_by_key[key]
        parent = ""
        for defn in definitions:
            if str(defn["key"]) == key:
                parent = str(defn.get("parent") or "")
                break
        depth_by_key[key] = 0 if not parent else depth_for(parent) + 1
        return depth_by_key[key]

    rows: List[Dict[str, Any]] = []
    for defn in definitions:
        predicate = defn["predicate"]
        members = [row for row in records if predicate(row)]
        if not members:
            continue
        members = sorted(
            members,
            key=lambda row: (
                float(row.get("identity_to_reference") or 0.0),
                str(row.get("species") or ""),
                str(row.get("protein_record_id") or ""),
            ),
        )
        least = members[0]
        most = members[-1]
        min_identity = float(least.get("identity_to_reference") or 0.0)
        max_identity = float(most.get("identity_to_reference") or 0.0)
        parent_key = str(defn.get("parent") or "")
        rows.append({
            "node_key": defn["key"],
            "node_label": defn["label"],
            "parent_key": parent_key,
            "parent_label": label_by_key.get(parent_key, ""),
            "tree_depth": depth_for(str(defn["key"])),
            "record_count": len(members),
            "species_count": len({str(row.get("species") or "") for row in members}),
            "identity_range_percent": f"{min_identity * 100.0:.2f}-{max_identity * 100.0:.2f}%",
            "min_identity_fraction": min_identity,
            "min_identity_percent": round(min_identity * 100.0, 4),
            "max_identity_fraction": max_identity,
            "max_identity_percent": round(max_identity * 100.0, 4),
            "least_conserved_species": least.get("species"),
            "least_conserved_label": node_conservation_record_label(least),
            "least_conserved_protein_id": node_conservation_protein_id(least),
            "least_conserved_record_id": least.get("record_id"),
            "least_conserved_length_aa": least.get("ungapped_length"),
            "least_comparable_to_reference_count": least.get("comparable_to_reference_count"),
            "least_gap_fraction": least.get("gap_fraction"),
            "least_conserved_input_status": least.get("node_extrema_input_status") or least.get("length_filter_status") or "kept",
            "least_conserved_metric_source": least.get("node_extrema_metric_source") or "reference_projected_alignment",
            "most_conserved_species": most.get("species"),
            "most_conserved_label": node_conservation_record_label(most),
            "most_conserved_protein_id": node_conservation_protein_id(most),
            "most_conserved_record_id": most.get("record_id"),
            "most_conserved_length_aa": most.get("ungapped_length"),
            "most_comparable_to_reference_count": most.get("comparable_to_reference_count"),
            "most_gap_fraction": most.get("gap_fraction"),
            "most_conserved_input_status": most.get("node_extrema_input_status") or most.get("length_filter_status") or "kept",
            "most_conserved_metric_source": most.get("node_extrema_metric_source") or "reference_projected_alignment",
            "membership_rule": defn.get("rule"),
        })
    return pd.DataFrame(rows, columns=columns)


def truncate_svg_text(text: Any, max_chars: int) -> str:
    value = str(text or "")
    return value if len(value) <= max_chars else value[:max(0, max_chars - 1)] + "..."


def write_node_conservation_tree_svg(node_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if node_df is None or node_df.empty:
        return None
    rows = dataframe_to_json_records(node_df)
    row_height = 44
    top = 96
    left = 24
    tree_x = 42
    depth_width = 28
    table_x = 435
    range_x = 675
    range_width = 260
    percent_x = range_x + range_width + 24
    least_x = 1050
    most_x = 1460
    width = 1840
    height = top + row_height * len(rows) + 56
    color_by_key = {
        "vertebrates": "#2563eb",
        "jawless_vertebrates": "#0891b2",
        "jawed_vertebrates": "#0891b2",
        "cartilaginous_fish": "#7c3aed",
        "bony_vertebrates": "#16a34a",
        "bony_fishes": "#16a34a",
        "teleosts": "#f97316",
        "holosteans": "#f97316",
        "coelacanth": "#f97316",
        "tetrapods": "#22a06b",
        "amphibians": "#f97316",
        "amniotes": "#ea580c",
        "reptiles_birds": "#dc2626",
        "reptiles": "#64748b",
        "birds": "#64748b",
        "mammals": "#ef4444",
        "monotremes": "#64748b",
        "marsupials": "#64748b",
        "placentals": "#64748b",
        "non_human_primates": "#475569",
        "apes": "#475569",
        "old_world_monkeys": "#475569",
        "non_primate_placentals": "#475569",
    }
    y_by_key = {
        str(row.get("node_key")): top + idx * row_height + 16
        for idx, row in enumerate(rows)
    }
    x_by_key = {
        str(row.get("node_key")): tree_x + int(row.get("tree_depth") or 0) * depth_width
        for row in rows
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<style>"
        ".title{font:700 19px Segoe UI,Tahoma,sans-serif;fill:#0f172a}"
        ".subtitle{font:600 10.5px Segoe UI,Tahoma,sans-serif;fill:#64748b}"
        ".colhead{font:800 10.5px Segoe UI,Tahoma,sans-serif;fill:#334155;text-transform:uppercase;letter-spacing:.04em}"
        ".tree-label{font:800 9.5px Segoe UI,Tahoma,sans-serif;fill:#1f2937}"
        ".node-label{font:800 11px Segoe UI,Tahoma,sans-serif;fill:#1f2937}"
        ".node-count{font:700 9.5px Segoe UI,Tahoma,sans-serif;fill:#64748b}"
        ".percent{font:900 13px Segoe UI,Tahoma,sans-serif;fill:#0f172a}"
        ".detail{font:700 10px Segoe UI,Tahoma,sans-serif;fill:#334155}"
        ".detail-id{font:600 9.5px Consolas,Menlo,monospace;fill:#64748b}"
        ".status{font:800 9px Segoe UI,Tahoma,sans-serif;fill:#b45309}"
        ".axis{font:700 8.5px Segoe UI,Tahoma,sans-serif;fill:#64748b}"
        ".link{fill:none;stroke:#78909c;stroke-width:1.15}"
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<rect x="0" y="0" width="100%" height="58" fill="#f8fafc"/>',
        f'<text class="title" x="{left}" y="27">{escape(v11_gene_label())} node conservation extremes</text>',
        f'<text class="subtitle" x="{left}" y="47">Compact evolutionary margin plus independent identity-range table; rejected length-filter records are included when raw sequence is available.</text>',
        f'<text class="colhead" x="{tree_x}" y="{top - 28}">evolutionary margin</text>',
        f'<text class="colhead" x="{table_x}" y="{top - 28}">node</text>',
        f'<text class="colhead" x="{range_x}" y="{top - 28}">identity range</text>',
        f'<text class="colhead" x="{least_x}" y="{top - 28}">least conserved</text>',
        f'<text class="colhead" x="{most_x}" y="{top - 28}">most conserved</text>',
        f'<line x1="{table_x - 18}" y1="{top - 44}" x2="{table_x - 18}" y2="{height - 30}" stroke="#cbd5e1" stroke-width="1"/>',
    ]
    for tick in (0, 50, 75, 90, 100):
        x = range_x + (tick / 100.0) * range_width
        parts.append(f'<line x1="{x:.2f}" y1="{top - 14}" x2="{x:.2f}" y2="{height - 32}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text class="axis" x="{x:.2f}" y="{top - 19}" text-anchor="middle">{tick}%</text>')

    for idx, row in enumerate(rows):
        y = top + idx * row_height
        fill = "#ffffff" if idx % 2 == 0 else "#f8fafc"
        parts.append(f'<rect x="{table_x - 24}" y="{y - 4}" width="{width - table_x + 4}" height="{row_height - 2}" fill="{fill}" opacity="0.94"/>')

    for row in rows:
        key = str(row.get("node_key") or "")
        parent = str(row.get("parent_key") or "")
        if parent and parent in y_by_key:
            x_child = x_by_key[key]
            x_parent = x_by_key.get(parent, tree_x)
            y = y_by_key[key]
            py = y_by_key[parent]
            elbow_x = max(x_parent + 12, x_child - 10)
            parts.append(f'<path class="link" d="M{x_parent:.2f} {py:.2f} H{elbow_x:.2f} V{y:.2f} H{x_child:.2f}"/>')
    parts.append(f'<path d="M{tree_x - 12} {top - 8} V{height - 42}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="3 5"/>')
    parts.append(f'<text class="axis" x="{tree_x - 8}" y="{top - 12}" text-anchor="start">ancestral</text>')

    for idx, row in enumerate(rows):
        y = top + idx * row_height + 16
        depth = int(row.get("tree_depth") or 0)
        key = str(row.get("node_key") or "")
        color = color_by_key.get(key, "#64748b")
        min_pct = float(row.get("min_identity_percent") or 0.0)
        max_pct = float(row.get("max_identity_percent") or min_pct)
        x1 = range_x + max(0.0, min(100.0, min_pct)) / 100.0 * range_width
        x2 = range_x + max(0.0, min(100.0, max_pct)) / 100.0 * range_width
        bar_width = max(2.0, x2 - x1)
        node_x = x_by_key[key]
        tree_label = truncate_svg_text(row.get("node_label"), 22)
        least = truncate_svg_text(row.get("least_conserved_label"), 42)
        most = truncate_svg_text(row.get("most_conserved_label"), 42)
        least_id = truncate_svg_text(row.get("least_conserved_protein_id"), 30)
        most_id = truncate_svg_text(row.get("most_conserved_protein_id"), 30)
        least_status = str(row.get("least_conserved_input_status") or "")
        most_status = str(row.get("most_conserved_input_status") or "")
        least_source = str(row.get("least_conserved_metric_source") or "")
        most_source = str(row.get("most_conserved_metric_source") or "")
        least_note = "rejected / pairwise" if least_status.startswith("rejected") else ("alignment" if least_source else "")
        most_note = "rejected / pairwise" if most_status.startswith("rejected") else ("alignment" if most_source else "")
        parts.append(f'<circle cx="{node_x:.2f}" cy="{y:.2f}" r="4.2" fill="{color}" stroke="#ffffff" stroke-width="1.1"/>')
        parts.append(f'<text class="tree-label" x="{node_x + 7:.2f}" y="{y + 3:.2f}">{escape(tree_label)}</text>')
        parts.append(f'<circle cx="{table_x - 10}" cy="{y - 3:.2f}" r="4" fill="{color}"/>')
        parts.append(f'<text class="node-label" x="{table_x}" y="{y - 6:.2f}">{escape(str(row.get("node_label") or ""))}</text>')
        parts.append(f'<text class="node-count" x="{table_x}" y="{y + 9:.2f}">{int(row.get("record_count") or 0)} records / {int(row.get("species_count") or 0)} species</text>')
        parts.append(f'<rect x="{range_x}" y="{y - 11}" width="{range_width}" height="20" rx="4" fill="#f1f5f9" stroke="#cbd5e1"/>')
        parts.append(f'<rect x="{x1:.2f}" y="{y - 10}" width="{bar_width:.2f}" height="18" rx="4" fill="{color}" opacity="0.82"><title>{escape(str(row.get("identity_range_percent") or ""))}</title></rect>')
        parts.append(f'<text class="percent" x="{percent_x}" y="{y + 4:.2f}">{escape(str(row.get("identity_range_percent") or ""))}</text>')
        parts.append(f'<text class="detail" x="{least_x}" y="{y - 7:.2f}">{escape(least)} - {min_pct:.2f}%</text>')
        parts.append(f'<text class="detail-id" x="{least_x}" y="{y + 8:.2f}">{escape(least_id)}{escape("  |  " + least_note if least_note else "")}</text>')
        parts.append(f'<text class="detail" x="{most_x}" y="{y - 7:.2f}">{escape(most)} - {max_pct:.2f}%</text>')
        parts.append(f'<text class="detail-id" x="{most_x}" y="{y + 8:.2f}">{escape(most_id)}{escape("  |  " + most_note if most_note else "")}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def node_conservation_extreme_short_label(value: Any, max_chars: int = 24) -> str:
    text = str(value or "").strip()
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    return truncate_svg_text(text, max_chars)


def write_node_conservation_paper_tree_svg(node_df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    if node_df is None or node_df.empty:
        return None
    rows_by_key = {str(row.get("node_key") or ""): row for row in dataframe_to_json_records(node_df)}
    leaf_order = [
        "jawless_vertebrates",
        "cartilaginous_fish",
        "teleosts",
        "holosteans",
        "coelacanth",
        "amphibians",
        "reptiles_birds",
        "monotremes",
        "marsupials",
        "non_primate_placentals",
        "non_human_primates",
    ]
    leaf_order = [key for key in leaf_order if key in rows_by_key]
    if not leaf_order:
        return None

    children_by_key = {
        "vertebrates": ["jawless_vertebrates", "jawed_vertebrates"],
        "jawed_vertebrates": ["cartilaginous_fish", "bony_vertebrates"],
        "bony_vertebrates": ["ray_finned_fishes", "sarcopterygian_lineage"],
        "ray_finned_fishes": ["teleosts", "holosteans"],
        "sarcopterygian_lineage": ["coelacanth", "tetrapods"],
        "tetrapods": ["amphibians", "amniotes"],
        "amniotes": ["reptiles_birds", "mammals"],
        "mammals": ["monotremes", "therians"],
        "therians": ["marsupials", "placentals"],
        "placentals": ["non_primate_placentals", "non_human_primates"],
    }
    color_by_key = {
        "vertebrates": "#2563eb",
        "jawless_vertebrates": "#0891b2",
        "jawed_vertebrates": "#0891b2",
        "cartilaginous_fish": "#7c3aed",
        "bony_vertebrates": "#16a34a",
        "ray_finned_fishes": "#16a34a",
        "sarcopterygian_lineage": "#22a06b",
        "teleosts": "#f97316",
        "holosteans": "#f97316",
        "coelacanth": "#f97316",
        "tetrapods": "#22a06b",
        "amphibians": "#f97316",
        "amniotes": "#ea580c",
        "reptiles_birds": "#dc2626",
        "mammals": "#ef4444",
        "therians": "#64748b",
        "monotremes": "#64748b",
        "marsupials": "#64748b",
        "placentals": "#64748b",
        "non_human_primates": "#475569",
        "non_primate_placentals": "#475569",
    }
    x_by_key = {
        "vertebrates": 70,
        "jawless_vertebrates": 210,
        "jawed_vertebrates": 165,
        "cartilaginous_fish": 300,
        "bony_vertebrates": 250,
        "ray_finned_fishes": 365,
        "sarcopterygian_lineage": 320,
        "teleosts": 540,
        "holosteans": 540,
        "coelacanth": 540,
        "tetrapods": 365,
        "amphibians": 540,
        "amniotes": 455,
        "reptiles_birds": 650,
        "mammals": 560,
        "therians": 640,
        "monotremes": 740,
        "marsupials": 760,
        "placentals": 710,
        "non_human_primates": 825,
        "non_primate_placentals": 825,
    }
    node_notes = {
        "vertebrates": ("Vertebrates", "~520-450 Ma"),
        "jawed_vertebrates": ("Jawed vertebrates", "~450-430 Ma"),
        "bony_vertebrates": ("Bony vertebrates", "~430-420 Ma"),
        "ray_finned_fishes": ("Ray-finned fishes", "~430 Ma"),
        "sarcopterygian_lineage": ("Lobe-finned lineage", "~420 Ma"),
        "tetrapods": ("Tetrapods", "~420-320 Ma"),
        "amniotes": ("Amniotes", "~320-225 Ma"),
        "mammals": ("Mammals", "~225-165 Ma"),
        "therians": ("Therian mammals", "~165-125 Ma"),
        "placentals": ("Placentals", "~100 Ma"),
    }
    top = 86
    leaf_gap = 42
    width = 1520
    height = top + max(1, len(leaf_order) - 1) * leaf_gap + 92
    y_by_key: Dict[str, float] = {
        key: top + idx * leaf_gap
        for idx, key in enumerate(leaf_order)
    }

    def present_children(key: str) -> List[str]:
        return [
            child for child in children_by_key.get(key, [])
            if child in rows_by_key or child in children_by_key
        ]

    def y_for(key: str) -> float:
        if key in y_by_key:
            return y_by_key[key]
        kids = [child for child in present_children(key) if child in rows_by_key or child in children_by_key]
        kid_ys = [y_for(child) for child in kids if child in rows_by_key or child in children_by_key]
        y_by_key[key] = sum(kid_ys) / len(kid_ys) if kid_ys else top
        return y_by_key[key]

    y_for("vertebrates")
    bar_x = 1000
    bar_width = 150
    bar_min = 45.0
    bar_max = 100.0

    def bar_pos(percent: Any) -> float:
        pct = max(bar_min, min(bar_max, float(percent or 0.0)))
        return bar_x + (pct - bar_min) / (bar_max - bar_min) * bar_width

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<style>"
        ".title{font:700 18px Arial,Helvetica,sans-serif;fill:#111827}"
        ".subtitle{font:500 10px Arial,Helvetica,sans-serif;fill:#4b5563}"
        ".tree{fill:none;stroke:#111827;stroke-width:3;stroke-linecap:square}"
        ".node{font:700 10.5px Arial,Helvetica,sans-serif;fill:#111827}"
        ".note-title{font:700 10px Arial,Helvetica,sans-serif;fill:#111827;paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round}"
        ".note-age{font:600 9.5px Arial,Helvetica,sans-serif;fill:#374151;paint-order:stroke;stroke:#fff;stroke-width:3px;stroke-linejoin:round}"
        ".leaf{font:700 12px Arial,Helvetica,sans-serif;fill:#111827}"
        ".range{font:700 12px Arial,Helvetica,sans-serif;fill:#111827}"
        ".extreme{font:500 9.5px Arial,Helvetica,sans-serif;fill:#374151}"
        ".axis{font:500 8.5px Arial,Helvetica,sans-serif;fill:#6b7280}"
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="24" y="30">{escape(v11_gene_label())} evolutionary conservation scan across sampled vertebrates</text>',
        f'<text class="subtitle" x="24" y="48">Simplified cladogram summary; ~Ma labels are approximate branch intervals, and percent ranges are identity to human {escape(v11_gene_label())}.</text>',
        f'<text class="axis" x="{bar_x}" y="61">45%</text>',
        f'<text class="axis" x="{bar_x + bar_width * (75 - bar_min) / (bar_max - bar_min):.1f}" y="61" text-anchor="middle">75%</text>',
        f'<text class="axis" x="{bar_x + bar_width}" y="61" text-anchor="end">100%</text>',
    ]
    for tick in (45, 75, 100):
        x = bar_pos(tick)
        parts.append(f'<line x1="{x:.1f}" y1="66" x2="{x:.1f}" y2="{height - 44}" stroke="#e5e7eb" stroke-width="1"/>')

    for parent, children in children_by_key.items():
        kids = [child for child in children if child in rows_by_key or child in children_by_key]
        if not kids:
            continue
        px = x_by_key[parent]
        py = y_for(parent)
        split_x = min(x_by_key.get(child, px + 70) for child in kids) - 26
        y_values = [y_for(child) for child in kids]
        parts.append(f'<line class="tree" x1="{px:.1f}" y1="{py:.1f}" x2="{split_x:.1f}" y2="{py:.1f}"/>')
        parts.append(f'<line class="tree" x1="{split_x:.1f}" y1="{min(y_values):.1f}" x2="{split_x:.1f}" y2="{max(y_values):.1f}"/>')
        for child in kids:
            cy = y_for(child)
            cx = x_by_key.get(child, split_x + 44)
            parts.append(f'<line class="tree" x1="{split_x:.1f}" y1="{cy:.1f}" x2="{cx:.1f}" y2="{cy:.1f}"/>')

    for key, (label, age) in node_notes.items():
        if key in y_by_key and key in x_by_key:
            kids = present_children(key)
            if kids:
                split_x = min(x_by_key.get(child, x_by_key[key] + 70) for child in kids) - 26
                note_x = (x_by_key[key] + split_x) / 2.0
            else:
                note_x = x_by_key[key] + 40
            note_y = y_by_key[key]
            parts.append(f'<text class="note-title" x="{note_x:.1f}" y="{note_y - 5:.1f}" text-anchor="middle">{escape(label)}</text>')
            if age:
                parts.append(f'<text class="note-age" x="{note_x:.1f}" y="{note_y + 10:.1f}" text-anchor="middle">{escape(age)}</text>')

    for key in leaf_order:
        row = rows_by_key[key]
        y = y_for(key)
        color = color_by_key.get(key, "#64748b")
        label = str(row.get("node_label") or key)
        count = int(row.get("species_count") or 0)
        min_pct = float(row.get("min_identity_percent") or 0.0)
        max_pct = float(row.get("max_identity_percent") or min_pct)
        range_text = f"{max_pct:.2f}%" if count <= 1 else str(row.get("identity_range_percent") or "")
        x1 = bar_pos(min_pct)
        x2 = bar_pos(max_pct)
        least = node_conservation_extreme_short_label(row.get("least_conserved_label"), 22)
        most = node_conservation_extreme_short_label(row.get("most_conserved_label"), 22)
        parts.append(f'<circle cx="{x_by_key[key]:.1f}" cy="{y:.1f}" r="7.2" fill="{color}" stroke="#ffffff" stroke-width="1.6"/>')
        parts.append(f'<text class="leaf" x="860" y="{y - 5:.1f}">{escape(label)}</text>')
        parts.append(f'<text class="extreme" x="860" y="{y + 9:.1f}">{count} species</text>')
        parts.append(f'<line x1="{bar_x}" y1="{y:.1f}" x2="{bar_x + bar_width}" y2="{y:.1f}" stroke="#d1d5db" stroke-width="4" stroke-linecap="round"/>')
        parts.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="8" stroke-linecap="round"/>')
        parts.append(f'<text class="range" x="{bar_x + bar_width + 18}" y="{y + 4:.1f}">{escape(range_text)}</text>')
        if count <= 1:
            parts.append(f'<text class="extreme" x="{bar_x + bar_width + 132}" y="{y + 2:.1f}">{escape(least)}</text>')
        else:
            parts.append(f'<text class="extreme" x="{bar_x + bar_width + 132}" y="{y - 5:.1f}">least {escape(least)}</text>')
            parts.append(f'<text class="extreme" x="{bar_x + bar_width + 132}" y="{y + 9:.1f}">most {escape(most)}</text>')

    root_y = y_for("vertebrates")
    parts.append(f'<path d="M{x_by_key["vertebrates"] - 30} {root_y - 15:.1f} v30" stroke="#111827" stroke-width="3"/>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def write_node_conservation_extremes_artifacts(outdir: Path,
                                               alignment_species_df: pd.DataFrame,
                                               reference_species: Optional[str] = None) -> pd.DataFrame:
    node_df = build_node_conservation_extremes(alignment_species_df, reference_species=reference_species, outdir=outdir)
    csv_path = outdir / NODE_CONSERVATION_EXTREMES_FILENAME
    node_df.to_csv(csv_path, index=False)
    write_node_conservation_tree_svg(node_df, outdir / NODE_CONSERVATION_TREE_SVG_FILENAME)
    write_node_conservation_paper_tree_svg(node_df, outdir / NODE_CONSERVATION_PAPER_TREE_SVG_FILENAME)
    return node_df


def load_reference_projected_records(outdir: Path) -> List[Dict[str, Any]]:
    fasta_path = outdir / "aligned_reference_projected.fasta"
    if not fasta_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        sequence = str(record.seq)
        species, symbol = parse_header_species_symbol(record.id)
        rows.append({
            "record_id": record.id,
            "species": species,
            "symbol": symbol,
            "sequence": sequence,
        })
    return rows


def build_pairwise_report_rows(outdir: Path, reference_species: Optional[str]) -> pd.DataFrame:
    records = load_reference_projected_records(outdir)
    if len(records) < 2:
        return pd.DataFrame(columns=[
            "reference_species", "reference_symbol", "reference_record_id", "record_id",
            "species", "symbol", "aligned_length", "shared_non_gap_sites",
            "exact_match_count", "exact_match_fraction", "mismatch_count",
            "target_gap_count", "target_gap_fraction", "text_report_path",
            "pdf_report_path", "png_page_paths", "png_page_count",
        ])

    ref_index = 0
    if reference_species:
        for idx, row in enumerate(records):
            if row["species"] == reference_species:
                ref_index = idx
                break

    reference_row = records[ref_index]
    ref_seq = reference_row["sequence"]
    pair_dir = outdir / PAIRWISE_DIRNAME
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(records):
        if idx == ref_index:
            continue
        seq = row["sequence"]
        aligned_length = min(len(ref_seq), len(seq))
        shared_non_gap = 0
        exact_match_count = 0
        for ref_aa, aa in zip(ref_seq[:aligned_length], seq[:aligned_length]):
            if ref_aa in {"-", "."} or aa in {"-", "."}:
                continue
            shared_non_gap += 1
            if ref_aa == aa:
                exact_match_count += 1
        target_gap_count = sum(1 for aa in seq[:aligned_length] if aa in {"-", "."})
        mismatch_count = max(shared_non_gap - exact_match_count, 0)
        exact_match_fraction = (exact_match_count / shared_non_gap) if shared_non_gap else None
        target_gap_fraction = (target_gap_count / aligned_length) if aligned_length else None

        stem = sanitize_filename(f"human_vs_{row['species']}")
        text_path = f"{PAIRWISE_DIRNAME}/{stem}.txt"
        pdf_path = f"{PAIRWISE_DIRNAME}/{stem}.pdf"
        png_paths = sorted(
            p.relative_to(outdir).as_posix()
            for p in pair_dir.glob(f"{stem}_page_*.png")
        ) if pair_dir.exists() else []

        rows.append({
            "reference_species": reference_row["species"],
            "reference_symbol": reference_row["symbol"],
            "reference_record_id": reference_row["record_id"],
            "record_id": row["record_id"],
            "species": row["species"],
            "symbol": row["symbol"],
            "aligned_length": aligned_length,
            "shared_non_gap_sites": shared_non_gap,
            "exact_match_count": exact_match_count,
            "exact_match_fraction": exact_match_fraction,
            "mismatch_count": mismatch_count,
            "target_gap_count": target_gap_count,
            "target_gap_fraction": target_gap_fraction,
            "text_report_path": text_path if (outdir / text_path).exists() else None,
            "pdf_report_path": pdf_path if (outdir / pdf_path).exists() else None,
            "png_page_paths": png_paths,
            "png_page_count": len(png_paths),
        })

    return pd.DataFrame(rows)


def normalize_clade_identity_profiles(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(columns=[
            "reference_position", "reference_residue", "clade", "n_non_gap",
            "identity_to_human", "hydrophobicity_agreement_to_human",
            "major_residue", "identity_to_human_smoothed",
        ])

    clades = sorted({
        col[: -len("_identity_to_human")]
        for col in df.columns
        if col.endswith("_identity_to_human")
    })
    for _, row in df.iterrows():
        for clade in clades:
            rows.append({
                "reference_position": row.get("reference_position"),
                "reference_residue": row.get("reference_residue"),
                "clade": clade,
                "n_non_gap": row.get(f"{clade}_n_non_gap"),
                "identity_to_human": row.get(f"{clade}_identity_to_human"),
                "hydrophobicity_agreement_to_human": row.get(f"{clade}_hydrophobicity_agreement_to_human"),
                "major_residue": row.get(f"{clade}_major_residue"),
                "identity_to_human_smoothed": row.get(f"{clade}_identity_to_human_smoothed"),
            })
    return pd.DataFrame(rows)


def normalize_clade_lowpass_profiles(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(columns=[
            "reference_position", "reference_residue", "clade", "identity_fft_lowpass",
        ])

    clades = sorted({
        col[: -len("_identity_fft_lowpass")]
        for col in df.columns
        if col.endswith("_identity_fft_lowpass")
    })
    for _, row in df.iterrows():
        for clade in clades:
            rows.append({
                "reference_position": row.get("reference_position"),
                "reference_residue": row.get("reference_residue"),
                "clade": clade,
                "identity_fft_lowpass": row.get(f"{clade}_identity_fft_lowpass"),
            })
    return pd.DataFrame(rows)


def normalize_clade_difference_profiles(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(columns=[
            "reference_position", "reference_residue", "clade",
            "minus_global_identity", "minus_global_identity_fft_lowpass",
        ])

    clades = sorted({
        col[: -len("_minus_global_identity")]
        for col in df.columns
        if col.endswith("_minus_global_identity")
    })
    for _, row in df.iterrows():
        for clade in clades:
            rows.append({
                "reference_position": row.get("reference_position"),
                "reference_residue": row.get("reference_residue"),
                "clade": clade,
                "minus_global_identity": row.get(f"{clade}_minus_global_identity"),
                "minus_global_identity_fft_lowpass": row.get(f"{clade}_minus_global_identity_fft_lowpass"),
            })
    return pd.DataFrame(rows)


def choose_existing_artifact(outdir: Path, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if (outdir / candidate).exists():
            return candidate
    return None


def evolutionary_divergence_figure_artifacts(outdir: Optional[Path],
                                             scope_key: str) -> Dict[str, Optional[str]]:
    if outdir is None:
        return {"svg_path": None, "png_path": None}
    svg_candidates = [f"evolutionary_segment_bars_{scope_key}.svg"]
    png_candidates = [f"evolutionary_segment_bars_{scope_key}.png"]
    if scope_key == "aligned_reference_projected":
        svg_candidates.append("evolutionary_segment_bars.svg")
        png_candidates.append("evolutionary_segment_bars.png")
    return {
        "svg_path": choose_existing_artifact(outdir, svg_candidates),
        "png_path": choose_existing_artifact(outdir, png_candidates),
    }


def alignment_browser_tick_positions(track_length: int,
                                     step: int = 100) -> List[int]:
    track_length = max(1, int(track_length))
    ticks = [1]
    ticks.extend(value for value in range(step, track_length + 1, step))
    if ticks[-1] != track_length:
        ticks.append(track_length)
    return sorted(dict.fromkeys(ticks))


def alignment_browser_sanitize_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "segment"
    chars: List[str] = []
    prev_sep = False
    for ch in text:
        if ch.isalnum():
            chars.append(ch)
            prev_sep = False
        elif not prev_sep:
            chars.append("_")
            prev_sep = True
    key = "".join(chars).strip("_")
    return key or "segment"


def load_alignment_browser_reference_sequence(outdir: Optional[Path],
                                              reference_species: Optional[str]) -> str:
    if outdir is None or not reference_species:
        return ""
    fasta_candidates = [
        outdir / "aligned_reference_projected.fasta",
        outdir / "aligned.fasta",
    ]
    species_key = str(reference_species).strip().lower()
    for fasta_path in fasta_candidates:
        if not fasta_path.exists():
            continue
        try:
            for record in SeqIO.parse(str(fasta_path), "fasta"):
                record_id = str(getattr(record, "id", "") or "")
                if species_key in record_id.lower():
                    seq = str(getattr(record, "seq", "") or "").upper()
                    return "".join(ch for ch in seq if ch not in GAP_CHARS)
        except Exception:
            continue
    return ""


def load_alignment_browser_reference_landmarks(outdir: Optional[Path],
                                               reference_species: Optional[str],
                                               gene_symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    if outdir is None or not reference_species:
        return []
    feature_df = load_output_table(outdir, "protein_features.tsv", "\t")
    if feature_df.empty:
        feature_df = load_output_table(outdir, "domains.tsv", "\t")
    if feature_df.empty or "species" not in feature_df.columns:
        return []

    ref_df = feature_df[feature_df["species"].astype(str) == str(reference_species).strip()].copy()
    if gene_symbol and "symbol" in ref_df.columns:
        symbol_matches = ref_df[ref_df["symbol"].astype(str) == str(gene_symbol).strip()].copy()
        if not symbol_matches.empty:
            ref_df = symbol_matches
    if ref_df.empty:
        return []

    landmarks: List[Dict[str, Any]] = []
    for description, display_label, color in ALIGNMENT_BROWSER_REFERENCE_LANDMARKS:
        sub = ref_df[ref_df["description"].astype(str) == description]
        if sub.empty:
            continue
        row = sub.iloc[0]
        try:
            start = int(float(row.get("start")))
            end = int(float(row.get("end")))
        except (TypeError, ValueError):
            continue
        if start > end:
            start, end = end, start
        landmarks.append({
            "key": alignment_browser_sanitize_key(display_label),
            "row_key": alignment_browser_sanitize_key(display_label),
            "row_label": display_label,
            "description": description,
            "label": display_label,
            "start": start,
            "end": end,
            "range_label": f"{start}-{end}",
            "color": color,
        })

    motif_specs = ALIGNMENT_BROWSER_REFERENCE_MOTIF_SEARCHES.get(str(gene_symbol or "").strip().upper(), ())
    reference_sequence = load_alignment_browser_reference_sequence(outdir, reference_species)
    for motif_spec in motif_specs:
        if not reference_sequence:
            continue
        found_motif = ""
        start = -1
        for motif in motif_spec.get("motifs", ()):
            motif_text = str(motif or "").strip().upper()
            if not motif_text:
                continue
            start = reference_sequence.find(motif_text)
            if start >= 0:
                found_motif = motif_text
                break
        if start < 0 or not found_motif:
            continue
        start += 1
        end = start + len(found_motif) - 1
        if found_motif in str(motif_spec.get("label") or ""):
            display_label = str(motif_spec.get("label") or "")
        elif "Caspase" in str(motif_spec.get("label") or ""):
            display_label = f'{motif_spec.get("label")} ({found_motif})'
        else:
            display_label = str(motif_spec.get("label") or found_motif)
        landmarks.append({
            "key": alignment_browser_sanitize_key(display_label),
            "row_key": "motif_sites",
            "row_label": "Import motif + caspase site",
            "description": str(motif_spec.get("description") or display_label),
            "label": display_label,
            "start": start,
            "end": end,
            "range_label": f"{start}-{end}",
            "color": str(motif_spec.get("color") or "#475569"),
        })
    position_specs = ALIGNMENT_BROWSER_REFERENCE_POSITION_LANDMARKS.get(str(gene_symbol or "").strip().upper(), ())
    for position_spec in position_specs:
        try:
            position = int(position_spec.get("position"))
        except (TypeError, ValueError):
            continue
        label = str(position_spec.get("label") or f"Residue {position}")
        landmarks.append({
            "key": alignment_browser_sanitize_key(label),
            "row_key": alignment_browser_sanitize_key(position_spec.get("row_label") or "Catalytic residues"),
            "row_label": str(position_spec.get("row_label") or "Catalytic residues"),
            "description": str(position_spec.get("description") or label),
            "label": label,
            "start": position,
            "end": position,
            "range_label": str(position),
            "color": str(position_spec.get("color") or "#0f172a"),
            "always_label": True,
        })
    return landmarks


def alignment_browser_landmark_rows(landmarks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: List[Dict[str, Any]] = []
    row_index: Dict[str, Dict[str, Any]] = {}
    for idx, landmark in enumerate(landmarks):
        row_key = alignment_browser_sanitize_key(
            landmark.get("row_key")
            or landmark.get("key")
            or landmark.get("row_label")
            or landmark.get("label")
            or f"landmark_{idx + 1}"
        )
        row = row_index.get(row_key)
        if row is None:
            row = {
                "key": row_key,
                "label": str(landmark.get("row_label") or landmark.get("label") or "Landmark"),
                "segments": [],
            }
            row_index[row_key] = row
            grouped.append(row)
        row["segments"].append(dict(landmark))
    for row in grouped:
        row["segments"].sort(key=lambda item: (int(item.get("start") or 0), int(item.get("end") or 0)))
        range_labels = [
            str(segment.get("range_label") or f'{int(segment.get("start") or 0)}-{int(segment.get("end") or 0)}')
            for segment in row["segments"]
        ]
        row["range_label"] = "; ".join(range_labels)
    return grouped


def architecture_svg_landmarks(track_x: float,
                               track_width: float,
                               track_length: int,
                               ruler_y: float,
                               domain_rows: Sequence[Dict[str, Any]]) -> List[str]:
    track_length = max(1, int(track_length))
    ticks = alignment_browser_tick_positions(track_length)
    parts = [
        f'<rect class="anno-track" x="{track_x}" y="{ruler_y + 6}" width="{track_width}" height="16" rx="3"/>'
    ]
    for tick in ticks:
        tick_x = track_x + ((tick - 1) / track_length) * track_width
        anchor = "middle"
        if tick == 1:
            anchor = "start"
        elif tick == track_length:
            anchor = "end"
        parts.append(f'<line class="anno-tick" x1="{tick_x:.3f}" y1="{ruler_y + 4}" x2="{tick_x:.3f}" y2="{ruler_y + 26}"/>')
        parts.append(
            f'<text class="anno-tick-label" x="{tick_x:.3f}" y="{ruler_y - 4}" text-anchor="{anchor}">{tick}</text>'
        )
    for row in domain_rows:
        y = float(row["y"])
        parts.append(f'<rect class="anno-track" x="{track_x}" y="{y + 6}" width="{track_width}" height="16" rx="3"/>')
        segments = row.get("segments") or [row]
        for segment in segments:
            start = int(segment["start"])
            end = int(segment["end"])
            span = max(1, end - start + 1)
            width = max(1.5, (span / track_length) * track_width)
            x = track_x + ((start - 1) / track_length) * track_width
            label = f'{segment["label"]} {segment["range_label"]}'
            parts.append(
                f'<rect x="{x:.3f}" y="{y + 6}" width="{width:.3f}" height="16" rx="3" '
                f'fill="{segment["color"]}" opacity="0.95"><title>{escape(label)}</title></rect>'
            )
            if span >= 24 and width >= 85.0:
                text_x = x + min(width / 2.0, track_width - 6)
                parts.append(
                    f'<text class="anno-band-label" x="{text_x:.3f}" y="{y + 18}" text-anchor="middle">{escape(label)}</text>'
                )
            elif segment.get("always_label"):
                text_x = track_x + ((start - 0.5) / track_length) * track_width
                parts.append(
                    f'<text class="anno-site-label" x="{text_x:.3f}" y="{y + 4}" text-anchor="middle" '
                    f'fill="{escape(str(segment["color"]))}">{escape(str(segment["label"]))}</text>'
                )
    return parts


def architecture_secondary_structure_ranges(secondary_structure: Optional[Dict[str, Any]],
                                            track_length: int) -> List[Dict[str, Any]]:
    if not isinstance(secondary_structure, dict):
        return []
    ranges = secondary_structure.get("ranges")
    if not isinstance(ranges, list):
        return []
    cleaned: List[Dict[str, Any]] = []
    max_residue = max(1, int(track_length or 1))
    for row in ranges:
        if not isinstance(row, dict):
            continue
        kind = str(row.get("kind") or "loop").lower()
        if kind not in {"helix", "sheet", "loop"}:
            kind = "loop"
        try:
            start = int(row.get("start"))
            end = int(row.get("end"))
        except (TypeError, ValueError):
            continue
        if start > end:
            start, end = end, start
        start = max(1, min(max_residue, start))
        end = max(1, min(max_residue, end))
        cleaned.append({
            "kind": kind,
            "start": start,
            "end": end,
            "length": max(1, end - start + 1),
            "color": ALPHAFOLD_STRUCTURE_COLORS.get(kind, ALPHAFOLD_STRUCTURE_COLORS["loop"]),
            "display_label": row.get("display_label"),
            "ordinal": row.get("ordinal"),
        })
    return cleaned


def architecture_local_charge_rows(local_charge: Optional[Dict[str, Any]],
                                   track_length: int) -> List[Dict[str, Any]]:
    if not isinstance(local_charge, dict) or not local_charge.get("available"):
        return []
    residues = local_charge.get("residues")
    if not isinstance(residues, list):
        return []
    max_residue = max(1, int(track_length or 1))
    rows: List[Dict[str, Any]] = []
    for row in residues:
        if not isinstance(row, dict):
            continue
        try:
            position = int(row.get("position"))
            charge = float(row.get("charge"))
        except (TypeError, ValueError):
            continue
        if position < 1 or position > max_residue:
            continue
        rows.append({
            "position": position,
            "charge": charge,
            "color": str(row.get("color") or local_charge_color(charge)),
            "window_start": row.get("window_start"),
            "window_end": row.get("window_end"),
            "window_sequence": row.get("window_sequence"),
        })
    return rows


def architecture_calcium_payload(calcium_binding: Optional[Dict[str, Any]],
                                 track_length: int) -> Dict[str, Any]:
    if not isinstance(calcium_binding, dict) or not calcium_binding.get("available"):
        return {"loops": [], "ligands": [], "sites": []}
    max_residue = max(1, int(track_length or 1))
    loops = []
    for row in calcium_binding.get("loops") or []:
        if not isinstance(row, dict):
            continue
        try:
            start = max(1, min(max_residue, int(row.get("start"))))
            end = max(1, min(max_residue, int(row.get("end"))))
        except (TypeError, ValueError):
            continue
        if start > end:
            start, end = end, start
        loops.append({**row, "start": start, "end": end})
    ligands = []
    for row in calcium_binding.get("ligands") or []:
        if not isinstance(row, dict):
            continue
        try:
            position = max(1, min(max_residue, int(row.get("position"))))
        except (TypeError, ValueError):
            continue
        ligands.append({**row, "position": position})
    return {"loops": loops, "ligands": ligands, "sites": calcium_binding.get("sites") or []}


def architecture_svg_helix_element(
    x1: float,
    x2: float,
    y: float,
    color: str = "",
    css_class: str = "",
    title: str = "",
    stroke_width: Optional[float] = None,
) -> str:
    """V11 helix renderer. Branches on css_class:

    * css_class == "snapshot-ss-helix": cursive teardrop loops (cubic Bezier
      per loop with overshooting control points so the curve self-crosses
      near the baseline). Used in the species-snapshot SVG download where
      the helix is the tiny annotation above each row -- the loop shape
      reads better than a zigzag at small sizes.

    * everything else (architecture, comparative AF SS, AlphaFold track,
      stacked-lane views): V9.7-style sharp zigzag polyline. The label/
      color-heavy architecture row reads much more cleanly with this
      compact ±4 px sawtooth than with tall loops that crowd the H1..Hn
      labels."""
    width = max(0.001, x2 - x1)
    is_snapshot = css_class == "snapshot-ss-helix"

    g_attrs: List[str] = ['fill="none"', 'stroke-linecap="round"', 'stroke-linejoin="round"']
    if css_class:
        g_attrs.append(f'class="{css_class}"')
    if color:
        g_attrs.append(f'stroke="{escape(color)}"')
    parts: List[str] = [f"<g {' '.join(g_attrs)}>"]

    if is_snapshot:
        # Bigger loops, thicker line so the curves anti-alias cleanly at the
        # viewer rasterization scale instead of looking pixelated. Sized so
        # the loop top (y - height) lands just above rowTop (no bleed into the
        # row-above residue cells, which end at rowTop-3 with rowHeight=40).
        out_sw = 2.80
        pitch = 14.0
        height = 20.0
        overshoot = 1.85
        num_loops = max(1, int(round(width / pitch)))
        step = width / num_loops
        chunks: List[str] = [f"M {x1:.2f} {y:.2f}"]
        for idx in range(num_loops):
            x_start = x1 + step * idx
            x_end = x1 + step * (idx + 1)
            cp1_x = x_start + overshoot * step
            cp2_x = x_start + (1.0 - overshoot) * step
            cp_y = y - height
            chunks.append(
                f"C {cp1_x:.2f} {cp_y:.2f} {cp2_x:.2f} {cp_y:.2f} {x_end:.2f} {y:.2f}"
            )
        parts.append(f'<path d="{" ".join(chunks)}" stroke-width="{out_sw:.2f}"/>')
    else:
        sw = stroke_width if stroke_width is not None else 4.0
        steps = max(2, int(math.ceil(width / 8.0)))
        points: List[str] = []
        for idx in range(steps + 1):
            xx = x1 + width * idx / steps
            dy = -4.0 if idx % 2 == 0 else 4.0
            points.append(f"{xx:.3f},{y + dy:.3f}")
        parts.append(f'<polyline points="{" ".join(points)}" stroke-width="{sw:.2f}"/>')

    if title:
        parts.append(f"<title>{escape(title)}</title>")
    parts.append("</g>")
    return "".join(parts)


def architecture_svg_secondary_structure(track_x: float,
                                         track_width: float,
                                         track_length: int,
                                         y: float,
                                         secondary_ranges: Sequence[Dict[str, Any]]) -> List[str]:
    parts = [
        f'<rect class="anno-track" x="{track_x}" y="{y + 6}" width="{track_width}" height="16" rx="3"/>',
        f'<line class="ss-loop-line" x1="{track_x}" y1="{y + 14}" x2="{track_x + track_width}" y2="{y + 14}"/>',
    ]
    for row in secondary_ranges:
        kind = str(row.get("kind") or "loop")
        if kind == "loop":
            continue
        start = int(row.get("start") or 1)
        end = int(row.get("end") or start)
        x1 = track_x + ((start - 1) / max(track_length, 1)) * track_width
        x2 = track_x + (end / max(track_length, 1)) * track_width
        width = max(2.0, x2 - x1)
        color = str(row.get("color") or ALPHAFOLD_STRUCTURE_COLORS.get(kind, ALPHAFOLD_STRUCTURE_COLORS["loop"]))
        display_label = str(row.get("display_label") or "")
        label = f"{display_label} {kind} {start}-{end}".strip()
        if kind == "helix":
            parts.append(
                architecture_svg_helix_element(x1, x2, y + 14, color=color, css_class="ss-helix-line", title=label)
            )
        elif kind == "sheet":
            tip = max(0.2, min(14.0, width * 0.45))
            parts.append(
                f'<polygon class="ss-sheet-arrow" points="'
                f'{x1:.3f},{y + 7:.3f} {x2 - tip:.3f},{y + 7:.3f} {x2 - tip:.3f},{y + 2:.3f} '
                f'{x2:.3f},{y + 14:.3f} {x2 - tip:.3f},{y + 26:.3f} {x2 - tip:.3f},{y + 21:.3f} '
                f'{x1:.3f},{y + 21:.3f}" fill="{escape(color)}"><title>{escape(label)}</title></polygon>'
            )
        if display_label:
            parts.append(
                f'<text class="ss-feature-label" x="{(x1 + x2) / 2.0:.3f}" y="{y + 5:.3f}" '
                f'text-anchor="middle" fill="{escape(color)}"><title>{escape(label)}</title>{escape(display_label)}</text>'
            )
    return parts


def architecture_svg_local_charge(track_x: float,
                                  track_width: float,
                                  track_length: int,
                                  y: float,
                                  charge_rows: Sequence[Dict[str, Any]]) -> List[str]:
    parts = [
        f'<rect class="anno-track" x="{track_x}" y="{y + 6}" width="{track_width}" height="16" rx="3"/>',
    ]
    width = track_width / max(track_length, 1)
    for row in charge_rows:
        position = int(row.get("position") or 1)
        x = track_x + ((position - 1) / max(track_length, 1)) * track_width
        charge = float(row.get("charge") or 0.0)
        title = f'pos {position} charge {charge:+.1f} window {row.get("window_start")}-{row.get("window_end")} {row.get("window_sequence") or ""}'
        parts.append(
            f'<rect x="{x:.3f}" y="{y + 6}" width="{max(0.6, width):.3f}" height="16" '
            f'fill="{escape(str(row.get("color") or local_charge_color(charge)))}"><title>{escape(title)}</title></rect>'
        )
    parts.append(f'<line x1="{track_x}" y1="{y + 14}" x2="{track_x + track_width}" y2="{y + 14}" stroke="#475569" stroke-width="0.7" opacity="0.35"/>')
    return parts


def architecture_svg_calcium(track_x: float,
                             track_width: float,
                             track_length: int,
                             y: float,
                             calcium_payload: Dict[str, Any]) -> List[str]:
    parts = [
        f'<rect class="anno-track" x="{track_x}" y="{y + 6}" width="{track_width}" height="16" rx="3"/>',
    ]
    for loop in calcium_payload.get("loops") or []:
        start = int(loop.get("start") or 1)
        end = int(loop.get("end") or start)
        x = track_x + ((start - 1) / max(track_length, 1)) * track_width
        w = ((end - start + 1) / max(track_length, 1)) * track_width
        label = f'{loop.get("label") or "CBR"} {start}-{end}'
        description = str(loop.get("description") or "")
        parts.append(
            f'<rect x="{x:.3f}" y="{y + 6}" width="{max(1.5, w):.3f}" height="16" rx="4" '
            f'fill="{escape(str(loop.get("color") or ALPHAFOLD_STRUCTURE_COLORS["calcium"]))}" opacity="0.88">'
            f'<title>{escape(label + (" | " + description if description else ""))}</title></rect>'
        )
        if w >= 24:
            parts.append(f'<text class="anno-band-label" x="{x + w / 2.0:.3f}" y="{y + 18}" text-anchor="middle">{escape(str(loop.get("label") or ""))}</text>')
    for ligand in calcium_payload.get("ligands") or []:
        position = int(ligand.get("position") or 1)
        cx = track_x + ((position - 0.5) / max(track_length, 1)) * track_width
        label = str(ligand.get("label") or f"residue {position}")
        sites = ", ".join(str(site) for site in ligand.get("sites") or [])
        color = str(ligand.get("color") or ALPHAFOLD_STRUCTURE_COLORS["calcium_ligand"])
        parts.append(
            f'<circle cx="{cx:.3f}" cy="{y + 14}" r="3.0" fill="{escape(color)}">'
            f'<title>{escape(label + (" | " + sites if sites else ""))}</title></circle>'
        )
    return parts


def plot_alignment_browser_architecture_png(out_png: Path,
                                            title: str,
                                            subtitle: str,
                                            groups: Sequence[Dict[str, Any]],
                                            track_length: int,
                                            compare_mode: str,
                                            landmarks: Sequence[Dict[str, Any]],
                                            secondary_structure: Optional[Dict[str, Any]] = None,
                                            local_charge: Optional[Dict[str, Any]] = None,
                                            calcium_binding: Optional[Dict[str, Any]] = None) -> None:
    active_groups = [group for group in groups if group.get("label")]
    landmark_rows = alignment_browser_landmark_rows(landmarks)
    secondary_ranges = architecture_secondary_structure_ranges(secondary_structure, track_length)
    charge_rows = architecture_local_charge_rows(local_charge, track_length)
    calcium_payload = architecture_calcium_payload(calcium_binding, track_length)
    has_calcium = bool(calcium_payload.get("loops") or calcium_payload.get("ligands"))
    annotation_rows = 1 + (1 if secondary_ranges else 0) + (1 if charge_rows else 0) + (1 if has_calcium else 0) + len(landmark_rows)
    total_rows = annotation_rows + max(len(active_groups), 1)
    fig_height = max(7.5, 0.34 * total_rows + 1.9)
    fig, ax = plt.subplots(figsize=(18, fig_height))
    x_max = max(1, int(track_length)) + 80
    ax.set_xlim(1, x_max)
    ax.set_ylim(-1.0, total_rows + 0.9)
    ax.axis("off")
    run_color = "#7c3aed" if compare_mode == "property" else "#2563eb"
    track_color = "#e5e7eb"
    track_edge = "#d0d7e2"
    count_x = max(1, int(track_length)) + 22
    tick_positions = alignment_browser_tick_positions(track_length)

    ax.text(1, total_rows + 0.45, title, fontsize=15, fontweight="bold", ha="left", va="bottom", color="#18202a")
    ax.text(1, total_rows + 0.1, subtitle, fontsize=10, fontweight="semibold", ha="left", va="bottom", color="#617083")

    row_cursor = total_rows - 1
    ax.text(-55, row_cursor, "Reference ruler", fontsize=9.5, fontweight="bold", ha="left", va="center", color="#18202a")
    ax.add_patch(Rectangle((1, row_cursor - 0.19), track_length, 0.38, facecolor=track_color, edgecolor=track_edge, linewidth=0.8))
    for tick in tick_positions:
        ax.plot([tick, tick], [row_cursor - 0.26, row_cursor + 0.26], color="#64748b", linewidth=0.8)
        ax.text(tick, row_cursor + 0.42, str(tick), fontsize=8, ha="center", va="bottom", color="#475569")
    ax.text(count_x, row_cursor, f"1-{track_length}", fontsize=9, fontweight="bold", ha="left", va="center", color="#617083")
    row_cursor -= 1

    if secondary_ranges:
        ax.text(-55, row_cursor, "AlphaFold SS", fontsize=9.5, fontweight="bold", ha="left", va="center", color="#18202a")
        ax.add_patch(Rectangle((1, row_cursor - 0.19), track_length, 0.38, facecolor="#eef2f7", edgecolor=track_edge, linewidth=0.8))
        ax.plot([1, track_length + 1], [row_cursor, row_cursor], color=ALPHAFOLD_STRUCTURE_COLORS["loop"], linewidth=2.8, alpha=0.75)
        for row in secondary_ranges:
            kind = str(row.get("kind") or "loop")
            if kind == "loop":
                continue
            start = int(row.get("start") or 1)
            end = int(row.get("end") or start)
            end_edge = end + 1
            color = str(row.get("color") or ALPHAFOLD_STRUCTURE_COLORS.get(kind, ALPHAFOLD_STRUCTURE_COLORS["loop"]))
            span = max(1, end - start + 1)
            if kind == "helix":
                points = max(5, min(80, span * 2))
                xs = np.linspace(start, end_edge, points)
                ys = row_cursor + 0.09 * np.where(np.arange(points) % 2 == 0, -1.0, 1.0)
                ax.plot(xs, ys, color=color, linewidth=2.2, solid_capstyle="round")
            elif kind == "sheet":
                tip = max(0.1, min(10.0, span * 0.45))
                shoulder = max(start, end_edge - tip)
                polygon = np.array([
                    [start, row_cursor - 0.16],
                    [shoulder, row_cursor - 0.16],
                    [shoulder, row_cursor - 0.27],
                    [end_edge, row_cursor],
                    [shoulder, row_cursor + 0.27],
                    [shoulder, row_cursor + 0.16],
                    [start, row_cursor + 0.16],
                ])
                ax.add_patch(Polygon(polygon, closed=True, facecolor=color, edgecolor="none", alpha=0.92))
            display_label = str(row.get("display_label") or "")
            if display_label:
                ax.text((start + end_edge) / 2.0, row_cursor + 0.28, display_label, fontsize=5.8,
                        fontweight="bold", ha="center", va="bottom", color=color)
        ax.text(count_x, row_cursor, "helix/sheet/loop", fontsize=9, fontweight="bold", ha="left", va="center", color="#617083")
        row_cursor -= 1

    if charge_rows:
        ax.text(-55, row_cursor, "5-aa charge", fontsize=9.5, fontweight="bold", ha="left", va="center", color="#18202a")
        ax.add_patch(Rectangle((1, row_cursor - 0.19), track_length, 0.38, facecolor="#eef2f7", edgecolor=track_edge, linewidth=0.8))
        for row in charge_rows:
            position = int(row.get("position") or 1)
            ax.add_patch(Rectangle((position, row_cursor - 0.19), 1.0, 0.38,
                                   facecolor=str(row.get("color") or local_charge_color(float(row.get("charge") or 0.0))),
                                   edgecolor="none", linewidth=0.0))
        ax.plot([1, track_length + 1], [row_cursor, row_cursor], color="#475569", linewidth=0.5, alpha=0.35)
        ax.text(count_x, row_cursor, "negative / neutral / positive", fontsize=9, fontweight="bold", ha="left", va="center", color="#617083")
        row_cursor -= 1

    if has_calcium:
        ax.text(-55, row_cursor, "Ca2+ binding", fontsize=9.5, fontweight="bold", ha="left", va="center", color="#18202a")
        ax.add_patch(Rectangle((1, row_cursor - 0.19), track_length, 0.38, facecolor="#eef2f7", edgecolor=track_edge, linewidth=0.8))
        for loop in calcium_payload.get("loops") or []:
            start = int(loop.get("start") or 1)
            end = int(loop.get("end") or start)
            span = max(1, end - start + 1)
            color = str(loop.get("color") or ALPHAFOLD_STRUCTURE_COLORS["calcium"])
            ax.add_patch(Rectangle((start, row_cursor - 0.19), span, 0.38, facecolor=color, edgecolor="none", alpha=0.88))
            ax.text(start + span / 2.0, row_cursor, str(loop.get("label") or ""), fontsize=7.2, fontweight="bold",
                    ha="center", va="center", color="white")
        for ligand in calcium_payload.get("ligands") or []:
            position = int(ligand.get("position") or 1)
            ax.plot([position + 0.5], [row_cursor], marker="o", markersize=3.2,
                    color=str(ligand.get("color") or ALPHAFOLD_STRUCTURE_COLORS["calcium_ligand"]))
        ax.text(count_x, row_cursor, "CBR1-3 + Ca/PC contacts", fontsize=9, fontweight="bold", ha="left", va="center", color="#617083")
        row_cursor -= 1

    for landmark_row in landmark_rows:
        label = str(landmark_row["label"])
        range_label = str(landmark_row["range_label"])
        ax.text(-55, row_cursor, label, fontsize=9.5, fontweight="bold", ha="left", va="center", color="#18202a")
        ax.add_patch(Rectangle((1, row_cursor - 0.19), track_length, 0.38, facecolor=track_color, edgecolor=track_edge, linewidth=0.8))
        for segment in landmark_row.get("segments") or []:
            start = int(segment["start"])
            end = int(segment["end"])
            span = max(1, end - start + 1)
            ax.add_patch(Rectangle((start, row_cursor - 0.19), span, 0.38, facecolor=str(segment["color"]), edgecolor="none", linewidth=0.0))
            if span >= 24:
                text_x = start + span / 2.0
                ax.text(text_x, row_cursor, f'{segment["label"]} {segment["range_label"]}', fontsize=8.2, fontweight="bold", ha="center", va="center", color="white")
            elif bool(segment.get("always_label")):
                ax.text(start + 0.5, row_cursor + 0.28, str(segment.get("label") or ""),
                        fontsize=6.2, fontweight="bold", ha="center", va="bottom",
                        color=str(segment.get("color") or "#0f172a"))
        ax.text(count_x, row_cursor, range_label, fontsize=9, fontweight="bold", ha="left", va="center", color="#617083")
        row_cursor -= 1

    for group in active_groups:
        runs = group.get("runs") or []
        count_label = f"{len(runs)} run" + ("" if len(runs) == 1 else "s")
        ax.text(-55, row_cursor, str(group.get("label") or ""), fontsize=9.5, fontweight="bold", ha="left", va="center", color="#18202a")
        ax.add_patch(Rectangle((1, row_cursor - 0.19), track_length, 0.38, facecolor=track_color, edgecolor=track_edge, linewidth=0.8))
        for run in runs:
            start = run.get("startRef")
            end = run.get("endRef")
            if start is None or end is None:
                start = int(run.get("startIdx", 0)) + 1
                end = int(run.get("endIdx", 0)) + 1
            start_num = int(start)
            end_num = int(end)
            ax.add_patch(Rectangle((start_num, row_cursor - 0.19), max(1, end_num - start_num + 1), 0.38, facecolor=run_color, edgecolor="none", linewidth=0.0))
        ax.text(count_x, row_cursor, count_label, fontsize=9, fontweight="bold", ha="left", va="center", color="#617083")
        row_cursor -= 1

    fig.tight_layout(pad=1.0)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_run_metadata(outdir: Path,
                       run_key: str,
                       db_path: Path,
                       report_filename: str,
                       tables: Dict[str, pd.DataFrame],
                       artifacts_df: pd.DataFrame,
                       pairwise_df: pd.DataFrame,
                       reference_species: Optional[str]) -> Dict[str, Any]:
    summary_meta = parse_run_summary_metadata(outdir / "run_summary.txt")
    seq_df = tables.get("sequence_retrieval", pd.DataFrame())
    recovered_count = None
    if "recovered_sequence_count" in summary_meta:
        recovered_count = summary_meta["recovered_sequence_count"]
    elif not seq_df.empty and "status" in seq_df.columns:
        recovered_count = int((seq_df["status"] == "ok").sum())

    candidate_count = summary_meta.get("candidate_sequence_count")
    if candidate_count is None and not seq_df.empty:
        candidate_count = len(seq_df)

    return {
        "run_key": run_key,
        "output_directory": str(outdir.resolve()),
        "output_name": outdir.name,
        "gene_symbol": summary_meta.get("Gene"),
        "source_species": summary_meta.get("Source species"),
        "reference_species": reference_species,
        "recovered_sequence_count": recovered_count,
        "candidate_sequence_count": candidate_count,
        "alignment_method": summary_meta.get("Alignment method"),
        "tree_method": summary_meta.get("Tree method"),
        "tree_built": summary_meta.get("Phylogeny built"),
        "full_alignment_length": summary_meta.get("Full alignment length"),
        "reference_projected_length": summary_meta.get("Reference-projected alignment length"),
        "selection_mode": summary_meta.get("Selection mode"),
        "annotated_sites": summary_meta.get("Annotated sites"),
        "annotated_site_window": summary_meta.get("Annotated-site comparison window residues"),
        "annotated_site_clade_mya": summary_meta.get("Annotated-site clade MyA overrides"),
        "ortholog_row_count": len(tables.get("orthologs", pd.DataFrame())),
        "pairwise_report_count": len(pairwise_df),
        "artifact_count": len(artifacts_df),
        "interactive_report_path": report_filename,
        "alignment_browser_path": ALIGNMENT_BROWSER_FILENAME if (outdir / ALIGNMENT_BROWSER_FILENAME).exists() else None,
        "sqlite_archive_path": str(db_path.resolve()),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def clean_json_value(value: Any, float_digits: int = 6) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [clean_json_value(item, float_digits=float_digits) for item in value]
    if isinstance(value, dict):
        return {str(k): clean_json_value(v, float_digits=float_digits) for k, v in value.items()}
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, float_digits)
    if isinstance(value, (np.floating,)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, float_digits)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return None
    return value


def dataframe_to_json_records(df: pd.DataFrame, float_digits: int = 6) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    records: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append({
            str(key): clean_json_value(value, float_digits=float_digits)
            for key, value in row.items()
        })
    return records


def filter_columns(df: pd.DataFrame, preferred_columns: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    keep = [column for column in preferred_columns if column in df.columns]
    return df[keep].copy() if keep else df.copy()


def read_json_artifact(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_text_artifact(path: Path, max_chars: Optional[int] = None) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if max_chars is not None and len(text) > max_chars:
        return ""
    return text


def comparative_alphafold_unavailable_payload(reason: str) -> Dict[str, Any]:
    return {
        "available": False,
        "filename": COMPARATIVE_ALPHAFOLD_SS_FILENAME,
        "reason": reason,
        "record_label_singular": "SS map",
        "record_label_plural": "SS maps",
        "records": [],
    }


def metadata_lookup_for_comparative_alphafold(outdir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    metadata_df = load_output_table(outdir, "protein_metadata.tsv", "\t")
    by_species: Dict[str, Dict[str, Any]] = {}
    by_protein: Dict[str, Dict[str, Any]] = {}
    if metadata_df.empty:
        return by_species, by_protein
    for row in dataframe_to_json_records(metadata_df):
        species = str(row.get("species") or "").strip()
        if species and species not in by_species:
            by_species[species] = row
        for key in (
            row.get("protein_record_id"),
            row.get("translation_id"),
            row.get("canonical_translation"),
        ):
            key_text = str(key or "").strip()
            if key_text and key_text not in by_protein:
                by_protein[key_text] = row
    return by_species, by_protein


def comparative_unique_keys(*values: Any) -> List[str]:
    keys: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        keys.append(text)
    return keys


def comparative_alphafold_override(species: str,
                                   record_id: str,
                                   protein_record_id: Any,
                                   protein_key: Any,
                                   protein_meta: Dict[str, Any]) -> Dict[str, Any]:
    species_key = str(species or "").strip().lower()
    candidates = comparative_unique_keys(
        protein_record_id,
        protein_key,
        parse_header_field(record_id, "ProteinRecordID"),
        parse_header_field(record_id, "Protein"),
        parse_header_field(record_id, "Gene"),
        parse_header_field(record_id, "EnsemblGene"),
        protein_meta.get("protein_record_id"),
        protein_meta.get("translation_id"),
        protein_meta.get("canonical_translation"),
        protein_meta.get("symbol"),
        protein_meta.get("ensembl_gene_id"),
    )
    for key in candidates:
        override = COMPARATIVE_ALPHAFOLD_ACCESSION_OVERRIDES.get((species_key, key))
        if override:
            return dict(override)
    return {}


def comparative_cached_alphafold_model_path(outdir: Path,
                                            accession: Any,
                                            entry_id: Any = None,
                                            filename: Any = None) -> Optional[Path]:
    model_dir = outdir / COMPARATIVE_ALPHAFOLD_MODEL_DIRNAME
    if filename:
        direct = model_dir / str(filename)
        if direct.exists():
            return direct
    accession_text = str(accession or "").strip()
    if accession_text:
        candidates = sorted(model_dir.glob(f"AF-{accession_text}-F1-model_v*.pdb"))
        if candidates:
            return candidates[-1]
    entry_text = str(entry_id or "").strip()
    if entry_text:
        candidates = sorted(model_dir.glob(f"{entry_text}-model_v*.pdb"))
        if candidates:
            return candidates[-1]
    return None


def build_comparative_full_alignment_reference_maps(outdir: Path,
                                                     reference_species: Optional[str] = None) -> Dict[str, Any]:
    fasta_path = outdir / "aligned.fasta"
    if not fasta_path.exists():
        return {"maps": {}, "reference_record_id": None, "reference_track_length": 0}
    try:
        records = list(SeqIO.parse(str(fasta_path), "fasta"))
    except Exception:
        return {"maps": {}, "reference_record_id": None, "reference_track_length": 0}
    if not records:
        return {"maps": {}, "reference_record_id": None, "reference_track_length": 0}

    target_species = str(reference_species or "").strip().lower()
    reference_record = records[0]
    for record in records:
        species = str(record.id).split("|", 1)[0].strip().lower()
        if (target_species and species == target_species) or (not target_species and species == "homo_sapiens"):
            reference_record = record
            break

    reference_positions: List[Optional[int]] = []
    reference_index = 0
    for aa in str(reference_record.seq).upper():
        if aa in GAP_CHARS:
            reference_positions.append(None)
        else:
            reference_index += 1
            reference_positions.append(reference_index)

    maps: Dict[str, Dict[str, Any]] = {}
    for record in records:
        record_id = str(record.id)
        species, _symbol = parse_header_species_symbol(record_id)
        protein_key = (
            parse_header_field(record_id, "ProteinRecordID")
            or parse_header_field(record_id, "Protein")
            or ""
        )
        species_position = 0
        mapped_count = 0
        reference_gap_count = 0
        position_to_reference: Dict[int, int] = {}
        for index, aa in enumerate(str(record.seq).upper()):
            if aa in GAP_CHARS:
                continue
            species_position += 1
            reference_position = reference_positions[index] if index < len(reference_positions) else None
            if reference_position is None:
                reference_gap_count += 1
                continue
            position_to_reference[species_position] = int(reference_position)
            mapped_count += 1
        species_ungapped_sequence = "".join(
            aa for aa in str(record.seq).upper() if aa not in GAP_CHARS
        )
        row = {
            "alignment_record_id": record_id,
            "species_sequence_length": species_position,
            # V12: full ungapped species sequence (positions align 1:1 with the
            # keys of position_to_reference) so a substituted AlphaFold model can
            # be pairwise-aligned to it before its SS ranges are mapped.
            "species_ungapped_sequence": species_ungapped_sequence,
            "reference_mapped_residue_count": mapped_count,
            "reference_gap_residue_count": reference_gap_count,
            "position_to_reference": position_to_reference,
        }
        for key in comparative_unique_keys(
            record_id,
            protein_key,
            f"{species}__{protein_key}" if protein_key else "",
            parse_header_field(record_id, "Gene"),
            parse_header_field(record_id, "EnsemblGene"),
        ):
            maps[key] = row
    return {
        "maps": maps,
        "reference_record_id": str(reference_record.id),
        "reference_track_length": reference_index,
    }


def close_comparative_alphafold_range(ranges: List[Dict[str, Any]],
                                      counters: Dict[str, int],
                                      kind: Optional[str],
                                      start_reference: Optional[int],
                                      end_reference: Optional[int],
                                      start_species: Optional[int],
                                      end_species: Optional[int],
                                      source: str = "reference_alphafold_secondary_structure_projection") -> None:
    if kind is None or start_reference is None or end_reference is None:
        return
    row = {
        "kind": kind,
        "secondary_structure": kind,
        "start": start_reference,
        "end": end_reference,
        "start_reference_position": start_reference,
        "end_reference_position": end_reference,
        "start_species_position": start_species,
        "end_species_position": end_species,
        "length": end_reference - start_reference + 1,
        "color": ALPHAFOLD_STRUCTURE_COLORS.get(kind, ALPHAFOLD_STRUCTURE_COLORS["loop"]),
        "source": source,
    }
    if kind in counters:
        counters[kind] += 1
        row["ordinal"] = counters[kind]
        row["display_label"] = f"{'H' if kind == 'helix' else 'S'}{counters[kind]}"
    ranges.append(row)


def project_secondary_structure_ranges_to_alignment(sequence: str,
                                                    reference_positions: Sequence[Optional[int]],
                                                    ss_by_reference_position: Dict[int, str]) -> Tuple[List[Dict[str, Any]], int, int]:
    ranges: List[Dict[str, Any]] = []
    counters = {"helix": 0, "sheet": 0}
    current_kind: Optional[str] = None
    start_reference: Optional[int] = None
    end_reference: Optional[int] = None
    start_species: Optional[int] = None
    end_species: Optional[int] = None
    species_position = 0
    mapped_residue_count = 0
    gap_reference_count = 0

    for index, reference_position in enumerate(reference_positions):
        aa = sequence[index].upper() if index < len(sequence) else "-"
        is_gap = aa in GAP_CHARS
        if not is_gap:
            species_position += 1
        if reference_position is None:
            continue
        if is_gap:
            gap_reference_count += 1
            close_comparative_alphafold_range(
                ranges, counters, current_kind, start_reference, end_reference, start_species, end_species
            )
            current_kind = None
            start_reference = None
            end_reference = None
            start_species = None
            end_species = None
            continue

        kind = ss_by_reference_position.get(int(reference_position), "loop")
        if current_kind == kind and end_reference is not None and int(reference_position) == end_reference + 1:
            end_reference = int(reference_position)
            end_species = species_position
        else:
            close_comparative_alphafold_range(
                ranges, counters, current_kind, start_reference, end_reference, start_species, end_species
            )
            current_kind = kind
            start_reference = int(reference_position)
            end_reference = int(reference_position)
            start_species = species_position
            end_species = species_position
        mapped_residue_count += 1

    close_comparative_alphafold_range(
        ranges, counters, current_kind, start_reference, end_reference, start_species, end_species
    )
    return ranges, mapped_residue_count, gap_reference_count


def map_species_secondary_structure_to_reference(ss_residues: Sequence[Dict[str, Any]],
                                                 position_to_reference: Dict[int, int]) -> Tuple[List[Dict[str, Any]], int, int, int]:
    ranges: List[Dict[str, Any]] = []
    counters = {"helix": 0, "sheet": 0}
    current_kind: Optional[str] = None
    start_reference: Optional[int] = None
    end_reference: Optional[int] = None
    start_species: Optional[int] = None
    end_species: Optional[int] = None
    mapped_residue_count = 0
    unmapped_residue_count = 0
    model_residue_count = 0

    for row in ss_residues:
        try:
            species_position = int(row.get("position"))
        except (TypeError, ValueError, AttributeError):
            continue
        model_residue_count += 1
        reference_position = position_to_reference.get(species_position)
        if reference_position is None:
            unmapped_residue_count += 1
            close_comparative_alphafold_range(
                ranges,
                counters,
                current_kind,
                start_reference,
                end_reference,
                start_species,
                end_species,
                source="species_alphafold_secondary_structure_reference_mapping",
            )
            current_kind = None
            start_reference = None
            end_reference = None
            start_species = None
            end_species = None
            continue
        kind = str(row.get("secondary_structure") or "loop").lower()
        if kind not in {"helix", "sheet", "loop"}:
            kind = "loop"
        if current_kind == kind and end_reference is not None and int(reference_position) == end_reference + 1:
            end_reference = int(reference_position)
            end_species = species_position
        else:
            close_comparative_alphafold_range(
                ranges,
                counters,
                current_kind,
                start_reference,
                end_reference,
                start_species,
                end_species,
                source="species_alphafold_secondary_structure_reference_mapping",
            )
            current_kind = kind
            start_reference = int(reference_position)
            end_reference = int(reference_position)
            start_species = species_position
            end_species = species_position
        mapped_residue_count += 1

    close_comparative_alphafold_range(
        ranges,
        counters,
        current_kind,
        start_reference,
        end_reference,
        start_species,
        end_species,
        source="species_alphafold_secondary_structure_reference_mapping",
    )
    return ranges, mapped_residue_count, unmapped_residue_count, model_residue_count


_THREE_TO_ONE_AA = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


def model_residue_one_letter_sequence(model_residues: Sequence[Dict[str, Any]]) -> Tuple[List[int], str]:
    """(V12) Return (pdb_positions, one_letter_sequence) for an AlphaFold model's
    residues, ordered by PDB residue number. Used to align a substituted model
    against the aligned species sequence so its secondary-structure ranges map by
    residue identity rather than by raw residue index."""
    rows: List[Tuple[int, str]] = []
    for entry in model_residues or []:
        try:
            position = int(entry.get("position"))
        except (TypeError, ValueError, AttributeError):
            continue
        name = str(entry.get("residue_name") or "").strip().upper()
        rows.append((position, _THREE_TO_ONE_AA.get(name, "X")))
    rows.sort(key=lambda item: item[0])
    return [pos for pos, _ in rows], "".join(aa for _, aa in rows)


def build_model_position_to_reference(model_residues: Sequence[Dict[str, Any]],
                                      species_ungapped_sequence: str,
                                      position_to_reference: Dict[int, int],
                                      min_identity: float = 0.70) -> Tuple[Dict[int, int], float]:
    """(V12) Map an AlphaFold model's PDB residue positions -> reference positions
    by pairwise-aligning the model sequence to the full ungapped species sequence
    and composing with the species-position -> reference-position map. This keeps
    the comparative secondary-structure bar correct when a substituted model
    differs in length/numbering from the Ensembl species sequence. Returns
    ({}, identity) when the two sequences agree below min_identity so the caller
    can fall back to the original positional mapping. When the model sequence
    equals the species sequence (the common case) the alignment is gapless 1:1,
    so the result is identical to the positional mapping -- no behaviour change."""
    positions, model_seq = model_residue_one_letter_sequence(model_residues)
    species_seq = "".join(
        aa for aa in str(species_ungapped_sequence or "").upper() if aa not in GAP_CHARS
    )
    if not model_seq or not species_seq:
        return {}, 0.0
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = BLOSUM62
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    try:
        alignment = aligner.align(model_seq, species_seq)[0]
    except (ValueError, IndexError):
        return {}, 0.0
    mapping: Dict[int, int] = {}
    identical = 0
    model_blocks, species_blocks = alignment.aligned
    for (m0, m1), (s0, s1) in zip(model_blocks, species_blocks):
        for offset in range(int(m1) - int(m0)):
            mi = int(m0) + offset
            si = int(s0) + offset
            if mi >= len(positions) or si >= len(species_seq):
                continue
            if model_seq[mi] == species_seq[si]:
                identical += 1
            reference_position = position_to_reference.get(si + 1)
            if reference_position is not None:
                mapping[positions[mi]] = int(reference_position)
    identity = identical / max(1, min(len(model_seq), len(species_seq)))
    if identity < min_identity:
        return {}, identity
    return mapping, identity


def build_comparative_alphafold_secondary_structure_payload(outdir: Path,
                                                           reference_species: Optional[str] = None) -> Dict[str, Any]:
    fasta_path = outdir / "aligned_reference_projected.fasta"
    if not fasta_path.exists():
        return comparative_alphafold_unavailable_payload("aligned_reference_projected.fasta was not found.")

    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        return comparative_alphafold_unavailable_payload("No aligned reference-projected sequences were found.")

    metadata = read_json_artifact(outdir / ALPHAFOLD_METADATA_FILENAME)
    model_filename = str(metadata.get("model_filename") or ALPHAFOLD_MODEL_FILENAME)
    pdb_text = read_text_artifact(outdir / model_filename)
    if not pdb_text:
        return comparative_alphafold_unavailable_payload(f"AlphaFold model file was not found: {model_filename}")

    secondary_structure = assign_pdb_secondary_structure(pdb_text)
    ss_residues = secondary_structure.get("residues") if isinstance(secondary_structure, dict) else []
    if not secondary_structure.get("available") or not isinstance(ss_residues, list) or not ss_residues:
        return comparative_alphafold_unavailable_payload(
            secondary_structure.get("reason") or "No reference AlphaFold secondary-structure residues were available."
        )

    ss_by_reference_position: Dict[int, str] = {}
    for row in ss_residues:
        try:
            position = int(row.get("position"))
        except (TypeError, ValueError, AttributeError):
            continue
        kind = str(row.get("secondary_structure") or "loop").lower()
        if kind not in {"helix", "sheet", "loop"}:
            kind = "loop"
        ss_by_reference_position[position] = kind

    target_species = str(reference_species or "").strip().lower()
    reference_record = records[0]
    for record in records:
        species = str(record.id).split("|", 1)[0].strip().lower()
        if (target_species and species == target_species) or (not target_species and species == "homo_sapiens"):
            reference_record = record
            break

    reference_positions: List[Optional[int]] = []
    reference_index = 0
    for aa in str(reference_record.seq).upper():
        if aa in GAP_CHARS:
            reference_positions.append(None)
        else:
            reference_index += 1
            reference_positions.append(reference_index)

    metadata_by_species, metadata_by_protein = metadata_lookup_for_comparative_alphafold(outdir)
    full_alignment_maps_payload = build_comparative_full_alignment_reference_maps(outdir, reference_species=reference_species)
    full_alignment_maps = full_alignment_maps_payload.get("maps") or {}
    payload_records: List[Dict[str, Any]] = []
    for record_index, record in enumerate(records, start=1):
        record_id = str(record.id)
        species, symbol = parse_header_species_symbol(record_id)
        protein_key = (
            parse_header_field(record_id, "ProteinRecordID")
            or parse_header_field(record_id, "Protein")
            or ""
        )
        protein_meta = (
            metadata_by_protein.get(str(protein_key).strip())
            or metadata_by_protein.get(f"{species}__{protein_key}".strip("_"))
            or metadata_by_species.get(species)
            or {}
        )
        protein_record_id = clean_json_value(protein_meta.get("protein_record_id")) or clean_json_value(protein_key)
        override = comparative_alphafold_override(species, record_id, protein_record_id, protein_key, protein_meta)
        uniprot_accession = clean_json_value(
            override.get("uniprot_accession")
            or protein_meta.get("uniprot_accession")
            or protein_meta.get("alphafold_accession")
        )
        alphafold_entry_id = clean_json_value(override.get("alphafold_entry_id") or protein_meta.get("alphafold_entry_id"))
        alphafold_source_label = clean_json_value(
            override.get("alphafold_source_label") or protein_meta.get("alphafold_source_label")
        )
        model_path = comparative_cached_alphafold_model_path(
            outdir,
            uniprot_accession,
            alphafold_entry_id,
            override.get("model_filename"),
        )
        alignment_map = None
        for map_key in comparative_unique_keys(
            record_id,
            protein_record_id,
            protein_key,
            f"{species}__{protein_key}" if protein_key else "",
            parse_header_field(record_id, "Gene"),
            parse_header_field(record_id, "EnsemblGene"),
        ):
            candidate_map = full_alignment_maps.get(map_key)
            if candidate_map:
                alignment_map = candidate_map
                break

        record_sequence_length = sum(1 for aa in str(record.seq).upper() if aa not in GAP_CHARS)
        ranges: List[Dict[str, Any]] = []
        mapped_residue_count = 0
        gap_reference_count = 0
        unmapped_species_residue_count: Optional[int] = None
        model_residue_count: Optional[int] = None
        model_relative_filename: Optional[str] = None
        model_secondary_structure_method: Optional[str] = None
        source_method = "reference_alphafold_secondary_structure_projected_through_alignment"
        source_note = (
            "Fallback projection from the human reference AlphaFold secondary-structure assignment; "
            "no cached species-specific AlphaFold model was available for this entry."
        )
        projection_warning = True

        if model_path is not None and alignment_map:
            model_text = read_text_artifact(model_path)
            model_secondary_structure = assign_pdb_secondary_structure(model_text) if model_text else {}
            model_residues = model_secondary_structure.get("residues") if isinstance(model_secondary_structure, dict) else []
            position_to_reference = alignment_map.get("position_to_reference") or {}
            if model_secondary_structure.get("available") and isinstance(model_residues, list) and isinstance(position_to_reference, dict):
                # V12: map the model's SS by residue identity (pairwise-align the
                # model sequence to the full species sequence) rather than by raw
                # residue index, so substituted / length-mismatched models still
                # land on the correct reference positions. Falls back to the
                # positional map when sequences match (gapless 1:1) or identity is
                # too low to trust the alignment.
                remapped_position_to_reference, _model_species_identity = build_model_position_to_reference(
                    model_residues,
                    alignment_map.get("species_ungapped_sequence") or "",
                    position_to_reference,
                )
                effective_position_to_reference = remapped_position_to_reference or position_to_reference
                ranges, mapped_residue_count, unmapped_count, model_count = map_species_secondary_structure_to_reference(
                    model_residues,
                    effective_position_to_reference,
                )
                if ranges:
                    unmapped_species_residue_count = unmapped_count
                    model_residue_count = model_count or clean_json_value(override.get("sequence_length"))
                    try:
                        model_relative_filename = str(model_path.relative_to(outdir)).replace("\\", "/")
                    except ValueError:
                        model_relative_filename = str(model_path).replace("\\", "/")
                    model_secondary_structure_method = clean_json_value(model_secondary_structure.get("method"))
                    source_method = "species_alphafold_secondary_structure_mapped_to_reference_alignment"
                    source_note = (
                        "Species-specific AlphaFold secondary structure mapped through aligned.fasta to the "
                        "human reference ruler; species residues aligned to reference gaps are omitted from the bar."
                    )
                    projection_warning = False
                    gap_reference_count = int(alignment_map.get("reference_gap_residue_count") or 0)

        if not ranges:
            ranges, mapped_residue_count, gap_reference_count = project_secondary_structure_ranges_to_alignment(
                str(record.seq),
                reference_positions,
                ss_by_reference_position,
            )
            model_residue_count = None
            unmapped_species_residue_count = gap_reference_count
        if not ranges:
            continue
        payload_records.append({
            "record_index": record_index,
            "record_id": record_id,
            "protein_record_id": protein_record_id,
            "species": species,
            "symbol": symbol,
            "scientific_name": clean_json_value(protein_meta.get("scientific_name")),
            "common_name": clean_json_value(protein_meta.get("common_name")),
            "preferred_public_label": clean_json_value(protein_meta.get("preferred_public_label")),
            "uniprot_accession": uniprot_accession,
            "alphafold_entry_id": alphafold_entry_id,
            "alphafold_source_label": alphafold_source_label,
            "source_method": source_method,
            "source_note": source_note,
            "projection_warning": projection_warning,
            "model_filename": model_relative_filename,
            "model_secondary_structure_method": model_secondary_structure_method,
            "model_residue_count": model_residue_count,
            "species_sequence_length": clean_json_value((alignment_map or {}).get("species_sequence_length")) or record_sequence_length,
            "mapped_residue_count": mapped_residue_count,
            "unmapped_species_residue_count": unmapped_species_residue_count,
            "gap_reference_position_count": gap_reference_count,
            "coverage_fraction": (mapped_residue_count / reference_index) if reference_index else None,
            "mapped_ranges": ranges,
        })

    if not payload_records:
        return comparative_alphafold_unavailable_payload("No species had residues that could be mapped to reference AlphaFold secondary structure.")

    return clean_json_value({
        "available": True,
        "filename": COMPARATIVE_ALPHAFOLD_SS_FILENAME,
        "method": "comparative_alphafold_secondary_structure_reference_mapping",
        "record_label_singular": "SS map",
        "record_label_plural": "SS maps",
        "note": (
            "Comparative secondary-structure ranges use cached species AlphaFold PDB models when available, "
            "mapped through aligned.fasta to the human reference ruler. Entries without a cached species "
            "model fall back to human reference AlphaFold secondary-structure projection through "
            "aligned_reference_projected.fasta and are flagged with projection_warning."
        ),
        "reference_species": str(reference_record.id).split("|", 1)[0],
        "reference_record_id": str(reference_record.id),
        "reference_alignment_record_id": full_alignment_maps_payload.get("reference_record_id"),
        "reference_model_filename": model_filename,
        "reference_secondary_structure_method": secondary_structure.get("method"),
        "track_length": reference_index,
        "record_count": len(payload_records),
        "records": payload_records,
    })


def write_comparative_alphafold_secondary_structure_bundle(outdir: Path | str,
                                                          reference_species: Optional[str] = None) -> Dict[str, Any]:
    outdir_path = Path(outdir)
    payload = build_comparative_alphafold_secondary_structure_payload(outdir_path, reference_species=reference_species)
    bundle_path = outdir_path / COMPARATIVE_ALPHAFOLD_SS_FILENAME
    bundle_path.write_text(
        json.dumps(clean_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def load_comparative_alphafold_secondary_structure(outdir: Path) -> Dict[str, Any]:
    payload = read_json_artifact(outdir / COMPARATIVE_ALPHAFOLD_SS_FILENAME)
    if not payload:
        payload = build_comparative_alphafold_secondary_structure_payload(outdir)
    records = payload.get("records")
    if not isinstance(records, list):
        records = payload.get("models")
    if not isinstance(records, list):
        records = []
    return clean_json_value({
        **payload,
        "available": bool(payload.get("available", bool(records))),
        "filename": payload.get("filename") or COMPARATIVE_ALPHAFOLD_SS_FILENAME,
        "records": records,
    })


def parse_pdb_backbone_residues(pdb_text: str) -> List[Dict[str, Any]]:
    residues: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name not in {"N", "CA", "C", "O"}:
            continue
        altloc = line[16:17].strip()
        if altloc not in {"", "A"}:
            continue
        chain = line[21:22].strip() or "A"
        insertion = line[26:27].strip()
        try:
            resi = int(line[22:26])
            coord = np.array([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ], dtype=float)
        except (TypeError, ValueError):
            continue
        key = (chain, resi, insertion)
        residue = residues.setdefault(key, {
            "chain": chain,
            "position": resi,
            "insertion_code": insertion,
            "residue_name": line[17:20].strip(),
            "atoms": {},
        })
        residue["atoms"][atom_name] = coord
    return sorted(residues.values(), key=lambda row: (str(row["chain"]), int(row["position"]), str(row["insertion_code"])))


def dihedral_degrees(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[float]:
    try:
        b0 = -(p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        norm = np.linalg.norm(b1)
        if not np.isfinite(norm) or norm == 0:
            return None
        b1 = b1 / norm
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        angle = float(np.degrees(np.arctan2(y, x)))
    except Exception:
        return None
    return angle if np.isfinite(angle) else None


def secondary_structure_kind(phi: Optional[float], psi: Optional[float]) -> str:
    if phi is None or psi is None:
        return "loop"
    if -100.0 <= phi <= -30.0 and -90.0 <= psi <= 0.0:
        return "helix"
    if -180.0 <= phi <= -45.0 and (45.0 <= psi <= 180.0 or -180.0 <= psi <= -145.0):
        return "sheet"
    return "loop"


def filter_short_secondary_runs(labels: List[str], kind: str, min_len: int) -> None:
    start = 0
    while start < len(labels):
        end = start
        while end + 1 < len(labels) and labels[end + 1] == labels[start]:
            end += 1
        if labels[start] == kind and end - start + 1 < min_len:
            for idx in range(start, end + 1):
                labels[idx] = "loop"
        start = end + 1


def parse_pdb_secondary_structure_records(pdb_text: str) -> List[Dict[str, Any]]:
    ranges: List[Dict[str, Any]] = []
    for line in pdb_text.splitlines():
        record = line[:6].strip().upper()
        if record == "HELIX":
            try:
                chain = (line[19:20].strip() or line[31:32].strip() or "A")
                start = int(line[21:25])
                end = int(line[33:37])
            except (TypeError, ValueError):
                parts = line.split()
                numeric_parts = [int(part) for part in parts if part.lstrip("-").isdigit()]
                if len(numeric_parts) < 2:
                    continue
                chain = "A"
                start, end = numeric_parts[-2], numeric_parts[-1]
            if start > end:
                start, end = end, start
            ranges.append({
                "kind": "helix",
                "chain": chain,
                "start": start,
                "end": end,
                "length": end - start + 1,
                "color": ALPHAFOLD_STRUCTURE_COLORS["helix"],
                "source": "pdb_helix_record",
            })
        elif record == "SHEET":
            try:
                chain = (line[21:22].strip() or line[32:33].strip() or "A")
                start = int(line[22:26])
                end = int(line[33:37])
            except (TypeError, ValueError):
                parts = line.split()
                numeric_parts = [int(part) for part in parts if part.lstrip("-").isdigit()]
                if len(numeric_parts) < 2:
                    continue
                chain = "A"
                start, end = numeric_parts[-2], numeric_parts[-1]
            if start > end:
                start, end = end, start
            ranges.append({
                "kind": "sheet",
                "chain": chain,
                "start": start,
                "end": end,
                "length": end - start + 1,
                "color": ALPHAFOLD_STRUCTURE_COLORS["sheet"],
                "source": "pdb_sheet_record",
            })
    return ranges


def apply_secondary_structure_ranges(backbone: Sequence[Dict[str, Any]],
                                     ss_ranges: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    by_position: Dict[Tuple[str, int], str] = {}
    for row in ss_ranges:
        kind = str(row.get("kind") or "loop")
        chain = str(row.get("chain") or "A")
        try:
            start = int(row.get("start"))
            end = int(row.get("end"))
        except (TypeError, ValueError):
            continue
        if start > end:
            start, end = end, start
        for position in range(start, end + 1):
            by_position[(chain, position)] = kind
    for residue in backbone:
        chain = str(residue.get("chain") or "A")
        position = int(residue.get("position"))
        kind = by_position.get((chain, position), "loop")
        annotated.append({
            "position": position,
            "chain": chain,
            "residue_name": residue.get("residue_name"),
            "secondary_structure": kind,
            "ss_code": {"helix": "h", "sheet": "s"}.get(kind, "c"),
        })
    return annotated


def contiguous_secondary_ranges(residues: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranges: List[Dict[str, Any]] = []
    if not residues:
        return ranges
    counters = {"helix": 0, "sheet": 0}
    start_idx = 0
    for idx in range(1, len(residues) + 1):
        if idx < len(residues) and residues[idx].get("secondary_structure") == residues[start_idx].get("secondary_structure"):
            continue
        start_res = residues[start_idx]
        end_res = residues[idx - 1]
        kind = str(start_res.get("secondary_structure") or "loop")
        row = {
            "kind": kind,
            "start": clean_json_value(start_res.get("position")),
            "end": clean_json_value(end_res.get("position")),
            "length": int(idx - start_idx),
            "color": ALPHAFOLD_STRUCTURE_COLORS.get(kind, ALPHAFOLD_STRUCTURE_COLORS["loop"]),
        }
        if kind in counters:
            counters[kind] += 1
            prefix = "H" if kind == "helix" else "S"
            row["ordinal"] = counters[kind]
            row["display_label"] = f"{prefix}{counters[kind]}"
        ranges.append(row)
        start_idx = idx
    return ranges


def smoothed_binary_track(labels: Sequence[str], kind: str, window: int = 9) -> List[float]:
    if not labels:
        return []
    half = max(0, int(window) // 2)
    values = [1.0 if label == kind else 0.0 for label in labels]
    smoothed: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        segment = values[start:end]
        smoothed.append(round(float(sum(segment) / len(segment)), 4) if segment else 0.0)
    return smoothed


def hex_to_rgb(color: str) -> Tuple[int, int, int]:
    text = str(color or "").strip().lstrip("#")
    if len(text) != 6:
        return (248, 250, 252)
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except ValueError:
        return (248, 250, 252)


def blend_hex(left: str, right: str, fraction: float) -> str:
    t = max(0.0, min(1.0, float(fraction)))
    l_rgb = hex_to_rgb(left)
    r_rgb = hex_to_rgb(right)
    rgb = tuple(int(round(l + (r - l) * t)) for l, r in zip(l_rgb, r_rgb))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def local_charge_color(charge: float, max_abs: float = 5.0) -> str:
    neutral = ALPHAFOLD_STRUCTURE_COLORS["charge_neutral"]
    if charge > 0:
        return blend_hex(neutral, ALPHAFOLD_STRUCTURE_COLORS["charge_positive"], min(abs(charge) / max_abs, 1.0))
    if charge < 0:
        return blend_hex(neutral, ALPHAFOLD_STRUCTURE_COLORS["charge_negative"], min(abs(charge) / max_abs, 1.0))
    return neutral


def build_local_charge_payload(sequence: str, window_size: int = 5) -> Dict[str, Any]:
    seq = str(sequence or "").upper()
    if not seq:
        return {
            "available": False,
            "reason": "No human reference or AlphaFold sequence was available for local charge calculation.",
            "window_size": window_size,
            "residues": [],
        }
    half = max(0, int(window_size) // 2)
    charge_map = {"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}
    residues: List[Dict[str, Any]] = []
    charges: List[float] = []
    for idx, aa in enumerate(seq):
        start_idx = max(0, idx - half)
        end_idx = min(len(seq), idx + half + 1)
        window = seq[start_idx:end_idx]
        charge = round(float(sum(charge_map.get(residue, 0.0) for residue in window)), 3)
        charges.append(charge)
        residues.append({
            "position": idx + 1,
            "residue": aa,
            "window_start": start_idx + 1,
            "window_end": end_idx,
            "window_sequence": window,
            "charge": charge,
            "color": local_charge_color(charge),
        })
    return {
        "available": True,
        "window_size": int(window_size),
        "method": "sequence_sidechain_charge_sliding_window",
        "requires_hydrogens": False,
        "pH_assumption": "approximate pH 7.4 side-chain charge; K/R=+1, D/E=-1, H=+0.1; termini excluded",
        "note": "This is a sequence-derived local charge track. Full electrostatic surface calculations require protonation states and hydrogens; AlphaFold PDB coordinates do not provide those by default.",
        "min_charge": min(charges) if charges else 0.0,
        "max_charge": max(charges) if charges else 0.0,
        "colors": {
            "positive": ALPHAFOLD_STRUCTURE_COLORS["charge_positive"],
            "negative": ALPHAFOLD_STRUCTURE_COLORS["charge_negative"],
            "neutral": ALPHAFOLD_STRUCTURE_COLORS["charge_neutral"],
        },
        "residues": clean_json_value(residues),
    }


def build_calcium_binding_payload(gene_symbol: Optional[str]) -> Dict[str, Any]:
    if str(gene_symbol or "").strip().upper() != "PLA2G4A":
        return {"available": False, "reason": "No curated Ca2+ binding annotation is bundled for this gene."}
    payload = dict(PLA2G4A_CALCIUM_BINDING_ANNOTATION)
    ranges = []
    for loop in payload.get("loops") or []:
        ranges.append({
            "source": "curated_calcium_binding",
            "kind": "calcium",
            "label": f"Ca2+ {loop['label']}",
            "start": loop["start"],
            "end": loop["end"],
            "length": int(loop["end"]) - int(loop["start"]) + 1,
            "color": loop.get("color") or ALPHAFOLD_STRUCTURE_COLORS["calcium"],
            "group": "calcium_binding_region",
            "description": loop.get("description"),
        })
    payload["ranges"] = ranges
    return clean_json_value(payload)


def assign_pdb_secondary_structure(pdb_text: str) -> Dict[str, Any]:
    backbone = parse_pdb_backbone_residues(pdb_text)
    if not backbone:
        return {
            "available": False,
            "method": "alphafold_pdb_secondary_structure",
            "reason": "No backbone residues could be parsed from the AlphaFold PDB.",
            "residues": [],
            "ranges": [],
            "helix_propensity": [],
            "sheet_propensity": [],
        }

    declared_ranges = parse_pdb_secondary_structure_records(pdb_text)
    if declared_ranges:
        annotated = apply_secondary_structure_ranges(backbone, declared_ranges)
        labels = [str(row.get("secondary_structure") or "loop") for row in annotated]
        helix_propensity = smoothed_binary_track(labels, "helix")
        sheet_propensity = smoothed_binary_track(labels, "sheet")
        for row, helix_value, sheet_value in zip(annotated, helix_propensity, sheet_propensity):
            row["helix_propensity"] = helix_value
            row["sheet_propensity"] = sheet_value
        return {
            "available": True,
            "method": "alphafold_pdb_helix_sheet_records",
            "note": "Secondary-structure overlay uses HELIX/SHEET annotations carried by the AlphaFold PDB model.",
            "residues": clean_json_value(annotated),
            "ranges": contiguous_secondary_ranges(annotated),
            "helix_propensity": helix_propensity,
            "sheet_propensity": sheet_propensity,
        }

    raw_labels: List[str] = []
    annotated: List[Dict[str, Any]] = []
    for idx, residue in enumerate(backbone):
        atoms = residue.get("atoms") or {}
        phi = None
        psi = None
        if idx > 0:
            prev_atoms = backbone[idx - 1].get("atoms") or {}
            if all(name in atoms for name in ("N", "CA", "C")) and "C" in prev_atoms:
                phi = dihedral_degrees(prev_atoms["C"], atoms["N"], atoms["CA"], atoms["C"])
        if idx + 1 < len(backbone):
            next_atoms = backbone[idx + 1].get("atoms") or {}
            if all(name in atoms for name in ("N", "CA", "C")) and "N" in next_atoms:
                psi = dihedral_degrees(atoms["N"], atoms["CA"], atoms["C"], next_atoms["N"])
        kind = secondary_structure_kind(phi, psi)
        raw_labels.append(kind)
        annotated.append({
            "position": int(residue["position"]),
            "chain": residue.get("chain") or "A",
            "residue_name": residue.get("residue_name"),
            "phi": clean_json_value(phi, float_digits=3),
            "psi": clean_json_value(psi, float_digits=3),
            "secondary_structure": kind,
            "ss_code": {"helix": "h", "sheet": "s"}.get(kind, "c"),
        })

    labels = list(raw_labels)
    filter_short_secondary_runs(labels, "helix", 4)
    filter_short_secondary_runs(labels, "sheet", 3)
    for row, kind in zip(annotated, labels):
        row["secondary_structure"] = kind
        row["ss_code"] = {"helix": "h", "sheet": "s"}.get(kind, "c")
    helix_propensity = smoothed_binary_track(labels, "helix")
    sheet_propensity = smoothed_binary_track(labels, "sheet")
    for row, helix_value, sheet_value in zip(annotated, helix_propensity, sheet_propensity):
        row["helix_propensity"] = helix_value
        row["sheet_propensity"] = sheet_value

    return {
        "available": True,
        "method": "phi_psi_from_alphafold_backbone",
        "note": "The AlphaFold PDB did not include HELIX/SHEET records, so the overlay falls back to local backbone torsion geometry derived from the model coordinates.",
        "residues": clean_json_value(annotated),
        "ranges": contiguous_secondary_ranges(annotated),
        "helix_propensity": helix_propensity,
        "sheet_propensity": sheet_propensity,
    }


def extract_reference_sequence(outdir: Path, reference_species: Optional[str]) -> str:
    fasta_path = outdir / "aligned_reference_projected.fasta"
    if not fasta_path.exists():
        return ""
    target = str(reference_species or "").strip().lower()
    try:
        records = list(SeqIO.parse(str(fasta_path), "fasta"))
    except Exception:
        return ""
    if not records:
        return ""
    chosen = records[0]
    if target:
        for record in records:
            species = str(record.id).split("|", 1)[0].strip().lower()
            if species == target:
                chosen = record
                break
    return "".join(ch for ch in str(chosen.seq).upper() if ch not in GAP_CHARS)


def add_structure_range(ranges: List[Dict[str, Any]],
                        *,
                        source: str,
                        kind: str,
                        label: str,
                        start: Any,
                        end: Any,
                        color: Optional[str] = None,
                        score: Any = None,
                        group: Any = None,
                        extra: Optional[Dict[str, Any]] = None) -> None:
    try:
        start_int = int(float(start))
        end_int = int(float(end))
    except (TypeError, ValueError):
        return
    if start_int <= 0 or end_int <= 0:
        return
    if start_int > end_int:
        start_int, end_int = end_int, start_int
    row = {
        "source": source,
        "kind": kind,
        "label": label,
        "start": start_int,
        "end": end_int,
        "length": end_int - start_int + 1,
        "color": color or ALPHAFOLD_STRUCTURE_COLORS.get(kind, ALPHAFOLD_STRUCTURE_COLORS["manual"]),
        "score": clean_json_value(score),
        "group": clean_json_value(group),
    }
    if extra:
        row.update({str(key): clean_json_value(value) for key, value in extra.items()})
    ranges.append(row)


def build_alphafold_structure_ranges(tables: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    ranges: List[Dict[str, Any]] = []
    conserved_df = tables.get("conserved_regions", pd.DataFrame())
    if conserved_df is not None and not conserved_df.empty:
        for _, row in conserved_df.iterrows():
            score_column = str(row.get("score_column") or "conserved")
            add_structure_range(
                ranges,
                source="conserved_regions",
                kind="conserved",
                label=f"{score_column} conserved",
                start=row.get("start_reference_position"),
                end=row.get("end_reference_position"),
                color=ALPHAFOLD_STRUCTURE_COLORS["conserved"],
                score=row.get("mean_score"),
                group=score_column,
                extra={"mean_occupancy": row.get("mean_occupancy")},
            )

    clade_df = tables.get("clade_fourier_regions", pd.DataFrame())
    if clade_df is not None and not clade_df.empty:
        for _, row in clade_df.iterrows():
            region_type = str(row.get("region_type") or "")
            kind = "divergent" if "divergent" in region_type.lower() else "conserved"
            clade = str(row.get("clade") or "clade")
            add_structure_range(
                ranges,
                source="clade_fourier_regions",
                kind=kind,
                label=f"{clade} {region_type}".strip(),
                start=row.get("start_reference_position"),
                end=row.get("end_reference_position"),
                color=ALPHAFOLD_STRUCTURE_COLORS["divergent"] if kind == "divergent" else ALPHAFOLD_STRUCTURE_COLORS["conserved"],
                score=row.get("mean_identity_to_human"),
                group=clade,
                extra={
                    "region_type": region_type,
                    "mean_global_identity_to_human": row.get("mean_global_identity_to_human"),
                    "mean_delta_vs_global": row.get("mean_delta_vs_global"),
                },
            )

    chunk_map_df = tables.get("selected_consensus_chunks_structure_map", pd.DataFrame())
    if chunk_map_df is not None and not chunk_map_df.empty:
        for _, row in chunk_map_df.iterrows():
            if str(row.get("mapping_status") or "").lower() == "missing_range":
                continue
            add_structure_range(
                ranges,
                source="selected_consensus_chunks_structure_map",
                kind="manual",
                label=str(row.get("label") or row.get("chunk_id") or "selected chunk"),
                start=row.get("start_structure_residue") or row.get("start_reference_position"),
                end=row.get("end_structure_residue") or row.get("end_reference_position"),
                color=str(row.get("color_hex") or ALPHAFOLD_STRUCTURE_COLORS["manual"]),
                score=row.get("score"),
                group=row.get("chunk_id"),
                extra={"mapping_status": row.get("mapping_status"), "notes": row.get("notes")},
            )

    annotated_df = tables.get("annotated_functional_sites", pd.DataFrame())
    if annotated_df is not None and not annotated_df.empty:
        for _, row in annotated_df.iterrows():
            add_structure_range(
                ranges,
                source="annotated_functional_sites",
                kind="manual",
                label=str(row.get("label") or "annotated site"),
                start=row.get("position"),
                end=row.get("position"),
                color=ALPHAFOLD_STRUCTURE_COLORS["manual"],
                score=row.get("reference_identity_fraction"),
                group="annotated_site",
            )
    for position_spec in ALIGNMENT_BROWSER_REFERENCE_POSITION_LANDMARKS.get("PLA2G4A", ()):
        add_structure_range(
            ranges,
            source="curated_catalytic_residue",
            kind="manual",
            label=str(position_spec.get("label") or "catalytic residue"),
            start=position_spec.get("position"),
            end=position_spec.get("position"),
            color=str(position_spec.get("color") or ALPHAFOLD_STRUCTURE_COLORS["manual"]),
            group="catalytic_residue",
            extra={"description": position_spec.get("description")},
        )
    ranges.sort(key=lambda row: (int(row["start"]), int(row["end"]), str(row["source"]), str(row["label"])))
    return ranges


def build_alphafold_structure_payload(outdir: Path,
                                      tables: Dict[str, pd.DataFrame],
                                      reference_species: Optional[str]) -> Tuple[Dict[str, Any], str]:
    metadata = read_json_artifact(outdir / ALPHAFOLD_METADATA_FILENAME)
    viewer_js = read_text_artifact(outdir / ALPHAFOLD_VIEWER_JS_FILENAME)
    comparative_ss = load_comparative_alphafold_secondary_structure(outdir)
    if not metadata.get("available"):
        return {
            "available": False,
            "reason": metadata.get("reason") or "Human AlphaFold bundle metadata is not available.",
            "viewer_js_available": bool(viewer_js),
            "colors": ALPHAFOLD_STRUCTURE_COLORS,
            "density_map": {
                "available": False,
                "reason": "No electron-density map is bundled with this AlphaFold report.",
            },
            "comparative_secondary_structure": comparative_ss,
            "ranges": [],
        }, viewer_js

    model_filename = str(metadata.get("model_filename") or ALPHAFOLD_MODEL_FILENAME)
    pdb_text = read_text_artifact(outdir / model_filename)
    if not pdb_text:
        return {
            "available": False,
            "reason": f"AlphaFold model file was not found or could not be read: {model_filename}",
            "viewer_js_available": bool(viewer_js),
            "colors": ALPHAFOLD_STRUCTURE_COLORS,
            "density_map": {
                "available": False,
                "reason": "No electron-density map is bundled with this AlphaFold report.",
            },
            "comparative_secondary_structure": comparative_ss,
            "ranges": build_alphafold_structure_ranges(tables),
        }, viewer_js

    prediction = metadata.get("alphafold_prediction") if isinstance(metadata.get("alphafold_prediction"), dict) else {}
    af_sequence = str(prediction.get("sequence") or prediction.get("uniprotSequence") or "").upper()
    reference_sequence = extract_reference_sequence(outdir, reference_species)
    sequence_match = bool(af_sequence and reference_sequence and af_sequence == reference_sequence)
    secondary_structure = assign_pdb_secondary_structure(pdb_text)
    gene_symbol = str(prediction.get("gene") or "").strip().upper()
    local_charge = build_local_charge_payload(af_sequence or reference_sequence)
    calcium_binding = build_calcium_binding_payload(gene_symbol)
    structure_ranges = build_alphafold_structure_ranges(tables)
    if calcium_binding.get("available"):
        structure_ranges.extend(calcium_binding.get("ranges") or [])
    structure_ranges.sort(key=lambda row: (int(row["start"]), int(row["end"]), str(row["source"]), str(row["label"])))
    residue_count = int(prediction.get("sequenceEnd") or len(af_sequence) or len(secondary_structure.get("residues") or []))
    return {
        "available": True,
        "viewer_js_available": bool(viewer_js),
        "viewer_js_filename": ALPHAFOLD_VIEWER_JS_FILENAME if viewer_js else None,
        "model_filename": model_filename,
        "pdb_text": pdb_text,
        "colors": ALPHAFOLD_STRUCTURE_COLORS,
        "metadata": {
            "uniprot_accession": metadata.get("uniprot_accession") or prediction.get("uniprotAccession"),
            "entry_id": prediction.get("entryId"),
            "description": prediction.get("uniprotDescription"),
            "global_metric_value": prediction.get("globalMetricValue"),
            "latest_version": prediction.get("latestVersion"),
            "model_created_date": prediction.get("modelCreatedDate"),
            "pdb_url": prediction.get("pdbUrl"),
        },
        "structure_provenance": {
            "coordinate_source": "AlphaFold Protein Structure Database",
            "viewer_engine": "3Dmol.js",
            "force_field": "none_applied_by_v9_7_viewer",
            "note": "V9.7 displays the downloaded AlphaFold coordinates and does not run AMBER, CHARMM, OPLS, OpenMM, or molecular-dynamics minimization.",
        },
        "density_map": {
            "available": False,
            "reason": "No experimental electron-density map is bundled with this AlphaFold coordinate model; AlphaFold PDB files contain predicted coordinates, not density.",
            "supported_formats": ["cube", "dx", "ccp4"],
        },
        "residue_count": residue_count,
        "sequence_validation": {
            "alphafold_sequence_length": len(af_sequence) if af_sequence else None,
            "reference_sequence_length": len(reference_sequence) if reference_sequence else None,
            "residue_numbering": "alphafold_residue_equals_human_reference_residue",
            "sequence_match": sequence_match,
            "status": "ok" if sequence_match else "not_checked" if not af_sequence or not reference_sequence else "mismatch",
        },
        "secondary_structure": secondary_structure,
        "comparative_secondary_structure": comparative_ss,
        "local_charge": local_charge,
        "calcium_binding": calcium_binding,
        "ranges": structure_ranges,
    }, viewer_js


def build_report_payload(outdir: Path,
                         tables: Dict[str, pd.DataFrame],
                         artifacts_df: pd.DataFrame,
                         pairwise_df: pd.DataFrame,
                         run_meta: Dict[str, Any]) -> Dict[str, Any]:
    downloads_df = artifacts_df.copy()
    downloads_df = downloads_df[["artifact_group", "file_type", "relative_path", "size_bytes"]] if not downloads_df.empty else pd.DataFrame(columns=["artifact_group", "file_type", "relative_path", "size_bytes"])

    # V11: load representatives TSV early so it can be folded into html_tables.
    _v11_reps_for_report = pd.DataFrame()
    _v11_reps_path = outdir / V11_DEFAULT_REPRESENTATIVE_CSV
    if _v11_reps_path.exists():
        try:
            _v11_reps_for_report = pd.read_csv(_v11_reps_path, sep="\t")
        except Exception:
            _v11_reps_for_report = pd.DataFrame()

    # V11 motif tables (separate so they can be folded into html_tables).
    def _v11_safe_read_tsv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return pd.DataFrame()

    def _v11_safe_read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    _v11_motifs_master_for_report = _v11_safe_read_tsv(outdir / V11_MOTIFS_MASTER_TSV)
    _v11_motif_per_clade_for_report = _v11_safe_read_tsv(outdir / V11_MOTIF_EVOLUTION_PER_CLADE_TSV)
    _v11_lineage_stab_for_report = _v11_safe_read_csv(outdir / V11_LINEAGE_STABILIZATION_CSV)
    # For the lineage stabilization table we display only the rows where the
    # |score| is non-trivial so the UI is browsable. Keep top 200 by |score|.
    if not _v11_lineage_stab_for_report.empty:
        _stab_disp = _v11_lineage_stab_for_report.copy()
        _stab_disp["_abs"] = _stab_disp["stabilization_score"].abs()
        _stab_disp = _stab_disp.sort_values("_abs", ascending=False).drop(columns=["_abs"]).head(200)
        _v11_lineage_stab_for_report = _stab_disp.reset_index(drop=True)

    alphafold_structure, alphafold_viewer_js = build_alphafold_structure_payload(
        outdir,
        tables,
        clean_json_value(run_meta.get("reference_species")),
    )

    html_tables: Dict[str, List[Dict[str, Any]]] = {
        "orthologs": dataframe_to_json_records(filter_columns(tables.get("orthologs", pd.DataFrame()), TABLE_DISPLAY_SPECS["orthologs"]["columns"])),
        "sequence_retrieval": dataframe_to_json_records(filter_columns(tables.get("sequence_retrieval", pd.DataFrame()), TABLE_DISPLAY_SPECS["sequence_retrieval"]["columns"])),
        "protein_metadata": dataframe_to_json_records(filter_columns(tables.get("protein_metadata", pd.DataFrame()), TABLE_DISPLAY_SPECS["protein_metadata"]["columns"])),
        "protein_features": dataframe_to_json_records(filter_columns(tables.get("protein_features", pd.DataFrame()), TABLE_DISPLAY_SPECS["protein_features"]["columns"])),
        "protein_xrefs": dataframe_to_json_records(filter_columns(tables.get("protein_xrefs", pd.DataFrame()), TABLE_DISPLAY_SPECS["protein_xrefs"]["columns"])),
        "length_filter": dataframe_to_json_records(filter_columns(tables.get("length_filter", pd.DataFrame()), TABLE_DISPLAY_SPECS["length_filter"]["columns"])),
        "conservation_scan": dataframe_to_json_records(filter_columns(tables.get("conservation_scan", pd.DataFrame()), TABLE_DISPLAY_SPECS["conservation_scan"]["columns"])),
        "conserved_regions": dataframe_to_json_records(filter_columns(tables.get("conserved_regions", pd.DataFrame()), TABLE_DISPLAY_SPECS["conserved_regions"]["columns"])),
        "annotated_functional_sites": dataframe_to_json_records(filter_columns(tables.get("annotated_functional_sites", pd.DataFrame()), TABLE_DISPLAY_SPECS["annotated_functional_sites"]["columns"])),
        "annotated_site_clade_comparison": dataframe_to_json_records(filter_columns(tables.get("annotated_site_clade_comparison", pd.DataFrame()), TABLE_DISPLAY_SPECS["annotated_site_clade_comparison"]["columns"])),
        "domains": dataframe_to_json_records(filter_columns(tables.get("domains", pd.DataFrame()), TABLE_DISPLAY_SPECS["domains"]["columns"])),
        "selected_consensus_chunks": dataframe_to_json_records(filter_columns(tables.get("selected_consensus_chunks", pd.DataFrame()), TABLE_DISPLAY_SPECS["selected_consensus_chunks"]["columns"])),
        "selected_consensus_chunks_structure_map": dataframe_to_json_records(filter_columns(tables.get("selected_consensus_chunks_structure_map", pd.DataFrame()), TABLE_DISPLAY_SPECS["selected_consensus_chunks_structure_map"]["columns"])),
        "clade_identity_profiles": dataframe_to_json_records(filter_columns(tables.get("clade_identity_profiles_wide", pd.DataFrame()), TABLE_DISPLAY_SPECS["clade_identity_profiles"]["columns"])),
        "clade_difference_from_global": dataframe_to_json_records(filter_columns(tables.get("clade_difference_from_global_wide", pd.DataFrame()), TABLE_DISPLAY_SPECS["clade_difference_from_global"]["columns"])),
        "clade_fourier_regions": dataframe_to_json_records(filter_columns(tables.get("clade_fourier_regions", pd.DataFrame()), TABLE_DISPLAY_SPECS["clade_fourier_regions"]["columns"])),
        "domain_clade_conservation_summary": dataframe_to_json_records(filter_columns(tables.get("domain_clade_conservation_summary", pd.DataFrame()), TABLE_DISPLAY_SPECS["domain_clade_conservation_summary"]["columns"])),
        "node_conservation_extremes": dataframe_to_json_records(filter_columns(tables.get("node_conservation_extremes", pd.DataFrame()), TABLE_DISPLAY_SPECS["node_conservation_extremes"]["columns"])),
        "evolutionary_segments": dataframe_to_json_records(filter_columns(tables.get("evolutionary_segments", pd.DataFrame()), TABLE_DISPLAY_SPECS["evolutionary_segments"]["columns"])),
        "evolutionary_segment_metrics": dataframe_to_json_records(filter_columns(tables.get("evolutionary_segment_metrics", pd.DataFrame()), TABLE_DISPLAY_SPECS["evolutionary_segment_metrics"]["columns"])),
        "evolutionary_alignment_windows_manifest": dataframe_to_json_records(filter_columns(tables.get("evolutionary_alignment_windows_manifest", pd.DataFrame()), TABLE_DISPLAY_SPECS["evolutionary_alignment_windows_manifest"]["columns"])),
        "pairwise_reports": dataframe_to_json_records(filter_columns(pairwise_df, TABLE_DISPLAY_SPECS["pairwise_reports"]["columns"])),
        "downloads": dataframe_to_json_records(filter_columns(downloads_df, TABLE_DISPLAY_SPECS["downloads"]["columns"])),
        "v11_representatives": dataframe_to_json_records(filter_columns(_v11_reps_for_report, TABLE_DISPLAY_SPECS["v11_representatives"]["columns"])),
        "v11_motifs_master": dataframe_to_json_records(filter_columns(_v11_motifs_master_for_report, TABLE_DISPLAY_SPECS["v11_motifs_master"]["columns"])),
        "v11_motif_evolution_per_clade": dataframe_to_json_records(filter_columns(_v11_motif_per_clade_for_report, TABLE_DISPLAY_SPECS["v11_motif_evolution_per_clade"]["columns"])),
        "v11_lineage_stabilization": dataframe_to_json_records(filter_columns(_v11_lineage_stab_for_report, TABLE_DISPLAY_SPECS["v11_lineage_stabilization"]["columns"])),
    }

    quick_links = [
        ("Run summary", choose_existing_artifact(outdir, ["run_summary.txt"])),
        ("Ortholog table", choose_existing_artifact(outdir, ["orthologs.tsv"])),
        ("Sequence retrieval", choose_existing_artifact(outdir, ["sequence_retrieval.tsv"])),
        ("Protein metadata", choose_existing_artifact(outdir, ["protein_metadata.tsv"])),
        ("Tree nomenclature cache", choose_existing_artifact(outdir, ["tree_nomenclature.json"])),
        ("Reference-projected FASTA", choose_existing_artifact(outdir, ["aligned_reference_projected.fasta"])),
        ("Conservation scan CSV", choose_existing_artifact(outdir, ["conservation_scan.csv"])),
        ("Evolutionary segments CSV", choose_existing_artifact(outdir, ["evolutionary_segments.csv"])),
        ("Evolutionary segment metrics CSV", choose_existing_artifact(outdir, ["evolutionary_segment_metrics_by_clade.csv"])),
        ("Evolutionary windows manifest", choose_existing_artifact(outdir, ["evolutionary_alignment_windows_manifest.csv"])),
        ("Evolutionary divergence bars", choose_existing_artifact(outdir, ["evolutionary_segment_bars.svg", "evolutionary_segment_bars.png"])),
        ("Evolutionary divergence bars (full alignment)", choose_existing_artifact(outdir, ["evolutionary_segment_bars_aligned_full.svg", "evolutionary_segment_bars_aligned_full.png"])),
        ("Node conservation extremes", choose_existing_artifact(outdir, [NODE_CONSERVATION_EXTREMES_FILENAME])),
        ("Node conservation tree", choose_existing_artifact(outdir, [NODE_CONSERVATION_TREE_SVG_FILENAME])),
        ("Node conservation paper tree", choose_existing_artifact(outdir, [NODE_CONSERVATION_PAPER_TREE_SVG_FILENAME])),
        ("AlphaFold metadata", choose_existing_artifact(outdir, ["human_reference_alphafold_metadata.json"])),
        ("AlphaFold overlay", choose_existing_artifact(outdir, ["human_reference_alphafold_overlay.pml"])),
        ("Colored alignment PDF", choose_existing_artifact(outdir, ["alignment_reference_projected_colored.pdf"])),
        ("Interactive alignment browser", choose_existing_artifact(outdir, [ALIGNMENT_BROWSER_FILENAME])),
        ("Default browser similar runs", choose_existing_artifact(outdir, [f"{ALIGNMENT_BROWSER_DEFAULT_EXPORT_PREFIX}_runs.csv"])),
        ("Default browser consensus", choose_existing_artifact(outdir, [f"{ALIGNMENT_BROWSER_DEFAULT_EXPORT_PREFIX}_consensus.csv"])),
        ("Default browser architecture", choose_existing_artifact(outdir, [f"{ALIGNMENT_BROWSER_DEFAULT_EXPORT_PREFIX}_architecture.svg", f"{ALIGNMENT_BROWSER_DEFAULT_EXPORT_PREFIX}_architecture.png"])),
    ]
    overview_figures = [
        ("Phylogeny", choose_existing_artifact(outdir, ["phylo_tree.svg", "phylo_tree.png"])),
        ("Phylogeny nomenclature", choose_existing_artifact(outdir, ["phylo_tree_nomenclature.svg", "phylo_tree_nomenclature.png"])),
        ("Node conservation paper tree", choose_existing_artifact(outdir, [NODE_CONSERVATION_PAPER_TREE_SVG_FILENAME])),
        ("Node conservation extremes", choose_existing_artifact(outdir, [NODE_CONSERVATION_TREE_SVG_FILENAME])),
        ("Reference conservation scan", choose_existing_artifact(outdir, ["conservation_scan.svg", "conservation_scan.png"])),
        ("Reference domain architecture", choose_existing_artifact(outdir, ["reference_domain_architecture.svg", "reference_domain_architecture.png"])),
        ("Annotated-site clade comparison", choose_existing_artifact(outdir, ["annotated_site_clade_comparison.svg", "annotated_site_clade_comparison.png"])),
    ]
    conservation_figures = [
        ("Exact conservation", choose_existing_artifact(outdir, ["exact_conservation.svg", "exact_conservation.png"])),
        ("Property conservation", choose_existing_artifact(outdir, ["property_conservation.svg", "property_conservation.png"])),
        ("Property heatmap", choose_existing_artifact(outdir, ["property_heatmap.svg", "property_heatmap.png"])),
    ]
    domain_figures = [
        ("Reference domain architecture", choose_existing_artifact(outdir, ["reference_domain_architecture.svg", "reference_domain_architecture.png"])),
        ("Annotated-site clade comparison", choose_existing_artifact(outdir, ["annotated_site_clade_comparison.svg", "annotated_site_clade_comparison.png"])),
    ]
    evolutionary_figures = [
        (
            "Domain and motif divergence bars (reference-projected)",
            choose_existing_artifact(outdir, ["evolutionary_segment_bars_aligned_reference_projected.svg", "evolutionary_segment_bars.svg", "evolutionary_segment_bars_aligned_reference_projected.png", "evolutionary_segment_bars.png"]),
        ),
        (
            "Domain and motif divergence bars (full alignment)",
            choose_existing_artifact(outdir, ["evolutionary_segment_bars_aligned_full.svg", "evolutionary_segment_bars_aligned_full.png"]),
        ),
    ]
    clade_figures = [
        ("Clade conservation profiles", choose_existing_artifact(outdir, ["clade_fourier_conservation_profiles.svg", "clade_fourier_conservation_profiles.png"])),
        ("Clade-vs-global differences", choose_existing_artifact(outdir, ["clade_difference_from_global_heatmap.svg", "clade_difference_from_global_heatmap.png"])),
        ("Clade Fourier spectrum", choose_existing_artifact(outdir, ["clade_fourier_spectrum.svg", "clade_fourier_spectrum.png"])),
        ("Domain-by-clade heatmap", choose_existing_artifact(outdir, ["domain_clade_conservation_heatmap.svg", "domain_clade_conservation_heatmap.png"])),
        ("Node conservation paper tree", choose_existing_artifact(outdir, [NODE_CONSERVATION_PAPER_TREE_SVG_FILENAME])),
        ("Node conservation extremes", choose_existing_artifact(outdir, [NODE_CONSERVATION_TREE_SVG_FILENAME])),
    ]

    v11_representative_figures = [
        ("V11 per-clade consolidated summary (consensus SS + mean net charge + domains)", choose_existing_artifact(outdir, [V11_CLADE_CONSOLIDATED_SUMMARY_SVG, V11_CLADE_CONSOLIDATED_SUMMARY_PNG])),
        ("V11 per-clade identity bubble grid (broad clades)", choose_existing_artifact(outdir, [V11_CLADE_IDENTITY_BUBBLE_SVG, V11_CLADE_IDENTITY_BUBBLE_PNG])),
        ("V11 per-clade identity bubble grid (subdivided 9-group: Primates / Rodents / OtherMammals split)", choose_existing_artifact(outdir, [V11_GROUPED_CLADE_IDENTITY_BUBBLE_SVG, V11_GROUPED_CLADE_IDENTITY_BUBBLE_PNG])),
        ("V11 paper-quality phylogeny", choose_existing_artifact(outdir, ["v11_phylo_tree_paper_quality.svg", "v11_phylo_tree_paper_quality.png"])),
        ("V11 representative net-charge heatmap (Δ vs human)", choose_existing_artifact(outdir, ["v11_representative_net_charge_heatmap.svg", "v11_representative_net_charge_heatmap.png"])),
        ("V11 representative aromaticity heatmap (Δ vs human)", choose_existing_artifact(outdir, ["v11_representative_aromaticity_heatmap.svg", "v11_representative_aromaticity_heatmap.png"])),
        ("V11 representative property traces (net-charge & aromaticity)", choose_existing_artifact(outdir, [V11_REPRESENTATIVE_PROPERTY_TRACES_SVG, V11_REPRESENTATIVE_PROPERTY_TRACES_PNG])),
    ]

    # V11 motif figure gallery — collect every per-motif logo + species heatmap
    # plus the landscape and annotated alignment SVG. Sorted with user-supplied
    # motifs first (we tag them via filename pattern), then library motifs.
    v11_motif_figures: List[Tuple[str, Optional[str]]] = []
    landscape_rel = choose_existing_artifact(outdir, [V11_LINEAGE_STABILIZATION_LANDSCAPE_SVG, V11_LINEAGE_STABILIZATION_LANDSCAPE_PNG])
    if landscape_rel:
        v11_motif_figures.append(("V11 lineage-stabilization landscape", landscape_rel))
    aln_anno_rel = choose_existing_artifact(outdir, [V11_ALIGNMENT_WITH_MOTIF_ANNOTATIONS_SVG])
    if aln_anno_rel:
        v11_motif_figures.append(("V11 MUSCLE alignment with motif annotations", aln_anno_rel))
    # Collect per-motif figures from disk. The filename pattern is
    # v11_motif_<motif_id>_clade_logos.svg and v11_motif_<motif_id>_species_heatmap.svg.
    motif_logo_files = sorted((outdir.glob("v11_motif_*_clade_logos.svg")), key=lambda p: (0 if "user__" in p.name else 1, p.name))
    for path in motif_logo_files:
        # Pull a human-friendly label out of motif_id.
        stem = path.stem.replace("v11_motif_", "").replace("_clade_logos", "")
        is_user = stem.startswith("user__")
        kind = "user-supplied" if is_user else "library"
        v11_motif_figures.append((f"V11 motif clade logos — {stem} ({kind})", path.name))
        heat = outdir / f"v11_motif_{stem}_species_heatmap.svg"
        if heat.exists():
            v11_motif_figures.append((f"V11 motif representative-species heatmap — {stem} ({kind})", heat.name))

    return {
        "title": f"{run_meta.get('gene_symbol') or outdir.name} evolutionary conservation report",
        "meta": run_meta,
        "alphafold_structure": clean_json_value(alphafold_structure),
        "alphafold_viewer_js": alphafold_viewer_js,
        "tables": html_tables,
        "table_configs": TABLE_DISPLAY_SPECS,
        "quick_links": [(label, href) for label, href in quick_links if href],
        "overview_figures": [(label, href) for label, href in overview_figures if href],
        "conservation_figures": [(label, href) for label, href in conservation_figures if href],
        "domain_figures": [(label, href) for label, href in domain_figures if href],
        "evolutionary_figures": [(label, href) for label, href in evolutionary_figures if href],
        "clade_figures": [(label, href) for label, href in clade_figures if href],
        "v11_representative_figures": [(label, href) for label, href in v11_representative_figures if href],
        "v11_representatives_df": _v11_reps_for_report,
        "v11_motif_figures": [(label, href) for label, href in v11_motif_figures if href],
        "v11_structure_overlay_href": choose_existing_artifact(outdir, [V11_STRUCTURE_OVERLAY_HTML]),
        "v11_grouped_structure_overlay_href": choose_existing_artifact(outdir, [V11_GROUPED_STRUCTURE_OVERLAY_HTML]),
        # Needed by _v11_overlay_iframe_markup to inline the overlay HTML
        # into iframe srcdoc so the report stays self-contained when shared.
        "output_directory": str(outdir.resolve()),
    }


def build_alignment_browser_payload(alignment_species_df: pd.DataFrame,
                                    run_meta: Dict[str, Any],
                                    tree_viewer_data: Optional[Dict[str, Any]] = None,
                                    outdir: Optional[Path] = None) -> Dict[str, Any]:
    scopes: Dict[str, Any] = {}
    reference_species = clean_json_value(run_meta.get("reference_species"))
    gene_symbol = clean_json_value(run_meta.get("gene_symbol"))
    alphafold_structure: Dict[str, Any] = {
        "available": False,
        "reason": "Output directory was not provided for AlphaFold structure payload construction.",
        "colors": ALPHAFOLD_STRUCTURE_COLORS,
        "ranges": [],
    }
    alphafold_viewer_js = ""
    if outdir is not None:
        alphafold_structure, alphafold_viewer_js = build_alphafold_structure_payload(
            outdir,
            {name: load_output_table(outdir, filename, sep) for name, (filename, sep) in CSV_TABLE_SPECS.items()},
            str(reference_species) if reference_species else None,
        )
    node_conservation = {
        "paper_tree_svg_path": choose_existing_artifact(outdir, [NODE_CONSERVATION_PAPER_TREE_SVG_FILENAME]) if outdir is not None else None,
        "detailed_tree_svg_path": choose_existing_artifact(outdir, [NODE_CONSERVATION_TREE_SVG_FILENAME]) if outdir is not None else None,
        "csv_path": choose_existing_artifact(outdir, [NODE_CONSERVATION_EXTREMES_FILENAME]) if outdir is not None else None,
    }
    # Inline the paper-tree SVG as a data URI so the rendered report renders
    # the figure without needing the sibling SVG file on disk (lets the
    # user email / zip just interactive_report.html and have it work).
    # Detailed tree gets the same treatment so the "Open detailed SVG" link
    # is also self-contained.
    if outdir is not None:
        for path_key, uri_key in (
            ("paper_tree_svg_path", "paper_tree_svg_data_uri"),
            ("detailed_tree_svg_path", "detailed_tree_svg_data_uri"),
        ):
            rel = node_conservation.get(path_key)
            if not rel:
                continue
            svg_path = Path(outdir) / rel
            if not svg_path.exists():
                continue
            try:
                svg_text = svg_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            b64 = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
            node_conservation[uri_key] = f"data:image/svg+xml;base64,{b64}"
    reference_landmarks = load_alignment_browser_reference_landmarks(
        outdir,
        str(reference_species) if reference_species else None,
        str(gene_symbol) if gene_symbol else None,
    )
    browser_columns = [
        "alignment_scope", "scope_label", "source_fasta", "record_index", "record_id",
        "species", "symbol", "protein_record_id", "taxonomy_level", "clade", "phylum", "broad_clade", "clade_age_mya", "clade_age_source",
        "species_display_label", "scientific_name", "common_name", "preferred_public_label",
        "preferred_public_gene_label", "preferred_protein_name", "nomenclature_leaf_label",
        "uniprot_accession", "reviewed_status", "alphafold_entry_id", "alphafold_source_label",
        "clade_order_rank", "tree_order", "tree_order_source", "is_reference",
        "aligned_length", "ungapped_length", "gap_count", "gap_fraction",
        "comparable_to_reference_count", "match_to_reference_count",
        "mismatch_to_reference_count", "identity_to_reference", "aligned_sequence",
        "reference_species", "reference_record_id",
    ]
    for scope, spec in ALIGNMENT_BROWSER_SCOPES.items():
        sub = alignment_species_df[alignment_species_df["alignment_scope"] == scope].copy() if not alignment_species_df.empty else pd.DataFrame()
        if sub.empty:
            continue
        sub = sub.sort_values(["tree_order", "record_index"], kind="stable")
        ref_rows = sub[sub["is_reference"].astype(bool)] if "is_reference" in sub.columns else pd.DataFrame()
        reference_row = ref_rows.iloc[0] if not ref_rows.empty else sub.iloc[0]
        reference_sequence = str(reference_row.get("aligned_sequence") or "")
        reference_positions: List[Optional[int]] = []
        reference_residues: List[str] = []
        ref_position = 0
        for aa in reference_sequence:
            aa_upper = aa.upper()
            reference_residues.append(aa_upper)
            if aa_upper not in GAP_CHARS:
                ref_position += 1
                reference_positions.append(ref_position)
            else:
                reference_positions.append(None)

        scopes[scope] = {
            "label": spec["label"],
            "source_fasta": spec["filename"],
            "alignment_length": len(reference_sequence),
            "reference_species": clean_json_value(reference_row.get("species")),
            "reference_record_id": clean_json_value(reference_row.get("record_id")),
            "reference_sequence": reference_sequence,
            "reference_positions": reference_positions,
            "reference_residues": reference_residues,
            "reference_landmarks": clean_json_value(reference_landmarks),
            "records": dataframe_to_json_records(filter_columns(sub, browser_columns)),
            "evolutionary_divergence": evolutionary_divergence_figure_artifacts(outdir, scope),
        }

    return {
        "title": f"{run_meta.get('gene_symbol') or 'Alignment'} interactive alignment browser",
        "meta": clean_json_value(run_meta),
        "aa_colors": AA_COLORS,
        "browser_default_state": clean_json_value(ALIGNMENT_BROWSER_DEFAULT_STATE),
        "clade_order": CLADE_ORDER,
        "scopes": scopes,
        "tree": clean_json_value(tree_viewer_data or {"available": False}),
        "node_conservation": clean_json_value(node_conservation),
        "alphafold_structure": clean_json_value(alphafold_structure),
        "alphafold_viewer_js": alphafold_viewer_js,
        # Per-clade 3D structure overlays (3Dmol.js); embedded in the alignment
        # browser alongside the per-residue AlphaFold range viewer.
        "v11_structure_overlay_href": choose_existing_artifact(outdir, [V11_STRUCTURE_OVERLAY_HTML]) if outdir is not None else None,
        "v11_grouped_structure_overlay_href": choose_existing_artifact(outdir, [V11_GROUPED_STRUCTURE_OVERLAY_HTML]) if outdir is not None else None,
        "v11_mod_structure_overlay_href": choose_existing_artifact(outdir, [V11_MOD_STRUCTURE_OVERLAY_HTML]) if outdir is not None else None,
        "v11_combined_structure_overlay_href": choose_existing_artifact(outdir, [V11_COMBINED_STRUCTURE_OVERLAY_HTML]) if outdir is not None else None,
        "v11_clade_identity_bubble_pdf_href": choose_existing_artifact(outdir, [V11_CLADE_IDENTITY_BUBBLE_PDF]) if outdir is not None else None,
        "v11_grouped_clade_identity_bubble_pdf_href": choose_existing_artifact(outdir, [V11_GROUPED_CLADE_IDENTITY_BUBBLE_PDF]) if outdir is not None else None,
        "v11_mod_clade_identity_bubble_pdf_href": choose_existing_artifact(outdir, [V11_MOD_CLADE_IDENTITY_BUBBLE_PDF]) if outdir is not None else None,
        "v11_per_clade_ss_csv_path": (outdir / V11_PER_CLADE_SS_CSV) if (outdir is not None and (outdir / V11_PER_CLADE_SS_CSV).exists()) else None,
        # Needed by v11_clade_overlay_panel_markup to inline each overlay HTML
        # into iframe srcdoc so the alignment browser stays self-contained.
        "output_directory": str(outdir.resolve()) if outdir is not None else None,
    }


def browser_is_informative_residue(aa: Any) -> bool:
    residue = str(aa or "").upper()
    return bool(residue) and residue not in NON_CONSENSUS_RESIDUES


def browser_residue_property_key(aa: Any, scheme: Dict[str, set[str]]) -> Optional[str]:
    residue = str(aa or "").upper()
    for key, residues in scheme.items():
        if residue in residues:
            return key
    return None


def browser_same_property_as_reference(ref: Any, aa: Any) -> bool:
    if not browser_is_informative_residue(ref) or not browser_is_informative_residue(aa):
        return False
    for scheme in PROPERTY_GROUPS.values():
        ref_key = browser_residue_property_key(ref, scheme)
        aa_key = browser_residue_property_key(aa, scheme)
        if ref_key and aa_key and ref_key == aa_key:
            return True
    return False


def browser_similar_to_reference(aa: Any, ref: Any, mode: str) -> bool:
    if not browser_is_informative_residue(aa) or not browser_is_informative_residue(ref):
        return False
    aa_upper = str(aa).upper()
    ref_upper = str(ref).upper()
    if mode == "exact":
        return aa_upper == ref_upper
    if mode == "property":
        return aa_upper == ref_upper or browser_same_property_as_reference(ref_upper, aa_upper)
    return False


def browser_consensus_source_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    without_reference = [row for row in rows if not bool(row.get("is_reference"))]
    return without_reference or list(rows)


def browser_choose_major_residue(counts: Dict[str, int], ref_residue: Any) -> Tuple[str, int]:
    if not counts:
        return "-", 0
    ref_upper = str(ref_residue or "").upper()
    entries = sorted(
        counts.items(),
        key=lambda item: (-item[1], 0 if item[0] == ref_upper else 1, item[0]),
    )
    return entries[0]


def browser_residue_at(row: Dict[str, Any], idx: int) -> str:
    seq = str(row.get("aligned_sequence") or "").upper()
    if idx < 0 or idx >= len(seq):
        return ""
    return seq[idx] or ""


def browser_build_consensus(rows: Sequence[Dict[str, Any]],
                            scope: Dict[str, Any],
                            columns: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    source_rows = browser_consensus_source_rows(rows)
    reference_residues = scope.get("reference_residues") or []
    cells: Dict[int, Dict[str, Any]] = {}
    for idx in columns:
        ref_residue = reference_residues[idx] if idx < len(reference_residues) else ""
        informative_counts: Dict[str, int] = {}
        fallback_counts: Dict[str, int] = {}
        for row in source_rows:
            aa = browser_residue_at(row, idx) or "-"
            fallback_counts[aa] = fallback_counts.get(aa, 0) + 1
            if browser_is_informative_residue(aa):
                informative_counts[aa] = informative_counts.get(aa, 0) + 1
        informative_total = sum(informative_counts.values())
        fallback_total = sum(fallback_counts.values())
        use_fallback = informative_total == 0
        major_aa, major_count = browser_choose_major_residue(
            fallback_counts if use_fallback else informative_counts,
            ref_residue,
        )
        denominator = fallback_total if use_fallback else informative_total
        cells[idx] = {
            "aa": major_aa,
            "count": major_count,
            "denominator": denominator,
            "informativeTotal": informative_total,
            "rowCount": len(source_rows),
            "support": (major_count / denominator) if denominator else None,
            "fromFallback": use_fallback,
        }
    return cells


def browser_compare_windows(consensus_cells: Dict[int, Dict[str, Any]],
                            scope: Dict[str, Any],
                            columns: Sequence[int],
                            mode: str,
                            min_run: int) -> Dict[str, Any]:
    reference_residues = scope.get("reference_residues") or []
    reference_positions = scope.get("reference_positions") or []
    highlighted: set[int] = set()
    runs: List[Dict[str, Any]] = []
    run: List[int] = []
    column_ordinals = {idx: ordinal for ordinal, idx in enumerate(columns)}
    min_run = max(1, int(min_run or 1))

    def flush_run() -> None:
        nonlocal run
        if len(run) >= min_run:
            highlighted.update(run)
            start_idx = run[0]
            end_idx = run[-1]
            runs.append({
                "startIdx": start_idx,
                "endIdx": end_idx,
                "startRef": reference_positions[start_idx] if start_idx < len(reference_positions) else None,
                "endRef": reference_positions[end_idx] if end_idx < len(reference_positions) else None,
                "startVisibleOrdinal": column_ordinals.get(start_idx, 0),
                "endVisibleOrdinal": column_ordinals.get(end_idx, 0),
                "length": len(run),
            })
        run = []

    if mode == "off":
        return {"columns": highlighted, "runs": runs, "runCount": 0}

    for idx in columns:
        consensus = consensus_cells.get(idx) or {}
        aa = consensus.get("aa") or ""
        ref = reference_residues[idx] if idx < len(reference_residues) else ""
        previous = run[-1] if run else None
        contiguous = previous is None or idx == previous + 1
        similar = contiguous and browser_similar_to_reference(aa, ref, mode)
        if similar:
            run.append(idx)
        else:
            flush_run()
            if browser_similar_to_reference(aa, ref, mode):
                run.append(idx)
    flush_run()
    return {"columns": highlighted, "runs": runs, "runCount": len(runs)}


def browser_reference_track_length(scope: Dict[str, Any]) -> int:
    positions = [
        int(value)
        for value in scope.get("reference_positions", [])
        if value is not None and not pd.isna(value)
    ]
    if positions:
        return max(positions)
    return int(scope.get("alignment_length") or 0)


def browser_format_group_label(value: Any) -> str:
    return str(value or "Unassigned").replace("_", " ")


def browser_group_rows(rows: Sequence[Dict[str, Any]],
                       group_field: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(group_field) or "Unassigned")
        groups.setdefault(key, []).append(row)
    grouped = [
        {"key": key, "label": browser_format_group_label(key), "rows": group_rows}
        for key, group_rows in groups.items()
    ]
    grouped.sort(key=lambda group: min(float(row.get("tree_order") or 999999) for row in group["rows"]))
    return grouped


def browser_default_scope(payload: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    scopes = payload.get("scopes") or {}
    preferred = str(ALIGNMENT_BROWSER_DEFAULT_STATE["scope"])
    if preferred in scopes:
        return preferred, scopes[preferred]
    for key, scope in scopes.items():
        return str(key), scope
    return preferred, None


def build_alignment_browser_architecture_svg(title: str,
                                             subtitle: str,
                                             groups: Sequence[Dict[str, Any]],
                                             track_length: int,
                                             compare_mode: str,
                                             landmarks: Sequence[Dict[str, Any]],
                                             secondary_structure: Optional[Dict[str, Any]] = None,
                                             local_charge: Optional[Dict[str, Any]] = None,
                                             calcium_binding: Optional[Dict[str, Any]] = None) -> str:
    active_groups = [group for group in groups if group.get("label")]
    landmark_rows = alignment_browser_landmark_rows(landmarks)
    secondary_ranges = architecture_secondary_structure_ranges(secondary_structure, track_length)
    charge_rows = architecture_local_charge_rows(local_charge, track_length)
    calcium_payload = architecture_calcium_payload(calcium_binding, track_length)
    has_calcium = bool(calcium_payload.get("loops") or calcium_payload.get("ligands"))
    label_width = 230
    track_width = 1180
    count_width = 90
    margin = 18
    header_height = 76
    annotation_row_height = 30
    row_height = 28
    width = label_width + track_width + count_width + margin * 2
    annotation_rows = 1 + (1 if secondary_ranges else 0) + (1 if charge_rows else 0) + (1 if has_calcium else 0) + len(landmark_rows)
    height = header_height + annotation_row_height * annotation_rows + row_height * max(len(active_groups), 1) + margin
    run_color = "#7c3aed" if compare_mode == "property" else "#2563eb"
    track_x = margin + label_width
    count_x = track_x + track_width + 12
    ruler_y = header_height
    extra_annotation_count = (1 if secondary_ranges else 0) + (1 if charge_rows else 0) + (1 if has_calcium else 0)
    domain_rows = [
        {**landmark_row, "y": ruler_y + annotation_row_height * (index + 1 + extra_annotation_count)}
        for index, landmark_row in enumerate(landmark_rows)
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<style>"
        ".title{font:700 18px Segoe UI,Tahoma,sans-serif;fill:#18202a}"
        ".subtitle{font:600 12px Segoe UI,Tahoma,sans-serif;fill:#617083}"
        ".label{font:700 13px Segoe UI,Tahoma,sans-serif;fill:#18202a}"
        ".count{font:700 12px Segoe UI,Tahoma,sans-serif;fill:#617083}"
        ".track{fill:#e5e7eb;stroke:#d0d7e2;stroke-width:1}"
        ".anno-track{fill:#eef2f7;stroke:#d0d7e2;stroke-width:1}"
        ".anno-tick{stroke:#64748b;stroke-width:1}"
        ".anno-tick-label{font:700 11px Segoe UI,Tahoma,sans-serif;fill:#475569}"
        ".anno-band-label{font:700 11px Segoe UI,Tahoma,sans-serif;fill:#ffffff}"
        ".anno-site-label{font:800 9px Segoe UI,Tahoma,sans-serif;paint-order:stroke;stroke:#ffffff;stroke-width:2px;stroke-linejoin:round}"
        ".ss-loop-line{stroke:#9ca3af;stroke-width:4;stroke-linecap:round;opacity:.72}"
        ".ss-helix-line{fill:none;stroke-width:4;stroke-linecap:round;stroke-linejoin:round}"
        ".ss-sheet-arrow{opacity:.9}"
        ".ss-feature-label{font:800 8px Segoe UI,Tahoma,sans-serif;paint-order:stroke;stroke:#ffffff;stroke-width:2px;stroke-linejoin:round}"
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="{margin}" y="24">{escape(title)}</text>',
        f'<text class="subtitle" x="{margin}" y="41">{escape(subtitle)}</text>',
        # V11: the specific domain ruler below is PLA2G4A coordinates; only show
        # it for PLA2G4A. Other genes get a neutral note (domains come from the
        # gene's own domains.tsv landmarks rendered just below).
        (f'<text class="subtitle" x="{margin}" y="58">Homo sapiens ruler annotations: PL 1-178, C2 6-122, PLA2c 140-740, PLLLLTP 263-269, DELD 519-522.</text>'
         if v11_is_pla2g4a() else
         f'<text class="subtitle" x="{margin}" y="58">Reference ({escape(v11_gene_label())}) domain landmarks below are derived from the gene\'s own UniProt/InterPro annotations.</text>'),
    ]
    parts.append(f'<text class="label" x="{margin}" y="{ruler_y + 17}">Reference ruler</text>')
    parts.append(f'<text class="count" x="{count_x}" y="{ruler_y + 17}">1-{track_length}</text>')
    parts.extend(architecture_svg_landmarks(track_x, track_width, track_length, ruler_y, domain_rows))
    if secondary_ranges:
        secondary_y = ruler_y + annotation_row_height
        parts.append(f'<text class="label" x="{margin}" y="{secondary_y + 17}">AlphaFold SS</text>')
        parts.append(f'<text class="count" x="{count_x}" y="{secondary_y + 17}">helix/sheet/loop</text>')
        parts.extend(architecture_svg_secondary_structure(track_x, track_width, track_length, secondary_y, secondary_ranges))
    charge_y = ruler_y + annotation_row_height * (1 + (1 if secondary_ranges else 0))
    if charge_rows:
        parts.append(f'<text class="label" x="{margin}" y="{charge_y + 17}">5-aa charge</text>')
        parts.append(f'<text class="count" x="{count_x}" y="{charge_y + 17}">negative / neutral / positive</text>')
        parts.extend(architecture_svg_local_charge(track_x, track_width, track_length, charge_y, charge_rows))
    calcium_y = charge_y + annotation_row_height * (1 if charge_rows else 0)
    if has_calcium:
        parts.append(f'<text class="label" x="{margin}" y="{calcium_y + 17}">Ca2+ binding</text>')
        parts.append(f'<text class="count" x="{count_x}" y="{calcium_y + 17}">CBR1-3 + Ca/PC contacts</text>')
        parts.extend(architecture_svg_calcium(track_x, track_width, track_length, calcium_y, calcium_payload))
    for row in domain_rows:
        y = float(row["y"])
        parts.append(f'<text class="label" x="{margin}" y="{y + 17}">{escape(str(row["label"]))}</text>')
        parts.append(f'<text class="count" x="{count_x}" y="{y + 17}">{escape(str(row["range_label"]))}</text>')
    group_base_y = header_height + annotation_row_height * annotation_rows + 4
    for index, group in enumerate(active_groups):
        y = group_base_y + index * row_height
        label = str(group.get("label") or "")
        runs = group.get("runs") or []
        count_label = f"{len(runs)} run" + ("" if len(runs) == 1 else "s")
        parts.append(f'<text class="label" x="{margin}" y="{y + 17}">{escape(label)}</text>')
        parts.append(f'<rect class="track" x="{track_x}" y="{y + 6}" width="{track_width}" height="15" rx="3"/>')
        for run in runs:
            start = run.get("startRef")
            end = run.get("endRef")
            if start is None or end is None:
                start = int(run.get("startIdx", 0)) + 1
                end = int(run.get("endIdx", 0)) + 1
            start_num = int(start)
            end_num = int(end)
            x = track_x + max(0.0, ((start_num - 1) / max(track_length, 1)) * track_width)
            rect_width = max(1.5, ((end_num - start_num + 1) / max(track_length, 1)) * track_width)
            range_label = f"{start_num}-{end_num}"
            parts.append(
                f'<rect x="{x:.3f}" y="{y + 6}" width="{rect_width:.3f}" height="15" rx="2" '
                f'fill="{run_color}"><title>{escape(label)} {escape(range_label)}</title></rect>'
            )
        parts.append(f'<text class="count" x="{count_x}" y="{y + 17}">{escape(count_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build_alignment_browser_default_exports(alignment_species_df: pd.DataFrame,
                                            run_meta: Dict[str, Any],
                                            outdir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, str, Dict[str, Any]]:
    payload = build_alignment_browser_payload(alignment_species_df, run_meta, outdir=outdir)
    scope_key, scope = browser_default_scope(payload)
    group_field = str(ALIGNMENT_BROWSER_DEFAULT_STATE["group"])
    compare_mode = str(ALIGNMENT_BROWSER_DEFAULT_STATE["compare"])
    min_run = int(ALIGNMENT_BROWSER_DEFAULT_STATE["min_similar_run"])

    run_columns = [
        "output_name", "gene_symbol", "preset_name", "alignment_scope", "scope_label",
        "group_field", "group_label", "group_key", "group_order", "group_row_count",
        "compare_mode", "min_similar_run", "run_number", "start_reference_position",
        "end_reference_position", "start_alignment_position", "end_alignment_position",
        "length_alignment_columns", "start_visible_column", "end_visible_column",
    ]
    consensus_columns = [
        "output_name", "gene_symbol", "preset_name", "alignment_scope", "scope_label",
        "group_field", "group_label", "group_key", "group_order", "group_row_count",
        "alignment_position", "reference_position", "reference_residue",
        "consensus_residue", "major_residue_count", "support_denominator",
        "informative_residue_count", "support_fraction", "support_percent",
        "used_fallback_counts", "similar_to_human_run", "similar_run_number",
        "similar_run_start_reference_position", "similar_run_end_reference_position",
    ]
    if scope is None:
        return pd.DataFrame(columns=run_columns), pd.DataFrame(columns=consensus_columns), "", {}

    columns = list(range(int(scope.get("alignment_length") or 0)))
    records = sorted(
        scope.get("records") or [],
        key=lambda row: (float(row.get("tree_order") or 999999), int(row.get("record_index") or 0)),
    )
    groups = browser_group_rows(records, group_field)
    reference_positions = scope.get("reference_positions") or []
    reference_residues = scope.get("reference_residues") or []
    preset_name = "taxa_compressed_exact_human_min6"
    common = {
        "output_name": run_meta.get("output_name"),
        "gene_symbol": run_meta.get("gene_symbol"),
        "preset_name": preset_name,
        "alignment_scope": scope_key,
        "scope_label": scope.get("label"),
        "group_field": group_field,
        "compare_mode": compare_mode,
        "min_similar_run": min_run,
    }

    run_rows: List[Dict[str, Any]] = []
    consensus_rows: List[Dict[str, Any]] = []
    svg_groups: List[Dict[str, Any]] = []
    for group_order, group in enumerate(groups, start=1):
        consensus_cells = browser_build_consensus(group["rows"], scope, columns)
        windows = browser_compare_windows(consensus_cells, scope, columns, compare_mode, min_run)
        run_by_idx: Dict[int, Dict[str, Any]] = {}
        for run_number, run in enumerate(windows["runs"], start=1):
            for idx in range(int(run["startIdx"]), int(run["endIdx"]) + 1):
                run_by_idx[idx] = {**run, "runNumber": run_number}
            run_rows.append({
                **common,
                "group_label": group["label"],
                "group_key": group["key"],
                "group_order": group_order,
                "group_row_count": len(group["rows"]),
                "run_number": run_number,
                "start_reference_position": run.get("startRef"),
                "end_reference_position": run.get("endRef"),
                "start_alignment_position": int(run["startIdx"]) + 1,
                "end_alignment_position": int(run["endIdx"]) + 1,
                "length_alignment_columns": run.get("length"),
                "start_visible_column": int(run.get("startVisibleOrdinal") or 0) + 1,
                "end_visible_column": int(run.get("endVisibleOrdinal") or 0) + 1,
            })
        for idx in columns:
            cell = consensus_cells.get(idx) or {}
            support = cell.get("support")
            similar_run = run_by_idx.get(idx)
            consensus_rows.append({
                **common,
                "group_label": group["label"],
                "group_key": group["key"],
                "group_order": group_order,
                "group_row_count": len(group["rows"]),
                "alignment_position": idx + 1,
                "reference_position": reference_positions[idx] if idx < len(reference_positions) else None,
                "reference_residue": reference_residues[idx] if idx < len(reference_residues) else None,
                "consensus_residue": cell.get("aa"),
                "major_residue_count": cell.get("count"),
                "support_denominator": cell.get("denominator"),
                "informative_residue_count": cell.get("informativeTotal"),
                "support_fraction": support,
                "support_percent": (support * 100) if support is not None else None,
                "used_fallback_counts": bool(cell.get("fromFallback")),
                "similar_to_human_run": similar_run is not None,
                "similar_run_number": similar_run.get("runNumber") if similar_run else None,
                "similar_run_start_reference_position": similar_run.get("startRef") if similar_run else None,
                "similar_run_end_reference_position": similar_run.get("endRef") if similar_run else None,
            })
        svg_groups.append({**group, "runs": windows["runs"]})

    track_length = browser_reference_track_length(scope)
    landmarks = scope.get("reference_landmarks") or []
    title = f"{run_meta.get('gene_symbol') or 'Alignment'} default similar-run architecture"
    subtitle = (
        f"{browser_format_group_label(group_field)} groups, {compare_mode}, "
        f"min run {min_run}, {track_length} reference indices"
    )
    svg = build_alignment_browser_architecture_svg(
        title=title,
        subtitle=subtitle,
        groups=svg_groups,
        track_length=track_length,
        compare_mode=compare_mode,
        landmarks=landmarks,
        secondary_structure=(payload.get("alphafold_structure") or {}).get("secondary_structure"),
        local_charge=(payload.get("alphafold_structure") or {}).get("local_charge"),
        calcium_binding=(payload.get("alphafold_structure") or {}).get("calcium_binding"),
    )
    plot_spec = {
        "title": title,
        "subtitle": subtitle,
        "groups": svg_groups,
        "track_length": track_length,
        "compare_mode": compare_mode,
        "landmarks": landmarks,
        "secondary_structure": (payload.get("alphafold_structure") or {}).get("secondary_structure"),
        "local_charge": (payload.get("alphafold_structure") or {}).get("local_charge"),
        "calcium_binding": (payload.get("alphafold_structure") or {}).get("calcium_binding"),
    }
    return pd.DataFrame(run_rows, columns=run_columns), pd.DataFrame(consensus_rows, columns=consensus_columns), svg, plot_spec


def write_alignment_browser_default_exports(outdir: Path,
                                            alignment_species_df: pd.DataFrame,
                                            run_meta: Dict[str, Any]) -> List[Path]:
    runs_df, consensus_df, svg, plot_spec = build_alignment_browser_default_exports(alignment_species_df, run_meta, outdir=outdir)
    paths = [
        outdir / f"{ALIGNMENT_BROWSER_DEFAULT_EXPORT_PREFIX}_runs.csv",
        outdir / f"{ALIGNMENT_BROWSER_DEFAULT_EXPORT_PREFIX}_consensus.csv",
    ]
    runs_df.to_csv(paths[0], index=False)
    consensus_df.to_csv(paths[1], index=False)
    if svg:
        svg_path = outdir / f"{ALIGNMENT_BROWSER_DEFAULT_EXPORT_PREFIX}_architecture.svg"
        svg_path.write_text(svg, encoding="utf-8")
        paths.append(svg_path)
    if plot_spec:
        png_path = outdir / f"{ALIGNMENT_BROWSER_DEFAULT_EXPORT_PREFIX}_architecture.png"
        plot_alignment_browser_architecture_png(
            png_path,
            str(plot_spec.get("title") or ""),
            str(plot_spec.get("subtitle") or ""),
            plot_spec.get("groups") or [],
            int(plot_spec.get("track_length") or 0),
            str(plot_spec.get("compare_mode") or "exact"),
            plot_spec.get("landmarks") or [],
            plot_spec.get("secondary_structure") or None,
            plot_spec.get("local_charge") or None,
            plot_spec.get("calcium_binding") or None,
        )
        paths.append(png_path)
    return paths


def alphafold_structure_css() -> str:
    return r"""
    .alphafold-structure-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      margin-bottom: 12px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    .alphafold-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 12px;
    }
    .alphafold-title h2 {
      margin: 0;
      font-size: 1.18rem;
      font-weight: 800;
    }
    .alphafold-title p {
      margin: 5px 0 0;
      color: var(--muted);
      line-height: 1.45;
    }
    .alphafold-model-note {
      margin: 0 0 12px;
      padding: 10px 12px;
      border: 1px solid #fed7aa;
      border-radius: 10px;
      background: #fff7ed;
      color: #7c2d12;
      font-size: 0.88rem;
      line-height: 1.45;
      font-weight: 650;
    }
    .alphafold-badges,
    .alphafold-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .alphafold-badge,
    .alphafold-legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 5px 9px;
      background: #eef3f8;
      color: #334155;
      font-size: 0.8rem;
      font-weight: 750;
    }
    .alphafold-swatch {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      border: 1px solid rgba(15, 23, 42, 0.15);
      flex: 0 0 auto;
    }
    .alphafold-grid {
      display: grid;
      grid-template-columns: minmax(320px, 0.95fr) minmax(360px, 1.05fr);
      gap: 14px;
      align-items: stretch;
    }
    .alphafold-viewer-card,
    .alphafold-controls-card {
      border: 1px solid #dbe4ee;
      border-radius: 12px;
      background: #fff;
      overflow: hidden;
    }
    .alphafold-viewer-canvas {
      min-height: 420px;
      height: 48vh;
      max-height: 640px;
      background: linear-gradient(135deg, #f8fafc, #eef2f7);
      position: relative;
    }
    .alphafold-viewer-message {
      min-height: 220px;
      display: grid;
      place-items: center;
      padding: 24px;
      color: var(--muted);
      text-align: center;
      line-height: 1.5;
      font-weight: 700;
    }
    .alphafold-track-wrap {
      border-top: 1px solid #e5edf6;
      padding: 10px;
      background: #fbfdff;
      overflow-x: auto;
    }
    .alphafold-track-svg {
      display: block;
      width: 100%;
      min-width: 720px;
      height: auto;
    }
    .alphafold-controls-card {
      padding: 12px;
      display: grid;
      grid-template-rows: auto auto auto auto minmax(160px, 1fr);
      gap: 10px;
    }
    .alphafold-control-row,
    .alphafold-render-row {
      display: grid;
      gap: 8px;
      align-items: end;
    }
    .alphafold-control-row {
      grid-template-columns: minmax(0, 1fr) auto auto auto;
    }
    .alphafold-render-row {
      grid-template-columns: minmax(0, 1fr) auto auto;
    }
    .alphafold-control-row label,
    .alphafold-render-row label {
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0;
      display: grid;
      gap: 5px;
    }
    .alphafold-control-row input,
    .alphafold-control-row select,
    .alphafold-render-row select {
      width: 100%;
      min-height: 36px;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      padding: 7px 9px;
      font: inherit;
      font-size: 0.92rem;
      background: #fff;
      color: var(--ink);
    }
    .alphafold-button {
      min-height: 36px;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      background: #f8fafc;
      color: var(--ink);
      padding: 7px 10px;
      font: inherit;
      font-weight: 750;
      cursor: pointer;
      white-space: nowrap;
    }
    .alphafold-button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    .alphafold-status {
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.4;
      padding: 8px 10px;
      border-radius: 10px;
      background: #f8fafc;
      border: 1px solid #e5edf6;
    }
    .alphafold-range-list {
      border: 1px solid #dbe4ee;
      border-radius: 10px;
      overflow: auto;
      max-height: 360px;
      background: #fff;
    }
    .alphafold-range-panel {
      border: 1px solid #dbe4ee;
      border-radius: 10px;
      overflow: hidden;
      background: #fff;
    }
    .alphafold-range-panel.collapsed .alphafold-range-list {
      display: none;
    }
    .alphafold-range-panel.collapsed {
      background: #f8fafc;
    }
    .alphafold-range-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      padding: 9px 10px;
      border-bottom: 1px solid #eef2f6;
    }
    .alphafold-range-panel.collapsed .alphafold-range-head {
      border-bottom: 0;
    }
    .alphafold-range-head strong {
      color: var(--ink);
      font-size: 0.9rem;
      font-weight: 850;
    }
    .alphafold-range-count {
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 750;
    }
    .alphafold-range-toggle {
      border: 1px solid #d8dee8;
      border-radius: 8px;
      background: #fff;
      color: var(--accent-strong);
      cursor: pointer;
      font: inherit;
      font-size: 0.78rem;
      font-weight: 800;
      padding: 5px 8px;
      white-space: nowrap;
    }
    .alphafold-range-toggle:hover {
      border-color: var(--accent);
      background: #ecfdf5;
    }
    .alphafold-range-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      width: 100%;
      padding: 9px 10px;
      border: 0;
      border-bottom: 1px solid #eef2f6;
      background: #fff;
      text-align: left;
      font: inherit;
      cursor: pointer;
    }
    .alphafold-range-row:hover,
    .alphafold-range-row.selected {
      background: #fffbeb;
    }
    .alphafold-range-row:last-child {
      border-bottom: 0;
    }
    .alphafold-range-main {
      min-width: 0;
    }
    .alphafold-range-title {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      font-weight: 800;
      color: var(--ink);
    }
    .alphafold-range-title span:last-child {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .alphafold-range-meta {
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 650;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .alphafold-range-coords {
      color: #334155;
      font-weight: 850;
      font-variant-numeric: tabular-nums;
    }
    .alphafold-empty {
      padding: 12px;
      color: var(--muted);
      font-weight: 750;
    }
    @media (max-width: 1050px) {
      .alphafold-grid {
        grid-template-columns: minmax(0, 1fr);
      }
      .alphafold-control-row {
        grid-template-columns: minmax(0, 1fr);
      }
      .alphafold-render-row {
        grid-template-columns: minmax(0, 1fr);
      }
      .alphafold-head {
        flex-direction: column;
      }
    }
"""


def alphafold_structure_panel_markup() -> str:
    return '<section class="alphafold-structure-panel" id="alphafold-structure-panel"></section>'


def v11_per_clade_ss_panel_markup(per_clade_ss_csv_path: Optional[Path]) -> str:
    """Render the per-clade consensus secondary structure as an
    alignment-browser-style grid: one row per clade, one cell per reference
    position, cell colour = consensus SS (helix orange, sheet purple, loop
    grey), cell opacity = consensus fraction (faint where the clade is split
    between SS states; bright where ≥80% agree). Empty / missing data render
    as a dim slash.

    Data source: v11_per_clade_secondary_structure.csv produced by
    v11_compute_per_clade_secondary_structure. If the CSV is missing or empty
    returns an empty string so the panel just isn't rendered.
    """
    if not per_clade_ss_csv_path or not Path(per_clade_ss_csv_path).exists():
        return ""
    try:
        df = pd.read_csv(per_clade_ss_csv_path)
    except Exception:  # noqa: BLE001
        return ""
    required = {"reference_position", "clade", "consensus_ss", "consensus_frac", "n_species"}
    if df.empty or not required.issubset(df.columns):
        return ""

    # Order clades the same way the bubble grid does (mammals → most basal).
    clades = sorted(df["clade"].dropna().unique().tolist(), key=_v11_bubble_clade_key)
    track_len = int(df["reference_position"].max())
    if not clades or track_len <= 0:
        return ""

    # Index for fast cell lookup.
    cell: Dict[Tuple[str, int], Tuple[str, float, int]] = {}
    for _, row in df.iterrows():
        clade = str(row["clade"])
        pos = int(row["reference_position"])
        ss = str(row.get("consensus_ss") or "loop")
        frac = float(row.get("consensus_frac") or 0.0)
        n = int(row.get("n_species") or 0)
        cell[(clade, pos)] = (ss, frac, n)

    SS_COLORS = {"helix": "#e88a22", "sheet": "#8d6af1", "loop": "#b8bec7"}
    SS_LETTERS = {"helix": "H", "sheet": "E", "loop": "C"}

    # Limit DOM size — wrap into blocks of 60 residues per row (matches the
    # bubble grid's pagination) so the panel stays readable for long proteins
    # without producing a 12000-cell single row.
    BLOCK = 60
    n_blocks = (track_len + BLOCK - 1) // BLOCK

    pieces: List[str] = []
    pieces.append('<section class="alphafold-structure-panel v11-per-clade-ss-panel" id="v11-per-clade-ss-panel" style="margin-top:14px;">')
    pieces.append('<details open><summary style="cursor:pointer;font-weight:700;font-size:0.95rem;">'
                  'Per-clade consensus secondary structure (per reference position)'
                  '</summary>')
    pieces.append('<p class="muted" style="margin:8px 0 8px 0;font-size:0.85rem;">'
                  'For every represented clade, the majority secondary-structure state at '
                  'each reference residue: '
                  '<span style="display:inline-block;width:10px;height:10px;background:#e88a22;border:1px solid #c8c8c8;"></span> helix (H), '
                  '<span style="display:inline-block;width:10px;height:10px;background:#8d6af1;border:1px solid #c8c8c8;"></span> sheet (E), '
                  '<span style="display:inline-block;width:10px;height:10px;background:#b8bec7;border:1px solid #c8c8c8;"></span> loop (C). '
                  'Cell opacity = consensus fraction; bright cells mean the clade strongly '
                  'agrees on that SS state, dim cells mean the clade is split. Slash = no '
                  'AlphaFold coverage for this clade at that position.'
                  '</p>')
    pieces.append('<style>'
                  '.v11-ss-grid{font-family:Consolas,monospace;font-size:10px;line-height:1;border-spacing:0;border-collapse:collapse;}'
                  '.v11-ss-grid th{padding:2px 6px;text-align:left;color:var(--muted);font-weight:600;font-size:11px;white-space:nowrap;}'
                  '.v11-ss-grid td.v11-ss-cell{width:11px;height:14px;text-align:center;color:#1a1a1a;}'
                  '.v11-ss-grid td.v11-ss-cart{padding:0;height:18px;width:11px;background:#fafbfc;}'
                  '.v11-ss-grid td.v11-ss-cart svg{display:block;width:100%;height:100%;}'
                  '.v11-ss-grid td.v11-ss-tick{font-size:9px;color:#888;padding-top:2px;}'
                  '.v11-ss-grid .v11-ss-blockwrap{margin-bottom:14px;overflow-x:auto;}'
                  '</style>')

    CELL_W = 11  # px; must match .v11-ss-cell CSS width.

    def _build_cartoon_svg(states: List[str]) -> str:
        """Render a cartoon SS strip as inline SVG: helix runs as a coil
        (sine wave), sheet runs as a flat arrow, loops as a thin grey line.
        `states` is a list of 'helix'/'sheet'/'loop'/'' (one per cell)."""
        cells = len(states)
        if cells == 0:
            return ""
        width = cells * CELL_W
        height = 18
        midy = height / 2
        # Identify contiguous runs.
        runs: List[Tuple[str, int, int]] = []  # (state, start_idx, end_idx_inclusive)
        i = 0
        while i < cells:
            s = states[i]
            j = i
            while j + 1 < cells and states[j + 1] == s:
                j += 1
            runs.append((s, i, j))
            i = j + 1
        bits: List[str] = []
        bits.append(f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none">')
        # Always draw a faint axis line.
        bits.append(f'<line x1="0" y1="{midy}" x2="{width}" y2="{midy}" stroke="#d8dee8" stroke-width="0.6"/>')
        for state, s_idx, e_idx in runs:
            x0 = s_idx * CELL_W
            x1 = (e_idx + 1) * CELL_W
            if state == "helix":
                # Build a sine wave path across the run.
                period = CELL_W * 2  # one full sine per 2 cells (≈ 22px)
                amp = 5.5
                path = [f"M {x0:.2f} {midy:.2f}"]
                # quadratic-bezier segments approximating the sine.
                step = CELL_W / 2  # half-period quarter
                x = x0
                up = True
                while x + step <= x1 + 0.01:
                    cx = x + step / 2
                    cy = midy - amp if up else midy + amp
                    path.append(f"Q {cx:.2f} {cy:.2f} {x + step:.2f} {midy:.2f}")
                    x += step
                    up = not up
                bits.append(f'<path d="{" ".join(path)}" stroke="#e88a22" stroke-width="1.6" fill="none" stroke-linecap="round"/>')
            elif state == "sheet":
                # Flat arrow: rectangle body + arrowhead at right end.
                head_w = min(CELL_W * 0.9, (x1 - x0) * 0.3)
                body_x1 = x1 - head_w
                bits.append(
                    f'<path d="M {x0:.2f} {midy - 4:.2f} L {body_x1:.2f} {midy - 4:.2f} '
                    f'L {body_x1:.2f} {midy - 6:.2f} L {x1:.2f} {midy:.2f} '
                    f'L {body_x1:.2f} {midy + 6:.2f} L {body_x1:.2f} {midy + 4:.2f} '
                    f'L {x0:.2f} {midy + 4:.2f} Z" fill="#8d6af1" stroke="#5d3fd1" stroke-width="0.5"/>'
                )
            elif state == "loop":
                # Thin grey horizontal line emphasising loop.
                bits.append(
                    f'<line x1="{x0:.2f}" y1="{midy:.2f}" x2="{x1:.2f}" y2="{midy:.2f}" '
                    f'stroke="#b8bec7" stroke-width="1.6" stroke-linecap="round"/>'
                )
            else:
                # Empty / no SS coverage: leave the faint axis line, add hatched dim band.
                bits.append(
                    f'<rect x="{x0:.2f}" y="{midy - 2:.2f}" width="{x1 - x0:.2f}" height="4" '
                    f'fill="#f2f2f4"/>'
                )
        bits.append('</svg>')
        return "".join(bits)

    for b in range(n_blocks):
        start = b * BLOCK + 1
        end = min(track_len, (b + 1) * BLOCK)
        cells_in_block = end - start + 1
        pieces.append('<div class="v11-ss-blockwrap"><table class="v11-ss-grid"><tbody>')
        # Position ticks row.
        pieces.append('<tr><th></th>')
        for pos in range(start, end + 1):
            label = str(pos) if (pos % 10 == 0 or pos == start) else ""
            pieces.append(f'<td class="v11-ss-tick">{label}</td>')
        pieces.append('</tr>')
        # One pair of rows per clade: cartoon strip + per-residue letters.
        for clade in clades:
            color = _v11_clade_color(clade)
            # Build per-cell SS state list for the cartoon SVG.
            states: List[str] = []
            for pos in range(start, end + 1):
                ent = cell.get((clade, pos))
                states.append(ent[0] if ent else "")
            cartoon_svg = _build_cartoon_svg(states)
            pieces.append(
                f'<tr><th style="color:{color};" rowspan="2">{escape(clade)}</th>'
                f'<td class="v11-ss-cart" colspan="{cells_in_block}">{cartoon_svg}</td>'
                f'</tr>'
            )
            pieces.append('<tr>')
            for pos in range(start, end + 1):
                ent = cell.get((clade, pos))
                if not ent:
                    pieces.append('<td class="v11-ss-cell" style="background:#f2f2f4;color:#bcbcc4;">/</td>')
                    continue
                ss, frac, n = ent
                bg = SS_COLORS.get(ss, "#b8bec7")
                letter = SS_LETTERS.get(ss, "C")
                alpha = max(0.35, min(1.0, frac))
                try:
                    r = int(bg[1:3], 16); g = int(bg[3:5], 16); bb = int(bg[5:7], 16)
                    bgcss = f"rgba({r},{g},{bb},{alpha:.2f})"
                except Exception:  # noqa: BLE001
                    bgcss = bg
                title = f"{clade} pos {pos}: {ss} ({int(frac*100)}% of {n} species)"
                pieces.append(
                    f'<td class="v11-ss-cell" style="background:{bgcss};" title="{escape(title)}">{letter}</td>'
                )
            pieces.append('</tr>')
        pieces.append('</tbody></table></div>')

    pieces.append('</details></section>')
    return "".join(pieces)


def _v11_html_to_srcdoc(path: Path) -> Optional[str]:
    """Read an HTML file and return an HTML-attribute-escaped string suitable
    for an `<iframe srcdoc="...">` attribute. Letting the report inline the
    full overlay HTML this way means the parent HTML can be shared on its
    own without breaking the embedded viewer."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return escape(text, quote=True)


def _v11_overlay_iframe_markup(href: Optional[str],
                                outdir: Optional[Any],
                                title: str,
                                link_text: str,
                                margin_top_px: int = 0) -> str:
    """Render the "Open in new tab" link + embedded iframe for a V11 structure
    overlay. When `outdir` is provided AND the overlay HTML file is readable,
    the iframe uses `srcdoc=` so the parent stays self-contained when shared.
    Otherwise falls back to `src=` (relative path, requires sibling files).
    Returns "" when no href is provided."""
    if not href:
        return ""
    iframe_style = (
        "width:100%;height:640px;border:1px solid var(--line);"
        "border-radius:12px;background:#05070a;"
    )
    margin_attr = f' style="margin-top:{margin_top_px}px"' if margin_top_px else ""
    link_html = (
        f'<p{margin_attr}><a class="chip" href="{escape(str(href))}" '
        f'target="_blank" rel="noreferrer">{link_text}</a></p>'
    )
    srcdoc = None
    if outdir:
        try:
            srcdoc = _v11_html_to_srcdoc(Path(outdir) / str(href))
        except Exception:  # noqa: BLE001
            srcdoc = None
    iframe_attr = (
        f'srcdoc="{srcdoc}"' if srcdoc
        else f'src="{escape(str(href))}"'
    )
    iframe_html = (
        f'<iframe {iframe_attr} title="{escape(title)}" style="{iframe_style}"></iframe>'
    )
    return link_html + iframe_html


def v11_clade_overlay_panel_markup(*,
                                   broad_href: Optional[str] = None,
                                   grouped_href: Optional[str] = None,
                                   mod_href: Optional[str] = None,
                                   combined_href: Optional[str] = None,
                                   bubble_pdf_href: Optional[str] = None,
                                   grouped_bubble_pdf_href: Optional[str] = None,
                                   mod_bubble_pdf_href: Optional[str] = None,
                                   outdir: Optional[Path] = None) -> str:
    """Per-clade 3D structure overlay block for the alignment browser. Renders
    up to three 3Dmol viewers (combined / broad-clade / subdivided 9-group)
    plus quick links to the printable bubble-grid PDFs. The combined viewer
    is the primary embedded one — its clade dropdown lists ~22 buckets so
    the user can pick any broad-clade OR any 9-group subdivision (Primates,
    Rodents, OtherMammals, ...) without switching files."""
    if not (broad_href or grouped_href or mod_href or combined_href):
        return ""
    iframe_style = (
        "width:100%;height:560px;border:1px solid var(--line);"
        "border-radius:8px;background:#05070a;"
    )
    pieces: List[str] = []
    pieces.append('<section class="alphafold-structure-panel v11-clade-overlay-panel" id="v11-clade-overlay-panel">')
    pieces.append('<details open><summary style="cursor:pointer;font-weight:700;font-size:0.95rem;">'
                  'Per-clade 3D structure overlay (Identity-to-human shaded onto AlphaFold)'
                  '</summary>')
    pieces.append('<p class="muted" style="margin:8px 0 12px 0;font-size:0.85rem;">'
                  'Per-residue clade identity painted onto the human AlphaFold model. '
                  'Pick any clade from the viewer dropdown to shade secondary structure '
                  'by how conserved each residue is in that clade. The primary viewer '
                  'below lists BOTH broad clades (Mammalia, Aves, Reptilia, Teleostei, '
                  '...) AND the 9-group subdivisions (Primates, Rodents, OtherMammals, '
                  'Teleosts, OtherFish, Birds, Reptiles, Amphibians, OtherVertebrates). '
                  'Separate broad-only and 9-group-only viewers are linked below.'
                  '</p>')
    if combined_href:
        # Try to inline the overlay's full HTML into a `srcdoc` attribute so
        # the report stays self-contained when shared without sibling files.
        # Falls back to plain `src=` (relative path) when outdir isn't
        # available or the file can't be read.
        combined_srcdoc = (
            _v11_html_to_srcdoc(Path(outdir) / str(combined_href))
            if outdir is not None else None
        )
        pieces.append('<div style="margin-bottom:18px;">')
        pieces.append(
            '<p style="margin:0 0 6px 0;font-weight:600;font-size:0.88rem;">'
            'Combined viewer — broad clades + 9-group subdivisions (all ~22 buckets)'
            '</p>'
        )
        pieces.append(
            f'<p><a class="chip" href="{escape(str(combined_href))}" target="_blank" rel="noreferrer">'
            f'Open in new tab &#8599;</a></p>'
        )
        iframe_attr = (
            f'srcdoc="{combined_srcdoc}"' if combined_srcdoc
            else f'src="{escape(str(combined_href))}"'
        )
        # V11: lazy load + content-visibility so the heavy 3Dmol viewer
        # doesn't block alignment-browser scroll. The browser defers iframe
        # render until it enters the viewport; `content-visibility:auto`
        # skips paint/layout while off-screen too, keeping the rest of the
        # page snappy on long alignments.
        pieces.append(
            f'<iframe {iframe_attr} loading="lazy" '
            f'title="V11 structure overlay (combined)" '
            f'style="{iframe_style};content-visibility:auto;contain-intrinsic-size:560px;"></iframe>'
        )
        pieces.append('</div>')
    # When the combined viewer is embedded above, surface broad / grouped /
    # PDF links as compact chips so the user still has one-click access to
    # the dedicated single-grouping viewers and the printable PDFs without
    # paying for two additional iframes on the page.
    extra_chips: List[str] = []
    if broad_href and combined_href:
        extra_chips.append(
            f'<a class="chip" href="{escape(str(broad_href))}" target="_blank" rel="noreferrer">'
            f'Broad-clade viewer (dedicated) &#8599;</a>'
        )
    if grouped_href and combined_href:
        extra_chips.append(
            f'<a class="chip" href="{escape(str(grouped_href))}" target="_blank" rel="noreferrer">'
            f'9-group viewer (dedicated) &#8599;</a>'
        )
    if mod_href and combined_href:
        extra_chips.append(
            f'<a class="chip" href="{escape(str(mod_href))}" target="_blank" rel="noreferrer">'
            f'Compact 9-group viewer (Other / x naming) &#8599;</a>'
        )
    if bubble_pdf_href:
        extra_chips.append(
            f'<a class="chip" href="{escape(str(bubble_pdf_href))}" target="_blank" rel="noreferrer">'
            f'Broad bubble PDF &#8599;</a>'
        )
    if grouped_bubble_pdf_href:
        extra_chips.append(
            f'<a class="chip" href="{escape(str(grouped_bubble_pdf_href))}" target="_blank" rel="noreferrer">'
            f'9-group bubble PDF &#8599;</a>'
        )
    if mod_bubble_pdf_href:
        extra_chips.append(
            f'<a class="chip" href="{escape(str(mod_bubble_pdf_href))}" target="_blank" rel="noreferrer">'
            f'Compact 9-group bubble PDF &#8599;</a>'
        )
    if extra_chips:
        pieces.append('<p style="margin:8px 0 12px 0;display:flex;gap:6px;flex-wrap:wrap;">')
        pieces.append(" ".join(extra_chips))
        pieces.append('</p>')

    # Fall back to inline iframes only when the combined viewer is missing —
    # otherwise the combined viewer is the primary embed and the chips above
    # cover the single-grouping cases.
    if not combined_href and broad_href:
        pieces.append('<div style="margin-bottom:18px;">')
        pieces.append('<p style="margin:0 0 6px 0;font-weight:600;font-size:0.88rem;">Broad clades</p>')
        pieces.append(
            f'<iframe src="{escape(str(broad_href))}" '
            f'title="V11 structure overlay (broad clades)" style="{iframe_style}"></iframe>'
        )
        pieces.append('</div>')
    if not combined_href and grouped_href:
        pieces.append('<div>')
        pieces.append(
            '<p style="margin:0 0 6px 0;font-weight:600;font-size:0.88rem;">'
            'Subdivided 9-group (Primates / Rodents / OtherMammals split)'
            '</p>'
        )
        pieces.append(
            f'<iframe src="{escape(str(grouped_href))}" '
            f'title="V11 structure overlay (9-group)" style="{iframe_style}"></iframe>'
        )
        pieces.append('</div>')
    pieces.append('</details></section>')
    return "".join(pieces)


def alphafold_structure_script() -> str:
    return r"""
    let ALPHAFOLD_VIEWER = null;
    let ALPHAFOLD_MODEL = null;
    let ALPHAFOLD_PANEL_READY = false;
    let ALPHAFOLD_SELECTED_RANGE = null;
    let ALPHAFOLD_RENDER_MODE = "cartoon";
    let ALPHAFOLD_RENDER_WARNING = "";
    let ALPHAFOLD_MANUAL_RANGES = [];
    let ALPHAFOLD_DYNAMIC_RANGES = [];
    let ALPHAFOLD_RANGE_LIST_COLLAPSED = true;
    const ALPHAFOLD_SIDECHAIN_ATOMS = [
      "CB", "CG", "CG1", "CG2", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3",
      "CZ", "CZ2", "CZ3", "CH2", "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2",
      "NZ", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH", "SD", "SG"
    ];
    const ALPHAFOLD_HETEROATOM_ELEMENTS = ["N", "O", "S", "P", "F", "Cl", "CL", "Br", "BR", "I"];
    const ALPHAFOLD_SIDECHAIN_ELEMENT_COLORS = {
      H: "#ffffff",
      C: "#c8c8c8",
      N: "#8f8fff",
      O: "#f00000",
      S: "#ffc832",
      P: "#ffa500",
      F: "#daa520",
      Cl: "#00ff00",
      CL: "#00ff00",
      Br: "#a52a2a",
      BR: "#a52a2a",
      I: "#940094"
    };
    const ALPHAFOLD_RESIDUE_CODES = {
      ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E", GLY: "G",
      HIS: "H", HSD: "H", HSE: "H", HSP: "H", ILE: "I", LEU: "L", LYS: "K", MET: "M",
      PHE: "F", PRO: "P", SER: "S", THR: "T", TRP: "W", TYR: "Y", VAL: "V",
      SEC: "U", PYL: "O", MSE: "M", ASX: "B", GLX: "Z", XLE: "J", UNK: "X"
    };

    function alphaFoldColor(kind) {
      const colors = (ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.colors) || {};
      return colors[kind] || colors.manual || "#0ea5e9";
    }

    function alphaFoldResidueCount() {
      const count = Number(ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.residue_count);
      return Number.isFinite(count) && count > 0 ? Math.floor(count) : 1;
    }

    function alphaFoldClampRange(range) {
      if (!range) return null;
      const maxResidue = alphaFoldResidueCount();
      let start = Math.floor(Number(range.start));
      let end = Math.floor(Number(range.end));
      if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
      if (start > end) [start, end] = [end, start];
      start = Math.max(1, Math.min(maxResidue, start));
      end = Math.max(1, Math.min(maxResidue, end));
      return {
        ...range,
        start,
        end,
        length: Math.max(1, end - start + 1),
        color: range.color || alphaFoldColor(range.kind || "manual"),
      };
    }

    function alphaFoldRangeKey(range) {
      if (!range) return "";
      return [range.source || "", range.kind || "", range.group || "", range.label || "", range.start, range.end].join(":");
    }

    function alphaFoldAllRanges(dynamicRanges) {
      const rows = []
        .concat((ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.ranges) || [])
        .concat(((ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.calcium_binding) || {}).ranges || [])
        .concat(dynamicRanges || ALPHAFOLD_DYNAMIC_RANGES || [])
        .concat(ALPHAFOLD_MANUAL_RANGES || [])
        .map(alphaFoldClampRange)
        .filter(Boolean);
      const seen = new Set();
      return rows.filter((row) => {
        const key = alphaFoldRangeKey(row);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      }).sort((a, b) => (
        Number(a.start) - Number(b.start)
        || Number(a.end) - Number(b.end)
        || String(a.source || "").localeCompare(String(b.source || ""))
        || String(a.label || "").localeCompare(String(b.label || ""))
      ));
    }

    function parseAlphaFoldManualRanges(text) {
      const ranges = [];
      const tokens = String(text || "").split(/[,\s;]+/).map((token) => token.trim()).filter(Boolean);
      tokens.forEach((token, index) => {
        const match = token.match(/^(\d+)(?:-(\d+))?$/);
        if (!match) return;
        ranges.push(alphaFoldClampRange({
          source: "manual",
          kind: "manual",
          label: `Manual range ${token}`,
          start: Number(match[1]),
          end: Number(match[2] || match[1]),
          color: alphaFoldColor("manual"),
          group: `manual_${Date.now()}_${index}`,
        }));
      });
      return ranges.filter(Boolean);
    }

    function alphaFoldSelectionSpec(range) {
      const clamped = alphaFoldClampRange(range);
      if (!clamped) return {};
      const residues = [];
      for (let residue = clamped.start; residue <= clamped.end; residue += 1) residues.push(residue);
      return { chain: "A", resi: residues };
    }

    function alphaFoldSelectionWithAtoms(range, atoms) {
      return { ...alphaFoldSelectionSpec(range), atom: atoms };
    }

    function alphaFoldSideChainColorScheme() {
      return { prop: "elem", map: ALPHAFOLD_SIDECHAIN_ELEMENT_COLORS };
    }

    function alphaFoldResidueCode(residueName) {
      const key = String(residueName || "").trim().toUpperCase();
      return ALPHAFOLD_RESIDUE_CODES[key] || (key.length === 1 ? key : "X");
    }

    function alphaFoldRenderModeLabel() {
      const labels = {
        cartoon: "cartoon",
        surface: "surface + cartoon",
        density: "electron density + cartoon",
      };
      return labels[ALPHAFOLD_RENDER_MODE] || labels.cartoon;
    }

    function alphaFoldSurfaceType(name) {
      if (!window.$3Dmol || !$3Dmol.SurfaceType) return name;
      return $3Dmol.SurfaceType[name] || $3Dmol.SurfaceType.VDW || $3Dmol.SurfaceType.SAS || name;
    }

    function alphaFoldClearRenderOverlays() {
      if (!ALPHAFOLD_VIEWER) return;
      try {
        if (ALPHAFOLD_VIEWER.removeAllSurfaces) ALPHAFOLD_VIEWER.removeAllSurfaces();
      } catch (error) {
        console.warn("Could not clear AlphaFold render overlays.", error);
      }
    }

    function alphaFoldApplySurfaceOverlay() {
      if (!ALPHAFOLD_VIEWER || !ALPHAFOLD_VIEWER.addSurface) {
        ALPHAFOLD_RENDER_WARNING = "surface rendering unavailable in this 3Dmol build";
        return false;
      }
      try {
        const surface = ALPHAFOLD_VIEWER.addSurface(
          alphaFoldSurfaceType("VDW"),
          { color: "#dbeafe", opacity: 0.30 },
          {}
        );
        if (surface && typeof surface.then === "function") {
          surface.then(() => {
            if (ALPHAFOLD_VIEWER) ALPHAFOLD_VIEWER.render();
          }).catch((error) => console.warn("Could not add AlphaFold surface.", error));
        }
        return true;
      } catch (error) {
        ALPHAFOLD_RENDER_WARNING = "surface rendering unavailable";
        console.warn("Could not add AlphaFold surface.", error);
        return false;
      }
    }

    function alphaFoldApplyDensityOverlay() {
      const density = (ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.density_map) || {};
      const volumeText = density.text || density.data || density.voldata || "";
      if (!density.available || !volumeText) {
        ALPHAFOLD_RENDER_WARNING = density.reason || "electron-density map unavailable for this model";
        return false;
      }
      if (!ALPHAFOLD_VIEWER || !ALPHAFOLD_VIEWER.addVolumetricData) {
        ALPHAFOLD_RENDER_WARNING = "electron-density rendering unavailable in this 3Dmol build";
        return false;
      }
      try {
        const format = String(density.format || density.volformat || "cube").toLowerCase();
        let volumePayload = volumeText;
        if (density.encoding === "base64" && window.$3Dmol && $3Dmol.base64ToArray) {
          volumePayload = $3Dmol.base64ToArray(volumeText);
        }
        ALPHAFOLD_VIEWER.addVolumetricData(volumePayload, format, {
          isoval: Number(density.isovalue || density.isoval || 1.0),
          color: density.color || "#38bdf8",
          opacity: Number(density.opacity || 0.34),
        });
        return true;
      } catch (error) {
        ALPHAFOLD_RENDER_WARNING = "could not render bundled electron-density map";
        console.warn("Could not add AlphaFold density overlay.", error);
        return false;
      }
    }

    function alphaFoldApplyRenderMode() {
      ALPHAFOLD_RENDER_WARNING = "";
      if (!ALPHAFOLD_VIEWER) return;
      if (ALPHAFOLD_RENDER_MODE === "surface") {
        alphaFoldApplySurfaceOverlay();
      } else if (ALPHAFOLD_RENDER_MODE === "density") {
        alphaFoldApplyDensityOverlay();
      }
    }

    function alphaFoldSetRenderMode(value) {
      const allowed = new Set(["cartoon", "surface", "density"]);
      ALPHAFOLD_RENDER_MODE = allowed.has(value) ? value : "cartoon";
      const select = document.getElementById("alphafold-render-mode");
      if (select && select.value !== ALPHAFOLD_RENDER_MODE) select.value = ALPHAFOLD_RENDER_MODE;
      if (!ALPHAFOLD_VIEWER) {
        alphaFoldUpdateStatus();
        return;
      }
      if (ALPHAFOLD_SELECTED_RANGE) {
        alphaFoldApplySelection(ALPHAFOLD_SELECTED_RANGE, false);
      } else {
        alphaFoldSetBaseStyle();
        alphaFoldApplyRenderMode();
        ALPHAFOLD_VIEWER.render();
        alphaFoldUpdateStatus();
      }
    }

    function alphaFoldSecondaryRanges(kind) {
      const ss = (ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.secondary_structure) || {};
      return (ss.ranges || []).filter((range) => range.kind === kind).map(alphaFoldClampRange).filter(Boolean);
    }

    function alphaFoldSecondaryKindFromCode(code) {
      const text = String(code || "").toLowerCase();
      if (text.startsWith("h")) return "helix";
      if (text.startsWith("s") || text.startsWith("e") || text.startsWith("b")) return "sheet";
      return "loop";
    }

    function alphaFoldSmoothedTrack(labels, kind, windowSize = 9) {
      const half = Math.max(0, Math.floor(Number(windowSize || 9) / 2));
      return labels.map((_, index) => {
        const start = Math.max(0, index - half);
        const end = Math.min(labels.length, index + half + 1);
        const segment = labels.slice(start, end);
        if (!segment.length) return 0;
        return Number((segment.filter((label) => label === kind).length / segment.length).toFixed(4));
      });
    }

    function alphaFoldSecondaryRangesFromResidues(residues) {
      const ranges = [];
      if (!residues.length) return ranges;
      const counters = { helix: 0, sheet: 0 };
      let startIndex = 0;
      for (let index = 1; index <= residues.length; index += 1) {
        if (index < residues.length && residues[index].secondary_structure === residues[startIndex].secondary_structure) continue;
        const first = residues[startIndex];
        const last = residues[index - 1];
        const kind = first.secondary_structure || "loop";
        const row = {
          kind,
          start: first.position,
          end: last.position,
          length: Math.max(1, Number(last.position) - Number(first.position) + 1),
          color: alphaFoldColor(kind),
        };
        if (kind === "helix" || kind === "sheet") {
          counters[kind] += 1;
          row.ordinal = counters[kind];
          row.display_label = `${kind === "helix" ? "H" : "S"}${counters[kind]}`;
        }
        ranges.push(row);
        startIndex = index;
      }
      return ranges;
    }

    function alphaFoldEnumerateSecondaryRanges(ranges) {
      const counters = { helix: 0, sheet: 0 };
      return (ranges || []).map((range) => {
        const kind = String(range.kind || "loop").toLowerCase();
        const row = { ...range, kind };
        if (kind === "helix" || kind === "sheet") {
          counters[kind] += 1;
          row.ordinal = Number(row.ordinal || counters[kind]);
          row.display_label = row.display_label || `${kind === "helix" ? "H" : "S"}${row.ordinal}`;
        }
        return row;
      });
    }

    function alphaFoldSecondaryStructureFromViewerModel(model) {
      if (!model || !model.selectedAtoms) return null;
      const atoms = model.selectedAtoms({}) || [];
      const byResidue = new Map();
      atoms.forEach((atom) => {
        const position = Number(atom.resi);
        if (!Number.isFinite(position)) return;
        const chain = atom.chain || atom.reschain || "A";
        const key = `${chain}:${position}`;
        const kind = alphaFoldSecondaryKindFromCode(atom.ss || atom.ssbegin || atom.ssend);
        const existing = byResidue.get(key);
        if (!existing || atom.atom === "CA" || existing.secondary_structure === "loop") {
          byResidue.set(key, {
            position,
            chain,
            residue_name: atom.resn || atom.residue || "",
            secondary_structure: kind,
            ss_code: { helix: "h", sheet: "s" }[kind] || "c",
          });
        }
      });
      const residues = [...byResidue.values()].sort((left, right) => (
        String(left.chain).localeCompare(String(right.chain)) || Number(left.position) - Number(right.position)
      ));
      const nonLoopCount = residues.filter((row) => row.secondary_structure !== "loop").length;
      if (!residues.length || nonLoopCount < 3) return null;
      const labels = residues.map((row) => row.secondary_structure || "loop");
      const helixPropensity = alphaFoldSmoothedTrack(labels, "helix");
      const sheetPropensity = alphaFoldSmoothedTrack(labels, "sheet");
      residues.forEach((row, index) => {
        row.helix_propensity = helixPropensity[index] || 0;
        row.sheet_propensity = sheetPropensity[index] || 0;
      });
      return {
        available: true,
        method: "3dmol_model_secondary_structure",
        note: "Secondary-structure overlay uses the assignments exposed by 3Dmol from the loaded AlphaFold structure model.",
        residues,
        ranges: alphaFoldSecondaryRangesFromResidues(residues),
        helix_propensity: helixPropensity,
        sheet_propensity: sheetPropensity,
      };
    }

    function alphaFoldApplyPayloadSecondaryStructureToAtoms(model) {
      if (!model || !model.selectedAtoms) return;
      const ssRows = (((ALPHAFOLD_STRUCTURE || {}).secondary_structure || {}).residues || []);
      const ssByResidue = new Map(ssRows.map((row) => [Number(row.position), row.ss_code || "c"]));
      model.selectedAtoms({}).forEach((atom) => {
        const ss = ssByResidue.get(Number(atom.resi));
        if (ss) atom.ss = ss;
      });
    }

    function alphaFoldSetBaseStyle(options = {}) {
      if (!ALPHAFOLD_VIEWER) return;
      try {
        if (ALPHAFOLD_VIEWER.removeAllLabels) ALPHAFOLD_VIEWER.removeAllLabels();
        if (ALPHAFOLD_VIEWER.removeAllShapes) ALPHAFOLD_VIEWER.removeAllShapes();
        alphaFoldClearRenderOverlays();
        const dimmed = Boolean(options.dimmed);
        const loopOpacity = dimmed ? 0.24 : 0.78;
        const secondaryOpacity = dimmed ? 0.30 : 0.96;
        ALPHAFOLD_VIEWER.setStyle({}, { cartoon: { color: alphaFoldColor("loop"), opacity: loopOpacity } });
        alphaFoldSecondaryRanges("helix").forEach((range) => {
          ALPHAFOLD_VIEWER.setStyle(alphaFoldSelectionSpec(range), { cartoon: { color: alphaFoldColor("helix"), opacity: secondaryOpacity } });
        });
        alphaFoldSecondaryRanges("sheet").forEach((range) => {
          ALPHAFOLD_VIEWER.setStyle(alphaFoldSelectionSpec(range), { cartoon: { color: alphaFoldColor("sheet"), arrows: true, opacity: secondaryOpacity } });
        });
      } catch (error) {
        console.warn("Could not style AlphaFold structure.", error);
      }
    }

    function alphaFoldResidueLabelRows(range) {
      if (!ALPHAFOLD_MODEL || !ALPHAFOLD_MODEL.selectedAtoms) return [];
      const atoms = ALPHAFOLD_MODEL.selectedAtoms(alphaFoldSelectionSpec(range)) || [];
      const byResidue = new Map();
      atoms.forEach((atom) => {
        const position = Number(atom.resi);
        if (!Number.isFinite(position)) return;
        const chain = atom.chain || atom.reschain || "A";
        const key = `${chain}:${position}`;
        const row = byResidue.get(key) || {
          chain,
          position,
          residue_name: atom.resn || atom.residue || "",
          atoms: [],
          sidechain_atoms: [],
          ca: null,
          cb: null,
        };
        row.atoms.push(atom);
        if (atom.atom === "CA") row.ca = atom;
        if (atom.atom === "CB") row.cb = atom;
        if (ALPHAFOLD_SIDECHAIN_ATOMS.includes(atom.atom)) row.sidechain_atoms.push(atom);
        byResidue.set(key, row);
      });
      return [...byResidue.values()].sort((left, right) => (
        String(left.chain).localeCompare(String(right.chain)) || Number(left.position) - Number(right.position)
      ));
    }

    function alphaFoldAveragePoint(atoms) {
      const usable = (atoms || []).filter((atom) => (
        Number.isFinite(Number(atom.x)) && Number.isFinite(Number(atom.y)) && Number.isFinite(Number(atom.z))
      ));
      if (!usable.length) return null;
      const sum = usable.reduce((acc, atom) => ({
        x: acc.x + Number(atom.x),
        y: acc.y + Number(atom.y),
        z: acc.z + Number(atom.z),
      }), { x: 0, y: 0, z: 0 });
      return { x: sum.x / usable.length, y: sum.y / usable.length, z: sum.z / usable.length };
    }

    function alphaFoldSelectionShowsResidueLabels(range) {
      if (!range || range.source !== "secondary_structure") return false;
      const kind = String(range.kind || "").toLowerCase();
      return kind === "helix" || kind === "sheet" || Number(range.length || 0) <= 24;
    }

    function alphaFoldAddResidueCalloutLabels(range) {
      if (!ALPHAFOLD_VIEWER || !alphaFoldSelectionShowsResidueLabels(range)) return;
      const rows = alphaFoldResidueLabelRows(range);
      const offsets = [
        { x: 1.7, y: 2.4, z: 1.1 },
        { x: -1.7, y: 2.2, z: 1.3 },
        { x: 1.9, y: -2.0, z: 1.2 },
        { x: -1.9, y: -2.1, z: 1.4 },
        { x: 0.5, y: 2.8, z: -1.5 },
        { x: -0.5, y: -2.8, z: -1.4 },
      ];
      rows.forEach((row, index) => {
        const anchor = alphaFoldAveragePoint(row.sidechain_atoms) || row.cb || row.ca || alphaFoldAveragePoint(row.atoms);
        if (!anchor) return;
        const offset = offsets[index % offsets.length];
        const scale = 1 + Math.floor(index / offsets.length) * 0.16;
        const labelPosition = {
          x: anchor.x + offset.x * scale,
          y: anchor.y + offset.y * scale,
          z: anchor.z + offset.z * scale,
        };
        const label = `${alphaFoldResidueCode(row.residue_name)}${row.position}`;
        if (ALPHAFOLD_VIEWER.addLine) {
          ALPHAFOLD_VIEWER.addLine({
            start: { x: anchor.x, y: anchor.y, z: anchor.z },
            end: labelPosition,
            color: "black",
            linewidth: 1.25,
          });
        }
        if (ALPHAFOLD_VIEWER.addLabel) {
          ALPHAFOLD_VIEWER.addLabel(
            label,
            {
              position: labelPosition,
              fontColor: "black",
              backgroundColor: "white",
              backgroundOpacity: 0.78,
              borderColor: "black",
              borderThickness: 1.2,
              fontSize: 13,
              inFront: true,
            }
          );
        }
      });
    }

    function alphaFoldApplySelection(range, zoom) {
      ALPHAFOLD_SELECTED_RANGE = alphaFoldClampRange(range);
      alphaFoldSetBaseStyle({ dimmed: Boolean(ALPHAFOLD_SELECTED_RANGE) });
      alphaFoldApplyRenderMode();
      if (ALPHAFOLD_VIEWER && ALPHAFOLD_SELECTED_RANGE) {
        const selection = alphaFoldSelectionSpec(ALPHAFOLD_SELECTED_RANGE);
        try {
          const selectedColor = ALPHAFOLD_SELECTED_RANGE.color || alphaFoldColor("selected");
          ALPHAFOLD_VIEWER.addStyle(selection, {
            cartoon: { color: selectedColor, opacity: 1.0 },
          });
          ALPHAFOLD_VIEWER.addStyle(selection, {
            stick: { radius: 0.13, color: selectedColor, opacity: 0.55 },
          });
          ALPHAFOLD_VIEWER.addStyle(alphaFoldSelectionWithAtoms(ALPHAFOLD_SELECTED_RANGE, ALPHAFOLD_SIDECHAIN_ATOMS), {
            stick: { radius: 0.24, colorscheme: alphaFoldSideChainColorScheme(), opacity: 1.0 },
          });
          ALPHAFOLD_VIEWER.addStyle({
            ...alphaFoldSelectionWithAtoms(ALPHAFOLD_SELECTED_RANGE, ALPHAFOLD_SIDECHAIN_ATOMS),
            elem: ALPHAFOLD_HETEROATOM_ELEMENTS,
          }, {
            sphere: { radius: 0.34, colorscheme: alphaFoldSideChainColorScheme(), opacity: 1.0 },
          });
          alphaFoldAddResidueCalloutLabels(ALPHAFOLD_SELECTED_RANGE);
          if (zoom) ALPHAFOLD_VIEWER.zoomTo(selection);
          ALPHAFOLD_VIEWER.render();
        } catch (error) {
          console.warn("Could not highlight AlphaFold selection.", error);
        }
      }
      alphaFoldRenderTrack();
      alphaFoldRenderRangeList();
      alphaFoldUpdateStatus();
    }

    function initializeAlphaFoldViewer() {
      const canvas = document.getElementById("alphafold-viewer-canvas");
      if (!canvas) return;
      if (!ALPHAFOLD_STRUCTURE || !ALPHAFOLD_STRUCTURE.available) {
        canvas.innerHTML = `<div class="alphafold-viewer-message">${escapeHtml((ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.reason) || "AlphaFold structure is not available for this run.")}</div>`;
        return;
      }
      if (!window.$3Dmol) {
        canvas.innerHTML = '<div class="alphafold-viewer-message">3Dmol.js is not available in this report. Re-run with the local 3Dmol asset available to enable the embedded structure viewer.</div>';
        return;
      }
      if (!ALPHAFOLD_STRUCTURE.pdb_text) {
        canvas.innerHTML = '<div class="alphafold-viewer-message">The AlphaFold PDB payload is empty.</div>';
        return;
      }
      try {
        canvas.innerHTML = "";
        ALPHAFOLD_VIEWER = $3Dmol.createViewer(canvas, { backgroundColor: "white", preserveDrawingBuffer: true });
        ALPHAFOLD_MODEL = ALPHAFOLD_VIEWER.addModel(ALPHAFOLD_STRUCTURE.pdb_text, "pdb");
        const modelSecondaryStructure = alphaFoldSecondaryStructureFromViewerModel(ALPHAFOLD_MODEL);
        if (modelSecondaryStructure && modelSecondaryStructure.ranges && modelSecondaryStructure.ranges.length) {
          ALPHAFOLD_STRUCTURE.secondary_structure = modelSecondaryStructure;
        } else {
          alphaFoldApplyPayloadSecondaryStructureToAtoms(ALPHAFOLD_MODEL);
        }
        alphaFoldSetBaseStyle();
        alphaFoldApplyRenderMode();
        ALPHAFOLD_VIEWER.zoomTo();
        ALPHAFOLD_VIEWER.render();
      } catch (error) {
        console.error(error);
        canvas.innerHTML = '<div class="alphafold-viewer-message">Could not initialize the embedded AlphaFold viewer.</div>';
      }
    }

    function alphaFoldTrackX(position, width, leftPad, rightPad) {
      const maxResidue = alphaFoldResidueCount();
      const inner = Math.max(1, width - leftPad - rightPad);
      if (maxResidue <= 1) return leftPad;
      return leftPad + ((Number(position) - 1) / (maxResidue - 1)) * inner;
    }

    function alphaFoldHelixElement(x1, x2, y, color, cssClass, title, strokeWidth) {
      // V11 helix renderer. Branches on cssClass:
      //   * "snapshot-ss-helix": cursive teardrop loops (cubic Bezier per loop
      //     with overshooting control points -> self-cross at the baseline).
      //     Used in the species-snapshot download, where loop shape reads
      //     better than a zigzag at small sizes.
      //   * anything else (architecture / comparative AF SS / stacked lanes
      //     / inline-styled AF track): V9.7-style compact zigzag polyline.
      //     The label/color-rich architecture row reads cleaner with the
      //     sawtooth than with tall loops crowding the H1..Hn labels.
      const width = Math.max(0.001, x2 - x1);
      const isSnapshot = cssClass === "snapshot-ss-helix";

      const gAttrs = ['fill="none"', 'stroke-linecap="round"', 'stroke-linejoin="round"'];
      if (cssClass) gAttrs.push(`class="${cssClass}"`);
      if (color !== undefined && color !== null && color !== "") gAttrs.push(`stroke="${color}"`);
      const parts = [`<g ${gAttrs.join(" ")}>`];

      if (isSnapshot) {
        // Bigger loops, thicker line so curves anti-alias cleanly at the
        // viewer rasterization scale (avoids pixelated look). Sized so loop
        // tops (y - height) land just above rowTop, no bleed into the
        // row-above residue cells (which end at rowTop-3 with rowHeight=40).
        const outSw = 2.80;
        const pitch = 14.0;
        const height = 20.0;
        const overshoot = 1.85;
        const numLoops = Math.max(1, Math.round(width / pitch));
        const step = width / numLoops;
        const chunks = [`M ${x1.toFixed(2)} ${y.toFixed(2)}`];
        for (let i = 0; i < numLoops; i += 1) {
          const xStart = x1 + step * i;
          const xEnd = x1 + step * (i + 1);
          const cp1x = xStart + overshoot * step;
          const cp2x = xStart + (1.0 - overshoot) * step;
          const cpY = y - height;
          chunks.push(`C ${cp1x.toFixed(2)} ${cpY.toFixed(2)} ${cp2x.toFixed(2)} ${cpY.toFixed(2)} ${xEnd.toFixed(2)} ${y.toFixed(2)}`);
        }
        parts.push(`<path d="${chunks.join(" ")}" stroke-width="${outSw.toFixed(2)}"/>`);
      } else {
        const sw = (strokeWidth === undefined || strokeWidth === null) ? 4.0 : strokeWidth;
        const steps = Math.max(2, Math.ceil(width / 8.0));
        const points = [];
        for (let i = 0; i <= steps; i += 1) {
          const xx = x1 + width * i / steps;
          const dy = i % 2 === 0 ? -4 : 4;
          points.push(`${xx.toFixed(1)},${(y + dy).toFixed(1)}`);
        }
        parts.push(`<polyline points="${points.join(" ")}" stroke-width="${sw.toFixed(2)}"/>`);
      }

      if (title) parts.push(`<title>${title}</title>`);
      parts.push('</g>');
      return parts.join("");
    }

    function alphaFoldRenderTrack() {
      const container = document.getElementById("alphafold-track");
      if (!container) return;
      const ss = (ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.secondary_structure) || {};
      const ranges = alphaFoldEnumerateSecondaryRanges((ss.ranges || []).map(alphaFoldClampRange).filter(Boolean));
      if (!ranges.length) {
        container.innerHTML = '<div class="alphafold-empty">No secondary-structure overlay could be derived for this model.</div>';
        return;
      }
      const width = 1000;
      const height = 218;
      const left = 52;
      const right = 28;
      const trackY = 60;
      const stripTop = 104;
      const chargeTop = 148;
      const calciumTop = 178;
      const maxResidue = alphaFoldResidueCount();
      const selected = alphaFoldClampRange(ALPHAFOLD_SELECTED_RANGE);
      const parts = [
        `<svg class="alphafold-track-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="AlphaFold secondary structure overlay">`,
        `<rect width="1000" height="${height}" fill="#ffffff"/>`,
        '<text x="52" y="24" font-size="13" font-weight="800" fill="#18202a">Secondary-structure overlay and smoothed propensity</text>',
        `<line x1="${left}" y1="${trackY}" x2="${width - right}" y2="${trackY}" stroke="#cbd5e1" stroke-width="3"/>`,
      ];
      [1, 100, 200, 300, 400, 500, 600, 700, maxResidue].filter((value, idx, arr) => value <= maxResidue && arr.indexOf(value) === idx).forEach((tick) => {
        const x = alphaFoldTrackX(tick, width, left, right);
        parts.push(`<line x1="${x.toFixed(1)}" y1="${trackY - 18}" x2="${x.toFixed(1)}" y2="${trackY + 18}" stroke="#94a3b8" stroke-width="1"/>`);
        parts.push(`<text x="${x.toFixed(1)}" y="${trackY + 34}" text-anchor="middle" font-size="10" font-weight="700" fill="#64748b">${tick}</text>`);
      });
      ranges.forEach((range) => {
        const x1 = alphaFoldTrackX(range.start, width, left, right);
        const x2 = alphaFoldTrackX(range.end, width, left, right);
        const color = alphaFoldColor(range.kind);
        if (range.kind === "helix") {
          parts.push(alphaFoldHelixElement(x1, x2, trackY, color, "", "", 4));
        } else if (range.kind === "sheet") {
          const tip = Math.min(14, Math.max(6, (x2 - x1) * 0.25));
          parts.push(`<polygon points="${x1.toFixed(1)},${trackY - 7} ${(x2 - tip).toFixed(1)},${trackY - 7} ${(x2 - tip).toFixed(1)},${trackY - 12} ${x2.toFixed(1)},${trackY} ${(x2 - tip).toFixed(1)},${trackY + 12} ${(x2 - tip).toFixed(1)},${trackY + 7} ${x1.toFixed(1)},${trackY + 7}" fill="${color}" opacity="0.9"/>`);
        } else {
          parts.push(`<rect x="${x1.toFixed(1)}" y="${trackY - 3}" width="${Math.max(1.5, x2 - x1).toFixed(1)}" height="6" rx="3" fill="${color}" opacity="0.72"/>`);
        }
        const displayLabel = range.display_label || "";
        const rangeTitle = `${displayLabel ? displayLabel + " " : ""}${range.kind} ${range.start}-${range.end}`;
        if (displayLabel) {
          parts.push(`<text x="${((x1 + x2) / 2).toFixed(1)}" y="${trackY - 16}" text-anchor="middle" font-size="9" font-weight="900" fill="${color}" paint-order="stroke" stroke="#ffffff" stroke-width="3" stroke-linejoin="round">${escapeHtml(displayLabel)}</text>`);
        }
        parts.push(`<rect data-af-start="${range.start}" data-af-end="${range.end}" data-af-label="${escapeHtml(range.kind)}" x="${x1.toFixed(1)}" y="${trackY - 24}" width="${Math.max(4, x2 - x1).toFixed(1)}" height="48" fill="transparent" style="cursor:pointer"><title>${escapeHtml(rangeTitle)}</title></rect>`);
      });
      if (selected) {
        const x1 = alphaFoldTrackX(selected.start, width, left, right);
        const x2 = alphaFoldTrackX(selected.end, width, left, right);
        parts.push(`<rect x="${x1.toFixed(1)}" y="36" width="${Math.max(3, x2 - x1).toFixed(1)}" height="92" fill="${alphaFoldColor("selected")}" opacity="0.23" stroke="${selected.color || alphaFoldColor("selected")}" stroke-width="2"/>`);
      }
      const helix = (((ALPHAFOLD_STRUCTURE.secondary_structure || {}).helix_propensity) || []);
      const sheet = (((ALPHAFOLD_STRUCTURE.secondary_structure || {}).sheet_propensity) || []);
      const barWidth = Math.max(0.4, (width - left - right) / Math.max(1, maxResidue));
      for (let i = 0; i < maxResidue; i += 1) {
        const x = alphaFoldTrackX(i + 1, width, left, right);
        const helixOpacity = Math.max(0.04, Math.min(0.9, Number(helix[i] || 0)));
        const sheetOpacity = Math.max(0.04, Math.min(0.9, Number(sheet[i] || 0)));
        parts.push(`<rect x="${x.toFixed(2)}" y="${stripTop}" width="${barWidth.toFixed(2)}" height="14" fill="${alphaFoldColor("helix")}" opacity="${helixOpacity.toFixed(3)}"/>`);
        parts.push(`<rect x="${x.toFixed(2)}" y="${stripTop + 20}" width="${barWidth.toFixed(2)}" height="14" fill="${alphaFoldColor("sheet")}" opacity="${sheetOpacity.toFixed(3)}"/>`);
      }
      parts.push(`<text x="8" y="${stripTop + 11}" font-size="10" font-weight="800" fill="#64748b">helix</text>`);
      parts.push(`<text x="8" y="${stripTop + 31}" font-size="10" font-weight="800" fill="#64748b">sheet</text>`);
      const chargeRows = architectureLocalChargeRows(maxResidue);
      if (chargeRows.length) {
        chargeRows.forEach((row) => {
          const x = alphaFoldTrackX(row.position, width, left, right);
          const title = `pos ${row.position} charge ${Number(row.charge).toFixed(1)} window ${row.window_start}-${row.window_end} ${row.window_sequence || ""}`;
          parts.push(`<rect x="${x.toFixed(2)}" y="${chargeTop}" width="${barWidth.toFixed(2)}" height="14" fill="${escapeHtml(row.color)}"><title>${escapeHtml(title)}</title></rect>`);
        });
        parts.push(`<line x1="${left}" y1="${chargeTop + 7}" x2="${width - right}" y2="${chargeTop + 7}" stroke="#475569" stroke-width="0.7" opacity="0.35"/>`);
        parts.push(`<text x="8" y="${chargeTop + 11}" font-size="10" font-weight="800" fill="#64748b">charge</text>`);
      }
      const calcium = architectureCalciumPayload(maxResidue);
      if (calcium.loops.length || calcium.ligands.length) {
        parts.push(`<line x1="${left}" y1="${calciumTop + 7}" x2="${width - right}" y2="${calciumTop + 7}" stroke="#cbd5e1" stroke-width="3"/>`);
        calcium.loops.forEach((loop) => {
          const x1 = alphaFoldTrackX(loop.start, width, left, right);
          const x2 = alphaFoldTrackX(loop.end, width, left, right);
          const title = `${loop.label || "CBR"} ${loop.start}-${loop.end}${loop.description ? " | " + loop.description : ""}`;
          parts.push(`<rect x="${x1.toFixed(1)}" y="${calciumTop - 2}" width="${Math.max(4, x2 - x1).toFixed(1)}" height="18" rx="5" fill="${escapeHtml(loop.color)}" opacity="0.88"><title>${escapeHtml(title)}</title></rect>`);
          parts.push(`<text x="${((x1 + x2) / 2).toFixed(1)}" y="${calciumTop + 11}" text-anchor="middle" font-size="9" font-weight="900" fill="#ffffff">${escapeHtml(loop.label || "")}</text>`);
        });
        calcium.ligands.forEach((ligand) => {
          const cx = alphaFoldTrackX(ligand.position, width, left, right);
          const sites = (ligand.sites || []).join(", ");
          const title = `${ligand.label || "ligand"} ${ligand.position}${sites ? " | " + sites : ""}`;
          parts.push(`<circle cx="${cx.toFixed(1)}" cy="${calciumTop + 7}" r="3.4" fill="${escapeHtml(ligand.color || alphaFoldColor("calcium_ligand"))}"><title>${escapeHtml(title)}</title></circle>`);
        });
        parts.push(`<text x="8" y="${calciumTop + 11}" font-size="10" font-weight="800" fill="#64748b">Ca2+</text>`);
      }
      parts.push('</svg>');
      container.innerHTML = parts.join("");
      container.querySelectorAll("[data-af-start]").forEach((node) => {
        node.addEventListener("click", () => {
          alphaFoldApplySelection({
            source: "secondary_structure",
            kind: node.dataset.afLabel || "manual",
            label: `${node.dataset.afLabel || "range"} ${node.dataset.afStart}-${node.dataset.afEnd}`,
            start: Number(node.dataset.afStart),
            end: Number(node.dataset.afEnd),
            color: alphaFoldColor(node.dataset.afLabel || "manual"),
          }, true);
        });
      });
    }

    function alphaFoldRangeRowHtml(range) {
      const key = escapeHtml(alphaFoldRangeKey(range));
      const selectedKey = alphaFoldRangeKey(ALPHAFOLD_SELECTED_RANGE);
      const selectedClass = key === escapeHtml(selectedKey) ? " selected" : "";
      const color = range.color || alphaFoldColor(range.kind || "manual");
      const source = [range.source, range.group].filter(Boolean).join(" | ");
      const score = range.score == null || range.score === "" ? "" : ` | score ${Number(range.score).toFixed ? Number(range.score).toFixed(3) : range.score}`;
      return `<button type="button" class="alphafold-range-row${selectedClass}" data-af-range-key="${key}"><div class="alphafold-range-main"><div class="alphafold-range-title"><span class="alphafold-swatch" style="background:${escapeHtml(color)}"></span><span>${escapeHtml(range.label || range.kind || "range")}</span></div><div class="alphafold-range-meta">${escapeHtml(`${range.kind || "range"} | ${source}${score}`)}</div></div><div class="alphafold-range-coords">${escapeHtml(`${range.start}-${range.end}`)}</div></button>`;
    }

    function alphaFoldRenderRangeList() {
      const panel = document.getElementById("alphafold-range-panel");
      const list = document.getElementById("alphafold-range-list");
      if (!list) return;
      const ranges = alphaFoldAllRanges();
      const countNode = document.getElementById("alphafold-range-count");
      if (countNode) countNode.textContent = `${ranges.length} selectable run${ranges.length === 1 ? "" : "s"}`;
      if (panel) panel.classList.toggle("collapsed", ALPHAFOLD_RANGE_LIST_COLLAPSED);
      const toggle = document.getElementById("alphafold-range-toggle");
      if (toggle) {
        toggle.textContent = ALPHAFOLD_RANGE_LIST_COLLAPSED ? "Show runs" : "Hide runs";
        toggle.setAttribute("aria-expanded", ALPHAFOLD_RANGE_LIST_COLLAPSED ? "false" : "true");
      }
      if (!ranges.length) {
        list.innerHTML = '<div class="alphafold-empty">No structure ranges are available yet. Type a manual residue range to start.</div>';
        return;
      }
      list.innerHTML = ranges.map(alphaFoldRangeRowHtml).join("");
      const lookup = new Map(ranges.map((range) => [alphaFoldRangeKey(range), range]));
      list.querySelectorAll("[data-af-range-key]").forEach((button) => {
        button.addEventListener("click", () => {
          const range = lookup.get(button.dataset.afRangeKey || "");
          if (range) alphaFoldApplySelection(range, true);
        });
      });
    }

    function alphaFoldUpdateStatus() {
      const status = document.getElementById("alphafold-status");
      if (!status) return;
      const meta = (ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.metadata) || {};
      const provenance = (ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.structure_provenance) || {};
      const validation = (ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.sequence_validation) || {};
      const ss = (ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.secondary_structure) || {};
      const selected = alphaFoldClampRange(ALPHAFOLD_SELECTED_RANGE);
      const parts = [];
      if (meta.uniprot_accession) parts.push(`UniProt ${meta.uniprot_accession}`);
      if (meta.entry_id) parts.push(meta.entry_id);
      parts.push(`${alphaFoldResidueCount()} residues`);
      if (validation.status) parts.push(`mapping ${validation.status}`);
      if (provenance.force_field) parts.push("force field: none in viewer");
      if (ss.method) parts.push(`secondary structure ${ss.method}`);
      parts.push(`display ${alphaFoldRenderModeLabel()}`);
      if (ALPHAFOLD_RENDER_WARNING) parts.push(ALPHAFOLD_RENDER_WARNING);
      if (selected) parts.push(`selected ${selected.label || selected.kind}: ${selected.start}-${selected.end}`);
      if (!(ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.viewer_js_available)) parts.push("3Dmol asset missing");
      status.textContent = parts.join(" | ");
    }

    function alphaFoldExportRanges(delimiter) {
      const ranges = alphaFoldAllRanges();
      const headers = ["source", "kind", "label", "group", "start", "end", "length", "score", "selected"];
      const selectedKey = alphaFoldRangeKey(ALPHAFOLD_SELECTED_RANGE);
      const lines = [headers.join(delimiter)];
      ranges.forEach((range) => {
        const row = {
          ...range,
          selected: alphaFoldRangeKey(range) === selectedKey,
        };
        lines.push(headers.map((header) => {
          const value = row[header] == null ? "" : String(row[header]);
          if (delimiter === ",") return `"${value.replaceAll('"', '""')}"`;
          return value.replaceAll("\t", " ");
        }).join(delimiter));
      });
      const suffix = delimiter === "," ? "csv" : "tsv";
      const blob = new Blob([lines.join("\n") + "\n"], { type: delimiter === "," ? "text/csv;charset=utf-8" : "text/tab-separated-values;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `alphafold_structure_ranges.${suffix}`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 250);
    }

    function alphaFoldViewerCanvas() {
      const container = document.getElementById("alphafold-viewer-canvas");
      return container ? container.querySelector("canvas") : null;
    }

    function alphaFoldDownloadBlob(filename, blob) {
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 250);
    }

    function alphaFoldDownloadDataUrl(filename, dataUrl) {
      const link = document.createElement("a");
      link.href = dataUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
    }

    function alphaFoldViewerDataUrl() {
      if (!ALPHAFOLD_VIEWER) return "";
      try {
        ALPHAFOLD_VIEWER.render();
        if (typeof ALPHAFOLD_VIEWER.pngURI === "function") return ALPHAFOLD_VIEWER.pngURI();
        const canvas = alphaFoldViewerCanvas();
        if (canvas && typeof canvas.toDataURL === "function") return canvas.toDataURL("image/png");
      } catch (error) {
        console.warn("Could not capture AlphaFold viewer.", error);
      }
      return "";
    }

    function alphaFoldExportPng() {
      const dataUrl = alphaFoldViewerDataUrl();
      if (!dataUrl) {
        ALPHAFOLD_RENDER_WARNING = "PNG export unavailable in this browser";
        alphaFoldUpdateStatus();
        return;
      }
      alphaFoldDownloadDataUrl("alphafold_structure_view.png", dataUrl);
    }

    function alphaFoldEscapeXml(value) {
      return String(value == null ? "" : value).replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&apos;",
      }[char]));
    }

    function alphaFoldExportSvg() {
      const dataUrl = alphaFoldViewerDataUrl();
      const canvas = alphaFoldViewerCanvas();
      if (!dataUrl || !canvas) {
        ALPHAFOLD_RENDER_WARNING = "SVG export unavailable in this browser";
        alphaFoldUpdateStatus();
        return;
      }
      const width = Math.max(1, Number(canvas.width || canvas.clientWidth || 1200));
      const height = Math.max(1, Number(canvas.height || canvas.clientHeight || 900));
      const selected = alphaFoldClampRange(ALPHAFOLD_SELECTED_RANGE);
      const title = selected
        ? `AlphaFold structure view, selected ${selected.start}-${selected.end}`
        : "AlphaFold structure view";
      const svg = [
        `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img">`,
        `<title>${alphaFoldEscapeXml(title)}</title>`,
        `<rect width="100%" height="100%" fill="white"/>`,
        `<image href="${alphaFoldEscapeXml(dataUrl)}" xlink:href="${alphaFoldEscapeXml(dataUrl)}" x="0" y="0" width="${width}" height="${height}"/>`,
        `</svg>`,
      ].join("");
      alphaFoldDownloadBlob("alphafold_structure_view.svg", new Blob([svg], { type: "image/svg+xml;charset=utf-8" }));
    }

    function renderAlphaFoldStructurePanel(dynamicRanges = []) {
      const panel = document.getElementById("alphafold-structure-panel");
      if (!panel) return;
      ALPHAFOLD_DYNAMIC_RANGES = (dynamicRanges || []).map(alphaFoldClampRange).filter(Boolean);
      if (!ALPHAFOLD_PANEL_READY) {
        const colors = (ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.colors) || {};
        panel.innerHTML = `
          <div class="alphafold-head">
            <div class="alphafold-title">
              <h2>Human AlphaFold structure</h2>
              <p>Human reference residues map directly onto the reference AlphaFold coordinate model; click browser runs, table-derived ranges, secondary-structure features, or a manual residue range to highlight the structure.</p>
            </div>
            <div class="alphafold-badges">
              <span class="alphafold-badge"><span class="alphafold-swatch" style="background:${escapeHtml(colors.helix || "#f97316")}"></span>alpha helix</span>
              <span class="alphafold-badge"><span class="alphafold-swatch" style="background:${escapeHtml(colors.sheet || "#7c3aed")}"></span>beta sheet</span>
              <span class="alphafold-badge"><span class="alphafold-swatch" style="background:${escapeHtml(colors.loop || "#9ca3af")}"></span>loop</span>
              <span class="alphafold-badge"><span class="alphafold-swatch" style="background:${escapeHtml(colors.selected || "#fde047")}"></span>selected run</span>
            </div>
          </div>
          <div class="alphafold-model-note"><strong>Structure provenance:</strong> this is an AlphaFold predicted coordinate model displayed by 3Dmol.js. V9.7 is not running AMBER, CHARMM, OPLS, or any molecular-mechanics force-field minimization in the viewer; it only maps conservation and residue selections onto the downloaded AlphaFold coordinates.</div>
          <div class="alphafold-grid">
            <div class="alphafold-viewer-card">
              <div class="alphafold-viewer-canvas" id="alphafold-viewer-canvas"></div>
              <div class="alphafold-track-wrap" id="alphafold-track"></div>
            </div>
            <div class="alphafold-controls-card">
              <div class="alphafold-control-row">
                <label>Manual human residue range
                  <input id="alphafold-manual-range" type="text" placeholder="e.g. 228-240, 505">
                </label>
                <button type="button" class="alphafold-button primary" id="alphafold-apply-range">Highlight</button>
                <button type="button" class="alphafold-button" id="alphafold-clear-range">Clear</button>
                <button type="button" class="alphafold-button" id="alphafold-zoom-range">Zoom</button>
              </div>
              <div class="alphafold-render-row">
                <label>Structure display
                  <select id="alphafold-render-mode">
                    <option value="cartoon">Cartoon</option>
                    <option value="surface">Surface + cartoon</option>
                    <option value="density">Electron density + cartoon</option>
                  </select>
                </label>
                <button type="button" class="alphafold-button" id="alphafold-export-png">Save PNG</button>
                <button type="button" class="alphafold-button" id="alphafold-export-svg">Save SVG</button>
              </div>
              <div class="alphafold-legend">
                <button type="button" class="alphafold-button" id="alphafold-export-csv">Export CSV</button>
                <button type="button" class="alphafold-button" id="alphafold-export-tsv">Export TSV</button>
                <span class="alphafold-legend-item"><span class="alphafold-swatch" style="background:${escapeHtml(colors.conserved || "#16a34a")}"></span>conserved</span>
                <span class="alphafold-legend-item"><span class="alphafold-swatch" style="background:${escapeHtml(colors.divergent || "#dc2626")}"></span>divergent/non-conserved</span>
                <span class="alphafold-legend-item"><span class="alphafold-swatch" style="background:#8f8fff"></span>N</span>
                <span class="alphafold-legend-item"><span class="alphafold-swatch" style="background:#f00000"></span>O</span>
                <span class="alphafold-legend-item"><span class="alphafold-swatch" style="background:#ffc832"></span>S</span>
                <span class="alphafold-legend-item"><span class="alphafold-swatch" style="background:#ffa500"></span>P</span>
              </div>
              <div class="alphafold-status" id="alphafold-status"></div>
              <div class="alphafold-range-panel collapsed" id="alphafold-range-panel">
                <div class="alphafold-range-head">
                  <div>
                    <strong>Selectable residue runs</strong>
                    <div class="alphafold-range-count" id="alphafold-range-count">0 selectable runs</div>
                  </div>
                  <button type="button" class="alphafold-range-toggle" id="alphafold-range-toggle" aria-expanded="false">Show runs</button>
                </div>
                <div class="alphafold-range-list" id="alphafold-range-list"></div>
              </div>
            </div>
          </div>
        `;
        ALPHAFOLD_PANEL_READY = true;
        initializeAlphaFoldViewer();
        const renderModeSelect = document.getElementById("alphafold-render-mode");
        if (renderModeSelect) {
          renderModeSelect.value = ALPHAFOLD_RENDER_MODE;
          renderModeSelect.addEventListener("change", (event) => alphaFoldSetRenderMode(event.target.value));
        }
        document.getElementById("alphafold-apply-range").addEventListener("click", () => {
          const input = document.getElementById("alphafold-manual-range");
          const ranges = parseAlphaFoldManualRanges(input ? input.value : "");
          if (!ranges.length) return;
          ALPHAFOLD_MANUAL_RANGES = ALPHAFOLD_MANUAL_RANGES.concat(ranges);
          alphaFoldApplySelection(ranges[0], true);
        });
        document.getElementById("alphafold-clear-range").addEventListener("click", () => {
          ALPHAFOLD_SELECTED_RANGE = null;
          alphaFoldSetBaseStyle();
          alphaFoldApplyRenderMode();
          if (ALPHAFOLD_VIEWER) {
            ALPHAFOLD_VIEWER.zoomTo();
            ALPHAFOLD_VIEWER.render();
          }
          alphaFoldRenderTrack();
          alphaFoldRenderRangeList();
          alphaFoldUpdateStatus();
        });
        document.getElementById("alphafold-zoom-range").addEventListener("click", () => {
          if (ALPHAFOLD_VIEWER && ALPHAFOLD_SELECTED_RANGE) ALPHAFOLD_VIEWER.zoomTo(alphaFoldSelectionSpec(ALPHAFOLD_SELECTED_RANGE));
          else if (ALPHAFOLD_VIEWER) ALPHAFOLD_VIEWER.zoomTo();
          if (ALPHAFOLD_VIEWER) ALPHAFOLD_VIEWER.render();
        });
        document.getElementById("alphafold-export-csv").addEventListener("click", () => alphaFoldExportRanges(","));
        document.getElementById("alphafold-export-tsv").addEventListener("click", () => alphaFoldExportRanges("\t"));
        document.getElementById("alphafold-export-png").addEventListener("click", alphaFoldExportPng);
        document.getElementById("alphafold-export-svg").addEventListener("click", alphaFoldExportSvg);
        document.getElementById("alphafold-range-toggle").addEventListener("click", () => {
          ALPHAFOLD_RANGE_LIST_COLLAPSED = !ALPHAFOLD_RANGE_LIST_COLLAPSED;
          alphaFoldRenderRangeList();
        });
      }
      alphaFoldRenderTrack();
      alphaFoldRenderRangeList();
      alphaFoldUpdateStatus();
    }
"""


def _v11_umap_panel_static(clusters_csv_path: Optional[Path]) -> str:
    """Static full-protein orthologue-UMAP scatter — the fallback used when the
    region-selectable JSON (V11_umap_regions.json) is absent.
    Each dot is one species' ortholog placed by its protein-informatic
    feature signature (per-quartile identity to human + composition + catalytic
    integrity + localization score, reduced to 2D). Hovering a dot reveals the
    species, its 9-group clade, the protein accession it corresponds to, the
    UMAP coordinates, and the key feature values; dots sitting away from their
    clade colour are functional-divergence candidates.

    Data source: V11_orthologue_clusters.csv (from _v11_orthologue_umap.py).
    Returns '' when the CSV is missing/empty so the panel is simply omitted.
    """
    if not clusters_csv_path or not Path(clusters_csv_path).exists():
        return ""
    try:
        df = pd.read_csv(clusters_csv_path)
    except Exception:  # noqa: BLE001
        return ""
    required = {"species", "grouped_clade_9", "umap_x", "umap_y"}
    if df.empty or not required.issubset(df.columns):
        return ""

    # Normalise missing string cells to "" so the `x or default` fallbacks
    # below behave: a NaN cell is truthy, which would defeat `or` and surface
    # a literal "nan" in the tooltip. (Same falsy/NaN class of bug guarded.)
    for _col in ("species", "grouped_clade_9", "protein_id"):
        if _col in df.columns:
            df[_col] = df[_col].fillna("")

    import html as _html

    xs = df["umap_x"].astype(float)
    ys = df["umap_y"].astype(float)
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if xmax - xmin < 1e-9:
        xmax = xmin + 1.0
    if ymax - ymin < 1e-9:
        ymax = ymin + 1.0

    W, H, PAD = 880.0, 520.0, 46.0

    def _sx(x):
        return PAD + (x - xmin) / (xmax - xmin) * (W - 2 * PAD)

    def _sy(y):  # flip so it reads like the PNG (UMAP y increases upward)
        return (H - PAD) - (y - ymin) / (ymax - ymin) * (H - 2 * PAD)

    spotlights = {"homo_sapiens", "mus_musculus", "danio_rerio"}
    feat_cols = [c for c in ("identity_quartile_1", "identity_quartile_2",
                             "identity_quartile_3", "identity_quartile_4",
                             "net_charge", "frac_aromatic", "catalytic_integrity",
                             "net_localization_score", "ungapped_length")
                 if c in df.columns]

    def _fmt(row, c):
        try:
            fv = float(row.get(c))
        except Exception:  # noqa: BLE001
            return ""
        if c == "ungapped_length":
            return f"len {int(fv)}"
        if c.startswith("identity_quartile_"):
            return f"Q{c[-1]} {fv:.2f}"
        label = {"net_charge": "net charge", "frac_aromatic": "aromatic",
                 "catalytic_integrity": "catalytic",
                 "net_localization_score": "localization"}.get(c, c)
        return f"{label} {fv:+.2f}" if c == "net_charge" else f"{label} {fv:.2f}"

    circles: List[str] = []
    labels: List[str] = []
    for _, row in df.iterrows():
        sp = str(row.get("species") or "")
        clade = str(row.get("grouped_clade_9") or "")
        prot = str(row.get("protein_id") or sp)
        cx = _sx(float(row["umap_x"]))
        cy = _sy(float(row["umap_y"]))
        color = _v11_clade_color(clade) if clade else "#888888"
        is_spot = sp.lower() in spotlights
        feat_txt = " · ".join(s for s in (_fmt(row, c) for c in feat_cols) if s)
        r = 6.5 if is_spot else 4.2
        sw = 1.6 if is_spot else 0.5
        circles.append(
            f'<circle class="v11u-dot" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
            f'fill="{color}" stroke="#1f2937" stroke-width="{sw}" '
            f'data-sp="{_html.escape(sp)}" data-clade="{_html.escape(clade)}" '
            f'data-prot="{_html.escape(prot)}" '
            f'data-x="{float(row["umap_x"]):.2f}" data-y="{float(row["umap_y"]):.2f}" '
            f'data-feat="{_html.escape(feat_txt)}"></circle>'
        )
        if is_spot:
            labels.append(f'<text class="v11u-splabel" x="{cx + 8:.1f}" '
                          f'y="{cy - 8:.1f}">{_html.escape(sp)}</text>')

    counts = df["grouped_clade_9"].astype(str).value_counts().to_dict()
    legend_order = [c for c in _V11_GROUPED_BUBBLE_CLADE_ORDER if c in counts]
    legend_order += [c for c in counts if c not in legend_order]
    legend_items = "".join(
        f'<span class="v11u-leg"><span class="v11u-sw" style="background:'
        f'{_v11_clade_color(c)};"></span>{_html.escape(str(c))} '
        f'<span class="v11u-cnt">({counts[c]})</span></span>'
        for c in legend_order
    )

    svg = (
        f'<svg viewBox="0 0 {int(W)} {int(H)}" class="v11u-svg" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'aria-label="Orthologue UMAP scatter">'
        f'<rect x="0" y="0" width="{int(W)}" height="{int(H)}" fill="#ffffff"/>'
        f'<text x="{PAD:.0f}" y="{H - 12:.0f}" class="v11u-axis">UMAP dim 1 &#8594;</text>'
        f'<text x="16" y="{PAD:.0f}" class="v11u-axis" '
        f'transform="rotate(-90 16 {PAD:.0f})">UMAP dim 2 &#8594;</text>'
        + "".join(circles) + "".join(labels) + '</svg>'
    )

    css = (
        "<style>"
        ".v11u-wrap{position:relative;max-width:920px;}"
        ".v11u-svg{width:100%;height:auto;border:1px solid var(--border,#d8dee6);"
        "border-radius:6px;background:#fff;}"
        ".v11u-dot{cursor:pointer;}"
        ".v11u-dot:hover{stroke:#111827;stroke-width:1.8;}"
        ".v11u-splabel{font:600 11px/1.1 system-ui,'Segoe UI',Arial;fill:#111827;"
        "paint-order:stroke;stroke:#fff;stroke-width:3px;pointer-events:none;}"
        ".v11u-axis{font:600 11px/1 system-ui;fill:#6b7280;}"
        ".v11u-legend{display:flex;flex-wrap:wrap;gap:8px 14px;margin-top:10px;font-size:0.82rem;}"
        ".v11u-leg{display:inline-flex;align-items:center;gap:5px;}"
        ".v11u-sw{display:inline-block;width:11px;height:11px;border:1px solid #c0c4cc;border-radius:2px;}"
        ".v11u-cnt{color:var(--muted,#888);}"
        ".v11u-tip{position:absolute;pointer-events:none;z-index:40;max-width:300px;"
        "background:#0f172a;color:#f1f5f9;border-radius:6px;padding:8px 10px;font-size:0.8rem;"
        "line-height:1.35;box-shadow:0 4px 14px rgba(15,23,42,.32);}"
        ".v11u-tip b{color:#fff;}"
        ".v11u-tfeat{color:#cbd5e1;margin-top:4px;display:block;font-size:0.75rem;}"
        "</style>"
    )

    js = (
        "<script>(function(){"
        "var panel=document.getElementById('v11-umap-panel');if(!panel)return;"
        "var svg=panel.querySelector('.v11u-svg'),tip=panel.querySelector('.v11u-tip'),"
        "wrap=panel.querySelector('.v11u-wrap');if(!svg||!tip||!wrap)return;"
        "function show(e){var c=e.target;if(!c||String(c.tagName).toLowerCase()!=='circle')return;"
        "var col=c.getAttribute('fill')||'#888';"
        "tip.innerHTML='<b>'+c.getAttribute('data-sp')+'</b> '"
        "+'<span style=\"font-weight:600;color:'+col+'\">'+c.getAttribute('data-clade')+'</span>'"
        "+'<br>protein: <b>'+c.getAttribute('data-prot')+'</b>'"
        "+'<br><span style=\"color:#94a3b8\">UMAP ('+c.getAttribute('data-x')+', '+c.getAttribute('data-y')+')</span>'"
        "+'<span class=\"v11u-tfeat\">'+c.getAttribute('data-feat')+'</span>';"
        "tip.hidden=false;move(e);}"
        "function move(e){var r=wrap.getBoundingClientRect();"
        "var tx=e.clientX-r.left+14,ty=e.clientY-r.top+14,tw=tip.offsetWidth,th=tip.offsetHeight;"
        "if(tx+tw>r.width)tx=e.clientX-r.left-tw-14;"
        "if(ty+th>r.height)ty=Math.max(2,r.height-th-4);"
        "tip.style.left=tx+'px';tip.style.top=ty+'px';}"
        "function hide(e){if(e.target&&String(e.target.tagName).toLowerCase()==='circle')tip.hidden=true;}"
        "svg.addEventListener('mouseover',show);"
        "svg.addEventListener('mousemove',function(e){if(!tip.hidden)move(e);});"
        "svg.addEventListener('mouseout',hide);"
        "})();</script>"
    )

    return (
        '<section class="alphafold-structure-panel v11-umap-panel" id="v11-umap-panel" '
        'style="margin-top:14px;">'
        '<details open>'
        '<summary style="cursor:pointer;font-weight:700;font-size:0.95rem;">'
        f'Orthologue UMAP — protein-informatic clustering ({len(df)} orthologs)'
        '</summary>'
        '<p class="muted" style="margin:8px 0;font-size:0.85rem;">'
        "Each dot is one species' ortholog, placed by its protein signature "
        '(per-quartile identity to human + amino-acid composition + catalytic '
        'integrity + localization score, reduced to 2D). Dots near each other have '
        'similar signatures; a dot that sits <em>away from its clade colour</em> is a '
        'functional-divergence candidate. <strong>Hover any dot</strong> to see its '
        'species, clade, the protein accession it corresponds to, and its feature '
        'values. Spotlight species (homo_sapiens / mus_musculus / danio_rerio) are '
        'drawn larger and labelled.'
        '</p>'
        + css +
        '<div class="v11u-wrap">' + svg +
        '<div class="v11u-tip" id="v11u-tip" hidden></div></div>'
        f'<div class="v11u-legend">{legend_items}</div>'
        '</details></section>'
        + js
    )


_V11_UMAP_CSS = (
    ".v11u-controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:8px 0;font-size:0.85rem;}"
    ".v11u-controls select,.v11u-controls input{font:inherit;padding:3px 6px;border:1px solid #cbd5e1;"
    "border-radius:4px;background:#fff;color:#0f172a;}"
    ".v11u-controls input{width:175px;}"
    ".v11u-controls code{background:#eef2ff;padding:0 3px;border-radius:3px;}"
    ".v11u-controls button{font:inherit;padding:3px 10px;border:1px solid #2563eb;background:#2563eb;"
    "color:#fff;border-radius:4px;cursor:pointer;}"
    ".v11u-controls button:hover{background:#1d4ed8;}"
    ".v11u-sep{color:#94a3b8;}"
    ".v11u-status{color:#64748b;font-size:0.78rem;}"
    ".v11u-wrap{position:relative;max-width:920px;}"
    ".v11u-svg{width:100%;height:auto;border:1px solid var(--border,#d8dee6);border-radius:6px;background:#fff;}"
    ".v11u-dot{cursor:pointer;}"
    ".v11u-dot:hover{stroke:#111827;stroke-width:1.8;}"
    ".v11u-splabel{font:600 11px/1.1 system-ui,'Segoe UI',Arial;fill:#111827;paint-order:stroke;"
    "stroke:#fff;stroke-width:3px;pointer-events:none;}"
    ".v11u-axis{font:600 11px/1 system-ui;fill:#6b7280;}"
    ".v11u-title{font:600 12px system-ui;fill:#334155;}"
    ".v11u-legend{display:flex;flex-wrap:wrap;gap:8px 14px;margin-top:10px;font-size:0.82rem;}"
    ".v11u-leg{display:inline-flex;align-items:center;gap:5px;}"
    ".v11u-sw{display:inline-block;width:11px;height:11px;border:1px solid #c0c4cc;border-radius:2px;}"
    ".v11u-cnt{color:var(--muted,#888);}"
    ".v11u-tip{position:absolute;pointer-events:none;z-index:40;max-width:300px;background:#0f172a;"
    "color:#f1f5f9;border-radius:6px;padding:8px 10px;font-size:0.8rem;line-height:1.35;"
    "box-shadow:0 4px 14px rgba(15,23,42,.32);}"
    ".v11u-tip b{color:#fff;}"
    ".v11u-tfeat{color:#cbd5e1;margin-top:4px;display:block;font-size:0.75rem;}"
)

# Interactive region-selectable scatter. Reads the embedded V11_umap_regions
# JSON: precomputed real-UMAP coords per region (dropdown) + per-species residue
# strings so any custom residue range recomputes a live PCA in-browser.
_V11_UMAP_JS = r'''<script>(function(){
  var dEl=document.getElementById('v11u-data'), cEl=document.getElementById('v11u-colors');
  if(!dEl||!cEl)return;
  var D=JSON.parse(dEl.textContent), COLORS=JSON.parse(cEl.textContent);
  var SP=D.species, SEQS=D.seqs||[], KD=D.hydropathy||{}, REF=D.ref_length||0, HI=D.human_index||0;
  var GAP={'-':1,'.':1,'X':1,'?':1,' ':1,'*':1};
  var SPOT={homo_sapiens:1,mus_musculus:1,danio_rerio:1};
  var W=880,H=520,PAD=46;
  var svg=document.getElementById('v11u-svg'), tip=document.getElementById('v11u-tip');
  var wrap=document.querySelector('#v11-umap-panel .v11u-wrap');
  var statusEl=document.getElementById('v11u-status');
  function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,function(m){return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[m];});}
  function render(points,label){
    var n=points.length; if(!n){svg.innerHTML='';return;}
    var xmin=1/0,xmax=-1/0,ymin=1/0,ymax=-1/0,k,p;
    for(k=0;k<n;k++){p=points[k];if(p.x<xmin)xmin=p.x;if(p.x>xmax)xmax=p.x;if(p.y<ymin)ymin=p.y;if(p.y>ymax)ymax=p.y;}
    if(xmax-xmin<1e-9)xmax=xmin+1; if(ymax-ymin<1e-9)ymax=ymin+1;
    function sx(x){return PAD+(x-xmin)/(xmax-xmin)*(W-2*PAD);}
    function sy(y){return (H-PAD)-(y-ymin)/(ymax-ymin)*(H-2*PAD);}
    var out=['<rect x="0" y="0" width="'+W+'" height="'+H+'" fill="#fff"/>'];
    out.push('<text x="'+PAD+'" y="'+(H-12)+'" class="v11u-axis">dim 1 →</text>');
    out.push('<text x="16" y="'+PAD+'" class="v11u-axis" transform="rotate(-90 16 '+PAD+')">dim 2 →</text>');
    out.push('<text x="'+(W/2)+'" y="20" text-anchor="middle" class="v11u-title">'+esc(label)+'</text>');
    var labs=[];
    for(k=0;k<n;k++){
      p=points[k]; var meta=SP[p.i]||{}, sp=meta.species||'', clade=meta.clade||'';
      var col=COLORS[clade]||'#888', spot=SPOT[String(sp).toLowerCase()]?1:0;
      var cx=sx(p.x), cy=sy(p.y);
      out.push('<circle class="v11u-dot" cx="'+cx.toFixed(1)+'" cy="'+cy.toFixed(1)+'" r="'+(spot?6.5:4.2)+'" fill="'+col+'" stroke="#1f2937" stroke-width="'+(spot?1.6:0.5)+'" data-i="'+p.i+'" data-rid="'+(p.rid!=null?p.rid:'')+'"></circle>');
      if(spot)labs.push('<text class="v11u-splabel" x="'+(cx+8).toFixed(1)+'" y="'+(cy-8).toFixed(1)+'">'+esc(sp)+'</text>');
    }
    svg.innerHTML=out.join('')+labs.join(''); svg.setAttribute('data-label',label);
  }
  function showTip(e){
    var c=e.target; if(!c||String(c.tagName).toLowerCase()!=='circle')return;
    var meta=SP[+c.getAttribute('data-i')]||{}, rid=c.getAttribute('data-rid');
    var col=c.getAttribute('fill')||'#888', label=svg.getAttribute('data-label')||'';
    tip.innerHTML='<b>'+esc(meta.species)+'</b> <span style="font-weight:600;color:'+col+'">'+esc(meta.clade)+'</span>'
      +'<br>protein: <b>'+esc(meta.protein_id)+'</b>'
      +'<br><span style="color:#94a3b8">'+esc(label)+'</span>'
      +((rid!==''&&rid!=null)?'<span class="v11u-tfeat">matches human at '+Math.round(parseFloat(rid)*100)+'% of this region</span>':'');
    tip.hidden=false; moveTip(e);
  }
  function moveTip(e){var r=wrap.getBoundingClientRect();var tx=e.clientX-r.left+14,ty=e.clientY-r.top+14,tw=tip.offsetWidth,th=tip.offsetHeight;if(tx+tw>r.width)tx=e.clientX-r.left-tw-14;if(ty+th>r.height)ty=Math.max(2,r.height-th-4);tip.style.left=tx+'px';tip.style.top=ty+'px';}
  function hideTip(e){if(e.target&&String(e.target.tagName).toLowerCase()==='circle')tip.hidden=true;}
  svg.addEventListener('mouseover',showTip);
  svg.addEventListener('mousemove',function(e){if(!tip.hidden)moveTip(e);});
  svg.addEventListener('mouseout',hideTip);
  function parseRanges(str){
    var cols=[],seen={};
    (str||'').split(',').forEach(function(tok){
      tok=tok.trim(); if(!tok)return; var mm=tok.split('-');
      var a=parseInt(mm[0],10), b=(mm.length>1?parseInt(mm[1],10):a);
      if(isNaN(a))return; if(isNaN(b))b=a; if(b<a){var t=a;a=b;b=t;}
      a=Math.max(1,a); b=Math.min(REF,b);
      for(var q=a;q<=b;q++){if(!seen[q]){seen[q]=1;cols.push(q-1);}}
    });
    return cols;
  }
  function topEig(G,n){
    var v=new Float64Array(n),i,j; for(i=0;i<n;i++)v[i]=Math.sin(i*12.9898+1.0);
    var lam=0;
    for(var it=0;it<140;it++){
      var w=new Float64Array(n);
      for(i=0;i<n;i++){var gi=G[i],s=0;for(j=0;j<n;j++)s+=gi[j]*v[j];w[i]=s;}
      var nrm=0;for(i=0;i<n;i++)nrm+=w[i]*w[i];nrm=Math.sqrt(nrm)||1;
      for(i=0;i<n;i++)w[i]/=nrm; lam=nrm; v=w;
    }
    return {vec:v,val:lam};
  }
  function livePCA(cols){
    var n=SEQS.length, hum=SEQS[HI]||'', m=cols.length, D2=m*2, i,j,k,d;
    var X=new Array(n), rid=new Array(n);
    for(i=0;i<n;i++){
      var s=SEQS[i], row=new Float64Array(D2), ids=0,cnt=0;
      for(k=0;k<m;k++){
        var c=cols[k];
        var sc=(c<s.length?s[c]:'-').toUpperCase(), hc=(c<hum.length?hum[c]:'-').toUpperCase();
        var ident=(!GAP[sc]&&sc===hc)?1:0;
        row[2*k]=ident; row[2*k+1]=(KD[sc]||0)/4.5;
        if(!GAP[hc]){cnt++; ids+=ident;}
      }
      X[i]=row; rid[i]=cnt?ids/cnt:0;
    }
    for(d=0;d<D2;d++){
      var mean=0;for(i=0;i<n;i++)mean+=X[i][d];mean/=n;
      var vv=0;for(i=0;i<n;i++){var tt=X[i][d]-mean;vv+=tt*tt;}vv=Math.sqrt(vv/n)||1;
      for(i=0;i<n;i++)X[i][d]=(X[i][d]-mean)/vv;
    }
    var G=new Array(n);for(i=0;i<n;i++)G[i]=new Float64Array(n);
    for(i=0;i<n;i++){for(j=i;j<n;j++){var s2=0,Xi=X[i],Xj=X[j];for(d=0;d<D2;d++)s2+=Xi[d]*Xj[d];G[i][j]=s2;G[j][i]=s2;}}
    var e1=topEig(G,n);
    for(i=0;i<n;i++)for(j=0;j<n;j++)G[i][j]-=e1.val*e1.vec[i]*e1.vec[j];
    var e2=topEig(G,n);
    var s1=Math.sqrt(Math.max(e1.val,0)), s2b=Math.sqrt(Math.max(e2.val,0));
    var pts=new Array(n);
    for(i=0;i<n;i++)pts[i]={i:i,x:e1.vec[i]*s1,y:e2.vec[i]*s2b,rid:Math.round(rid[i]*1000)/1000};
    return pts;
  }
  var regionSel=document.getElementById('v11u-region');
  function drawRegion(idx){var r=D.regions[idx];if(!r)return;render(r.points,r.name+' — '+(r.method||'UMAP'));statusEl.textContent='precomputed '+(r.method||'UMAP')+' · '+r.name;}
  regionSel.addEventListener('change',function(){drawRegion(+regionSel.value);});
  document.getElementById('v11u-go').addEventListener('click',function(){
    var cols=parseRanges(document.getElementById('v11u-range').value);
    if(cols.length<1){statusEl.textContent='enter residues, e.g. 263-269 or 6-122,228';return;}
    statusEl.textContent='computing live PCA over '+cols.length+' residues…';
    setTimeout(function(){
      try{var pts=livePCA(cols);regionSel.selectedIndex=-1;render(pts,'custom: '+cols.length+' residues (live PCA)');statusEl.textContent='live PCA · '+cols.length+' residues';}
      catch(err){statusEl.textContent='PCA failed: '+err.message;}
    },10);
  });
  document.getElementById('v11u-range').addEventListener('keydown',function(e){if(e.key==='Enter'){e.preventDefault();document.getElementById('v11u-go').click();}});
  drawRegion(0);
})();</script>'''


def v11_umap_panel_markup(outdir: Optional[Path]) -> str:
    """Orthologue-UMAP panel for the bottom of the alignment browser.

    Prefers the region-selectable interactive panel (precomputed real UMAP per
    domain via a dropdown + live client-side PCA for any custom residue range,
    from V11_umap_regions.json). Falls back to the static full-protein scatter
    (V11_orthologue_clusters.csv) when the region JSON is absent.
    """
    if not outdir:
        return ""
    outdir = Path(outdir)
    regions_json = outdir / "V11_umap_regions.json"
    if regions_json.exists():
        try:
            panel = _v11_umap_panel_interactive(regions_json)
            if panel:
                return panel
        except Exception:  # noqa: BLE001
            pass
    clusters_csv = outdir / "V11_orthologue_clusters.csv"
    if clusters_csv.exists():
        return _v11_umap_panel_static(clusters_csv)
    return ""


def _v11_umap_panel_interactive(regions_json: Path) -> str:
    import html as _html
    import json as _json
    raw = regions_json.read_text(encoding="utf-8")
    data = _json.loads(raw)
    regions = data.get("regions") or []
    species = data.get("species") or []
    if not regions or not species:
        return ""
    ref_len = int(data.get("ref_length") or 0)

    clades: List[str] = []
    counts: Dict[str, int] = {}
    for s in species:
        c = str(s.get("clade") or "")
        counts[c] = counts.get(c, 0) + 1
        if c and c not in clades:
            clades.append(c)
    clades.sort(key=lambda c: (_V11_GROUPED_BUBBLE_CLADE_ORDER.index(c)
                               if c in _V11_GROUPED_BUBBLE_CLADE_ORDER else 99))
    colors = {c: _v11_clade_color(c) for c in clades}

    region_options = "".join(
        f'<option value="{i}">{_html.escape(str(r.get("name")))}</option>'
        for i, r in enumerate(regions))
    legend_items = "".join(
        f'<span class="v11u-leg"><span class="v11u-sw" style="background:{colors[c]};"></span>'
        f'{_html.escape(c)} <span class="v11u-cnt">({counts.get(c, 0)})</span></span>'
        for c in clades)

    data_script = (
        '<script id="v11u-data" type="application/json">'
        + raw.replace("</", "<\\/") + "</script>"
        + '<script id="v11u-colors" type="application/json">'
        + _json.dumps(colors) + "</script>")

    return (
        '<section class="alphafold-structure-panel v11-umap-panel" id="v11-umap-panel" '
        'style="margin-top:14px;">'
        '<details open>'
        '<summary style="cursor:pointer;font-weight:700;font-size:0.95rem;">'
        f'Orthologue UMAP — region-selectable ({len(species)} orthologs, {ref_len} ref residues)'
        '</summary>'
        '<p class="muted" style="margin:8px 0;font-size:0.85rem;">'
        "Each dot is one species' ortholog. <strong>Pick a region</strong> to load its "
        'precomputed <em>real UMAP</em> (full protein or an annotated domain), or '
        '<strong>type residues</strong> (e.g. <code>263-269</code> or <code>6-122,228</code>) and '
        'hit Draw to recompute a <em>live PCA</em> over exactly those positions across all species. '
        'Dots that drift from their clade colour diverge in that region. Hover any dot for its '
        'species, clade, protein accession, and how much of the region matches the human residue.'
        '</p>'
        "<style>" + _V11_UMAP_CSS + "</style>"
        '<div class="v11u-controls">'
        '<label>Region <select id="v11u-region">' + region_options + '</select></label>'
        '<span class="v11u-sep">or residues</span>'
        '<input id="v11u-range" type="text" placeholder="263-269 or 6-122,228" '
        'aria-label="custom residue range" />'
        '<button type="button" id="v11u-go">Draw (live PCA)</button>'
        '<span id="v11u-status" class="v11u-status"></span>'
        '</div>'
        '<div class="v11u-wrap">'
        '<svg id="v11u-svg" class="v11u-svg" viewBox="0 0 880 520" '
        'preserveAspectRatio="xMidYMid meet" role="img" aria-label="Orthologue embedding scatter"></svg>'
        '<div class="v11u-tip" id="v11u-tip" hidden></div></div>'
        f'<div class="v11u-legend">{legend_items}</div>'
        '</details></section>'
        + data_script + _V11_UMAP_JS
    )


def build_alignment_browser_html(payload: Dict[str, Any]) -> str:
    viewer_js = str(payload.get("alphafold_viewer_js") or "")
    payload_for_json = dict(payload)
    payload_for_json.pop("alphafold_viewer_js", None)
    payload_for_json.pop("v11_per_clade_ss_csv_path", None)  # Path object; not JSON-safe and consumed server-side.
    payload_json = json.dumps(payload_for_json, ensure_ascii=False).replace("</script>", "<\\/script>")
    viewer_js = viewer_js.replace("</script>", "<\\/script>")
    title = escape(str(payload.get("title") or "Interactive alignment browser"))
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__ALIGNMENT_BROWSER_TITLE__</title>
  <script>__ALPHAFOLD_VIEWER_JS__</script>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #18202a;
      --muted: #617083;
      --line: #d8dee8;
      --accent: #0f766e;
      --accent-strong: #0b4f4a;
      --warn: #b42318;
      --label-width: 282px;
      --cell-size: 18px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", Tahoma, sans-serif;
    }
__ALPHAFOLD_STRUCTURE_CSS__
    .page {
      width: min(100% - 28px, 1680px);
      margin: 0 auto;
      padding: 18px 0 28px;
    }
    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 14px;
    }
    h1 {
      margin: 0;
      font-size: 1.55rem;
      font-weight: 750;
      letter-spacing: 0;
    }
    .meta {
      color: var(--muted);
      font-size: 0.92rem;
      text-align: right;
    }
    .toolbar, .metrics, .tree-panel, .snapshot-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    input, select {
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 6px 8px;
      font: inherit;
      font-size: 0.92rem;
      text-transform: none;
      font-weight: 500;
    }
    .range-pair {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }
    .metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
      color: var(--muted);
      font-size: 0.92rem;
    }
    .metric {
      display: inline-flex;
      gap: 6px;
      align-items: baseline;
      padding: 5px 8px;
      border-radius: 6px;
      background: #eef3f8;
    }
    .metric strong {
      color: var(--ink);
      font-size: 0.98rem;
    }
    .tree-panel {
      margin-bottom: 12px;
    }
    .tree-panel summary {
      cursor: pointer;
      font-weight: 750;
    }
    .tree-controls {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      align-items: end;
      margin-top: 10px;
    }
    .tree-tools {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      justify-content: flex-start;
    }
    .tree-filter-label {
      display: flex;
      min-height: 34px;
      gap: 8px;
      align-items: center;
      color: var(--ink);
      font-size: 0.9rem;
      font-weight: 600;
      text-transform: none;
    }
    .tree-filter-label input {
      width: auto;
      min-height: 0;
    }
    .tree-controls button {
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fafc;
      color: var(--ink);
      padding: 6px 10px;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    .tree-zoom-controls {
      display: inline-flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
      min-height: 34px;
      padding: 4px 6px;
      border: 1px solid #dbe4ee;
      border-radius: 6px;
      background: #f8fafc;
    }
    .tree-zoom-controls button {
      min-height: 28px;
      padding: 4px 9px;
    }
    .tree-zoom-value {
      min-width: 58px;
      color: #475569;
      font-size: 0.86rem;
      font-weight: 750;
      text-align: center;
    }
    .tree-status {
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.9rem;
    }
    .tree-scale-note {
      margin-top: 6px;
      color: #64748b;
      font-size: 0.82rem;
      line-height: 1.35;
    }
    .tree-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .tree-legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 8px;
      border-radius: 999px;
      background: #eef3f8;
      color: #334155;
      font-size: 0.82rem;
      font-weight: 700;
    }
    .tree-legend-swatch {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      flex: 0 0 auto;
    }
    .tree-legend-note {
      flex-basis: 100%;
      color: #64748b;
      font-size: 0.8rem;
      line-height: 1.35;
    }
    .tree-svg-wrap {
      margin-top: 10px;
      overflow: auto;
      min-height: 240px;
      max-height: 440px;
      border: 1px solid #eef2f6;
      border-radius: 6px;
      background: #fbfdff;
    }
    .tree-svg {
      display: block;
      max-width: none;
    }
    .tree-link {
      fill: none;
      stroke: #94a3b8;
      stroke-width: 1.4;
    }
    .tree-node {
      cursor: pointer;
    }
    .tree-node circle {
      fill: var(--tree-node-fill, #fff);
      stroke: var(--tree-node-stroke, var(--accent));
      stroke-width: 1.5;
    }
    .tree-node.selected circle {
      fill: var(--tree-node-selected-fill, #ccfbf1);
      stroke: var(--tree-node-selected-stroke, var(--accent-strong));
      stroke-width: 2.3;
    }
    .tree-node.search-match circle {
      fill: var(--tree-node-search-fill, #fef3c7);
      stroke: var(--tree-node-search-stroke, #b45309);
      stroke-width: 2;
    }
    .tree-label {
      fill: var(--tree-label-fill, var(--ink));
      font: 600 11px "Segoe UI", Tahoma, sans-serif;
    }
    .tree-node-popup {
      position: fixed;
      inset: 0;
      z-index: 80;
      display: grid;
      place-items: center;
      padding: 24px;
      background: rgba(15, 23, 42, 0.35);
      font-family: "Segoe UI", Tahoma, sans-serif;
    }
    .tree-node-popup[hidden] {
      display: none;
    }
    .tree-node-popup-card {
      width: min(960px, calc(100vw - 32px));
      max-height: min(760px, calc(100vh - 48px));
      display: flex;
      flex-direction: column;
      overflow: hidden;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      background: #ffffff;
      box-shadow: 0 24px 70px rgba(15, 23, 42, 0.24);
    }
    .tree-node-popup-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      padding: 14px 16px 10px;
      border-bottom: 1px solid #e5e7eb;
    }
    .tree-node-popup-title {
      min-width: 0;
    }
    .tree-node-popup-title strong {
      display: block;
      color: var(--ink);
      font-size: 1rem;
    }
    .tree-node-popup-title span {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 0.82rem;
      line-height: 1.35;
    }
    .tree-node-popup-actions {
      display: flex;
      gap: 8px;
      flex: 0 0 auto;
    }
    .tree-node-popup-actions button {
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #f8fafc;
      color: #0f172a;
      font-size: 0.78rem;
      font-weight: 750;
      padding: 6px 9px;
      cursor: pointer;
    }
    .tree-node-popup-actions button:hover {
      border-color: var(--accent);
      background: #ecfdf5;
    }
    .tree-node-popup-table-wrap {
      overflow: auto;
    }
    .tree-node-popup table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
    }
    .tree-node-popup th,
    .tree-node-popup td {
      padding: 8px 10px;
      border-bottom: 1px solid #edf2f7;
      text-align: left;
      vertical-align: top;
    }
    .tree-node-popup th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f8fafc;
      color: #475569;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .tree-node-popup .identity {
      font-weight: 800;
      font-variant-numeric: tabular-nums;
      color: #0f172a;
      white-space: nowrap;
    }
    .tree-node-popup .protein-id,
    .tree-node-popup .record-id {
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.76rem;
      color: #475569;
      word-break: break-all;
    }
    .tree-node-popup .species-name {
      font-weight: 800;
      color: #111827;
    }
    .tree-node-popup .species-meta {
      margin-top: 2px;
      color: #64748b;
      font-size: 0.76rem;
    }
    .tree-scale line {
      stroke: #64748b;
      stroke-width: 1.5;
    }
    .tree-scale text {
      fill: #475569;
      font: 650 10px "Segoe UI", Tahoma, sans-serif;
    }
    .snapshot-panel {
      margin-bottom: 12px;
      font-family: "Segoe UI", Tahoma, sans-serif;
    }
    .snapshot-topbar {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 10px;
    }
    .snapshot-title h2 {
      margin: 0;
      font-size: 1.15rem;
      font-weight: 750;
      letter-spacing: 0;
    }
    .snapshot-title p {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.4;
    }
    .snapshot-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: flex-start;
      justify-content: flex-end;
    }
    .snapshot-export, .snapshot-remove {
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--accent);
      padding: 8px 12px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    .snapshot-export[disabled], .snapshot-remove[disabled] {
      opacity: 0.55;
      cursor: not-allowed;
    }
    .snapshot-search-label {
      margin-bottom: 8px;
    }
    .snapshot-selected-list,
    .snapshot-candidate-list {
      border: 1px solid #dbe4ee;
      border-radius: 8px;
      background: #fff;
      overflow: auto;
    }
    .snapshot-selected-list {
      margin-bottom: 10px;
      max-height: 190px;
    }
    .snapshot-candidate-list {
      max-height: 270px;
    }
    .snapshot-empty-state {
      padding: 11px 12px;
      color: #617083;
      font-size: 0.94rem;
      font-weight: 750;
      text-transform: uppercase;
      letter-spacing: 0;
      background: #f8fbff;
    }
    .snapshot-selected-row,
    .snapshot-candidate-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
      width: 100%;
      padding: 9px 10px;
      border: 0;
      border-bottom: 1px solid #eef2f6;
      background: #fff;
      text-align: left;
      font: inherit;
    }
    .snapshot-candidate-row {
      cursor: pointer;
    }
    .snapshot-candidate-row:hover {
      background: #ecfdf5;
    }
    .snapshot-selected-row:last-child,
    .snapshot-candidate-row:last-child {
      border-bottom: 0;
    }
    .snapshot-row-main {
      min-width: 0;
    }
    .snapshot-row-head {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
    }
    .snapshot-row-name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--ink);
      font-size: 0.98rem;
      font-weight: 750;
    }
    .snapshot-row-badge {
      display: inline-flex;
      align-items: center;
      padding: 2px 7px;
      border-radius: 999px;
      background: #dcfce7;
      color: #166534;
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 0;
      text-transform: uppercase;
      flex: 0 0 auto;
    }
    .snapshot-row-meta {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--muted);
      font-size: 0.88rem;
      font-weight: 650;
    }
    .snapshot-row-detail {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #7b8794;
      font-size: 0.79rem;
      font-weight: 600;
    }
    .snapshot-footer {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 12px;
      margin-top: 10px;
    }
    .snapshot-footer-note {
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.35;
    }
    .snapshot-residues-label {
      min-width: 180px;
    }
    .snapshot-residues-label input {
      max-width: 120px;
      justify-self: end;
    }
    .snapshot-highlight-label {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.85rem;
      color: #334155;
      cursor: pointer;
    }
    .snapshot-highlight-status {
      font-size: 0.8rem;
      font-weight: 600;
      color: #b91c6b;
    }
    .alignment-shell {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }
    .alignment-grid {
      --col-count: 1;
      overflow: auto;
      max-height: calc(100vh - 290px);
      min-height: 360px;
      font-family: Consolas, "Courier New", monospace;
      font-size: 11px;
      line-height: 1;
    }
    .ruler, .alignment-row, .consensus-row {
      display: grid;
      grid-template-columns: var(--label-width) repeat(var(--col-count), var(--cell-size));
      min-width: max-content;
    }
    /* Large-content engine: let the browser skip layout + paint for off-screen
       alignment rows (the full grid is often 150k+ cells). Rows are a uniform
       23px tall inside the scroll container; the `auto` keyword remembers each
       row's real size after its first paint so horizontal scroll width and the
       scrollbar stay stable. Unsupported browsers ignore this and render as
       before. The sticky ruler is excluded so it always stays pinned. */
    .alignment-row, .consensus-row {
      content-visibility: auto;
      contain-intrinsic-size: auto 1200px auto 23px;
    }
    .ruler {
      position: sticky;
      top: 0;
      z-index: 4;
      background: #edf2f7;
      border-bottom: 1px solid var(--line);
      color: #334155;
      font-family: "Segoe UI", Tahoma, sans-serif;
      font-size: 10px;
      font-weight: 700;
    }
    .row-label, .ruler-label {
      position: sticky;
      left: 0;
      z-index: 3;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      border-right: 1px solid var(--line);
      background: #fff;
    }
    .ruler-label {
      z-index: 5;
      background: #edf2f7;
      padding: 6px 8px;
    }
    .row-label {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      min-height: 23px;
      padding: 3px 8px;
      border-bottom: 1px solid #eef2f6;
      font-family: "Segoe UI", Tahoma, sans-serif;
      font-size: 11px;
    }
    .row-name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--ink);
      font-weight: 650;
    }
    .row-stats {
      color: var(--muted);
      font-size: 10px;
    }
    .aa-cell, .ruler-cell {
      width: var(--cell-size);
      height: 23px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-right: 1px solid rgba(24,32,42,0.08);
      border-bottom: 1px solid rgba(24,32,42,0.08);
    }
    .ruler-cell {
      color: #475569;
      background: #edf2f7;
      writing-mode: vertical-rl;
      text-orientation: mixed;
      overflow: hidden;
    }
    .aa-cell.gap {
      color: #9aa6b2;
    }
    .aa-cell.consensus {
      font-weight: 800;
      color: #111827;
    }
    .aa-cell.low-support {
      opacity: 0.72;
    }
    .aa-cell.compare-window {
      box-shadow: inset 0 0 0 2px rgba(37,99,235,0.82);
    }
    .aa-cell.compare-window.property {
      box-shadow: inset 0 0 0 2px rgba(124,58,237,0.78);
    }
    .aa-cell.mismatch {
      box-shadow: inset 0 -2px 0 var(--warn);
    }
    .aa-cell.mismatch.compare-window {
      box-shadow: inset 0 -2px 0 var(--warn), inset 0 0 0 2px rgba(37,99,235,0.82);
    }
    .aa-cell.mismatch.compare-window.property {
      box-shadow: inset 0 -2px 0 var(--warn), inset 0 0 0 2px rgba(124,58,237,0.78);
    }
    .aa-cell.reference {
      font-weight: 800;
      outline: 1px solid rgba(15,118,110,0.55);
      outline-offset: -1px;
    }
    .consensus-row .row-label {
      background: #f8fafc;
      border-bottom-color: #dbeafe;
    }
    .row-label.offset-suspect {
      background: #fff7ed;
    }
    .row-label.group-outlier {
      background: #fef2f2;
    }
    .row-label.tree-row-selected {
      background: #ecfdf5;
      box-shadow: inset 3px 0 0 var(--accent);
    }
    .row-label.tree-row-match {
      box-shadow: inset 0 0 0 2px rgba(180,83,9,0.35);
    }
    .badge {
      display: inline-flex;
      align-items: center;
      max-width: 92px;
      margin-left: 5px;
      padding: 1px 5px;
      border-radius: 5px;
      font-size: 9px;
      font-weight: 800;
      line-height: 1.4;
      color: #374151;
      background: #e5e7eb;
      vertical-align: middle;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .badge.offset {
      color: #7c2d12;
      background: #fed7aa;
    }
    .badge.outlier {
      color: #7f1d1d;
      background: #fecaca;
    }
    .badge.compare {
      color: #1e3a8a;
      background: #bfdbfe;
    }
    .group-row {
      position: sticky;
      left: 0;
      z-index: 2;
      min-width: max-content;
      background: #e8f3f2;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      font-family: "Segoe UI", Tahoma, sans-serif;
    }
    .group-row button {
      border: 0;
      background: transparent;
      color: var(--accent-strong);
      font: inherit;
      font-weight: 750;
      padding: 7px 9px;
      cursor: pointer;
    }
    .group-meta {
      color: #617083;
      font-size: 11px;
      font-weight: 650;
      margin-left: 8px;
    }
    .run-architecture-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 12px;
      font-family: "Segoe UI", Tahoma, sans-serif;
    }
    .run-architecture-panel[hidden] {
      display: none;
    }
    .run-architecture-panel.collapsed {
      padding-bottom: 10px;
    }
    .run-architecture-panel.collapsed .architecture-body {
      display: none;
    }
    .architecture-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 9px;
      color: var(--muted);
      font-size: 0.84rem;
    }
    .run-architecture-panel.collapsed .architecture-head {
      margin-bottom: 0;
    }
    .architecture-title {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      min-width: 0;
      flex: 1 1 auto;
    }
    .architecture-head strong {
      color: var(--ink);
      font-size: 0.92rem;
    }
    .architecture-selected {
      min-height: 1.2em;
      color: var(--accent-strong);
      font-weight: 750;
      text-align: right;
    }
    .architecture-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      justify-content: flex-end;
    }
    .architecture-export {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fafc;
      color: var(--accent-strong);
      cursor: pointer;
      font: inherit;
      font-size: 0.78rem;
      font-weight: 750;
      padding: 5px 8px;
    }
    .architecture-export:hover {
      border-color: var(--accent);
      background: #ecfdf5;
    }
    .architecture-toggle {
      border-color: var(--accent);
      background: #ecfdf5;
    }
    .architecture-toggle[aria-expanded="true"] {
      background: #fff7ed;
      border-color: #fed7aa;
      color: #9a3412;
    }
    .architecture-note {
      margin: 2px 0 10px;
      color: var(--muted);
      font-size: 0.79rem;
      line-height: 1.4;
    }
    .architecture-row {
      display: grid;
      grid-template-columns: minmax(140px, 220px) minmax(180px, 1fr) auto;
      gap: 10px;
      align-items: center;
      min-height: 24px;
      margin-top: 7px;
    }
    .architecture-row.annotation {
      min-height: 36px;
      margin-top: 5px;
    }
    .architecture-label {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--ink);
      font-size: 0.82rem;
      font-weight: 700;
    }
    .architecture-row.annotation .architecture-label {
      color: #475569;
      font-size: 0.78rem;
      font-weight: 750;
    }
    .architecture-track {
      position: relative;
      height: 15px;
      border: 1px solid #d0d7e2;
      border-radius: 3px;
      background: #e5e7eb;
      overflow: hidden;
    }
    .architecture-track.annotation-track {
      height: 22px;
      border-radius: 5px;
      background: #eef2f7;
      overflow: visible;
    }
    .architecture-track.domain-track {
      height: 18px;
      overflow: hidden;
    }
    .architecture-ruler-tick {
      position: absolute;
      top: -1px;
      bottom: -1px;
      width: 1px;
      background: #64748b;
      opacity: 0.72;
      pointer-events: none;
    }
    .architecture-ruler-label {
      position: absolute;
      top: -18px;
      transform: translateX(-50%);
      color: #475569;
      font-size: 0.68rem;
      font-weight: 700;
      white-space: nowrap;
      pointer-events: none;
    }
    .architecture-ruler-label.start {
      transform: translateX(0);
    }
    .architecture-ruler-label.end {
      transform: translateX(-100%);
    }
    .architecture-domain-band {
      position: absolute;
      top: 1px;
      bottom: 1px;
      min-width: 2px;
      box-sizing: border-box;
      border-radius: 4px;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0 6px;
      color: #ffffff;
      font-size: 0.7rem;
      font-weight: 800;
      line-height: 1;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      pointer-events: none;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.12);
    }
    .architecture-site-label {
      position: absolute;
      top: -10px;
      transform: translateX(-50%);
      z-index: 2;
      pointer-events: none;
      font-size: 0.58rem;
      font-weight: 900;
      line-height: 1;
      white-space: nowrap;
      text-shadow: 0 1px 0 #ffffff, 1px 0 0 #ffffff, -1px 0 0 #ffffff, 0 -1px 0 #ffffff;
    }
    .architecture-secondary-svg {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      overflow: visible;
    }
    .architecture-secondary-svg .architecture-svg-ss-label {
      display: none;
    }
    .architecture-secondary-svg .architecture-comparative-lane-label {
      font: 800 8px "Segoe UI", Tahoma, sans-serif;
      fill: #334155;
      paint-order: stroke;
      stroke: #ffffff;
      stroke-width: 2px;
      stroke-linejoin: round;
      pointer-events: none;
    }
    .architecture-secondary-svg .architecture-comparative-lane-id {
      font: 700 7px "Segoe UI", Tahoma, sans-serif;
      fill: #64748b;
      paint-order: stroke;
      stroke: #ffffff;
      stroke-width: 2px;
      stroke-linejoin: round;
      pointer-events: none;
    }
    .architecture-ss-number {
      position: absolute;
      top: -1px;
      transform: translateX(-50%);
      z-index: 2;
      pointer-events: none;
      font-size: 0.56rem;
      font-weight: 900;
      line-height: 1;
      paint-order: stroke;
      text-shadow: 0 1px 0 #ffffff, 1px 0 0 #ffffff, -1px 0 0 #ffffff, 0 -1px 0 #ffffff;
    }
    .architecture-comparative-empty {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      color: #64748b;
      font-size: 0.68rem;
      font-weight: 800;
      pointer-events: none;
    }
    .architecture-segment {
      position: absolute;
      top: 0;
      bottom: 0;
      min-width: 2px;
      border: 0;
      border-radius: 2px;
      background: #2563eb;
      cursor: pointer;
    }
    .architecture-segment.property {
      background: #7c3aed;
    }
    .architecture-segment.selected {
      outline: 2px solid #0f172a;
      outline-offset: -2px;
    }
    .architecture-count {
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 700;
      white-space: nowrap;
    }
    .architecture-count.range {
      font-variant-numeric: tabular-nums;
    }
    .evolutionary-divergence-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 12px;
      font-family: "Segoe UI", Tahoma, sans-serif;
    }
    .evolutionary-divergence-panel[hidden] {
      display: none;
    }
    .evolutionary-divergence-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }
    .evolutionary-divergence-title {
      min-width: 0;
      flex: 1 1 auto;
    }
    .evolutionary-divergence-title strong {
      display: block;
      color: var(--ink);
      font-size: 0.98rem;
    }
    .evolutionary-divergence-title span {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.35;
    }
    .evolutionary-divergence-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      justify-content: flex-end;
    }
    .evolutionary-divergence-actions a {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fafc;
      color: var(--accent-strong);
      text-decoration: none;
      font-size: 0.78rem;
      font-weight: 750;
      padding: 5px 8px;
      white-space: nowrap;
    }
    .evolutionary-divergence-actions a:hover {
      border-color: var(--accent);
      background: #ecfdf5;
    }
    .evolutionary-divergence-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 10px;
      margin-bottom: 10px;
    }
    .legend-chip {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 4px 9px;
      border: 1px solid #dbe4ee;
      border-radius: 999px;
      background: #f8fafc;
      color: var(--ink);
      font-size: 0.8rem;
      font-weight: 600;
    }
    .legend-swatch {
      width: 14px;
      height: 14px;
      border-radius: 3px;
      border: 1px solid #14213d;
      background: linear-gradient(135deg, #2a9d8f 0%, #3b82f6 100%);
    }
    .legend-line {
      width: 18px;
      border-top: 2px dashed #991b1b;
      transform: translateY(-1px);
    }
    .legend-hatch {
      width: 14px;
      height: 14px;
      border-radius: 3px;
      border: 1px solid #991b1b;
      background: repeating-linear-gradient(
        -45deg,
        #ffffff 0,
        #ffffff 4px,
        #ef4444 4px,
        #ef4444 6px
      );
    }
    .evolutionary-divergence-figure {
      display: block;
      width: 100%;
      border: 1px solid #dbe4ee;
      border-radius: 10px;
      background: #ffffff;
      overflow: hidden;
    }
    .evolutionary-divergence-figure img {
      display: block;
      width: 100%;
      height: auto;
    }
    .empty {
      padding: 24px;
      color: var(--muted);
      font-family: "Segoe UI", Tahoma, sans-serif;
    }
    @media (max-width: 850px) {
      :root { --label-width: 220px; --cell-size: 16px; }
      header { display: block; }
      .meta { text-align: left; margin-top: 6px; }
      .tree-controls { grid-template-columns: 1fr; align-items: stretch; }
      .tree-tools { flex-direction: column; align-items: stretch; }
      .tree-zoom-controls { width: 100%; justify-content: flex-start; }
      .snapshot-topbar, .snapshot-footer { flex-direction: column; align-items: stretch; }
      .evolutionary-divergence-head { flex-direction: column; }
      .evolutionary-divergence-actions { justify-content: flex-start; }
      .snapshot-actions { justify-content: flex-start; }
      .snapshot-residues-label { min-width: 0; }
      .snapshot-residues-label input { max-width: none; justify-self: stretch; }
      .alignment-grid { max-height: calc(100vh - 360px); }
    }
  </style>
</head>
<body>
  <main class="page">
    <header>
      <div>
        <h1>__ALIGNMENT_BROWSER_TITLE__</h1>
      </div>
      <div class="meta" id="run-meta"></div>
    </header>

    <section class="toolbar">
      <label>Alignment
        <select id="scope-select"></select>
      </label>
      <label>Species search
        <input id="species-search" type="search" placeholder="Species, symbol, record id">
      </label>
      <label>Clade
        <select id="clade-filter"></select>
      </label>
      <label>Taxa
        <select id="taxonomy-filter"></select>
      </label>
      <label>Sort
        <select id="sort-select">
          <option value="tree_order">Tree/evolution order</option>
          <option value="species">Species</option>
          <option value="clade">Clade</option>
          <option value="taxonomy_level">Taxa</option>
          <option value="gap_count">Gap count</option>
          <option value="identity_to_reference">Identity to reference</option>
        </select>
      </label>
      <label>Direction
        <select id="sort-direction">
          <option value="asc">Ascending</option>
          <option value="desc">Descending</option>
        </select>
      </label>
      <label>Group
        <select id="group-select">
          <option value="none">None</option>
          <option value="clade">Clade</option>
          <option value="taxonomy_level">Taxa</option>
        </select>
      </label>
      <label>View
        <select id="view-mode">
          <option value="detailed">Detailed</option>
          <option value="compressed">Compressed consensus</option>
        </select>
      </label>
      <label>Compare to human
        <select id="compare-mode">
          <option value="off">Off</option>
          <option value="exact">Exact</option>
          <option value="property">Property</option>
        </select>
      </label>
      <label>Min similar run
        <input id="compare-min-run" type="number" min="1" value="6">
      </label>
      <label>Offset correction
        <select id="offset-mode">
          <option value="flag">Flag only</option>
          <option value="visual">Visual correction</option>
        </select>
      </label>
      <label>Residue
        <input id="residue-filter" type="text" maxlength="20" placeholder="A, ST, gap">
      </label>
      <label>Gaps
        <select id="gap-filter">
          <option value="all">Any</option>
          <option value="has_gap">Has gap</option>
          <option value="no_gap">No gaps</option>
        </select>
      </label>
      <label>Reference match
        <select id="match-filter">
          <option value="all">Any</option>
          <option value="has_match">Has match</option>
          <option value="has_mismatch">Has mismatch</option>
          <option value="all_visible_match">All visible match</option>
        </select>
      </label>
      <label>Reference position
        <span class="range-pair"><input id="ref-start" type="number" min="1" placeholder="Start"><input id="ref-end" type="number" min="1" placeholder="End"></span>
      </label>
      <label>Alignment position
        <span class="range-pair"><input id="align-start" type="number" min="1" placeholder="Start"><input id="align-end" type="number" min="1" placeholder="End"></span>
      </label>
    </section>

    <section class="metrics" id="metrics"></section>
    <section class="evolutionary-divergence-panel" id="node-conservation-panel" hidden></section>
    <details class="tree-panel" id="tree-panel" open>
      <summary>Protein phylogenetic tree</summary>
      <div class="tree-controls">
        <label>Tree labels
          <select id="tree-view-mode">
            <option value="species">Species</option>
            <option value="nomenclature">Nomenclature</option>
            <option value="clades">Clades</option>
            <option value="taxa">Taxa</option>
            <option value="phyla">Phyla</option>
            <option value="broad_clades">Broad clades</option>
          </select>
        </label>
        <label>Tree spacing
          <select id="tree-layout-mode">
            <option value="classic">Classic</option>
            <option value="branch_lengths">Branch lengths</option>
          </select>
        </label>
        <label>Tree search
          <input id="tree-search" type="search" placeholder="Species, clade, record ID">
        </label>
        <label class="tree-filter-label">
          <input id="tree-filter-rows" type="checkbox">
          Filter alignment rows to selection
        </label>
        <div class="tree-tools">
          <div class="tree-zoom-controls" aria-label="Tree zoom controls">
            <button type="button" id="tree-zoom-out" aria-label="Zoom out">-</button>
            <span class="tree-zoom-value" id="tree-zoom-value">100%</span>
            <button type="button" id="tree-zoom-in" aria-label="Zoom in">+</button>
            <button type="button" id="tree-zoom-fit">Fit</button>
            <button type="button" id="tree-zoom-reset">100%</button>
          </div>
          <button type="button" id="tree-clear-selection">Clear selection</button>
        </div>
      </div>
      <div class="tree-status" id="tree-status"></div>
      <div class="tree-scale-note" id="tree-scale-note"></div>
      <div class="tree-legend" id="tree-legend"></div>
      <div class="tree-svg-wrap" id="tree-svg-wrap"></div>
    </details>
    <section class="run-architecture-panel" id="run-architecture-panel" hidden></section>
    __ALPHAFOLD_STRUCTURE_PANEL__
    __V11_PER_CLADE_SS_PANEL__
    __V11_CLADE_OVERLAY_PANEL__
    <section class="snapshot-panel" id="species-snapshot-panel"></section>
    <section class="evolutionary-divergence-panel" id="evolutionary-divergence-panel" hidden></section>

    <section class="alignment-shell">
      <div id="alignment-grid" class="alignment-grid"></div>
    </section>
    __V11_UMAP_PANEL__
  </main>

  <script id="alignment-payload" type="application/json">__ALIGNMENT_BROWSER_PAYLOAD__</script>
  <div class="tree-node-popup" id="tree-node-popup" hidden>
    <div class="tree-node-popup-card" role="dialog" aria-modal="true" aria-labelledby="tree-node-popup-title">
      <div class="tree-node-popup-head">
        <div class="tree-node-popup-title">
          <strong id="tree-node-popup-title">Selected tree node</strong>
          <span id="tree-node-popup-summary"></span>
        </div>
        <div class="tree-node-popup-actions">
          <button type="button" id="tree-node-popup-copy">Copy TSV</button>
          <button type="button" id="tree-node-popup-close">Close</button>
        </div>
      </div>
      <div class="tree-node-popup-table-wrap" id="tree-node-popup-body"></div>
    </div>
  </div>
  <script>
    const PAYLOAD = JSON.parse(document.getElementById("alignment-payload").textContent);
    const ALPHAFOLD_STRUCTURE = PAYLOAD.alphafold_structure || {};
    const DEFAULT_BROWSER_STATE = PAYLOAD.browser_default_state || {};
    const COLLAPSED_GROUPS = new Set();
    let LAST_COMPRESSED_COLLAPSE_SIGNATURE = "";
    let SELECTED_RUN_KEY = "";
    let RUN_ARCHITECTURE_COLLAPSED = true;
    let TREE_SELECTED_NODE_ID = "";
    let TREE_SELECTED_RECORDS = new Set();
    let TREE_SEARCH_MATCH_RECORDS = new Set();
    let TREE_ZOOM = 1;
    let TREE_ZOOM_MODE = "fit";
    let TREE_LAYOUT_SIGNATURE = "";
    let LAST_TREE_NODE_POPUP_ROWS = [];
    let SNAPSHOT_SELECTED_RECORDS = [];
    let SNAPSHOT_HIGHLIGHT_HD_RODENT = false;
    let SNAPSHOT_RAW_ALIGNMENT = false;
    let SNAPSHOT_SEARCH_QUERY = "";
    let SNAPSHOT_RESIDUES_PER_LINE = 100;
    const TREE_MIN_ZOOM = 0.35;
    const TREE_MAX_ZOOM = 4.5;
    const TREE_COLOR_OVERRIDES = {
      tetrapods: "#2563eb",
      dipnoi: "#0f766e",
      actinistia: "#7c3aed",
      holostei: "#ea580c",
      teleosts: "#16a34a",
      other_fish: "#0891b2",
      other_vertebrates: "#64748b",
      chordata: "#2563eb",
      arthropoda: "#dc2626",
      nematoda: "#ca8a04",
      mollusca: "#7c2d12",
      annelida: "#be185d",
      echinodermata: "#0ea5e9",
      cnidaria: "#db2777",
      fungi: "#6d28d9",
    };
    const TREE_COLOR_PALETTE = ["#2563eb", "#0f766e", "#7c3aed", "#ea580c", "#16a34a", "#dc2626", "#ca8a04", "#0891b2"];
    const GAP_CHARS = new Set(["-", "."]);
    const NON_CONSENSUS_RESIDUES = new Set(["-", ".", "X", "?"]);
    const OFFSET_RANGE = 12;
    const OFFSET_MIN_COMPARABLE = 30;
    const OFFSET_RAW_MAX = 0.70;
    const OFFSET_MIN_IMPROVEMENT = 0.20;
    const OFFSET_MIN_SHIFTED = 0.75;
    const PROPERTY_GROUPS = {
      hydrophobicity: {
        hydrophobic: new Set("AVILMFWYC".split("")),
        non_hydrophobic: new Set("RNDQEKHSTPG".split(""))
      },
      charge: {
        positive: new Set("KRH".split("")),
        negative: new Set("DE".split("")),
        neutral: new Set("AVILMFWYCNQSTPG".split(""))
      },
      polarity: {
        polar: new Set("RNDQEKHSTYC".split("")),
        nonpolar: new Set("AVILMFWPG".split(""))
      },
      size: {
        small: new Set("AGSCTPDN".split("")),
        medium: new Set("VILQEH".split("")),
        large: new Set("MFKRWY".split(""))
      },
      aromaticity: {
        aromatic: new Set("FWYH".split("")),
        non_aromatic: new Set("AVILMCGPSTNQDEKR".split(""))
      }
    };
    const FILTER_IDS = [
      "scope-select", "species-search", "clade-filter", "taxonomy-filter", "sort-select",
      "sort-direction", "group-select", "view-mode", "compare-mode", "compare-min-run",
      "offset-mode", "residue-filter", "gap-filter", "match-filter", "ref-start",
      "ref-end", "align-start", "align-end"
    ];

    function escapeHtml(value) {
      return String(value == null ? "" : value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

__ALPHAFOLD_STRUCTURE_SCRIPT__

    function treeData() {
      return PAYLOAD.tree || { available: false, message: "Tree viewer data was not built." };
    }

    function treeViewMode() {
      const node = document.getElementById("tree-view-mode");
      return node ? node.value : "species";
    }

    function treeLayoutMode() {
      const node = document.getElementById("tree-layout-mode");
      return node ? node.value : "classic";
    }

    function treeChildren(node) {
      return (node && Array.isArray(node.children)) ? node.children : [];
    }

    function treeViewField(node, mode) {
      if (mode === "nomenclature") return node.nomenclature_leaf_label || node.preferred_public_label || node.species_display_label || node.common_name || node.species || node.name || node.record_id || "";
      if (mode === "clades") return node.clade || node.broad_clade || node.phylum || node.taxonomy_level || node.species || node.name || node.record_id || "";
      if (mode === "taxa") return node.taxonomy_level || node.species || node.name || node.record_id || "";
      if (mode === "phyla") return node.phylum || node.taxonomy_level || node.species || node.name || node.record_id || "";
      if (mode === "broad_clades") return node.broad_clade || node.clade || node.phylum || node.taxonomy_level || node.species || node.name || node.record_id || "";
      return node.species || node.name || node.record_id || "";
    }

    function treeHash(text) {
      let hash = 0;
      for (const char of String(text || "")) {
        hash = ((hash << 5) - hash) + char.charCodeAt(0);
        hash |= 0;
      }
      return Math.abs(hash);
    }

    function treeColorForKey(key) {
      const normalized = String(key || "").trim().toLowerCase();
      if (!normalized) return "";
      if (TREE_COLOR_OVERRIDES[normalized]) return TREE_COLOR_OVERRIDES[normalized];
      return TREE_COLOR_PALETTE[treeHash(normalized) % TREE_COLOR_PALETTE.length];
    }

    function formatMa(value) {
      const numeric = Number(value);
      return Number.isFinite(numeric) ? `~${numeric.toFixed(0)} Ma` : "";
    }

    function treeNodeMatches(node, query) {
      if (!query) return false;
      const fields = [
        node.name, node.record_id, node.species, node.symbol, node.clade, node.taxonomy_level,
        node.phylum, node.broad_clade, node.display_label, node.nomenclature_group_label,
        node.nomenclature_leaf_label, node.preferred_public_label, node.preferred_protein_name,
        node.common_name, node.scientific_name,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return fields.includes(query);
    }

    function collectTreeTipRecords(node, records = []) {
      if (node && Array.isArray(node.tip_records) && node.tip_records.length) {
        node.tip_records.forEach((recordId) => records.push(String(recordId)));
        return records;
      }
      const children = treeChildren(node);
      if (!children.length) {
        if (node && node.record_id) records.push(String(node.record_id));
        return records;
      }
      children.forEach((child) => collectTreeTipRecords(child, records));
      return records;
    }

    function findTreeNode(node, nodeId) {
      if (!node) return null;
      if (node.id === nodeId) return node;
      for (const child of treeChildren(node)) {
        const found = findTreeNode(child, nodeId);
        if (found) return found;
      }
      return null;
    }

    function treeSearchMatches(node, query, matches = new Set()) {
      if (!node || !query) return matches;
      const children = treeChildren(node);
      if (treeNodeMatches(node, query)) {
        collectTreeTipRecords(node).forEach((recordId) => matches.add(recordId));
      }
      children.forEach((child) => treeSearchMatches(child, query, matches));
      return matches;
    }

    function buildDisplayTree(node, mode) {
      const children = treeChildren(node).map((child) => buildDisplayTree(child, mode));
      const ownLabel = treeViewField(node, mode);
      if (!children.length) {
        const recordId = node && node.record_id ? String(node.record_id) : "";
        return {
          ...node,
          display_label: formatLabel(ownLabel),
          display_group: ownLabel,
          summary_group: ownLabel,
          summary_tip_count: recordId ? 1 : 0,
          tip_records: recordId ? [recordId] : [],
          tip_groups: ownLabel ? [ownLabel] : [],
          children: [],
        };
      }

      const tipRecords = [...new Set(children.flatMap((child) => child.tip_records || []).map((recordId) => String(recordId)))];
      const tipGroups = [...new Set(children.flatMap((child) => child.tip_groups || []).filter(Boolean))];
      const nomenclatureGroup = node.nomenclature_group_label || "";
      const shouldCollapse = (
        ((mode === "clades" || mode === "broad_clades") && node.id !== "0" && tipGroups.length === 1 && tipRecords.length > 1)
        || (mode === "nomenclature" && node.id !== "0" && nomenclatureGroup && tipRecords.length > 1)
      );
      const summaryGroup = tipGroups.length === 1 ? tipGroups[0] : "";
      const displayLabel = mode === "nomenclature" && shouldCollapse && nomenclatureGroup
        ? `${formatLabel(nomenclatureGroup)}: ${tipRecords.length} homologs`
        : `${formatLabel(summaryGroup)} (${tipRecords.length})`;
      return {
        ...node,
        display_label: shouldCollapse ? displayLabel : "",
        display_group: (mode === "nomenclature" && nomenclatureGroup) ? nomenclatureGroup : (summaryGroup || ownLabel),
        summary_group: (mode === "nomenclature" && nomenclatureGroup) ? nomenclatureGroup : summaryGroup,
        summary_tip_count: tipRecords.length,
        tip_records: tipRecords,
        tip_groups: tipGroups,
        children: shouldCollapse ? [] : children,
      };
    }

    function refreshTreeSearchMatches() {
      const tree = treeData();
      const search = document.getElementById("tree-search");
      const query = search ? search.value.trim().toLowerCase() : "";
      TREE_SEARCH_MATCH_RECORDS = tree.available && tree.root ? treeSearchMatches(tree.root, query) : new Set();
    }

    function treeNodeAccent(node, mode) {
      const key = mode === "nomenclature"
        ? (node.broad_clade || node.clade || node.phylum || node.nomenclature_group_label)
        : mode === "phyla"
        ? (node.summary_group || node.phylum)
        : (mode === "clades"
          ? (node.summary_group || node.clade || node.broad_clade || node.phylum)
          : (node.summary_group || node.broad_clade || node.clade || node.phylum));
      return treeColorForKey(key);
    }

    function treeNodeLabel(node, mode) {
      if (node.display_label) return node.display_label;
      const label = treeViewField(node, mode);
      return formatLabel(label);
    }

    function treeAgeSourceLabel(source) {
      if (source === "taxonomy_inferred") return "age inferred from taxonomy";
      if (source === "phylum_anchor") return "age inferred from other anchored members of this phylum";
      return "";
    }

    function treeNodeTitle(node, mode) {
      const parts = [];
      const label = treeNodeLabel(node, mode);
      if (label) parts.push(label);
      if (node.summary_tip_count > 1) parts.push(`${node.summary_tip_count} tips`);
      if (node.preferred_protein_name && node.summary_tip_count <= 1) parts.push(`Protein: ${node.preferred_protein_name}`);
      if (node.nomenclature_event_type) parts.push(`Event: ${formatLabel(node.nomenclature_event_type)}`);
      if (node.phylum) parts.push(`Phylum: ${formatLabel(node.phylum)}`);
      if (node.broad_clade) {
        const broadText = formatLabel(node.broad_clade);
        const ageText = formatMa(node.clade_age_mya);
        parts.push(ageText ? `${broadText} ${ageText}` : broadText);
      } else {
        const ageText = formatMa(node.clade_age_mya);
        if (ageText) parts.push(ageText);
      }
      const ageSourceText = treeAgeSourceLabel(node.clade_age_source);
      if (ageSourceText && formatMa(node.clade_age_mya)) parts.push(ageSourceText);
      if (node.taxonomy_level) parts.push(`Taxa: ${formatLabel(node.taxonomy_level)}`);
      if (node.scientific_name && node.scientific_name !== node.species) parts.push(`Scientific: ${node.scientific_name}`);
      if (node.common_name) parts.push(`Common: ${node.common_name}`);
      if (node.alphafold_entry_id) parts.push(`AlphaFold: ${node.alphafold_entry_id}`);
      if (node.record_id && node.summary_tip_count <= 1) parts.push(node.record_id);
      return parts.filter(Boolean).join(" | ");
    }

    function proteinIdFromRecordId(recordId) {
      const text = String(recordId || "");
      const match = text.match(/(?:Protein|ProteinRecordID)=([^|]+)/);
      return match ? match[1] : "";
    }

    function treePopupProteinId(row) {
      const raw = String(row.protein_record_id || proteinIdFromRecordId(row.record_id) || row.alphafold_entry_id || row.uniprot_accession || "").trim();
      return raw.includes("__") ? raw.split("__").pop() : raw;
    }

    function treePopupSpeciesLabel(row) {
      return (
        row.common_name
        || row.species_display_label
        || row.scientific_name
        || formatLabel(row.species)
        || row.record_id
        || "Unknown species"
      );
    }

    function treePopupNodeLabel(node) {
      if (!node) return "Selected tree node";
      const label = (
        node.nomenclature_group_label
        || node.common_name
        || node.species_display_label
        || node.preferred_public_label
        || node.broad_clade
        || node.clade
        || node.taxonomy_level
        || node.phylum
        || node.species
        || node.name
      );
      return label ? formatLabel(label) : "Selected tree node";
    }

    function treePopupIdentity(value) {
      const number = Number(value);
      return Number.isFinite(number) ? `${(number * 100).toFixed(2)}%` : "";
    }

    function treeNodeIdentityRows(node) {
      const scope = currentScope();
      const records = collectTreeTipRecords(node).map((recordId) => String(recordId || "")).filter(Boolean);
      const wantedRecords = new Set(records);
      const wantedProteins = new Set(records.map((recordId) => proteinIdFromRecordId(recordId)).filter(Boolean));
      const wantedSpecies = new Set((node && Array.isArray(node.descendant_species) ? node.descendant_species : []).map((species) => String(species || "")).filter(Boolean));
      const rows = [];
      const seen = new Set();
      for (const row of (scope.records || [])) {
        const recordId = String(row.record_id || "");
        const proteinId = treePopupProteinId(row);
        const species = String(row.species || "");
        const matchesNode = (
          (recordId && wantedRecords.has(recordId))
          || (proteinId && (wantedRecords.has(proteinId) || wantedProteins.has(proteinId)))
          || (species && wantedSpecies.has(species))
        );
        if (!matchesNode) continue;
        const key = recordId || `${species}:${proteinId}`;
        if (seen.has(key)) continue;
        seen.add(key);
        rows.push(row);
      }
      return rows.sort((a, b) => {
        const aIdentity = Number(a.identity_to_reference);
        const bIdentity = Number(b.identity_to_reference);
        const aHas = Number.isFinite(aIdentity);
        const bHas = Number.isFinite(bIdentity);
        if (aHas && bHas && bIdentity !== aIdentity) return bIdentity - aIdentity;
        if (aHas !== bHas) return aHas ? -1 : 1;
        return treePopupSpeciesLabel(a).localeCompare(treePopupSpeciesLabel(b));
      });
    }

    function closeTreeNodePopup() {
      const popup = document.getElementById("tree-node-popup");
      if (popup) popup.hidden = true;
    }

    function treeNodePopupTsv(rows) {
      const headers = ["rank", "species_label", "species", "identity_to_reference_percent", "protein_id", "symbol", "taxonomy_level", "clade", "record_id"];
      const lines = [headers.join("\\t")];
      rows.forEach((row, index) => {
        const values = [
          index + 1,
          treePopupSpeciesLabel(row),
          row.species || "",
          treePopupIdentity(row.identity_to_reference),
          treePopupProteinId(row),
          row.symbol || row.preferred_public_gene_label || "",
          row.taxonomy_level || "",
          row.clade || row.broad_clade || "",
          row.record_id || "",
        ];
        lines.push(values.map((value) => String(value == null ? "" : value).replaceAll("\\t", " ").replaceAll("\\n", " ")).join("\\t"));
      });
      return lines.join("\\n");
    }

    async function copyTreeNodePopupTsv() {
      const text = treeNodePopupTsv(LAST_TREE_NODE_POPUP_ROWS);
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(text);
          return;
        }
      } catch (error) {
        // Fall back to a temporary textarea below.
      }
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "readonly");
      textarea.style.position = "fixed";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }

    function showTreeNodeIdentityPopup(node) {
      const popup = document.getElementById("tree-node-popup");
      const title = document.getElementById("tree-node-popup-title");
      const summary = document.getElementById("tree-node-popup-summary");
      const body = document.getElementById("tree-node-popup-body");
      if (!popup || !title || !summary || !body || !node) return;
      const rows = treeNodeIdentityRows(node);
      LAST_TREE_NODE_POPUP_ROWS = rows;
      const scope = currentScope();
      const scopeLabel = scope.label || currentScopeKey();
      const nodeLabel = treePopupNodeLabel(node);
      title.textContent = nodeLabel;
      summary.textContent = `${rows.length} species/record${rows.length === 1 ? "" : "s"} in ${scopeLabel}, sorted highest to lowest by identity to the reference.`;
      if (!rows.length) {
        body.innerHTML = '<div class="empty">No records from this node are available in the current alignment scope.</div>';
      } else {
        const tableRows = rows.map((row, index) => {
          const proteinId = treePopupProteinId(row);
          const taxa = [row.taxonomy_level, row.clade || row.broad_clade].filter(Boolean).map(formatLabel).join(" | ");
          const speciesMeta = [row.scientific_name, row.species].filter(Boolean).filter((value, idx, arr) => arr.indexOf(value) === idx).join(" | ");
          return `
            <tr title="${escapeHtml(row.record_id || "")}">
              <td>${index + 1}</td>
              <td>
                <div class="species-name">${escapeHtml(treePopupSpeciesLabel(row))}</div>
                <div class="species-meta">${escapeHtml(speciesMeta)}</div>
              </td>
              <td class="identity">${escapeHtml(treePopupIdentity(row.identity_to_reference))}</td>
              <td>
                <div class="protein-id">${escapeHtml(proteinId || row.alphafold_entry_id || row.uniprot_accession || "")}</div>
                <div class="record-id">${escapeHtml(row.record_id || "")}</div>
              </td>
              <td>${escapeHtml(row.symbol || row.preferred_public_gene_label || "")}</td>
              <td>${escapeHtml(taxa)}</td>
            </tr>
          `;
        }).join("");
        body.innerHTML = `
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Species</th>
                <th>% identity</th>
                <th>Protein / record ID</th>
                <th>Gene</th>
                <th>Taxa</th>
              </tr>
            </thead>
            <tbody>${tableRows}</tbody>
          </table>
        `;
      }
      popup.hidden = false;
    }

    function niceTreeScale(maxDistance) {
      const rough = Number(maxDistance) / 5;
      if (!Number.isFinite(rough) || rough <= 0) return 0;
      const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
      const normalized = rough / magnitude;
      const step = normalized >= 5 ? 5 : (normalized >= 2 ? 2 : 1);
      return step * magnitude;
    }

    function buildTreeLayout(root, spacingMode) {
      const nodes = [];
      const links = [];
      let leafIndex = 0;
      let maxDepth = 0;
      let maxDistance = 0;
      let hasBranchLengths = false;

      function walk(node, depth, parent, distance) {
        const children = treeChildren(node);
        const branchLength = parent ? Number(node.branch_length || 0) : 0;
        const safeBranchLength = Number.isFinite(branchLength) && branchLength > 0 ? branchLength : 0;
        const currentDistance = parent ? distance + safeBranchLength : 0;
        if (safeBranchLength > 0) hasBranchLengths = true;
        const item = { node, depth, distance: currentDistance, x: 0, y: 0 };
        maxDepth = Math.max(maxDepth, depth);
        maxDistance = Math.max(maxDistance, currentDistance);
        nodes.push(item);
        if (parent) links.push({ parent, child: item });
        if (!children.length) {
          item.y = 30 + leafIndex * 24;
          leafIndex += 1;
        } else {
          const childItems = children.map((child) => walk(child, depth + 1, item, currentDistance));
          item.y = childItems.reduce((sum, child) => sum + child.y, 0) / childItems.length;
        }
        return item;
      }

      walk(root, 0, null, 0);
      const branchScaled = spacingMode === "branch_lengths" && hasBranchLengths && maxDistance > 0;
      const labelPad = 330;
      const treeWidth = branchScaled
        ? Math.max(540, Math.min(2200, Math.ceil(maxDistance * 900)))
        : Math.max(500, Math.max(maxDepth, 1) * 130);
      const width = Math.max(760, labelPad + treeWidth);
      const height = Math.max(140, 60 + Math.max(leafIndex, 1) * 24);
      const span = branchScaled ? maxDistance : Math.max(maxDepth, 1);
      nodes.forEach((item) => {
        const offset = branchScaled ? item.distance : item.depth;
        item.x = 24 + (offset / Math.max(span, 1e-9)) * treeWidth;
      });
      return {
        nodes,
        links,
        width,
        height,
        leafCount: leafIndex,
        tipCount: leafIndex,
        hasBranchLengths,
        usesBranchLengths: branchScaled,
        requestedSpacing: spacingMode,
        maxDistance,
        scaleSize: branchScaled ? niceTreeScale(maxDistance) : 0,
      };
    }

    function renderTreeLegend(tree) {
      const legend = document.getElementById("tree-legend");
      if (!legend) return;
      const items = Array.isArray(tree.mya_legend) ? tree.mya_legend.filter((item) => Number.isFinite(Number(item && item.mya))) : [];
      const eventTypes = Array.isArray(tree.event_types) ? tree.event_types.filter(Boolean) : [];
      if (!items.length) {
        legend.innerHTML = eventTypes.length
          ? `<div class="tree-legend-note">Mapped tree events: ${escapeHtml(eventTypes.join(", "))}</div>`
          : "";
        return;
      }
      legend.innerHTML = items.map((item) => {
        const color = treeColorForKey(item.clade);
        return `<span class="tree-legend-item"><span class="tree-legend-swatch" style="background:${escapeHtml(color || "#94a3b8")}"></span>${escapeHtml(formatLabel(item.clade))} ${escapeHtml(formatMa(item.mya))}</span>`;
      }).join("") + `<div class="tree-legend-note">Approximate divergence guide only: <code>~Ma</code> labels use broad-clade anchors, and missing coarse clades are inferred from the run taxonomy when possible.${eventTypes.length ? ` Mapped tree events: ${escapeHtml(eventTypes.join(", "))}.` : ""}</div>`;
    }

    function renderTreeScaleNote(layout) {
      const note = document.getElementById("tree-scale-note");
      if (!note) return;
      const zoomHint = "Use the scrollbars to pan; use the zoom buttons or Ctrl/Cmd + mouse wheel to zoom.";
      if (layout && layout.usesBranchLengths) {
        note.textContent = `Branch-length spacing is active for this tree. ${zoomHint}`;
        return;
      }
      if (layout && layout.requestedSpacing === "branch_lengths" && !layout.usesBranchLengths) {
        note.textContent = `This tree does not expose usable branch lengths, so the view fell back to the classic split-depth layout. ${zoomHint}`;
        return;
      }
      note.textContent = `Classic split-depth spacing is active for readability. Switch to Branch lengths when you want the inferred protein-distance geometry. ${zoomHint}`;
    }

    function clampTreeZoom(value) {
      return Math.max(TREE_MIN_ZOOM, Math.min(TREE_MAX_ZOOM, Number(value) || 1));
    }

    function updateTreeZoomValue() {
      const node = document.getElementById("tree-zoom-value");
      if (node) node.textContent = `${Math.round(TREE_ZOOM * 100)}%`;
    }

    function treeSvg() {
      return document.querySelector("#tree-svg-wrap svg");
    }

    function applyTreeZoom() {
      const svg = treeSvg();
      if (!svg) return;
      const baseWidth = Number(svg.dataset.baseWidth || 0);
      const baseHeight = Number(svg.dataset.baseHeight || 0);
      if (!baseWidth || !baseHeight) return;
      svg.style.width = `${baseWidth * TREE_ZOOM}px`;
      svg.style.height = `${baseHeight * TREE_ZOOM}px`;
      updateTreeZoomValue();
    }

    function fitTreeZoom() {
      const wrap = document.getElementById("tree-svg-wrap");
      const svg = treeSvg();
      if (!wrap || !svg) return;
      const baseWidth = Number(svg.dataset.baseWidth || 0);
      if (!baseWidth) return;
      TREE_ZOOM = clampTreeZoom(Math.min(1, Math.max(TREE_MIN_ZOOM, (wrap.clientWidth - 12) / baseWidth)));
      TREE_ZOOM_MODE = "fit";
      applyTreeZoom();
    }

    function setTreeZoom(nextZoom) {
      TREE_ZOOM = clampTreeZoom(nextZoom);
      TREE_ZOOM_MODE = "manual";
      applyTreeZoom();
    }

    function treeScaleSvg(layout) {
      if (!layout.usesBranchLengths || !(layout.scaleSize > 0) || !(layout.maxDistance > 0)) return "";
      const barWidth = Math.max(30, ((layout.scaleSize / layout.maxDistance) * Math.max(1, layout.width - 120)));
      const x2 = layout.width - 22;
      const x1 = Math.max(24, x2 - barWidth);
      const y = 22;
      return `<g class="tree-scale"><line x1="${x1.toFixed(1)}" y1="${y}" x2="${x2.toFixed(1)}" y2="${y}"></line><line x1="${x1.toFixed(1)}" y1="${(y - 4).toFixed(1)}" x2="${x1.toFixed(1)}" y2="${(y + 4).toFixed(1)}"></line><line x1="${x2.toFixed(1)}" y1="${(y - 4).toFixed(1)}" x2="${x2.toFixed(1)}" y2="${(y + 4).toFixed(1)}"></line><text x="${((x1 + x2) / 2).toFixed(1)}" y="${(y - 7).toFixed(1)}" text-anchor="middle">${escapeHtml(layout.scaleSize.toFixed(3))} branch length</text></g>`;
    }

    function renderTreePanel(forceFit = false) {
      const tree = treeData();
      const status = document.getElementById("tree-status");
      const wrap = document.getElementById("tree-svg-wrap");
      refreshTreeSearchMatches();
      if (!tree.available || !tree.root) {
        TREE_SELECTED_NODE_ID = "";
        TREE_SELECTED_RECORDS = new Set();
        status.textContent = tree.message || "No tree is available for this run.";
        const note = document.getElementById("tree-scale-note");
        if (note) note.textContent = "";
        const legend = document.getElementById("tree-legend");
        if (legend) legend.innerHTML = "";
        wrap.innerHTML = "";
        return;
      }

      const mode = treeViewMode();
      const spacingMode = treeLayoutMode();
      const displayRoot = buildDisplayTree(tree.root, mode);
      const layout = buildTreeLayout(displayRoot, spacingMode);
      const linkHtml = layout.links.map(({ parent, child }) => (
        `<path class="tree-link" d="M${parent.x.toFixed(1)} ${parent.y.toFixed(1)}H${child.x.toFixed(1)}V${child.y.toFixed(1)}"/>`
      )).join("");
      const scaleHtml = treeScaleSvg(layout);
      const nodeHtml = layout.nodes.map((item) => {
        const node = item.node;
        const tipRecords = collectTreeTipRecords(node);
        const selected = node.id === TREE_SELECTED_NODE_ID || tipRecords.some((recordId) => TREE_SELECTED_RECORDS.has(recordId));
        const searchMatch = tipRecords.some((recordId) => TREE_SEARCH_MATCH_RECORDS.has(recordId));
        const classes = ["tree-node"];
        if (selected) classes.push("selected");
        if (searchMatch) classes.push("search-match");
        const children = treeChildren(node);
        const radius = children.length ? 4.5 : (node.summary_tip_count > 1 ? 4.2 : 3.8);
        const label = (!children.length || node.display_label) ? treeNodeLabel(node, mode) : "";
        const accent = treeNodeAccent(node, mode);
        const styleBits = accent
          ? [
            `--tree-node-stroke:${accent}`,
            `--tree-node-selected-stroke:${accent}`,
            `--tree-node-search-stroke:${accent}`,
            `--tree-label-fill:${accent}`,
          ]
          : [];
        const styleAttr = styleBits.length ? ` style="${styleBits.join(";")}"` : "";
        const labelHtml = label
          ? `<text class="tree-label" x="${(item.x + 8).toFixed(1)}" y="${(item.y + 4).toFixed(1)}"${accent ? ` style="fill:${escapeHtml(accent)}"` : ""}>${escapeHtml(label)}</text>`
          : "";
        return `<g class="${classes.join(" ")}" data-tree-node-id="${escapeHtml(node.id)}" transform="translate(${item.x.toFixed(1)} ${item.y.toFixed(1)})"${styleAttr}><title>${escapeHtml(treeNodeTitle(node, mode))}</title><circle r="${radius}"></circle></g>${labelHtml}`;
      }).join("");

      wrap.innerHTML = `<svg class="tree-svg" width="${layout.width}" height="${layout.height}" viewBox="0 0 ${layout.width} ${layout.height}" role="img" data-base-width="${layout.width}" data-base-height="${layout.height}">${scaleHtml}${linkHtml}${nodeHtml}</svg>`;
      renderTreeLegend(tree);
      renderTreeScaleNote(layout);
      const layoutSignature = `${mode}|${spacingMode}|${layout.width}|${layout.height}|${layout.usesBranchLengths}|${layout.leafCount}`;
      if (forceFit || TREE_ZOOM_MODE === "fit" || layoutSignature !== TREE_LAYOUT_SIGNATURE) {
        fitTreeZoom();
      } else {
        applyTreeZoom();
      }
      TREE_LAYOUT_SIGNATURE = layoutSignature;
      wrap.querySelectorAll("[data-tree-node-id]").forEach((node) => {
        node.addEventListener("click", () => {
          const selected = findTreeNode(tree.root, node.dataset.treeNodeId);
          const selectedRecords = selected ? collectTreeTipRecords(selected) : [];
          const nextRecords = new Set(TREE_SELECTED_RECORDS);
          const allAlreadySelected = selectedRecords.length > 0 && selectedRecords.every((recordId) => nextRecords.has(recordId));
          selectedRecords.forEach((recordId) => {
            if (allAlreadySelected) {
              nextRecords.delete(recordId);
            } else {
              nextRecords.add(recordId);
            }
          });
          TREE_SELECTED_NODE_ID = nextRecords.size === selectedRecords.length && !allAlreadySelected ? (selected ? selected.id : "") : "";
          TREE_SELECTED_RECORDS = nextRecords;
          renderTreePanel();
          renderAlignment();
          showTreeNodeIdentityPopup(selected);
        });
      });
      wrap.onwheel = (event) => {
        if (!(event.ctrlKey || event.metaKey)) return;
        event.preventDefault();
        const factor = event.deltaY < 0 ? 1.15 : (1 / 1.15);
        setTreeZoom(TREE_ZOOM * factor);
      };
      const selectedText = TREE_SELECTED_RECORDS.size ? `${TREE_SELECTED_RECORDS.size} selected` : "No selection";
      const search = document.getElementById("tree-search").value.trim();
      const searchText = search ? `${TREE_SEARCH_MATCH_RECORDS.size} search match${TREE_SEARCH_MATCH_RECORDS.size === 1 ? "" : "es"}` : "";
      const structureText = layout.leafCount === tree.tip_count ? `${layout.leafCount} visible tips` : `${layout.leafCount} visible groups, ${tree.tip_count} original tips`;
      const spacingText = layout.usesBranchLengths ? "spacing: branch lengths" : "spacing: classic";
      const nomenclatureText = mode === "nomenclature" ? `nomenclature: ${tree.nomenclature_source || "local"}` : "";
      status.textContent = [structureText, `view: ${formatLabel(mode)}`, nomenclatureText, spacingText, selectedText, searchText, `source: ${tree.source_tree || "unknown"}`].filter(Boolean).join(" | ");
    }

    function numberOrNull(id) {
      const raw = document.getElementById(id).value.trim();
      if (!raw) return null;
      const value = Number(raw);
      return Number.isFinite(value) ? value : null;
    }

    function currentScopeKey() {
      return document.getElementById("scope-select").value || Object.keys(PAYLOAD.scopes)[0];
    }

    function currentScope() {
      return PAYLOAD.scopes[currentScopeKey()] || { records: [], reference_positions: [], reference_residues: [] };
    }

    function setSelectValue(id, value) {
      const node = document.getElementById(id);
      if (!node || value == null) return;
      const text = String(value);
      if ([...node.options].some((option) => option.value === text)) {
        node.value = text;
      }
    }

    function setInputValue(id, value) {
      const node = document.getElementById(id);
      if (!node) return;
      node.value = value == null ? "" : String(value);
    }

    function applyBrowserDefaultState() {
      setSelectValue("scope-select", DEFAULT_BROWSER_STATE.scope || "aligned_reference_projected");
      populateDynamicFilters();
      setInputValue("species-search", DEFAULT_BROWSER_STATE.species_search || "");
      setSelectValue("clade-filter", DEFAULT_BROWSER_STATE.clade || "all");
      setSelectValue("taxonomy-filter", DEFAULT_BROWSER_STATE.taxonomy || "all");
      setSelectValue("sort-select", DEFAULT_BROWSER_STATE.sort || "tree_order");
      setSelectValue("sort-direction", DEFAULT_BROWSER_STATE.direction || "asc");
      setSelectValue("group-select", DEFAULT_BROWSER_STATE.group || "taxonomy_level");
      setSelectValue("view-mode", DEFAULT_BROWSER_STATE.view || "compressed");
      setSelectValue("compare-mode", DEFAULT_BROWSER_STATE.compare || "exact");
      setInputValue("compare-min-run", DEFAULT_BROWSER_STATE.min_similar_run || 6);
      setSelectValue("offset-mode", DEFAULT_BROWSER_STATE.offset || "flag");
      setInputValue("residue-filter", DEFAULT_BROWSER_STATE.residue || "");
      setSelectValue("gap-filter", DEFAULT_BROWSER_STATE.gaps || "all");
      setSelectValue("match-filter", DEFAULT_BROWSER_STATE.reference_match || "all");
      setInputValue("ref-start", DEFAULT_BROWSER_STATE.reference_start || "");
      setInputValue("ref-end", DEFAULT_BROWSER_STATE.reference_end || "");
      setInputValue("align-start", DEFAULT_BROWSER_STATE.alignment_start || "");
      setInputValue("align-end", DEFAULT_BROWSER_STATE.alignment_end || "");
    }

    function formatPercent(value) {
      return value == null || Number.isNaN(Number(value)) ? "" : `${(Number(value) * 100).toFixed(1)}%`;
    }

    function formatLabel(value) {
      return String(value || "Unassigned").replaceAll("_", " ");
    }

    function formatSigned(value) {
      const number = Number(value || 0);
      return number > 0 ? `+${number}` : `${number}`;
    }

    function isGapResidue(aa) {
      return GAP_CHARS.has(String(aa || "").toUpperCase());
    }

    function isInformativeResidue(aa) {
      return aa && !NON_CONSENSUS_RESIDUES.has(String(aa).toUpperCase());
    }

    function residueAt(row, idx, shift = 0) {
      const seq = String(row.aligned_sequence || "").toUpperCase();
      const shifted = idx + Number(shift || 0);
      if (shifted < 0 || shifted >= seq.length) return "";
      return seq[shifted] || "";
    }

    function currentGroupField() {
      return document.getElementById("group-select").value;
    }

    function compressedActive() {
      return document.getElementById("view-mode").value === "compressed" && currentGroupField() !== "none";
    }

    function offsetVisualEnabled() {
      return document.getElementById("offset-mode").value === "visual";
    }

    function architecturePanelActive() {
      return compressedActive() && document.getElementById("compare-mode").value !== "off";
    }

    function consensusSourceRows(rows) {
      const withoutReference = rows.filter((row) => !row.is_reference);
      return withoutReference.length ? withoutReference : rows;
    }

    function residuePropertyKey(aa, scheme) {
      const residue = String(aa || "").toUpperCase();
      for (const [key, residues] of Object.entries(scheme)) {
        if (residues.has(residue)) return key;
      }
      return null;
    }

    function samePropertyAsReference(ref, aa) {
      if (!isInformativeResidue(ref) || !isInformativeResidue(aa)) return false;
      for (const scheme of Object.values(PROPERTY_GROUPS)) {
        const refKey = residuePropertyKey(ref, scheme);
        const aaKey = residuePropertyKey(aa, scheme);
        if (refKey && aaKey && refKey === aaKey) return true;
      }
      return false;
    }

    function similarToReference(aa, ref, mode) {
      if (!isInformativeResidue(aa) || !isInformativeResidue(ref)) return false;
      if (mode === "exact") return aa === ref;
      if (mode === "property") return aa === ref || samePropertyAsReference(ref, aa);
      return false;
    }

    function populateScopeSelect() {
      const select = document.getElementById("scope-select");
      select.innerHTML = Object.entries(PAYLOAD.scopes).map(([key, scope]) => (
        `<option value="${escapeHtml(key)}">${escapeHtml(scope.label)}</option>`
      )).join("");
    }

    function populateDynamicFilters() {
      const scope = currentScope();
      const clades = [...new Set(scope.records.map((row) => row.clade || "Unassigned"))]
        .sort((a, b) => (PAYLOAD.clade_order.indexOf(a) + 1 || 999) - (PAYLOAD.clade_order.indexOf(b) + 1 || 999) || String(a).localeCompare(String(b)));
      const taxa = [...new Set(scope.records.map((row) => row.taxonomy_level || "Unassigned"))]
        .sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: "base" }));
      const cladeSelect = document.getElementById("clade-filter");
      const taxSelect = document.getElementById("taxonomy-filter");
      const selectedClade = cladeSelect.value;
      const selectedTax = taxSelect.value;
      cladeSelect.innerHTML = '<option value="all">All</option>' + clades.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(formatLabel(value))}</option>`).join("");
      taxSelect.innerHTML = '<option value="all">All</option>' + taxa.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(formatLabel(value))}</option>`).join("");
      cladeSelect.value = [...cladeSelect.options].some((opt) => opt.value === selectedClade) ? selectedClade : "all";
      taxSelect.value = [...taxSelect.options].some((opt) => opt.value === selectedTax) ? selectedTax : "all";
    }

    function selectedColumns(scope) {
      const refStart = numberOrNull("ref-start");
      const refEnd = numberOrNull("ref-end");
      const alignStart = numberOrNull("align-start");
      const alignEnd = numberOrNull("align-end");
      const cols = [];
      for (let idx = 0; idx < scope.alignment_length; idx += 1) {
        const alignPos = idx + 1;
        const refPos = scope.reference_positions[idx];
        if (alignStart != null && alignPos < alignStart) continue;
        if (alignEnd != null && alignPos > alignEnd) continue;
        if (refStart != null || refEnd != null) {
          if (refPos == null) continue;
          if (refStart != null && refPos < refStart) continue;
          if (refEnd != null && refPos > refEnd) continue;
        }
        cols.push(idx);
      }
      return cols;
    }

    function snapshotResiduesPerLine() {
      const parsed = Number(SNAPSHOT_RESIDUES_PER_LINE);
      if (!Number.isFinite(parsed) || parsed < 1) return 100;
      return Math.max(1, Math.floor(parsed));
    }

    function snapshotRowMeta(row) {
      const parts = [];
      const length = snapshotUngappedLength(row);
      if (length != null) parts.push(`${length} aa`);
      const identity = formatPercent(row.identity_to_reference);
      if (identity) parts.push(identity);
      if (row.taxonomy_level) parts.push(String(row.taxonomy_level));
      if (row.clade) parts.push(String(row.clade));
      return parts.join(" | ");
    }

    function snapshotIdentityValue(row) {
      const value = Number(row.identity_to_reference);
      return Number.isFinite(value) ? value : null;
    }

    function snapshotUngappedLength(row) {
      const numeric = Number(row.ungapped_length);
      if (Number.isFinite(numeric) && numeric > 0) return Math.round(numeric);
      const sequence = String(row.aligned_sequence || "");
      if (!sequence) return null;
      return sequence.replaceAll("-", "").replaceAll(".", "").length;
    }

    function snapshotScopeRecordMap(scope) {
      const map = new Map();
      (scope.records || []).forEach((row) => {
          const recordId = String(row.record_id || "");
        if (recordId) map.set(recordId, row);
      });
      return map;
    }

    function snapshotReferenceRow(scope) {
      return (scope.records || []).find((row) => Boolean(row.is_reference)) || (scope.records || [])[0] || null;
    }

    function snapshotReferenceTargetLength(scope) {
      const referenceRow = snapshotReferenceRow(scope);
      const length = referenceRow ? snapshotUngappedLength(referenceRow) : null;
      return length != null ? length : 749;
    }

    function preserveSnapshotSelection(scope) {
      const recordMap = snapshotScopeRecordMap(scope);
      const seen = new Set();
      SNAPSHOT_SELECTED_RECORDS = SNAPSHOT_SELECTED_RECORDS
        .map((recordId) => String(recordId || ""))
        .filter((recordId) => {
          if (!recordId || !recordMap.has(recordId) || seen.has(recordId)) return false;
          seen.add(recordId);
          return true;
        });
      return recordMap;
    }

    function snapshotSelectedRows(scope, recordMap) {
      return SNAPSHOT_SELECTED_RECORDS.map((recordId) => recordMap.get(String(recordId || ""))).filter(Boolean);
    }

    function snapshotExportRows(scope, recordMap) {
      const rows = snapshotSelectedRows(scope, recordMap);
      const referenceRow = snapshotReferenceRow(scope);
      const referenceId = referenceRow ? String(referenceRow.record_id || "") : "";
      if (referenceRow && !rows.some((row) => String(row.record_id || "") === referenceId)) {
        rows.push(referenceRow);
      }
      return rows;
    }

    function snapshotCandidateRows(scope, recordMap) {
      const selected = new Set(SNAPSHOT_SELECTED_RECORDS.map((recordId) => String(recordId || "")));
      const referenceRow = snapshotReferenceRow(scope);
      const referenceId = referenceRow ? String(referenceRow.record_id || "") : "";
      const query = SNAPSHOT_SEARCH_QUERY.trim().toLowerCase();
      const targetLength = snapshotReferenceTargetLength(scope);
      const filteredRows = [...(scope.records || [])]
        .filter((row) => {
          const recordId = String(row.record_id || "");
          if (!recordId || recordId === referenceId || selected.has(recordId)) return false;
          if (!query) return true;
          const haystack = [
            row.species,
            row.symbol,
            row.record_id,
            row.taxonomy_level,
            row.clade,
          ].filter(Boolean).join(" ").toLowerCase();
          return haystack.includes(query);
        });
      const grouped = new Map();
      filteredRows.forEach((row) => {
        const speciesKey = String(row.species || row.record_id || "unknown");
        if (!grouped.has(speciesKey)) grouped.set(speciesKey, []);
        grouped.get(speciesKey).push(row);
      });
      const ranked = [];
      [...grouped.entries()]
        .sort((left, right) => String(left[0]).localeCompare(String(right[0]), undefined, { numeric: true, sensitivity: "base" }))
        .forEach(([, rows]) => {
          const ordered = rows.slice().sort((left, right) => {
            const leftLength = snapshotUngappedLength(left);
            const rightLength = snapshotUngappedLength(right);
            const leftDelta = leftLength == null ? Number.POSITIVE_INFINITY : Math.abs(leftLength - targetLength);
            const rightDelta = rightLength == null ? Number.POSITIVE_INFINITY : Math.abs(rightLength - targetLength);
            if (leftDelta !== rightDelta) return leftDelta - rightDelta;
            const leftIdentity = snapshotIdentityValue(left);
            const rightIdentity = snapshotIdentityValue(right);
            if (leftIdentity != null || rightIdentity != null) {
              const leftRank = leftIdentity == null ? -1 : leftIdentity;
              const rightRank = rightIdentity == null ? -1 : rightIdentity;
              if (leftRank !== rightRank) return rightRank - leftRank;
            }
            return (
              String(left.symbol || "").localeCompare(String(right.symbol || ""), undefined, { numeric: true, sensitivity: "base" }) ||
              String(left.record_id || "").localeCompare(String(right.record_id || ""), undefined, { numeric: true, sensitivity: "base" })
            );
          });
          ordered.forEach((row, index) => {
            const recordId = String(row.record_id || "");
            ranked.push({
              ...row,
              snapshot_variant_count: ordered.length,
              snapshot_variant_rank: index + 1,
              snapshot_recommended: index === 0,
              snapshot_length_delta: (() => {
                const length = snapshotUngappedLength(row);
                return length == null ? null : Math.abs(length - targetLength);
              })(),
              snapshot_reference_target_length: targetLength,
              snapshot_record_id: recordId,
            });
          });
        });
      return ranked;
    }

    function snapshotCandidateNameHtml(row) {
      const name = escapeHtml(row.species || row.record_id || "unknown");
      const badge = row.snapshot_variant_count > 1 && row.snapshot_recommended
        ? '<span class="snapshot-row-badge">Recommended</span>'
        : "";
      return `<div class="snapshot-row-head"><span class="snapshot-row-name">${name}</span>${badge}</div>`;
    }

    function snapshotCandidateDetail(row) {
      if (!(row.snapshot_variant_count > 1)) return "";
      const parts = [`variant ${row.snapshot_variant_rank}/${row.snapshot_variant_count}`];
      const delta = row.snapshot_length_delta;
      const target = row.snapshot_reference_target_length;
      if (delta != null && target != null) {
        parts.push(`delta ${delta} aa vs human ${target}`);
      }
      if (row.snapshot_record_id) parts.push(String(row.snapshot_record_id));
      return parts.join(" | ");
    }

    function snapshotExportBaseName() {
      const parts = ["alignment_browser", currentScopeKey(), "species_snapshot", `rpl${snapshotResiduesPerLine()}`];
      return parts.join("_").replace(/[^a-zA-Z0-9]+/g, "_").replace(/^_+|_+$/g, "").toLowerCase();
    }

    function buildEmptySnapshotSvg(title, message) {
      return [
        '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="140" viewBox="0 0 980 140" role="img">',
        '<style>.title{font:700 20px Segoe UI,Tahoma,sans-serif;fill:#18202a}.message{font:600 14px Segoe UI,Tahoma,sans-serif;fill:#617083}</style>',
        '<rect width="980" height="140" fill="#ffffff"/>',
        `<text class="title" x="24" y="34">${escapeXml(title)}</text>`,
        `<text class="message" x="24" y="72">${escapeXml(message)}</text>`,
        '</svg>',
      ].join("");
    }

    function snapshotRangeLabel(scope, chunk) {
      const values = chunk.map((idx) => {
        const refPos = scope.reference_positions[idx];
        return refPos != null ? Number(refPos) : idx + 1;
      });
      if (!values.length) return "0-0";
      return `${values[0]}-${values[values.length - 1]}`;
    }

    function snapshotSpeciesLabel(row, referenceSpecies) {
      const species = formatLabel(row.species || row.record_id || "unknown");
      if (!row.is_reference) return species;
      const refTag = String(referenceSpecies || "").toLowerCase() === "homo_sapiens"
        ? "human ref"
        : `${formatLabel(referenceSpecies || "reference")} ref`;
      return `${species} (${refTag})`;
    }

    function snapshotSecondaryRangesForRow(row, scope, lookup) {
      const maxReferencePosition = (scope.reference_positions || []).reduce((maxValue, value) => {
        const numeric = Number(value);
        return Number.isFinite(numeric) ? Math.max(maxValue, numeric) : maxValue;
      }, 0);
      const trackLength = Math.max(1, maxReferencePosition, Number(scope.alignment_length || 0), alphaFoldResidueCount());
      const entry = comparativeAlphaFoldEntryForRow(row, lookup);
      if (entry) return comparativeAlphaFoldEntryDisplayRanges(entry, trackLength);
      if (row && row.is_reference) return architectureSecondaryRanges(trackLength);
      return [];
    }

    function snapshotReferenceSpans(scope, chunk, start, end) {
      const left = Math.min(Number(start), Number(end));
      const right = Math.max(Number(start), Number(end));
      if (!Number.isFinite(left) || !Number.isFinite(right)) return [];
      const spans = [];
      let active = null;
      chunk.forEach((idx, ordinal) => {
        const refPos = Number((scope.reference_positions || [])[idx]);
        const inRange = Number.isFinite(refPos) && refPos >= left && refPos <= right;
        if (!inRange) {
          if (active) {
            spans.push(active);
            active = null;
          }
          return;
        }
        if (active && ordinal === active.endOrdinal + 1) {
          active.endOrdinal = ordinal;
        } else {
          if (active) spans.push(active);
          active = { startOrdinal: ordinal, endOrdinal: ordinal };
        }
      });
      if (active) spans.push(active);
      return spans;
    }

    function snapshotSecondaryTraceSvg(row, scope, chunk, gridX, rowTop, cellWidth, ranges) {
      if (!Array.isArray(ranges) || !ranges.length) return "";
      // V11: SS baseline scaled with the snapshot row (rowTop+8 originally,
      // +12 after the first up-scale, now +18) so the bigger cursive helix loops
      // sit cleanly above the residue cells without crashing into the row above.
      const y = rowTop + 18;
      const parts = [];
      ranges.forEach((range) => {
        const kind = comparativeRangeKind(range);
        const start = Math.floor(comparativeRangeStart(range));
        const end = Math.floor(comparativeRangeEnd(range));
        snapshotReferenceSpans(scope, chunk, start, end).forEach((span) => {
          const x1 = gridX + span.startOrdinal * cellWidth;
          const x2 = gridX + (span.endOrdinal + 1) * cellWidth;
          if (!(x2 > x1)) return;
          const label = snapshotSpeciesLabel(row, scope.reference_species || "reference");
          const title = range.title || `${label}; ${kind}; reference ${Math.min(start, end)}-${Math.max(start, end)}`;
          if (kind === "sheet") {
            const tip = Math.max(7, Math.min(17, (x2 - x1) * 0.45));
            parts.push(`<polygon class="snapshot-ss-sheet" points="${x1.toFixed(2)},${(y - 6).toFixed(2)} ${(x2 - tip).toFixed(2)},${(y - 6).toFixed(2)} ${(x2 - tip).toFixed(2)},${(y - 13).toFixed(2)} ${x2.toFixed(2)},${y.toFixed(2)} ${(x2 - tip).toFixed(2)},${(y + 13).toFixed(2)} ${(x2 - tip).toFixed(2)},${(y + 6).toFixed(2)} ${x1.toFixed(2)},${(y + 6).toFixed(2)}"><title>${escapeXml(title)}</title></polygon>`);
          } else if (kind === "helix") {
            parts.push(alphaFoldHelixElement(x1, x2, y, "", "snapshot-ss-helix", escapeXml(title), null));
          } else {
            parts.push(`<line class="snapshot-ss-loop" x1="${x1.toFixed(2)}" y1="${y.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y.toFixed(2)}"><title>${escapeXml(title)}</title></line>`);
          }
        });
      });
      return parts.join("");
    }

    const SNAPSHOT_STRONG_CONSERVATION_GROUPS = [
      "STA", "NEQK", "NHQK", "NDEQ", "QHRK", "MILV", "MILF", "HY", "FYW"
    ].map((group) => new Set(group.split("")));

    const SNAPSHOT_WEAK_CONSERVATION_GROUPS = [
      "CSA", "ATV", "SAG", "STNK", "STPA", "SGND", "SNDEQK", "NDEQHK", "NEQHRK", "FVLIM", "HFY"
    ].map((group) => new Set(group.split("")));

    function snapshotAllResiduesInAnyGroup(residues, groups) {
      return groups.some((group) => residues.every((aa) => group.has(aa)));
    }

    function snapshotConservationSymbol(rows, idx) {
      const residues = rows
        .map((row) => residueAt(row, idx) || "")
        .filter((aa) => isInformativeResidue(aa));
      if (residues.length < 2) return " ";
      const unique = new Set(residues);
      if (unique.size === 1) return "*";
      if (snapshotAllResiduesInAnyGroup(residues, SNAPSHOT_STRONG_CONSERVATION_GROUPS)) return ":";
      if (snapshotAllResiduesInAnyGroup(residues, SNAPSHOT_WEAK_CONSERVATION_GROUPS)) return ".";
      return " ";
    }

    function snapshotConservationRowSvg(rows, chunk, legendX, matrixX, gridX, y, cellWidth) {
      if (!Array.isArray(rows) || rows.length < 2) return "";
      const parts = [
        `<text class="snapshot-conservation-label" x="${legendX}" y="${y}">conservation</text>`,
        `<text class="row-number" x="${matrixX + 20}" y="${y}" text-anchor="end"></text>`,
      ];
      chunk.forEach((idx, ordinal) => {
        const symbol = snapshotConservationSymbol(rows, idx);
        if (!symbol.trim()) return;
        const x = gridX + ordinal * cellWidth + cellWidth / 2;
        const title = symbol === "*"
          ? "identical residue across snapshot rows"
          : symbol === ":"
            ? "strongly similar residue properties across snapshot rows"
            : "weakly similar residue properties across snapshot rows";
        parts.push(`<text class="snapshot-conservation-symbol" x="${x.toFixed(1)}" y="${y}" text-anchor="middle"><title>${escapeXml(title)}</title>${escapeXml(symbol)}</text>`);
      });
      return parts.join("");
    }

    const SNAPSHOT_RODENT_CLADE_RE = /rodent|glires|muroidea|myomorph|murin|hystricomorph|sciuromorph|castorimorph/i;
    const SNAPSHOT_RODENT_SPECIES_RE = /^(mus_|mus$|rattus|cricetulus|mesocricetus|peromyscus|microtus|cavia|ictidomys|urocitellus|marmota|castor|jaculus|nannospalax|fukomys|heterocephalus|chinchilla|octodon|dipodomys|perognathus|meriones|psammomys|spermophilus|myodes|ondatra|sigmodon|apodemus|acomys|grammomys|arvicanthis|mastomys)/i;

    function snapshotIsRodentRow(row) {
      if (!row) return false;
      const fields = [row.clade, row.broad_clade, row.taxonomy_level].map((v) => String(v || "")).join(" ");
      if (SNAPSHOT_RODENT_CLADE_RE.test(fields)) return true;
      return SNAPSHOT_RODENT_SPECIES_RE.test(String(row.species || "").toLowerCase());
    }

    // V12: columns where homo_sapiens and danio_rerio carry the IDENTICAL residue
    // but every selected rodent row differs from it -- the "shared in human+fish,
    // diverged in rodents" signature. Only active when human + zebrafish + at
    // least one rodent are among the snapshot rows.
    function snapshotHumanDanioSharedRodentDivergentColumns(rows, columns) {
      const result = { columns: new Set(), bovineColumns: new Set(), human: null, danio: null, bovine: null, rodents: [], active: false, reason: "" };
      const list = Array.isArray(rows) ? rows : [];
      result.human = list.find((r) => r && (r.is_reference || String(r.species || "").toLowerCase() === "homo_sapiens")) || null;
      result.danio = list.find((r) => r && String(r.species || "").toLowerCase() === "danio_rerio") || null;
      result.bovine = list.find((r) => r && String(r.species || "").toLowerCase() === "bos_taurus") || null;
      result.rodents = list.filter(snapshotIsRodentRow);
      if (!result.human || !result.danio || !result.rodents.length) {
        result.reason = "needs homo_sapiens + danio_rerio + ≥1 rodent selected";
        return result;
      }
      result.active = true;
      (columns || []).forEach((idx) => {
        const h = String(residueAt(result.human, idx) || "").toUpperCase();
        const d = String(residueAt(result.danio, idx) || "").toUpperCase();
        if (!isInformativeResidue(h) || !isInformativeResidue(d) || h !== d) return;
        for (const rod of result.rodents) {
          const r = String(residueAt(rod, idx) || "").toUpperCase();
          if (!isInformativeResidue(r) || r === h) return;
        }
        result.columns.add(idx);
        // V14: sub-flag columns where bos_taurus ALSO carries the shared residue --
        // identical across all three ACTIVE species (human + zebrafish + bovine)
        // while both inactive rodents differ. Candidate activity-linked residues.
        if (result.bovine) {
          const b = String(residueAt(result.bovine, idx) || "").toUpperCase();
          if (isInformativeResidue(b) && b === h) result.bovineColumns.add(idx);
        }
      });
      return result;
    }

    // Raw-alignment mode: render the snapshot from the full MUSCLE alignment
    // (aligned_full scope) subset to the selected species with all-gap columns
    // removed -- every residue is shown (like UGENE), with natural gaps rather
    // than the human-projection padding. Numbers still use human positions.
    function snapshotResolveScopeAndColumns() {
      const rawScope = PAYLOAD.scopes && (PAYLOAD.scopes["selected_raw"] || PAYLOAD.scopes["aligned_full"]);
      if (SNAPSHOT_RAW_ALIGNMENT && rawScope) {
        const full = rawScope;
        const recordMap = preserveSnapshotSelection(full);
        const rows = snapshotExportRows(full, recordMap);
        const total = (full.reference_positions || []).length
          || Number(full.alignment_length || 0)
          || (rows[0] ? String(rows[0].aligned_sequence || "").length : 0);
        const cols = [];
        for (let i = 0; i < total; i += 1) {
          let keep = false;
          for (let r = 0; r < rows.length; r += 1) {
            const aa = residueAt(rows[r], i);
            if (aa && aa !== "-" && aa !== ".") { keep = true; break; }
          }
          if (keep) cols.push(i);
        }
        if (cols.length) return { scope: full, columns: cols };
      }
      const scope = currentScope();
      return { scope: scope, columns: selectedColumns(scope) };
    }

    // Adobe Illustrator / Inkscape do not reliably apply a CSS <style> block or
    // the `font:` shorthand, which broke the exported snapshot outside browsers
    // (lost helix strokes, mis-sized/mis-anchored text). Map each snapshot class
    // to explicit SVG presentation attributes and inline them on export.
    const SNAPSHOT_SVG_STYLE_ATTRS = {
      "title": `font-family="Segoe UI, Tahoma, sans-serif" font-size="36" font-weight="700" fill="#18202a"`,
      "subtitle": `font-family="Segoe UI, Tahoma, sans-serif" font-size="21" font-weight="600" fill="#617083"`,
      "legend-range": `font-family="Segoe UI, Tahoma, sans-serif" font-size="28" font-weight="700" fill="#405774"`,
      "legend-name": `font-family="Segoe UI, Tahoma, sans-serif" font-size="28" font-weight="700" fill="#18202a"`,
      "legend-ref": `font-family="Segoe UI, Tahoma, sans-serif" font-size="28" font-weight="700" fill="#0f766e"`,
      "legend-ref-bg": `fill="#eefaf8"`,
      "axis": `stroke="#7b8188" stroke-width="3"`,
      "tick": `stroke="#7b8188" stroke-width="2.2"`,
      "tick-label": `font-family="Segoe UI, Tahoma, sans-serif" font-size="22" font-weight="500" fill="#6b7280"`,
      "row-number": `font-family="Segoe UI, Tahoma, sans-serif" font-size="22" font-weight="500" fill="#18202a"`,
      "aa": `font-family="Consolas, 'Courier New', monospace" font-size="28" font-weight="700" fill="#18202a"`,
      "snapshot-ss-loop": `fill="none" stroke="#6b7280" stroke-width="2.2" stroke-linecap="round" opacity="0.5"`,
      "snapshot-ss-helix": `stroke="#6b7280" stroke-width="2.8"`,
      "snapshot-ss-sheet": `fill="#6b7280" opacity="0.86"`,
      "snapshot-conservation-label": `font-family="Segoe UI, Tahoma, sans-serif" font-size="24" font-weight="700" fill="#405774"`,
      "snapshot-conservation-symbol": `font-family="Consolas, 'Courier New', monospace" font-size="30" font-weight="800" fill="#111827"`,
      "snapshot-conservation-key": `font-family="Segoe UI, Tahoma, sans-serif" font-size="22" font-weight="600" fill="#617083"`,
      "snapshot-hd-frame": `fill="none" stroke="#000000" stroke-width="3.6"`,
      "snapshot-hd-tab": `fill="#000000"`,
      "snapshot-hd-trio-star": `fill="none" stroke="#dc2626" stroke-width="2.8" stroke-linecap="round"`,
    };

    function inlineSnapshotSvgStyles(svg) {
      let out = String(svg).replace(/<style>[\s\S]*?<\/style>/, "");
      out = out.replace(/ class="([^"]+)"/g, function (m, cls) {
        const attrs = SNAPSHOT_SVG_STYLE_ATTRS[cls];
        return attrs ? " " + attrs : "";
      });
      return out;
    }

    function buildSpeciesSnapshotSvg(scope, columns) {
      const recordMap = preserveSnapshotSelection(scope);
      const rows = snapshotExportRows(scope, recordMap);
      const titleGene = PAYLOAD.meta && PAYLOAD.meta.gene_symbol ? String(PAYLOAD.meta.gene_symbol) : "Alignment";
      const snapshotTitle = `${titleGene} species snapshot`;
      if (!(scope.records || []).length) {
        return buildEmptySnapshotSvg(snapshotTitle, "No alignment records are available for this scope.");
      }
      if (!columns.length) {
        return buildEmptySnapshotSvg(snapshotTitle, "No alignment columns match the current filters.");
      }
      if (!rows.length) {
        return buildEmptySnapshotSvg(snapshotTitle, "No species are available for snapshot export.");
      }

      const residuesPerLine = snapshotResiduesPerLine();
      const chunks = [];
      for (let start = 0; start < columns.length; start += residuesPerLine) {
        chunks.push(columns.slice(start, start + residuesPerLine));
      }
      // V11 snapshot panel sized up again ~45% (was 22/32/56 grid + 14px AA font).
      const margin = 36;
      const legendWidth = 480;
      const legendGap = 44;
      const rowNumberWidth = 44;
      const cellWidth = 32;
      const cellHeight = 46;
      const cellTopOffset = 32;
      const rowHeight = 80;
      const conservationRowHeight = rows.length > 1 ? 52 : 0;
      const axisHeight = 70;
      const blockGap = 16;
      const titleBlockHeight = 100;
      const footerKeyHeight = rows.length > 1 ? 30 : 0;
      const maxColumns = Math.max(...chunks.map((chunk) => chunk.length), 1);
      const gridWidth = maxColumns * cellWidth;
      const blockHeight = axisHeight + rows.length * rowHeight + conservationRowHeight;
      const width = margin * 2 + legendWidth + legendGap + rowNumberWidth + gridWidth + 8;
      const height = margin + titleBlockHeight + chunks.length * blockHeight + Math.max(0, chunks.length - 1) * blockGap + footerKeyHeight + margin;
      const selectedCount = Math.max(0, rows.length - (snapshotReferenceRow(scope) ? 1 : 0));
      const referenceSpecies = scope.reference_species || (snapshotReferenceRow(scope) && snapshotReferenceRow(scope).species) || "reference";
      const compareMode = document.getElementById("compare-mode") ? document.getElementById("compare-mode").value : "exact";
      const minRun = Math.max(1, numberOrNull("compare-min-run") || 6);
      const highlightOn = SNAPSHOT_HIGHLIGHT_HD_RODENT;
      const highlight = highlightOn
        ? snapshotHumanDanioSharedRodentDivergentColumns(rows, columns)
        : { columns: new Set(), active: false, reason: "" };
      const highlightColumns = highlight.columns;
      let subtitle = `${selectedCount} requested species + ${formatLabel(referenceSpecies)} | ${compareMode} | min run ${minRun} | ${columns.length} columns${SNAPSHOT_RAW_ALIGNMENT ? " | raw MUSCLE alignment (all residues)" : ""}`;
      if (highlightOn) {
        if (highlight.active) {
          subtitle += ` | ${highlightColumns.size} human+zebrafish-shared rodent-divergent site${highlightColumns.size === 1 ? "" : "s"}`;
          const trioN = highlight.bovineColumns ? highlight.bovineColumns.size : 0;
          if (trioN) subtitle += ` (${trioN} also in bos_taurus, red asterisk)`;
        } else {
          subtitle += ` | highlight off (${highlight.reason})`;
        }
      }
      const secondaryLookup = comparativeAlphaFoldRecordLookup();
      const parts = [
        `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img">`,
        '<style>.title{font:700 36px Segoe UI,Tahoma,sans-serif;fill:#18202a}.subtitle{font:600 21px Segoe UI,Tahoma,sans-serif;fill:#617083}.legend-range{font:700 28px Segoe UI,Tahoma,sans-serif;fill:#405774}.legend-name{font:700 28px Segoe UI,Tahoma,sans-serif;fill:#18202a}.legend-ref{font:700 28px Segoe UI,Tahoma,sans-serif;fill:#0f766e}.legend-ref-bg{fill:#eefaf8}.axis{stroke:#7b8188;stroke-width:3}.tick{stroke:#7b8188;stroke-width:2.2}.tick-label{font:500 20px Segoe UI,Tahoma,sans-serif;fill:#6b7280}.row-number{font:500 22px Segoe UI,Tahoma,sans-serif;fill:#18202a}.aa{font:700 28px Consolas,Courier New,monospace;fill:#18202a}.snapshot-ss-loop{stroke:#6b7280;stroke-width:2.2;stroke-linecap:round;opacity:.5}.snapshot-ss-helix{fill:none;stroke:#6b7280;stroke-width:2.8;stroke-linecap:round;stroke-linejoin:round}.snapshot-ss-sheet{fill:#6b7280;opacity:.86}.snapshot-conservation-label{font:700 24px Segoe UI,Tahoma,sans-serif;fill:#405774}.snapshot-conservation-symbol{font:800 30px Consolas,Courier New,monospace;fill:#111827}.snapshot-conservation-key{font:600 22px Segoe UI,Tahoma,sans-serif;fill:#617083}.snapshot-hd-frame{fill:none;stroke:#000000;stroke-width:3.6}.snapshot-hd-tab{fill:#000000}.snapshot-hd-trio-star{fill:none;stroke:#dc2626;stroke-width:2.8;stroke-linecap:round}</style>',
        `<rect width="${width}" height="${height}" fill="#ffffff"/>`,
        `<text class="title" x="${margin}" y="${margin + 20}">${escapeXml(snapshotTitle)}</text>`,
        `<text class="subtitle" x="${margin}" y="${margin + 40}">${escapeXml(subtitle)}</text>`,
      ];

      let y = margin + titleBlockHeight;
      const legendX = margin;
      const matrixX = margin + legendWidth + legendGap;
      const gridX = matrixX + rowNumberWidth;
      chunks.forEach((chunk, blockIndex) => {
        const rangeLabel = snapshotRangeLabel(scope, chunk);
        const axisY = y + 18;
        const tickLabelY = y + 54;
        const gridTop = y + axisHeight;
        parts.push(`<text class="legend-range" x="${legendX}" y="${y + 34}">${escapeXml(rangeLabel)}</text>`);
        parts.push(`<line class="axis" x1="${gridX - 2}" y1="${axisY}" x2="${gridX + chunk.length * cellWidth + 2}" y2="${axisY}"/>`);
        // Candidate labelled ticks: the block's first & last column (boundary
        // anchors) plus every reference position divisible by 3.
        const labeledTicks = [];
        chunk.forEach((idx, ordinal) => {
          const refPos = scope.reference_positions[idx];
          if (refPos == null) return;
          const isEnd = ordinal === 0 || ordinal === chunk.length - 1;
          if (!(isEnd || Number(refPos) % 3 === 0)) return;
          labeledTicks.push({ refPos: refPos, x: gridX + ordinal * cellWidth + cellWidth / 2, isEnd: isEnd });
        });
        // Tick marks: one per candidate (thin lines never visually collide).
        labeledTicks.forEach((t) => {
          parts.push(`<line class="tick" x1="${t.x.toFixed(1)}" y1="${axisY}" x2="${t.x.toFixed(1)}" y2="${(axisY + 13).toFixed(1)}"/>`);
        });
        // Tick LABELS: 3-digit numbers on adjacent cells collide at block edges
        // (e.g. 279|280, 281|282). Keep the boundary label (first/last column);
        // drop an interior multiple-of-3 label only when it would overlap a kept
        // one. Distance test uses an estimated label width, so it self-adjusts
        // to 1/2/3-digit numbers and any residues-per-line / cell width.
        const TICK_LABEL_CHAR_W = 12.5;   // ~digit advance at the tick-label size
        const TICK_LABEL_GAP = 4;         // extra breathing room between labels
        const tickHalf = (t) => (String(t.refPos).length * TICK_LABEL_CHAR_W) / 2;
        const keptTickLabels = [];
        const tryKeepTickLabel = (t) => {
          const clash = keptTickLabels.some((k) => Math.abs(k.x - t.x) < tickHalf(k) + tickHalf(t) + TICK_LABEL_GAP);
          if (!clash) keptTickLabels.push(t);
        };
        labeledTicks.filter((t) => t.isEnd).forEach(tryKeepTickLabel);   // boundaries win
        labeledTicks.filter((t) => !t.isEnd).forEach(tryKeepTickLabel);  // then fill interior
        keptTickLabels.forEach((t) => {
          parts.push(`<text class="tick-label" x="${t.x.toFixed(1)}" y="${tickLabelY}" text-anchor="middle">${escapeXml(t.refPos)}</text>`);
        });
        rows.forEach((row, rowIndex) => {
          const rowTop = gridTop + rowIndex * rowHeight;
          const rowMid = rowTop + cellTopOffset + 29;
          const speciesLabel = snapshotSpeciesLabel(row, referenceSpecies);
          if (row.is_reference) {
            parts.push(`<rect class="legend-ref-bg" x="${legendX}" y="${rowTop + 1}" width="${legendWidth - 12}" height="${rowHeight - 3}" rx="0"/>`);
          }
          parts.push(`<text class="${row.is_reference ? "legend-ref" : "legend-name"}" x="${legendX}" y="${rowMid}">${escapeXml(speciesLabel)}</text>`);
          parts.push(`<text class="row-number" x="${matrixX + rowNumberWidth - 8}" y="${rowMid}" text-anchor="end">${rowIndex + 1}</text>`);
          const secondaryRanges = snapshotSecondaryRangesForRow(row, scope, secondaryLookup);
          parts.push(snapshotSecondaryTraceSvg(row, scope, chunk, gridX, rowTop, cellWidth, secondaryRanges));
          chunk.forEach((idx, ordinal) => {
            const aa = residueAt(row, idx) || "-";
            const ref = scope.reference_residues[idx] || "";
            const refPos = scope.reference_positions[idx];
            const color = PAYLOAD.aa_colors[aa] || PAYLOAD.aa_colors.X || "#f2f2f2";
            const x = gridX + ordinal * cellWidth;
            const title = `${speciesLabel} | alignment ${idx + 1}${refPos == null ? "" : ` | reference ${refPos}`} | ${aa || " "} vs ${ref || " "}`;
            parts.push(`<rect x="${x}" y="${rowTop + cellTopOffset}" width="${cellWidth}" height="${cellHeight}" fill="${escapeXml(color)}"><title>${escapeXml(title)}</title></rect>`);
            parts.push(`<text class="aa" x="${(x + cellWidth / 2).toFixed(1)}" y="${rowTop + cellTopOffset + 34}" text-anchor="middle">${escapeXml(aa)}</text>`);
          });
        });
        if (conservationRowHeight) {
          const conservationY = gridTop + rows.length * rowHeight + 34;
          parts.push(snapshotConservationRowSvg(rows, chunk, legendX, matrixX, gridX, conservationY, cellWidth));
        }
        if (highlightOn && highlight.active && highlightColumns.size) {
          const frameTop = gridTop + cellTopOffset;
          const frameHeight = (rows.length - 1) * rowHeight + cellHeight;
          const trioColumns = highlight.bovineColumns || new Set();
          chunk.forEach((idx, ordinal) => {
            if (!highlightColumns.has(idx)) return;
            const x = gridX + ordinal * cellWidth;
            const refPos = scope.reference_positions[idx];
            const trio = trioColumns.has(idx);
            const tip = trio
              ? `Shared in homo_sapiens + danio_rerio + bos_taurus (all active species), divergent in every selected rodent | alignment ${idx + 1}${refPos == null ? "" : ` | reference ${refPos}`}`
              : `Shared in homo_sapiens + danio_rerio, divergent in every selected rodent | alignment ${idx + 1}${refPos == null ? "" : ` | reference ${refPos}`}`;
            parts.push(`<rect class="snapshot-hd-frame" x="${x}" y="${frameTop}" width="${cellWidth}" height="${frameHeight}"><title>${escapeXml(tip)}</title></rect>`);
            parts.push(`<rect class="snapshot-hd-tab" x="${x}" y="${(frameTop + frameHeight + 5).toFixed(1)}" width="${cellWidth}" height="9"><title>${escapeXml(tip)}</title></rect>`);
            if (trio) {
              // V14: red asterisk sitting on the black tab but lifted up a little
              // into the bottom of the sequence row (offset above the bar centre,
              // not centred). These columns are identical across all three ACTIVE
              // species (human + zebrafish + bovine) while both inactive rodents
              // differ -- kept additive to the black frame. Drawn as solid-red
              // strokes (no halo) so it survives SVG export in Illustrator /
              // Inkscape where glyph fonts may be missing.
              const cx = x + cellWidth / 2;
              const cy = frameTop + frameHeight + 2;
              const r = 10;
              const arm = 0.866 * r;
              const segs = [[0, -r, 0, r], [-arm, -r / 2, arm, r / 2], [-arm, r / 2, arm, -r / 2]];
              segs.forEach((s) => parts.push(`<line class="snapshot-hd-trio-star" x1="${(cx + s[0]).toFixed(1)}" y1="${(cy + s[1]).toFixed(1)}" x2="${(cx + s[2]).toFixed(1)}" y2="${(cy + s[3]).toFixed(1)}"><title>${escapeXml(tip)}</title></line>`));
            }
          });
        }
        y += blockHeight + blockGap;
      });
      if (footerKeyHeight) {
        const trioLegendN = (highlightOn && highlight.active && highlight.bovineColumns) ? highlight.bovineColumns.size : 0;
        const highlightLegend = (highlightOn && highlight.active && highlightColumns.size)
          ? "  |  Black frames: identical in homo_sapiens + danio_rerio, different in every selected rodent."
            + (trioLegendN ? "  Red asterisk: also identical in bos_taurus (shared by all active species: human + zebrafish + bovine)." : "")
          : "";
        parts.push(`<text class="snapshot-conservation-key" x="${legendX}" y="${height - margin - 8}">Conservation key: * identical across snapshot rows; : strongly similar amino-acid properties; . weakly similar amino-acid properties.${escapeXml(highlightLegend)}</text>`);
      }
      parts.push("</svg>");
      return inlineSnapshotSvgStyles(parts.join(""));
    }

    function downloadBlob(filename, blob) {
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 250);
    }

    async function downloadSvgAsPng(filename, svgText) {
      const match = svgText.match(/<svg[^>]*width="([0-9.]+)"[^>]*height="([0-9.]+)"/i);
      const width = match ? Math.max(1, Math.ceil(Number(match[1]))) : 1200;
      const height = match ? Math.max(1, Math.ceil(Number(match[2]))) : 800;
      const svgBlob = new Blob([svgText], { type: "image/svg+xml;charset=utf-8" });
      const svgUrl = URL.createObjectURL(svgBlob);
      try {
        const image = await new Promise((resolve, reject) => {
          const img = new Image();
          img.onload = () => resolve(img);
          img.onerror = () => reject(new Error("Could not rasterize SVG."));
          img.src = svgUrl;
        });
        const scale = Math.max(1, Math.min(3, Number(window.devicePixelRatio) || 1));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.ceil(width * scale));
        canvas.height = Math.max(1, Math.ceil(height * scale));
        const context = canvas.getContext("2d");
        if (!context) throw new Error("Could not create canvas context.");
        context.scale(scale, scale);
        context.fillStyle = "#ffffff";
        context.fillRect(0, 0, width, height);
        context.drawImage(image, 0, 0, width, height);
        const blob = await new Promise((resolve, reject) => {
          canvas.toBlob((value) => {
            if (value) resolve(value);
            else reject(new Error("Could not create PNG."));
          }, "image/png");
        });
        downloadBlob(filename, blob);
      } finally {
        URL.revokeObjectURL(svgUrl);
      }
    }

    function renderSpeciesSnapshotPanel(scope, columns) {
      const panel = document.getElementById("species-snapshot-panel");
      if (!panel) return;
      const recordMap = preserveSnapshotSelection(scope);
      const referenceRow = snapshotReferenceRow(scope);
      const referenceSpecies = scope.reference_species || (referenceRow && referenceRow.species) || "reference";
      const selectedRows = snapshotSelectedRows(scope, recordMap);
      const candidateRows = snapshotCandidateRows(scope, recordMap);
      const exportDisabled = !(scope.records || []).length || !columns.length;
      const highlightInfo = SNAPSHOT_HIGHLIGHT_HD_RODENT
        ? snapshotHumanDanioSharedRodentDivergentColumns(snapshotExportRows(scope, recordMap), columns)
        : null;
      const highlightStatusText = !highlightInfo
        ? ""
        : (highlightInfo.active
            ? `${highlightInfo.columns.size} site${highlightInfo.columns.size === 1 ? "" : "s"} will be framed`
              + (highlightInfo.bovineColumns && highlightInfo.bovineColumns.size
                  ? ` (${highlightInfo.bovineColumns.size} also in bos_taurus → red ✱)`
                  : "")
            : `inactive — ${highlightInfo.reason}`);
      const selectedHtml = selectedRows.length
        ? selectedRows.map((row) => {
          const recordId = String(row.record_id || "");
          return `<div class="snapshot-selected-row"><div class="snapshot-row-main"><div class="snapshot-row-head"><div class="snapshot-row-name">${escapeHtml(row.species || row.record_id || "unknown")}</div></div><div class="snapshot-row-meta">${escapeHtml(snapshotRowMeta(row))}</div></div><button type="button" class="snapshot-remove" data-snapshot-remove="${escapeHtml(recordId)}">Remove</button></div>`;
        }).join("")
        : '<div class="snapshot-empty-state">NO SPECIES ADDED YET.</div>';
      const candidateHtml = candidateRows.length
        ? candidateRows.map((row) => {
          const recordId = String(row.record_id || "");
          const detail = snapshotCandidateDetail(row);
          return `<button type="button" class="snapshot-candidate-row" data-snapshot-add="${escapeHtml(recordId)}"><div class="snapshot-row-main">${snapshotCandidateNameHtml(row)}<div class="snapshot-row-meta">${escapeHtml(snapshotRowMeta(row))}</div>${detail ? `<div class="snapshot-row-detail">${escapeHtml(detail)}</div>` : ""}</div></button>`;
        }).join("")
        : '<div class="snapshot-empty-state">NO SPECIES MATCH SEARCH.</div>';
      panel.innerHTML = `
        <div class="snapshot-topbar">
          <div class="snapshot-title">
            <h2>Species snapshot export</h2>
            <p>Search and add species; ${escapeHtml(referenceSpecies)} is added at the bottom.</p>
          </div>
          <div class="snapshot-actions">
            <button type="button" class="snapshot-export" data-snapshot-export="svg"${exportDisabled ? " disabled" : ""}>Export SVG</button>
            <button type="button" class="snapshot-export" data-snapshot-export="png"${exportDisabled ? " disabled" : ""}>Export PNG</button>
          </div>
        </div>
        <label class="snapshot-search-label">Species list
          <input id="snapshot-species-search" type="search" placeholder="Search species to add" value="${escapeHtml(SNAPSHOT_SEARCH_QUERY)}">
        </label>
        <div class="snapshot-selected-list">${selectedHtml}</div>
        <div class="snapshot-candidate-list">${candidateHtml}</div>
        <div class="snapshot-footer">
          <div class="snapshot-footer-note">${escapeHtml(`${selectedRows.length} selected species${referenceRow ? `; ${referenceSpecies} will be appended automatically.` : "."}`)}</div>
          <label class="snapshot-highlight-label" title="When ON, the exported SVG/PNG frames columns where homo_sapiens and danio_rerio carry the identical amino acid but every selected rodent differs. Columns where bos_taurus also shares that residue (identical across all active species) additionally get a red asterisk. Needs human + zebrafish + at least one rodent selected.">
            <input type="checkbox" id="snapshot-highlight-hd-rodent"${SNAPSHOT_HIGHLIGHT_HD_RODENT ? " checked" : ""}>
            Highlight human+zebrafish shared, rodent-different sites (red ✱ = also in bovine)
          </label>
          ${SNAPSHOT_HIGHLIGHT_HD_RODENT ? `<div class="snapshot-highlight-status">${escapeHtml(highlightStatusText)}</div>` : ""}
          <label class="snapshot-highlight-label" title="Export the raw MUSCLE alignment of the selected species (all residues, natural gaps like a standard aligner) instead of the human reference-projected view. Numbers still use human positions; insertion columns are unnumbered.">
            <input type="checkbox" id="snapshot-raw-alignment"${SNAPSHOT_RAW_ALIGNMENT ? " checked" : ""}>
            Raw alignment (all residues)
          </label>
          <label class="snapshot-residues-label">Residues per line
            <input id="snapshot-residues-per-line" type="number" min="1" step="1" value="${snapshotResiduesPerLine()}">
          </label>
        </div>
      `;

      const searchInput = document.getElementById("snapshot-species-search");
      searchInput.addEventListener("input", () => {
        const caret = searchInput.selectionStart == null ? searchInput.value.length : searchInput.selectionStart;
        SNAPSHOT_SEARCH_QUERY = searchInput.value;
        renderSpeciesSnapshotPanel(currentScope(), selectedColumns(currentScope()));
        const nextSearch = document.getElementById("snapshot-species-search");
        if (nextSearch) {
          nextSearch.focus();
          nextSearch.setSelectionRange(caret, caret);
        }
      });

      const residuesInput = document.getElementById("snapshot-residues-per-line");
      const updateResidues = () => {
        const parsed = Number(residuesInput.value);
        if (Number.isFinite(parsed) && parsed >= 1) {
          SNAPSHOT_RESIDUES_PER_LINE = Math.max(1, Math.floor(parsed));
        }
      };
      residuesInput.addEventListener("input", updateResidues);
      residuesInput.addEventListener("change", () => {
        updateResidues();
        renderSpeciesSnapshotPanel(currentScope(), selectedColumns(currentScope()));
      });

      const highlightToggleInput = document.getElementById("snapshot-highlight-hd-rodent");
      if (highlightToggleInput) {
        highlightToggleInput.addEventListener("change", () => {
          SNAPSHOT_HIGHLIGHT_HD_RODENT = !!highlightToggleInput.checked;
          renderSpeciesSnapshotPanel(currentScope(), selectedColumns(currentScope()));
        });
      }

      const rawAlignmentInput = document.getElementById("snapshot-raw-alignment");
      if (rawAlignmentInput) {
        rawAlignmentInput.addEventListener("change", () => {
          SNAPSHOT_RAW_ALIGNMENT = !!rawAlignmentInput.checked;
          renderSpeciesSnapshotPanel(currentScope(), selectedColumns(currentScope()));
        });
      }

      panel.querySelectorAll("[data-snapshot-add]").forEach((button) => {
        button.addEventListener("click", () => {
          const recordId = String(button.dataset.snapshotAdd || "");
          if (!recordId || SNAPSHOT_SELECTED_RECORDS.includes(recordId)) return;
          SNAPSHOT_SELECTED_RECORDS.push(recordId);
          renderAlignment();
        });
      });

      panel.querySelectorAll("[data-snapshot-remove]").forEach((button) => {
        button.addEventListener("click", () => {
          const recordId = String(button.dataset.snapshotRemove || "");
          SNAPSHOT_SELECTED_RECORDS = SNAPSHOT_SELECTED_RECORDS.filter((value) => String(value || "") !== recordId);
          renderAlignment();
        });
      });

      panel.querySelectorAll("[data-snapshot-export]").forEach((button) => {
        button.addEventListener("click", async () => {
          const exportType = button.dataset.snapshotExport;
          const resolved = snapshotResolveScopeAndColumns();
          const liveScope = resolved.scope;
          const liveColumns = resolved.columns;
          const svgText = buildSpeciesSnapshotSvg(liveScope, liveColumns);
          try {
            if (exportType === "png") {
              await downloadSvgAsPng(`${snapshotExportBaseName()}.png`, svgText);
            } else {
              downloadText(`${snapshotExportBaseName()}.svg`, "image/svg+xml;charset=utf-8", svgText);
            }
          } catch (error) {
            console.error(error);
            window.alert("Could not export the species snapshot.");
          }
        });
      });
    }

    function rowMatches(row, scope, columns) {
      const query = document.getElementById("species-search").value.trim().toLowerCase();
      if (query) {
        const haystack = [row.species, row.symbol, row.record_id, row.taxonomy_level, row.clade].join(" ").toLowerCase();
        if (!haystack.includes(query)) return false;
      }
      const clade = document.getElementById("clade-filter").value;
      if (clade !== "all" && (row.clade || "Unassigned") !== clade) return false;
      const tax = document.getElementById("taxonomy-filter").value;
      if (tax !== "all" && (row.taxonomy_level || "Unassigned") !== tax) return false;
      const treeFilter = document.getElementById("tree-filter-rows");
      if (treeFilter && treeFilter.checked && TREE_SELECTED_RECORDS.size) {
        if (!TREE_SELECTED_RECORDS.has(String(row.record_id || ""))) return false;
      }

      const residueRaw = document.getElementById("residue-filter").value.trim().toUpperCase().replaceAll(",", "");
      const gapMode = document.getElementById("gap-filter").value;
      const matchMode = document.getElementById("match-filter").value;
      const seq = String(row.aligned_sequence || "").toUpperCase();
      const residueSet = new Set(residueRaw === "GAP" ? ["-", "."] : residueRaw.split(""));
      let hasResidue = residueSet.size === 0 || !residueRaw;
      let hasGap = false;
      let hasMatch = false;
      let hasMismatch = false;
      let comparable = 0;
      let visibleMatches = 0;

      for (const idx of columns) {
        const aa = seq[idx] || "";
        const ref = scope.reference_residues[idx];
        const isGap = aa === "-" || aa === ".";
        if (isGap) hasGap = true;
        if (!hasResidue && residueSet.has(aa)) hasResidue = true;
        if (!isGap && ref && ref !== "-" && ref !== ".") {
          comparable += 1;
          if (aa === ref) {
            hasMatch = true;
            visibleMatches += 1;
          } else {
            hasMismatch = true;
          }
        }
      }

      if (!hasResidue) return false;
      if (gapMode === "has_gap" && !hasGap) return false;
      if (gapMode === "no_gap" && hasGap) return false;
      if (matchMode === "has_match" && !hasMatch) return false;
      if (matchMode === "has_mismatch" && !hasMismatch) return false;
      if (matchMode === "all_visible_match" && (!comparable || visibleMatches !== comparable)) return false;
      return true;
    }

    function compareRows(left, right) {
      const field = document.getElementById("sort-select").value;
      const direction = document.getElementById("sort-direction").value;
      let a = left[field];
      let b = right[field];
      if (field === "identity_to_reference") {
        a = a == null ? -1 : Number(a);
        b = b == null ? -1 : Number(b);
      }
      let result = 0;
      if (typeof a === "number" && typeof b === "number") {
        result = a - b;
      } else {
        result = String(a == null ? "" : a).localeCompare(String(b == null ? "" : b), undefined, { numeric: true, sensitivity: "base" });
      }
      if (result === 0) {
        result = Number(left.record_index || 0) - Number(right.record_index || 0);
      }
      return direction === "desc" ? -result : result;
    }

    function chooseMajorResidue(counts, refResidue) {
      const entries = [...counts.entries()];
      if (!entries.length) return { aa: "-", count: 0 };
      entries.sort((left, right) => {
        const countDelta = right[1] - left[1];
        if (countDelta !== 0) return countDelta;
        if (left[0] === refResidue && right[0] !== refResidue) return -1;
        if (right[0] === refResidue && left[0] !== refResidue) return 1;
        return String(left[0]).localeCompare(String(right[0]), undefined, { sensitivity: "base" });
      });
      return { aa: entries[0][0], count: entries[0][1] };
    }

    function buildConsensus(rows, scope, columns, diagnostics, applyOffsets) {
      const sourceRows = consensusSourceRows(rows);
      const diagMap = diagnostics || new Map();
      const cells = new Map();
      for (const idx of columns) {
        const refResidue = scope.reference_residues[idx];
        const informativeCounts = new Map();
        const fallbackCounts = new Map();
        let shiftedRows = 0;
        for (const row of sourceRows) {
          const diag = diagMap.get(row.record_id);
          const shift = applyOffsets && diag && diag.offsetSuspect ? Number(diag.bestShift || 0) : 0;
          if (shift !== 0) shiftedRows += 1;
          const aa = residueAt(row, idx, shift) || "-";
          fallbackCounts.set(aa, (fallbackCounts.get(aa) || 0) + 1);
          if (isInformativeResidue(aa)) {
            informativeCounts.set(aa, (informativeCounts.get(aa) || 0) + 1);
          }
        }
        const informativeTotal = [...informativeCounts.values()].reduce((sum, count) => sum + count, 0);
        const fallbackTotal = [...fallbackCounts.values()].reduce((sum, count) => sum + count, 0);
        const useFallback = informativeTotal === 0;
        const major = chooseMajorResidue(useFallback ? fallbackCounts : informativeCounts, refResidue);
        const denominator = useFallback ? fallbackTotal : informativeTotal;
        cells.set(idx, {
          aa: major.aa,
          count: major.count,
          denominator,
          informativeTotal,
          rowCount: sourceRows.length,
          support: denominator ? major.count / denominator : null,
          fromFallback: useFallback,
          shiftedRows
        });
      }
      return cells;
    }

    function scoreRowAgainstConsensus(row, consensusCells, columns, shift) {
      let comparable = 0;
      let matches = 0;
      for (const idx of columns) {
        const aa = residueAt(row, idx, shift);
        const consensus = consensusCells.get(idx);
        const consensusAa = consensus ? consensus.aa : "";
        if (!isInformativeResidue(aa) || !isInformativeResidue(consensusAa)) continue;
        comparable += 1;
        if (aa === consensusAa) matches += 1;
      }
      return {
        agreement: comparable ? matches / comparable : null,
        comparable,
        matches
      };
    }

    function analyzeGroupOffsets(rows, scope, columns) {
      const diagnostics = new Map();
      if (!columns.length || rows.length < 2) return diagnostics;
      for (const row of rows) {
        if (row.is_reference) continue;
        let comparisonRows = consensusSourceRows(rows).filter((candidate) => candidate !== row);
        if (!comparisonRows.length) {
          comparisonRows = rows.filter((candidate) => candidate !== row);
        }
        if (!comparisonRows.length) continue;
        const leaveOneConsensus = buildConsensus(comparisonRows, scope, columns, new Map(), false);
        const raw = scoreRowAgainstConsensus(row, leaveOneConsensus, columns, 0);
        let best = { ...raw, shift: 0 };
        for (let shift = -OFFSET_RANGE; shift <= OFFSET_RANGE; shift += 1) {
          if (shift === 0) continue;
          const shifted = scoreRowAgainstConsensus(row, leaveOneConsensus, columns, shift);
          if (shifted.agreement != null && (best.agreement == null || shifted.agreement > best.agreement)) {
            best = { ...shifted, shift };
          }
        }
        const improvement = raw.agreement != null && best.agreement != null ? best.agreement - raw.agreement : 0;
        const offsetSuspect = (
          raw.comparable >= OFFSET_MIN_COMPARABLE &&
          raw.agreement != null &&
          raw.agreement < OFFSET_RAW_MAX &&
          best.shift !== 0 &&
          improvement >= OFFSET_MIN_IMPROVEMENT &&
          best.agreement >= OFFSET_MIN_SHIFTED
        );
        const groupOutlier = (
          raw.comparable >= OFFSET_MIN_COMPARABLE &&
          raw.agreement != null &&
          raw.agreement < OFFSET_RAW_MAX &&
          !offsetSuspect
        );
        diagnostics.set(row.record_id, {
          rawAgreement: raw.agreement,
          bestAgreement: best.agreement,
          bestShift: best.shift,
          comparable: raw.comparable,
          improvement,
          offsetSuspect,
          groupOutlier
        });
      }
      return diagnostics;
    }

    function diagnosticSummary(diagnostics) {
      const values = diagnostics ? [...diagnostics.values()] : [];
      return {
        offsetSuspects: values.filter((diag) => diag.offsetSuspect).length,
        outliers: values.filter((diag) => diag.groupOutlier).length
      };
    }

    function compareWindows(consensusCells, scope, columns) {
      const mode = document.getElementById("compare-mode").value;
      const minRun = Math.max(1, numberOrNull("compare-min-run") || 6);
      const highlighted = new Set();
      const runs = [];
      let run = [];

      function flushRun() {
        if (run.length >= minRun) {
          run.forEach((idx) => highlighted.add(idx));
          const startIdx = run[0];
          const endIdx = run[run.length - 1];
          const startRef = scope.reference_positions[startIdx];
          const endRef = scope.reference_positions[endIdx];
          runs.push({
            startIdx,
            endIdx,
            startRef,
            endRef,
            startVisibleOrdinal: columns.indexOf(startIdx),
            endVisibleOrdinal: columns.indexOf(endIdx),
            length: run.length
          });
        }
        run = [];
      }

      if (mode === "off") return { columns: highlighted, runs, runCount: 0 };
      for (const idx of columns) {
        const consensus = consensusCells.get(idx);
        const aa = consensus ? consensus.aa : "";
        const ref = scope.reference_residues[idx];
        const previous = run.length ? run[run.length - 1] : null;
        const contiguous = previous == null || idx === previous + 1;
        const similar = contiguous && similarToReference(aa, ref, mode);
        if (similar) {
          run.push(idx);
        } else {
          flushRun();
          if (similarToReference(aa, ref, mode)) run.push(idx);
        }
      }
      flushRun();
      return { columns: highlighted, runs, runCount: runs.length };
    }

    function compareDissimilarWindows(consensusCells, scope, columns) {
      const mode = document.getElementById("compare-mode").value;
      const minRun = Math.max(1, numberOrNull("compare-min-run") || 6);
      const highlighted = new Set();
      const runs = [];
      let run = [];

      function flushRun() {
        if (run.length >= minRun) {
          run.forEach((idx) => highlighted.add(idx));
          const startIdx = run[0];
          const endIdx = run[run.length - 1];
          const startRef = scope.reference_positions[startIdx];
          const endRef = scope.reference_positions[endIdx];
          runs.push({
            startIdx,
            endIdx,
            startRef,
            endRef,
            startVisibleOrdinal: columns.indexOf(startIdx),
            endVisibleOrdinal: columns.indexOf(endIdx),
            length: run.length
          });
        }
        run = [];
      }

      if (mode === "off") return { columns: highlighted, runs, runCount: 0 };
      for (const idx of columns) {
        const consensus = consensusCells.get(idx);
        const aa = consensus ? consensus.aa : "";
        const ref = scope.reference_residues[idx];
        const previous = run.length ? run[run.length - 1] : null;
        const contiguous = previous == null || idx === previous + 1;
        const dissimilar = (
          contiguous &&
          isInformativeResidue(aa) &&
          isInformativeResidue(ref) &&
          !similarToReference(aa, ref, mode)
        );
        if (dissimilar) {
          run.push(idx);
        } else {
          flushRun();
          if (isInformativeResidue(aa) && isInformativeResidue(ref) && !similarToReference(aa, ref, mode)) run.push(idx);
        }
      }
      flushRun();
      return { columns: highlighted, runs, runCount: runs.length };
    }

    function alphaFoldRangesFromGroups(groups) {
      if (!Array.isArray(groups)) return [];
      const compareMode = document.getElementById("compare-mode").value;
      const minRun = Math.max(1, numberOrNull("compare-min-run") || 6);
      const ranges = [];
      groups.forEach((group) => {
        const similarRuns = (group.windows && group.windows.runs) || [];
        similarRuns.forEach((run, index) => {
          if (run.startRef == null || run.endRef == null) return;
          ranges.push({
            source: "alignment_browser",
            kind: "conserved",
            label: `${group.label || "group"} conserved run ${index + 1}`,
            start: run.startRef,
            end: run.endRef,
            color: alphaFoldColor("conserved"),
            group: group.label || group.key || "group",
            score: "",
            compare_mode: compareMode,
            min_similar_run: minRun,
          });
        });
        const dissimilarRuns = (group.dissimilarWindows && group.dissimilarWindows.runs) || [];
        dissimilarRuns.forEach((run, index) => {
          if (run.startRef == null || run.endRef == null) return;
          ranges.push({
            source: "alignment_browser",
            kind: "divergent",
            label: `${group.label || "group"} non-conserved run ${index + 1}`,
            start: run.startRef,
            end: run.endRef,
            color: alphaFoldColor("divergent"),
            group: group.label || group.key || "group",
            score: "",
            compare_mode: compareMode,
            min_similar_run: minRun,
          });
        });
      });
      return ranges;
    }

    function rulerCell(scope, idx) {
      const alignPos = idx + 1;
      const refPos = scope.reference_positions[idx];
      const label = refPos != null ? refPos : alignPos;
      const show = alignPos === 1 || alignPos === scope.alignment_length || label % 10 === 0;
      const title = `Alignment ${alignPos}${refPos == null ? "" : `, reference ${refPos}`}`;
      return `<span class="ruler-cell" title="${escapeHtml(title)}">${show ? escapeHtml(label) : ""}</span>`;
    }

    function residueCell(row, scope, idx) {
      const seq = String(row.aligned_sequence || "").toUpperCase();
      const aa = seq[idx] || "";
      const ref = scope.reference_residues[idx];
      const refPos = scope.reference_positions[idx];
      const isGap = aa === "-" || aa === ".";
      const comparable = !isGap && ref && ref !== "-" && ref !== ".";
      const match = comparable && aa === ref;
      const color = PAYLOAD.aa_colors[aa] || PAYLOAD.aa_colors.X || "#f2f2f2";
      const classes = ["aa-cell"];
      if (isGap) classes.push("gap");
      if (row.is_reference) classes.push("reference");
      if (comparable && match) classes.push("match");
      if (comparable && !match) classes.push("mismatch");
      const title = `${row.species} | alignment ${idx + 1}${refPos == null ? "" : ` | reference ${refPos}`} | ${aa || " "} vs ${ref || " "}`;
      return `<span class="${classes.join(" ")}" style="background:${escapeHtml(color)}" title="${escapeHtml(title)}">${escapeHtml(aa)}</span>`;
    }

    function renderConsensusCell(group, scope, idx, consensusCell, highlightedColumns) {
      const aa = consensusCell.aa || "-";
      const ref = scope.reference_residues[idx];
      const refPos = scope.reference_positions[idx];
      const isGap = isGapResidue(aa);
      const comparable = !isGap && isInformativeResidue(ref);
      const exactMatch = comparable && aa === ref;
      const color = PAYLOAD.aa_colors[aa] || PAYLOAD.aa_colors.X || "#f2f2f2";
      const classes = ["aa-cell", "consensus"];
      const compareMode = document.getElementById("compare-mode").value;
      const highlighted = highlightedColumns.has(idx);
      if (isGap) classes.push("gap");
      if (consensusCell.support != null && consensusCell.support < 0.5) classes.push("low-support");
      if (comparable && !exactMatch) classes.push("mismatch");
      if (highlighted) {
        classes.push("compare-window");
        if (compareMode === "property") classes.push("property");
      }
      const supportText = consensusCell.denominator
        ? `${consensusCell.count}/${consensusCell.denominator} (${(consensusCell.support * 100).toFixed(1)}%)`
        : "0/0";
      const fallbackText = consensusCell.fromFallback ? " | no informative residues" : "";
      const shiftedText = consensusCell.shiftedRows ? ` | visual offset correction applied to ${consensusCell.shiftedRows} row(s)` : "";
      const title = `${group.label} consensus | alignment ${idx + 1}${refPos == null ? "" : ` | reference ${refPos}`} | ${aa || " "} vs ${ref || " "} | support ${supportText}${fallbackText}${shiftedText}`;
      return `<span class="${classes.join(" ")}" style="background:${escapeHtml(color)}" title="${escapeHtml(title)}">${escapeHtml(aa)}</span>`;
    }

    function renderConsensusRow(group, scope, columns) {
      const applyOffsets = offsetVisualEnabled();
      const consensusCells = group.consensusCells || buildConsensus(group.rows, scope, columns, group.diagnostics, applyOffsets);
      const windows = group.windows || compareWindows(consensusCells, scope, columns);
      const supportValues = [...consensusCells.values()].filter((cell) => cell.support != null).map((cell) => cell.support);
      const meanSupport = supportValues.length ? supportValues.reduce((sum, value) => sum + value, 0) / supportValues.length : null;
      const compareBadge = windows.runCount
        ? `<span class="badge compare">${windows.runCount} run${windows.runCount === 1 ? "" : "s"}</span>`
        : "";
      const offsetCount = [...(group.diagnostics || new Map()).values()].filter((diag) => diag.offsetSuspect).length;
      const offsetBadge = applyOffsets && offsetCount ? `<span class="badge offset">${offsetCount} shifted</span>` : "";
      const name = `${group.label} consensus`;
      const stats = `${group.rows.length} rows ${formatPercent(meanSupport)} ${compareBadge}${offsetBadge}`;
      const cells = columns.map((idx) => renderConsensusCell(group, scope, idx, consensusCells.get(idx), windows.columns)).join("");
      return `<div class="consensus-row"><div class="row-label consensus-label" title="${escapeHtml(name)}"><span class="row-name">${escapeHtml(name)}</span><span class="row-stats">${stats}</span></div>${cells}</div>`;
    }

    function renderRow(row, scope, columns, diagnostics) {
      const name = `${row.species} (${row.symbol || "unknown"})`;
      const diag = diagnostics ? diagnostics.get(row.record_id) : null;
      const labelClasses = ["row-label"];
      const recordKey = String(row.record_id || "");
      const badges = [];
      const details = [];
      if (TREE_SELECTED_RECORDS.has(recordKey)) labelClasses.push("tree-row-selected");
      if (TREE_SEARCH_MATCH_RECORDS.has(recordKey)) labelClasses.push("tree-row-match");
      if (diag && diag.offsetSuspect) {
        labelClasses.push("offset-suspect");
        badges.push(`<span class="badge offset">offset ${formatSigned(diag.bestShift)}</span>`);
        details.push(`offset candidate ${formatSigned(diag.bestShift)}: ${formatPercent(diag.rawAgreement)} to ${formatPercent(diag.bestAgreement)}`);
      } else if (diag && diag.groupOutlier) {
        labelClasses.push("group-outlier");
        badges.push('<span class="badge outlier">outlier</span>');
        details.push(`low group agreement: ${formatPercent(diag.rawAgreement)}`);
      }
      const stats = `${escapeHtml(formatLabel(row.clade))} ${escapeHtml(formatPercent(row.identity_to_reference))} ${badges.join("")}`;
      const cells = columns.map((idx) => residueCell(row, scope, idx)).join("");
      const title = [row.record_id].concat(details).filter(Boolean).join(" | ");
      return `<div class="alignment-row" data-row-record-id="${escapeHtml(recordKey)}"><div class="${labelClasses.join(" ")}" title="${escapeHtml(title)}"><span class="row-name">${escapeHtml(name)}</span><span class="row-stats">${stats}</span></div>${cells}</div>`;
    }

    function groupedRows(rows) {
      const groupField = currentGroupField();
      if (groupField === "none") return [{ key: "all", label: "", rows }];
      const groups = new Map();
      for (const row of rows) {
        const raw = row[groupField] || "Unassigned";
        const key = String(raw);
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(row);
      }
      return [...groups.entries()]
        .map(([key, groupRows]) => ({ key, label: formatLabel(key), rows: groupRows }))
        .sort((a, b) => Math.min(...a.rows.map((r) => Number(r.tree_order || 999999))) - Math.min(...b.rows.map((r) => Number(r.tree_order || 999999))));
    }

    function renderGroupHeader(group) {
      const collapseKey = `${currentGroupField()}:${group.key}`;
      const collapsed = COLLAPSED_GROUPS.has(collapseKey);
      const summary = diagnosticSummary(group.diagnostics);
      const metaParts = [];
      if (summary.offsetSuspects) metaParts.push(`<span class="badge offset">${summary.offsetSuspects} offset</span>`);
      if (summary.outliers) metaParts.push(`<span class="badge outlier">${summary.outliers} outlier</span>`);
      const meta = metaParts.length ? `<span class="group-meta">${metaParts.join("")}</span>` : "";
      return `<div class="group-row"><button type="button" data-collapse-key="${escapeHtml(collapseKey)}">${collapsed ? "+" : "-"} ${escapeHtml(group.label)} (${group.rows.length})</button>${meta}</div>`;
    }

    function renderMetrics(scope, rows, columns, groups) {
      const meta = PAYLOAD.meta || {};
      const allDiagnostics = [];
      for (const group of groups) {
        for (const diag of (group.diagnostics || new Map()).values()) {
          allDiagnostics.push(diag);
        }
      }
      const offsetCount = allDiagnostics.filter((diag) => diag.offsetSuspect).length;
      const outlierCount = allDiagnostics.filter((diag) => diag.groupOutlier).length;
      document.getElementById("run-meta").textContent = [meta.gene_symbol, meta.output_name].filter(Boolean).join(" | ");
      document.getElementById("metrics").innerHTML = [
        ["Rows", rows.length],
        ["Visible columns", columns.length],
        ["Alignment length", scope.alignment_length || 0],
        ["Reference", scope.reference_species || ""],
        ["View", compressedActive() ? "Compressed" : "Detailed"],
        ["Compare", document.getElementById("compare-mode").value],
        ["Offset flags", offsetCount],
        ["Outliers", outlierCount],
        ["Tree source", rows[0] ? rows[0].tree_order_source : ""],
      ].map(([label, value]) => `<span class="metric">${escapeHtml(label)} <strong>${escapeHtml(value)}</strong></span>`).join("");
    }

    function compressedCollapseSignature() {
      return `${currentScopeKey()}:${currentGroupField()}:compressed`;
    }

    function ensureCompressedDefaultCollapse(groups) {
      if (!compressedActive()) {
        LAST_COMPRESSED_COLLAPSE_SIGNATURE = "";
        return;
      }
      const signature = compressedCollapseSignature();
      if (signature === LAST_COMPRESSED_COLLAPSE_SIGNATURE) return;
      for (const group of groups) {
        if (!group.label) continue;
        COLLAPSED_GROUPS.add(`${currentGroupField()}:${group.key}`);
      }
      LAST_COMPRESSED_COLLAPSE_SIGNATURE = signature;
      SELECTED_RUN_KEY = "";
    }

    function expandGroupsForTreeState(groups) {
      const activeRecords = new Set([...TREE_SELECTED_RECORDS, ...TREE_SEARCH_MATCH_RECORDS]);
      if (!activeRecords.size) return;
      for (const group of groups) {
        if (!group.label) continue;
        if (group.rows.some((row) => activeRecords.has(String(row.record_id || "")))) {
          COLLAPSED_GROUPS.delete(`${currentGroupField()}:${group.key}`);
        }
      }
    }

    function referenceTrackLength(scope) {
      const positions = (scope.reference_positions || []).filter((value) => value != null).map((value) => Number(value));
      if (positions.length) return Math.max(...positions);
      return Number(scope.alignment_length || 0);
    }

    function architectureLandmarks(scope) {
      return scope && Array.isArray(scope.reference_landmarks) ? scope.reference_landmarks : [];
    }

    function architectureLandmarkRows(scope) {
      const landmarks = architectureLandmarks(scope);
      const rows = [];
      const rowMap = new Map();
      landmarks.forEach((landmark, index) => {
        const rowKey = String(landmark.row_key || landmark.key || landmark.row_label || landmark.label || `landmark_${index + 1}`);
        let row = rowMap.get(rowKey);
        if (!row) {
          row = {
            key: rowKey,
            label: String(landmark.row_label || landmark.label || landmark.description || "Landmark"),
            segments: []
          };
          rowMap.set(rowKey, row);
          rows.push(row);
        }
        row.segments.push({ ...landmark });
      });
      rows.forEach((row) => {
        row.segments.sort((a, b) => Number(a.start || 0) - Number(b.start || 0) || Number(a.end || 0) - Number(b.end || 0));
        row.range_label = row.segments.map((segment) => String(segment.range_label || `${segment.start}-${segment.end}`)).join("; ");
      });
      return rows;
    }

    function architectureTickPositions(trackLength, step = 100) {
      const safeLength = Math.max(1, Number(trackLength || 0));
      const ticks = [1];
      for (let value = step; value <= safeLength; value += step) ticks.push(value);
      if (ticks[ticks.length - 1] !== safeLength) ticks.push(safeLength);
      return [...new Set(ticks)].sort((a, b) => a - b);
    }

    function architectureRangePercent(start, end, trackLength) {
      const safeTrackLength = Math.max(1, Number(trackLength || 0));
      const startNum = Number(start || 1);
      const endNum = Number(end || startNum);
      const left = Math.max(0, ((startNum - 1) / safeTrackLength) * 100);
      const width = Math.max(0.25, ((endNum - startNum + 1) / safeTrackLength) * 100);
      return { left, width };
    }

    function architectureShowInlineLandmarkLabel(start, end, trackLength, bandWidthPx = null) {
      const span = Math.max(1, Number(end || start || 1) - Number(start || 1) + 1);
      const range = architectureRangePercent(start, end, trackLength);
      const widthPercent = Number(range.width || 0);
      if (span < 24 || widthPercent < 6) return false;
      if (bandWidthPx != null && Number(bandWidthPx) < 85) return false;
      return true;
    }

    function architectureSecondaryRanges(trackLength) {
      const ss = (ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.secondary_structure) || {};
      const maxResidue = Math.max(1, Number(trackLength || alphaFoldResidueCount() || 1));
      const ranges = (ss.ranges || []).map((range) => {
        const clamped = alphaFoldClampRange(range);
        if (!clamped) return null;
        return {
          ...clamped,
          start: Math.max(1, Math.min(maxResidue, Number(clamped.start))),
          end: Math.max(1, Math.min(maxResidue, Number(clamped.end))),
          kind: ["helix", "sheet", "loop"].includes(clamped.kind) ? clamped.kind : "loop",
        };
      }).filter(Boolean);
      return alphaFoldEnumerateSecondaryRanges(ranges);
    }

    function architectureLocalChargeRows(trackLength) {
      const localCharge = (ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.local_charge) || {};
      if (!localCharge.available || !Array.isArray(localCharge.residues)) return [];
      const maxResidue = Math.max(1, Number(trackLength || alphaFoldResidueCount() || 1));
      return localCharge.residues.map((row) => {
        const position = Number(row.position);
        if (!Number.isFinite(position) || position < 1 || position > maxResidue) return null;
        return {
          ...row,
          position,
          charge: Number(row.charge || 0),
          color: row.color || alphaFoldColor("charge_neutral"),
        };
      }).filter(Boolean);
    }

    function architectureCalciumPayload(trackLength) {
      const calcium = (ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.calcium_binding) || {};
      if (!calcium.available) return { loops: [], ligands: [], sites: [] };
      const maxResidue = Math.max(1, Number(trackLength || alphaFoldResidueCount() || 1));
      const loops = (calcium.loops || []).map((loop) => {
        const start = Math.max(1, Math.min(maxResidue, Number(loop.start || 1)));
        const end = Math.max(1, Math.min(maxResidue, Number(loop.end || loop.start || 1)));
        return { ...loop, start: Math.min(start, end), end: Math.max(start, end), color: loop.color || alphaFoldColor("calcium") };
      });
      const ligands = (calcium.ligands || []).map((ligand) => {
        const position = Math.max(1, Math.min(maxResidue, Number(ligand.position || 1)));
        return { ...ligand, position };
      });
      return { loops, ligands, sites: calcium.sites || [] };
    }

    function architectureSecondarySvg(trackLength) {
      const ranges = architectureSecondaryRanges(trackLength);
      if (!ranges.length) return "";
      const width = 1000;
      const height = 36;
      const parts = [
        `<svg class="architecture-secondary-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="AlphaFold secondary structure overlay">`,
        `<line x1="0" y1="18" x2="${width}" y2="18" stroke="${escapeHtml(alphaFoldColor("loop"))}" stroke-width="6" stroke-linecap="round" opacity="0.72"/>`,
      ];
      ranges.forEach((range) => {
        if (range.kind === "loop") return;
        const start = Number(range.start || 1);
        const end = Number(range.end || start);
        const x1 = Math.max(0, ((start - 1) / Math.max(trackLength, 1)) * width);
        const x2 = Math.min(width, (end / Math.max(trackLength, 1)) * width);
        const color = alphaFoldColor(range.kind);
        if (range.kind === "helix") {
          parts.push(alphaFoldHelixElement(x1, x2, 18, escapeHtml(color), "", escapeHtml(`helix ${start}-${end}`), 5));
        } else if (range.kind === "sheet") {
          const tip = Math.max(0.01, Math.min(16, (x2 - x1) * 0.45));
          parts.push(`<polygon points="${x1.toFixed(2)},9 ${(x2 - tip).toFixed(2)},9 ${(x2 - tip).toFixed(2)},4 ${x2.toFixed(2)},18 ${(x2 - tip).toFixed(2)},32 ${(x2 - tip).toFixed(2)},27 ${x1.toFixed(2)},27" fill="${escapeHtml(color)}" opacity="0.9"><title>${escapeHtml(`sheet ${start}-${end}`)}</title></polygon>`);
        }
        const displayLabel = range.display_label || "";
        if (displayLabel) {
          const labelTitle = `${displayLabel} ${range.kind} ${start}-${end}`;
          parts.push(`<text class="architecture-svg-ss-label" x="${((x1 + x2) / 2).toFixed(2)}" y="7" text-anchor="middle" font-size="8" font-weight="900" fill="${escapeHtml(color)}" paint-order="stroke" stroke="#ffffff" stroke-width="2" stroke-linejoin="round"><title>${escapeHtml(labelTitle)}</title>${escapeHtml(displayLabel)}</text>`);
        }
      });
      parts.push("</svg>");
      return parts.join("");
    }

    function architectureLocalChargeSvg(trackLength) {
      const rows = architectureLocalChargeRows(trackLength);
      if (!rows.length) return "";
      const width = 1000;
      const height = 24;
      const binWidth = Math.max(0.6, width / Math.max(trackLength, 1));
      const parts = [
        `<svg class="architecture-secondary-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="5 amino acid local charge overlay">`,
      ];
      rows.forEach((row) => {
        const x = ((row.position - 1) / Math.max(trackLength, 1)) * width;
        const title = `pos ${row.position} charge ${Number(row.charge).toFixed(1)} window ${row.window_start}-${row.window_end} ${row.window_sequence || ""}`;
        parts.push(`<rect x="${x.toFixed(3)}" y="0" width="${binWidth.toFixed(3)}" height="${height}" fill="${escapeHtml(row.color)}"><title>${escapeHtml(title)}</title></rect>`);
      });
      parts.push(`<line x1="0" y1="${height / 2}" x2="${width}" y2="${height / 2}" stroke="#475569" stroke-width="0.7" opacity="0.35"/>`);
      parts.push("</svg>");
      return parts.join("");
    }

    function architectureCalciumSvg(trackLength) {
      const calcium = architectureCalciumPayload(trackLength);
      if (!calcium.loops.length && !calcium.ligands.length) return "";
      const width = 1000;
      const height = 24;
      const parts = [
        `<svg class="architecture-secondary-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="Ca2+ binding site overlay">`,
      ];
      calcium.loops.forEach((loop) => {
        const x = ((loop.start - 1) / Math.max(trackLength, 1)) * width;
        const w = ((loop.end - loop.start + 1) / Math.max(trackLength, 1)) * width;
        const title = `${loop.label || "CBR"} ${loop.start}-${loop.end}${loop.description ? " | " + loop.description : ""}`;
        parts.push(`<rect x="${x.toFixed(3)}" y="3" width="${Math.max(1.5, w).toFixed(3)}" height="18" rx="4" fill="${escapeHtml(loop.color)}" opacity="0.88"><title>${escapeHtml(title)}</title></rect>`);
        if (w >= 26) parts.push(`<text x="${(x + w / 2).toFixed(3)}" y="16" text-anchor="middle" font-size="9" font-weight="900" fill="#ffffff">${escapeHtml(loop.label || "")}</text>`);
      });
      calcium.ligands.forEach((ligand) => {
        const cx = ((Number(ligand.position) - 0.5) / Math.max(trackLength, 1)) * width;
        const sites = (ligand.sites || []).join(", ");
        const title = `${ligand.label || "ligand"} ${ligand.position}${sites ? " | " + sites : ""}`;
        parts.push(`<circle cx="${cx.toFixed(3)}" cy="12" r="3" fill="${escapeHtml(ligand.color || alphaFoldColor("calcium_ligand"))}"><title>${escapeHtml(title)}</title></circle>`);
      });
      parts.push("</svg>");
      return parts.join("");
    }

    function architectureSecondaryRowHtml(trackLength) {
      const overlay = architectureSecondarySvg(trackLength);
      if (!overlay) return "";
      const labels = architectureSecondaryRanges(trackLength).map((range) => {
        const displayLabel = range.display_label || "";
        if (!displayLabel || !["helix", "sheet"].includes(range.kind)) return "";
        const center = (((Number(range.start || 1) - 1) + (Number(range.end || range.start || 1))) / 2) / Math.max(trackLength, 1) * 100;
        const title = `${displayLabel} ${range.kind} ${range.start}-${range.end}`;
        return `<span class="architecture-ss-number" style="left:${center.toFixed(4)}%;color:${escapeHtml(alphaFoldColor(range.kind))}" title="${escapeHtml(title)}">${escapeHtml(displayLabel)}</span>`;
      }).join("");
      return `<div class="architecture-row annotation"><div class="architecture-label">AlphaFold SS</div><div class="architecture-track annotation-track domain-track">${overlay}${labels}</div><div class="architecture-count range">helix/sheet/loop</div></div>`;
    }

    function comparativeAlphaFoldPayload() {
      return (ALPHAFOLD_STRUCTURE && ALPHAFOLD_STRUCTURE.comparative_secondary_structure) || {};
    }

    function comparativeAlphaFoldRecords() {
      const payload = comparativeAlphaFoldPayload();
      if (Array.isArray(payload.records)) return payload.records;
      if (Array.isArray(payload.models)) return payload.models;
      return [];
    }

    function comparativeAlphaFoldRecordLabel(count) {
      const payload = comparativeAlphaFoldPayload();
      const singular = payload.record_label_singular || "AF model";
      const plural = payload.record_label_plural || `${singular}s`;
      return Number(count) === 1 ? singular : plural;
    }

    function comparativeAlphaFoldRecordLookup() {
      const lookup = new Map();
      comparativeAlphaFoldRecords().forEach((entry) => {
        [
          entry.record_id,
          entry.protein_record_id,
          entry.species,
          entry.alphafold_entry_id,
          entry.uniprot_accession,
        ].filter(Boolean).forEach((key) => lookup.set(String(key), entry));
      });
      return lookup;
    }

    function architectureSelectedComparativeRows(scope) {
      const rows = Array.isArray(scope.records) ? scope.records : [];
      const byRecord = new Map(rows.map((row) => [String(row.record_id || ""), row]));
      const selectedIds = new Set();
      TREE_SELECTED_RECORDS.forEach((recordId) => selectedIds.add(String(recordId || "")));
      SNAPSHOT_SELECTED_RECORDS.forEach((recordId) => selectedIds.add(String(recordId || "")));
      if (!selectedIds.size) return [];
      return [...selectedIds].map((recordId) => byRecord.get(recordId)).filter(Boolean);
    }

    function comparativeAlphaFoldEntryForRow(row, lookup) {
      if (!row) return null;
      const keys = [row.record_id, row.protein_record_id, row.species, row.alphafold_entry_id, row.uniprot_accession]
        .filter(Boolean)
        .map((value) => String(value));
      for (const key of keys) {
        if (lookup.has(key)) return lookup.get(key);
      }
      return null;
    }

    function comparativeAlphaFoldRanges(entry) {
      if (!entry) return [];
      if (Array.isArray(entry.mapped_ranges)) return entry.mapped_ranges;
      if (Array.isArray(entry.reference_mapped_ranges)) return entry.reference_mapped_ranges;
      if (Array.isArray(entry.ranges)) return entry.ranges;
      return [];
    }

    function comparativeAlphaFoldSourceSummary(entries) {
      const normalizedEntries = Array.isArray(entries) ? entries : [];
      return normalizedEntries.map((entry) => {
        const method = String(entry.source_method || "").includes("species_alphafold")
          ? "real species AF"
          : "human-projected";
        const accession = entry.uniprot_accession || entry.alphafold_entry_id || entry.protein_record_id || entry.species || "SS map";
        const modelCount = Number(entry.model_residue_count || entry.species_sequence_length || 0);
        const mappedCount = Number(entry.mapped_residue_count || 0);
        const unmappedCount = Number(entry.unmapped_species_residue_count || 0);
        const proteinLen = Number(entry.species_sequence_length || entry.model_residue_count || 0);
        const lenText = Number.isFinite(proteinLen) && proteinLen > 0 ? `full ${proteinLen}-aa protein` : "length unavailable";
        const insertionText = Number.isFinite(unmappedCount) && unmappedCount > 0 ? ` (${unmappedCount} species-specific insertion${unmappedCount === 1 ? "" : "s"} align to human gaps, no reference mark)` : "";
        const ssText = Number.isFinite(mappedCount) && mappedCount > 0 ? `; SS mapped at ${mappedCount} human-reference positions${insertionText}` : "";
        return `${accession}: ${method}; ${lenText}${ssText}`;
      }).join("; ");
    }

    function comparativeAlphaFoldEntryLabel(entry) {
      const speciesLabel = (entry && (
        entry.common_name
        || entry.species_display_label
        || entry.species
      )) || "species";
      const text = formatLabel(speciesLabel);
      return text ? text.charAt(0).toUpperCase() + text.slice(1) : "Species";
    }

    function comparativeAlphaFoldProteinId(entry) {
      if (!entry) return "";
      const recordId = String(entry.record_id || "");
      const proteinMatch = recordId.match(/(?:^|\|)Protein=([^|]+)/);
      if (proteinMatch && proteinMatch[1]) return proteinMatch[1];
      const proteinRecord = String(entry.protein_record_id || "").trim();
      if (proteinRecord.includes("__")) return proteinRecord.split("__").pop();
      return proteinRecord || String(entry.alphafold_entry_id || entry.uniprot_accession || "");
    }

    function comparativeAlphaFoldShortLabel(entry) {
      const label = comparativeAlphaFoldEntryLabel(entry);
      return label.length > 16 ? `${label.slice(0, 15)}...` : label;
    }

    function comparativeAlphaFoldShortProteinId(entry) {
      const label = comparativeAlphaFoldProteinId(entry);
      return label.length > 22 ? `${label.slice(0, 21)}...` : label;
    }

    function comparativeRangeStart(range) {
      return Number(range.start_reference_position ?? range.startRef ?? range.reference_start ?? range.start ?? NaN);
    }

    function comparativeRangeEnd(range) {
      return Number(range.end_reference_position ?? range.endRef ?? range.reference_end ?? range.end ?? NaN);
    }

    function comparativeRangeKind(range) {
      const kind = String(range.kind || range.secondary_structure || range.ss || "loop").toLowerCase();
      if (kind.startsWith("h")) return "helix";
      if (kind.startsWith("s") || kind.startsWith("e") || kind.startsWith("b")) return "sheet";
      return "loop";
    }

    function comparativeAlphaFoldRangeTitle(entry, range, kind, referenceStart, referenceEnd) {
      const label = String(range.display_label || "").trim();
      const prefix = `${label ? `${label} ` : ""}${kind}`;
      const parts = [prefix];
      const speciesStart = Number(range.start_species_position ?? range.species_start ?? range.startSpecies ?? NaN);
      const speciesEnd = Number(range.end_species_position ?? range.species_end ?? range.endSpecies ?? NaN);
      const realAf = String((entry && entry.source_method) || "").includes("species_alphafold");
      const modelCount = Number((entry && entry.model_residue_count) || 0);
      if (Number.isFinite(speciesStart) && Number.isFinite(speciesEnd)) {
        const left = Math.min(speciesStart, speciesEnd);
        const right = Math.max(speciesStart, speciesEnd);
        const modelSuffix = realAf && Number.isFinite(modelCount) && modelCount > 0 ? ` of ${modelCount}` : "";
        parts.push(`${realAf ? "AF residues" : "aligned species residues"} ${left}-${right}${modelSuffix}`);
      }
      parts.push(`ref ${referenceStart}-${referenceEnd}`);
      if (Number.isFinite(modelCount) && modelCount > 0) {
        parts.push(`${modelCount} aa model`);
      }
      const unmappedCount = Number((entry && entry.unmapped_species_residue_count) || 0);
      if (Number.isFinite(unmappedCount) && unmappedCount > 0) {
        parts.push(`${unmappedCount} no-reference residues omitted from this human-ruler track`);
      }
      return parts.join("; ");
    }

    function comparativeAlphaFoldDisplayRanges(entries, trackLength) {
      const maxResidue = Math.max(1, Number(trackLength || 1));
      const normalizedEntries = Array.isArray(entries) ? entries : [];
      if (!normalizedEntries.length) return [];
      if (normalizedEntries.length === 1) {
        const entry = normalizedEntries[0];
        return comparativeAlphaFoldEntryDisplayRanges(entry, trackLength);
      }

      const bins = Array.from({ length: maxResidue }, () => ({ helix: 0, sheet: 0, loop: 0, total: 0 }));
      normalizedEntries.forEach((entry) => {
        comparativeAlphaFoldRanges(entry).forEach((range) => {
          const start = Math.max(1, Math.min(maxResidue, Math.floor(comparativeRangeStart(range))));
          const end = Math.max(1, Math.min(maxResidue, Math.floor(comparativeRangeEnd(range))));
          if (!Number.isFinite(start) || !Number.isFinite(end)) return;
          const kind = comparativeRangeKind(range);
          for (let pos = Math.min(start, end); pos <= Math.max(start, end); pos += 1) {
            bins[pos - 1][kind] += 1;
            bins[pos - 1].total += 1;
          }
        });
      });

      const counters = { helix: 0, sheet: 0 };
      const ranges = [];
      let active = null;
      const closeActive = (endPosition) => {
        if (!active) return;
        const row = {
          kind: active.kind,
          start: active.start,
          end: endPosition,
          length: endPosition - active.start + 1,
          title: `consensus ${active.kind}; ref ${active.start}-${endPosition}; across ${normalizedEntries.length} SS maps`,
        };
        if (row.kind === "helix" || row.kind === "sheet") {
          counters[row.kind] += 1;
          row.display_label = `${row.kind === "helix" ? "H" : "S"}${counters[row.kind]}`;
        }
        ranges.push(row);
        active = null;
      };
      bins.forEach((bin, index) => {
        const position = index + 1;
        if (!bin.total) {
          closeActive(position - 1);
          return;
        }
        const dominant = bin.helix >= bin.sheet && bin.helix >= bin.loop
          ? "helix"
          : (bin.sheet >= bin.loop ? "sheet" : "loop");
        if (active && active.kind === dominant) return;
        closeActive(position - 1);
        active = { kind: dominant, start: position };
      });
      closeActive(maxResidue);
      return ranges;
    }

    function comparativeAlphaFoldEntryDisplayRanges(entry, trackLength) {
      const maxResidue = Math.max(1, Number(trackLength || 1));
      return comparativeAlphaFoldRanges(entry).map((range) => {
        const start = Math.max(1, Math.min(maxResidue, Math.floor(comparativeRangeStart(range))));
        const end = Math.max(1, Math.min(maxResidue, Math.floor(comparativeRangeEnd(range))));
        if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
        const kind = comparativeRangeKind(range);
        const left = Math.min(start, end);
        const right = Math.max(start, end);
        return {
          ...range,
          kind,
          start: left,
          end: right,
          display_label: range.display_label || "",
          title: comparativeAlphaFoldRangeTitle(entry, range, kind, left, right),
        };
      }).filter(Boolean).sort((left, right) => Number(left.start) - Number(right.start) || Number(left.end) - Number(right.end));
    }

    function architectureComparativeSelection(scope) {
      const selectedRows = architectureSelectedComparativeRows(scope);
      const lookup = comparativeAlphaFoldRecordLookup();
      const entries = selectedRows
        .map((row) => comparativeAlphaFoldEntryForRow(row, lookup))
        .filter(Boolean);
      const seen = new Set();
      const uniqueEntries = entries.filter((entry) => {
        const key = String(entry.record_id || entry.protein_record_id || entry.species || entry.alphafold_entry_id || JSON.stringify(entry));
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
      return { selectedRows, entries: uniqueEntries };
    }

    function architectureComparativeSecondarySvg(trackLength, scope) {
      const payload = comparativeAlphaFoldPayload();
      const selection = architectureComparativeSelection(scope);
      const selectedCount = selection.selectedRows.length;
      const entries = selection.entries;
      const width = 1000;
      const laneLimit = entries.length > 1 ? Math.min(entries.length, 16) : 1;
      const laneStep = 24;
      const laneGutter = entries.length > 1 ? 150 : 0;
      const plotX = laneGutter;
      const plotWidth = Math.max(1, width - plotX);
      const height = entries.length > 1 ? Math.max(36, laneLimit * laneStep + 8) : 36;
      const parts = [
        `<svg class="architecture-secondary-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="Comparative AlphaFold secondary structure overlay">`,
      ];
      let countLabel = "select species";
      let emptyLabel = "empty until species are selected";
      let detailLabel = "";
      let displayRanges = [];
      const appendRange = (range, y, stacked) => {
        if (range.kind === "loop") return;
        const start = Number(range.start || 1);
        const end = Number(range.end || start);
        const x1 = Math.max(plotX, plotX + ((start - 1) / Math.max(trackLength, 1)) * plotWidth);
        const x2 = Math.min(width, plotX + (end / Math.max(trackLength, 1)) * plotWidth);
        const color = alphaFoldColor(range.kind);
        const title = range.title || `${range.kind}; ref ${start}-${end}`;
        if (range.kind === "helix") {
          parts.push(alphaFoldHelixElement(x1, x2, y, escapeHtml(color), "", escapeHtml(title), stacked ? 3.2 : 5));
        } else if (range.kind === "sheet") {
          const tip = Math.max(0.01, Math.min(stacked ? 10 : 16, (x2 - x1) * 0.45));
          const top = stacked ? y - 5 : 9;
          const high = stacked ? y - 9 : 4;
          const bottom = stacked ? y + 5 : 27;
          const low = stacked ? y + 9 : 32;
          const mid = stacked ? y : 18;
          parts.push(`<polygon points="${x1.toFixed(2)},${top.toFixed(2)} ${(x2 - tip).toFixed(2)},${top.toFixed(2)} ${(x2 - tip).toFixed(2)},${high.toFixed(2)} ${x2.toFixed(2)},${mid.toFixed(2)} ${(x2 - tip).toFixed(2)},${low.toFixed(2)} ${(x2 - tip).toFixed(2)},${bottom.toFixed(2)} ${x1.toFixed(2)},${bottom.toFixed(2)}" fill="${escapeHtml(color)}" opacity="0.9"><title>${escapeHtml(title)}</title></polygon>`);
        }
        if (!stacked) {
          const displayLabel = range.display_label || "";
          if (displayLabel) {
            const labelTitle = range.title || `${displayLabel} ${range.kind}; ref ${start}-${end}`;
            parts.push(`<text class="architecture-svg-ss-label" x="${((x1 + x2) / 2).toFixed(2)}" y="7" text-anchor="middle" font-size="8" font-weight="900" fill="${escapeHtml(color)}" paint-order="stroke" stroke="#ffffff" stroke-width="2" stroke-linejoin="round"><title>${escapeHtml(labelTitle)}</title>${escapeHtml(displayLabel)}</text>`);
          }
        }
      };
      if (selectedCount && !entries.length) {
        parts.push(`<line x1="0" y1="18" x2="${width}" y2="18" stroke="${escapeHtml(alphaFoldColor("loop"))}" stroke-width="6" stroke-linecap="round" opacity="0.72"/>`);
        countLabel = payload.available ? `0/${selectedCount} ${comparativeAlphaFoldRecordLabel(0)}` : "bundle not built";
        emptyLabel = payload.available ? "selected species have no mapped AF SS" : "comparative AF SS bundle not built";
        detailLabel = emptyLabel;
      } else if (entries.length === 1) {
        parts.push(`<line x1="0" y1="18" x2="${width}" y2="18" stroke="${escapeHtml(alphaFoldColor("loop"))}" stroke-width="6" stroke-linecap="round" opacity="0.72"/>`);
        countLabel = "helix/sheet/loop";
        detailLabel = comparativeAlphaFoldSourceSummary(entries);
        displayRanges = comparativeAlphaFoldDisplayRanges(entries, trackLength);
        displayRanges.forEach((range) => appendRange(range, 18, false));
        emptyLabel = "";
      } else if (entries.length > 1) {
        countLabel = "helix/sheet/loop";
        const visibleEntries = entries.slice(0, laneLimit);
        detailLabel = `${entries.length} selected SS maps${entries.length > laneLimit ? `; showing first ${laneLimit}` : ""}; ${comparativeAlphaFoldSourceSummary(entries)}`;
        parts.push(`<line x1="${laneGutter - 6}" y1="3" x2="${laneGutter - 6}" y2="${height - 3}" stroke="#cbd5e1" stroke-width="1"/>`);
        visibleEntries.forEach((entry, index) => {
          const y = 15 + index * laneStep;
          const laneTitle = `${comparativeAlphaFoldEntryLabel(entry)}; ${comparativeAlphaFoldSourceSummary([entry])}`;
          parts.push(`<line x1="${laneGutter}" y1="${y}" x2="${width}" y2="${y}" stroke="${escapeHtml(alphaFoldColor("loop"))}" stroke-width="4" stroke-linecap="round" opacity="0.64"><title>${escapeHtml(laneTitle)}</title></line>`);
          comparativeAlphaFoldEntryDisplayRanges(entry, trackLength).forEach((range) => appendRange(range, y, true));
          parts.push(`<text x="4" y="${Math.max(8, y - 4)}"><title>${escapeHtml(laneTitle)}</title><tspan class="architecture-comparative-lane-label" x="4">${escapeHtml(comparativeAlphaFoldShortLabel(entry))}</tspan><tspan class="architecture-comparative-lane-id" x="4" dy="9">${escapeHtml(comparativeAlphaFoldShortProteinId(entry))}</tspan></text>`);
        });
        emptyLabel = "";
      }
      parts.push("</svg>");
      const empty = emptyLabel ? `<span class="architecture-comparative-empty">${escapeHtml(emptyLabel)}</span>` : "";
      return { overlay: parts.join(""), empty, countLabel, detailLabel, ranges: displayRanges, stacked: entries.length > 1, trackHeight: height };
    }

    function architectureComparativeSecondaryRowHtml(trackLength, scope) {
      const result = architectureComparativeSecondarySvg(trackLength, scope);
      const labels = result.stacked ? "" : (result.ranges || []).map((range) => {
        const displayLabel = range.display_label || "";
        if (!displayLabel || !["helix", "sheet"].includes(range.kind)) return "";
        const center = (((Number(range.start || 1) - 1) + (Number(range.end || range.start || 1))) / 2) / Math.max(trackLength, 1) * 100;
        const title = range.title || `${displayLabel} ${range.kind}; ref ${range.start}-${range.end}`;
        return `<span class="architecture-ss-number" style="left:${center.toFixed(4)}%;color:${escapeHtml(alphaFoldColor(range.kind))}" title="${escapeHtml(title)}">${escapeHtml(displayLabel)}</span>`;
      }).join("");
      const rowStyle = result.trackHeight > 36 ? ` style="min-height:${Math.ceil(result.trackHeight + 8)}px"` : "";
      const trackStyle = result.trackHeight ? ` style="height:${Math.ceil(result.trackHeight)}px"` : "";
      return `<div class="architecture-row annotation"${rowStyle}><div class="architecture-label">Comparative AF SS</div><div class="architecture-track annotation-track domain-track"${trackStyle}>${result.overlay}${labels}${result.empty}</div><div class="architecture-count range" title="${escapeHtml(result.detailLabel || result.countLabel)}">${escapeHtml(result.countLabel)}</div></div>`;
    }

    function architectureLocalChargeRowHtml(trackLength) {
      const overlay = architectureLocalChargeSvg(trackLength);
      if (!overlay) return "";
      return `<div class="architecture-row annotation"><div class="architecture-label">5-aa charge</div><div class="architecture-track annotation-track domain-track">${overlay}</div><div class="architecture-count range">negative / neutral / positive</div></div>`;
    }

    function architectureCalciumRowHtml(trackLength) {
      const overlay = architectureCalciumSvg(trackLength);
      if (!overlay) return "";
      return `<div class="architecture-row annotation"><div class="architecture-label">Ca2+ binding</div><div class="architecture-track annotation-track domain-track">${overlay}</div><div class="architecture-count range">CBR1-3 + Ca/PC contacts</div></div>`;
    }

    function architectureSecondarySvgExport(trackX, trackWidth, y, trackLength) {
      const ranges = architectureSecondaryRanges(trackLength);
      if (!ranges.length) return [];
      const parts = [
        `<text class="label" x="${18}" y="${y + 17}">AlphaFold SS</text>`,
        `<text class="count" x="${trackX + trackWidth + 12}" y="${y + 17}">helix/sheet/loop</text>`,
        `<rect class="anno-track" x="${trackX}" y="${y + 6}" width="${trackWidth}" height="16" rx="3"/>`,
        `<line class="ss-loop-line" x1="${trackX}" y1="${y + 14}" x2="${trackX + trackWidth}" y2="${y + 14}"/>`,
      ];
      ranges.forEach((range) => {
        if (range.kind === "loop") return;
        const start = Number(range.start || 1);
        const end = Number(range.end || start);
        const x1 = trackX + ((start - 1) / Math.max(trackLength, 1)) * trackWidth;
        const x2 = trackX + (end / Math.max(trackLength, 1)) * trackWidth;
        const color = alphaFoldColor(range.kind);
        if (range.kind === "helix") {
          parts.push(alphaFoldHelixElement(x1, x2, y + 14, escapeXml(color), "ss-helix-line", escapeXml(`helix ${start}-${end}`), null));
        } else if (range.kind === "sheet") {
          const tip = Math.max(0.01, Math.min(14, (x2 - x1) * 0.45));
          parts.push(`<polygon class="ss-sheet-arrow" points="${x1.toFixed(3)},${y + 7} ${(x2 - tip).toFixed(3)},${y + 7} ${(x2 - tip).toFixed(3)},${y + 2} ${x2.toFixed(3)},${y + 14} ${(x2 - tip).toFixed(3)},${y + 26} ${(x2 - tip).toFixed(3)},${y + 21} ${x1.toFixed(3)},${y + 21}" fill="${escapeXml(color)}"><title>${escapeXml(`sheet ${start}-${end}`)}</title></polygon>`);
        }
        const displayLabel = range.display_label || "";
        if (displayLabel) {
          const labelTitle = `${displayLabel} ${range.kind} ${start}-${end}`;
          parts.push(`<text class="ss-feature-label" x="${((x1 + x2) / 2).toFixed(3)}" y="${y + 5}" text-anchor="middle" fill="${escapeXml(color)}"><title>${escapeXml(labelTitle)}</title>${escapeXml(displayLabel)}</text>`);
        }
      });
      return parts;
    }

    function architectureComparativeExportRowHeight(scope) {
      const entries = architectureComparativeSelection(scope).entries || [];
      if (entries.length <= 1) return 30;
      return Math.max(30, Math.min(entries.length, 16) * 24 + 10);
    }

    function architectureComparativeSecondarySvgExport(trackX, trackWidth, y, trackLength, scope, rowHeight = 30) {
      const payload = comparativeAlphaFoldPayload();
      const selection = architectureComparativeSelection(scope);
      const selectedCount = selection.selectedRows.length;
      const entries = selection.entries;
      let countLabel = "select species";
      let emptyLabel = "empty until species are selected";
      const parts = [
        `<text class="label" x="${18}" y="${y + 17}">Comparative AF SS</text>`,
        `<rect class="anno-track" x="${trackX}" y="${y + 6}" width="${trackWidth}" height="${Math.max(16, rowHeight - 12)}" rx="3"/>`,
      ];
      if (selectedCount && !entries.length) {
        parts.push(`<line class="ss-loop-line" x1="${trackX}" y1="${y + 14}" x2="${trackX + trackWidth}" y2="${y + 14}"/>`);
        countLabel = payload.available ? `0/${selectedCount} ${comparativeAlphaFoldRecordLabel(0)}` : "bundle not built";
        emptyLabel = payload.available ? "selected species have no mapped AF SS" : "comparative AF SS bundle not built";
      } else if (entries.length === 1) {
        parts.push(`<line class="ss-loop-line" x1="${trackX}" y1="${y + 14}" x2="${trackX + trackWidth}" y2="${y + 14}"/>`);
        countLabel = "helix/sheet/loop";
        emptyLabel = "";
        comparativeAlphaFoldDisplayRanges(entries, trackLength).forEach((range) => {
          if (range.kind === "loop") return;
          const start = Number(range.start || 1);
          const end = Number(range.end || start);
          const x1 = trackX + ((start - 1) / Math.max(trackLength, 1)) * trackWidth;
          const x2 = trackX + (end / Math.max(trackLength, 1)) * trackWidth;
          const color = alphaFoldColor(range.kind);
          const title = range.title || `${range.kind} ${start}-${end}`;
          if (range.kind === "helix") {
            parts.push(alphaFoldHelixElement(x1, x2, y + 14, escapeXml(color), "ss-helix-line", escapeXml(title), null));
          } else if (range.kind === "sheet") {
            const tip = Math.max(0.01, Math.min(14, (x2 - x1) * 0.45));
            parts.push(`<polygon class="ss-sheet-arrow" points="${x1.toFixed(3)},${y + 7} ${(x2 - tip).toFixed(3)},${y + 7} ${(x2 - tip).toFixed(3)},${y + 2} ${x2.toFixed(3)},${y + 14} ${(x2 - tip).toFixed(3)},${y + 26} ${(x2 - tip).toFixed(3)},${y + 21} ${x1.toFixed(3)},${y + 21}" fill="${escapeXml(color)}"><title>${escapeXml(title)}</title></polygon>`);
          }
          const displayLabel = range.display_label || "";
          if (displayLabel) {
            const labelTitle = range.title || `${displayLabel} ${range.kind}; ref ${start}-${end}`;
            parts.push(`<text class="ss-feature-label" x="${((x1 + x2) / 2).toFixed(3)}" y="${y + 5}" text-anchor="middle" fill="${escapeXml(color)}"><title>${escapeXml(labelTitle)}</title>${escapeXml(displayLabel)}</text>`);
          }
        });
      } else if (entries.length > 1) {
        countLabel = "helix/sheet/loop";
        emptyLabel = "";
        const labelGutter = Math.min(170, Math.max(140, trackWidth * 0.13));
        const plotX = trackX + labelGutter;
        const plotWidth = Math.max(1, trackWidth - labelGutter);
        parts.push(`<line x1="${(plotX - 6).toFixed(3)}" y1="${y + 4}" x2="${(plotX - 6).toFixed(3)}" y2="${y + rowHeight - 4}" stroke="#cbd5e1" stroke-width="1"/>`);
        entries.slice(0, Math.min(entries.length, 16)).forEach((entry, index) => {
          const laneY = y + 15 + index * 24;
          const laneTitle = `${comparativeAlphaFoldEntryLabel(entry)}; ${comparativeAlphaFoldSourceSummary([entry])}`;
          parts.push(`<line class="ss-loop-line" x1="${plotX.toFixed(3)}" y1="${laneY}" x2="${trackX + trackWidth}" y2="${laneY}"><title>${escapeXml(laneTitle)}</title></line>`);
          comparativeAlphaFoldEntryDisplayRanges(entry, trackLength).forEach((range) => {
            if (range.kind === "loop") return;
            const start = Number(range.start || 1);
            const end = Number(range.end || start);
            const x1 = plotX + ((start - 1) / Math.max(trackLength, 1)) * plotWidth;
            const x2 = plotX + (end / Math.max(trackLength, 1)) * plotWidth;
            const color = alphaFoldColor(range.kind);
            const title = range.title || `${range.kind}; ref ${start}-${end}`;
            if (range.kind === "helix") {
              parts.push(alphaFoldHelixElement(x1, x2, laneY, escapeXml(color), "ss-helix-line", escapeXml(title), 3.2));
            } else if (range.kind === "sheet") {
              const tip = Math.max(0.01, Math.min(10, (x2 - x1) * 0.45));
              parts.push(`<polygon class="ss-sheet-arrow" points="${x1.toFixed(3)},${laneY - 5} ${(x2 - tip).toFixed(3)},${laneY - 5} ${(x2 - tip).toFixed(3)},${laneY - 9} ${x2.toFixed(3)},${laneY} ${(x2 - tip).toFixed(3)},${laneY + 9} ${(x2 - tip).toFixed(3)},${laneY + 5} ${x1.toFixed(3)},${laneY + 5}" fill="${escapeXml(color)}"><title>${escapeXml(title)}</title></polygon>`);
            }
          });
          parts.push(`<text class="ss-lane-label" x="${trackX + 4}" y="${laneY - 4}"><title>${escapeXml(laneTitle)}</title>${escapeXml(comparativeAlphaFoldShortLabel(entry))}</text>`);
          parts.push(`<text class="ss-lane-id" x="${trackX + 4}" y="${laneY + 6}">${escapeXml(comparativeAlphaFoldShortProteinId(entry))}</text>`);
        });
      }
      if (emptyLabel) {
        parts.push(`<text class="count" x="${trackX + trackWidth / 2}" y="${y + 18}" text-anchor="middle">${escapeXml(emptyLabel)}</text>`);
      }
      parts.push(`<text class="count" x="${trackX + trackWidth + 12}" y="${y + 17}">${escapeXml(countLabel)}</text>`);
      return parts;
    }

    function architectureLocalChargeSvgExport(trackX, trackWidth, y, trackLength) {
      const rows = architectureLocalChargeRows(trackLength);
      if (!rows.length) return [];
      const parts = [
        `<text class="label" x="${18}" y="${y + 17}">5-aa charge</text>`,
        `<text class="count" x="${trackX + trackWidth + 12}" y="${y + 17}">negative / neutral / positive</text>`,
        `<rect class="anno-track" x="${trackX}" y="${y + 6}" width="${trackWidth}" height="16" rx="3"/>`,
      ];
      const binWidth = Math.max(0.6, trackWidth / Math.max(trackLength, 1));
      rows.forEach((row) => {
        const x = trackX + ((row.position - 1) / Math.max(trackLength, 1)) * trackWidth;
        const title = `pos ${row.position} charge ${Number(row.charge).toFixed(1)} window ${row.window_start}-${row.window_end} ${row.window_sequence || ""}`;
        parts.push(`<rect x="${x.toFixed(3)}" y="${y + 6}" width="${binWidth.toFixed(3)}" height="16" fill="${escapeXml(row.color)}"><title>${escapeXml(title)}</title></rect>`);
      });
      parts.push(`<line x1="${trackX}" y1="${y + 14}" x2="${trackX + trackWidth}" y2="${y + 14}" stroke="#475569" stroke-width="0.7" opacity="0.35"/>`);
      return parts;
    }

    function architectureCalciumSvgExport(trackX, trackWidth, y, trackLength) {
      const calcium = architectureCalciumPayload(trackLength);
      if (!calcium.loops.length && !calcium.ligands.length) return [];
      const parts = [
        `<text class="label" x="${18}" y="${y + 17}">Ca2+ binding</text>`,
        `<text class="count" x="${trackX + trackWidth + 12}" y="${y + 17}">CBR1-3 + Ca/PC contacts</text>`,
        `<rect class="anno-track" x="${trackX}" y="${y + 6}" width="${trackWidth}" height="16" rx="3"/>`,
      ];
      calcium.loops.forEach((loop) => {
        const x = trackX + ((loop.start - 1) / Math.max(trackLength, 1)) * trackWidth;
        const w = ((loop.end - loop.start + 1) / Math.max(trackLength, 1)) * trackWidth;
        const title = `${loop.label || "CBR"} ${loop.start}-${loop.end}${loop.description ? " | " + loop.description : ""}`;
        parts.push(`<rect x="${x.toFixed(3)}" y="${y + 6}" width="${Math.max(1.5, w).toFixed(3)}" height="16" rx="4" fill="${escapeXml(loop.color)}" opacity="0.88"><title>${escapeXml(title)}</title></rect>`);
        if (w >= 24) parts.push(`<text class="anno-band-label" x="${(x + w / 2).toFixed(3)}" y="${y + 18}" text-anchor="middle">${escapeXml(loop.label || "")}</text>`);
      });
      calcium.ligands.forEach((ligand) => {
        const cx = trackX + ((Number(ligand.position) - 0.5) / Math.max(trackLength, 1)) * trackWidth;
        const sites = (ligand.sites || []).join(", ");
        const title = `${ligand.label || "ligand"} ${ligand.position}${sites ? " | " + sites : ""}`;
        parts.push(`<circle cx="${cx.toFixed(3)}" cy="${y + 14}" r="3" fill="${escapeXml(ligand.color || alphaFoldColor("calcium_ligand"))}"><title>${escapeXml(title)}</title></circle>`);
      });
      return parts;
    }

    function runRangeLabel(run) {
      if (run.startRef != null && run.endRef != null) return `${run.startRef}-${run.endRef}`;
      return `${run.startIdx + 1}-${run.endIdx + 1}`;
    }

    function cssPixels(varName, fallback) {
      const raw = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
      const value = parseFloat(raw);
      return Number.isFinite(value) ? value : fallback;
    }

    function scrollToRun(run) {
      const grid = document.getElementById("alignment-grid");
      const cellSize = cssPixels("--cell-size", 18);
      const target = Math.max(0, Number(run.startVisibleOrdinal || 0) * cellSize);
      grid.scrollTo({ left: target, behavior: "smooth" });
    }

    function exportBaseName() {
      const minRun = Math.max(1, numberOrNull("compare-min-run") || 6);
      const parts = ["alignment_browser", currentScopeKey(), currentGroupField(), document.getElementById("compare-mode").value, `min${minRun}`];
      return parts.join("_").replace(/[^a-zA-Z0-9]+/g, "_").replace(/^_+|_+$/g, "").toLowerCase();
    }

    function csvEscape(value) {
      const text = value == null ? "" : String(value);
      return /[",\\r\\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    }

    function rowsToCsv(rows, headers) {
      const lines = [headers.join(",")];
      for (const row of rows) {
        lines.push(headers.map((header) => csvEscape(row[header])).join(","));
      }
      return lines.join("\\r\\n") + "\\r\\n";
    }

    function downloadText(filename, mimeType, text) {
      downloadBlob(filename, new Blob([text], { type: mimeType }));
    }

    function exportRunRows(scope, groups) {
      const compareMode = document.getElementById("compare-mode").value;
      const minRun = Math.max(1, numberOrNull("compare-min-run") || 6);
      const scopeKey = currentScopeKey();
      const groupField = currentGroupField();
      const rows = [];
      groups.filter((group) => group.label).forEach((group, groupIndex) => {
        const runs = (group.windows && group.windows.runs) || [];
        runs.forEach((run, runIndex) => {
          rows.push({
            alignment_scope: scopeKey,
            scope_label: scope.label || "",
            group_field: groupField,
            group_label: group.label,
            group_key: group.key,
            group_order: groupIndex + 1,
            group_row_count: group.rows.length,
            compare_mode: compareMode,
            min_similar_run: minRun,
            run_number: runIndex + 1,
            start_reference_position: run.startRef,
            end_reference_position: run.endRef,
            start_alignment_position: run.startIdx + 1,
            end_alignment_position: run.endIdx + 1,
            length_alignment_columns: run.length,
            start_visible_column: Number(run.startVisibleOrdinal || 0) + 1,
            end_visible_column: Number(run.endVisibleOrdinal || 0) + 1
          });
        });
      });
      return rows;
    }

    function exportConsensusRows(scope, groups, columns) {
      const compareMode = document.getElementById("compare-mode").value;
      const minRun = Math.max(1, numberOrNull("compare-min-run") || 6);
      const scopeKey = currentScopeKey();
      const groupField = currentGroupField();
      const rows = [];
      groups.filter((group) => group.label).forEach((group, groupIndex) => {
        const consensusCells = group.consensusCells || buildConsensus(group.rows, scope, columns, group.diagnostics, offsetVisualEnabled());
        const windows = group.windows || compareWindows(consensusCells, scope, columns);
        const runByIdx = new Map();
        windows.runs.forEach((run, runIndex) => {
          for (let idx = run.startIdx; idx <= run.endIdx; idx += 1) {
            runByIdx.set(idx, { ...run, runNumber: runIndex + 1 });
          }
        });
        for (const idx of columns) {
          const cell = consensusCells.get(idx) || {};
          const support = cell.support == null ? null : Number(cell.support);
          const run = runByIdx.get(idx);
          rows.push({
            alignment_scope: scopeKey,
            scope_label: scope.label || "",
            group_field: groupField,
            group_label: group.label,
            group_key: group.key,
            group_order: groupIndex + 1,
            group_row_count: group.rows.length,
            compare_mode: compareMode,
            min_similar_run: minRun,
            alignment_position: idx + 1,
            reference_position: scope.reference_positions[idx],
            reference_residue: scope.reference_residues[idx],
            consensus_residue: cell.aa || "",
            major_residue_count: cell.count,
            support_denominator: cell.denominator,
            informative_residue_count: cell.informativeTotal,
            support_fraction: support == null ? "" : support.toFixed(6),
            support_percent: support == null ? "" : (support * 100).toFixed(2),
            used_fallback_counts: Boolean(cell.fromFallback),
            similar_to_human_run: Boolean(run),
            similar_run_number: run ? run.runNumber : "",
            similar_run_start_reference_position: run ? run.startRef : "",
            similar_run_end_reference_position: run ? run.endRef : ""
          });
        }
      });
      return rows;
    }

    function escapeXml(value) {
      return String(value == null ? "" : value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function architectureSvg(scope, groups) {
      const activeGroups = groups.filter((group) => group.label);
      const trackLength = referenceTrackLength(scope);
      const landmarkRows = architectureLandmarkRows(scope);
      const compareMode = document.getElementById("compare-mode").value;
      const labelWidth = 230;
      const trackWidth = 1180;
      const countWidth = 90;
      const margin = 18;
      const headerHeight = 76;
      const annotationRowHeight = 30;
      const rowHeight = 28;
      const width = labelWidth + trackWidth + countWidth + margin * 2;
      const secondaryRows = architectureSecondaryRanges(trackLength).length ? 1 : 0;
      const comparativeRowHeight = architectureComparativeExportRowHeight(scope);
      const chargeRows = architectureLocalChargeRows(trackLength).length ? 1 : 0;
      const calciumRows = (architectureCalciumPayload(trackLength).loops.length || architectureCalciumPayload(trackLength).ligands.length) ? 1 : 0;
      const fixedAnnotationHeight = annotationRowHeight * (1 + secondaryRows + chargeRows + calciumRows + landmarkRows.length);
      const height = headerHeight + fixedAnnotationHeight + comparativeRowHeight + rowHeight * Math.max(activeGroups.length, 1) + margin;
      const runColor = compareMode === "property" ? "#7c3aed" : "#2563eb";
      const title = `${PAYLOAD.meta && PAYLOAD.meta.gene_symbol ? PAYLOAD.meta.gene_symbol : "Alignment"} similar-run architecture`;
      const subtitle = `${formatLabel(currentGroupField())} groups, ${compareMode}, min run ${Math.max(1, numberOrNull("compare-min-run") || 6)}, ${trackLength} reference indices`;
      const trackX = margin + labelWidth;
      const countX = trackX + trackWidth + 12;
      const rulerY = headerHeight;
      const groupBaseY = headerHeight + fixedAnnotationHeight + comparativeRowHeight + 4;
      const ticks = architectureTickPositions(trackLength);
      const parts = [
        `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img">`,
        "<style>.title{font:700 18px Segoe UI,Tahoma,sans-serif;fill:#18202a}.subtitle{font:600 12px Segoe UI,Tahoma,sans-serif;fill:#617083}.label{font:700 13px Segoe UI,Tahoma,sans-serif;fill:#18202a}.count{font:700 12px Segoe UI,Tahoma,sans-serif;fill:#617083}.track{fill:#e5e7eb;stroke:#d0d7e2;stroke-width:1}.anno-track{fill:#eef2f7;stroke:#d0d7e2;stroke-width:1}.anno-tick{stroke:#64748b;stroke-width:1}.anno-tick-label{font:700 11px Segoe UI,Tahoma,sans-serif;fill:#475569}.anno-band-label{font:700 11px Segoe UI,Tahoma,sans-serif;fill:#ffffff}.anno-site-label{font:800 9px Segoe UI,Tahoma,sans-serif;paint-order:stroke;stroke:#ffffff;stroke-width:2px;stroke-linejoin:round}.ss-loop-line{stroke:#9ca3af;stroke-width:4;stroke-linecap:round;opacity:.72}.ss-helix-line{fill:none;stroke-width:4;stroke-linecap:round;stroke-linejoin:round}.ss-sheet-arrow{opacity:.9}.ss-feature-label{font:800 8px Segoe UI,Tahoma,sans-serif;paint-order:stroke;stroke:#ffffff;stroke-width:2px;stroke-linejoin:round}.ss-lane-label{font:800 8px Segoe UI,Tahoma,sans-serif;fill:#334155;paint-order:stroke;stroke:#ffffff;stroke-width:2px;stroke-linejoin:round}.ss-lane-id{font:700 7px Segoe UI,Tahoma,sans-serif;fill:#64748b;paint-order:stroke;stroke:#ffffff;stroke-width:2px;stroke-linejoin:round}</style>",
        `<rect width="${width}" height="${height}" fill="#ffffff"/>`,
        `<text class="title" x="${margin}" y="24">${escapeXml(title)}</text>`,
        `<text class="subtitle" x="${margin}" y="41">${escapeXml(subtitle)}</text>`,
        `<text class="subtitle" x="${margin}" y="58">Reference domain landmarks are derived from the gene's own UniProt/InterPro annotations (shown below).</text>`,
        `<text class="label" x="${margin}" y="${rulerY + 17}">Reference ruler</text>`,
        `<text class="count" x="${countX}" y="${rulerY + 17}">1-${escapeXml(trackLength)}</text>`,
        `<rect class="anno-track" x="${trackX}" y="${rulerY + 6}" width="${trackWidth}" height="16" rx="3"/>`
      ];
      ticks.forEach((tick) => {
        const tickX = trackX + ((tick - 1) / Math.max(trackLength, 1)) * trackWidth;
        const anchor = tick === 1 ? "start" : (tick === trackLength ? "end" : "middle");
        parts.push(`<line class="anno-tick" x1="${tickX.toFixed(3)}" y1="${rulerY + 4}" x2="${tickX.toFixed(3)}" y2="${rulerY + 26}"/>`);
        parts.push(`<text class="anno-tick-label" x="${tickX.toFixed(3)}" y="${rulerY - 4}" text-anchor="${anchor}">${escapeXml(tick)}</text>`);
      });
      if (secondaryRows) {
        parts.push(...architectureSecondarySvgExport(trackX, trackWidth, rulerY + annotationRowHeight, trackLength));
      }
      const comparativeY = rulerY + annotationRowHeight * (1 + secondaryRows);
      parts.push(...architectureComparativeSecondarySvgExport(trackX, trackWidth, comparativeY, trackLength, scope, comparativeRowHeight));
      const afterComparativeY = comparativeY + comparativeRowHeight;
      if (chargeRows) {
        parts.push(...architectureLocalChargeSvgExport(trackX, trackWidth, afterComparativeY, trackLength));
      }
      if (calciumRows) {
        parts.push(...architectureCalciumSvgExport(trackX, trackWidth, afterComparativeY + annotationRowHeight * chargeRows, trackLength));
      }
      landmarkRows.forEach((landmarkRow, index) => {
        const y = afterComparativeY + annotationRowHeight * (index + chargeRows + calciumRows);
        parts.push(`<text class="label" x="${margin}" y="${y + 17}">${escapeXml(landmarkRow.label || "")}</text>`);
        parts.push(`<text class="count" x="${countX}" y="${y + 17}">${escapeXml(landmarkRow.range_label || "")}</text>`);
        parts.push(`<rect class="anno-track" x="${trackX}" y="${y + 6}" width="${trackWidth}" height="16" rx="3"/>`);
        landmarkRow.segments.forEach((segment) => {
          const start = Number(segment.start || 1);
          const end = Number(segment.end || start);
          const range = architectureRangePercent(start, end, trackLength);
          const bandX = trackX + (range.left / 100) * trackWidth;
          const bandWidth = Math.max(1.5, (range.width / 100) * trackWidth);
          const label = `${segment.label || ""} ${segment.range_label || `${start}-${end}`}`.trim();
          parts.push(`<rect x="${bandX.toFixed(3)}" y="${y + 6}" width="${bandWidth.toFixed(3)}" height="16" rx="3" fill="${escapeXml(segment.color || "#0f766e")}" opacity="0.95"><title>${escapeXml(label)}</title></rect>`);
          if (architectureShowInlineLandmarkLabel(start, end, trackLength, bandWidth)) {
            parts.push(`<text class="anno-band-label" x="${(bandX + bandWidth / 2).toFixed(3)}" y="${y + 18}" text-anchor="middle">${escapeXml(label)}</text>`);
          } else if (segment.always_label) {
            const labelX = trackX + ((start - 0.5) / Math.max(trackLength, 1)) * trackWidth;
            parts.push(`<text class="anno-site-label" x="${labelX.toFixed(3)}" y="${y + 4}" text-anchor="middle" fill="${escapeXml(segment.color || "#0f172a")}">${escapeXml(segment.label || "")}</text>`);
          }
        });
      });
      activeGroups.forEach((group, index) => {
        const y = groupBaseY + index * rowHeight;
        const runs = (group.windows && group.windows.runs) || [];
        const countLabel = `${runs.length} run${runs.length === 1 ? "" : "s"}`;
        parts.push(`<text class="label" x="${margin}" y="${y + 17}">${escapeXml(group.label)}</text>`);
        parts.push(`<rect class="track" x="${trackX}" y="${y + 6}" width="${trackWidth}" height="15" rx="3"/>`);
        runs.forEach((run) => {
          const start = run.startRef != null ? Number(run.startRef) : run.startIdx + 1;
          const end = run.endRef != null ? Number(run.endRef) : run.endIdx + 1;
          const x = trackX + Math.max(0, ((start - 1) / Math.max(trackLength, 1)) * trackWidth);
          const rectWidth = Math.max(1.5, ((end - start + 1) / Math.max(trackLength, 1)) * trackWidth);
          const range = `${start}-${end}`;
          parts.push(`<rect x="${x.toFixed(3)}" y="${y + 6}" width="${rectWidth.toFixed(3)}" height="15" rx="2" fill="${runColor}"><title>${escapeXml(`${group.label} ${range}`)}</title></rect>`);
        });
        parts.push(`<text class="count" x="${countX}" y="${y + 17}">${escapeXml(countLabel)}</text>`);
      });
      parts.push("</svg>");
      return parts.join("");
    }

    function renderArchitecturePanel(scope, groups, columns) {
      const panel = document.getElementById("run-architecture-panel");
      const compareMode = document.getElementById("compare-mode").value;
      const activeGroups = groups.filter((group) => group.label);
      const trackLength = referenceTrackLength(scope);
      const landmarkRows = architectureLandmarkRows(scope);
      if (!architecturePanelActive() || !columns.length || !activeGroups.length || !trackLength) {
        panel.hidden = true;
        panel.innerHTML = "";
        return;
      }

      const runLookup = new Map();
      let selectedLabel = "";
      const annotationRows = [];
      const tickSegments = architectureTickPositions(trackLength).map((tick) => {
        const left = Math.max(0, ((tick - 1) / Math.max(trackLength, 1)) * 100);
        const edgeClass = tick === 1 ? " start" : (tick === trackLength ? " end" : "");
        return `<span class="architecture-ruler-tick" style="left:${left.toFixed(4)}%"></span><span class="architecture-ruler-label${edgeClass}" style="left:${left.toFixed(4)}%">${escapeHtml(tick)}</span>`;
      }).join("");
      annotationRows.push(`<div class="architecture-row annotation"><div class="architecture-label">Reference ruler</div><div class="architecture-track annotation-track">${tickSegments}</div><div class="architecture-count range">1-${escapeHtml(trackLength)}</div></div>`);
      const secondaryRowHtml = architectureSecondaryRowHtml(trackLength);
      if (secondaryRowHtml) annotationRows.push(secondaryRowHtml);
      annotationRows.push(architectureComparativeSecondaryRowHtml(trackLength, scope));
      const chargeRowHtml = architectureLocalChargeRowHtml(trackLength);
      if (chargeRowHtml) annotationRows.push(chargeRowHtml);
      const calciumRowHtml = architectureCalciumRowHtml(trackLength);
      if (calciumRowHtml) annotationRows.push(calciumRowHtml);
      landmarkRows.forEach((landmarkRow) => {
        const rangeLabel = landmarkRow.range_label || "";
        const label = landmarkRow.label || "Landmark";
        const segmentsHtml = (landmarkRow.segments || []).map((segment) => {
          const start = Number(segment.start || 1);
          const end = Number(segment.end || start);
          const range = architectureRangePercent(start, end, trackLength);
          const bandLabel = `${segment.label || ""} ${segment.range_label || `${start}-${end}`}`.trim();
          const inlineLabel = architectureShowInlineLandmarkLabel(start, end, trackLength) ? escapeHtml(bandLabel) : "";
          const siteLabel = segment.always_label
            ? `<span class="architecture-site-label" style="left:${(((start - 0.5) / Math.max(trackLength, 1)) * 100).toFixed(4)}%;color:${escapeHtml(segment.color || "#0f172a")}">${escapeHtml(segment.label || "")}</span>`
            : "";
          return `<span class="architecture-domain-band" style="left:${range.left.toFixed(4)}%;width:${range.width.toFixed(4)}%;background:${escapeHtml(segment.color || "#0f766e")}" title="${escapeHtml(bandLabel)}">${inlineLabel}</span>${siteLabel}`;
        }).join("");
        annotationRows.push(
          `<div class="architecture-row annotation">` +
          `<div class="architecture-label" title="${escapeHtml(label)}">${escapeHtml(label)}</div>` +
          `<div class="architecture-track annotation-track domain-track">` +
          `${segmentsHtml}` +
          `</div>` +
          `<div class="architecture-count range">${escapeHtml(rangeLabel)}</div>` +
          `</div>`
        );
      });
      const rowsHtml = activeGroups.map((group) => {
        const runs = (group.windows && group.windows.runs) || [];
        const segments = runs.map((run) => {
          const start = run.startRef != null ? Number(run.startRef) : run.startIdx + 1;
          const end = run.endRef != null ? Number(run.endRef) : run.endIdx + 1;
          const span = architectureRangePercent(start, end, trackLength);
          const selected = run.key && run.key === SELECTED_RUN_KEY;
          const range = runRangeLabel(run);
          if (selected) selectedLabel = `${group.label}: ${range}`;
          runLookup.set(run.key, run);
          return `<button type="button" class="architecture-segment ${compareMode === "property" ? "property" : ""}${selected ? " selected" : ""}" data-run-key="${escapeHtml(run.key)}" style="left:${span.left.toFixed(4)}%;width:${span.width.toFixed(4)}%" title="${escapeHtml(`${group.label} ${range}`)}" aria-label="${escapeHtml(`${group.label} ${range}`)}"></button>`;
        }).join("");
        const countLabel = `${runs.length} run${runs.length === 1 ? "" : "s"}`;
        return `<div class="architecture-row"><div class="architecture-label" title="${escapeHtml(group.label)}">${escapeHtml(group.label)}</div><div class="architecture-track">${segments}</div><div class="architecture-count">${escapeHtml(countLabel)}</div></div>`;
      }).join("");

      panel.hidden = false;
      panel.classList.toggle("collapsed", RUN_ARCHITECTURE_COLLAPSED);
      const totalRuns = activeGroups.reduce((total, group) => total + (((group.windows && group.windows.runs) || []).length), 0);
      const collapsedSummary = RUN_ARCHITECTURE_COLLAPSED
        ? `${totalRuns} conserved run${totalRuns === 1 ? "" : "s"} hidden to save space`
        : (selectedLabel || "Click a run to jump to it");
      const toggleLabel = RUN_ARCHITECTURE_COLLAPSED ? "Show bar" : "Hide bar";
      panel.innerHTML = `<div class="architecture-head"><div class="architecture-title"><strong>Similar conserved-runs bar across ${escapeHtml(trackLength)} reference indices</strong><span class="architecture-selected">${escapeHtml(collapsedSummary)}</span></div><div class="architecture-actions"><button type="button" class="architecture-export architecture-toggle" data-architecture-toggle aria-expanded="${RUN_ARCHITECTURE_COLLAPSED ? "false" : "true"}">${escapeHtml(toggleLabel)}</button><button type="button" class="architecture-export" data-export="runs">Export runs CSV</button><button type="button" class="architecture-export" data-export="consensus">Export consensus CSV</button><button type="button" class="architecture-export" data-export="svg">Export bar SVG</button><button type="button" class="architecture-export" data-export="png">Export bar PNG</button></div></div><div class="architecture-body"><div class="architecture-note">Homo sapiens reference ruler with domain, catalytic, and motif landmarks derived from the gene's own UniProt/InterPro annotations, plus sequence-derived 5-aa local charge.</div>${annotationRows.join("")}${rowsHtml}</div>`;
      panel.querySelector("[data-architecture-toggle]").addEventListener("click", () => {
        RUN_ARCHITECTURE_COLLAPSED = !RUN_ARCHITECTURE_COLLAPSED;
        renderArchitecturePanel(scope, groups, columns);
      });
      panel.querySelectorAll("[data-run-key]").forEach((button) => {
        button.addEventListener("click", () => {
          const run = runLookup.get(button.dataset.runKey);
          if (!run) return;
          SELECTED_RUN_KEY = run.key;
          renderArchitecturePanel(scope, groups, columns);
          scrollToRun(run);
          if (run.startRef != null && run.endRef != null) {
            alphaFoldApplySelection({
              source: "alignment_browser",
              kind: "conserved",
              label: `${run.groupLabel || "group"} conserved run`,
              start: run.startRef,
              end: run.endRef,
              color: alphaFoldColor("conserved"),
              group: run.groupLabel || run.groupKey || "group",
            }, true);
          }
        });
      });
      panel.querySelector('[data-export="runs"]').addEventListener("click", () => {
        const headers = [
          "alignment_scope", "scope_label", "group_field", "group_label", "group_key",
          "group_order", "group_row_count", "compare_mode", "min_similar_run",
          "run_number", "start_reference_position", "end_reference_position",
          "start_alignment_position", "end_alignment_position", "length_alignment_columns",
          "start_visible_column", "end_visible_column"
        ];
        downloadText(`${exportBaseName()}_runs.csv`, "text/csv;charset=utf-8", rowsToCsv(exportRunRows(scope, groups), headers));
      });
      panel.querySelector('[data-export="consensus"]').addEventListener("click", () => {
        const headers = [
          "alignment_scope", "scope_label", "group_field", "group_label", "group_key",
          "group_order", "group_row_count", "compare_mode", "min_similar_run",
          "alignment_position", "reference_position", "reference_residue", "consensus_residue",
          "major_residue_count", "support_denominator", "informative_residue_count",
          "support_fraction", "support_percent", "used_fallback_counts",
          "similar_to_human_run", "similar_run_number", "similar_run_start_reference_position",
          "similar_run_end_reference_position"
        ];
        downloadText(`${exportBaseName()}_consensus.csv`, "text/csv;charset=utf-8", rowsToCsv(exportConsensusRows(scope, groups, columns), headers));
      });
      panel.querySelector('[data-export="svg"]').addEventListener("click", () => {
        downloadText(`${exportBaseName()}_architecture.svg`, "image/svg+xml;charset=utf-8", architectureSvg(scope, groups));
      });
      panel.querySelector('[data-export="png"]').addEventListener("click", async () => {
        await downloadSvgAsPng(`${exportBaseName()}_architecture.png`, architectureSvg(scope, groups));
      });
    }

    function renderEvolutionaryDivergencePanel(scope) {
      const panel = document.getElementById("evolutionary-divergence-panel");
      if (!panel) return;
      const divergence = scope && scope.evolutionary_divergence ? scope.evolutionary_divergence : {};
      const imagePath = divergence.svg_path || divergence.png_path;
      if (!imagePath) {
        panel.hidden = true;
        panel.innerHTML = "";
        return;
      }
      const scopeLabel = scope && scope.label ? scope.label : currentScopeKey();
      const actions = [
        divergence.svg_path ? `<a href="${escapeHtml(divergence.svg_path)}">Open SVG</a>` : "",
        divergence.png_path ? `<a href="${escapeHtml(divergence.png_path)}">Open PNG</a>` : "",
      ].filter(Boolean).join("");
      panel.hidden = false;
      panel.innerHTML = `
        <div class="evolutionary-divergence-head">
          <div class="evolutionary-divergence-title">
            <strong>Evolutionary divergence across curated domains and motifs</strong>
            <span>
              Scope-aware summary for ${escapeHtml(scopeLabel)}. Dashed red lines mark the divergence cutoffs; hatched bars
              show clade means below those cutoffs. This panel follows the alignment-basis selector above.
            </span>
          </div>
          <div class="evolutionary-divergence-actions">${actions}</div>
        </div>
        <div class="evolutionary-divergence-legend">
          <span class="legend-chip"><span class="legend-swatch"></span>Clade mean bars</span>
          <span class="legend-chip"><span class="legend-line"></span>Divergence cutoff</span>
          <span class="legend-chip"><span class="legend-hatch"></span>Below cutoff</span>
        </div>
        <a class="evolutionary-divergence-figure" href="${escapeHtml(imagePath)}">
          <img loading="lazy" decoding="async" src="${escapeHtml(imagePath)}" alt="Evolutionary divergence bars for ${escapeHtml(scopeLabel)}">
        </a>
      `;
    }

    function renderNodeConservationPanel() {
      const panel = document.getElementById("node-conservation-panel");
      if (!panel) return;
      const conservation = PAYLOAD.node_conservation || {};
      // Prefer the inlined data URI (works when the HTML is shared without
      // its sibling SVG files); fall back to the on-disk relative path.
      const imageSrc = conservation.paper_tree_svg_data_uri
                       || conservation.paper_tree_svg_path
                       || "";
      const imageHref = conservation.paper_tree_svg_data_uri
                        || conservation.paper_tree_svg_path
                        || "";
      const detailedSrc = conservation.detailed_tree_svg_data_uri
                          || conservation.detailed_tree_svg_path
                          || "";
      if (!imageSrc) {
        panel.hidden = true;
        panel.innerHTML = "";
        return;
      }
      const actions = [
        imageHref ? `<a href="${escapeHtml(imageHref)}" target="_blank">Open paper SVG</a>` : "",
        detailedSrc ? `<a href="${escapeHtml(detailedSrc)}" target="_blank">Open detailed SVG</a>` : "",
        conservation.csv_path ? `<a href="${escapeHtml(conservation.csv_path)}">Open CSV</a>` : "",
      ].filter(Boolean).join("");
      panel.hidden = false;
      panel.innerHTML = `
        <div class="evolutionary-divergence-head">
          <div class="evolutionary-divergence-title">
            <strong>Node conservation paper tree</strong>
            <span>Paper-style evolutionary scan across sampled vertebrate nodes with node-level identity ranges and least/most conserved orthologues.</span>
          </div>
          <div class="evolutionary-divergence-actions">${actions}</div>
        </div>
        <a class="evolutionary-divergence-figure" href="${escapeHtml(imageHref)}" target="_blank">
          <img loading="lazy" decoding="async" src="${escapeHtml(imageSrc)}" alt="Node conservation paper tree">
        </a>
      `;
    }

    function renderAlignment() {
      const scope = currentScope();
      const columns = selectedColumns(scope);
      refreshTreeSearchMatches();
      let rows = scope.records.filter((row) => rowMatches(row, scope, columns)).sort(compareRows);
      const groups = groupedRows(rows);
      for (const group of groups) {
        group.diagnostics = group.label ? analyzeGroupOffsets(group.rows, scope, columns) : new Map();
        if (group.label && compressedActive()) {
          group.consensusCells = buildConsensus(group.rows, scope, columns, group.diagnostics, offsetVisualEnabled());
          group.windows = compareWindows(group.consensusCells, scope, columns);
          group.dissimilarWindows = compareDissimilarWindows(group.consensusCells, scope, columns);
          group.windows.runs.forEach((run, index) => {
            run.groupKey = group.key;
            run.groupLabel = group.label;
            run.key = `${currentScopeKey()}:${currentGroupField()}:${group.key}:${index}:${run.startIdx}:${run.endIdx}`;
          });
        }
      }
      ensureCompressedDefaultCollapse(groups);
      expandGroupsForTreeState(groups);
      renderSpeciesSnapshotPanel(scope, columns);
      renderMetrics(scope, rows, columns, groups);
      renderAlphaFoldStructurePanel(alphaFoldRangesFromGroups(groups));
      renderArchitecturePanel(scope, groups, columns);
      renderEvolutionaryDivergencePanel(scope);
      const grid = document.getElementById("alignment-grid");
      grid.style.setProperty("--col-count", Math.max(columns.length, 1));
      if (!columns.length) {
        grid.innerHTML = '<div class="empty">No alignment columns match the current filters.</div>';
        return;
      }
      if (!rows.length) {
        grid.innerHTML = '<div class="empty">No species match the current filters.</div>';
        return;
      }
      const ruler = `<div class="ruler"><div class="ruler-label">Species</div>${columns.map((idx) => rulerCell(scope, idx)).join("")}</div>`;
      const parts = [ruler];
      const compressed = compressedActive();
      for (const group of groups) {
        if (group.label) {
          const collapseKey = `${currentGroupField()}:${group.key}`;
          const collapsed = COLLAPSED_GROUPS.has(collapseKey);
          parts.push(renderGroupHeader(group));
          if (compressed) {
            parts.push(renderConsensusRow(group, scope, columns));
            if (collapsed) continue;
          } else if (collapsed) {
            continue;
          }
        }
        for (const row of group.rows) {
          parts.push(renderRow(row, scope, columns, group.diagnostics));
        }
      }
      grid.innerHTML = parts.join("");
      grid.querySelectorAll("[data-collapse-key]").forEach((button) => {
        button.addEventListener("click", () => {
          const key = button.dataset.collapseKey;
          if (COLLAPSED_GROUPS.has(key)) COLLAPSED_GROUPS.delete(key);
          else COLLAPSED_GROUPS.add(key);
          renderAlignment();
        });
      });
    }

    function refreshAfterScopeChange() {
      populateDynamicFilters();
      renderAlignment();
    }

    populateScopeSelect();
    applyBrowserDefaultState();
    for (const id of FILTER_IDS) {
      const node = document.getElementById(id);
      node.addEventListener("input", id === "scope-select" ? refreshAfterScopeChange : renderAlignment);
      node.addEventListener("change", id === "scope-select" ? refreshAfterScopeChange : renderAlignment);
    }
    document.getElementById("tree-view-mode").addEventListener("change", () => {
      TREE_ZOOM_MODE = "fit";
      renderTreePanel(true);
    });
    document.getElementById("tree-layout-mode").addEventListener("change", () => {
      TREE_ZOOM_MODE = "fit";
      renderTreePanel(true);
    });
    document.getElementById("tree-search").addEventListener("input", () => {
      renderTreePanel();
      renderAlignment();
    });
    document.getElementById("tree-filter-rows").addEventListener("change", renderAlignment);
    document.getElementById("tree-zoom-in").addEventListener("click", () => setTreeZoom(TREE_ZOOM * 1.2));
    document.getElementById("tree-zoom-out").addEventListener("click", () => setTreeZoom(TREE_ZOOM / 1.2));
    document.getElementById("tree-zoom-reset").addEventListener("click", () => setTreeZoom(1));
    document.getElementById("tree-zoom-fit").addEventListener("click", () => {
      TREE_ZOOM_MODE = "fit";
      fitTreeZoom();
    });
    document.getElementById("tree-clear-selection").addEventListener("click", () => {
      TREE_SELECTED_NODE_ID = "";
      TREE_SELECTED_RECORDS = new Set();
      document.getElementById("tree-filter-rows").checked = false;
      closeTreeNodePopup();
      renderTreePanel();
      renderAlignment();
    });
    document.getElementById("tree-node-popup-close").addEventListener("click", closeTreeNodePopup);
    document.getElementById("tree-node-popup-copy").addEventListener("click", copyTreeNodePopupTsv);
    document.getElementById("tree-node-popup").addEventListener("click", (event) => {
      if (event.target && event.target.id === "tree-node-popup") closeTreeNodePopup();
    });
    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeTreeNodePopup();
    });
    window.addEventListener("resize", () => {
      if (TREE_ZOOM_MODE === "fit") renderTreePanel(true);
      else applyTreeZoom();
    });
    renderNodeConservationPanel();
    renderTreePanel();
    renderAlignment();
  </script>
</body>
</html>
"""
    # Passing `outdir` lets the panel builder inline each viewer HTML via
    # iframe `srcdoc` so the alignment browser stays self-contained when
    # shared without sibling files.
    overlay_panel = v11_clade_overlay_panel_markup(
        broad_href=payload.get("v11_structure_overlay_href"),
        grouped_href=payload.get("v11_grouped_structure_overlay_href"),
        mod_href=payload.get("v11_mod_structure_overlay_href"),
        combined_href=payload.get("v11_combined_structure_overlay_href"),
        bubble_pdf_href=payload.get("v11_clade_identity_bubble_pdf_href"),
        grouped_bubble_pdf_href=payload.get("v11_grouped_clade_identity_bubble_pdf_href"),
        mod_bubble_pdf_href=payload.get("v11_mod_clade_identity_bubble_pdf_href"),
        outdir=Path(payload.get("output_directory")) if payload.get("output_directory") else None,
    )
    per_clade_ss_panel = v11_per_clade_ss_panel_markup(payload.get("v11_per_clade_ss_csv_path"))
    _umap_outdir = payload.get("output_directory")
    umap_panel = v11_umap_panel_markup(Path(_umap_outdir)) if _umap_outdir else ""
    rendered = (
        html_template
        .replace("__ALIGNMENT_BROWSER_TITLE__", title)
        .replace("__ALIGNMENT_BROWSER_PAYLOAD__", payload_json)
        .replace("__ALPHAFOLD_VIEWER_JS__", viewer_js)
        .replace("__ALPHAFOLD_STRUCTURE_CSS__", alphafold_structure_css())
        .replace("__ALPHAFOLD_STRUCTURE_PANEL__", alphafold_structure_panel_markup())
        .replace("__V11_PER_CLADE_SS_PANEL__", per_clade_ss_panel)
        .replace("__V11_CLADE_OVERLAY_PANEL__", overlay_panel)
        .replace("__V11_UMAP_PANEL__", umap_panel)
        .replace("__ALPHAFOLD_STRUCTURE_SCRIPT__", alphafold_structure_script())
    )
    # Post-process: inline any locally-referenced SVG/PNG/JPG artefacts as
    # data URIs so the alignment browser stays valid when shared without
    # its sibling files (same pass we apply in build_interactive_report_html).
    outdir_str = payload.get("output_directory")
    if outdir_str:
        try:
            rendered = _v11_inline_local_artifacts(rendered, Path(outdir_str))
        except Exception:  # noqa: BLE001
            pass
    return rendered


def render_summary_cards(run_meta: Dict[str, Any]) -> str:
    cards = [
        ("Gene", run_meta.get("gene_symbol")),
        ("Source species", run_meta.get("source_species")),
        ("Reference species", run_meta.get("reference_species")),
        ("Recovered sequences", run_meta.get("recovered_sequence_count")),
        ("Alignment method", run_meta.get("alignment_method")),
        ("Tree method", run_meta.get("tree_method") or ("not run" if not run_meta.get("tree_built") else None)),
        ("Projected length", run_meta.get("reference_projected_length")),
        ("Artifacts", run_meta.get("artifact_count")),
    ]
    parts = ['<div class="card-grid">']
    for label, value in cards:
        parts.append(
            '<div class="summary-card">'
            f'<div class="summary-label">{escape(str(label))}</div>'
            f'<div class="summary-value">{escape("" if value is None else str(value))}</div>'
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def render_link_chips(links: Sequence[Tuple[str, str]]) -> str:
    if not links:
        return "<p class=\"muted\">No quick links were available for this run.</p>"
    parts = ['<div class="chip-row">']
    for label, href in links:
        parts.append(f'<a class="chip" href="{escape(href, quote=True)}">{escape(label)}</a>')
    parts.append("</div>")
    return "".join(parts)


def render_figure_gallery(figures: Sequence[Tuple[str, str]],
                          wide: bool = False) -> str:
    if not figures:
        return "<p class=\"muted\">No figures were available for this section.</p>"
    grid_class = "figure-grid figure-grid-wide" if wide else "figure-grid"
    card_class = "figure-card figure-card-wide" if wide else "figure-card"
    parts = [f'<div class="{grid_class}">']
    for label, href in figures:
        parts.append(
            f'<figure class="{card_class}">'
            f'<figcaption>{escape(label)}</figcaption>'
            f'<a href="{escape(href, quote=True)}"><img loading="lazy" decoding="async" src="{escape(href, quote=True)}" alt="{escape(label, quote=True)}"></a>'
            "</figure>"
        )
    parts.append("</div>")
    return "".join(parts)


def render_table_section(table_key: str, title: str, description: str) -> str:
    return (
        '<section class="table-card">'
        '<div class="table-header">'
        f"<div><h3>{escape(title)}</h3><p>{escape(description)}</p></div>"
        f'<label class="table-search">Search<input type="search" data-table-search="{escape(table_key, quote=True)}" placeholder="Filter rows"></label>'
        "</div>"
        f'<div class="table-count" data-table-count="{escape(table_key, quote=True)}"></div>'
        f'<div class="table-wrap"><table data-table="{escape(table_key, quote=True)}"></table></div>'
        "</section>"
    )


def build_interactive_report_html(payload: Dict[str, Any]) -> str:
    # V11: make gene-dependent labels resolve to the actual gene symbol.
    v11_set_active_gene(str((payload.get("meta") or {}).get("gene_symbol") or ""))
    viewer_js = str(payload.get("alphafold_viewer_js") or "").replace("</script>", "<\\/script>")
    report_tables_json = json.dumps(payload["tables"], ensure_ascii=False).replace("</script>", "<\\/script>")
    table_configs_json = json.dumps(payload["table_configs"], ensure_ascii=False).replace("</script>", "<\\/script>")
    alphafold_structure_json = json.dumps(payload.get("alphafold_structure") or {}, ensure_ascii=False).replace("</script>", "<\\/script>")
    title = escape(str(payload["title"]))
    meta = payload["meta"]

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__REPORT_TITLE__</title>
  <script>__ALPHAFOLD_VIEWER_JS__</script>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4efe4;
      --panel: #fffaf3;
      --panel-alt: #efe7d6;
      --ink: #14213d;
      --muted: #5f6b7a;
      --accent: #0f766e;
      --accent-2: #b45309;
      --line: #d5c7ad;
      --shadow: 0 10px 30px rgba(20, 33, 61, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background:
        radial-gradient(circle at top right, rgba(15, 118, 110, 0.10), transparent 30%),
        linear-gradient(180deg, #fbf7f0 0%, var(--bg) 100%);
      color: var(--ink);
    }
    a { color: var(--accent); }
    .page {
      max-width: 1420px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }
    .hero {
      background: linear-gradient(135deg, rgba(20,33,61,0.96), rgba(15,118,110,0.9));
      color: #fdfbf6;
      border-radius: 24px;
      padding: 26px 28px;
      box-shadow: var(--shadow);
      margin-bottom: 20px;
    }
    .hero h1 {
      margin: 0 0 8px;
      font-size: 2rem;
      letter-spacing: 0.02em;
    }
    .hero p {
      margin: 0;
      color: rgba(253, 251, 246, 0.88);
      max-width: 920px;
      line-height: 1.5;
    }
    .card-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 16px 0 18px;
    }
    .summary-card, .note-card, .table-card, .tab-panel, .section-card {
      background: var(--panel);
      border: 1px solid rgba(213, 199, 173, 0.9);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }
    .summary-card {
      padding: 14px 16px;
      min-height: 92px;
    }
    .summary-label {
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }
    .summary-value {
      font-size: 1.15rem;
      font-weight: 700;
      line-height: 1.3;
      word-break: break-word;
    }
    .chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 16px;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 12px;
      border-radius: 999px;
      text-decoration: none;
      background: rgba(15, 118, 110, 0.10);
      border: 1px solid rgba(15, 118, 110, 0.18);
      color: var(--accent);
      font-weight: 600;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      background: var(--panel-alt);
      border: 1px solid rgba(213, 199, 173, 0.9);
      border-radius: 18px;
      padding: 14px 16px;
      margin-bottom: 20px;
    }
    .controls label {
      font-weight: 600;
      color: var(--ink);
    }
    .controls input {
      margin-left: 8px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      min-width: 180px;
    }
    .controls button {
      border: 0;
      border-radius: 10px;
      padding: 9px 12px;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      font-weight: 600;
    }
    .muted {
      color: var(--muted);
      line-height: 1.5;
    }
    .tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 14px;
    }
    .tab-button {
      border: 1px solid rgba(20, 33, 61, 0.12);
      background: rgba(255,255,255,0.72);
      color: var(--ink);
      border-radius: 12px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
    }
    .tab-button.active {
      background: var(--ink);
      color: #fff;
      border-color: var(--ink);
    }
    .tab-panel {
      display: none;
      padding: 18px;
      margin-bottom: 18px;
    }
    .tab-panel.active { display: block; }
    .panel-grid {
      display: grid;
      gap: 16px;
    }
    .figure-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
      margin-top: 12px;
    }
    .figure-grid.figure-grid-wide {
      grid-template-columns: minmax(0, 1fr);
    }
    .figure-card {
      margin: 0;
      background: #fff;
      border: 1px solid rgba(213, 199, 173, 0.9);
      border-radius: 16px;
      overflow: hidden;
      /* Large-content engine: off-screen figure cards skip layout/paint, which
         also defers decoding their (inlined data-URI) image until scrolled near.
         `auto` remembers the real height after first paint. */
      content-visibility: auto;
      contain-intrinsic-size: auto 420px;
    }
    .figure-card figcaption {
      padding: 12px 14px 0;
      font-weight: 700;
      color: var(--ink);
    }
    .figure-card img {
      width: 100%;
      display: block;
      padding: 10px 12px 14px;
      object-fit: contain;
      max-height: 360px;
    }
    .figure-card.figure-card-wide img {
      max-height: none;
    }
    .table-card {
      padding: 14px;
    }
    .table-header {
      display: flex;
      gap: 12px;
      justify-content: space-between;
      align-items: end;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }
    .table-header h3 {
      margin: 0 0 4px;
    }
    .table-header p {
      margin: 0;
      color: var(--muted);
      max-width: 720px;
      line-height: 1.45;
    }
    .table-search {
      color: var(--muted);
      font-size: 0.92rem;
      white-space: nowrap;
    }
    .table-search input {
      margin-left: 8px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      min-width: 210px;
    }
    .table-count {
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 8px;
    }
    .table-wrap {
      overflow: auto;
      border: 1px solid rgba(213, 199, 173, 0.9);
      border-radius: 14px;
      background: #fff;
      max-height: 520px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }
    thead th {
      position: sticky;
      top: 0;
      background: #f7f2e8;
      z-index: 1;
      cursor: pointer;
      text-align: left;
    }
    th, td {
      border-bottom: 1px solid #efe7d6;
      padding: 8px 10px;
      vertical-align: top;
      line-height: 1.4;
    }
    tbody tr:nth-child(odd) { background: rgba(239, 231, 214, 0.28); }
    code {
      background: rgba(20, 33, 61, 0.06);
      border-radius: 6px;
      padding: 2px 6px;
    }
    @media (max-width: 900px) {
      .hero h1 { font-size: 1.6rem; }
      .table-header { align-items: start; }
      .table-search input, .controls input { min-width: 0; width: 100%; }
    }
__ALPHAFOLD_STRUCTURE_CSS__
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>__REPORT_TITLE__</h1>
      <p>
        Offline summary-and-drilldown report for the current V9.7 run. Large alignment matrices stay file-backed,
        while the key tables, clade summaries, pairwise-vs-reference metrics, and figure links are available directly in this page.
      </p>
      __SUMMARY_CARDS__
      <div class="muted">Output directory: <code>__OUTPUT_DIRECTORY__</code></div>
    </section>

    <section class="controls">
      <label>Reference residue jump<input id="position-filter" type="number" min="1" step="1" placeholder="e.g. 228"></label>
      <button id="clear-position-filter" type="button">Clear position filter</button>
      <div class="muted">This filter is applied across conservation, domain/site, and clade tables wherever a reference position is available.</div>
    </section>

    <div class="tabs">
      <button class="tab-button active" data-tab-target="overview">Overview</button>
      <button class="tab-button" data-tab-target="alphafold">AlphaFold Structure</button>
      <button class="tab-button" data-tab-target="species">Species &amp; QC</button>
      <button class="tab-button" data-tab-target="conservation">Conservation</button>
      <button class="tab-button" data-tab-target="domains">Domains &amp; Sites</button>
      <button class="tab-button" data-tab-target="clade">Clade Analysis</button>
      <button class="tab-button" data-tab-target="v11-representatives">V11 Representatives</button>
      <button class="tab-button" data-tab-target="pairwise">Pairwise vs Human</button>
      <button class="tab-button" data-tab-target="downloads">Downloads</button>
    </div>

    <section class="tab-panel active" id="tab-overview">
      <div class="panel-grid">
        <div class="section-card" style="padding: 16px;">
          <h2>Quick Links</h2>
          __QUICK_LINKS__
        </div>
        <div class="section-card" style="padding: 16px;">
          <h2>Main Figures</h2>
          __OVERVIEW_FIGURES__
        </div>
      </div>
    </section>

    <section class="tab-panel" id="tab-alphafold">
      __ALPHAFOLD_STRUCTURE_PANEL__
    </section>

    <section class="tab-panel" id="tab-species">
      <div class="panel-grid">
        __TABLE_SPECIES_ORTHOLOGS__
        __TABLE_SPECIES_RETRIEVAL__
        __TABLE_PROTEIN_METADATA__
        __TABLE_SPECIES_LENGTH__
      </div>
    </section>

    <section class="tab-panel" id="tab-conservation">
      <div class="panel-grid">
        <div class="section-card" style="padding: 16px;">
          <h2>Conservation Figures</h2>
          __CONSERVATION_FIGURES__
        </div>
        __TABLE_CONSERVATION_SCAN__
        __TABLE_CONSERVED_REGIONS__
      </div>
    </section>

    <section class="tab-panel" id="tab-domains">
      <div class="panel-grid">
        <div class="section-card" style="padding: 16px;">
          <h2>Domain and Annotated-Site Figures</h2>
          __DOMAIN_FIGURES__
        </div>
        <div class="section-card" style="padding: 16px;">
          <h2>Evolutionary Domain and Motif Divergence</h2>
          <p class="muted">
            First-divergence labels use the earliest recovered lineage in the current tree order when a tree is available,
            and otherwise fall back to the standard clade-plus-species ordering used by the pipeline. Dashed red lines mark
            the divergence cutoffs for each metric, and hatched bars show clade means that fall below those cutoffs.
          </p>
          __EVOLUTIONARY_FIGURES__
        </div>
        __TABLE_DOMAINS__
        __TABLE_PROTEIN_FEATURES__
        __TABLE_PROTEIN_XREFS__
        __TABLE_ANNOTATED_FUNCTIONAL_SITES__
        __TABLE_ANNOTATED_SITE_CLADE_COMPARISON__
        __TABLE_SELECTED_CONSENSUS_CHUNKS__
        __TABLE_SELECTED_CONSENSUS_CHUNK_MAP__
        __TABLE_EVOLUTIONARY_SEGMENTS__
        __TABLE_EVOLUTIONARY_ALIGNMENT_WINDOWS_MANIFEST__
      </div>
    </section>

    <section class="tab-panel" id="tab-clade">
      <div class="panel-grid">
        <div class="section-card" style="padding: 16px;">
          <h2>Clade Figures</h2>
          __CLADE_FIGURES__
        </div>
        __TABLE_CLADE_IDENTITY_PROFILES__
        __TABLE_CLADE_DIFFERENCE_FROM_GLOBAL__
        __TABLE_CLADE_FOURIER_REGIONS__
        __TABLE_DOMAIN_CLADE_CONSERVATION_SUMMARY__
        __TABLE_NODE_CONSERVATION_EXTREMES__
        __TABLE_EVOLUTIONARY_SEGMENT_METRICS__
      </div>
    </section>

    <section class="tab-panel" id="tab-v11-representatives">
      <div class="panel-grid">
        <div class="section-card" style="padding: 16px;">
          <h2>V11 — Representative comparison vs human</h2>
          <p class="muted">
            V11 picks the most-conserved species per broad clade (highest mean identity to the human reference)
            and always retains <em>danio rerio</em> for experimental convenience. The tracks below compare
            net-charge (pH 7.4 model) and local aromaticity (5-residue centered window) of each representative
            against the human reference, alongside the focused secondary-structure view.
          </p>
        </div>
        <div class="section-card" style="padding: 16px;">
          <h2>V11 Figures</h2>
          __V11_REPRESENTATIVE_FIGURES__
        </div>
        __TABLE_V11_REPRESENTATIVES__
        <div class="section-card" style="padding: 16px;">
          <h2>V11 — Motif evolution &amp; lineage stabilization</h2>
          <p class="muted">
            Per-clade Shannon entropy and Jensen-Shannon divergence (Capra &amp; Singh, 2007) per reference position;
            stabilization score = H<sub>ancestral</sub> − H<sub>derived</sub>, a simplified entropy-based approximation
            of Gu's Type I functional-divergence test (Gu, MBE 1999/2006). Default ancestral bucket:
            Cyclostomata ∪ Tunicata ∪ Chondrichthyes; default derived bucket: Mammalia ∪ Aves ∪ Reptilia.
            User-supplied motifs (<code>--annotated_motifs</code>) and curated regex-library hits
            (NLS / NES / PEST / PxxP / LxxLL / leucine zipper / acidic / basic blobs) flow through the same
            per-clade analysis pipeline. No function is assumed for any motif label.
          </p>
        </div>
        <div class="section-card" style="padding: 16px;">
          <h2>Motif &amp; stabilization figures</h2>
          __V11_MOTIF_FIGURES__
        </div>
        __TABLE_V11_MOTIFS_MASTER__
        __TABLE_V11_MOTIF_EVOLUTION_PER_CLADE__
        __TABLE_V11_LINEAGE_STABILIZATION__
        <div class="section-card" style="padding: 16px;">
          <h2>V11 — Interactive 3D structure overlay</h2>
          <p class="muted">
            Per-clade identity-to-human painted onto the reference AlphaFold model (3Dmol.js). Pick a clade from the
            Overlay menu to shade secondary structure by how conserved each residue is in that clade; switch palettes,
            visual modes, surface, and (for genes with a known active site) the pocket guide. Opens in its own tab for
            full-screen use.
          </p>
          __V11_STRUCTURE_OVERLAY_LINK__
        </div>
        <div class="section-card" style="padding: 16px;">
          <h2>V11 — Region-selectable ortholog UMAP</h2>
          <p class="muted">
            Interactive embedding of every species' ortholog. Choose a domain to load its precomputed real UMAP, or
            type residues (e.g. <code>263-269</code> or <code>6-122,228</code>) to recompute a live PCA over just
            those positions across all species. Hover any dot for its species, clade, protein accession and how much
            of the region matches the human residue. (Mirrors the panel at the bottom of the standalone alignment browser.)
          </p>
          __V11_UMAP_INTERACTIVE__
        </div>
      </div>
    </section>

    <section class="tab-panel" id="tab-pairwise">
      <div class="panel-grid">
        <div class="section-card" style="padding: 16px;">
          <h2>Pairwise Report Summary</h2>
          <p class="muted">
            This table is computed from the reference-projected alignment and linked back to the existing text/PDF/PNG pairwise reports.
          </p>
        </div>
        __TABLE_PAIRWISE_REPORTS__
      </div>
    </section>

    <section class="tab-panel" id="tab-downloads">
      <div class="panel-grid">
        <div class="section-card" style="padding: 16px;">
          <h2>Artifact Manifest</h2>
          <p class="muted">
            The raw matrices are intentionally left file-backed and linked here instead of being loaded into page memory.
          </p>
        </div>
        __TABLE_DOWNLOADS__
      </div>
    </section>
  </div>

  <script id="report-tables" type="application/json">__REPORT_TABLES__</script>
  <script id="table-configs" type="application/json">__TABLE_CONFIGS__</script>
  <script id="alphafold-structure-payload" type="application/json">__ALPHAFOLD_STRUCTURE_PAYLOAD__</script>
  <script>
    const REPORT_TABLES = JSON.parse(document.getElementById("report-tables").textContent);
    const TABLE_CONFIGS = JSON.parse(document.getElementById("table-configs").textContent);
    const ALPHAFOLD_STRUCTURE = JSON.parse(document.getElementById("alphafold-structure-payload").textContent || "{}");
    const SORT_STATE = {};

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

__ALPHAFOLD_STRUCTURE_SCRIPT__

    function formatHeader(column) {
      return column.replaceAll("_", " ");
    }

    function currentPositionFilter() {
      const raw = document.getElementById("position-filter").value.trim();
      if (!raw) return null;
      const value = Number(raw);
      return Number.isFinite(value) ? value : null;
    }

    function cellToSearchText(value) {
      if (Array.isArray(value)) return value.map(cellToSearchText).join(" ");
      if (value && typeof value === "object") return Object.values(value).map(cellToSearchText).join(" ");
      return value == null ? "" : String(value);
    }

    function rowMatchesSearch(row, query) {
      if (!query) return true;
      const haystack = Object.values(row).map(cellToSearchText).join(" ").toLowerCase();
      return haystack.includes(query);
    }

    function rowMatchesPosition(row, config, position) {
      if (position == null) return true;
      if (config.positionField) {
        const value = Number(row[config.positionField]);
        return Number.isFinite(value) && value === position;
      }
      if (config.rangeStartField && config.rangeEndField) {
        const start = Number(row[config.rangeStartField]);
        const end = Number(row[config.rangeEndField]);
        return Number.isFinite(start) && Number.isFinite(end) && start <= position && position <= end;
      }
      return true;
    }

    function compareValues(a, b) {
      if (a == null && b == null) return 0;
      if (a == null) return 1;
      if (b == null) return -1;
      if (typeof a === "number" && typeof b === "number") return a - b;
      return String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: "base" });
    }

    function renderCellValue(column, value, config) {
      const linkColumns = config.linkColumns || [];
      if (linkColumns.includes(column)) {
        const values = Array.isArray(value) ? value : (value ? [value] : []);
        if (!values.length) return "";
        return values.map((href) => {
          const label = String(href).split("/").pop();
          return `<a href="${escapeHtml(href)}">${escapeHtml(label)}</a>`;
        }).join("<br>");
      }
      if (Array.isArray(value)) {
        return value.map((item) => escapeHtml(item)).join("<br>");
      }
      return value == null ? "" : escapeHtml(value);
    }

    function renderTable(tableKey) {
      const table = document.querySelector(`table[data-table="${tableKey}"]`);
      if (!table) return;
      const config = TABLE_CONFIGS[tableKey] || {};
      const position = currentPositionFilter();
      const searchInput = document.querySelector(`[data-table-search="${tableKey}"]`);
      const query = searchInput ? searchInput.value.trim().toLowerCase() : "";
      const sourceRows = Array.isArray(REPORT_TABLES[tableKey]) ? REPORT_TABLES[tableKey].slice() : [];
      const columns = (config.columns || []).filter((column) => sourceRows.some((row) => Object.prototype.hasOwnProperty.call(row, column)));
      const sortState = SORT_STATE[tableKey] || {};

      let rows = sourceRows.filter((row) => rowMatchesPosition(row, config, position)).filter((row) => rowMatchesSearch(row, query));
      if (sortState.column) {
        rows.sort((left, right) => {
          const result = compareValues(left[sortState.column], right[sortState.column]);
          return sortState.direction === "desc" ? -result : result;
        });
      }

      const countEl = document.querySelector(`[data-table-count="${tableKey}"]`);
      if (countEl) {
        countEl.textContent = `${rows.length} row${rows.length === 1 ? "" : "s"}`;
      }

      if (!columns.length) {
        table.innerHTML = "<tbody><tr><td>No data available.</td></tr></tbody>";
        return;
      }

      const headerHtml = columns.map((column) => {
        const arrow = sortState.column === column ? (sortState.direction === "desc" ? " ↓" : " ↑") : "";
        return `<th data-sort-column="${escapeHtml(column)}">${escapeHtml(formatHeader(column))}${arrow}</th>`;
      }).join("");

      const bodyHtml = rows.length ? rows.map((row) => {
        const cells = columns.map((column) => `<td>${renderCellValue(column, row[column], config)}</td>`).join("");
        return `<tr>${cells}</tr>`;
      }).join("") : `<tr><td colspan="${columns.length}">No rows match the current filters.</td></tr>`;

      table.innerHTML = `<thead><tr>${headerHtml}</tr></thead><tbody>${bodyHtml}</tbody>`;
      table.querySelectorAll("th[data-sort-column]").forEach((header) => {
        header.addEventListener("click", () => {
          const column = header.dataset.sortColumn;
          const current = SORT_STATE[tableKey] || {};
          let direction = "asc";
          if (current.column === column && current.direction === "asc") {
            direction = "desc";
          }
          SORT_STATE[tableKey] = { column, direction };
          renderTable(tableKey);
        });
      });
    }

    function renderAllTables() {
      Object.keys(REPORT_TABLES).forEach(renderTable);
    }

    document.querySelectorAll(".tab-button").forEach((button) => {
      button.addEventListener("click", () => {
        const target = button.dataset.tabTarget;
        document.querySelectorAll(".tab-button").forEach((node) => node.classList.toggle("active", node === button));
        document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${target}`));
        if (target === "alphafold") renderAlphaFoldStructurePanel();
      });
    });

    document.querySelectorAll("[data-table-search]").forEach((input) => {
      input.addEventListener("input", () => renderTable(input.dataset.tableSearch));
    });

    document.getElementById("position-filter").addEventListener("input", renderAllTables);
    document.getElementById("clear-position-filter").addEventListener("click", () => {
      document.getElementById("position-filter").value = "";
      renderAllTables();
    });

    renderAlphaFoldStructurePanel();
    renderAllTables();
  </script>
</body>
</html>
"""

    replacements = {
        "__REPORT_TITLE__": title,
        "__SUMMARY_CARDS__": render_summary_cards(meta),
        "__OUTPUT_DIRECTORY__": escape(str(meta.get("output_directory") or "")),
        "__QUICK_LINKS__": render_link_chips(payload["quick_links"]),
        "__OVERVIEW_FIGURES__": render_figure_gallery(payload["overview_figures"]),
        "__CONSERVATION_FIGURES__": render_figure_gallery(payload["conservation_figures"]),
        "__DOMAIN_FIGURES__": render_figure_gallery(payload["domain_figures"]),
        "__EVOLUTIONARY_FIGURES__": render_figure_gallery(payload["evolutionary_figures"], wide=True),
        "__CLADE_FIGURES__": render_figure_gallery(payload["clade_figures"]),
        "__TABLE_SPECIES_ORTHOLOGS__": render_table_section("orthologs", "Ortholog table", "Recovered orthologue rows from Ensembl for the current run."),
        "__TABLE_SPECIES_RETRIEVAL__": render_table_section("sequence_retrieval", "Sequence retrieval", "Per-species protein retrieval status and metadata."),
        "__TABLE_PROTEIN_METADATA__": render_table_section("protein_metadata", "Protein metadata", "Stable per-protein metadata and AlphaFold-ready identifiers for future comparative work."),
        "__TABLE_SPECIES_LENGTH__": render_table_section("length_filter", "Length QC", "Reference-relative length filtering summary."),
        "__TABLE_CONSERVATION_SCAN__": render_table_section("conservation_scan", "Reference-mapped conservation scan", "Per-reference-position exact and property conservation scores."),
        "__TABLE_CONSERVED_REGIONS__": render_table_section("conserved_regions", "Conserved regions", "Contiguous windows that exceed the configured conservation thresholds."),
        "__TABLE_DOMAINS__": render_table_section("domains", "Domain annotations", "UniProt / InterPro-derived domain and feature annotations."),
        "__TABLE_PROTEIN_FEATURES__": render_table_section("protein_features", "Protein features", "Normalized per-feature intervals collected for the recovered proteins."),
        "__TABLE_PROTEIN_XREFS__": render_table_section("protein_xrefs", "Protein cross-references", "External identifiers and cross-references retained for downstream evolutionary interpretation."),
        "__TABLE_ANNOTATED_FUNCTIONAL_SITES__": render_table_section("annotated_functional_sites", "Annotated functional sites", "Named residues mapped against conservation scores."),
        "__TABLE_ANNOTATED_SITE_CLADE_COMPARISON__": render_table_section("annotated_site_clade_comparison", "Annotated-site clade comparison", "Reference motifs and clade-specific residue usage around annotated sites."),
        "__TABLE_SELECTED_CONSENSUS_CHUNKS__": render_table_section("selected_consensus_chunks", "Selected consensus chunks", "Optional chunk ranges prepared for human AlphaFold overlay workflows."),
        "__TABLE_SELECTED_CONSENSUS_CHUNK_MAP__": render_table_section("selected_consensus_chunks_structure_map", "Chunk-to-structure map", "Reference-position chunk ranges mapped onto the human AlphaFold model residue numbering."),
        "__TABLE_EVOLUTIONARY_SEGMENTS__": render_table_section("evolutionary_segments", "Evolutionary first-divergence summary", "Human domains, regions, motif windows, and clade windows annotated with the first clade and species that fall below the identity, similarity, and polarity rules."),
        "__TABLE_EVOLUTIONARY_ALIGNMENT_WINDOWS_MANIFEST__": render_table_section("evolutionary_alignment_windows_manifest", "Alignment-window excerpts", "Downloadable text excerpts for curated human features and clade-divergent windows, ordered by the current evolutionary/tree order."),
        "__TABLE_CLADE_IDENTITY_PROFILES__": render_table_section("clade_identity_profiles", "Clade identity profiles", "Reference-position identity to the human/reference sequence across clades."),
        "__TABLE_CLADE_DIFFERENCE_FROM_GLOBAL__": render_table_section("clade_difference_from_global", "Clade-vs-global differences", "Per-position departures from the global identity profile."),
        "__TABLE_CLADE_FOURIER_REGIONS__": render_table_section("clade_fourier_regions", "Clade Fourier regions", "FFT-smoothed conserved and divergent clade windows."),
        "__TABLE_DOMAIN_CLADE_CONSERVATION_SUMMARY__": render_table_section("domain_clade_conservation_summary", "Domain-by-clade summary", "Domain-level conservation statistics split by clade."),
        "__TABLE_NODE_CONSERVATION_EXTREMES__": render_table_section("node_conservation_extremes", "Node conservation extremes", f"Most and least human-reference-conserved {v11_gene_label()} orthologues per named vertebrate node; human query excluded."),
        "__TABLE_EVOLUTIONARY_SEGMENT_METRICS__": render_table_section("evolutionary_segment_metrics", "Per-clade segment metrics", "Mean identity, BLOSUM-positive similarity, and polarity agreement for each human segment across the clade bins used for the divergence bar figure."),
        "__TABLE_PAIRWISE_REPORTS__": render_table_section("pairwise_reports", "Pairwise vs reference", "Searchable summary of the pairwise human/reference report bundle."),
        "__TABLE_DOWNLOADS__": render_table_section("downloads", "Artifact manifest", "Direct links to raw CSV/TSV/FASTA/tree/PDF/SVG/PNG outputs."),
        "__V11_REPRESENTATIVE_FIGURES__": render_figure_gallery(payload.get("v11_representative_figures") or [], wide=True),
        "__TABLE_V11_REPRESENTATIVES__": render_table_section("v11_representatives", "V11 clade representatives", "Most-conserved species per broad clade (highest mean identity to human reference). Mandatory inclusions: homo_sapiens, danio_rerio."),
        "__V11_MOTIF_FIGURES__": render_figure_gallery(payload.get("v11_motif_figures") or [], wide=True),
        "__TABLE_V11_MOTIFS_MASTER__": render_table_section("v11_motifs_master", "V11 motifs (user + library)", "All motif ranges investigated by V11: user-supplied via --annotated_motifs (source=user) and curated regex-library hits in the human reference (source=library). No function is assumed for any label."),
        "__TABLE_V11_MOTIF_EVOLUTION_PER_CLADE__": render_table_section("v11_motif_evolution_per_clade", "V11 motif evolution by clade", "For each motif × broad-clade pair: consensus motif inside that clade, fraction of clade members matching the human reference motif, and dominant alternative."),
        "__TABLE_V11_LINEAGE_STABILIZATION__": render_table_section("v11_lineage_stabilization", "V11 lineage stabilization (top 200 |score|)", "Stabilization score = H_ancestral − H_derived per reference position. Positive = position is more variable in ancestral clades and fixed in derived clades (Gu-style Type I divergence approximation)."),
        "__V11_STRUCTURE_OVERLAY_LINK__": (
            (
                # Broad-clade viewer (13 vertebrate clades). Inline the full
                # overlay HTML into the iframe via srcdoc so the report
                # stays self-contained when shared without sibling files.
                _v11_overlay_iframe_markup(
                    payload.get("v11_structure_overlay_href"),
                    payload.get("output_directory"),
                    "V11 structure overlay (broad clades)",
                    "Open interactive 3D structure overlay &mdash; broad clades &#8599;",
                )
                +
                # Subdivided 9-group viewer (Primates/Rodents/OtherMammals split, etc.).
                _v11_overlay_iframe_markup(
                    payload.get("v11_grouped_structure_overlay_href"),
                    payload.get("output_directory"),
                    "V11 structure overlay (9 groups)",
                    "Open interactive 3D structure overlay &mdash; subdivided 9-group (Primates / Rodents / OtherMammals split) &#8599;",
                    margin_top_px=24,
                )
            )
            if (payload.get("v11_structure_overlay_href") or payload.get("v11_grouped_structure_overlay_href"))
            else '<p class="muted">Structure overlay not available (requires the human reference AlphaFold model).</p>'
        ),
        "__V11_UMAP_INTERACTIVE__": (
            v11_umap_panel_markup(Path(payload["output_directory"]))
            if payload.get("output_directory")
            else '<p class="muted">Region-selectable UMAP not available (region JSON missing).</p>'
        ),
        "__REPORT_TABLES__": report_tables_json,
        "__TABLE_CONFIGS__": table_configs_json,
        "__ALPHAFOLD_VIEWER_JS__": viewer_js,
        "__ALPHAFOLD_STRUCTURE_CSS__": alphafold_structure_css(),
        "__ALPHAFOLD_STRUCTURE_PANEL__": alphafold_structure_panel_markup(),
        "__ALPHAFOLD_STRUCTURE_SCRIPT__": alphafold_structure_script(),
        "__ALPHAFOLD_STRUCTURE_PAYLOAD__": alphafold_structure_json,
    }
    for token, value in replacements.items():
        html_template = html_template.replace(token, value)
    # Post-process: inline locally-referenced binary artefacts (SVG / PNG /
    # JPG) as data URIs so the rendered HTML stays valid when shared on its
    # own without the sibling output-dir files. Keeps the "Open ..." href
    # links pointing at the inlined data URI too so they open in a new tab
    # without breaking. Only runs when the payload carries `output_directory`.
    outdir_str = payload.get("output_directory")
    if outdir_str:
        try:
            html_template = _v11_inline_local_artifacts(html_template, Path(outdir_str))
        except Exception:  # noqa: BLE001
            pass
    return html_template


_V11_INLINE_ARTIFACT_REGEX = re.compile(
    r'\b(src|href)="([^"#?][^"#?]*?\.(svg|png|jpe?g|pdf))"',
    re.IGNORECASE,
)
_V11_INLINE_ARTIFACT_MIME = {
    "svg": "image/svg+xml",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "pdf": "application/pdf",
}
# Per-extension size caps (MB). PDFs get a higher cap because the typical
# bubble-grid PDFs (~0.5-1.2 MB) are the main artefacts users want to keep
# embedded in a shareable report. SVG/PNG cap is tighter so giant composite
# figures (full-alignment architecture SVGs ~5 MB) stay as relative links
# rather than ballooning the HTML 10×.
_V11_INLINE_ARTIFACT_CAP_MB = {
    "svg": 0.5,
    "png": 0.5,
    "jpg": 0.5,
    "jpeg": 0.5,
    "pdf": 1.5,
}


def _v11_inline_local_artifacts(html: str, outdir: Path) -> str:
    """Walk every src=/href= attribute that points to a relative
    .svg/.png/.jpg/.pdf file in `outdir`, read the file, and replace the
    attribute with a base64 data URI. Per-extension size caps live in
    `_V11_INLINE_ARTIFACT_CAP_MB` (SVG/PNG 0.5 MB, PDF 1.5 MB). Skips data
    URIs, absolute URLs, and oversized files so the report stays send-able
    without ballooning.

    The same data URI is substituted into BOTH `src` and `href` references
    pointing at the same path on subsequent matches via a cache, so an
    `<a href="paper_tree.svg"><img src="paper_tree.svg"></a>` pair both get
    the inline URI in one pass — including the case of a PDF link clicked
    via `<a href="v11_clade_identity_bubble_9group.pdf">`.
    """
    cache: Dict[str, Optional[str]] = {}

    def repl(match):
        attr = match.group(1)
        path_rel = match.group(2)
        ext = match.group(3).lower()
        if path_rel in cache:
            uri = cache[path_rel]
        else:
            target = outdir / path_rel
            if not target.is_file():
                cache[path_rel] = None
                return match.group(0)
            cap_mb = _V11_INLINE_ARTIFACT_CAP_MB.get(ext, 0.5)
            if target.stat().st_size > int(cap_mb * 1024 * 1024):
                cache[path_rel] = None
                return match.group(0)
            try:
                data = target.read_bytes()
            except OSError:
                cache[path_rel] = None
                return match.group(0)
            mime = _V11_INLINE_ARTIFACT_MIME.get(ext, "application/octet-stream")
            uri = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
            cache[path_rel] = uri
        if uri is None:
            return match.group(0)
        # Chromium-based browsers (Chrome 60+, Edge 79+) BLOCK top-frame
        # navigation to data: URLs as a phishing/XSS mitigation. So a chip
        # like <a href="data:application/pdf;..." target="_blank"> would
        # silently do nothing when clicked, even though the URI is valid.
        # Adding the `download="<filename>"` attribute changes the browser's
        # behaviour to a file save instead of a navigation, which Chromium
        # explicitly allows. We do this only for PDF + href so other refs
        # (img src, etc.) aren't affected.
        if ext == "pdf" and attr.lower() == "href":
            filename = Path(path_rel).name
            return f'{attr}="{uri}" download="{escape(filename)}"'
        return f'{attr}="{uri}"'

    return _V11_INLINE_ARTIFACT_REGEX.sub(repl, html)


def quote_ident(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def normalize_sqlite_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(clean_json_value(value), ensure_ascii=False)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, bool):
        return int(value)
    return value


def prepare_dataframe_for_sqlite(df: pd.DataFrame, run_key: Optional[str] = None) -> pd.DataFrame:
    work = df.copy()
    if run_key is not None and "run_key" not in work.columns:
        work.insert(0, "run_key", run_key)
    for column in work.columns:
        work[column] = work[column].map(normalize_sqlite_value)
    return work


def infer_sqlite_type(values: Iterable[Any]) -> str:
    seen: List[Any] = [value for value in values if value is not None]
    if not seen:
        return "TEXT"
    if all(isinstance(value, bool) for value in seen):
        return "INTEGER"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in seen):
        return "INTEGER"
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in seen):
        return "REAL" if any(isinstance(value, float) for value in seen) else "INTEGER"
    return "TEXT"


def ensure_sqlite_table(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
    columns = list(df.columns)
    if not columns:
        return
    existing = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    if not existing:
        definitions: List[str] = []
        for column in columns:
            if table_name == "runs" and column == "run_key":
                definitions.append(f'{quote_ident(column)} TEXT PRIMARY KEY')
            else:
                definitions.append(f"{quote_ident(column)} {infer_sqlite_type(df[column].tolist())}")
        conn.execute(f"CREATE TABLE {quote_ident(table_name)} ({', '.join(definitions)})")
        if table_name != "runs" and "run_key" in columns:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {quote_ident(f'idx_{table_name}_run_key')} "
                f"ON {quote_ident(table_name)} ({quote_ident('run_key')})"
            )
        return

    existing_cols = {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({quote_ident(table_name)})").fetchall()
    }
    for column in columns:
        if column in existing_cols:
            continue
        conn.execute(
            f"ALTER TABLE {quote_ident(table_name)} ADD COLUMN "
            f"{quote_ident(column)} {infer_sqlite_type(df[column].tolist())}"
        )


def table_has_run_key(conn: sqlite3.Connection, table_name: str) -> bool:
    for row in conn.execute(f"PRAGMA table_info({quote_ident(table_name)})").fetchall():
        if str(row[1]) == "run_key":
            return True
    return False


def insert_dataframe(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    columns = list(df.columns)
    placeholders = ", ".join("?" for _ in columns)
    sql = (
        f"INSERT INTO {quote_ident(table_name)} "
        f"({', '.join(quote_ident(column) for column in columns)}) VALUES ({placeholders})"
    )
    rows = [
        tuple(row[column] for column in columns)
        for row in df.to_dict(orient="records")
    ]
    conn.executemany(sql, rows)


def ensure_index_if_columns_exist(conn: sqlite3.Connection,
                                  table_name: str,
                                  index_name: str,
                                  columns: Sequence[str]) -> None:
    existing_cols = {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({quote_ident(table_name)})").fetchall()
    }
    if not set(columns).issubset(existing_cols):
        return
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {quote_ident(index_name)} "
        f"ON {quote_ident(table_name)} ({', '.join(quote_ident(column) for column in columns)})"
    )


def ensure_alignment_indexes(conn: sqlite3.Connection) -> None:
    index_specs = [
        ("alignment_species", "idx_alignment_species_run_scope", ["run_key", "alignment_scope"]),
        ("alignment_species", "idx_alignment_species_run_scope_species", ["run_key", "alignment_scope", "species"]),
        ("alignment_species", "idx_alignment_species_run_scope_clade", ["run_key", "alignment_scope", "clade"]),
        ("alignment_species", "idx_alignment_species_run_scope_taxonomy", ["run_key", "alignment_scope", "taxonomy_level"]),
        ("alignment_species", "idx_alignment_species_run_scope_tree", ["run_key", "alignment_scope", "tree_order"]),
        ("alignment_cells", "idx_alignment_cells_run_scope", ["run_key", "alignment_scope"]),
        ("alignment_cells", "idx_alignment_cells_run_scope_species", ["run_key", "alignment_scope", "species"]),
        ("alignment_cells", "idx_alignment_cells_run_scope_clade", ["run_key", "alignment_scope", "clade"]),
        ("alignment_cells", "idx_alignment_cells_run_scope_taxonomy", ["run_key", "alignment_scope", "taxonomy_level"]),
        ("alignment_cells", "idx_alignment_cells_run_scope_alignment_pos", ["run_key", "alignment_scope", "alignment_position"]),
        ("alignment_cells", "idx_alignment_cells_run_scope_reference_pos", ["run_key", "alignment_scope", "reference_position"]),
        ("alignment_cells", "idx_alignment_cells_run_scope_tree", ["run_key", "alignment_scope", "tree_order"]),
    ]
    existing_tables = {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    for table_name, index_name, columns in index_specs:
        if table_name in existing_tables:
            ensure_index_if_columns_exist(conn, table_name, index_name, columns)


def write_sqlite_archive(db_path: Path,
                         run_key: str,
                         tables: Dict[str, pd.DataFrame]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        existing_tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        for table_name in existing_tables:
            if table_has_run_key(conn, table_name):
                conn.execute(f"DELETE FROM {quote_ident(table_name)} WHERE run_key = ?", (run_key,))

        for table_name, df in tables.items():
            ensure_sqlite_table(conn, table_name, df)
            insert_dataframe(conn, table_name, df)
        ensure_alignment_indexes(conn)


def export_output_archive(outdir: Path | str,
                          db_path: Path | str = DEFAULT_SQLITE_ARCHIVE_PATH,
                          report_filename: str = INTERACTIVE_REPORT_FILENAME) -> Dict[str, Any]:
    outdir_path = Path(outdir).resolve()
    if not outdir_path.exists():
        raise FileNotFoundError(f"Output directory does not exist: {outdir_path}")

    loaded_tables = {
        table_name: load_output_table(outdir_path, filename, sep)
        for table_name, (filename, sep) in CSV_TABLE_SPECS.items()
    }
    msa_sequences_df = collect_msa_sequences(outdir_path)
    reference_species = infer_reference_species(loaded_tables, msa_sequences_df)
    pairwise_df = build_pairwise_report_rows(outdir_path, reference_species=reference_species)
    run_key = normalize_run_key(outdir_path)
    sqlite_path = Path(db_path).resolve()

    alignment_species_df, alignment_cells_df = build_alignment_archive_tables(
        outdir=outdir_path,
        tables=loaded_tables,
        reference_species=reference_species,
    )
    tree_viewer_data = build_tree_viewer_data(outdir_path, alignment_species_df)
    node_conservation_df = write_node_conservation_extremes_artifacts(
        outdir_path,
        alignment_species_df,
        reference_species=reference_species,
    )
    loaded_tables["node_conservation_extremes"] = node_conservation_df
    _archive_gene_symbol = parse_run_summary_metadata(outdir_path / "run_summary.txt").get("Gene")
    # V11: make any figure/HTML rebuilt here label itself with the real gene.
    v11_set_active_gene(_archive_gene_symbol)
    preliminary_meta = {
        "run_key": run_key,
        "output_directory": str(outdir_path),
        "output_name": outdir_path.name,
        "gene_symbol": _archive_gene_symbol,
        "reference_species": reference_species,
    }
    alignment_browser_path = outdir_path / ALIGNMENT_BROWSER_FILENAME
    alignment_browser_path.write_text(
        build_alignment_browser_html(build_alignment_browser_payload(alignment_species_df, preliminary_meta, tree_viewer_data, outdir=outdir_path)),
        encoding="utf-8",
    )

    artifacts_df = collect_artifacts(outdir_path)
    run_meta = build_run_metadata(
        outdir=outdir_path,
        run_key=run_key,
        db_path=sqlite_path,
        report_filename=report_filename,
        tables=loaded_tables,
        artifacts_df=artifacts_df,
        pairwise_df=pairwise_df,
        reference_species=reference_species,
    )
    alignment_browser_path.write_text(
        build_alignment_browser_html(build_alignment_browser_payload(alignment_species_df, run_meta, tree_viewer_data, outdir=outdir_path)),
        encoding="utf-8",
    )
    artifacts_df = collect_artifacts(outdir_path)
    run_meta = build_run_metadata(
        outdir=outdir_path,
        run_key=run_key,
        db_path=sqlite_path,
        report_filename=report_filename,
        tables=loaded_tables,
        artifacts_df=artifacts_df,
        pairwise_df=pairwise_df,
        reference_species=reference_species,
    )
    write_alignment_browser_default_exports(outdir_path, alignment_species_df, run_meta)
    artifacts_df = collect_artifacts(outdir_path)
    run_meta = build_run_metadata(
        outdir=outdir_path,
        run_key=run_key,
        db_path=sqlite_path,
        report_filename=report_filename,
        tables=loaded_tables,
        artifacts_df=artifacts_df,
        pairwise_df=pairwise_df,
        reference_species=reference_species,
    )
    alignment_browser_path.write_text(
        build_alignment_browser_html(build_alignment_browser_payload(alignment_species_df, run_meta, tree_viewer_data, outdir=outdir_path)),
        encoding="utf-8",
    )
    artifacts_df = collect_artifacts(outdir_path)
    run_meta = build_run_metadata(
        outdir=outdir_path,
        run_key=run_key,
        db_path=sqlite_path,
        report_filename=report_filename,
        tables=loaded_tables,
        artifacts_df=artifacts_df,
        pairwise_df=pairwise_df,
        reference_species=reference_species,
    )

    sqlite_tables = {
        "runs": prepare_dataframe_for_sqlite(pd.DataFrame([run_meta])),
        "artifacts": prepare_dataframe_for_sqlite(artifacts_df, run_key=run_key),
        "orthologs": prepare_dataframe_for_sqlite(loaded_tables["orthologs"], run_key=run_key),
        "sequence_retrieval": prepare_dataframe_for_sqlite(loaded_tables["sequence_retrieval"], run_key=run_key),
        "protein_metadata": prepare_dataframe_for_sqlite(loaded_tables["protein_metadata"], run_key=run_key),
        "protein_features": prepare_dataframe_for_sqlite(loaded_tables["protein_features"], run_key=run_key),
        "protein_xrefs": prepare_dataframe_for_sqlite(loaded_tables["protein_xrefs"], run_key=run_key),
        "length_filter": prepare_dataframe_for_sqlite(loaded_tables["length_filter"], run_key=run_key),
        "msa_sequences": prepare_dataframe_for_sqlite(msa_sequences_df, run_key=run_key),
        "conservation_scan": prepare_dataframe_for_sqlite(loaded_tables["conservation_scan"], run_key=run_key),
        "conservation_per_position": prepare_dataframe_for_sqlite(loaded_tables["conservation_per_position"], run_key=run_key),
        "annotated_functional_sites": prepare_dataframe_for_sqlite(loaded_tables["annotated_functional_sites"], run_key=run_key),
        "annotated_site_clade_comparison": prepare_dataframe_for_sqlite(loaded_tables["annotated_site_clade_comparison"], run_key=run_key),
        "domains": prepare_dataframe_for_sqlite(loaded_tables["domains"], run_key=run_key),
        "selected_consensus_chunks": prepare_dataframe_for_sqlite(loaded_tables["selected_consensus_chunks"], run_key=run_key),
        "selected_consensus_chunks_structure_map": prepare_dataframe_for_sqlite(loaded_tables["selected_consensus_chunks_structure_map"], run_key=run_key),
        "conserved_regions": prepare_dataframe_for_sqlite(loaded_tables["conserved_regions"], run_key=run_key),
        "domain_clade_conservation_summary": prepare_dataframe_for_sqlite(loaded_tables["domain_clade_conservation_summary"], run_key=run_key),
        "node_conservation_extremes": prepare_dataframe_for_sqlite(loaded_tables["node_conservation_extremes"], run_key=run_key),
        "evolutionary_segments": prepare_dataframe_for_sqlite(loaded_tables["evolutionary_segments"], run_key=run_key),
        "evolutionary_segment_metrics": prepare_dataframe_for_sqlite(loaded_tables["evolutionary_segment_metrics"], run_key=run_key),
        "evolutionary_alignment_windows_manifest": prepare_dataframe_for_sqlite(loaded_tables["evolutionary_alignment_windows_manifest"], run_key=run_key),
        "clade_identity_profiles": prepare_dataframe_for_sqlite(normalize_clade_identity_profiles(loaded_tables["clade_identity_profiles_wide"]), run_key=run_key),
        "clade_fourier_lowpass_profiles": prepare_dataframe_for_sqlite(normalize_clade_lowpass_profiles(loaded_tables["clade_fourier_lowpass_profiles_wide"]), run_key=run_key),
        "clade_fourier_spectrum": prepare_dataframe_for_sqlite(loaded_tables["clade_fourier_spectrum"], run_key=run_key),
        "clade_difference_from_global": prepare_dataframe_for_sqlite(normalize_clade_difference_profiles(loaded_tables["clade_difference_from_global_wide"]), run_key=run_key),
        "clade_fourier_regions": prepare_dataframe_for_sqlite(loaded_tables["clade_fourier_regions"], run_key=run_key),
        "pairwise_reports": prepare_dataframe_for_sqlite(pairwise_df, run_key=run_key),
        "alignment_species": prepare_dataframe_for_sqlite(alignment_species_df, run_key=run_key),
        "alignment_cells": prepare_dataframe_for_sqlite(alignment_cells_df, run_key=run_key),
    }
    write_sqlite_archive(sqlite_path, run_key=run_key, tables=sqlite_tables)

    payload = build_report_payload(
        outdir=outdir_path,
        tables=loaded_tables,
        artifacts_df=artifacts_df,
        pairwise_df=pairwise_df,
        run_meta=run_meta,
    )
    report_path = outdir_path / report_filename
    report_path.write_text(build_interactive_report_html(payload), encoding="utf-8")

    return {
        "run_key": run_key,
        "database_path": sqlite_path,
        "html_path": report_path,
        "alignment_browser_path": alignment_browser_path,
        "artifact_count": len(artifacts_df),
        "pairwise_report_count": len(pairwise_df),
    }


# =============================================================================
# V11 additions: per-species net-charge & aromaticity tracks, clade-
# representative selection, focused comparative views vs. human reference.
# =============================================================================

# Residue → net charge at physiological pH 7.4. Histidine partially protonated
# (pKa ~6.0 ⇒ ~10% protonated at pH 7.4). Cysteine ignored (pKa ~8.3, only
# partially deprotonated, usually treated as neutral for net-charge tracks).
V11_RESIDUE_NET_CHARGE_PH74: Dict[str, float] = {
    "K": 1.0, "R": 1.0, "H": 0.1,
    "D": -1.0, "E": -1.0,
}

# Aromatic residue indicator (F, W, Y). Histidine has aromatic character but
# is canonically excluded from "aromatic" property tracks in protein analyses.
V11_RESIDUE_AROMATICITY: Dict[str, float] = {"F": 1.0, "W": 1.0, "Y": 1.0}

# Default sliding-window width for the smoothed property tracks (residues).
V11_DEFAULT_PROPERTY_WINDOW = 5

# Species that V11 always retains in the focused comparative view even if
# they would not otherwise be picked as the most-conserved representative.
V11_MANDATORY_FOCUS_SPECIES: Tuple[str, ...] = ("homo_sapiens", "danio_rerio")

V11_DEFAULT_REPRESENTATIVE_CSV = "v11_clade_representatives.tsv"
V11_NET_CHARGE_PER_SPECIES_CSV = "v11_net_charge_per_species.csv"
V11_AROMATICITY_PER_SPECIES_CSV = "v11_aromaticity_per_species.csv"
V11_NET_CHARGE_DELTA_CSV = "v11_net_charge_delta_vs_human.csv"
V11_AROMATICITY_DELTA_CSV = "v11_aromaticity_delta_vs_human.csv"
V11_REPRESENTATIVE_PROPERTY_HEATMAP_PNG = "v11_representative_property_heatmap.png"
V11_REPRESENTATIVE_PROPERTY_HEATMAP_SVG = "v11_representative_property_heatmap.svg"
V11_REPRESENTATIVE_PROPERTY_TRACES_PNG = "v11_representative_property_traces.png"
V11_REPRESENTATIVE_PROPERTY_TRACES_SVG = "v11_representative_property_traces.svg"
V11_REPRESENTATIVE_SS_BUNDLE_JSON = "v11_representative_secondary_structure.json"


def _v11_residue_value(aa: str, table: Dict[str, float]) -> float:
    """Return the property value of a single residue. Gaps/unknowns → 0."""
    if not aa:
        return 0.0
    return float(table.get(str(aa).upper(), 0.0))


def _v11_centered_window_mean(values: Sequence[float], window: int) -> List[float]:
    """Centered sliding-window mean. window must be >= 1; we use mean of the
    [i - window//2 .. i + window//2] slice clipped to array bounds."""
    if window <= 1 or not values:
        return [float(v) for v in values]
    arr = np.asarray(values, dtype=float)
    half = window // 2
    cumulative = np.cumsum(np.insert(arr, 0, 0.0))
    out = np.empty_like(arr)
    n = len(arr)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = (cumulative[hi] - cumulative[lo]) / max(1, hi - lo)
    return out.tolist()


def v11_compute_per_species_property_track(alignment: Any,
                                           residue_property_map: Dict[str, float],
                                           reference_species: str,
                                           smoothing_window: int = V11_DEFAULT_PROPERTY_WINDOW) -> pd.DataFrame:
    """For a reference-projected alignment, return one row per species with
    the smoothed property value at every reference (ungapped) column.

    Output frame columns:
        species, protein_id, plus one column per reference position labelled
        "pos_<n>" (1-based, matching reference_ungapped_position).
    """
    if alignment is None or len(alignment) == 0:
        return pd.DataFrame()
    target = (reference_species or "").strip().lower()
    ref_record = None
    for record in alignment:
        species, _ = parse_header_species_symbol(record.id)
        if species.lower() == target:
            ref_record = record
            break
    if ref_record is None:
        ref_record = alignment[0]

    # Determine reference (ungapped) positions of each alignment column.
    ref_seq = str(ref_record.seq).upper()
    aln_to_ref: List[Optional[int]] = []
    ref_residues: List[str] = []
    ref_counter = 0
    for aa in ref_seq:
        if aa in GAP_CHARS:
            aln_to_ref.append(None)
        else:
            ref_counter += 1
            aln_to_ref.append(ref_counter)
            ref_residues.append(aa)

    # Columns to keep = alignment columns that correspond to a ref position.
    keep_cols = [idx for idx, ref_pos in enumerate(aln_to_ref) if ref_pos is not None]

    rows: List[Dict[str, Any]] = []
    for record in alignment:
        species, symbol = parse_header_species_symbol(record.id)
        seq_str = str(record.seq).upper()
        raw = [
            _v11_residue_value(seq_str[idx], residue_property_map) if idx < len(seq_str) else 0.0
            for idx in keep_cols
        ]
        smoothed = _v11_centered_window_mean(raw, smoothing_window)
        row: Dict[str, Any] = {
            "species": species,
            "protein_label": symbol,
            "record_id": record.id,
        }
        for ref_idx, value in enumerate(smoothed, start=1):
            row[f"pos_{ref_idx}"] = float(value)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Move ref species to top.
    df["_is_ref"] = (df["species"].str.lower() == target).astype(int)
    df = df.sort_values(["_is_ref", "species"], ascending=[False, True]).drop(columns=["_is_ref"]).reset_index(drop=True)
    return df


def v11_compute_identity_to_reference(alignment: Any, reference_species: str) -> pd.DataFrame:
    """Return per-species mean identity to the reference across overlapping
    (both non-gap) columns. Frame columns: species, protein_label, record_id,
    overlapping_positions, identical_positions, mean_identity."""
    if alignment is None or len(alignment) == 0:
        return pd.DataFrame()
    target = (reference_species or "").strip().lower()
    ref_record = None
    for record in alignment:
        species, _ = parse_header_species_symbol(record.id)
        if species.lower() == target:
            ref_record = record
            break
    if ref_record is None:
        ref_record = alignment[0]
    ref_seq = str(ref_record.seq).upper()

    rows: List[Dict[str, Any]] = []
    for record in alignment:
        species, symbol = parse_header_species_symbol(record.id)
        seq = str(record.seq).upper()
        overlap = 0
        identical = 0
        for r, s in zip(ref_seq, seq):
            if r in GAP_CHARS or s in GAP_CHARS:
                continue
            overlap += 1
            if r == s:
                identical += 1
        mean_id = float(identical) / overlap if overlap else 0.0
        rows.append({
            "species": species,
            "protein_label": symbol,
            "record_id": record.id,
            "overlapping_positions": overlap,
            "identical_positions": identical,
            "mean_identity": mean_id,
        })
    return pd.DataFrame(rows)


def _v11_resolve_taxonomy_level(taxonomy_lookup: Any, species: str) -> str:
    """Accept either the flat Dict[str, str] from build_species_taxonomy_lookup
    or the nested Dict[str, Dict] form, and return the species' taxonomy_level
    string (empty if unknown)."""
    if not taxonomy_lookup or not species:
        return ""
    key = str(species)
    raw = taxonomy_lookup.get(key)
    if raw is None:
        raw = taxonomy_lookup.get(key.lower())
    if isinstance(raw, dict):
        return str(raw.get("taxonomy_level") or raw.get("clade") or raw.get("phylum") or "").strip()
    if isinstance(raw, str):
        return raw.strip()
    return ""


# MRCA-with-human values seen in Ensembl ortholog "taxonomy_level" → V11
# clade bucket. Note: Ensembl reports MRCA, *not* the species' own lineage,
# so e.g. "Eutheria" means "any non-primate placental mammal" (the MRCA with
# human is Eutheria), and "Euteleostomi" covers every bony fish. We use MRCA
# only when it pins the clade uniquely; otherwise we fall through to a
# species-name pattern check below.
_V11_MRCA_BUCKETS: Dict[str, str] = {
    # Primate-line MRCAs ⇒ Mammalia.
    "catarrhini": "Mammalia", "simiiformes": "Mammalia", "hominoidea": "Mammalia",
    "hominidae": "Mammalia", "homininae": "Mammalia", "haplorrhini": "Mammalia",
    "primates": "Mammalia", "euarchontoglires": "Mammalia",
    # Generic mammal-line MRCAs.
    "eutheria": "Mammalia", "boreoeutheria": "Mammalia", "theria": "Mammalia",
    "mammalia": "Mammalia", "metatheria": "Mammalia", "marsupialia": "Mammalia",
    "monotremata": "Mammalia", "laurasiatheria": "Mammalia", "afrotheria": "Mammalia",
    "xenarthra": "Mammalia",
    # Reptile/bird (Amniota MRCA covers both).
    "amniota": "Reptilia",  # default; gallus_gallus etc. overridden by species.
    "sauropsida": "Reptilia", "squamata": "Reptilia", "testudines": "Reptilia",
    "crocodylia": "Reptilia", "lepidosauria": "Reptilia",
    "aves": "Aves", "neognathae": "Aves", "palaeognathae": "Aves",
    # Amphibian-line MRCA.
    "tetrapoda": "Amphibia",
    "amphibia": "Amphibia", "anura": "Amphibia", "caudata": "Amphibia",
    # Sarcopterygii (non-tetrapod): coelacanth or lungfish — disambiguate via name.
    "sarcopterygii": "Coelacanthiformes",
    "coelacanthiformes": "Coelacanthiformes", "actinistia": "Coelacanthiformes",
    "dipnoi": "Dipnoi", "ceratodontiformes": "Dipnoi", "lepidosireniformes": "Dipnoi",
    # Euteleostomi covers ALL bony fish (and tetrapods). Default fish, but we
    # still need name-based disambiguation for holostei/polypterids/etc.
    "euteleostomi": "Teleostei",
    "actinopterygii": "Teleostei", "neopterygii": "Teleostei",
    "teleostei": "Teleostei", "clupeocephala": "Teleostei",
    "holostei": "Holostei", "semionotiformes": "Holostei",
    "chondrostei": "Chondrostei", "acipenseriformes": "Chondrostei",
    "polypteriformes": "Polypteriformes", "cladistia": "Polypteriformes",
    # Cartilaginous fish.
    "gnathostomata": "Chondrichthyes",  # default; bony fish overridden by species/MRCA above.
    "chondrichthyes": "Chondrichthyes", "elasmobranchii": "Chondrichthyes",
    "holocephali": "Chondrichthyes",
    # Jawless fish.
    "cyclostomata": "Cyclostomata", "hyperoartia": "Cyclostomata",
    "petromyzontidae": "Cyclostomata", "hyperotreti": "Cyclostomata",
    "vertebrata": "Cyclostomata",  # default for unannotated cyclostomes.
    # Non-vertebrate chordates.
    "chordata": "Tunicata",  # default for unannotated ciona-likes.
    "tunicata": "Tunicata", "ascidiacea": "Tunicata",
    "cephalochordata": "Cephalochordata", "branchiostoma": "Cephalochordata",
    # Non-chordates (Opisthokonta = animal/fungi MRCA → yeast in our runs).
    "opisthokonta": "Fungi",
    "fungi": "Fungi", "ascomycota": "Fungi", "saccharomycotina": "Fungi",
    "saccharomycetes": "Fungi", "saccharomycetales": "Fungi", "dikarya": "Fungi",
    "arthropoda": "Arthropoda", "insecta": "Arthropoda",
    "nematoda": "Nematoda", "mollusca": "Mollusca",
    "echinodermata": "Echinodermata", "cnidaria": "Cnidaria",
}

# Species-name patterns: each (substring tested against species name with
# word-boundary discipline, bucket). Used to disambiguate cases where the
# MRCA-based map can't (e.g. all bony fish reporting "Euteleostomi").
# Pattern must occur as a whole _-delimited token, NOT a substring.
_V11_SPECIES_NAME_PATTERNS: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    # Bird genera (some have Amniota MRCA, need bird override).
    (("gallus", "meleagris", "coturnix", "anas", "anser", "ficedula", "geospiza",
      "parus", "passer", "serinus", "taeniopygia", "aquila", "strigops",
      "struthio", "pygocentrus", "stegastes", "amphiprion"  # latter two: fish, see below
      ), "Aves"),
    # Reptile genera.
    (("anolis", "podarcis", "salvator", "sphenodon", "chrysemys", "pelodiscus",
      "crocodylus", "gopherus", "chelonoidis", "terrapene", "laticauda",
      "notechis", "pseudonaja", "naja"), "Reptilia"),
    # Amphibian genera.
    (("xenopus", "leptobrachium"), "Amphibia"),
    # Coelacanth.
    (("latimeria",), "Coelacanthiformes"),
    # Lungfish.
    (("neoceratodus", "protopterus", "lepidosiren"), "Dipnoi"),
    # Cartilaginous fish.
    (("callorhinchus", "scyliorhinus", "carcharodon", "raja"), "Chondrichthyes"),
    # Jawless fish.
    (("petromyzon", "eptatretus", "lampetra"), "Cyclostomata"),
    # Holostean (gars and bowfin only — actual genera, not "vulgaris").
    (("lepisosteus", "atractosteus"), "Holostei"),
    (("amia",), "Holostei"),
    # Polypterids.
    (("polypterus", "erpetoichthys"), "Polypteriformes"),
    # Chondrosteans.
    (("acipenser", "scaphirhynchus", "polyodon"), "Chondrostei"),
    # Tunicates / cephalochordates.
    (("ciona", "halocynthia", "molgula", "oikopleura"), "Tunicata"),
    (("branchiostoma",), "Cephalochordata"),
    # Fungi.
    (("saccharomyces", "candida", "aspergillus", "neurospora"), "Fungi"),
)

# Override fish species incorrectly captured by the bird Aves heuristic above.
# (Aves keywords like "stegastes"/"amphiprion"/"pygocentrus" are fish genera —
# explicit teleost listing here outranks the Aves block.)
_V11_TELEOST_GENUS_TOKENS: Tuple[str, ...] = (
    "danio", "oryzias", "gasterosteus", "takifugu", "tetraodon", "xiphophorus",
    "salmo", "oncorhynchus", "salvelinus", "poecilia", "oreochromis", "astyanax",
    "ictalurus", "gadus", "fundulus", "amphilophus", "seriola", "betta",
    "carassius", "cyprinus", "labrus", "lates", "larimichthys", "hippocampus",
    "sander", "nothobranchius", "scleropages", "denticeps", "kryptolebias",
    "pundamilia", "neolamprologus", "stegastes", "amphiprion", "pygocentrus",
    "paramormyrops", "amphiprion", "haplochromis", "astatotilapia", "sparus",
    "cyclopterus", "anabas", "myripristis", "scophthalmus", "cottoperca",
    "maylandia", "esox", "hucho", "electrophorus", "clupea", "salmo_trutta",
    "salmo_salar", "salmo_trutta", "amphiprion_ocellaris", "amphiprion_percula",
    "acanthochromis", "cynoglossus", "mastacembelus", "sparus_aurata",
    "dicentrarchus", "labrus_bergylta", "nothobranchius_furzeri",
    "cyprinodon", "oryzias_javanicus", "oryzias_melastigma", "oryzias_sinensis",
    "lates_calcarifer", "hippocampus_comes", "sander_lucioperca",
    "takifugu_rubripes", "tetraodon_nigroviridis",
)


def _v11_species_tokens(species: str) -> List[str]:
    """Split species name on underscores into lowercase tokens for word-
    boundary keyword matching."""
    return [tok for tok in str(species or "").strip().lower().split("_") if tok]


# Mammal genus tokens. Covers human (homo) and the rest of the placental/
# marsupial mammal set the pipeline routinely sees so Ensembl's MRCA-style
# taxonomy_level (which collapses many mammals into "Eutheria" or the query
# species' own name) doesn't leak human/chimp into a "tetrapods" fallback.
_V11_MAMMAL_GENUS_TOKENS: Tuple[str, ...] = (
    "homo", "pan", "gorilla", "pongo", "nomascus", "hylobates",
    "macaca", "papio", "mandrillus", "cercocebus", "chlorocebus", "rhinopithecus",
    "saimiri", "aotus", "cebus", "callithrix", "carlito", "tarsius", "otolemur",
    "microcebus", "prolemur", "propithecus", "lemur", "indri",
    "mus", "rattus", "peromyscus", "cricetulus", "mesocricetus", "microtus",
    "jaculus", "dipodomys", "octodon", "chinchilla", "cavia", "ictidomys",
    "urocitellus", "marmota", "sciurus", "spermophilus", "nannospalax",
    "heterocephalus", "fukomys", "ochotona", "oryctolagus", "lepus",
    "canis", "felis", "panthera", "ursus", "ailuropoda", "mustela", "vulpes",
    "neovison", "lynx", "leptonychotes", "neomonachus", "odobenus",
    "bos", "bison", "ovis", "capra", "sus", "catagonus", "camelus", "vicugna",
    "cervus", "moschus", "rangifer",
    "tursiops", "delphinapterus", "phocoena", "balaenoptera", "physeter",
    "monodon", "orcinus",
    "equus",
    "pteropus", "myotis", "rhinolophus", "rousettus", "miniopterus",
    "erinaceus", "sorex", "condylura",
    "dasypus", "choloepus", "trichechus", "elephas", "loxodonta", "procavia",
    "echinops", "orycteropus", "chrysochloris",
    "tupaia", "galeopterus",
    # Marsupials / monotremes.
    "monodelphis", "phascolarctos", "sarcophilus", "vombatus", "notamacropus",
    "macropus", "thylacinus", "ornithorhynchus", "tachyglossus",
)


def v11_resolve_broad_clade(species: str, taxonomy_level: Optional[str]) -> str:
    """Return a coarse clade label suitable for the V11 representative
    bucketing. Combines species-name token matching (authoritative when it
    fires) with the Ensembl MRCA taxonomy_level (which is used as a fallback
    default since MRCA aliasing groups many distinct clades under one label,
    e.g. "Euteleostomi" covers every bony fish)."""
    tokens = _v11_species_tokens(species)
    token_set = set(tokens)

    # 1. Mammal genus tokens — cover human (whose MRCA == species name) and
    #    the rest of the placental/marsupial set Ensembl ortho-tags via the
    #    overly broad "Eutheria" / "Catarrhini" / "Theria" MRCA labels.
    if token_set & set(_V11_MAMMAL_GENUS_TOKENS):
        return "Mammalia"

    # 2. Teleost genus tokens win next (fixes Aves/teleost name ambiguity).
    if token_set & set(_V11_TELEOST_GENUS_TOKENS):
        return "Teleostei"

    # 3. Per-pattern species-name matches with whole-token boundary.
    for keywords, bucket in _V11_SPECIES_NAME_PATTERNS:
        if token_set & set(keywords):
            return bucket

    # 4. MRCA-based fallback.
    tax = str(taxonomy_level or "").strip().lower()
    if tax in _V11_MRCA_BUCKETS:
        return _V11_MRCA_BUCKETS[tax]
    # Token containment (handles multi-word MRCAs we did not list literally).
    for keyword, bucket in _V11_MRCA_BUCKETS.items():
        if keyword in tax:
            return bucket

    # 5. Last-resort fallback (best effort using V9.8 helpers).
    phylum = classify_alignment_phylum(species, taxonomy_level)
    if phylum and phylum != "Chordata":
        return phylum
    inferred = classify_alignment_clade(species, taxonomy_level)
    return inferred or "Unassigned"


def v11_select_clade_representatives(alignment: Any,
                                     reference_species: str,
                                     taxonomy_lookup: Optional[Dict[str, Any]] = None,
                                     always_include: Sequence[str] = V11_MANDATORY_FOCUS_SPECIES) -> pd.DataFrame:
    """Pick the species with the highest mean identity-to-reference inside
    each broad clade. Always include `always_include` (default: human +
    zebrafish) regardless of clade ranking.

    Returns a frame with columns: clade, species, protein_label, record_id,
    mean_identity, overlapping_positions, identical_positions, is_reference,
    is_mandatory, selection_rank.
    """
    identity_df = v11_compute_identity_to_reference(alignment, reference_species)
    if identity_df.empty:
        return identity_df

    # Resolve broad clade per species using the same classifiers as the rest
    # of the pipeline. Robust to either taxonomy_lookup shape.
    identity_df["taxonomy_level"] = identity_df["species"].apply(
        lambda s: _v11_resolve_taxonomy_level(taxonomy_lookup, s)
    )
    identity_df["clade"] = identity_df.apply(
        lambda r: v11_resolve_broad_clade(r["species"], r["taxonomy_level"]) or "Unassigned",
        axis=1,
    )

    target = (reference_species or "").strip().lower()
    mandatory = {str(s).strip().lower() for s in (always_include or ())} | {target}
    identity_df["is_reference"] = (identity_df["species"].str.lower() == target).astype(bool)
    identity_df["is_mandatory"] = identity_df["species"].str.lower().isin(mandatory)

    # Per-clade winner (max identity, tie-broken by overlap then species).
    identity_df = identity_df.sort_values(
        ["clade", "mean_identity", "overlapping_positions", "species"],
        ascending=[True, False, False, True],
    )
    picked = identity_df.groupby("clade", as_index=False).head(1).copy()
    picked["selection_rank"] = "clade_top"

    # Force-add mandatory species not already in the picked set.
    chosen_keys = set(picked["species"].str.lower().tolist())
    extras = identity_df[
        identity_df["species"].str.lower().isin(mandatory) & ~identity_df["species"].str.lower().isin(chosen_keys)
    ].drop_duplicates(subset=["species"]).copy()
    extras["selection_rank"] = extras["is_reference"].map({True: "reference", False: "mandatory"})
    picked = pd.concat([picked, extras], ignore_index=True)

    # Order: reference first, then mandatory non-reference, then by clade name.
    picked["_order"] = picked.apply(
        lambda r: (0 if r["is_reference"] else (1 if r["selection_rank"] == "mandatory" else 2),
                   str(r["clade"]).lower()),
        axis=1,
    )
    picked = picked.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return picked[[
        "clade", "species", "protein_label", "record_id",
        "taxonomy_level",
        "mean_identity", "overlapping_positions", "identical_positions",
        "is_reference", "is_mandatory", "selection_rank",
    ]]


def _v11_delta_vs_reference(per_species_df: pd.DataFrame, reference_species: str) -> pd.DataFrame:
    """Subtract the reference row from every other row, column-wise."""
    if per_species_df.empty:
        return per_species_df
    target = (reference_species or "").strip().lower()
    ref_rows = per_species_df[per_species_df["species"].str.lower() == target]
    if ref_rows.empty:
        return per_species_df.copy()
    pos_cols = [c for c in per_species_df.columns if c.startswith("pos_")]
    ref_vals = ref_rows.iloc[0][pos_cols].astype(float).to_numpy()
    # Vectorized: subtract the reference row across all rows in one shot.
    meta = per_species_df[["species", "protein_label", "record_id"]].reset_index(drop=True)
    values = per_species_df[pos_cols].astype(float).to_numpy() - ref_vals[None, :]
    delta = pd.DataFrame(values, columns=pos_cols)
    return pd.concat([meta, delta], axis=1)


def _v11_plot_property_heatmap(delta_df: pd.DataFrame,
                               representatives_df: pd.DataFrame,
                               property_label: str,
                               *,
                               cmap_name: str,
                               vmin: float,
                               vmax: float,
                               outpath_png: Path,
                               outpath_svg: Path,
                               reference_species: str) -> None:
    """Heatmap of delta-from-human across reference positions for the focused
    set of representative species."""
    if delta_df.empty or representatives_df.empty:
        return
    # Filter by the exact record_id of each rep so we don't accidentally
    # include sibling paralogs from the same species (e.g. danio_rerio has
    # two PLA2G4A paralogs in the alignment but the reps table only picks
    # one).
    keep_records = set(representatives_df["record_id"].astype(str))
    focus = delta_df[delta_df["record_id"].astype(str).isin(keep_records)].copy()
    if focus.empty:
        # Fallback: filter by species in case record_id columns don't line up.
        keep_species = representatives_df["species"].str.lower().tolist()
        focus = delta_df[delta_df["species"].str.lower().isin(keep_species)].copy()
    if focus.empty:
        return
    # Sort focus rows in representative order using record_id.
    rep_order = list(representatives_df["record_id"].astype(str))
    focus["_order"] = focus["record_id"].astype(str).apply(
        lambda r: rep_order.index(r) if r in rep_order else 1e9
    )
    focus = focus.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    pos_cols = [c for c in focus.columns if c.startswith("pos_")]
    matrix = focus[pos_cols].to_numpy(dtype=float)
    species_labels = focus["species"].tolist()
    clade_lookup = dict(zip(representatives_df["species"], representatives_df["clade"]))
    y_labels = [
        f"{sp} [{clade_lookup.get(sp, 'Unassigned')}]"
        + (" (ref)" if sp.lower() == reference_species.lower() else "")
        for sp in species_labels
    ]

    width = max(10.0, len(pos_cols) * 0.012)
    height = max(2.5, 0.42 * len(species_labels) + 1.6)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap_name,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(species_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    n_ticks = min(20, max(2, len(pos_cols) // 50))
    tick_step = max(1, len(pos_cols) // n_ticks)
    ax.set_xticks(range(0, len(pos_cols), tick_step))
    ax.set_xticklabels([str(i + 1) for i in range(0, len(pos_cols), tick_step)], rotation=90, fontsize=7)
    ax.set_xlabel(f"Reference ({reference_species}) ungapped position")
    ax.set_title(f"{property_label} delta vs {reference_species} (5-residue smoothed)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cbar.set_label(f"Δ{property_label}")

    # The reference row is Δ-vs-itself = 0 across the board, which renders as a
    # blank band and reads like missing data. Make it obviously intentional:
    # overlay a hatched grey band + centred caption on the reference row(s).
    ref_lc = reference_species.lower()
    n_cols = len(pos_cols)
    for row_idx, sp in enumerate(species_labels):
        if sp.lower() != ref_lc:
            continue
        ax.add_patch(Rectangle(
            (-0.5, row_idx - 0.5), n_cols, 1.0,
            facecolor="#e5e7eb", edgecolor="#9ca3af", linewidth=0.6,
            hatch="////", zorder=3,
        ))
        ax.text(
            n_cols / 2.0, row_idx,
            f"{reference_species} — reference baseline (Δ = 0 by definition)",
            ha="center", va="center", fontsize=8, fontstyle="italic",
            color="#374151", zorder=4,
        )
        # Thin divider just below the reference row to set it apart.
        ax.axhline(row_idx + 0.5, color="#6b7280", linewidth=0.8, zorder=4)

    fig.tight_layout()
    fig.savefig(outpath_png, dpi=180)
    fig.savefig(outpath_svg)
    plt.close(fig)


def _v11_plot_property_traces(charge_df: pd.DataFrame,
                              aromaticity_df: pd.DataFrame,
                              representatives_df: pd.DataFrame,
                              reference_species: str,
                              outpath_png: Path,
                              outpath_svg: Path) -> None:
    """Faceted line plots: one ROW per representative, two columns
    (net-charge | aromaticity), each showing the species track (solid)
    overlaid on the human reference track (dashed) along the reference
    ungapped axis. Row-major layout keeps each panel wide and readable
    for long proteins (PLA2G4A: 749 aa)."""
    if representatives_df.empty or charge_df.empty or aromaticity_df.empty:
        return
    # Match by record_id so paralogs of the same species don't sneak in.
    rep_records = list(representatives_df["record_id"].astype(str))
    species_order = list(representatives_df["species"])
    ref = reference_species.lower()
    ref_charge_row = charge_df[charge_df["species"].str.lower() == ref]
    ref_arom_row = aromaticity_df[aromaticity_df["species"].str.lower() == ref]
    if ref_charge_row.empty or ref_arom_row.empty:
        return
    pos_cols_c = [c for c in charge_df.columns if c.startswith("pos_")]
    pos_cols_a = [c for c in aromaticity_df.columns if c.startswith("pos_")]
    x_c = np.arange(1, len(pos_cols_c) + 1)
    x_a = np.arange(1, len(pos_cols_a) + 1)
    ref_c = ref_charge_row.iloc[0][pos_cols_c].to_numpy(dtype=float)
    ref_a = ref_arom_row.iloc[0][pos_cols_a].to_numpy(dtype=float)

    nspec = len(species_order)
    fig, axes = plt.subplots(
        nrows=nspec,
        ncols=2,
        figsize=(16.0, max(2.2 * nspec, 6.0)),
        sharex="col",
        squeeze=False,
    )
    clade_lookup = dict(zip(representatives_df["record_id"].astype(str), representatives_df["clade"]))
    species_lookup = dict(zip(representatives_df["record_id"].astype(str), representatives_df["species"]))
    for row_idx, rec_id in enumerate(rep_records):
        species = species_lookup.get(rec_id, "")
        spc_charge = charge_df[charge_df["record_id"].astype(str) == rec_id]
        spc_arom = aromaticity_df[aromaticity_df["record_id"].astype(str) == rec_id]
        ax_c = axes[row_idx][0]
        ax_a = axes[row_idx][1]
        clade = clade_lookup.get(rec_id, "Unassigned")
        is_ref = (species.lower() == ref)
        clade_color = _v11_clade_color(clade)
        row_label = f"{species}\n[{clade}]" + ("  (ref)" if is_ref else "")
        ax_c.set_ylabel(row_label, fontsize=9, rotation=0, ha="right", va="center", labelpad=70,
                         color=clade_color)
        # Reference baseline (dashed grey)
        ax_c.plot(x_c, ref_c, color="#aaa", lw=0.8, ls="--", label=reference_species)
        ax_a.plot(x_a, ref_a, color="#aaa", lw=0.8, ls="--", label=reference_species)
        if not spc_charge.empty:
            ax_c.fill_between(x_c, ref_c, spc_charge.iloc[0][pos_cols_c].to_numpy(dtype=float),
                              color=clade_color, alpha=0.35, linewidth=0)
            ax_c.plot(x_c, spc_charge.iloc[0][pos_cols_c].to_numpy(dtype=float),
                      color=clade_color, lw=0.9)
        if not spc_arom.empty:
            ax_a.fill_between(x_a, ref_a, spc_arom.iloc[0][pos_cols_a].to_numpy(dtype=float),
                              color=clade_color, alpha=0.35, linewidth=0)
            ax_a.plot(x_a, spc_arom.iloc[0][pos_cols_a].to_numpy(dtype=float),
                      color=clade_color, lw=0.9)
        ax_c.axhline(0.0, color="#ddd", lw=0.5)
        ax_a.axhline(0.0, color="#ddd", lw=0.5)
        ax_c.set_xlim(1, max(2, len(pos_cols_c)))
        ax_a.set_xlim(1, max(2, len(pos_cols_a)))
        ax_c.set_ylim(-1.05, 1.05)
        ax_a.set_ylim(-0.05, 1.05)
        for ax_ in (ax_c, ax_a):
            ax_.tick_params(axis="both", labelsize=7)
            for spine in ("top", "right"):
                ax_.spines[spine].set_visible(False)
        if row_idx == 0:
            ax_c.set_title("Net charge (pH 7.4, 5aa smoothed)", fontsize=10)
            ax_a.set_title("Aromaticity fraction (5aa smoothed)", fontsize=10)
        if row_idx == nspec - 1:
            ax_c.set_xlabel(f"Reference ({reference_species}) ungapped position")
            ax_a.set_xlabel(f"Reference ({reference_species}) ungapped position")
    fig.suptitle(f"Representative property tracks vs {reference_species} (filled area = Δ to reference)",
                 fontsize=12, y=0.998)
    fig.tight_layout()
    fig.savefig(outpath_png, dpi=170)
    fig.savefig(outpath_svg)
    plt.close(fig)


def v11_build_representative_focused_ss_payload(comparative_ss_payload: Dict[str, Any],
                                                representatives_df: pd.DataFrame) -> Dict[str, Any]:
    """Filter the existing comparative AlphaFold SS bundle to the focused
    representative set. Returns a payload with the same outer shape as
    `comparative_ss_payload` so downstream readers/render code can re-use it."""
    if not isinstance(comparative_ss_payload, dict):
        return {"available": False, "reason": "No comparative SS payload."}
    if representatives_df is None or representatives_df.empty:
        return {"available": False, "reason": "No representatives selected."}
    keep = set(representatives_df["species"].str.lower().tolist())
    records = comparative_ss_payload.get("records") or comparative_ss_payload.get("species") or []
    if not isinstance(records, list):
        return {"available": False, "reason": "Comparative SS payload missing records list."}
    filtered = [r for r in records if isinstance(r, dict) and str(r.get("species", "")).lower() in keep]
    out: Dict[str, Any] = dict(comparative_ss_payload)
    out["records"] = filtered
    out["focused_species_count"] = len(filtered)
    out["representative_species"] = list(representatives_df["species"])
    out["focused_view"] = True
    out["available"] = bool(filtered)
    if not filtered:
        out["reason"] = "No representative species had usable AlphaFold SS data."
    return out


# ---------- V11 paper-quality phylogenetic tree --------------------------- #

# Coarse broad-clade → presentation colour. Maps the V11 buckets to a
# consistent palette used across the paper tree, heatmaps, and any future
# interactive views.
V11_CLADE_PALETTE: Dict[str, str] = {
    "Mammalia":         "#2563eb",  # indigo
    "Aves":             "#16a34a",  # green
    "Reptilia":         "#0891b2",  # cyan
    "Amphibia":         "#22c55e",  # light green
    "Teleostei":        "#f97316",  # orange
    "Holostei":         "#ef4444",  # red
    "Polypteriformes":  "#dc2626",  # darker red
    "Chondrostei":      "#b91c1c",  # deeper red
    "Coelacanthiformes":"#a855f7",  # purple
    "Dipnoi":           "#7c3aed",  # darker purple
    "Chondrichthyes":   "#ec4899",  # pink
    "Cyclostomata":     "#8b5cf6",  # violet
    "Tunicata":         "#64748b",  # slate
    "Cephalochordata":  "#475569",  # darker slate
    "Fungi":            "#92400e",  # brown
    "Arthropoda":       "#a16207",  # mustard
    "Unassigned":       "#9ca3af",  # grey
    # Subdivided 9-group palette (Primates/Rodents/OtherMammals/Teleosts/
    # OtherFish/Birds/Reptiles/Amphibians/OtherVertebrates). Originally a
    # MATLAB clade-analysis grouping; reimplemented in Python. The mammal
    # subdivision is the key extra information vs the broad-clade view.
    "Primates":         "#1d4ed8",  # deep blue
    "Rodents":          "#fbbf24",  # amber
    "OtherMammals":     "#9333ea",  # purple-violet
    "Teleosts":         "#f97316",  # orange  (shares with Teleostei)
    "OtherFish":        "#ef4444",  # red     (shares with Holostei)
    "Birds":            "#16a34a",  # green   (Aves)
    "Reptiles":         "#0891b2",  # cyan    (Reptilia)
    "Amphibians":       "#22c55e",  # light green (Amphibia)
    "OtherVertebrates": "#475569",  # slate (jawless + tunicates + cephalochordates)
    # Compact 9-group buckets — match the user's clade_identity_by_reference_
    # position_mod.csv column names exactly (Other, x).
    "Other":            "#ef4444",  # non-teleost fish (sharks, gar, coelacanth, lungfish)
    "x":                "#475569",  # jawless / tunicates / invertebrates / unassigned
}

# Genus-token sets for the MATLAB 9-group remap. Drawn from
# _V11_MAMMAL_GENUS_TOKENS so the subdivision is consistent with how the
# broad-clade resolver already buckets species into "Mammalia".
_V11_PRIMATE_GENUS_TOKENS: Tuple[str, ...] = (
    "homo", "pan", "gorilla", "pongo", "nomascus", "hylobates",
    "macaca", "papio", "mandrillus", "cercocebus", "chlorocebus",
    "rhinopithecus", "saimiri", "aotus", "cebus", "callithrix",
    "carlito", "tarsius", "otolemur", "microcebus", "prolemur",
    "propithecus", "lemur", "indri",
)
_V11_RODENT_GENUS_TOKENS: Tuple[str, ...] = (
    # Rodentia
    "mus", "rattus", "peromyscus", "cricetulus", "mesocricetus", "microtus",
    "jaculus", "dipodomys", "octodon", "chinchilla", "cavia", "ictidomys",
    "urocitellus", "marmota", "sciurus", "spermophilus", "nannospalax",
    "heterocephalus", "fukomys", "castor", "perognathus",
    # Lagomorpha — grouped with rodents (Glires) for the MATLAB chart.
    "ochotona", "oryctolagus", "lepus",
)
_V11_OTHER_FISH_BROAD_CLADES: frozenset = frozenset({
    "Holostei", "Polypteriformes", "Chondrostei", "Chondrichthyes",
    "Coelacanthiformes", "Dipnoi",
})

# Display order for the subdivided 9-group bubble grid (most-conserved →
# most-distant), matching the row order of the user's reference MATLAB chart
# we are reproducing in Python.
_V11_GROUPED_BUBBLE_CLADE_ORDER: Tuple[str, ...] = (
    "Primates", "Rodents", "OtherMammals",
    "Teleosts", "OtherFish",
    "Birds", "Reptiles", "Amphibians",
    "OtherVertebrates",
)


# Display order for the compact 9-group "mod" bubble grid — exact column order
# from the user's reference clade_identity_by_reference_position_mod.csv.
_V11_MOD_BUBBLE_CLADE_ORDER: Tuple[str, ...] = (
    "Primates", "Rodents", "OtherMammals",
    "Teleosts", "Birds", "Reptiles", "Amphibians",
    "Other", "x",
)


def v11_resolve_mod_clade(species: str, broad_clade: str) -> str:
    """Remap a V11 broad clade label onto the user's MATLAB-era 9-group
    column set: Primates / Rodents / OtherMammals / Teleosts / Birds /
    Reptiles / Amphibians / Other / x. Same as the 9-group resolver but
    using the exact column names from the user's reference CSV
    `clade_identity_by_reference_position_mod.csv` (with `Other` for
    non-teleost fish and `x` for jawless / tunicates / unassigned).
    """
    tokens = set(_v11_species_tokens(species or ""))
    if broad_clade == "Mammalia":
        if tokens & set(_V11_PRIMATE_GENUS_TOKENS):
            return "Primates"
        if tokens & set(_V11_RODENT_GENUS_TOKENS):
            return "Rodents"
        return "OtherMammals"
    if broad_clade == "Teleostei":
        return "Teleosts"
    if broad_clade in _V11_OTHER_FISH_BROAD_CLADES:
        return "Other"
    if broad_clade == "Aves":
        return "Birds"
    if broad_clade == "Reptilia":
        return "Reptiles"
    if broad_clade == "Amphibia":
        return "Amphibians"
    return "x"


def v11_resolve_grouped_clade(species: str, broad_clade: str) -> str:
    """Remap a V11 broad clade label (`Mammalia`, `Aves`, ...) onto the
    subdivided 9-group categorization: Primates / Rodents / OtherMammals /
    Teleosts / OtherFish / Birds / Reptiles / Amphibians / OtherVertebrates.

    Mammalia is subdivided by primate/rodent genus tokens (with Lagomorpha
    pooled with Rodents under Glires); non-teleost fish (sharks, gar,
    coelacanths, lungfish, ...) collapse into OtherFish; jawless fish +
    tunicates + cephalochordates + truly unassigned records collapse into
    OtherVertebrates. This is the Python reimplementation of the MATLAB
    clade-analysis grouping; the original was developed in MATLAB but is now
    delivered alongside the V11 outputs for any input gene.
    """
    tokens = set(_v11_species_tokens(species or ""))
    if broad_clade == "Mammalia":
        if tokens & set(_V11_PRIMATE_GENUS_TOKENS):
            return "Primates"
        if tokens & set(_V11_RODENT_GENUS_TOKENS):
            return "Rodents"
        return "OtherMammals"
    if broad_clade == "Teleostei":
        return "Teleosts"
    if broad_clade in _V11_OTHER_FISH_BROAD_CLADES:
        return "OtherFish"
    if broad_clade == "Aves":
        return "Birds"
    if broad_clade == "Reptilia":
        return "Reptiles"
    if broad_clade == "Amphibia":
        return "Amphibians"
    return "OtherVertebrates"


def _v11_clade_color(clade: str) -> str:
    return V11_CLADE_PALETTE.get(clade or "Unassigned", "#374151")


def v11_plot_paper_quality_tree_svg(treefile: Path | str,
                                    out_svg: Path | str,
                                    *,
                                    representatives_df: Optional[pd.DataFrame] = None,
                                    taxonomy_lookup: Optional[Dict[str, Any]] = None,
                                    title: str = "",
                                    show_bootstrap_threshold: float = 70.0) -> Optional[Path]:
    """Render the IQ-TREE phylogeny with paper-quality styling:

    * Tip labels colored by V11 broad clade.
    * Representative species (from `representatives_df`) bolded and marked
      with a leading ★ glyph.
    * Bootstrap support shown only when ≥ `show_bootstrap_threshold`.
    * A clade colour legend in the lower-right corner.
    * A scale bar for the inferred protein-distance branch lengths.
    """
    treefile = Path(treefile)
    out_svg = Path(out_svg)
    if not treefile.exists():
        return None
    try:
        tree = Phylo.read(str(treefile), "newick")
    except Exception:
        return None

    tips = tree.get_terminals()
    n_tips = len(tips)
    if n_tips == 0:
        return None

    representative_keys: set = set()
    if representatives_df is not None and not representatives_df.empty:
        representative_keys = {str(s).lower() for s in representatives_df["species"]}

    # Determine clade per tip via the V11 classifier.
    tip_clade: Dict[str, str] = {}
    for tip in tips:
        raw = str(tip.name or "")
        species_key = raw.split("|", 1)[0].strip()
        species_key_norm = species_key.split(" ")[0].lower()
        tax_level = ""
        if taxonomy_lookup:
            raw_lookup = taxonomy_lookup.get(species_key) or taxonomy_lookup.get(species_key_norm) or {}
            if isinstance(raw_lookup, dict):
                tax_level = str(raw_lookup.get("taxonomy_level") or raw_lookup.get("clade") or "")
            elif isinstance(raw_lookup, str):
                tax_level = raw_lookup
        clade = v11_resolve_broad_clade(species_key_norm, tax_level)
        tip_clade[raw] = clade

    # Figure & layout: scale aggressively for large tip counts so labels do
    # not overlap. ~0.22 inch per tip vertical → ~45 inches for 204 tips.
    fig_height = max(10.0, 0.22 * n_tips + 2.0)
    fig_width = 18.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Custom label function so Phylo.draw uses our friendly names.
    def label_func(clade_obj: Any) -> str:
        raw = str(getattr(clade_obj, "name", "") or "")
        if not raw:
            return ""
        species, symbol = parse_header_species_symbol(raw)
        prefix = "★ " if species.lower() in representative_keys else "  "
        nice_species = species.replace("_", " ").capitalize()
        if symbol and symbol != species:
            return f"{prefix}{nice_species} ({symbol})"
        return f"{prefix}{nice_species}"

    # Bootstrap-confidence filter on internal nodes.
    for node in tree.get_nonterminals():
        conf = getattr(node, "confidence", None)
        try:
            conf_val = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            conf_val = None
        if conf_val is None or conf_val < show_bootstrap_threshold:
            node.confidence = None
        else:
            # Phylo.draw shows confidence as a string; render with one decimal.
            node.confidence = f"{conf_val:.0f}"

    Phylo.draw(tree, axes=ax, do_show=False, label_func=label_func, branch_labels=lambda c: getattr(c, "confidence", None))

    # Recolor tip labels by clade after Phylo.draw placed them.
    for text in ax.texts:
        label = text.get_text()
        if not label:
            continue
        is_rep = label.strip().startswith("★")
        species_part = label.strip().lstrip("★").strip()
        species_norm = species_part.split(" (")[0].strip().lower().replace(" ", "_")
        clade = None
        for tip_raw, c in tip_clade.items():
            tip_species, _ = parse_header_species_symbol(tip_raw)
            if tip_species.lower() == species_norm:
                clade = c
                break
        if clade:
            text.set_color(_v11_clade_color(clade))
        if is_rep:
            text.set_fontweight("bold")
            text.set_fontsize(11.0)
        else:
            text.set_fontsize(8.0)

    # Hide spines/ticks for a clean look.
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=7)

    if title:
        ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Branch length (substitutions per site)", fontsize=10)
    # Hide the y-axis (Phylo.draw uses it for tip index — meaningless here).
    ax.set_ylabel("")
    ax.set_yticks([])

    # Clade legend in lower-right.
    present_clades = sorted({c for c in tip_clade.values() if c})
    if present_clades:
        legend_text_lines = ["Clade colors (V11)"]
        for c in present_clades:
            color = _v11_clade_color(c)
            legend_text_lines.append(f"  {c}")
        # Render legend as colored text via figtext for clarity.
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="s", color=_v11_clade_color(c), linestyle="", markersize=9, label=c)
            for c in present_clades
        ]
        handles.append(Line2D([0], [0], marker=r"$\bigstar$", color="#374151", linestyle="", markersize=10,
                              label="V11 clade representative"))
        ax.legend(
            handles=handles,
            loc="lower right",
            frameon=True,
            framealpha=0.92,
            fontsize=8,
            title="V11",
        )

    fig.tight_layout()
    fig.savefig(out_svg)
    png_path = out_svg.with_suffix(".png")
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return out_svg


def v11_write_representative_comparison_outputs(outdir: Path | str,
                                                reference_projected_alignment: Any,
                                                reference_species: str,
                                                taxonomy_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
                                                smoothing_window: int = V11_DEFAULT_PROPERTY_WINDOW,
                                                always_include: Sequence[str] = V11_MANDATORY_FOCUS_SPECIES) -> Dict[str, Any]:
    """Compute and persist the V11 representative-comparison artifacts. Returns
    a payload dict describing what was written (for logging / interactive
    report integration)."""
    outdir = Path(outdir)

    reps_df = v11_select_clade_representatives(
        reference_projected_alignment,
        reference_species=reference_species,
        taxonomy_lookup=taxonomy_lookup,
        always_include=always_include,
    )
    reps_path = outdir / V11_DEFAULT_REPRESENTATIVE_CSV
    reps_df.to_csv(reps_path, sep="\t", index=False)

    charge_df = v11_compute_per_species_property_track(
        reference_projected_alignment,
        V11_RESIDUE_NET_CHARGE_PH74,
        reference_species=reference_species,
        smoothing_window=smoothing_window,
    )
    charge_df.to_csv(outdir / V11_NET_CHARGE_PER_SPECIES_CSV, index=False)
    charge_delta_df = _v11_delta_vs_reference(charge_df, reference_species)
    charge_delta_df.to_csv(outdir / V11_NET_CHARGE_DELTA_CSV, index=False)

    arom_df = v11_compute_per_species_property_track(
        reference_projected_alignment,
        V11_RESIDUE_AROMATICITY,
        reference_species=reference_species,
        smoothing_window=smoothing_window,
    )
    arom_df.to_csv(outdir / V11_AROMATICITY_PER_SPECIES_CSV, index=False)
    arom_delta_df = _v11_delta_vs_reference(arom_df, reference_species)
    arom_delta_df.to_csv(outdir / V11_AROMATICITY_DELTA_CSV, index=False)

    # Heatmaps: keep the focused-set rows from the delta frames.
    heatmap_png = outdir / V11_REPRESENTATIVE_PROPERTY_HEATMAP_PNG
    heatmap_svg = outdir / V11_REPRESENTATIVE_PROPERTY_HEATMAP_SVG
    # Stack net-charge and aromaticity vertically in a single panel via two-pass call.
    # We emit one heatmap per property under suffix names for clarity.
    charge_heat_png = outdir / "v11_representative_net_charge_heatmap.png"
    charge_heat_svg = outdir / "v11_representative_net_charge_heatmap.svg"
    arom_heat_png = outdir / "v11_representative_aromaticity_heatmap.png"
    arom_heat_svg = outdir / "v11_representative_aromaticity_heatmap.svg"
    _v11_plot_property_heatmap(
        charge_delta_df, reps_df,
        property_label="Net charge (pH 7.4)",
        cmap_name="RdBu_r",
        vmin=-1.0, vmax=1.0,
        outpath_png=charge_heat_png, outpath_svg=charge_heat_svg,
        reference_species=reference_species,
    )
    _v11_plot_property_heatmap(
        arom_delta_df, reps_df,
        property_label="Aromaticity fraction",
        cmap_name="PuOr_r",
        vmin=-1.0, vmax=1.0,
        outpath_png=arom_heat_png, outpath_svg=arom_heat_svg,
        reference_species=reference_species,
    )

    # Combined heatmap legacy filename (net-charge primary, aromaticity below)
    # — kept as alias to charge heatmap so report indexing finds something.
    if charge_heat_png.exists():
        try:
            heatmap_png.write_bytes(charge_heat_png.read_bytes())
            heatmap_svg.write_bytes(charge_heat_svg.read_bytes())
        except OSError:
            pass

    traces_png = outdir / V11_REPRESENTATIVE_PROPERTY_TRACES_PNG
    traces_svg = outdir / V11_REPRESENTATIVE_PROPERTY_TRACES_SVG
    _v11_plot_property_traces(charge_df, arom_df, reps_df, reference_species,
                              outpath_png=traces_png, outpath_svg=traces_svg)

    # Focused SS bundle (filter the existing comparative bundle if present).
    focused_ss_payload: Dict[str, Any] = {"available": False, "reason": "Comparative SS bundle not yet built."}
    comparative_ss_path = outdir / COMPARATIVE_ALPHAFOLD_SS_FILENAME
    if comparative_ss_path.exists():
        try:
            comp_payload = json.loads(comparative_ss_path.read_text(encoding="utf-8"))
            focused_ss_payload = v11_build_representative_focused_ss_payload(comp_payload, reps_df)
            (outdir / V11_REPRESENTATIVE_SS_BUNDLE_JSON).write_text(
                json.dumps(focused_ss_payload, indent=2), encoding="utf-8"
            )
        except (OSError, json.JSONDecodeError) as exc:
            focused_ss_payload = {"available": False, "reason": f"Could not parse comparative SS: {exc}"}

    return {
        "representatives_path": str(reps_path),
        "representative_count": int(len(reps_df)),
        "net_charge_per_species_path": str(outdir / V11_NET_CHARGE_PER_SPECIES_CSV),
        "aromaticity_per_species_path": str(outdir / V11_AROMATICITY_PER_SPECIES_CSV),
        "net_charge_delta_path": str(outdir / V11_NET_CHARGE_DELTA_CSV),
        "aromaticity_delta_path": str(outdir / V11_AROMATICITY_DELTA_CSV),
        "net_charge_heatmap_png": str(charge_heat_png) if charge_heat_png.exists() else None,
        "aromaticity_heatmap_png": str(arom_heat_png) if arom_heat_png.exists() else None,
        "representative_traces_png": str(traces_png) if traces_png.exists() else None,
        "focused_ss_bundle_path": str(outdir / V11_REPRESENTATIVE_SS_BUNDLE_JSON) if comparative_ss_path.exists() else None,
        "focused_ss_payload_summary": {
            "available": focused_ss_payload.get("available"),
            "focused_species_count": focused_ss_payload.get("focused_species_count"),
        },
        "smoothing_window": smoothing_window,
    }


# =============================================================================
# V11 motif & lineage-stabilization extension                                 #
# -----------------------------------------------------------------------------#
# Two pieces, both keyed off ungapped reference positions and the V11 broad-  #
# clade classifier (v11_resolve_broad_clade):                                  #
#                                                                              #
# 1. **User-supplied motif inspector** — `--annotated_motifs                   #
#    "263-269:cPLA2_PL_rich_263_269"` lets the user point at any ref-position  #
#    range and gets per-species residues + per-clade consensus + sequence      #
#    logos. No function is assumed; the label is whatever the user wrote.     #
# 2. **Regex motif-library auto-scan** — V11_REGULATORY_MOTIF_LIBRARY holds a  #
#    small curated set (classical/bipartite NLS, NES, PEST, PxxP, LxxLL,       #
#    leucine zipper). Each human-reference hit becomes a labeled range and    #
#    flows through the same per-clade analysis pipeline. The user can add    #
#    extra regexes via --v11_extra_motif_regex.                                #
#                                                                              #
# Plus a "lineage stabilization" signal: per position, Shannon entropy is      #
# computed within each broad clade; the difference between ancestral and      #
# derived clade entropies is the stabilization score, a simplified            #
# entropy-based approximation of Gu's Type I functional divergence            #
# (Gu, MBE 1999; Gu, MBE 2006). For pair-wise clade differences we use        #
# Jensen-Shannon divergence between per-clade residue frequency vectors,      #
# following Capra & Singh (Bioinformatics 2007).                              #
# =============================================================================

import re as _v11_re
from collections import Counter as _V11_Counter

V11_ANNOTATED_MOTIFS_TSV = "v11_annotated_motifs.tsv"
V11_MOTIF_LIBRARY_HITS_TSV = "v11_motif_library_hits.tsv"
V11_MOTIFS_MASTER_TSV = "v11_motifs_master.tsv"
V11_PER_CLADE_ENTROPY_CSV = "v11_per_clade_entropy.csv"
V11_PER_CLADE_CONSENSUS_CSV = "v11_per_clade_consensus.csv"
V11_CLADE_PAIR_JS_DIVERGENCE_CSV = "v11_clade_pair_js_divergence.csv"
V11_LINEAGE_STABILIZATION_CSV = "v11_lineage_stabilization.csv"
V11_MOTIF_EVOLUTION_PER_SPECIES_TSV = "v11_motif_evolution_per_species.tsv"
V11_MOTIF_EVOLUTION_PER_CLADE_TSV = "v11_motif_evolution_per_clade.tsv"
V11_LINEAGE_STABILIZATION_LANDSCAPE_PNG = "v11_lineage_stabilization_landscape.png"
V11_LINEAGE_STABILIZATION_LANDSCAPE_SVG = "v11_lineage_stabilization_landscape.svg"
V11_ALIGNMENT_WITH_MOTIF_ANNOTATIONS_SVG = "v11_alignment_with_motif_annotations.svg"
V11_ALIGNMENT_WITH_MOTIF_ANNOTATIONS_HTML = "v11_alignment_with_motif_annotations.html"

# Default ancestral / derived clade buckets for the stabilization score. They
# match the typical "before vs after jawed-vertebrate emergence" comparison
# used for cPLA2-style proteins. Users can override via CLI flags.
V11_DEFAULT_ANCESTRAL_CLADES: Tuple[str, ...] = ("Cyclostomata", "Tunicata", "Chondrichthyes")
V11_DEFAULT_DERIVED_CLADES: Tuple[str, ...] = ("Mammalia", "Aves", "Reptilia")

# Standard 20 amino acids for entropy normalization (gaps & ambiguities are
# tracked separately as occupancy).
_V11_AA_ALPHABET: Tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")

# Curated regulatory-motif library. Each entry is (name, regex, description,
# citation). Regexes are case-sensitive and applied to the upper-case human
# reference (ungapped). They are intentionally permissive — V11 reports each
# hit; the user decides which to trust.
V11_REGULATORY_MOTIF_LIBRARY: Tuple[Tuple[str, str, str, str], ...] = (
    ("NLS_classical_monopartite", r"K(K|R)[A-Z](K|R)", "Classical monopartite nuclear localization signal (K-rich)", "Lange et al., Trends Cell Biol 2007"),
    ("NLS_bipartite", r"[KR][KR][A-Z]{10,12}[KR][KR][A-Z][KR][KR]", "Bipartite nuclear localization signal", "Robbins et al., Cell 1991"),
    ("NES_leucine_rich", r"L[A-Z]{2,3}[LIMVF][A-Z]{2,3}[LIMVF][A-Z]L", "Leucine-rich nuclear export signal (CRM1-binding)", "la Cour et al., Protein Eng Des Sel 2004"),
    ("PEST_PEST_rich", r"[PEST]{12,}", "PEST sequence (P/E/S/T-enriched, degron)", "Rogers et al., Science 1986"),
    ("PxxP_SH3_binding", r"P[A-Z][A-Z]P", "SH3-binding minimal core (PxxP)", "Mayer & Saksela, Curr Top Microbiol Immunol 2002"),
    ("LxxLL_nuclear_receptor", r"L[A-Z][A-Z]LL", "Nuclear receptor coactivator box (LxxLL)", "Heery et al., Nature 1997"),
    ("Leucine_zipper_heptad", r"L[A-Z]{6}L[A-Z]{6}L[A-Z]{6}L", "Leucine zipper (L at every 7th position over 4 heptads)", "Landschulz et al., Science 1988"),
    ("Proline_rich_stretch", r"P[A-Z]{0,2}P[A-Z]{0,2}P[A-Z]{0,2}P", "Generic proline-rich stretch (≥4 P within 12 residues)", "Williamson, Biochem J 1994"),
    ("Acidic_blob", r"[DE]{4,}", "Acidic blob (≥4 contiguous D/E)", "Sigler, Nature 1988"),
    ("Basic_blob", r"[KR]{4,}", "Basic blob (≥4 contiguous K/R)", "Generic"),
    # ----- V11.1 additions (2026-06-01): regulatory landscape expansion ----- #
    # Phospho-dependent partner-binding motifs (14-3-3 family). Sequences
    # written without explicit phospho marker — S/T is the phospho-acceptor
    # and any per-species loss of that S/T is downstream of the motif scan.
    ("14_3_3_mode_I", r"R[A-Z][A-Z](S|T)[A-Z]P", "14-3-3 binding mode I (RSxpSxP)", "Yaffe et al., Cell 1997"),
    ("14_3_3_mode_II", r"R[A-Z](Y|F)[A-Z](S|T)[A-Z]P", "14-3-3 binding mode II (RxY/FxpSxP)", "Yaffe et al., Cell 1997"),
    # Phosphodegrons — kinase-primed ubiquitin ligase recruitment motifs.
    ("phosphodegron_betaTrCP", r"D(S|T)G[A-Z]{2,3}(S|T)", "β-TrCP phosphodegron (DSGxxS-style)", "Wu et al., Mol Cell 2003"),
    ("phosphodegron_FBW7", r"(L|I|P)[A-Z](S|T)P[A-Z][A-Z](S|T|E|D)", "FBW7 phosphodegron (CPD/CdcD)", "Welcker & Clurman, Nat Rev Cancer 2008"),
    # Cell-cycle ubiquitin ligase recognition motifs (APC/C substrates).
    ("KEN_box", r"KEN[A-Z]{0,3}(D|E)", "KEN box (APC/C-Cdh1 recognition motif)", "Pfleger & Kirschner, Genes Dev 2000"),
    ("D_box", r"R[A-Z][A-Z]L[A-Z]{2,3}[ILV][A-Z]N", "Destruction box (APC/C recognition motif)", "Glotzer et al., Nature 1991"),
    # Post-translational modification — sumoylation consensus.
    ("SUMO_consensus", r"[VILMFP]K[A-Z]E", "SUMOylation consensus (Ψ-K-x-E)", "Rodriguez et al., JBC 2001"),
    # Lipid / membrane targeting motifs.
    ("polybasic_membrane", r"[KR]{5,8}", "Polybasic membrane-binding cluster (PIP2)", "Heo et al., Science 2006"),
    ("CAAX_prenylation", r"C[A-Z][A-Z](S|T|M|Q|A|L)$", "C-terminal CAAX prenylation motif", "Reid et al., Curr Opin Cell Biol 1999"),
    # Heuristic N-terminal targeting tag (mitochondrial import signal start).
    ("Mitochondrial_targeting_R", r"^M[A-Z]{0,3}[KR][KR]", "N-terminal arginine-rich mitochondrial targeting prefix", "Vögtle et al., Cell 2009"),
)


# ----------------------- residue / motif parsing helpers ----------------------#

def v11_parse_annotated_motifs(motif_text: Optional[str]) -> pd.DataFrame:
    """Parse `"263-269:cPLA2_PL_rich_263_269,500-505:label_b"` into
    [motif_id, start, end, label, source]. Start and end are 1-based,
    inclusive, and both must be present (single positions still need
    `pos-pos`). No function/sequence is assumed."""
    rows: List[Dict[str, Any]] = []
    if not motif_text:
        return pd.DataFrame(columns=["motif_id", "start", "end", "label", "source"])
    for raw in str(motif_text).split(","):
        token = raw.strip()
        if not token:
            continue
        # Accept "start-end:label" or "start-end" (label auto-derived).
        parts = token.split(":", 1)
        range_part = parts[0].strip()
        label = parts[1].strip() if len(parts) > 1 else ""
        if "-" not in range_part:
            continue
        try:
            start_str, end_str = range_part.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
        except ValueError:
            continue
        if start <= 0 or end <= 0 or end < start:
            continue
        if not label:
            label = f"user_motif_{start}_{end}"
        motif_id = f"user__{label}__{start}_{end}"
        rows.append({
            "motif_id": motif_id,
            "start": start,
            "end": end,
            "label": label,
            "source": "user",
        })
    return pd.DataFrame(rows, columns=["motif_id", "start", "end", "label", "source"])


def v11_parse_extra_motif_regex(text: Optional[str]) -> List[Tuple[str, str]]:
    """Parse `"name1:regex1,name2:regex2"` into list of (name, regex)."""
    out: List[Tuple[str, str]] = []
    if not text:
        return out
    for raw in str(text).split(","):
        token = raw.strip()
        if not token or ":" not in token:
            continue
        name, regex = token.split(":", 1)
        name = name.strip()
        regex = regex.strip()
        if name and regex:
            out.append((name, regex))
    return out


def v11_scan_motif_library(reference_sequence: str,
                           library: Sequence[Tuple[str, str, str, str]] = V11_REGULATORY_MOTIF_LIBRARY,
                           extra_regex: Optional[Sequence[Tuple[str, str]]] = None) -> pd.DataFrame:
    """Scan an ungapped reference sequence with every regex in the library.
    Coordinates returned are 1-based, inclusive. Each library entry can hit
    multiple times; overlapping hits are kept (we leave de-duplication to the
    user via the report)."""
    ref = str(reference_sequence or "").upper()
    rows: List[Dict[str, Any]] = []
    full_lib: List[Tuple[str, str, str, str]] = list(library)
    for name, regex in (extra_regex or []):
        full_lib.append((str(name), str(regex), "User-supplied extra motif regex", "user"))
    for name, regex, description, citation in full_lib:
        try:
            compiled = _v11_re.compile(regex)
        except _v11_re.error:
            continue
        for m in compiled.finditer(ref):
            start_1b = m.start() + 1
            end_1b = m.end()  # finditer end is exclusive (0-based); inclusive 1-based = end()
            motif_id = f"lib__{name}__{start_1b}_{end_1b}"
            rows.append({
                "motif_id": motif_id,
                "motif_name": name,
                "regex": regex,
                "description": description,
                "citation": citation,
                "start": int(start_1b),
                "end": int(end_1b),
                "matched_seq": ref[m.start():m.end()],
                "label": f"{name}_{start_1b}_{end_1b}",
                "source": "library",
            })
    return pd.DataFrame(rows, columns=[
        "motif_id", "motif_name", "regex", "description", "citation",
        "start", "end", "matched_seq", "label", "source",
    ])


def v11_merge_motif_tables(user_df: pd.DataFrame, library_df: pd.DataFrame) -> pd.DataFrame:
    """Union user-supplied motifs with library hits; return a single table
    keyed by motif_id. User motifs come first."""
    frames: List[pd.DataFrame] = []
    if not user_df.empty:
        u = user_df.copy()
        for col in ("motif_name", "regex", "description", "citation", "matched_seq"):
            if col not in u.columns:
                u[col] = ""
        frames.append(u[["motif_id", "label", "source", "start", "end",
                          "motif_name", "regex", "description", "citation", "matched_seq"]])
    if not library_df.empty:
        l = library_df.copy()
        frames.append(l[["motif_id", "label", "source", "start", "end",
                          "motif_name", "regex", "description", "citation", "matched_seq"]])
    if not frames:
        return pd.DataFrame(columns=["motif_id", "label", "source", "start", "end",
                                      "motif_name", "regex", "description", "citation", "matched_seq"])
    return pd.concat(frames, ignore_index=True)


# ----------------- per-clade entropy / consensus / JS divergence -------------#

def _v11_ref_position_records(alignment: Any, reference_species: str) -> Tuple[List[Optional[int]], Any]:
    """Return (alignment_col → ref_ungapped_position, reference_record)."""
    target = (reference_species or "").strip().lower()
    ref_record = None
    for record in alignment:
        species, _ = parse_header_species_symbol(record.id)
        if species.lower() == target:
            ref_record = record
            break
    if ref_record is None:
        ref_record = alignment[0]
    ref_aligned = str(ref_record.seq).upper()
    aln_to_ref: List[Optional[int]] = []
    counter = 0
    for aa in ref_aligned:
        if aa in GAP_CHARS:
            aln_to_ref.append(None)
        else:
            counter += 1
            aln_to_ref.append(counter)
    return aln_to_ref, ref_record


def _v11_species_clade_map(alignment: Any,
                           taxonomy_lookup: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Return record_id → broad clade, using v11_resolve_broad_clade."""
    out: Dict[str, str] = {}
    for record in alignment:
        species, _ = parse_header_species_symbol(record.id)
        tax_level = _v11_resolve_taxonomy_level(taxonomy_lookup, species) if taxonomy_lookup is not None else ""
        out[str(record.id)] = v11_resolve_broad_clade(species, tax_level) or "Unassigned"
    return out


def _v11_shannon_entropy_from_counts(counts: Dict[str, int]) -> Tuple[float, int]:
    """Return (Shannon entropy in bits, total non-gap count)."""
    total = sum(int(v) for k, v in counts.items() if k not in GAP_CHARS)
    if total == 0:
        return 0.0, 0
    h = 0.0
    for aa, c in counts.items():
        if aa in GAP_CHARS or c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return float(h), int(total)


def v11_compute_per_clade_entropy(alignment: Any,
                                  taxonomy_lookup: Optional[Dict[str, Any]],
                                  reference_species: str) -> pd.DataFrame:
    """Per ungapped reference position, return Shannon entropy and occupancy
    (number of non-gap residues) per broad clade. Output is wide:
        reference_ungapped_position, reference_residue,
        entropy_<clade>, occupancy_<clade>, ...
    """
    aln_to_ref, ref_record = _v11_ref_position_records(alignment, reference_species)
    species_clade = _v11_species_clade_map(alignment, taxonomy_lookup)
    # Pre-collect records grouped by clade.
    clade_records: Dict[str, List[Any]] = {}
    for record in alignment:
        clade = species_clade.get(str(record.id), "Unassigned")
        clade_records.setdefault(clade, []).append(record)
    ref_seq = str(ref_record.seq).upper()

    # For every alignment column that has a ref-position, compute per-clade
    # entropy from the column's residues.
    rows: List[Dict[str, Any]] = []
    for col_idx, ref_pos in enumerate(aln_to_ref):
        if ref_pos is None:
            continue
        row: Dict[str, Any] = {
            "reference_ungapped_position": int(ref_pos),
            "reference_residue": ref_seq[col_idx] if col_idx < len(ref_seq) else "-",
        }
        for clade, records in clade_records.items():
            counts: _V11_Counter = _V11_Counter()
            for rec in records:
                seq = str(rec.seq)
                if col_idx < len(seq):
                    counts[seq[col_idx].upper()] += 1
            h, occ = _v11_shannon_entropy_from_counts(dict(counts))
            row[f"entropy_{clade}"] = h
            row[f"occupancy_{clade}"] = occ
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------- V11.1 Subgroup-Discriminating Positions ------------#
# Central pillar of V11.1 Functional Divergence module: user picks two named
# subgroups of species; per ungapped reference position, scores how strongly
# the two subgroups conserve DIFFERENT residues. Captures both subcellular-
# targeting phenotype divergence (e.g. nuclear-vs-cytoplasmic localisation)
# and catalytic/substrate phenotype divergence (e.g. active-vs-inactive on
# a substrate the lab tests) — one statistic, two phenotype classes.
# -----------------------------------------------------------------------------

V11_SUBGROUP_SDP_CSV_TEMPLATE = "v11_subgroup_sdp_{label_a}_vs_{label_b}.csv"


def _v11_fisher_exact_2x2_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Hand-rolled two-sided Fisher's exact for a 2x2 contingency table

        |          | residue X | other |
        | subgroup A |    a    |   b   |
        | subgroup B |    c    |   d   |

    Returns a two-sided p-value. Uses math.comb (stdlib) so no scipy needed.
    For empty marginals (any row or column total 0) returns 1.0.
    """
    n = a + b + c + d
    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d
    if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0 or n == 0:
        return 1.0

    def _pmf(x: int) -> float:
        # Hypergeometric PMF P(X = x) given marginals row1, row2, col1.
        if x < 0 or x > col1 or (row1 - x) < 0 or (row2 - (col1 - x)) < 0:
            return 0.0
        try:
            return (
                math.comb(col1, x) * math.comb(col2, row1 - x) / math.comb(n, row1)
            )
        except ValueError:
            return 0.0

    observed = _pmf(a)
    if observed <= 0:
        return 1.0
    x_min = max(0, col1 - row2)
    x_max = min(col1, row1)
    total = 0.0
    for x in range(x_min, x_max + 1):
        p = _pmf(x)
        if p <= observed + 1e-12:
            total += p
    return float(max(0.0, min(1.0, total)))


def _v11_property_class_for_aa(aa: str, scheme: Dict[str, set[str]]) -> Optional[str]:
    aa = (aa or "").upper()
    for cls, residues in scheme.items():
        if aa in residues:
            return cls
    return None


def v11_compute_subgroup_sdp(alignment: Any,
                             reference_species: str,
                             subgroup_a_species: Sequence[str],
                             subgroup_b_species: Sequence[str],
                             label_a: str = "A",
                             label_b: str = "B",
                             property_scheme: Optional[Dict[str, set[str]]] = None,
                             min_occupancy: int = 2) -> pd.DataFrame:
    """Subgroup-Discriminating Position analyzer (Gu Type II-style).

    For every ungapped reference position, computes:
      consensus_A, consensus_B, consensus_frac_A, consensus_frac_B
      entropy_A, entropy_B (Shannon, bits, gap-excluded)
      occupancy_A, occupancy_B (non-gap residue count in each subgroup)
      delta_entropy = H_A - H_B
      consensus_changed = bool(consensus_A != consensus_B)
      property_class_changed = bool(prop_class(consensus_A) != prop_class(consensus_B))
      sdp_score = (1 - normalized_H_A) * (1 - normalized_H_B) * consensus_changed
                  (peaks where BOTH subgroups are conserved internally but the
                  conserved residues differ — classic Gu Type II signal)
      sdp_pvalue = two-sided Fisher's exact on (consensus_A, !consensus_A) ×
                   (consensus_A in B count, !consensus_A in B count)

    `property_scheme` defaults to AA_GROUP_SCHEMES["charge"] (most biology-
    relevant single class); pass any AA_GROUP_SCHEMES entry to swap.
    """
    if property_scheme is None:
        # Lazy import from the pipeline module to avoid a hard import cycle;
        # fall back to a built-in charge scheme if the pipeline isn't loaded.
        try:
            from gene_phylo_conservation_pipeline import AA_GROUP_SCHEMES  # type: ignore
            property_scheme = AA_GROUP_SCHEMES.get("charge", {})
        except Exception:  # noqa: BLE001
            property_scheme = {
                "positive": set("KRH"),
                "negative": set("DE"),
                "neutral": set("AVILMFWYCNQSTPG"),
            }

    aln_to_ref, ref_record = _v11_ref_position_records(alignment, reference_species)
    ref_seq = str(ref_record.seq).upper()

    a_keys = {str(s).strip().lower() for s in (subgroup_a_species or ()) if str(s).strip()}
    b_keys = {str(s).strip().lower() for s in (subgroup_b_species or ()) if str(s).strip()}
    records_a: List[Any] = []
    records_b: List[Any] = []
    for rec in alignment:
        species, _ = parse_header_species_symbol(str(rec.id))
        sp_lower = species.lower()
        if sp_lower in a_keys:
            records_a.append(rec)
        if sp_lower in b_keys:
            records_b.append(rec)
    if not records_a or not records_b:
        return pd.DataFrame(columns=[
            "reference_position", "reference_residue",
            "consensus_A", "consensus_B", "consensus_frac_A", "consensus_frac_B",
            "entropy_A", "entropy_B", "occupancy_A", "occupancy_B",
            "delta_entropy", "consensus_changed", "property_class_A",
            "property_class_B", "property_class_changed",
            "sdp_score", "sdp_pvalue",
        ])

    def _column_counts(records: Sequence[Any], col_idx: int) -> Dict[str, int]:
        ctr: Dict[str, int] = {}
        for rec in records:
            seq = str(rec.seq)
            if col_idx < len(seq):
                aa = seq[col_idx].upper()
                if aa in GAP_CHARS or aa in {"X", "?"}:
                    continue
                ctr[aa] = ctr.get(aa, 0) + 1
        return ctr

    def _consensus(counts: Dict[str, int]) -> Tuple[str, float, int]:
        if not counts:
            return "-", float("nan"), 0
        total = sum(counts.values())
        aa, cnt = max(counts.items(), key=lambda kv: (kv[1], -ord(kv[0])))
        return aa, (cnt / total) if total else float("nan"), total

    # Pre-compute max possible entropy across all amino acids (log2(20))
    # so we can normalize per-subgroup entropy into [0, 1] for the score.
    max_h = math.log2(20.0)

    rows: List[Dict[str, Any]] = []
    for col_idx, ref_pos in enumerate(aln_to_ref):
        if ref_pos is None:
            continue
        ref_aa = ref_seq[col_idx] if col_idx < len(ref_seq) else "-"
        counts_a = _column_counts(records_a, col_idx)
        counts_b = _column_counts(records_b, col_idx)
        cons_a, frac_a, n_a = _consensus(counts_a)
        cons_b, frac_b, n_b = _consensus(counts_b)
        h_a, _ = _v11_shannon_entropy_from_counts(counts_a)
        h_b, _ = _v11_shannon_entropy_from_counts(counts_b)
        delta_h = h_a - h_b

        consensus_changed = (cons_a != cons_b) and cons_a != "-" and cons_b != "-"
        prop_a = _v11_property_class_for_aa(cons_a, property_scheme) if property_scheme else None
        prop_b = _v11_property_class_for_aa(cons_b, property_scheme) if property_scheme else None
        prop_changed = bool(prop_a and prop_b and prop_a != prop_b)

        if n_a >= min_occupancy and n_b >= min_occupancy and consensus_changed:
            norm_h_a = h_a / max_h if max_h > 0 else 0.0
            norm_h_b = h_b / max_h if max_h > 0 else 0.0
            sdp_score = (1.0 - norm_h_a) * (1.0 - norm_h_b)
        else:
            sdp_score = 0.0

        if n_a >= min_occupancy and n_b >= min_occupancy and cons_a != "-":
            a_with = counts_a.get(cons_a, 0)
            a_without = n_a - a_with
            b_with = counts_b.get(cons_a, 0)
            b_without = n_b - b_with
            p_value = _v11_fisher_exact_2x2_two_sided(a_with, a_without, b_with, b_without)
        else:
            p_value = float("nan")

        rows.append({
            "reference_position": int(ref_pos),
            "reference_residue": ref_aa,
            "consensus_A": cons_a,
            "consensus_B": cons_b,
            "consensus_frac_A": frac_a,
            "consensus_frac_B": frac_b,
            "entropy_A": h_a,
            "entropy_B": h_b,
            "occupancy_A": n_a,
            "occupancy_B": n_b,
            "delta_entropy": delta_h,
            "consensus_changed": bool(consensus_changed),
            "property_class_A": prop_a or "",
            "property_class_B": prop_b or "",
            "property_class_changed": prop_changed,
            "sdp_score": float(sdp_score),
            "sdp_pvalue": float(p_value) if p_value == p_value else float("nan"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.attrs["label_a"] = label_a
        df.attrs["label_b"] = label_b
        df.attrs["n_species_A"] = len(records_a)
        df.attrs["n_species_B"] = len(records_b)
    return df


def v11_write_subgroup_sdp_outputs(outdir: Path | str,
                                   alignment: Any,
                                   reference_species: str,
                                   subgroup_a_species: Sequence[str],
                                   subgroup_b_species: Sequence[str],
                                   label_a: str = "A",
                                   label_b: str = "B") -> Optional[Path]:
    """Compute the SDP DataFrame and persist it as
    v11_subgroup_sdp_<label_a>_vs_<label_b>.csv.
    Returns the output Path or None if the analyzer found no overlap between
    requested subgroups and the alignment.
    """
    outdir = Path(outdir)
    df = v11_compute_subgroup_sdp(
        alignment,
        reference_species,
        subgroup_a_species,
        subgroup_b_species,
        label_a=label_a,
        label_b=label_b,
    )
    if df is None or df.empty:
        return None
    safe_a = _v11_re.sub(r"[^A-Za-z0-9_-]+", "_", str(label_a)).strip("_") or "A"
    safe_b = _v11_re.sub(r"[^A-Za-z0-9_-]+", "_", str(label_b)).strip("_") or "B"
    out = outdir / V11_SUBGROUP_SDP_CSV_TEMPLATE.format(label_a=safe_a, label_b=safe_b)
    df.to_csv(out, index=False)
    return out


# ----------------- V11.1 Phospho-regulatable signal detector -----------------#
# For every regulatory motif hit (e.g. NLS, NES, 14-3-3-binding) AND any
# active-site / catalytic residue from KNOWN_SITE_CONFIGS-derived annotation,
# scan ±10 residues for S/T/Y. Classify the parent motif as
# "phospho_regulatable" if any S/T/Y sits in the window; for each such S/T/Y
# list the per-species residue at that aligned position so subgroup-specific
# phospho-site gain/loss is immediately visible (e.g. zebrafish has S where
# human has A near an NLS → candidate phospho-mediated localisation switch).
# -----------------------------------------------------------------------------

V11_PHOSPHO_REGULATABLE_TSV = "v11_phospho_regulatable_signals.tsv"
V11_PHOSPHO_REGULATABLE_WINDOW = 10


def v11_compute_phospho_regulatable_signals(alignment: Any,
                                            reference_species: str,
                                            library_hits_df: Optional[pd.DataFrame] = None,
                                            extra_anchor_positions: Optional[Sequence[Tuple[int, str]]] = None,
                                            window: int = V11_PHOSPHO_REGULATABLE_WINDOW) -> pd.DataFrame:
    """Find every S / T / Y within `window` residues of a motif hit or
    user-supplied anchor (catalytic / active-site residue) and report:
      - parent_motif_id (or "active_site")
      - parent_label
      - parent_start / parent_end (or single position for active sites)
      - sty_position (1-based ungapped reference)
      - sty_context_window (11mer: pos±5 of human reference)
      - phospho_capable_species_count (S/T/Y in that species)
      - phospho_loss_species (semicolon-joined species that mutated S/T/Y to a
        non-phospho residue)

    `library_hits_df` is the DataFrame from `v11_scan_motif_library`; rows
    with `start` and `end` (1-based, inclusive) are consumed. Pass
    `extra_anchor_positions = [(228, "S228 nucleophile"), (549, "D549 proton
    acceptor")]` to also probe around catalytic residues.
    """
    aln_to_ref, ref_record = _v11_ref_position_records(alignment, reference_species)
    ref_seq = str(ref_record.seq).upper()
    ref_to_col: Dict[int, int] = {p: i for i, p in enumerate(aln_to_ref) if p is not None}
    if not ref_to_col:
        return pd.DataFrame(columns=[
            "parent_motif_id", "parent_label", "parent_start", "parent_end",
            "sty_position", "sty_reference_residue", "sty_context_window",
            "phospho_capable_species_count", "phospho_lose_species_count",
            "phospho_loss_species",
        ])

    # Per ungapped reference position, list (species, residue) for every
    # species in the alignment at that aligned column.
    species_at_pos: Dict[int, List[Tuple[str, str]]] = {}
    for pos, col_idx in ref_to_col.items():
        sp_list: List[Tuple[str, str]] = []
        for rec in alignment:
            seq = str(rec.seq)
            if col_idx < len(seq):
                aa = seq[col_idx].upper()
                if aa in GAP_CHARS or aa in {"X", "?"}:
                    continue
                species, _ = parse_header_species_symbol(str(rec.id))
                sp_list.append((species, aa))
        species_at_pos[pos] = sp_list

    track_len = max(ref_to_col)
    phospho_capable = {"S", "T", "Y"}

    anchors: List[Dict[str, Any]] = []
    if library_hits_df is not None and not library_hits_df.empty and "start" in library_hits_df.columns and "end" in library_hits_df.columns:
        for _, row in library_hits_df.iterrows():
            try:
                start = int(row["start"])
                end = int(row["end"])
            except (TypeError, ValueError):
                continue
            anchors.append({
                "kind": "motif",
                "motif_id": str(row.get("motif_id") or ""),
                "label": str(row.get("label") or row.get("motif_name") or ""),
                "start": start,
                "end": end,
            })
    if extra_anchor_positions:
        for pos, label in extra_anchor_positions:
            try:
                p = int(pos)
            except (TypeError, ValueError):
                continue
            anchors.append({
                "kind": "active_site",
                "motif_id": f"active_site__{p}",
                "label": str(label or f"active_site_{p}"),
                "start": p,
                "end": p,
            })

    rows: List[Dict[str, Any]] = []
    for anchor in anchors:
        lo = max(1, anchor["start"] - window)
        hi = min(track_len, anchor["end"] + window)
        for pos in range(lo, hi + 1):
            ref_aa = ref_seq[ref_to_col[pos]] if pos in ref_to_col else "-"
            if ref_aa not in phospho_capable:
                continue
            sp_data = species_at_pos.get(pos, [])
            cap = sum(1 for _, aa in sp_data if aa in phospho_capable)
            loss = [sp for sp, aa in sp_data if aa not in phospho_capable]
            ctx_lo = max(1, pos - 5)
            ctx_hi = min(track_len, pos + 5)
            ctx = "".join(
                ref_seq[ref_to_col[p]] if p in ref_to_col and ref_to_col[p] < len(ref_seq) else "."
                for p in range(ctx_lo, ctx_hi + 1)
            )
            rows.append({
                "parent_motif_id": anchor["motif_id"],
                "parent_label": anchor["label"],
                "parent_start": int(anchor["start"]),
                "parent_end": int(anchor["end"]),
                "sty_position": int(pos),
                "sty_reference_residue": ref_aa,
                "sty_context_window": ctx,
                "phospho_capable_species_count": int(cap),
                "phospho_lose_species_count": int(len(loss)),
                "phospho_loss_species": ";".join(sorted(set(loss))),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "parent_motif_id", "parent_label", "parent_start", "parent_end",
            "sty_position", "sty_reference_residue", "sty_context_window",
            "phospho_capable_species_count", "phospho_lose_species_count",
            "phospho_loss_species",
        ])
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["parent_motif_id", "sty_position"]).reset_index(drop=True)
    return df


def v11_write_phospho_regulatable_outputs(outdir: Path | str,
                                          alignment: Any,
                                          reference_species: str,
                                          library_hits_df: Optional[pd.DataFrame],
                                          extra_anchor_positions: Optional[Sequence[Tuple[int, str]]] = None,
                                          window: int = V11_PHOSPHO_REGULATABLE_WINDOW) -> Path:
    outdir = Path(outdir)
    df = v11_compute_phospho_regulatable_signals(
        alignment, reference_species,
        library_hits_df=library_hits_df,
        extra_anchor_positions=extra_anchor_positions,
        window=window,
    )
    out = outdir / V11_PHOSPHO_REGULATABLE_TSV
    df.to_csv(out, sep="\t", index=False)
    return out


# ----------------- V11.1 Intrinsically Disordered Region (IDR) predictor ----#
# Lightweight, no external dependency. Per residue, combines Wootton-Federhen
# complexity (normalised log of multinomial coefficient over an L-residue
# window) with a disorder-promoting-amino-acid fraction (P,E,S,Q,K,A). IDR
# regions are stretches of consecutive residues with smoothed score above
# threshold over ≥ min_length contiguous positions.
#
# Run PER SPECIES because IDR boundaries can shift between species — a
# zebrafish ortholog can have a longer C-terminal disordered tail that
# contains regulatory motifs missing from the human ortholog.
# -----------------------------------------------------------------------------

V11_IDR_PREDICTIONS_CSV = "v11_idr_predictions_per_species.csv"
V11_IDR_DISORDER_PROMOTING_RESIDUES: frozenset = frozenset("PESQKA")
V11_IDR_WINDOW = 20
V11_IDR_SMOOTHING = 5
V11_IDR_THRESHOLD = 0.55
V11_IDR_MIN_LENGTH = 15


def _v11_wootton_federhen_complexity(window_seq: str) -> float:
    """Normalised Wootton-Federhen complexity K_normalised in [0, 1].

    K = (1/L) * log2(L! / prod(n_i!)) for the standard 20-aa alphabet.
    Maximum K when every residue is different (capped at log2(L) for L≤20,
    log2(20) for L>20). We normalise by the maximum value the metric can
    take for the given window length so the score lives in [0, 1]
    independent of window size.
    """
    L = len(window_seq)
    if L == 0:
        return 0.0
    counts: Dict[str, int] = {}
    for aa in window_seq:
        if aa in {"-", "."}:
            continue
        counts[aa] = counts.get(aa, 0) + 1
    n_eff = sum(counts.values())
    if n_eff <= 1:
        return 0.0
    try:
        log_numer = math.lgamma(n_eff + 1)
        log_denom = sum(math.lgamma(c + 1) for c in counts.values())
    except (OverflowError, ValueError):
        return 0.0
    K = (log_numer - log_denom) / math.log(2.0) / n_eff
    # Theoretical maximum for the window: log2(min(L, 20))
    K_max = math.log2(min(n_eff, 20)) if n_eff > 1 else 1.0
    return float(max(0.0, min(1.0, K / K_max if K_max > 0 else 0.0)))


def _v11_idr_score_for_window(window_seq: str) -> float:
    cleaned = [aa for aa in window_seq if aa not in {"-", "."}]
    if not cleaned:
        return 0.0
    K = _v11_wootton_federhen_complexity("".join(cleaned))
    disorder_frac = sum(1 for aa in cleaned if aa in V11_IDR_DISORDER_PROMOTING_RESIDUES) / len(cleaned)
    return float(0.5 * (1.0 - K) + 0.5 * disorder_frac)


def _v11_per_residue_idr_track(seq_ungapped: str,
                               window: int = V11_IDR_WINDOW,
                               smoothing: int = V11_IDR_SMOOTHING) -> List[float]:
    n = len(seq_ungapped)
    raw: List[float] = []
    half_w = window // 2
    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        raw.append(_v11_idr_score_for_window(seq_ungapped[lo:hi]))
    if smoothing <= 1:
        return raw
    half_s = smoothing // 2
    smoothed: List[float] = []
    for i in range(n):
        lo = max(0, i - half_s)
        hi = min(n, i + half_s + 1)
        smoothed.append(sum(raw[lo:hi]) / (hi - lo))
    return smoothed


def _v11_call_idrs(track: Sequence[float],
                   threshold: float = V11_IDR_THRESHOLD,
                   min_length: int = V11_IDR_MIN_LENGTH) -> List[Tuple[int, int, float]]:
    """Walk the track and return [(start_1b, end_1b_inclusive, mean_score), ...]
    for contiguous runs above threshold of length >= min_length."""
    spans: List[Tuple[int, int, float]] = []
    run_start: Optional[int] = None
    run_scores: List[float] = []
    for i, v in enumerate(track):
        if v >= threshold:
            if run_start is None:
                run_start = i
                run_scores = [v]
            else:
                run_scores.append(v)
        else:
            if run_start is not None and (i - run_start) >= min_length:
                mean_score = sum(run_scores) / len(run_scores)
                spans.append((run_start + 1, i, float(mean_score)))
            run_start = None
            run_scores = []
    if run_start is not None and (len(track) - run_start) >= min_length:
        mean_score = sum(run_scores) / len(run_scores)
        spans.append((run_start + 1, len(track), float(mean_score)))
    return spans


def v11_compute_idr_per_species(alignment: Any,
                                library_hits_df: Optional[pd.DataFrame] = None,
                                window: int = V11_IDR_WINDOW,
                                smoothing: int = V11_IDR_SMOOTHING,
                                threshold: float = V11_IDR_THRESHOLD,
                                min_length: int = V11_IDR_MIN_LENGTH) -> pd.DataFrame:
    """One row per IDR call per species: species, start, end, length,
    mean_idr_score, contains_motifs (semicolon-joined motif_ids overlapping
    the region, only relevant for the reference species since motif hits
    use reference coords).
    """
    motif_intervals: List[Tuple[int, int, str]] = []
    if library_hits_df is not None and not library_hits_df.empty:
        for _, row in library_hits_df.iterrows():
            try:
                s = int(row["start"])
                e = int(row["end"])
            except (TypeError, ValueError):
                continue
            motif_intervals.append((s, e, str(row.get("motif_id") or row.get("label") or "")))

    rows: List[Dict[str, Any]] = []
    for rec in alignment:
        seq = "".join(aa for aa in str(rec.seq).upper() if aa not in GAP_CHARS and aa not in {"X", "?"})
        if not seq:
            continue
        species, _ = parse_header_species_symbol(str(rec.id))
        track = _v11_per_residue_idr_track(seq, window=window, smoothing=smoothing)
        for start, end, mean_score in _v11_call_idrs(track, threshold=threshold, min_length=min_length):
            overlaps: List[str] = []
            for ms, me, mid in motif_intervals:
                if me >= start and ms <= end:
                    overlaps.append(mid)
            rows.append({
                "species": species,
                "record_id": str(rec.id),
                "start": int(start),
                "end": int(end),
                "length": int(end - start + 1),
                "mean_idr_score": float(mean_score),
                "contains_motifs": ";".join(overlaps),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "species", "record_id", "start", "end", "length",
            "mean_idr_score", "contains_motifs",
        ])
    return pd.DataFrame(rows)


def v11_write_idr_predictions(outdir: Path | str,
                              alignment: Any,
                              library_hits_df: Optional[pd.DataFrame] = None,
                              window: int = V11_IDR_WINDOW,
                              smoothing: int = V11_IDR_SMOOTHING,
                              threshold: float = V11_IDR_THRESHOLD,
                              min_length: int = V11_IDR_MIN_LENGTH) -> Path:
    outdir = Path(outdir)
    df = v11_compute_idr_per_species(
        alignment,
        library_hits_df=library_hits_df,
        window=window, smoothing=smoothing,
        threshold=threshold, min_length=min_length,
    )
    out = outdir / V11_IDR_PREDICTIONS_CSV
    df.to_csv(out, index=False)
    return out


# ----------------- V11.1 AlphaFold pocket detector --------------------------#
# Lightweight Cα-packing-based pocket scoring (no FreeSASA dependency).
# For every residue in the human reference AlphaFold model, count the number
# of Cα atoms within 10 Å (`packing_density`). Classify:
#   core           : packing_density ≥ 14 AND pLDDT ≥ 70
#   pocket-lining  : 9 ≤ packing_density ≤ 13 AND adjacent to ≥1 core residue
#   surface        : packing_density < 9
#   buried-static  : packing_density ≥ 14 BUT pLDDT < 70 (low confidence)
# Also flags `near_active_site` if any active-site residue lies within 8 Å.
# -----------------------------------------------------------------------------

V11_POCKET_RESIDUES_CSV = "v11_pocket_residues.csv"
V11_POCKET_CORE_THRESHOLD = 14
V11_POCKET_LINING_MIN = 9
V11_POCKET_LINING_MAX = 13
V11_POCKET_NEIGHBOUR_RADIUS = 10.0
V11_POCKET_ACTIVE_SITE_RADIUS = 8.0


def _v11_parse_pdb_ca_atoms(pdb_text: str) -> List[Dict[str, Any]]:
    """Parse Cα ATOM records from an AlphaFold PDB. Returns a list of
    {resi, resn, aa, x, y, z, plddt} dicts. Skips alternate locations
    other than the empty / 'A' altloc."""
    aa3_to_1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
        "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
        "TYR": "Y", "VAL": "V",
    }
    atoms: List[Dict[str, Any]] = []
    for line in (pdb_text or "").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        altloc = line[16]
        if altloc not in (" ", "A"):
            continue
        try:
            resi = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        try:
            plddt = float(line[60:66])
        except ValueError:
            plddt = float("nan")
        resn = line[17:20].strip()
        atoms.append({
            "resi": resi,
            "resn": resn,
            "aa": aa3_to_1.get(resn, "X"),
            "x": x, "y": y, "z": z,
            "plddt": plddt,
        })
    return atoms


def v11_compute_pocket_residues(pdb_text: str,
                                active_site_positions: Optional[Sequence[int]] = None,
                                neighbour_radius: float = V11_POCKET_NEIGHBOUR_RADIUS,
                                active_site_radius: float = V11_POCKET_ACTIVE_SITE_RADIUS,
                                core_threshold: int = V11_POCKET_CORE_THRESHOLD,
                                lining_min: int = V11_POCKET_LINING_MIN,
                                lining_max: int = V11_POCKET_LINING_MAX) -> pd.DataFrame:
    """Return one row per residue: residue_number, residue_aa, packing_density,
    classification ∈ {core, pocket_lining, surface, buried_static}, plddt,
    near_active_site (bool), distance_to_active_site (Å)."""
    atoms = _v11_parse_pdb_ca_atoms(pdb_text)
    if not atoms:
        return pd.DataFrame(columns=[
            "residue_number", "residue_aa", "packing_density", "classification",
            "plddt", "near_active_site", "distance_to_active_site",
        ])

    n = len(atoms)
    r2 = neighbour_radius * neighbour_radius
    asr2 = active_site_radius * active_site_radius
    active_set = set(int(p) for p in (active_site_positions or []))
    active_atoms = [a for a in atoms if a["resi"] in active_set]

    densities: List[int] = [0] * n
    for i in range(n):
        ai = atoms[i]
        d = 0
        for j in range(n):
            if i == j:
                continue
            aj = atoms[j]
            dx = ai["x"] - aj["x"]
            dy = ai["y"] - aj["y"]
            dz = ai["z"] - aj["z"]
            if dx * dx + dy * dy + dz * dz <= r2:
                d += 1
        densities[i] = d

    distances_to_active: List[float] = []
    near_flags: List[bool] = []
    for ai in atoms:
        if not active_atoms:
            distances_to_active.append(float("nan"))
            near_flags.append(False)
            continue
        best = float("inf")
        for aj in active_atoms:
            dx = ai["x"] - aj["x"]
            dy = ai["y"] - aj["y"]
            dz = ai["z"] - aj["z"]
            best = min(best, dx * dx + dy * dy + dz * dz)
        d = math.sqrt(best) if best != float("inf") else float("nan")
        distances_to_active.append(d)
        near_flags.append(d <= active_site_radius)

    classes: List[str] = []
    core_set: set = set()
    for i, ai in enumerate(atoms):
        plddt = ai["plddt"]
        if densities[i] >= core_threshold:
            if (plddt != plddt) or plddt >= 70.0:
                classes.append("core")
                core_set.add(i)
            else:
                classes.append("buried_static")
        elif lining_min <= densities[i] <= lining_max:
            classes.append("pocket_lining_candidate")
        else:
            classes.append("surface")

    # Tighten "pocket_lining_candidate" → "pocket_lining" only if adjacent to
    # at least one core residue along the sequence ±3 OR within neighbour_radius.
    core_idx_set = core_set
    for i in range(n):
        if classes[i] != "pocket_lining_candidate":
            continue
        is_lining = False
        for j in core_idx_set:
            if abs(atoms[i]["resi"] - atoms[j]["resi"]) <= 3:
                is_lining = True
                break
            dx = atoms[i]["x"] - atoms[j]["x"]
            dy = atoms[i]["y"] - atoms[j]["y"]
            dz = atoms[i]["z"] - atoms[j]["z"]
            if dx * dx + dy * dy + dz * dz <= r2:
                is_lining = True
                break
        classes[i] = "pocket_lining" if is_lining else "surface"

    # V11.1 SASA proxy: relative burial = packing_density / max(observed
    # packing). Higher = more buried; lower = more surface-exposed.
    max_density = max(densities) if densities else 1
    rel_burial = [d / max_density if max_density > 0 else 0.0 for d in densities]

    rows: List[Dict[str, Any]] = []
    for i, atom in enumerate(atoms):
        # Primary headline classification — what residues LINE the substrate
        # pocket. SDR enzymes (DHRS7) and most globular enzymes have the
        # catalytic site at the bottom of a deep cavity, so the "pocket
        # residues" are simply the residues sitting within the active-site
        # neighbourhood. Packing density / pLDDT remain as secondary
        # descriptors for interpreting why a given pocket residue is
        # exposed (surface entrance) vs buried (core of the pocket).
        is_substrate_pocket = bool(near_flags[i])
        rows.append({
            "residue_number": int(atom["resi"]),
            "residue_aa": atom["aa"],
            "is_substrate_pocket": is_substrate_pocket,
            "distance_to_active_site": float(distances_to_active[i]) if distances_to_active[i] == distances_to_active[i] else float("nan"),
            "packing_density": int(densities[i]),
            "relative_burial": float(rel_burial[i]),
            "packing_classification": classes[i],
            "plddt": float(atom["plddt"]) if atom["plddt"] == atom["plddt"] else float("nan"),
            "near_active_site": bool(near_flags[i]),  # alias kept for back-compat
        })
    return pd.DataFrame(rows)


def v11_write_pocket_residues(outdir: Path | str,
                              active_site_positions: Optional[Sequence[int]] = None) -> Optional[Path]:
    """Wrapper: read the human AlphaFold PDB from outdir and write the
    pocket-residue CSV. Returns the path or None if the PDB isn't there."""
    outdir = Path(outdir)
    pdb_path = outdir / ALPHAFOLD_MODEL_FILENAME
    if not pdb_path.exists():
        # The pipeline may have stored the model under an alternate filename
        # captured in the metadata.
        meta_path = outdir / ALPHAFOLD_METADATA_FILENAME
        meta = read_json_artifact(meta_path) if meta_path.exists() else {}
        alt = (meta or {}).get("model_filename") if isinstance(meta, dict) else None
        if alt and (outdir / alt).exists():
            pdb_path = outdir / alt
    if not pdb_path.exists():
        return None
    pdb_text = pdb_path.read_text(encoding="utf-8")
    df = v11_compute_pocket_residues(pdb_text, active_site_positions=active_site_positions)
    if df.empty:
        return None
    out = outdir / V11_POCKET_RESIDUES_CSV
    df.to_csv(out, index=False)
    return out


# ----------------- V11.1 Pairwise species diff framework --------------------#
# User picks two species (`--diff_species A,B`); emits a per-position table
# of where the two orthologs differ, with structural / regulatory / SDP
# context cross-referenced. The "print and stare at it" view for a bench
# scientist comparing e.g. mouse vs human DHRS7.
# -----------------------------------------------------------------------------

V11_SPECIES_DIFF_CSV_TEMPLATE = "v11_species_diff_{species_a}_vs_{species_b}.csv"
V11_SPECIES_DIFF_HTML_TEMPLATE = "v11_species_diff_{species_a}_vs_{species_b}.html"


def v11_compute_species_diff(alignment: Any,
                             reference_species: str,
                             species_a: str,
                             species_b: str,
                             sdp_df: Optional[pd.DataFrame] = None,
                             pocket_df: Optional[pd.DataFrame] = None,
                             library_hits_df: Optional[pd.DataFrame] = None,
                             active_site_positions: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Per ungapped reference position, return aa_A, aa_B, property class
    change, and (when provided) sdp_score, in_substrate_pocket,
    distance_to_active_site, overlapping_motif_ids, near_active_site.
    """
    try:
        from gene_phylo_conservation_pipeline import AA_GROUP_SCHEMES  # type: ignore
        prop_scheme = AA_GROUP_SCHEMES.get("charge", {})
    except Exception:  # noqa: BLE001
        prop_scheme = {
            "positive": set("KRH"),
            "negative": set("DE"),
            "neutral": set("AVILMFWYCNQSTPG"),
        }

    aln_to_ref, ref_record = _v11_ref_position_records(alignment, reference_species)
    ref_seq = str(ref_record.seq).upper()

    rec_a = None
    rec_b = None
    a_key = (species_a or "").strip().lower()
    b_key = (species_b or "").strip().lower()
    for rec in alignment:
        species, _ = parse_header_species_symbol(str(rec.id))
        sp_lower = species.lower()
        if rec_a is None and sp_lower == a_key:
            rec_a = rec
        if rec_b is None and sp_lower == b_key:
            rec_b = rec
        if rec_a is not None and rec_b is not None:
            break
    if rec_a is None or rec_b is None:
        return pd.DataFrame(columns=[
            "reference_position", "reference_residue",
            "species_a", "aa_a", "species_b", "aa_b",
            "differs", "property_class_a", "property_class_b",
            "property_class_changed", "sdp_score",
            "is_substrate_pocket", "distance_to_active_site",
            "near_active_site", "overlapping_motif_ids",
        ])

    seq_a = str(rec_a.seq).upper()
    seq_b = str(rec_b.seq).upper()

    sdp_by_pos: Dict[int, float] = {}
    if sdp_df is not None and not sdp_df.empty and "reference_position" in sdp_df.columns:
        for _, r in sdp_df.iterrows():
            try:
                sdp_by_pos[int(r["reference_position"])] = float(r.get("sdp_score") or 0.0)
            except (TypeError, ValueError):
                continue
    pocket_by_pos: Dict[int, Dict[str, Any]] = {}
    if pocket_df is not None and not pocket_df.empty and "residue_number" in pocket_df.columns:
        for _, r in pocket_df.iterrows():
            try:
                pocket_by_pos[int(r["residue_number"])] = dict(r)
            except (TypeError, ValueError):
                continue
    motif_intervals: List[Tuple[int, int, str]] = []
    if library_hits_df is not None and not library_hits_df.empty:
        for _, r in library_hits_df.iterrows():
            try:
                motif_intervals.append((int(r["start"]), int(r["end"]),
                                        str(r.get("motif_id") or r.get("label") or "")))
            except (TypeError, ValueError):
                continue
    active_set = set(int(p) for p in (active_site_positions or []))

    rows: List[Dict[str, Any]] = []
    for col_idx, ref_pos in enumerate(aln_to_ref):
        if ref_pos is None:
            continue
        ref_aa = ref_seq[col_idx] if col_idx < len(ref_seq) else "-"
        aa_a = seq_a[col_idx].upper() if col_idx < len(seq_a) else "-"
        aa_b = seq_b[col_idx].upper() if col_idx < len(seq_b) else "-"
        differs = (aa_a != aa_b) and aa_a not in {"-", "X", "?"} and aa_b not in {"-", "X", "?"}
        prop_a = _v11_property_class_for_aa(aa_a, prop_scheme) or ""
        prop_b = _v11_property_class_for_aa(aa_b, prop_scheme) or ""
        prop_changed = bool(prop_a and prop_b and prop_a != prop_b)
        sdp_score = float(sdp_by_pos.get(int(ref_pos), 0.0))
        pocket_row = pocket_by_pos.get(int(ref_pos)) or {}
        is_pocket = bool(pocket_row.get("is_substrate_pocket", False)) if pocket_row else False
        d_active = pocket_row.get("distance_to_active_site")
        try:
            d_active_f = float(d_active) if d_active is not None and d_active == d_active else float("nan")
        except (TypeError, ValueError):
            d_active_f = float("nan")
        near_active = int(ref_pos) in active_set
        overlapping = [mid for ms, me, mid in motif_intervals if ms <= int(ref_pos) <= me]
        rows.append({
            "reference_position": int(ref_pos),
            "reference_residue": ref_aa,
            "species_a": species_a,
            "aa_a": aa_a,
            "species_b": species_b,
            "aa_b": aa_b,
            "differs": bool(differs),
            "property_class_a": prop_a,
            "property_class_b": prop_b,
            "property_class_changed": prop_changed,
            "sdp_score": sdp_score,
            "is_substrate_pocket": is_pocket,
            "distance_to_active_site": d_active_f,
            "near_active_site": near_active,
            "overlapping_motif_ids": ";".join(overlapping),
        })
    return pd.DataFrame(rows)


def v11_write_species_diff_outputs(outdir: Path | str,
                                   alignment: Any,
                                   reference_species: str,
                                   species_a: str,
                                   species_b: str,
                                   sdp_df: Optional[pd.DataFrame] = None,
                                   pocket_df: Optional[pd.DataFrame] = None,
                                   library_hits_df: Optional[pd.DataFrame] = None,
                                   active_site_positions: Optional[Sequence[int]] = None,
                                   gene_label: str = "") -> Optional[Path]:
    """Write the per-position diff CSV + a self-contained scannable HTML
    summary (table + counts; structure context links out to the existing
    v11_structure_overlay_combined.html). Returns the HTML path."""
    outdir = Path(outdir)
    df = v11_compute_species_diff(
        alignment, reference_species,
        species_a, species_b,
        sdp_df=sdp_df, pocket_df=pocket_df,
        library_hits_df=library_hits_df,
        active_site_positions=active_site_positions,
    )
    if df.empty:
        return None
    safe_a = _v11_re.sub(r"[^A-Za-z0-9_-]+", "_", species_a).strip("_") or "A"
    safe_b = _v11_re.sub(r"[^A-Za-z0-9_-]+", "_", species_b).strip("_") or "B"
    csv_out = outdir / V11_SPECIES_DIFF_CSV_TEMPLATE.format(species_a=safe_a, species_b=safe_b)
    df.to_csv(csv_out, index=False)

    # Build a scannable HTML — counts header + sortable table of differing
    # positions only (with optional pocket / SDP / motif highlights).
    diffs = df[df["differs"]].copy()
    diffs = diffs.sort_values(["is_substrate_pocket", "sdp_score", "property_class_changed"],
                               ascending=[False, False, False])
    n_total = int((df["reference_residue"] != "-").sum())
    n_diff = int(len(diffs))
    n_prop = int(diffs["property_class_changed"].sum())
    n_pocket = int(diffs["is_substrate_pocket"].sum())
    n_motif = int((diffs["overlapping_motif_ids"].astype(str).str.len() > 0).sum())
    n_near = int(diffs["near_active_site"].sum())

    def _row_cells(r: pd.Series) -> str:
        bg = ""
        if r["is_substrate_pocket"]:
            bg = "#fef3c7"
        elif r["near_active_site"]:
            bg = "#fde68a"
        elif r["property_class_changed"]:
            bg = "#fee2e2"
        elif r["sdp_score"] >= 0.7:
            bg = "#e0f2fe"
        cells = [
            f'<td>{int(r["reference_position"])}</td>',
            f'<td><b>{escape(str(r["reference_residue"]))}</b></td>',
            f'<td>{escape(str(r["aa_a"]))}</td>',
            f'<td>{escape(str(r["aa_b"]))}</td>',
            f'<td>{escape(str(r.get("property_class_a") or ""))}</td>',
            f'<td>{escape(str(r.get("property_class_b") or ""))}</td>',
            f'<td>{float(r.get("sdp_score") or 0):.3f}</td>',
            f'<td>{"✓" if r["is_substrate_pocket"] else ""}</td>',
            (f'<td>{float(r["distance_to_active_site"]):.2f}</td>'
             if r["distance_to_active_site"] == r["distance_to_active_site"] else "<td></td>"),
            f'<td>{escape(str(r.get("overlapping_motif_ids") or ""))}</td>',
        ]
        return f'<tr style="background:{bg};">{"".join(cells)}</tr>'

    table_rows = "\n".join(_row_cells(r) for _, r in diffs.iterrows())
    html_path = outdir / V11_SPECIES_DIFF_HTML_TEMPLATE.format(species_a=safe_a, species_b=safe_b)
    html_path.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>{escape(gene_label or 'gene')} — {escape(species_a)} vs {escape(species_b)} diff</title>
<style>
 body{{font:14px/1.45 'Segoe UI',sans-serif;margin:24px;color:#1f2937;background:#f8fafc;}}
 h1{{margin:0 0 6px 0;font-size:20px;}} .muted{{color:#6b7280;font-size:13px;}}
 .counts{{display:flex;gap:14px;margin:18px 0;flex-wrap:wrap;}}
 .counts .pill{{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:8px 12px;}}
 .counts b{{display:block;font-size:18px;color:#0f172a;}}
 table{{border-collapse:collapse;background:#fff;width:100%;font-size:13px;border:1px solid #e5e7eb;}}
 th{{background:#1f2937;color:#fff;text-align:left;padding:8px;position:sticky;top:0;}}
 td{{padding:6px 8px;border-top:1px solid #f3f4f6;font-family:Consolas,monospace;}}
 .legend{{display:flex;gap:10px;margin:10px 0;font-size:12px;}}
 .legend span{{padding:2px 8px;border-radius:4px;}}
</style></head>
<body>
<h1>{escape(gene_label or 'gene')} — {escape(species_a)} vs {escape(species_b)}</h1>
<p class="muted">Per ungapped reference position differences between the two orthologs, sorted with substrate-pocket residues first, then by SDP score and property class changes.</p>
<div class="counts">
  <div class="pill"><b>{n_total}</b>reference residues</div>
  <div class="pill"><b>{n_diff}</b>positions differ</div>
  <div class="pill"><b>{n_prop}</b>property-class change</div>
  <div class="pill"><b>{n_pocket}</b>substrate-pocket</div>
  <div class="pill"><b>{n_motif}</b>in regulatory motif</div>
  <div class="pill"><b>{n_near}</b>at active site</div>
</div>
<div class="legend">
  <span style="background:#fef3c7;">substrate pocket</span>
  <span style="background:#fde68a;">active site</span>
  <span style="background:#fee2e2;">property-class change</span>
  <span style="background:#e0f2fe;">high SDP score (≥0.70)</span>
</div>
<table>
<thead><tr>
 <th>pos</th><th>ref</th><th>{escape(species_a)}</th><th>{escape(species_b)}</th>
 <th>class A</th><th>class B</th><th>SDP</th><th>pocket</th><th>Å to active site</th><th>motifs</th>
</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
<p class="muted" style="margin-top:18px;">For 3D context open <code>v11_structure_overlay_combined.html</code> in this output folder and toggle "Fish-conserved / rodent-diverged residues" or paste a position into the residue input.</p>
</body></html>""", encoding="utf-8")
    return html_path


def v11_compute_per_clade_consensus(alignment: Any,
                                    taxonomy_lookup: Optional[Dict[str, Any]],
                                    reference_species: str) -> pd.DataFrame:
    """Per ungapped reference position, return per-clade consensus residue
    + consensus fraction + count of agreeing residues + total non-gap count.
    Wide: reference_ungapped_position, consensus_<clade>, consensus_frac_<clade>."""
    aln_to_ref, ref_record = _v11_ref_position_records(alignment, reference_species)
    species_clade = _v11_species_clade_map(alignment, taxonomy_lookup)
    clade_records: Dict[str, List[Any]] = {}
    for record in alignment:
        clade = species_clade.get(str(record.id), "Unassigned")
        clade_records.setdefault(clade, []).append(record)
    ref_seq = str(ref_record.seq).upper()

    rows: List[Dict[str, Any]] = []
    for col_idx, ref_pos in enumerate(aln_to_ref):
        if ref_pos is None:
            continue
        row: Dict[str, Any] = {
            "reference_ungapped_position": int(ref_pos),
            "reference_residue": ref_seq[col_idx] if col_idx < len(ref_seq) else "-",
        }
        for clade, records in clade_records.items():
            counts: _V11_Counter = _V11_Counter()
            for rec in records:
                seq = str(rec.seq)
                if col_idx < len(seq):
                    aa = seq[col_idx].upper()
                    if aa not in GAP_CHARS:
                        counts[aa] += 1
            total = sum(counts.values())
            if total > 0:
                best_aa, best_n = counts.most_common(1)[0]
            else:
                best_aa, best_n = "-", 0
            row[f"consensus_{clade}"] = best_aa
            row[f"consensus_frac_{clade}"] = (best_n / total) if total > 0 else 0.0
            row[f"consensus_count_{clade}"] = int(best_n)
            row[f"total_count_{clade}"] = int(total)
        rows.append(row)
    return pd.DataFrame(rows)


def _v11_jsd(p: List[float], q: List[float]) -> float:
    """Jensen-Shannon divergence in bits. p, q must be same-length probability
    vectors. Returns 0..1 (we report sqrt to get distance in [0, 1])."""
    if not p or not q or len(p) != len(q):
        return 0.0
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    def _kld(a: List[float], b: List[float]) -> float:
        out = 0.0
        for ai, bi in zip(a, b):
            if ai > 0 and bi > 0:
                out += ai * math.log2(ai / bi)
        return out
    jsd = 0.5 * _kld(p, m) + 0.5 * _kld(q, m)
    # Convert to [0, 1] by taking sqrt of the bit-valued JSD (clamped).
    return float(max(0.0, min(1.0, math.sqrt(max(0.0, jsd)))))


def v11_compute_clade_pair_js_divergence(alignment: Any,
                                         taxonomy_lookup: Optional[Dict[str, Any]],
                                         reference_species: str,
                                         min_clade_count: int = 3) -> pd.DataFrame:
    """For every pair of broad clades, compute Jensen-Shannon divergence
    between per-clade residue frequency vectors at every reference position.
    Long format. Cite Capra & Singh, Bioinformatics 2007.

    Pairs with either clade having < `min_clade_count` non-gap residues at a
    position are skipped (under-sampled noise)."""
    aln_to_ref, ref_record = _v11_ref_position_records(alignment, reference_species)
    species_clade = _v11_species_clade_map(alignment, taxonomy_lookup)
    clade_records: Dict[str, List[Any]] = {}
    for record in alignment:
        clade = species_clade.get(str(record.id), "Unassigned")
        clade_records.setdefault(clade, []).append(record)
    clades_sorted = sorted(clade_records.keys())

    def _freq_vector(records: Sequence[Any], col_idx: int) -> Tuple[List[float], int]:
        counts: _V11_Counter = _V11_Counter()
        for rec in records:
            seq = str(rec.seq)
            if col_idx < len(seq):
                aa = seq[col_idx].upper()
                if aa not in GAP_CHARS:
                    counts[aa] += 1
        total = sum(counts.values())
        if total == 0:
            return [0.0] * len(_V11_AA_ALPHABET), 0
        return [counts.get(aa, 0) / total for aa in _V11_AA_ALPHABET], total

    rows: List[Dict[str, Any]] = []
    for col_idx, ref_pos in enumerate(aln_to_ref):
        if ref_pos is None:
            continue
        freq_cache: Dict[str, Tuple[List[float], int]] = {}
        for clade in clades_sorted:
            freq_cache[clade] = _freq_vector(clade_records[clade], col_idx)
        for i, clade_a in enumerate(clades_sorted):
            pa, na = freq_cache[clade_a]
            if na < min_clade_count:
                continue
            for clade_b in clades_sorted[i + 1:]:
                pb, nb = freq_cache[clade_b]
                if nb < min_clade_count:
                    continue
                jsd = _v11_jsd(pa, pb)
                rows.append({
                    "reference_ungapped_position": int(ref_pos),
                    "clade_a": clade_a,
                    "clade_b": clade_b,
                    "n_a": int(na),
                    "n_b": int(nb),
                    "js_divergence": jsd,
                })
    return pd.DataFrame(rows)


def v11_compute_lineage_stabilization(entropy_df: pd.DataFrame,
                                      ancestral_clades: Sequence[str] = V11_DEFAULT_ANCESTRAL_CLADES,
                                      derived_clades: Sequence[str] = V11_DEFAULT_DERIVED_CLADES,
                                      min_occupancy: int = 3) -> pd.DataFrame:
    """Stabilization score per reference position:

        score = H_ancestral - H_derived

    where H_ancestral is the occupancy-weighted mean entropy across the
    ancestral clade bucket and H_derived likewise for the derived bucket.
    Positive scores → position is more variable in ancestral clades and
    has fixed in derived clades (the "stabilization" signal).

    This is a simplified, entropy-only approximation of Gu's Type I
    functional-divergence test (Gu, MBE 1999; Gu, MBE 2006). The full DIVERGE
    framework uses ML estimation of θ-I; we use entropy because it requires
    no tree or branch lengths and remains interpretable from the alignment
    alone. Positions are reported only when both buckets have at least
    `min_occupancy` residues."""
    if entropy_df.empty:
        return entropy_df

    def _weighted_entropy(row: pd.Series, clades: Sequence[str]) -> Tuple[float, int]:
        h_sum = 0.0
        n_sum = 0
        for clade in clades:
            h_col = f"entropy_{clade}"
            n_col = f"occupancy_{clade}"
            if h_col not in row or n_col not in row:
                continue
            n = int(row[n_col] or 0)
            if n <= 0:
                continue
            h_sum += float(row[h_col]) * n
            n_sum += n
        if n_sum == 0:
            return float("nan"), 0
        return h_sum / n_sum, n_sum

    rows: List[Dict[str, Any]] = []
    for _, row in entropy_df.iterrows():
        h_anc, n_anc = _weighted_entropy(row, ancestral_clades)
        h_der, n_der = _weighted_entropy(row, derived_clades)
        if n_anc < min_occupancy or n_der < min_occupancy:
            score = float("nan")
        else:
            score = float(h_anc - h_der)
        rows.append({
            "reference_ungapped_position": int(row["reference_ungapped_position"]),
            "reference_residue": row.get("reference_residue", ""),
            "ancestral_clades": "|".join(ancestral_clades),
            "derived_clades": "|".join(derived_clades),
            "ancestral_entropy": h_anc,
            "ancestral_n": n_anc,
            "derived_entropy": h_der,
            "derived_n": n_der,
            "stabilization_score": score,
        })
    return pd.DataFrame(rows)


# --------------------- motif evolution per species/clade ---------------------#

def _v11_alignment_columns_for_ref_range(aln_to_ref: List[Optional[int]],
                                         start_1b: int, end_1b: int) -> List[int]:
    """Return the alignment-column indices (0-based) that map to ref positions
    in [start_1b, end_1b] inclusive."""
    return [i for i, rp in enumerate(aln_to_ref)
            if rp is not None and start_1b <= rp <= end_1b]


def v11_compute_motif_evolution(alignment: Any,
                                motifs_df: pd.DataFrame,
                                reference_species: str,
                                taxonomy_lookup: Optional[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-species table, per-clade table) for the given motifs.

    per-species columns:
        motif_id, label, start, end, species, record_id, clade,
        motif_residues, identity_to_reference, identical_count, length, classification
    per-clade columns:
        motif_id, label, start, end, clade, n_species, consensus_motif,
        reference_motif, fraction_matching_reference, dominant_alternative
    """
    if motifs_df.empty:
        return (pd.DataFrame(columns=[
                    "motif_id", "label", "start", "end", "species", "record_id",
                    "clade", "motif_residues", "identity_to_reference",
                    "identical_count", "length", "classification"]),
                pd.DataFrame(columns=[
                    "motif_id", "label", "start", "end", "clade", "n_species",
                    "consensus_motif", "reference_motif",
                    "fraction_matching_reference", "dominant_alternative"]))

    aln_to_ref, ref_record = _v11_ref_position_records(alignment, reference_species)
    species_clade = _v11_species_clade_map(alignment, taxonomy_lookup)

    per_species_rows: List[Dict[str, Any]] = []
    per_clade_rows: List[Dict[str, Any]] = []

    for _, motif in motifs_df.iterrows():
        start = int(motif["start"])
        end = int(motif["end"])
        motif_id = str(motif["motif_id"])
        label = str(motif.get("label", motif_id))
        col_idxs = _v11_alignment_columns_for_ref_range(aln_to_ref, start, end)
        if not col_idxs:
            continue
        ref_motif = "".join(
            (str(ref_record.seq)[c].upper() if c < len(str(ref_record.seq)) else "-")
            for c in col_idxs
        )
        # Per-species residues at motif columns.
        for record in alignment:
            species, _ = parse_header_species_symbol(record.id)
            seq = str(record.seq)
            residues = "".join(
                (seq[c].upper() if c < len(seq) else "-")
                for c in col_idxs
            )
            # Identity vs ref, gap-aware.
            identical = sum(1 for r, m in zip(ref_motif, residues)
                            if r not in GAP_CHARS and m not in GAP_CHARS and r == m)
            comparable = sum(1 for r, m in zip(ref_motif, residues)
                             if r not in GAP_CHARS and m not in GAP_CHARS)
            length = len(ref_motif)
            if comparable == 0:
                identity = 0.0
                classification = "absent"
            else:
                identity = identical / comparable
                if identity >= 0.9:
                    classification = "preserved"
                elif identity >= 0.5:
                    classification = "altered"
                else:
                    classification = "lost"
            per_species_rows.append({
                "motif_id": motif_id,
                "label": label,
                "start": start,
                "end": end,
                "species": species,
                "record_id": str(record.id),
                "clade": species_clade.get(str(record.id), "Unassigned"),
                "motif_residues": residues,
                "reference_motif": ref_motif,
                "identical_count": int(identical),
                "comparable_positions": int(comparable),
                "length": int(length),
                "identity_to_reference": float(identity),
                "classification": classification,
            })

        # Per-clade aggregation.
        per_clade_groups: Dict[str, List[str]] = {}
        for row in per_species_rows:
            if row["motif_id"] != motif_id:
                continue
            per_clade_groups.setdefault(row["clade"], []).append(row["motif_residues"])
        for clade, motifs_in_clade in per_clade_groups.items():
            counts = _V11_Counter(motifs_in_clade)
            (best_motif, best_n) = counts.most_common(1)[0] if counts else ("-", 0)
            n_species = len(motifs_in_clade)
            n_match_ref = sum(1 for m in motifs_in_clade if m == ref_motif)
            frac_match_ref = n_match_ref / n_species if n_species else 0.0
            second = counts.most_common(2)[1] if len(counts) > 1 else None
            dominant_alt = second[0] if (second and second[0] != ref_motif) else best_motif if best_motif != ref_motif else ""
            per_clade_rows.append({
                "motif_id": motif_id,
                "label": label,
                "start": start,
                "end": end,
                "clade": clade,
                "n_species": int(n_species),
                "consensus_motif": best_motif,
                "consensus_count": int(best_n),
                "reference_motif": ref_motif,
                "fraction_matching_reference": float(frac_match_ref),
                "dominant_alternative": dominant_alt,
            })

    per_species_df = pd.DataFrame(per_species_rows)
    per_clade_df = pd.DataFrame(per_clade_rows)
    return per_species_df, per_clade_df


# ------------------------- plotting helpers ----------------------------------#

# Coarse amino-acid colour scheme for the motif figures (Lesk-style).
_V11_AA_COLORS: Dict[str, str] = {
    # Hydrophobic (orange)
    "A": "#f97316", "V": "#f97316", "I": "#f97316", "L": "#f97316", "M": "#f97316",
    # Aromatic (red)
    "F": "#dc2626", "W": "#dc2626", "Y": "#dc2626",
    # Polar (green)
    "S": "#16a34a", "T": "#16a34a", "N": "#16a34a", "Q": "#16a34a",
    # Positive (blue)
    "K": "#2563eb", "R": "#2563eb", "H": "#0891b2",
    # Negative (purple)
    "D": "#7c3aed", "E": "#7c3aed",
    # Special
    "C": "#facc15", "G": "#94a3b8", "P": "#ec4899",
    "-": "#e5e7eb", "X": "#9ca3af",
}


def _v11_aa_color(aa: str) -> str:
    return _V11_AA_COLORS.get(str(aa).upper(), "#9ca3af")


def v11_plot_motif_clade_logos(per_species_df: pd.DataFrame,
                               motif_row: pd.Series,
                               outpath_png: Path,
                               outpath_svg: Path) -> None:
    """For one motif, render a small panel of per-clade sequence logos
    (matplotlib-only; pseudo-logo built from rectangles whose height is
    information content × frequency, coloured by amino-acid class).

    Layout: one row per clade, motif positions along x. Reference motif is
    shown on top in plain text for visual anchoring."""
    motif_id = str(motif_row["motif_id"])
    label = str(motif_row.get("label", motif_id))
    motif_data = per_species_df[per_species_df["motif_id"] == motif_id]
    if motif_data.empty:
        return
    motif_len = int(motif_data.iloc[0]["length"])
    ref_motif = str(motif_data.iloc[0]["reference_motif"])
    clades = sorted(motif_data["clade"].unique())
    fig_w = max(6.0, 0.7 * motif_len + 3.5)
    fig_h = max(3.5, 0.7 * len(clades) + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for row_idx, clade in enumerate(clades):
        clade_motifs = list(motif_data[motif_data["clade"] == clade]["motif_residues"])
        n = len(clade_motifs)
        if n == 0:
            continue
        # Frequency per position.
        for pos_idx in range(motif_len):
            col_chars = [m[pos_idx] if pos_idx < len(m) else "-" for m in clade_motifs]
            counts: _V11_Counter = _V11_Counter(col_chars)
            total = sum(c for aa, c in counts.items() if aa not in GAP_CHARS)
            if total == 0:
                continue
            # Information content (max 2 bits / log2(20)≈4.32 ; cap at 1 for display).
            ic = 0.0
            for aa, c in counts.items():
                if aa in GAP_CHARS or c == 0:
                    continue
                p = c / total
                ic += p * math.log2(p)
            ic = math.log2(20.0) + ic  # in bits, range ~ 0..4.32
            ic_norm = min(1.0, ic / 4.32)  # 0..1
            # Stack glyphs from tallest to shortest, height ∝ p × ic_norm.
            cum_y = 0.0
            for aa, c in sorted(counts.items(), key=lambda kv: -kv[1]):
                if aa in GAP_CHARS:
                    continue
                p = c / total
                h = p * ic_norm
                color = _v11_aa_color(aa)
                ax.add_patch(plt.Rectangle((pos_idx, row_idx + cum_y),
                                            1.0, h, facecolor=color,
                                            edgecolor="white", linewidth=0.3, alpha=0.9))
                if h > 0.18:
                    ax.text(pos_idx + 0.5, row_idx + cum_y + h / 2.0,
                            aa, ha="center", va="center", fontsize=8,
                            color="white", fontweight="bold")
                cum_y += h
        ax.text(-0.4, row_idx + 0.5, f"{clade}\n(n={n})", ha="right", va="center", fontsize=8)

    # Reference motif glyph row at the top.
    ref_y = len(clades) + 0.3
    for pos_idx, aa in enumerate(ref_motif):
        ax.add_patch(plt.Rectangle((pos_idx, ref_y), 1.0, 0.55,
                                    facecolor=_v11_aa_color(aa), edgecolor="black",
                                    linewidth=0.4, alpha=0.95))
        ax.text(pos_idx + 0.5, ref_y + 0.27, aa, ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
    ax.text(-0.4, ref_y + 0.27, "Reference (human)", ha="right", va="center", fontsize=8, fontweight="bold")

    ax.set_xlim(-0.5, motif_len + 0.2)
    ax.set_ylim(-0.5, ref_y + 1.0)
    # Position labels (1-based ref coords).
    start = int(motif_row["start"])
    ax.set_xticks([i + 0.5 for i in range(motif_len)])
    ax.set_xticklabels([str(start + i) for i in range(motif_len)], fontsize=8)
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#999")
    ax.set_title(f"V11 motif '{label}' — per-clade sequence logo (ref pos {start}-{int(motif_row['end'])})", fontsize=10)
    ax.set_xlabel("Reference ungapped position")
    fig.tight_layout()
    fig.savefig(outpath_png, dpi=180)
    fig.savefig(outpath_svg)
    plt.close(fig)


def v11_plot_motif_species_heatmap(per_species_df: pd.DataFrame,
                                   motif_row: pd.Series,
                                   representatives_df: Optional[pd.DataFrame],
                                   outpath_png: Path,
                                   outpath_svg: Path) -> None:
    """Heatmap for one motif: representative species (rows) × motif positions
    (cols). Cell colour = amino-acid class. Cell label = single-letter code.
    Matches the V11 representative-comparison aesthetic."""
    motif_id = str(motif_row["motif_id"])
    label = str(motif_row.get("label", motif_id))
    motif_data = per_species_df[per_species_df["motif_id"] == motif_id]
    if motif_data.empty:
        return
    motif_len = int(motif_data.iloc[0]["length"])
    ref_motif = str(motif_data.iloc[0]["reference_motif"])
    # Filter to representative record_ids when provided; otherwise show all clade-distinct species.
    if representatives_df is not None and not representatives_df.empty:
        rep_records = list(representatives_df["record_id"].astype(str))
        rep_set = set(rep_records)
        focus = motif_data[motif_data["record_id"].astype(str).isin(rep_set)].copy()
        order = [r for r in rep_records if r in set(focus["record_id"].astype(str))]
        focus["_order"] = focus["record_id"].astype(str).apply(lambda r: order.index(r) if r in order else 1e9)
        focus = focus.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    else:
        focus = motif_data.copy().reset_index(drop=True)
    if focus.empty:
        return

    fig_w = max(5.0, 0.7 * motif_len + 4.5)
    fig_h = max(3.0, 0.35 * len(focus) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for row_idx, (_, row) in enumerate(focus.iterrows()):
        residues = str(row["motif_residues"])
        for pos_idx, aa in enumerate(residues):
            color = _v11_aa_color(aa)
            ax.add_patch(plt.Rectangle((pos_idx, row_idx), 1.0, 1.0,
                                        facecolor=color, edgecolor="white", linewidth=0.4))
            ax.text(pos_idx + 0.5, row_idx + 0.5, aa, ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    # Reference row on top.
    ref_y = len(focus) + 0.4
    for pos_idx, aa in enumerate(ref_motif):
        ax.add_patch(plt.Rectangle((pos_idx, ref_y), 1.0, 0.7,
                                    facecolor=_v11_aa_color(aa), edgecolor="black", linewidth=0.5))
        ax.text(pos_idx + 0.5, ref_y + 0.35, aa, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

    y_labels = [f"{r['species']} [{r['clade']}]" for _, r in focus.iterrows()]
    ax.set_yticks([i + 0.5 for i in range(len(focus))])
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_yticks(list(ax.get_yticks()) + [ref_y + 0.35])
    new_labels = list(y_labels) + ["Reference (human)"]
    ax.set_yticklabels(new_labels, fontsize=8)
    start = int(motif_row["start"])
    ax.set_xticks([i + 0.5 for i in range(motif_len)])
    ax.set_xticklabels([str(start + i) for i in range(motif_len)], fontsize=8)
    ax.set_xlim(-0.2, motif_len + 0.2)
    ax.set_ylim(-0.2, ref_y + 1.2)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title(f"V11 motif '{label}' — representative species residues (ref pos {start}-{int(motif_row['end'])})", fontsize=10)
    ax.set_xlabel("Reference ungapped position")
    fig.tight_layout()
    fig.savefig(outpath_png, dpi=180)
    fig.savefig(outpath_svg)
    plt.close(fig)


def v11_plot_lineage_stabilization_landscape(stabilization_df: pd.DataFrame,
                                              motifs_df: pd.DataFrame,
                                              outpath_png: Path,
                                              outpath_svg: Path,
                                              max_position: Optional[int] = None,
                                              js_top_df: Optional[pd.DataFrame] = None) -> None:
    """Strip plot along the reference position axis:
      - Top track: stabilization score per position (bars; positive = more
        stable in derived lineages, negative = lost stability).
      - Middle track: top-1 JS-divergent clade pair per position (line).
      - Bottom track: motif-hit rug (user vs library).

    No knowledge of `domains.tsv` is assumed here; the pipeline can overlay
    domain rectangles separately if desired."""
    if stabilization_df.empty:
        return
    positions = stabilization_df["reference_ungapped_position"].astype(int).to_numpy()
    scores = stabilization_df["stabilization_score"].astype(float).to_numpy()
    if max_position is None:
        max_position = int(positions.max())

    nrows = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(14, 6.5),
                              sharex=True, gridspec_kw={"height_ratios": [3, 2, 1]})
    ax_score, ax_js, ax_motif = axes
    # Score bars (red = stabilized in derived, blue = stabilized in ancestral).
    colours = ["#dc2626" if s > 0 else "#2563eb" for s in scores]
    ax_score.bar(positions, scores, color=colours, width=1.0, linewidth=0)
    ax_score.axhline(0.0, color="#999", lw=0.6)
    ax_score.set_ylabel("Stabilization\nscore (bits)\n← ancestral | derived →", fontsize=9)
    ax_score.set_title("V11 lineage stabilization landscape — ancestral vs derived clades",
                        fontsize=11, pad=8)
    for spine in ("top", "right"):
        ax_score.spines[spine].set_visible(False)

    # JS divergence line (top clade pair per position).
    if js_top_df is not None and not js_top_df.empty:
        ax_js.plot(js_top_df["reference_ungapped_position"].astype(int),
                    js_top_df["js_divergence"].astype(float),
                    color="#7c3aed", lw=1.0)
        ax_js.set_ylabel("Top clade-pair\nJS divergence", fontsize=9)
    else:
        ax_js.text(0.5, 0.5, "(JS divergence not computed)", ha="center", va="center",
                    transform=ax_js.transAxes, fontsize=9, color="#999")
    ax_js.set_ylim(0, 1.0)
    for spine in ("top", "right"):
        ax_js.spines[spine].set_visible(False)

    # Motif rug.
    if motifs_df is not None and not motifs_df.empty:
        for _, m in motifs_df.iterrows():
            start = int(m["start"])
            end = int(m["end"])
            color = "#0f766e" if m.get("source") == "user" else "#b45309"
            ax_motif.add_patch(plt.Rectangle((start - 0.5, 0.1), end - start + 1.0, 0.8,
                                              facecolor=color, alpha=0.7, edgecolor="black", linewidth=0.4))
            if m.get("source") == "user":
                ax_motif.text((start + end) / 2.0, 1.05, str(m.get("label", "")),
                                ha="center", va="bottom", fontsize=7, color="#0f766e")
    ax_motif.set_ylim(0, 1.6)
    ax_motif.set_yticks([])
    for spine in ("top", "right", "left"):
        ax_motif.spines[spine].set_visible(False)
    ax_motif.set_ylabel("Motif hits", fontsize=9)
    ax_motif.set_xlabel("Reference ungapped position")
    ax_motif.set_xlim(0.5, max_position + 0.5)

    fig.tight_layout()
    fig.savefig(outpath_png, dpi=180)
    fig.savefig(outpath_svg)
    plt.close(fig)


def v11_render_alignment_with_motif_annotations(outdir: Path,
                                                 motifs_df: pd.DataFrame,
                                                 representatives_df: pd.DataFrame,
                                                 reference_species: str,
                                                 per_species_df: pd.DataFrame,
                                                 alignment: Any) -> Optional[Path]:
    """Render a single SVG showing the projected MUSCLE alignment for the V11
    representative species with colored rectangles overlaid on motif hits.

    The figure has one row per representative species (plus a top row for the
    human reference) and one column per ungapped reference position. Cells
    display the residue glyph; motif spans are drawn as semi-transparent
    coloured boxes with labels above the reference row.

    Also emits a sibling HTML wrapper with the SVG inline and small per-motif
    tooltips (hover anywhere over the box → see per-clade consensus)."""
    if representatives_df is None or representatives_df.empty:
        return None
    aln_to_ref, ref_record = _v11_ref_position_records(alignment, reference_species)
    rep_records = list(representatives_df["record_id"].astype(str))
    species_order = list(representatives_df["species"])
    clade_lookup = dict(zip(representatives_df["record_id"].astype(str), representatives_df["clade"]))
    # Build a record_id -> Seq map for fast lookup.
    seq_by_record: Dict[str, str] = {str(rec.id): str(rec.seq) for rec in alignment}
    ref_seq = str(ref_record.seq).upper()
    ref_residues = [ref_seq[i] for i, rp in enumerate(aln_to_ref) if rp is not None]
    n_pos = len(ref_residues)

    # Layout: cell 9px wide × 14px tall; top margin 80px (for motif labels);
    # left margin 220px (species labels). Wrap every 100 ref positions into a
    # new "block" so the SVG isn't a giant horizontal strip.
    cell_w = 9
    cell_h = 14
    wrap = 100
    rows_per_block = len(rep_records) + 2  # +1 ref row, +1 motif label space
    blocks = math.ceil(n_pos / wrap)
    block_h = rows_per_block * cell_h + 40
    svg_w = 220 + wrap * cell_w + 30
    svg_h = blocks * block_h + 60

    # Helper: color palette for motifs.
    def _motif_color(motif_row: pd.Series) -> str:
        return "#0f766e" if motif_row.get("source") == "user" else "#b45309"

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" font-family="Consolas,monospace" font-size="11">')
    parts.append('<rect width="100%" height="100%" fill="#fffaf3"/>')
    parts.append(f'<text x="20" y="24" font-size="14" font-weight="bold" fill="#14213d">V11 reference-projected alignment with motif annotations</text>')
    parts.append(f'<text x="20" y="42" font-size="10" fill="#5f6b7a">Reference = {escape(reference_species)} ; representatives bolded ; motif hits boxed (teal=user, amber=library)</text>')

    motifs_sorted = motifs_df.sort_values(["start", "end"]).reset_index(drop=True) if (motifs_df is not None and not motifs_df.empty) else pd.DataFrame()

    for block_idx in range(blocks):
        block_start = block_idx * wrap
        block_end = min(n_pos, block_start + wrap)
        y_base = 60 + block_idx * block_h

        # Motif boxes that span any position in this block.
        if not motifs_sorted.empty:
            for _, m in motifs_sorted.iterrows():
                s, e = int(m["start"]), int(m["end"])
                # Block coords (0-based among kept positions).
                a = max(s - 1, block_start)
                b = min(e, block_end) - 1
                if a > b:
                    continue
                x0 = 220 + (a - block_start) * cell_w
                x1 = 220 + (b - block_start + 1) * cell_w
                color = _motif_color(m)
                opacity = 0.32 if m.get("source") == "user" else 0.20
                title_txt = f"{m.get('label','motif')} ({m.get('source','')}) ref {s}-{e}"
                parts.append(f'<rect x="{x0}" y="{y_base}" width="{x1 - x0}" height="{(len(rep_records) + 1) * cell_h}" fill="{color}" fill-opacity="{opacity}" stroke="{color}" stroke-width="0.7"><title>{escape(title_txt)}</title></rect>')
                if m.get("source") == "user":
                    parts.append(f'<text x="{(x0 + x1) / 2}" y="{y_base - 4}" text-anchor="middle" font-size="9" fill="{color}" font-weight="bold">{escape(str(m.get("label", "")))}</text>')

        # Reference row.
        y = y_base + cell_h
        parts.append(f'<text x="200" y="{y + 11}" text-anchor="end" font-size="10" font-weight="bold" fill="#14213d">Reference (human)</text>')
        for i in range(block_start, block_end):
            aa = ref_residues[i]
            color = _v11_aa_color(aa)
            x = 220 + (i - block_start) * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}" fill-opacity="0.25"/>')
            parts.append(f'<text x="{x + cell_w / 2}" y="{y + 11}" text-anchor="middle" fill="#14213d">{escape(aa)}</text>')

        # Per-rep rows.
        for ridx, rec_id in enumerate(rep_records):
            y = y_base + (ridx + 2) * cell_h
            species_label = species_order[ridx] if ridx < len(species_order) else rec_id
            clade = clade_lookup.get(rec_id, "Unassigned")
            parts.append(f'<text x="200" y="{y + 11}" text-anchor="end" font-size="10" fill="#14213d">{escape(species_label)} <tspan fill="#5f6b7a" font-size="9">[{escape(clade)}]</tspan></text>')
            spc_seq = seq_by_record.get(rec_id, "")
            for i in range(block_start, block_end):
                col_idx = None
                # Map block-relative i to the actual alignment column (since
                # we only kept columns that correspond to ref ungapped pos).
                ref_idx = i + 1
                # Walk aln_to_ref to find the column for ref pos i+1.
                # We rebuild a quick lookup once outside the loop for perf.
                pass
            # Build pos→col lookup just once per block from the cached ref-residue iter.
            # (Lookup happens lazily in next loop.)

        # Build a position→column index once for all rows.
        # (Done outside the rows loop for efficiency.)
        # We re-iterate now to fill in the per-rep glyphs.
    # End of block iteration sketch (we need a real per-rep glyph emission).

    # ---- Real per-rep glyph emission (refactor) ----
    parts = []  # restart parts cleanly with the proper second pass
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" font-family="Consolas,monospace" font-size="11">')
    parts.append('<rect width="100%" height="100%" fill="#fffaf3"/>')
    parts.append(f'<text x="20" y="24" font-size="14" font-weight="bold" fill="#14213d">V11 reference-projected alignment with motif annotations</text>')
    parts.append(f'<text x="20" y="42" font-size="10" fill="#5f6b7a">Reference = {escape(reference_species)} ; representatives bolded ; motif hits boxed (teal=user, amber=library)</text>')

    # Pre-extract every record's ungapped-projected sequence (one residue per
    # ref position) so we can index directly by block position.
    rep_projected: Dict[str, str] = {}
    for rec_id in rep_records:
        seq = seq_by_record.get(rec_id, "")
        rep_projected[rec_id] = "".join(
            (seq[c] if c < len(seq) else "-") for c, rp in enumerate(aln_to_ref) if rp is not None
        ).upper()

    for block_idx in range(blocks):
        block_start = block_idx * wrap
        block_end = min(n_pos, block_start + wrap)
        y_base = 60 + block_idx * block_h

        # Motif boxes for this block.
        if not motifs_sorted.empty:
            for _, m in motifs_sorted.iterrows():
                s, e = int(m["start"]), int(m["end"])
                a = max(s - 1, block_start)
                b = min(e, block_end) - 1
                if a > b:
                    continue
                x0 = 220 + (a - block_start) * cell_w
                x1 = 220 + (b - block_start + 1) * cell_w
                color = _motif_color(m)
                opacity = 0.32 if m.get("source") == "user" else 0.20
                title_txt = f"{m.get('label','motif')} ({m.get('source','')}) ref {s}-{e}"
                parts.append(f'<rect x="{x0}" y="{y_base}" width="{x1 - x0}" height="{(len(rep_records) + 1) * cell_h}" fill="{color}" fill-opacity="{opacity}" stroke="{color}" stroke-width="0.7"><title>{escape(title_txt)}</title></rect>')
                if m.get("source") == "user":
                    parts.append(f'<text x="{(x0 + x1) / 2}" y="{y_base - 4}" text-anchor="middle" font-size="9" fill="{color}" font-weight="bold">{escape(str(m.get("label", "")))}</text>')

        # Position ruler.
        ruler_y = y_base - 4
        for tick in range(block_start, block_end, 10):
            x = 220 + (tick - block_start) * cell_w
            parts.append(f'<text x="{x}" y="{ruler_y}" font-size="8" fill="#999">{tick + 1}</text>')

        # Reference row.
        y = y_base + cell_h
        parts.append(f'<text x="200" y="{y + 11}" text-anchor="end" font-size="10" font-weight="bold" fill="#14213d">Reference (human)</text>')
        for i in range(block_start, block_end):
            aa = ref_residues[i]
            color = _v11_aa_color(aa)
            x = 220 + (i - block_start) * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}" fill-opacity="0.25"/>')
            parts.append(f'<text x="{x + cell_w / 2}" y="{y + 11}" text-anchor="middle" fill="#14213d">{escape(aa)}</text>')

        # Representative rows.
        for ridx, rec_id in enumerate(rep_records):
            y = y_base + (ridx + 2) * cell_h
            species_label = species_order[ridx] if ridx < len(species_order) else rec_id
            clade = clade_lookup.get(rec_id, "Unassigned")
            parts.append(f'<text x="200" y="{y + 11}" text-anchor="end" font-size="10" fill="#14213d">{escape(species_label)} <tspan fill="#5f6b7a" font-size="9">[{escape(clade)}]</tspan></text>')
            seq = rep_projected.get(rec_id, "")
            for i in range(block_start, block_end):
                if i >= len(seq):
                    break
                aa = seq[i]
                color = _v11_aa_color(aa)
                x = 220 + (i - block_start) * cell_w
                # Highlight identity to reference with stronger background.
                ref_aa = ref_residues[i]
                opacity = 0.35 if aa == ref_aa else 0.15
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}" fill-opacity="{opacity}"/>')
                parts.append(f'<text x="{x + cell_w / 2}" y="{y + 11}" text-anchor="middle" fill="#14213d">{escape(aa)}</text>')

    parts.append('</svg>')

    svg_path = outdir / V11_ALIGNMENT_WITH_MOTIF_ANNOTATIONS_SVG
    svg_path.write_text("\n".join(parts), encoding="utf-8")

    html_path = outdir / V11_ALIGNMENT_WITH_MOTIF_ANNOTATIONS_HTML
    html_content = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>V11 alignment with motif annotations</title>"
        "<style>body{margin:0;padding:14px;background:#f4efe4;font-family:Segoe UI,Tahoma,sans-serif;}"
        "h1{font-size:1.2rem;color:#14213d;}p{color:#5f6b7a;}svg{display:block;background:#fffaf3;}</style></head>"
        "<body><h1>V11 reference-projected alignment with motif annotations</h1>"
        "<p>Teal boxes: user-supplied motifs (<code>--annotated_motifs</code>). "
        "Amber boxes: hits from the V11 regulatory-motif regex library. "
        "Hover any box to see its label and ref-position span.</p>"
        + "\n".join(parts) +
        "</body></html>"
    )
    html_path.write_text(html_content, encoding="utf-8")
    return svg_path


# ---------------------- main orchestrator -------------------------------------#

def v11_write_motif_analysis_outputs(outdir: Path | str,
                                     alignment: Any,
                                     reference_species: str,
                                     taxonomy_lookup: Optional[Dict[str, Any]] = None,
                                     annotated_motifs_text: Optional[str] = None,
                                     extra_motif_regex_text: Optional[str] = None,
                                     ancestral_clades: Sequence[str] = V11_DEFAULT_ANCESTRAL_CLADES,
                                     derived_clades: Sequence[str] = V11_DEFAULT_DERIVED_CLADES,
                                     representatives_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """End-to-end V11 motif & lineage-stabilization orchestrator. Produces
    every artifact described in the V11 README's motif section. Returns a
    summary dict (for pipeline-step logging)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Parse user / library motifs and union them.
    user_df = v11_parse_annotated_motifs(annotated_motifs_text)
    extra_regex = v11_parse_extra_motif_regex(extra_motif_regex_text)
    _, ref_record = _v11_ref_position_records(alignment, reference_species)
    ref_ungapped = "".join(a for a in str(ref_record.seq).upper() if a not in GAP_CHARS)
    library_df = v11_scan_motif_library(ref_ungapped, V11_REGULATORY_MOTIF_LIBRARY, extra_regex)
    motifs_master = v11_merge_motif_tables(user_df, library_df)

    user_df.to_csv(outdir / V11_ANNOTATED_MOTIFS_TSV, sep="\t", index=False)
    library_df.to_csv(outdir / V11_MOTIF_LIBRARY_HITS_TSV, sep="\t", index=False)
    motifs_master.to_csv(outdir / V11_MOTIFS_MASTER_TSV, sep="\t", index=False)

    # --- Per-clade entropy / consensus.
    entropy_df = v11_compute_per_clade_entropy(alignment, taxonomy_lookup, reference_species)
    entropy_df.to_csv(outdir / V11_PER_CLADE_ENTROPY_CSV, index=False)
    consensus_df = v11_compute_per_clade_consensus(alignment, taxonomy_lookup, reference_species)
    consensus_df.to_csv(outdir / V11_PER_CLADE_CONSENSUS_CSV, index=False)

    # --- Clade-pair JS divergence (long format).
    js_df = v11_compute_clade_pair_js_divergence(alignment, taxonomy_lookup, reference_species)
    js_df.to_csv(outdir / V11_CLADE_PAIR_JS_DIVERGENCE_CSV, index=False)

    # --- Lineage stabilization score.
    stab_df = v11_compute_lineage_stabilization(entropy_df, ancestral_clades, derived_clades)
    stab_df.to_csv(outdir / V11_LINEAGE_STABILIZATION_CSV, index=False)

    # --- Motif evolution tables.
    per_species_df, per_clade_df = v11_compute_motif_evolution(
        alignment, motifs_master, reference_species, taxonomy_lookup
    )
    per_species_df.to_csv(outdir / V11_MOTIF_EVOLUTION_PER_SPECIES_TSV, sep="\t", index=False)
    per_clade_df.to_csv(outdir / V11_MOTIF_EVOLUTION_PER_CLADE_TSV, sep="\t", index=False)

    # --- Per-motif figures (logos + species heatmap). Only render for motifs
    # the user explicitly supplied OR a handful of "interesting" library hits
    # (top-N longest / most matched). To keep the figure count bounded we cap
    # library motif renders at 12 (the longest matches by default).
    motif_figure_paths: List[str] = []
    rendered: List[pd.Series] = []
    user_motif_rows = motifs_master[motifs_master["source"] == "user"]
    library_rows = motifs_master[motifs_master["source"] == "library"].copy()
    if not library_rows.empty:
        library_rows["_len"] = library_rows["end"].astype(int) - library_rows["start"].astype(int) + 1
        library_rows = library_rows.sort_values("_len", ascending=False).head(12)
    for _, row in pd.concat([user_motif_rows, library_rows], ignore_index=True).iterrows():
        motif_id = str(row["motif_id"])
        logo_png = outdir / f"v11_motif_{motif_id}_clade_logos.png"
        logo_svg = outdir / f"v11_motif_{motif_id}_clade_logos.svg"
        v11_plot_motif_clade_logos(per_species_df, row, logo_png, logo_svg)
        heat_png = outdir / f"v11_motif_{motif_id}_species_heatmap.png"
        heat_svg = outdir / f"v11_motif_{motif_id}_species_heatmap.svg"
        v11_plot_motif_species_heatmap(per_species_df, row, representatives_df, heat_png, heat_svg)
        if logo_svg.exists():
            motif_figure_paths.append(str(logo_svg))
        if heat_svg.exists():
            motif_figure_paths.append(str(heat_svg))
        rendered.append(row)

    # --- Top-clade-pair JS divergence summary (one row per position with the
    # single highest JS pair) — used by the landscape figure.
    js_top: pd.DataFrame
    if not js_df.empty:
        js_top = js_df.sort_values(["reference_ungapped_position", "js_divergence"],
                                    ascending=[True, False]).drop_duplicates(
            subset=["reference_ungapped_position"], keep="first"
        )[["reference_ungapped_position", "clade_a", "clade_b", "js_divergence"]]
    else:
        js_top = pd.DataFrame(columns=["reference_ungapped_position", "clade_a", "clade_b", "js_divergence"])

    # --- Landscape figure.
    landscape_png = outdir / V11_LINEAGE_STABILIZATION_LANDSCAPE_PNG
    landscape_svg = outdir / V11_LINEAGE_STABILIZATION_LANDSCAPE_SVG
    if not stab_df.empty:
        v11_plot_lineage_stabilization_landscape(
            stab_df, motifs_master, landscape_png, landscape_svg,
            max_position=int(stab_df["reference_ungapped_position"].max()),
            js_top_df=js_top,
        )

    # --- Annotated alignment SVG/HTML.
    if representatives_df is not None and not representatives_df.empty:
        try:
            v11_render_alignment_with_motif_annotations(
                outdir, motifs_master, representatives_df, reference_species,
                per_species_df, alignment,
            )
        except Exception:
            pass

    return {
        "user_motif_count": int(len(user_df)),
        "library_motif_count": int(len(library_df)),
        "motif_total": int(len(motifs_master)),
        "entropy_rows": int(len(entropy_df)),
        "js_rows": int(len(js_df)),
        "stabilization_rows": int(len(stab_df)),
        "motif_evolution_species_rows": int(len(per_species_df)),
        "motif_evolution_clade_rows": int(len(per_clade_df)),
        "motif_figure_count": int(len(motif_figure_paths)),
        "ancestral_clades": list(ancestral_clades),
        "derived_clades": list(derived_clades),
        "landscape_svg": str(landscape_svg) if landscape_svg.exists() else None,
        "annotated_alignment_svg": str(outdir / V11_ALIGNMENT_WITH_MOTIF_ANNOTATIONS_SVG)
            if (outdir / V11_ALIGNMENT_WITH_MOTIF_ANNOTATIONS_SVG).exists() else None,
    }


# =============================================================================
# V11 per-clade consolidated summary (default)                                 #
# -----------------------------------------------------------------------------#
# A single default view that, for EVERY represented vertebrate clade, overlays #
# the three things that matter together along the human reference axis:        #
#   1. consensus AlphaFold secondary structure (helix / sheet / loop) —        #
#      aggregated from each clade's member species (majority vote per pos),    #
#   2. mean net charge (pH 7.4, 5-aa smoothed) across the clade's members,     #
#   3. the reference domain architecture (from domains.tsv) as a top band.     #
# This generalises the per-representative-species comparison into a clade-     #
# level "what is conserved across each clade" picture, inspired by the MATLAB  #
# clade-bubble reference/charge figure.                                        #
# =============================================================================

V11_PER_CLADE_SS_CSV = "v11_per_clade_secondary_structure.csv"
V11_PER_CLADE_NET_CHARGE_CSV = "v11_per_clade_net_charge.csv"
V11_CLADE_CONSOLIDATED_SUMMARY_PNG = "v11_clade_consolidated_summary.png"
V11_CLADE_CONSOLIDATED_SUMMARY_SVG = "v11_clade_consolidated_summary.svg"

_V11_SS_COLORS = {"helix": "#dc2626", "sheet": "#7c3aed", "loop": "#cbd5e1"}

# Preferred vertebrate clade ordering (basal → derived) for the consolidated
# figure; clades not listed fall to the end alphabetically.
_V11_CLADE_DISPLAY_ORDER = (
    "Cyclostomata", "Chondrichthyes", "Polypteriformes", "Chondrostei",
    "Holostei", "Teleostei", "Coelacanthiformes", "Dipnoi",
    "Amphibia", "Reptilia", "Aves", "Mammalia",
    "Tunicata", "Cephalochordata", "Fungi",
)


def _v11_clade_sort_key(clade: str):
    try:
        return (0, _V11_CLADE_DISPLAY_ORDER.index(clade))
    except ValueError:
        return (1, clade)


def v11_compute_per_clade_secondary_structure(outdir: Path | str,
                                              taxonomy_lookup: Optional[Dict[str, Any]],
                                              reference_species: str) -> pd.DataFrame:
    """Aggregate the comparative AlphaFold SS bundle into a per-clade consensus
    secondary structure at every reference position.

    Returns long-format columns:
        reference_position, clade, n_species, helix_frac, sheet_frac,
        loop_frac, consensus_ss, consensus_frac
    """
    outdir = Path(outdir)
    ss_path = outdir / COMPARATIVE_ALPHAFOLD_SS_FILENAME
    if not ss_path.exists():
        return pd.DataFrame(columns=[
            "reference_position", "clade", "n_species",
            "helix_frac", "sheet_frac", "loop_frac", "consensus_ss", "consensus_frac"])
    try:
        bundle = json.loads(ss_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()
    records = bundle.get("records") or []
    track_length = int(bundle.get("track_length") or 0)
    if not records or track_length <= 0:
        return pd.DataFrame()

    # position (1-based) -> clade -> Counter of ss states.
    counts: Dict[int, Dict[str, "_V11_Counter"]] = {}
    clade_species_seen: Dict[str, set] = {}
    for rec in records:
        species = str(rec.get("species") or "")
        tax = _v11_resolve_taxonomy_level(taxonomy_lookup, species) if taxonomy_lookup is not None else ""
        clade = v11_resolve_broad_clade(species, tax) or "Unassigned"
        clade_species_seen.setdefault(clade, set()).add(rec.get("record_id") or species)
        for rng in (rec.get("mapped_ranges") or []):
            kind = str(rng.get("secondary_structure") or rng.get("kind") or "loop").lower()
            if kind not in _V11_SS_COLORS:
                kind = "loop"
            try:
                start = int(rng.get("start_reference_position") or rng.get("start"))
                end = int(rng.get("end_reference_position") or rng.get("end"))
            except (TypeError, ValueError):
                continue
            for pos in range(start, end + 1):
                counts.setdefault(pos, {}).setdefault(clade, _V11_Counter())[kind] += 1

    rows: List[Dict[str, Any]] = []
    for pos in range(1, track_length + 1):
        per_clade = counts.get(pos, {})
        for clade, ctr in per_clade.items():
            total = sum(ctr.values())
            if total == 0:
                continue
            helix = ctr.get("helix", 0) / total
            sheet = ctr.get("sheet", 0) / total
            loop = ctr.get("loop", 0) / total
            consensus = max(("helix", "sheet", "loop"), key=lambda k: ctr.get(k, 0))
            rows.append({
                "reference_position": pos,
                "clade": clade,
                "n_species": total,
                "helix_frac": helix,
                "sheet_frac": sheet,
                "loop_frac": loop,
                "consensus_ss": consensus,
                "consensus_frac": ctr.get(consensus, 0) / total,
            })
    return pd.DataFrame(rows)


def v11_compute_per_clade_net_charge(alignment: Any,
                                     taxonomy_lookup: Optional[Dict[str, Any]],
                                     reference_species: str,
                                     smoothing_window: int = V11_DEFAULT_PROPERTY_WINDOW) -> pd.DataFrame:
    """Mean net charge (pH 7.4, smoothed) per clade per reference position.

    Wide: reference_ungapped_position + mean_charge_<clade> columns, plus a
    parallel n_<clade> count set. Built by averaging the per-species net-charge
    track within each broad clade."""
    per_species = v11_compute_per_species_property_track(
        alignment, V11_RESIDUE_NET_CHARGE_PH74, reference_species=reference_species,
        smoothing_window=smoothing_window,
    )
    if per_species.empty:
        return pd.DataFrame()
    species_clade = {}
    for rec_id in per_species["record_id"]:
        species = str(rec_id).split("|", 1)[0].strip()
        tax = _v11_resolve_taxonomy_level(taxonomy_lookup, species) if taxonomy_lookup is not None else ""
        species_clade[rec_id] = v11_resolve_broad_clade(species, tax) or "Unassigned"
    per_species = per_species.copy()
    per_species["clade"] = per_species["record_id"].map(species_clade)
    pos_cols = [c for c in per_species.columns if c.startswith("pos_")]
    grouped = per_species.groupby("clade")[pos_cols].mean()
    counts = per_species.groupby("clade").size()

    n_positions = len(pos_cols)
    out_rows: List[Dict[str, Any]] = []
    for pos_idx in range(n_positions):
        row: Dict[str, Any] = {"reference_ungapped_position": pos_idx + 1}
        for clade in grouped.index:
            row[f"mean_charge_{clade}"] = float(grouped.loc[clade, pos_cols[pos_idx]])
            row[f"n_{clade}"] = int(counts.loc[clade])
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _v11_load_reference_domains(outdir: Path) -> List[Dict[str, Any]]:
    """Best-effort load of reference domain spans from domains.tsv for the
    consolidated figure's top band. Returns [{label, start, end}]."""
    path = outdir / "domains.tsv"
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        return []
    # Find usable start/end + label columns flexibly.
    cols = {c.lower(): c for c in df.columns}
    start_col = next((cols[c] for c in ("start", "reference_start", "ref_start", "begin") if c in cols), None)
    end_col = next((cols[c] for c in ("end", "reference_end", "ref_end", "stop") if c in cols), None)
    label_col = next((cols[c] for c in ("label", "domain", "name", "description", "feature_type") if c in cols), None)
    if not start_col or not end_col:
        return []
    spans: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        try:
            s = int(float(r[start_col]))
            e = int(float(r[end_col]))
        except (TypeError, ValueError):
            continue
        if e < s:
            s, e = e, s
        spans.append({
            "label": str(r[label_col]) if label_col else "",
            "start": s,
            "end": e,
        })
    return spans


def v11_plot_clade_consolidated_summary(per_clade_ss: pd.DataFrame,
                                        per_clade_charge: pd.DataFrame,
                                        domain_spans: List[Dict[str, Any]],
                                        reference_species: str,
                                        gene_label: str,
                                        outpath_png: Path,
                                        outpath_svg: Path) -> None:
    """One figure, stacked per-clade tracks along the reference axis. Each clade
    gets a thin band: consensus SS as colored segments (helix/sheet/loop) with
    the mean net-charge drawn as a centered line on top. A domain architecture
    band sits at the very top."""
    if per_clade_ss.empty and per_clade_charge.empty:
        return
    # Determine clades present (union), ordered basal → derived.
    clades = set()
    if not per_clade_ss.empty:
        clades |= set(per_clade_ss["clade"].unique())
    if not per_clade_charge.empty:
        clades |= {c[len("mean_charge_"):] for c in per_clade_charge.columns if c.startswith("mean_charge_")}
    clades = [c for c in clades if c]
    clades.sort(key=_v11_clade_sort_key)
    if not clades:
        return

    # Reference length.
    if not per_clade_ss.empty:
        track_len = int(per_clade_ss["reference_position"].max())
    else:
        track_len = int(per_clade_charge["reference_ungapped_position"].max())

    n = len(clades)
    domain_band_h = 0.7
    fig_h = max(5.0, 0.62 * n + 1.8 + domain_band_h)
    fig_w = 16.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # SS lookups: position -> clade -> consensus_ss.
    ss_by_clade_pos: Dict[str, Dict[int, str]] = {}
    if not per_clade_ss.empty:
        for clade, sub in per_clade_ss.groupby("clade"):
            ss_by_clade_pos[clade] = dict(zip(sub["reference_position"].astype(int), sub["consensus_ss"]))

    row_h = 1.0
    charge_amp = 0.42  # vertical half-amplitude for the charge line within a row

    for row_idx, clade in enumerate(clades):
        y0 = n - 1 - row_idx  # top clade at top
        # SS band as colored segments.
        ss_map = ss_by_clade_pos.get(clade, {})
        if ss_map:
            run_start = None
            run_kind = None
            for pos in range(1, track_len + 2):
                kind = ss_map.get(pos)
                if kind != run_kind or pos > track_len:
                    if run_kind is not None and run_start is not None:
                        ax.add_patch(Rectangle(
                            (run_start - 0.5, y0 + 0.12), (pos - run_start), row_h - 0.5,
                            facecolor=_V11_SS_COLORS.get(run_kind, "#cbd5e1"),
                            edgecolor="none", alpha=0.85, zorder=2))
                    run_start = pos
                    run_kind = kind
        # Mean net-charge line centered in the row.
        if not per_clade_charge.empty:
            col = f"mean_charge_{clade}"
            if col in per_clade_charge.columns:
                x = per_clade_charge["reference_ungapped_position"].to_numpy()
                q = per_clade_charge[col].to_numpy(dtype=float)
                q_clipped = np.clip(q, -1.0, 1.0)
                yline = y0 + 0.5 + charge_amp * q_clipped
                ax.plot(x, yline, color="#111827", lw=0.7, zorder=3)
                ax.axhline  # noop guard
                ax.plot([1, track_len], [y0 + 0.5, y0 + 0.5], color="#9ca3af", lw=0.4, ls=":", zorder=2)
        # Clade label.
        n_sp = ""
        if not per_clade_charge.empty and f"n_{clade}" in per_clade_charge.columns:
            n_sp = f" (n={int(per_clade_charge[f'n_{clade}'].iloc[0])})"
        ax.text(-0.012 * track_len, y0 + 0.5, f"{clade}{n_sp}", ha="right", va="center",
                fontsize=9, color=_v11_clade_color(clade), fontweight="bold")

    # Domain architecture band on top.
    top_y = n + 0.25
    if domain_spans:
        seen_labels = set()
        for d in domain_spans:
            s = max(1, int(d["start"]))
            e = min(track_len, int(d["end"]))
            if e < s:
                continue
            ax.add_patch(Rectangle((s - 0.5, top_y), (e - s + 1), domain_band_h,
                                    facecolor="#0f766e", edgecolor="#0b4f49",
                                    alpha=0.30, zorder=2))
            lbl = str(d.get("label") or "")
            if lbl and lbl not in seen_labels and (e - s) > track_len * 0.03:
                ax.text((s + e) / 2.0, top_y + domain_band_h / 2.0, lbl[:18],
                        ha="center", va="center", fontsize=6.5, color="#0b4f49")
                seen_labels.add(lbl)
    else:
        # No coordinate-mapped domains in domains.tsv (e.g. only InterPro IDs
        # without start/end). Say so explicitly rather than leaving a blank band.
        ax.text(track_len / 2.0, top_y + domain_band_h / 2.0,
                "no coordinate-mapped reference domains in domains.tsv",
                ha="center", va="center", fontsize=7.5, fontstyle="italic", color="#9ca3af")
    ax.text(-0.012 * track_len, top_y + domain_band_h / 2.0, "Domains", ha="right", va="center",
            fontsize=9, color="#0f766e", fontweight="bold")

    ax.set_xlim(-0.14 * track_len, track_len + 1)
    ax.set_ylim(-0.3, top_y + domain_band_h + 0.4)
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel(f"Reference ({reference_species}) ungapped position")
    ax.set_title(f"{gene_label} — per-clade consensus secondary structure + mean net charge + domains",
                 fontsize=12)

    # Legend.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor=_V11_SS_COLORS["helix"], label="Consensus helix", alpha=0.85),
        Patch(facecolor=_V11_SS_COLORS["sheet"], label="Consensus sheet", alpha=0.85),
        Patch(facecolor=_V11_SS_COLORS["loop"], label="Consensus loop", alpha=0.85),
        Line2D([0], [0], color="#111827", lw=1.0, label="Mean net charge (pH 7.4, clipped ±1)"),
        Patch(facecolor="#0f766e", alpha=0.30, label="Reference domain"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, ncol=2, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(outpath_png, dpi=170)
    fig.savefig(outpath_svg)
    plt.close(fig)


def v11_write_clade_consolidated_outputs(outdir: Path | str,
                                         alignment: Any,
                                         reference_species: str,
                                         gene_label: str,
                                         taxonomy_lookup: Optional[Dict[str, Any]] = None,
                                         smoothing_window: int = V11_DEFAULT_PROPERTY_WINDOW) -> Dict[str, Any]:
    """Default per-clade consolidated analysis: consensus SS + mean net charge
    + domains for every represented clade. Writes CSVs + one combined figure."""
    outdir = Path(outdir)
    ss_df = v11_compute_per_clade_secondary_structure(outdir, taxonomy_lookup, reference_species)
    ss_df.to_csv(outdir / V11_PER_CLADE_SS_CSV, index=False)
    charge_df = v11_compute_per_clade_net_charge(alignment, taxonomy_lookup, reference_species, smoothing_window)
    charge_df.to_csv(outdir / V11_PER_CLADE_NET_CHARGE_CSV, index=False)
    domain_spans = _v11_load_reference_domains(outdir)

    png = outdir / V11_CLADE_CONSOLIDATED_SUMMARY_PNG
    svg = outdir / V11_CLADE_CONSOLIDATED_SUMMARY_SVG
    try:
        v11_plot_clade_consolidated_summary(
            ss_df, charge_df, domain_spans, reference_species, gene_label, png, svg)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"clade consolidated figure failed: {exc}",
                "per_clade_ss_rows": int(len(ss_df)),
                "per_clade_charge_rows": int(len(charge_df))}

    clades = sorted(set(ss_df["clade"].unique()) if not ss_df.empty else set(), key=_v11_clade_sort_key)
    return {
        "per_clade_ss_rows": int(len(ss_df)),
        "per_clade_charge_rows": int(len(charge_df)),
        "clades": clades,
        "clade_count": len(clades),
        "domain_span_count": len(domain_spans),
        "summary_svg": str(svg) if svg.exists() else None,
    }


# =============================================================================
# V11 interactive 3D structure overlay (3Dmol.js)                              #
# -----------------------------------------------------------------------------#
# Emits the per-clade identity CSV in the format consumed by the standalone    #
# build_structure_overlay.py viewer (one *_IdentityFraction / *_CoveredSpecies #
# / *_IdentitySpeciesEquivalent / *_TotalSpeciesInGroup quartet per broad      #
# clade, per reference position), then invokes the generator to write a self-  #
# contained <gene>_structure_overlay.html that paints those clade identities   #
# onto the human reference AlphaFold model. Gene-agnostic.                     #
# =============================================================================

V11_STRUCTURE_OVERLAY_CSV = "v11_clade_identity_by_reference_position.csv"
V11_STRUCTURE_OVERLAY_HTML = "v11_structure_overlay.html"
V11_CLADE_IDENTITY_BUBBLE_PNG = "v11_clade_identity_bubble.png"
V11_CLADE_IDENTITY_BUBBLE_SVG = "v11_clade_identity_bubble.svg"
V11_CLADE_IDENTITY_BUBBLE_PDF = "v11_clade_identity_bubble.pdf"
# Subdivided 9-group clade analysis (Primates/Rodents/OtherMammals/Teleosts/
# OtherFish/Birds/Reptiles/Amphibians/OtherVertebrates). Originally a MATLAB
# clade-analysis grouping; reimplemented in Python so it runs for any gene.
V11_GROUPED_CLADE_IDENTITY_CSV = "v11_clade_identity_by_reference_position_9group.csv"
V11_GROUPED_CLADE_IDENTITY_BUBBLE_PNG = "v11_clade_identity_bubble_9group.png"
V11_GROUPED_CLADE_IDENTITY_BUBBLE_SVG = "v11_clade_identity_bubble_9group.svg"
V11_GROUPED_CLADE_IDENTITY_BUBBLE_PDF = "v11_clade_identity_bubble_9group.pdf"
V11_GROUPED_STRUCTURE_OVERLAY_HTML = "v11_structure_overlay_9group.html"
# Mirror of the user's MATLAB-era reference CSV: exact column names
# (Primates, Rodents, OtherMammals, Teleosts, Birds, Reptiles, Amphibians,
# Other, x), same row layout. Filename matches the user's reference too so
# tooling that consumed clade_identity_by_reference_position_mod.csv works.
V11_MOD_CLADE_IDENTITY_CSV = "v11_clade_identity_by_reference_position_mod.csv"
V11_MOD_CLADE_IDENTITY_BUBBLE_PNG = "v11_clade_identity_bubble_mod.png"
V11_MOD_CLADE_IDENTITY_BUBBLE_SVG = "v11_clade_identity_bubble_mod.svg"
V11_MOD_CLADE_IDENTITY_BUBBLE_PDF = "v11_clade_identity_bubble_mod.pdf"
V11_MOD_STRUCTURE_OVERLAY_HTML = "v11_structure_overlay_mod.html"
# Combined viewer that exposes broad clades + 9-group subdivisions + the
# compact 9-group "mod" groups as selectable rows in a single 3Dmol dropdown,
# so the user can pick any bucket to shade the structure without switching
# overlay files.
V11_COMBINED_CLADE_IDENTITY_CSV = "v11_clade_identity_by_reference_position_combined.csv"
V11_COMBINED_STRUCTURE_OVERLAY_HTML = "v11_structure_overlay_combined.html"


# Display order for the bubble grid: most-conserved (mammals) at top, basal /
# non-vertebrate clades at the bottom — echoing the MATLAB figure's layout.
_V11_BUBBLE_CLADE_ORDER = (
    "Mammalia", "Aves", "Reptilia", "Amphibia",
    "Coelacanthiformes", "Dipnoi", "Teleostei", "Holostei",
    "Polypteriformes", "Chondrostei", "Chondrichthyes", "Cyclostomata",
    "Tunicata", "Cephalochordata", "Fungi", "Arthropoda",
    "Nematoda", "Mollusca", "Echinodermata", "Cnidaria", "Unassigned",
)


def _v11_bubble_clade_key(clade: str):
    try:
        return (0, _V11_BUBBLE_CLADE_ORDER.index(clade))
    except ValueError:
        return (1, clade)


V11_TELEOST_CONSERVED_RODENT_DIVERGED_CSV = "v11_teleost_conserved_rodent_diverged_sites.csv"
V11_HIGHLIGHT_SITES_POCKET_DISTANCE_CSV = "v11_teleost_conserved_rodent_diverged_sites_pocket_distance.csv"
V11_POCKET_RESIDUES_CSV = "v11_pocket_residues.csv"  # V11.1 pillar-5 output
# Pocket-proximity class thresholds (Cα-Cα distance in Angstroms from the
# highlight residue to the nearest hand-curated active-site residue):
#   ≤ 5 Å  → pocket_lining   (direct contact, side-chain accessible)
#   ≤ 8 Å  → pocket_proximal (typical "in-pocket" SDP definition; what V11
#                              pillar 5 uses for substrate-pocket calls)
#   ≤ 12 Å → near_pocket     (allosteric / first-shell)
#   else   → distal
V11_POCKET_DISTANCE_LINING_THRESHOLD_AA = 5.0
V11_POCKET_DISTANCE_PROXIMAL_THRESHOLD_AA = 8.0
V11_POCKET_DISTANCE_NEAR_THRESHOLD_AA = 12.0
_V11_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
# Strict tier (existing behaviour): highly-conserved teleosts × strongly
# diverged rodents. Captures the most clear-cut "fish-conserved, rodent-lost"
# sites; few in number but high-confidence.
V11_TELEOST_CONSERVED_RODENT_DIVERGED_TELEOST_THRESHOLD = 0.70
V11_TELEOST_CONSERVED_RODENT_DIVERGED_RODENT_THRESHOLD = 0.50
# Mid tier (added 2026-06-01 per user feedback): positions where teleosts
# have OK conservation and rodents are lacking, but the absolute values
# don't meet the strict cut. Catches the broader landscape — especially
# rodent-lacking positions with moderate but real teleost agreement.
V11_TELEOST_CONSERVED_RODENT_DIVERGED_TELEOST_THRESHOLD_MID = 0.55
V11_TELEOST_CONSERVED_RODENT_DIVERGED_RODENT_THRESHOLD_MID = 0.65
V11_TELEOST_CONSERVED_RODENT_DIVERGED_MIN_DELTA_MID = 0.25
# Broad tier (added 2026-06-01 follow-up): fish identity is meaningfully
# HIGHER than rodent identity, even though both are above the mid-tier
# rodent ceiling. Catches "fish > rodent" gaps that the strict / mid cuts
# miss — e.g. DHRS7 position 160 (Teleosts=0.96, Rodents=0.75, Δ=0.21):
# fish is clearly more conserved than rodents but the absolute thresholds
# aren't tripped. Mutually exclusive with strict and mid.
V11_TELEOST_CONSERVED_RODENT_DIVERGED_TELEOST_THRESHOLD_BROAD = 0.70
V11_TELEOST_CONSERVED_RODENT_DIVERGED_MIN_DELTA_BROAD = 0.15


def _v11_highlight_teleost_rodent_diverged_tiered(df: pd.DataFrame,
                                                  teleost_col: str = "Teleosts_IdentityFraction",
                                                  rodent_col: str = "Rodents_IdentityFraction",
                                                  teleost_min_strict: float = V11_TELEOST_CONSERVED_RODENT_DIVERGED_TELEOST_THRESHOLD,
                                                  rodent_max_strict: float = V11_TELEOST_CONSERVED_RODENT_DIVERGED_RODENT_THRESHOLD,
                                                  teleost_min_mid: float = V11_TELEOST_CONSERVED_RODENT_DIVERGED_TELEOST_THRESHOLD_MID,
                                                  rodent_max_mid: float = V11_TELEOST_CONSERVED_RODENT_DIVERGED_RODENT_THRESHOLD_MID,
                                                  min_delta_mid: float = V11_TELEOST_CONSERVED_RODENT_DIVERGED_MIN_DELTA_MID,
                                                  teleost_min_broad: float = V11_TELEOST_CONSERVED_RODENT_DIVERGED_TELEOST_THRESHOLD_BROAD,
                                                  min_delta_broad: float = V11_TELEOST_CONSERVED_RODENT_DIVERGED_MIN_DELTA_BROAD,
                                                  ) -> Dict[str, List[int]]:
    """Return tiered lists of fish-conserved, rodent-lost positions.

    Returns {"strict": [...], "mid": [...], "broad": [...]} — the three
    lists are mutually exclusive (each position is reported in at most
    one tier, in priority strict > mid > broad).

    - strict: Teleosts ≥ teleost_min_strict AND Rodents ≤ rodent_max_strict
      (currently 0.70 / 0.50 → the original tight cut)
    - mid:    NOT strict, AND Teleosts ≥ teleost_min_mid (0.55) AND
              Rodents ≤ rodent_max_mid (0.65) AND
              (Teleosts − Rodents) ≥ min_delta_mid (0.25)
      Captures "teleost ok, rodent lacking" positions where the absolute
      conservation isn't extreme but the cross-clade gap is real and large.
    - broad:  NOT strict AND NOT mid, AND Teleosts ≥ teleost_min_broad
              (0.70) AND (Teleosts − Rodents) ≥ min_delta_broad (0.15)
      Captures the broader "fish noticeably better than rodent" landscape
      even when rodent identity is moderately high (e.g. 0.65–0.85). User
      example: DHRS7 pos 160 (T=0.96, R=0.75, Δ=0.21) — fish is clearly
      more conserved with human but the mid cuts miss it because R > 0.65.
    """
    empty: Dict[str, List[int]] = {"strict": [], "mid": [], "broad": []}
    if df is None or df.empty:
        return empty
    if teleost_col not in df.columns or rodent_col not in df.columns:
        return empty
    if "ReferenceResidueNumber" not in df.columns:
        return empty
    sub = df[["ReferenceResidueNumber", teleost_col, rodent_col]].dropna()
    if sub.empty:
        return empty
    teleo = sub[teleost_col].astype(float)
    rod = sub[rodent_col].astype(float)
    delta = teleo - rod

    strict_mask = (teleo >= float(teleost_min_strict)) & (rod <= float(rodent_max_strict))
    mid_mask = (~strict_mask) & (
        (teleo >= float(teleost_min_mid)) &
        (rod <= float(rodent_max_mid)) &
        (delta >= float(min_delta_mid))
    )
    broad_mask = (~strict_mask) & (~mid_mask) & (
        (teleo >= float(teleost_min_broad)) &
        (delta >= float(min_delta_broad))
    )
    strict_positions = [int(p) for p in sub.loc[strict_mask, "ReferenceResidueNumber"].tolist()]
    mid_positions = [int(p) for p in sub.loc[mid_mask, "ReferenceResidueNumber"].tolist()]
    broad_positions = [int(p) for p in sub.loc[broad_mask, "ReferenceResidueNumber"].tolist()]
    return {"strict": strict_positions, "mid": mid_positions, "broad": broad_positions}


def _v11_highlight_teleost_rodent_diverged(df: pd.DataFrame,
                                           teleost_col: str = "Teleosts_IdentityFraction",
                                           rodent_col: str = "Rodents_IdentityFraction",
                                           teleost_min: float = V11_TELEOST_CONSERVED_RODENT_DIVERGED_TELEOST_THRESHOLD,
                                           rodent_max: float = V11_TELEOST_CONSERVED_RODENT_DIVERGED_RODENT_THRESHOLD,
                                           include_mid: bool = False) -> List[int]:
    """Return reference positions where Teleosts is highly conserved (≥
    teleost_min) AND Rodents has substantially diverged (≤ rodent_max).

    When `include_mid=True` ALSO includes the mid-tier positions (broader
    threshold; see _v11_highlight_teleost_rodent_diverged_tiered docstring).
    Default False so back-compat callers only get the strict tier.
    """
    if df is None or df.empty:
        return []
    if teleost_col not in df.columns or rodent_col not in df.columns:
        return []
    if "ReferenceResidueNumber" not in df.columns:
        return []
    if not include_mid:
        sub = df[["ReferenceResidueNumber", teleost_col, rodent_col]].dropna()
        if sub.empty:
            return []
        mask = (sub[teleost_col] >= float(teleost_min)) & (sub[rodent_col] <= float(rodent_max))
        return [int(p) for p in sub.loc[mask, "ReferenceResidueNumber"].tolist()]
    tiers = _v11_highlight_teleost_rodent_diverged_tiered(df, teleost_col, rodent_col,
                                                         teleost_min_strict=teleost_min,
                                                         rodent_max_strict=rodent_max)
    return tiers["strict"] + tiers["mid"]


def v11_write_teleost_conserved_rodent_diverged_sites(csv_path: Path | str, outdir: Path | str,
                                                      teleost_col: str = "Teleosts_IdentityFraction",
                                                      rodent_col: str = "Rodents_IdentityFraction") -> Optional[Path]:
    """Emit a per-position CSV of the "fish-conserved, rodent-lost" sites
    (positions where teleosts are highly conserved with human but rodents are
    not). Driven by a 9-group / mod / combined clade-identity CSV; columns:
    reference_position, reference_residue, teleost_identity, rodent_identity,
    delta (= teleost - rodent), plus per-group identity fractions for primates
    / OtherMammals / Birds / Reptiles / Amphibians when available.
    """
    csv_path = Path(csv_path)
    outdir = Path(outdir)
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:  # noqa: BLE001
        return None
    if df.empty or teleost_col not in df.columns or rodent_col not in df.columns:
        return None
    tiers = _v11_highlight_teleost_rodent_diverged_tiered(df, teleost_col, rodent_col)
    strict_set = set(tiers.get("strict") or [])
    mid_set = set(tiers.get("mid") or [])
    broad_set = set(tiers.get("broad") or [])
    all_positions = sorted(strict_set | mid_set | broad_set)
    if not all_positions:
        out = outdir / V11_TELEOST_CONSERVED_RODENT_DIVERGED_CSV
        pd.DataFrame(columns=[
            "reference_position", "reference_residue", "tier",
            "teleost_identity", "rodent_identity", "delta_teleost_minus_rodent",
        ]).to_csv(out, index=False)
        return out
    keep = df[df["ReferenceResidueNumber"].isin(all_positions)].copy()
    out_rows: List[Dict[str, Any]] = []
    extra_cols = [c for c in (
        "Primates_IdentityFraction", "OtherMammals_IdentityFraction",
        "Birds_IdentityFraction", "Reptiles_IdentityFraction",
        "Amphibians_IdentityFraction", "OtherFish_IdentityFraction",
        "Other_IdentityFraction", "OtherVertebrates_IdentityFraction", "x_IdentityFraction",
    ) if c in df.columns]
    for _, row in keep.iterrows():
        pos = int(row["ReferenceResidueNumber"])
        if pos in strict_set:
            tier = "strict"
        elif pos in mid_set:
            tier = "mid"
        elif pos in broad_set:
            tier = "broad"
        else:
            tier = ""
        out_rows.append({
            "reference_position": pos,
            "reference_residue": row.get("reference_residue", ""),
            "tier": tier,
            "teleost_identity": float(row[teleost_col]),
            "rodent_identity": float(row[rodent_col]),
            "delta_teleost_minus_rodent": float(row[teleost_col] - row[rodent_col]),
            **{c.replace("_IdentityFraction", "_identity"): float(row[c]) if isinstance(row[c], (int, float)) and np.isfinite(row[c]) else None for c in extra_cols},
        })
    out = outdir / V11_TELEOST_CONSERVED_RODENT_DIVERGED_CSV
    # Sort: strict tier first, then mid, then broad, then by position.
    out_df = pd.DataFrame(out_rows)
    if not out_df.empty:
        out_df["_tier_order"] = out_df["tier"].map({"strict": 0, "mid": 1, "broad": 2}).fillna(3)
        out_df = out_df.sort_values(["_tier_order", "reference_position"]).drop(columns=["_tier_order"])
    out_df.to_csv(out, index=False)
    return out


def _v11_parse_pdb_ca_atoms(pdb_path: Path) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[int, str]]:
    """Parse a PDB file and return (residue_number -> (x, y, z)) for CA atoms
    plus (residue_number -> one-letter AA) lookup. Skips alt-loc entries past
    the first one encountered for each residue."""
    coords: Dict[int, Tuple[float, float, float]] = {}
    aa1: Dict[int, str] = {}
    with pdb_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                resi = int(line[22:26])
            except ValueError:
                continue
            if resi in coords:
                continue  # honour first occurrence (alt-loc A)
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            coords[resi] = (x, y, z)
            aa1[resi] = _V11_AA3_TO_1.get(line[17:20].strip(), "X")
    return coords, aa1


def _v11_compute_per_clade_consensus_at_positions(
    alignment: Any,
    reference_species: str,
    positions: Sequence[int],
    taxonomy_lookup: Optional[Dict[str, Any]] = None,
    individual_species_keys: Sequence[str] = ("mus_musculus", "danio_rerio"),
) -> Dict[int, Dict[str, Any]]:
    """For every reference position in `positions`, return:
      • teleost_consensus_aa + teleost_consensus_pct + teleost_n_sampled
        — most common AA across `Teleostei` records and its frequency.
      • rodent_consensus_aa + rodent_consensus_pct + rodent_n_sampled
        — same for the 9-group `Rodents` bucket (Mammalia narrowed by
        genus token; same resolver the bubble grid uses).
      • For every species key in `individual_species_keys`, the single
        residue that species carries at this position, exposed as
        `{species_key}_aa`. Default tracks Mus musculus + Danio rerio —
        the canonical model rodent + the mandatory teleost reference in
        V11's design.

    Returns: {pos: {...above keys}}. Positions that don't map back to a
    reference column (gap in human) are skipped.
    """
    if not positions:
        return {}
    try:
        aln_to_ref, ref_record = _v11_ref_position_records(alignment, reference_species)
    except Exception:  # noqa: BLE001
        return {}
    if ref_record is None:
        return {}
    pos_to_col: Dict[int, int] = {}
    for col_idx, ref_pos in enumerate(aln_to_ref):
        if ref_pos is not None:
            pos_to_col[int(ref_pos)] = col_idx

    species_clade_broad = _v11_species_clade_map(alignment, taxonomy_lookup)

    teleost_records: List[Any] = []
    rodent_records: List[Any] = []
    # Normalise individual species keys to lower-case once.
    indiv_keys = tuple(str(k).strip().lower() for k in (individual_species_keys or ()) if str(k).strip())
    individual_records: Dict[str, Any] = {k: None for k in indiv_keys}

    for rec in alignment:
        species, _sym = parse_header_species_symbol(str(rec.id))
        sp_lower = species.lower()
        broad = species_clade_broad.get(str(rec.id), "")
        if broad == "Teleostei":
            teleost_records.append(rec)
        try:
            grouped = v11_resolve_grouped_clade(species, broad)
        except Exception:  # noqa: BLE001
            grouped = ""
        if grouped == "Rodents":
            rodent_records.append(rec)
        if sp_lower in individual_records and individual_records[sp_lower] is None:
            individual_records[sp_lower] = rec

    def _consensus(records: Sequence[Any], col_idx: int) -> Tuple[str, float, int]:
        ctr: Dict[str, int] = {}
        for rec in records:
            seq = str(rec.seq)
            if col_idx < len(seq):
                aa = seq[col_idx].upper()
                if aa in GAP_CHARS or aa in {"X", "?"}:
                    continue
                ctr[aa] = ctr.get(aa, 0) + 1
        if not ctr:
            return ("-", float("nan"), 0)
        total = sum(ctr.values())
        aa, cnt = max(ctr.items(), key=lambda kv: (kv[1], -ord(kv[0])))
        return (aa, (cnt / total) if total else float("nan"), total)

    def _residue_at(rec: Any, col_idx: int) -> Tuple[str, Optional[int]]:
        """Return (one-letter AA, species-specific 1-based position) at the
        alignment column for this species's record. Position is the count
        of non-gap residues in the species's sequence from index 0 through
        and including `col_idx`, so it matches how the species's own
        ungapped protein is numbered. Returns ('-', None) when gap / X."""
        if rec is None:
            return ("-", None)
        seq = str(rec.seq)
        if col_idx >= len(seq):
            return ("-", None)
        ch = seq[col_idx].upper()
        if ch in GAP_CHARS or ch in {"X", "?"}:
            return ("-", None)
        # 1-based species position = count of non-gap, non-X residues from
        # start of the species's aligned sequence up to and including the
        # column. Used so mutagenesis primers can target the EXACT
        # species-specific residue number (e.g. mouse V161, zebrafish L172).
        count = 0
        for c in seq[: col_idx + 1]:
            cu = c.upper()
            if cu in GAP_CHARS or cu in {"X", "?"}:
                continue
            count += 1
        return (ch, count)

    out: Dict[int, Dict[str, Any]] = {}
    for pos in positions:
        try:
            p = int(pos)
        except (TypeError, ValueError):
            continue
        col_idx = pos_to_col.get(p)
        if col_idx is None:
            continue
        t_aa, t_pct, t_n = _consensus(teleost_records, col_idx)
        r_aa, r_pct, r_n = _consensus(rodent_records, col_idx)
        entry: Dict[str, Any] = {
            "teleost_consensus_aa": t_aa,
            "teleost_consensus_pct": (round(t_pct, 3) if math.isfinite(t_pct) else None),
            "teleost_n_sampled": int(t_n),
            "rodent_consensus_aa": r_aa,
            "rodent_consensus_pct": (round(r_pct, 3) if math.isfinite(r_pct) else None),
            "rodent_n_sampled": int(r_n),
        }
        for sp_key in indiv_keys:
            aa, sp_pos = _residue_at(individual_records.get(sp_key), col_idx)
            entry[f"{sp_key}_aa"] = aa
            entry[f"{sp_key}_pos"] = sp_pos
        # Also record the alignment column for downstream verification.
        entry["alignment_column_index"] = int(col_idx)
        out[p] = entry
    return out


def _v11_pocket_proximity_class(distance_aa: Optional[float]) -> str:
    """Categorical proximity bucket for a Cα-Cα distance in Angstroms."""
    if distance_aa is None or not math.isfinite(distance_aa):
        return "unknown"
    if distance_aa <= V11_POCKET_DISTANCE_LINING_THRESHOLD_AA:
        return "pocket_lining"
    if distance_aa <= V11_POCKET_DISTANCE_PROXIMAL_THRESHOLD_AA:
        return "pocket_proximal"
    if distance_aa <= V11_POCKET_DISTANCE_NEAR_THRESHOLD_AA:
        return "near_pocket"
    return "distal"


def v11_write_highlight_sites_pocket_distance(
    outdir: Path | str,
    gene_label: str,
    active_site_residues: Optional[Sequence[Any]] = None,
    *,
    alignment: Any = None,
    reference_species: str = "homo_sapiens",
    taxonomy_lookup: Optional[Dict[str, Any]] = None,
    sites_csv_filename: str = V11_TELEOST_CONSERVED_RODENT_DIVERGED_CSV,
    pdb_filename: str = ALPHAFOLD_MODEL_FILENAME,
    output_filename: str = V11_HIGHLIGHT_SITES_POCKET_DISTANCE_CSV,
) -> Optional[Path]:
    """For every fish-conserved / rodent-diverged highlight position, compute
    the Cα-Cα distance to the gene's nearest hand-curated active-site
    residue plus the distance to the active-site centroid. Output is the
    existing sites CSV augmented with five new columns and SORTED BY
    DISTANCE ASCENDING (closest-to-pocket first) so the most likely
    function-impacting sites land at the top of the table.

    Parameters
    ----------
    outdir : Path
        V11 output directory containing the highlight-sites CSV and the
        human reference AlphaFold PDB.
    gene_label : str
        Gene symbol, used for the output filename / annotation only.
    active_site_residues : Sequence
        Iterable of either bare integers (residue numbers) or 2-tuples
        ``(position, label)`` where label is a human-friendly catalytic
        annotation like ``"S228 nucleophile"``. When None or empty, the
        helper emits the file with distance columns set to None and a
        ``pocket_proximity_class`` of ``"n/a (no active-site definition)"``
        so downstream tooling still finds the expected schema for any gene.

    New columns
    -----------
    - ``nearest_active_site_position`` (int)
    - ``nearest_active_site_residue`` (str, e.g. ``"S228 nucleophile"``)
    - ``min_CA_distance_to_active_site_AA`` (float, 2 dp)
    - ``CA_distance_to_pocket_centroid_AA`` (float, 2 dp)
    - ``pocket_proximity_class`` (categorical: pocket_lining / pocket_proximal
      / near_pocket / distal)
    """
    outdir = Path(outdir)
    sites_csv = outdir / sites_csv_filename
    pdb_path = outdir / pdb_filename
    if not sites_csv.exists() or not pdb_path.exists():
        return None
    try:
        sites_df = pd.read_csv(sites_csv)
    except Exception:  # noqa: BLE001
        return None
    if sites_df.empty:
        out_path = outdir / output_filename
        sites_df.assign(
            nearest_active_site_position=pd.Series(dtype="float"),
            nearest_active_site_residue=pd.Series(dtype="object"),
            min_CA_distance_to_active_site_AA=pd.Series(dtype="float"),
            CA_distance_to_pocket_centroid_AA=pd.Series(dtype="float"),
            pocket_proximity_class=pd.Series(dtype="object"),
        ).to_csv(out_path, index=False)
        return out_path

    ca_coords, aa1_by_resi = _v11_parse_pdb_ca_atoms(pdb_path)

    # Normalise active-site spec into a list of (position, label) tuples.
    active_pairs: List[Tuple[int, str]] = []
    for entry in (active_site_residues or []):
        pos: Optional[int] = None
        label: Optional[str] = None
        if isinstance(entry, (tuple, list)) and len(entry) >= 1:
            try:
                pos = int(entry[0])
            except (TypeError, ValueError):
                continue
            if len(entry) >= 2 and entry[1] is not None:
                label = str(entry[1])
        else:
            try:
                pos = int(entry)  # bare int / numpy int
            except (TypeError, ValueError):
                continue
        if pos is None:
            continue
        if not label:
            aa = aa1_by_resi.get(pos, "X")
            label = f"{aa}{pos}"
        active_pairs.append((pos, label))

    # If no active-site info, still produce the file with annotated columns
    # so downstream tooling sees a consistent schema.
    no_active_site = not active_pairs
    centroid: Optional[Tuple[float, float, float]] = None
    if not no_active_site:
        active_coords = [ca_coords[p] for p, _ in active_pairs if p in ca_coords]
        if active_coords:
            n = len(active_coords)
            centroid = (
                sum(c[0] for c in active_coords) / n,
                sum(c[1] for c in active_coords) / n,
                sum(c[2] for c in active_coords) / n,
            )

    def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        dx = a[0] - b[0]; dy = a[1] - b[1]; dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    enriched_rows: List[Dict[str, Any]] = []
    for _, row in sites_df.iterrows():
        rec: Dict[str, Any] = {col: row[col] for col in sites_df.columns}
        nearest_pos: Optional[int] = None
        nearest_label: Optional[str] = None
        min_d: Optional[float] = None
        centroid_d: Optional[float] = None
        class_label: str

        try:
            pos = int(row["reference_position"])
        except (KeyError, ValueError, TypeError):
            class_label = "missing_position"
            enriched_rows.append({
                **rec,
                "nearest_active_site_position": None,
                "nearest_active_site_residue": None,
                "min_CA_distance_to_active_site_AA": None,
                "CA_distance_to_pocket_centroid_AA": None,
                "pocket_proximity_class": class_label,
            })
            continue

        highlight_xyz = ca_coords.get(pos)
        if no_active_site:
            class_label = "n/a (no active-site definition for this gene)"
        elif highlight_xyz is None:
            class_label = "highlight_not_in_PDB"
        elif centroid is None:
            class_label = "active_sites_not_in_PDB"
        else:
            best_d = float("inf")
            for ap, alabel in active_pairs:
                ac = ca_coords.get(ap)
                if ac is None:
                    continue
                dd = _dist(highlight_xyz, ac)
                if dd < best_d:
                    best_d = dd
                    nearest_pos = ap
                    nearest_label = alabel
            if math.isfinite(best_d):
                min_d = round(best_d, 2)
                centroid_d = round(_dist(highlight_xyz, centroid), 2)
                class_label = _v11_pocket_proximity_class(min_d)
            else:
                class_label = "active_sites_not_in_PDB"

        enriched_rows.append({
            **rec,
            "nearest_active_site_position": nearest_pos,
            "nearest_active_site_residue": nearest_label,
            "min_CA_distance_to_active_site_AA": min_d,
            "CA_distance_to_pocket_centroid_AA": centroid_d,
            "pocket_proximity_class": class_label,
        })

    out_df = pd.DataFrame(enriched_rows)

    # Per-clade consensus residue at each highlight position: what AA do
    # the teleosts carry (consensus + fraction), what do the rodents
    # carry, and specifically what does Mus musculus have? Empty / "-"
    # when alignment is None or the position doesn't map back to a
    # reference column (e.g. gap in human reference).
    if alignment is not None and not out_df.empty:
        try:
            positions = [int(p) for p in out_df["reference_position"].dropna().tolist()]
        except Exception:  # noqa: BLE001
            positions = []
        try:
            per_clade = _v11_compute_per_clade_consensus_at_positions(
                alignment, reference_species, positions, taxonomy_lookup,
            )
        except Exception:  # noqa: BLE001
            per_clade = {}
        if per_clade:
            # Use the union of keys across all per-position entries so any
            # individual-species residue columns (e.g. mus_musculus_aa,
            # danio_rerio_aa) flow through automatically.
            all_keys: List[str] = []
            seen: set = set()
            for entry in per_clade.values():
                for k in entry.keys():
                    if k not in seen:
                        seen.add(k)
                        all_keys.append(k)
            new_cols: Dict[str, List[Any]] = {k: [] for k in all_keys}
            for _, row in out_df.iterrows():
                try:
                    p = int(row["reference_position"])
                except (KeyError, ValueError, TypeError):
                    p = None
                entry = per_clade.get(p) if p is not None else None
                for k in all_keys:
                    new_cols[k].append(entry.get(k) if entry else None)
            for k, vals in new_cols.items():
                out_df[k] = vals

    # V11.1 pillar-5 enrichment: if v11_pocket_residues.csv exists in the
    # same dir, left-merge its per-residue structural classification onto
    # the highlight rows. Adds packing_classification (core /
    # pocket_lining / surface from Cα-packing density), is_substrate_pocket
    # (boolean: residue is within 8 Å of any active-site Cα — pillar-5's
    # definition), relative_burial (SASA proxy, 0..1) and plddt. Silently
    # skipped when the pocket CSV is absent (e.g. pipeline-time call
    # before the V11.1 module has run).
    pocket_resi_csv = outdir / V11_POCKET_RESIDUES_CSV
    if pocket_resi_csv.exists():
        try:
            pkt = pd.read_csv(pocket_resi_csv)
            keep_cols = [c for c in (
                "residue_number", "packing_classification", "is_substrate_pocket",
                "relative_burial", "plddt",
            ) if c in pkt.columns]
            if "residue_number" in keep_cols and len(keep_cols) > 1:
                pkt_small = pkt[keep_cols].copy()
                out_df = out_df.merge(
                    pkt_small,
                    left_on="reference_position", right_on="residue_number",
                    how="left",
                )
                if "residue_number" in out_df.columns:
                    out_df = out_df.drop(columns=["residue_number"])
        except Exception:  # noqa: BLE001
            pass

    # Flag whether Mus musculus carries a residue DIFFERENT from the human
    # reference at this position. Goes hand-in-hand with the consensus
    # columns; True for both "diverged" and "altogether different" mouse
    # residues. Striking signal when combined with fish-conserved tier.
    if "mus_musculus_aa" in out_df.columns and "reference_residue" in out_df.columns:
        def _diverged(row):
            ref = str(row.get("reference_residue") or "").strip().upper()
            mus = str(row.get("mus_musculus_aa") or "").strip().upper()
            if not ref or not mus or mus == "-":
                return None
            return ref != mus
        out_df["mouse_diverged_from_human"] = out_df.apply(_diverged, axis=1)

    # Sort closest-first (NaN distances drop to bottom); within identical
    # distance keep strict above mid above broad.
    if "min_CA_distance_to_active_site_AA" in out_df.columns:
        tier_order = out_df.get("tier", pd.Series("strict", index=out_df.index)).map(
            {"strict": 0, "mid": 1, "broad": 2}
        ).fillna(3)
        out_df = out_df.assign(_tier_order=tier_order)
        out_df = out_df.sort_values(
            ["min_CA_distance_to_active_site_AA", "_tier_order", "reference_position"],
            ascending=[True, True, True],
            na_position="last",
        ).drop(columns=["_tier_order"])

    out_path = outdir / output_filename
    out_df.to_csv(out_path, index=False)
    return out_path


def _v11_render_allosteric_summary_figures(
    allosteric_summary_csv: Path,
    gene_label: str,
    *,
    coevolution_csv: Optional[Path] = None,
    coevol_top_k_per_source: int = 3,
    max_rows_per_page: int = 22,
) -> List[Any]:
    """Render the pilot allosteric / coevolution outputs as PDF pages.

    Two-page-style output:
      Page A — per-highlight allosteric profile (one row per highlight
               residue): tier band, distance proximity class, direct
               contact counts, candidate-allosteric partner list,
               nearest-allo-residue / hops.
      Page B — top mutual-information coevolution partners (up to K rows
               per highlight source): partner residue, Cα-Cα distance,
               raw MI in bits, coupling signal (proximal vs remote vs
               substrate-pocket vs candidate-allosteric).

    Returns [] when the input CSV is missing.
    """
    if not allosteric_summary_csv or not Path(allosteric_summary_csv).exists():
        return []
    try:
        summary_df = pd.read_csv(allosteric_summary_csv)
    except Exception:  # noqa: BLE001
        return []
    if summary_df.empty:
        return []

    figs: List[Any] = []

    # ---- Page A: allosteric profile ---- #
    TIER_BANDS = {"strict": "#fbbf24", "mid": "#fef3c7"}
    ROW_BG_DEFAULT = "#f5f7fa"
    ALLO_CONTACT_BG = "#fecaca"  # rose when row has any candidate allo contact

    def _split_chunks(df_in: pd.DataFrame, n: int) -> List[pd.DataFrame]:
        return [df_in.iloc[i : i + n] for i in range(0, len(df_in), n)] or [df_in]

    A_cols = [
        ("source_position", "Pos"),
        ("source_residue", "AA"),
        ("source_tier", "Tier"),
        ("source_proximity_class", "Proximity (distance)"),
        ("source_packing_classification", "Packing class"),
        ("n_direct_contacts", "# direct contacts"),
        ("n_contacts_in_substrate_pocket", "# substr. pocket"),
        ("n_contacts_with_candidate_allosteric_residues", "# cand. allosteric"),
        ("direct_candidate_allosteric_partners", "Direct allo partners"),
        ("nearest_candidate_allosteric_residue", "Nearest allo"),
        ("nearest_candidate_allosteric_hops", "hops"),
    ]
    A_cols = [(k, h) for k, h in A_cols if k in summary_df.columns]
    summary_sorted = summary_df.sort_values(
        ["n_contacts_with_candidate_allosteric_residues",
         "n_direct_contacts"],
        ascending=[False, False],
        na_position="last",
    )
    chunks_A = _split_chunks(summary_sorted, max_rows_per_page)
    n_pages_A = len(chunks_A)
    for page_idx, chunk in enumerate(chunks_A):
        rows_text: List[List[str]] = []
        rows_bg: List[List[str]] = []
        for _, row in chunk.iterrows():
            tier_val = str(row.get("source_tier") or "").strip().lower()
            tier_bg = TIER_BANDS.get(tier_val)
            n_allo = row.get("n_contacts_with_candidate_allosteric_residues")
            has_allo = bool(n_allo and float(n_allo) > 0)
            base_bg = ALLO_CONTACT_BG if has_allo else ROW_BG_DEFAULT
            txt: List[str] = []
            bg: List[str] = []
            for k, _h in A_cols:
                v = row.get(k)
                if pd.isna(v):
                    txt.append("—")
                elif isinstance(v, float) and v.is_integer():
                    txt.append(str(int(v)))
                elif isinstance(v, float):
                    txt.append(f"{v:.2f}")
                else:
                    txt.append(str(v) if str(v).strip() else "—")
                if k == "source_tier" and tier_bg:
                    bg.append(tier_bg)
                else:
                    bg.append(base_bg)
            rows_text.append(txt)
            rows_bg.append(bg)
        headers = [h for _k, h in A_cols]
        fig = plt.figure(figsize=(13.0, max(4.5, 0.40 * (len(chunk) + 4))))
        title_extra = f"  (page {page_idx + 1} of {n_pages_A})" if n_pages_A > 1 else ""
        fig.suptitle(
            f"{gene_label} — PILOT: structural allosteric profile of highlight residues{title_extra}\n"
            f"Cα contact graph (≤ 8 Å, |i−j| ≥ 3); candidate allosteric residue = V11.1 packing 'pocket_lining' AND NOT in substrate pocket",
            fontsize=11, y=0.97,
        )
        ax = fig.add_subplot(111)
        ax.axis("off")
        table = ax.table(
            cellText=rows_text, cellColours=rows_bg, colLabels=headers,
            loc="upper center", cellLoc="center", colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.0)
        table.auto_set_column_width(col=list(range(len(headers))))
        table.scale(1.0, 1.4)
        for col_idx in range(len(headers)):
            try:
                hcell = table[0, col_idx]
            except KeyError:
                continue
            hcell.set_facecolor("#1f2937")
            hcell.set_text_props(color="white", weight="bold")
        for r_idx, (_, row) in enumerate(chunk.iterrows(), start=1):
            if str(row.get("source_tier") or "").strip().lower() == "strict":
                for col_idx in range(len(headers)):
                    try:
                        c = table[r_idx, col_idx]
                    except KeyError:
                        continue
                    c.set_text_props(weight="bold")
        footer = (
            "Row colour: ROSE when residue has ≥1 direct contact with a candidate allosteric "
            "(pocket_lining, NOT in substrate pocket) residue. Tier cell: dark amber = strict, "
            "light amber = mid. Pilot only — not a production allosteric mapper; needs DCA/MD/wet-lab follow-up."
        )
        fig.text(0.5, 0.025, footer, ha="center", fontsize=7.5, color="#374151", wrap=True)
        figs.append(fig)

    # ---- Page B: top MI coevolution partners ---- #
    if coevolution_csv is not None and Path(coevolution_csv).exists():
        try:
            coevol_df = pd.read_csv(coevolution_csv)
        except Exception:  # noqa: BLE001
            coevol_df = pd.DataFrame()
        if not coevol_df.empty:
            COEVOL_SIGNAL_BG = {
                "proximal_co_evolving": "#bbf7d0",                                  # emerald-200
                "remote_co_evolving_at_candidate_allosteric_residue": "#fda4af",    # rose-300
                "co_evolving_with_substrate_pocket": "#fed7aa",                     # orange-200
                "remote_co_evolving": "#fef9c3",                                    # yellow-100
            }
            B_cols = [
                ("source_position", "Source Pos"),
                ("source_residue", "Src AA"),
                ("source_tier", "Tier"),
                ("partner_position", "Partner Pos"),
                ("partner_residue", "Ptn AA"),
                ("partner_packing_classification", "Partner packing"),
                ("partner_is_substrate_pocket", "Sub. pocket?"),
                ("ca_distance_source_partner_AA", "Cα dist (Å)"),
                ("mutual_information_bits", "MI (bits)"),
                ("n_species_paired", "n paired"),
                ("coupling_signal", "Coupling signal"),
            ]
            B_cols = [(k, h) for k, h in B_cols if k in coevol_df.columns]
            # Keep top-K per source by MI desc, then concat in source order.
            coevol_df = coevol_df.copy()
            coevol_df = coevol_df.sort_values(
                ["source_position", "mutual_information_bits"], ascending=[True, False]
            )
            limited = coevol_df.groupby("source_position", sort=False).head(coevol_top_k_per_source)
            chunks_B = _split_chunks(limited, max_rows_per_page)
            n_pages_B = len(chunks_B)
            for page_idx, chunk in enumerate(chunks_B):
                rows_text = []
                rows_bg = []
                last_source: Optional[int] = None
                for _, row in chunk.iterrows():
                    tier_val = str(row.get("source_tier") or "").strip().lower()
                    tier_bg = TIER_BANDS.get(tier_val)
                    signal = str(row.get("coupling_signal") or "").strip()
                    base_bg = COEVOL_SIGNAL_BG.get(signal, ROW_BG_DEFAULT)
                    txt = []
                    bg = []
                    for k, _h in B_cols:
                        v = row.get(k)
                        if pd.isna(v):
                            txt.append("—")
                        elif k == "partner_is_substrate_pocket":
                            txt.append("Yes" if bool(v) else "No")
                        elif k == "mutual_information_bits":
                            txt.append(f"{float(v):.3f}")
                        elif k == "ca_distance_source_partner_AA":
                            txt.append(f"{float(v):.2f}" if isinstance(v, (int, float)) else str(v))
                        elif isinstance(v, float) and v.is_integer():
                            txt.append(str(int(v)))
                        else:
                            txt.append(str(v) if str(v).strip() else "—")
                        if k == "source_tier" and tier_bg:
                            bg.append(tier_bg)
                        else:
                            bg.append(base_bg)
                    rows_text.append(txt)
                    rows_bg.append(bg)
                    last_source = row.get("source_position")
                headers = [h for _k, h in B_cols]
                fig = plt.figure(figsize=(13.0, max(4.5, 0.40 * (len(chunk) + 4))))
                title_extra = f"  (page {page_idx + 1} of {n_pages_B})" if n_pages_B > 1 else ""
                fig.suptitle(
                    f"{gene_label} — PILOT: top mutual-information coevolution partners (top {coevol_top_k_per_source} / highlight)"
                    f"{title_extra}\n"
                    f"raw MI in bits across the species alignment (gap-/X-skipping joint distribution); NOT APC-corrected",
                    fontsize=11, y=0.97,
                )
                ax = fig.add_subplot(111)
                ax.axis("off")
                table = ax.table(
                    cellText=rows_text, cellColours=rows_bg, colLabels=headers,
                    loc="upper center", cellLoc="center", colLoc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8.0)
                table.auto_set_column_width(col=list(range(len(headers))))
                table.scale(1.0, 1.4)
                for col_idx in range(len(headers)):
                    try:
                        hcell = table[0, col_idx]
                    except KeyError:
                        continue
                    hcell.set_facecolor("#1f2937")
                    hcell.set_text_props(color="white", weight="bold")
                # Bold strict rows.
                for r_idx, (_, row) in enumerate(chunk.iterrows(), start=1):
                    if str(row.get("source_tier") or "").strip().lower() == "strict":
                        for col_idx in range(len(headers)):
                            try:
                                c = table[r_idx, col_idx]
                            except KeyError:
                                continue
                            c.set_text_props(weight="bold")
                footer = (
                    "Coupling signal cell colour: GREEN = proximal_co_evolving (direct interaction, Cα ≤ 8 Å); "
                    "ROSE = remote_co_evolving_at_candidate_allosteric_residue (high MI to a pocket_lining residue "
                    "outside the substrate pocket — candidate ALLOSTERIC coupling); ORANGE = co_evolving_with_substrate_pocket; "
                    "YELLOW = remote_co_evolving (other). Raw MI; APC correction is future work."
                )
                fig.text(0.5, 0.025, footer, ha="center", fontsize=7.5, color="#374151", wrap=True)
                figs.append(fig)

    return figs


def _v11_render_pocket_distance_table_figures(
    pocket_distance_csv: Path,
    gene_label: str,
    max_rows_per_page: int = 22,
) -> List[Any]:
    """Render the pocket-distance CSV as one or more matplotlib Figures
    formatted as a colour-banded table (one row per highlight site, sorted
    closest-first). Returns a list of Figures the caller can pump into an
    open `PdfPages` instance. Empty list when the CSV is missing or empty."""
    if not pocket_distance_csv or not Path(pocket_distance_csv).exists():
        return []
    try:
        df = pd.read_csv(pocket_distance_csv)
    except Exception:  # noqa: BLE001
        return []
    if df.empty:
        return []

    # Column layout (key, header, formatter)
    def _fmt_dist(v):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "—"
        return f"{float(v):.2f}"

    def _fmt_id(v):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "—"
        return f"{float(v):.3f}"

    def _fmt_pct(v):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "—"
        return f"{float(v) * 100:.0f}%"

    def _fmt_pocket_bool(v):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "—"
        if isinstance(v, str):
            return "Yes" if v.strip().lower() in {"true", "1", "yes", "y"} else "No"
        return "Yes" if bool(v) else "No"

    def _fmt_packing(v):
        if v is None or not isinstance(v, str) or not v.strip():
            return "—"
        # Compact label for tighter columns: "pocket_lining" -> "PocketLining"
        s = str(v).strip()
        parts = s.split("_")
        return "".join(p.capitalize() for p in parts) if "_" in s else s

    has_packing = "packing_classification" in df.columns
    has_substrate = "is_substrate_pocket" in df.columns
    has_burial = "relative_burial" in df.columns
    has_consensus = "teleost_consensus_aa" in df.columns and "rodent_consensus_aa" in df.columns
    has_mus = "mus_musculus_aa" in df.columns
    has_danio = "danio_rerio_aa" in df.columns

    def _fmt_aa_with_two_pcts(aa: Any, cons_pct: Any, id_pct: Any) -> str:
        """Combined consensus / identity display, e.g.:
              `L 95% | 95%`  when teleost consensus L at 95 % matches human (id 95 %)
              `F 97% | 0%`   when rodent consensus F at 97 % diverged from human (id 0 %)
        The header is `Teleost AA (cons % | id %)`."""
        if aa is None or (isinstance(aa, float) and not math.isfinite(aa)):
            return "—"
        letter = str(aa).strip()
        if not letter or letter == "-":
            return "—"
        def _pct(v):
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                return None
            try:
                return round(float(v) * 100)
            except (TypeError, ValueError):
                return None
        c = _pct(cons_pct)
        i = _pct(id_pct)
        if c is None and i is None:
            return letter
        c_s = f"{c}%" if c is not None else "—"
        i_s = f"{i}%" if i is not None else "—"
        return f"{letter} {c_s} | {i_s}"

    # Pre-compute the combined "AA cons% | id%" cells so the per-cell
    # formatter stays single-value.
    if has_consensus:
        df = df.copy()
        df["__teleost_display"] = [
            _fmt_aa_with_two_pcts(aa, cp, ip)
            for aa, cp, ip in zip(
                df["teleost_consensus_aa"], df["teleost_consensus_pct"],
                df.get("teleost_identity", [None] * len(df)),
            )
        ]
        df["__rodent_display"] = [
            _fmt_aa_with_two_pcts(aa, cp, ip)
            for aa, cp, ip in zip(
                df["rodent_consensus_aa"], df["rodent_consensus_pct"],
                df.get("rodent_identity", [None] * len(df)),
            )
        ]

    columns_spec: List[Tuple[str, str, Any]] = [
        ("reference_position", "Pos", lambda v: str(int(v)) if pd.notna(v) else "—"),
        ("reference_residue", "Human AA", lambda v: str(v) if pd.notna(v) else "—"),
        ("tier", "Tier", lambda v: str(v) if pd.notna(v) else "—"),
        ("nearest_active_site_residue", "Nearest active site",
         lambda v: str(v) if pd.notna(v) else "—"),
        ("min_CA_distance_to_active_site_AA", "Cα dist (Å)", _fmt_dist),
        ("pocket_proximity_class", "Proximity (distance)",
         lambda v: str(v) if pd.notna(v) else "—"),
    ]
    if has_packing:
        columns_spec.append(("packing_classification", "Packing class (V11.1)", _fmt_packing))
    if has_substrate:
        columns_spec.append(("is_substrate_pocket", "≤8 Å pocket?", _fmt_pocket_bool))
    if has_burial:
        columns_spec.append(("relative_burial", "Buried %", _fmt_pct))
    # Per-clade consensus residue at this position. Columns merge consensus
    # AA + its frequency in the clade with the clade-to-human identity %
    # (separated by '|') so the user sees both signals in one cell.
    if has_consensus:
        columns_spec.append(("__teleost_display", "Teleost AA (cons % | id %)",
                             lambda v: str(v) if v else "—"))
        columns_spec.append(("__rodent_display",  "Rodent AA (cons % | id %)",
                             lambda v: str(v) if v else "—"))
    # Individual canonical model species — full Latin names in headers,
    # cell tinted by match-vs-human. Danio rerio first (canonical model
    # teleost / mandatory zebrafish ortholog), then Mus musculus
    # (canonical model rodent), per user-preferred order.
    # Cells now show the species's OWN residue position (e.g. "V161" /
    # "L172") so mutagenesis primers can target the exact species
    # position; falls back to just the letter if position is missing.
    def _fmt_species_cell(letter: Any, pos: Any) -> str:
        if letter is None or (isinstance(letter, float) and not math.isfinite(letter)):
            return "—"
        s = str(letter).strip()
        if not s or s == "-":
            return "—"
        try:
            if pos is None or (isinstance(pos, float) and not math.isfinite(pos)):
                return s
            p = int(pos)
            if p <= 0:
                return s
            return f"{s}{p}"
        except (TypeError, ValueError):
            return s

    if has_danio:
        if "danio_rerio_pos" in df.columns:
            df = df.copy() if not getattr(df, "_pd_cell_copy", False) else df
            df["__danio_cell"] = [
                _fmt_species_cell(aa, p)
                for aa, p in zip(df["danio_rerio_aa"], df["danio_rerio_pos"])
            ]
            columns_spec.append(("__danio_cell", "danio_rerio (AA + pos)",
                                 lambda v: str(v) if v else "—"))
        else:
            columns_spec.append(("danio_rerio_aa", "danio_rerio",
                                 lambda v: (str(v) if (isinstance(v, str) and v and v != "-") else "—")))
    if has_mus:
        if "mus_musculus_pos" in df.columns:
            df = df.copy() if not getattr(df, "_pd_cell_copy", False) else df
            df["__mus_cell"] = [
                _fmt_species_cell(aa, p)
                for aa, p in zip(df["mus_musculus_aa"], df["mus_musculus_pos"])
            ]
            columns_spec.append(("__mus_cell", "mus_musculus (AA + pos)",
                                 lambda v: str(v) if v else "—"))
        else:
            columns_spec.append(("mus_musculus_aa", "mus_musculus",
                                 lambda v: (str(v) if (isinstance(v, str) and v and v != "-") else "—")))

    # Background colours per proximity class (paler tones so dark text reads).
    CLASS_COLORS = {
        "pocket_lining":   "#fecaca",  # rose-200 — most actionable
        "pocket_proximal": "#fed7aa",  # orange-200
        "near_pocket":     "#fef08a",  # yellow-200
        "distal":          "#f5f7fa",  # very pale grey
    }
    DEFAULT_CELL_BG = "#ffffff"
    # Tier-cell band colours — mirror the dark / light amber bands the user
    # sees on the bubble grid highlight strip (strict = dark amber, mid =
    # light amber). Override the row's class background for the Tier cell.
    TIER_BANDS = {
        "strict": "#fbbf24",  # amber-400 — dark amber
        "mid":    "#fef3c7",  # amber-100 — light amber
        "broad":  "#fef9c3",  # yellow-100 — palest cream (fish > rodent gap)
    }
    # Packing-class cell highlight — packing_lining gets a red ring; surface
    # gets a green tint (likely solvent-exposed); core is neutral.
    PACKING_CELL_BG = {
        "pocket_lining": "#fca5a5",  # rose-300, signals "structurally lines a pocket"
        "surface":       "#a7f3d0",  # emerald-200
        "core":          None,
    }
    SUBSTRATE_POCKET_BG = "#fda4af"  # rose-300 when boolean is true
    # Consensus-vs-human cell tints. Cell turns pale rose when the clade
    # consensus residue DIFFERS from human (the actionable "diverged"
    # signal); pale green when it MATCHES (conserved with reference).
    CONSENSUS_DIVERGED_BG = "#fecaca"  # rose-200
    CONSENSUS_CONSERVED_BG = "#bbf7d0"  # emerald-200

    n_rows = len(df)
    max_rows_per_page = max(6, int(max_rows_per_page))
    n_pages = math.ceil(n_rows / max_rows_per_page)
    figs: List[Any] = []

    for page_idx in range(n_pages):
        start = page_idx * max_rows_per_page
        end = min(n_rows, start + max_rows_per_page)
        chunk = df.iloc[start:end]

        cell_text: List[List[str]] = []
        cell_colors: List[List[str]] = []
        for _, row in chunk.iterrows():
            text_row: List[str] = []
            color_row: List[str] = []
            cls = str(row.get("pocket_proximity_class", "")) if "pocket_proximity_class" in row else ""
            row_bg = CLASS_COLORS.get(cls, DEFAULT_CELL_BG)
            tier_val = str(row.get("tier", "")) if "tier" in row else ""
            tier_bg = TIER_BANDS.get(tier_val)
            pkt_raw = row.get("packing_classification") if "packing_classification" in row else None
            pkt_cls = str(pkt_raw).strip().lower() if isinstance(pkt_raw, str) else ""
            packing_cell_bg = PACKING_CELL_BG.get(pkt_cls)
            substrate_raw = row.get("is_substrate_pocket") if "is_substrate_pocket" in row else None
            substrate_true = False
            if isinstance(substrate_raw, str):
                substrate_true = substrate_raw.strip().lower() in {"true", "1", "yes", "y"}
            elif substrate_raw is not None and pd.notna(substrate_raw):
                substrate_true = bool(substrate_raw)
            human_aa_raw = row.get("reference_residue") if "reference_residue" in row else None
            human_aa = str(human_aa_raw).strip().upper() if isinstance(human_aa_raw, str) else ""
            teleost_aa = str(row.get("teleost_consensus_aa") or "").strip().upper()
            rodent_aa = str(row.get("rodent_consensus_aa") or "").strip().upper()
            mus_aa = str(row.get("mus_musculus_aa") or "").strip().upper()
            danio_aa = str(row.get("danio_rerio_aa") or "").strip().upper()
            def _consensus_bg(clade_aa: str) -> Optional[str]:
                if not clade_aa or clade_aa == "-" or not human_aa:
                    return None
                return CONSENSUS_CONSERVED_BG if clade_aa == human_aa else CONSENSUS_DIVERGED_BG
            for col_idx, (key, _header, fmt) in enumerate(columns_spec):
                raw = row.get(key) if key in row.index else None
                text_row.append(fmt(raw))
                cell_bg = row_bg
                if key == "tier" and tier_bg:
                    cell_bg = tier_bg
                elif key == "packing_classification" and packing_cell_bg:
                    cell_bg = packing_cell_bg
                elif key == "is_substrate_pocket" and substrate_true:
                    cell_bg = SUBSTRATE_POCKET_BG
                elif key == "__teleost_display":
                    bg = _consensus_bg(teleost_aa)
                    if bg:
                        cell_bg = bg
                elif key == "__rodent_display":
                    bg = _consensus_bg(rodent_aa)
                    if bg:
                        cell_bg = bg
                elif key in ("mus_musculus_aa", "__mus_cell"):
                    bg = _consensus_bg(mus_aa)
                    if bg:
                        cell_bg = bg
                elif key in ("danio_rerio_aa", "__danio_cell"):
                    bg = _consensus_bg(danio_aa)
                    if bg:
                        cell_bg = bg
                color_row.append(cell_bg)
            cell_text.append(text_row)
            cell_colors.append(color_row)

        col_headers = [hdr for _, hdr, _ in columns_spec]

        # Figure size: width matches the bubble plot (11"); height scales
        # with row count so dense pages stay readable.
        fig = plt.figure(figsize=(13.0, max(4.5, 0.40 * (len(chunk) + 4))))
        title_extra = ""
        if n_pages > 1:
            title_extra = f"  (page {page_idx + 1} of {n_pages})"
        fig.suptitle(
            f"{gene_label} — fish-conserved / rodent-diverged highlight sites × "
            f"distance to active-site pocket{title_extra}\n"
            f"sorted closest-to-pocket first (Cα-Cα Å to nearest hand-curated catalytic residue)",
            fontsize=11.5, y=0.97,
        )
        ax = fig.add_subplot(111)
        ax.axis("off")
        table = ax.table(
            cellText=cell_text,
            cellColours=cell_colors,
            colLabels=col_headers,
            loc="upper center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        # Auto-size every column to its widest cell (header included) so
        # long catalytic labels like "S228 nucleophile" don't overflow.
        table.auto_set_column_width(col=list(range(len(col_headers))))
        table.scale(1.0, 1.35)
        # Style the header row.
        for col_idx, _hdr in enumerate(col_headers):
            try:
                cell = table[0, col_idx]
            except KeyError:
                continue
            cell.set_facecolor("#1f2937")
            cell.set_text_props(color="white", weight="bold")
        # Bold the strict-tier rows so they pop slightly more than mid.
        try:
            tier_col_idx = next(i for i, (k, _, _) in enumerate(columns_spec) if k == "tier")
        except StopIteration:
            tier_col_idx = -1
        for r_idx, (_, row) in enumerate(chunk.iterrows(), start=1):
            tier_val = str(row.get("tier", "")) if "tier" in row else ""
            if tier_val == "strict":
                for col_idx in range(len(col_headers)):
                    try:
                        cell = table[r_idx, col_idx]
                    except KeyError:
                        continue
                    cell.set_text_props(weight="bold")

        # Footer with proximity-class legend + threshold defs.
        footer_lines = [
            "Row colour = distance-based proximity class (Cα-Cα Å to nearest catalytic residue): "
            f"pocket_lining ≤ {V11_POCKET_DISTANCE_LINING_THRESHOLD_AA:.0f} (rose) | "
            f"pocket_proximal ≤ {V11_POCKET_DISTANCE_PROXIMAL_THRESHOLD_AA:.0f} (orange) | "
            f"near_pocket ≤ {V11_POCKET_DISTANCE_NEAR_THRESHOLD_AA:.0f} (yellow) | "
            f"distal > {V11_POCKET_DISTANCE_NEAR_THRESHOLD_AA:.0f} (pale grey)."
        ]
        tier_legend = (
            "Tier cell band: strict = dark amber (matches bubble-grid dark amber, bold text), "
            "mid = light amber (matches bubble-grid light amber, regular), "
            "broad = palest cream (fish > rodent gap when the absolute thresholds miss it). "
            "When 'mouse_diverged_from_human' is True (CSV) the cell becomes a striking-target candidate "
            "(fish-conserved + rodent-diverged + mouse-altered)."
        )
        footer_lines.append(tier_legend)
        if has_packing or has_substrate:
            footer_lines.append(
                "V11.1 pocket detector: 'Packing class' cell turns rose when residue lines a pocket by Cα packing "
                "density; '≤8 Å pocket?' turns rose when residue is within 8 Å of any active-site residue (pillar-5 "
                "substrate-pocket call)."
            )
        if has_consensus:
            footer_lines.append(
                "Per-clade consensus: 'Teleost AA (cons % | id %)' and 'Rodent AA (cons % | id %)' show the most common "
                "residue in that clade (left %) and the clade's identity-to-human at this position (right %); '"
                "mus_musculus' and 'danio_rerio' show those species' own residues. Cell tint: GREEN = matches the human "
                "reference, ROSE = DIVERGED from human."
            )
        footer = "  •  ".join(footer_lines)
        fig.text(0.5, 0.025, footer, ha="center", fontsize=7.5, color="#374151", wrap=True)
        figs.append(fig)

    return figs


def v11_plot_clade_identity_bubble(csv_path: Path | str,
                                   outdir: Path | str,
                                   gene_label: str,
                                   reference_species: str = "homo_sapiens",
                                   residues_per_block: int = 60,
                                   *,
                                   clade_order: Optional[Sequence[str]] = None,
                                   png_filename: str = V11_CLADE_IDENTITY_BUBBLE_PNG,
                                   svg_filename: str = V11_CLADE_IDENTITY_BUBBLE_SVG,
                                   pdf_filename: Optional[str] = V11_CLADE_IDENTITY_BUBBLE_PDF,
                                   pdf_blocks_per_page: int = 3,
                                   title_suffix: str = "",
                                   pocket_distance_csv: Optional[Path] = None,
                                   allosteric_summary_csv: Optional[Path] = None,
                                   allosteric_coevolution_csv: Optional[Path] = None) -> Optional[Path]:
    """MATLAB-style 'clade bubble — reference identity' grid. Rows = broad
    clades (most-conserved at top); columns = reference residues wrapped into
    blocks of `residues_per_block`, each block stacked vertically like an
    alignment view. Each bubble is coloured by its CLADE (fixed palette) and
    SIZED by that clade's identity-to-human fraction (big = conserved, tiny =
    divergent). The reference amino-acid letter is printed above each column;
    letters turn red where the column is poorly covered (gaps in most clades).
    Reads the viewer-format CSV from v11_write_clade_identity_overlay_csv."""
    csv_path = Path(csv_path)
    outdir = Path(outdir)
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty or "ReferenceResidueNumber" not in df.columns:
        return None
    clades = [c[: -len("_IdentityFraction")] for c in df.columns if c.endswith("_IdentityFraction")]
    clades = [c for c in clades if c]
    if clade_order:
        order_index = {c: i for i, c in enumerate(clade_order)}
        clades.sort(key=lambda c: (order_index.get(c, len(order_index)), c))
    else:
        clades.sort(key=_v11_bubble_clade_key)
    if not clades:
        return None

    df = df.sort_values("ReferenceResidueNumber").reset_index(drop=True)
    positions = df["ReferenceResidueNumber"].astype(int).to_numpy()
    track_len = int(positions.max())
    ref_residues = (df["reference_residue"].astype(str).tolist()
                    if "reference_residue" in df.columns else ["" for _ in positions])
    pos_to_idx = {int(p): i for i, p in enumerate(positions)}

    n_blocks = math.ceil(track_len / residues_per_block)
    n_clades = len(clades)
    # Wider per-residue spacing so adjacent bubbles can't visually collide
    # even at maximum size (was 0.19" / residue → 13.7 pt spacing; now 0.24"
    # → 17.3 pt spacing, vs the new max bubble diameter ≈ 11.3 pt).
    fig_w = max(11.0, residues_per_block * 0.24)
    # Per-block height: tight but with enough room for position-label /
    # AA-letter / highlight-label strips above and the tick-label strip
    # below, plus a thin breathing margin so the next block's content can
    # never visually touch this one.
    block_h = max(1.8, 0.32 * n_clades + 0.9)
    # Vertical spacing between consecutive blocks. 0.18 = ~18% of block_h is
    # blank space between blocks — visually distinct gap without the
    # cavernous separation the previous 0.55 setting produced.
    inter_block_hspace = 0.18

    # Size scaling: identity 0..1 → marker area (pt²). Capped so the maximum
    # bubble diameter is comfortably smaller than the per-residue spacing in
    # the figure, eliminating horizontal touching of neighbours. With fig_w
    # = residues_per_block × 0.24 inches the per-residue spacing is ≈ 17.3
    # pt; max area=100 → diameter ≈ 11.3 pt → ~6 pt clearance.
    def _size(frac: float) -> float:
        return 5.0 + 95.0 * max(0.0, min(1.0, frac))

    suffix = f" {title_suffix}".rstrip() if title_suffix else ""
    title_text = (
        f"{gene_label} — clade identity fractions across reference residues{suffix} "
        f"(bubble color = clade; size = identity to {reference_species})"
    )

    def _render_blocks(block_indices: Sequence[int], page_num: Optional[int] = None,
                       total_pages: Optional[int] = None) -> Any:
        """Render the requested block indices into a single Figure."""
        n_local = len(block_indices)
        # Use gridspec_kw['hspace'] to force a fixed vertical gap between
        # stacked blocks so AA-letter / position-label strips can never
        # overlap the position-tick strip of the next block.
        fig_height = block_h * (n_local + inter_block_hspace * max(0, n_local - 1))
        fig_local, axes_local = plt.subplots(
            n_local, 1,
            figsize=(fig_w, fig_height),
            squeeze=False,
            gridspec_kw={"hspace": inter_block_hspace},
        )
        axes_local = axes_local[:, 0]
        # Precompute the "highlight" positions: residues where Teleosts are
        # more conserved with human than Rodents are. Three tiers in
        # priority order (mutually exclusive):
        #   strict — bright amber band + bold `<pos>*` label (high-conf:
        #            T ≥ 0.70 AND R ≤ 0.50)
        #   mid    — soft cream band + lighter `<pos>·` label (teleost ok,
        #            rodent lacking — T ≥ 0.55, R ≤ 0.65, gap ≥ 0.25)
        #   broad  — palest yellow band + `<pos>` label (the broader
        #            "fish > rodent" landscape: T ≥ 0.70, gap ≥ 0.15, even
        #            when rodent identity itself is moderately high)
        highlight_tiers = _v11_highlight_teleost_rodent_diverged_tiered(df)
        strict_positions = set(highlight_tiers.get("strict") or [])
        mid_positions = set(highlight_tiers.get("mid") or [])
        broad_positions = set(highlight_tiers.get("broad") or [])
        for local_idx, b in enumerate(block_indices):
            ax = axes_local[local_idx]
            start = b * residues_per_block + 1
            end = min(track_len, (b + 1) * residues_per_block)
            # Highlight bands UNDER the bubbles. Render broad → mid → strict
            # so the brighter strict tier draws on top where they overlap
            # (they shouldn't — the tiers are mutually exclusive — but be
            # defensive). All bands sit at zorder=1, under the bubbles.
            for pos in range(start, end + 1):
                if pos in broad_positions:
                    ax.axvspan(pos - 0.45, pos + 0.45, color="#fef9c3",
                               alpha=0.40, zorder=1, linewidth=0)
                if pos in mid_positions:
                    ax.axvspan(pos - 0.45, pos + 0.45, color="#fef3c7",
                               alpha=0.50, zorder=1, linewidth=0)
                if pos in strict_positions:
                    ax.axvspan(pos - 0.45, pos + 0.45, color="#fde68a",
                               alpha=0.55, zorder=1, linewidth=0)
            xs, ys, cols, szs, vals = [], [], [], [], []
            for row_idx, clade in enumerate(clades):
                y = n_clades - 1 - row_idx
                color = _v11_clade_color(clade)
                frac_col = f"{clade}_IdentityFraction"
                # Connecting line per clade row — bubbles render as beads on
                # this thread, making each clade's per-residue track easy to
                # follow across the block. Drawn under the bubbles (zorder=2)
                # in the same clade color, faded so the bubbles stay the
                # primary visual.
                row_line_xs: List[int] = []
                for pos in range(start, end + 1):
                    i = pos_to_idx.get(pos)
                    if i is None:
                        continue
                    f = df[frac_col].iloc[i]
                    if not (isinstance(f, (int, float)) and np.isfinite(f)):
                        continue
                    xs.append(pos)
                    ys.append(y)
                    cols.append(color)
                    szs.append(_size(float(f)))
                    vals.append(float(f))
                    row_line_xs.append(pos)
                if len(row_line_xs) >= 2:
                    ax.plot(row_line_xs, [y] * len(row_line_xs),
                            color=color, alpha=0.40, linewidth=0.8,
                            solid_capstyle="round", zorder=2)
            if xs:
                # Visible dark outline so bubbles read as distinct dots.
                ax.scatter(xs, ys, c=cols, s=szs,
                           edgecolors="#1f2937", linewidths=0.6,
                           alpha=0.95, zorder=3)
                # Inline fraction text inside the largest bubbles only — the
                # threshold is set so the text fits comfortably inside the
                # bubble disc and doesn't spill into neighbours.
                for xv, yv, sv, vv in zip(xs, ys, szs, vals):
                    if sv >= 75 and residues_per_block <= 80:
                        ax.text(xv, yv, f"{vv:.2f}", ha="center", va="center",
                                fontsize=3.8, color="#0b0b0b", zorder=4)
            # Three vertical strips above the bubble rows:
            #   aa_letter_y     — the reference AA at every column (always shown)
            #   highlight_y     — the residue NUMBER + * for highlighted
            #                      positions only (so the user can read which
            #                      residue the yellow band marks without
            #                      hunting along the tick row)
            # And one strip below:
            #   position_label_y — every-10 tick labels
            aa_letter_y = n_clades + 0.15
            highlight_y = n_clades + 0.90
            position_label_y = -0.85
            for pos in range(start, end + 1):
                i = pos_to_idx.get(pos)
                aa = ref_residues[i] if (i is not None and i < len(ref_residues)) else ""
                if pos in strict_positions:
                    ax.text(pos, highlight_y, f"{pos}*",
                            ha="center", va="bottom",
                            fontsize=7, color="#b45309",
                            fontweight="bold", zorder=5)
                elif pos in mid_positions:
                    # Lighter, smaller marker for the mid tier so the strict
                    # tier still pops as the headline.
                    ax.text(pos, highlight_y, f"{pos}·",
                            ha="center", va="bottom",
                            fontsize=6, color="#d97706",
                            zorder=5)
                elif pos in broad_positions:
                    # Smallest, lightest marker for broad — the "fish >
                    # rodent gap" tier; readable but visually subordinate
                    # to strict / mid so the headlines still dominate.
                    ax.text(pos, highlight_y, f"{pos}",
                            ha="center", va="bottom",
                            fontsize=5.5, color="#a16207",
                            zorder=5)
                ax.text(pos, aa_letter_y, aa, ha="center", va="bottom",
                        fontsize=6.5, color="#333")
                if pos % 10 == 0 or pos == start:
                    ax.text(pos, position_label_y, str(pos), ha="center", va="top",
                            fontsize=6, color="#777")
            ax.set_xlim(start - 0.6, start + residues_per_block - 0.4)
            # Generous top margin (room for AA letter + highlight residue
            # number + a buffer) and bottom margin (room for position ticks +
            # a buffer); combined with inter_block_hspace this eliminates any
            # vertical overlap between consecutive blocks on the same page OR
            # between consecutive pages of the multi-page PDF.
            ax.set_ylim(-1.6, n_clades + 1.85)
            ax.set_yticks(range(n_clades))
            ax.set_yticklabels(
                [clades[n_clades - 1 - i] for i in range(n_clades)], fontsize=7.5
            )
            for tick, clade in zip(
                ax.get_yticklabels(),
                [clades[n_clades - 1 - i] for i in range(n_clades)],
            ):
                tick.set_color(_v11_clade_color(clade))
            ax.set_xticks([])
            for spine in ("top", "right", "bottom"):
                ax.spines[spine].set_visible(False)
            ax.set_ylabel("Clade", fontsize=8)
        # Title above the FIRST subplot (suptitle would be safer than axes
        # title but a regular set_title with explicit pad keeps it close to
        # block 1 instead of floating between the top and block 1).
        axes_local[0].set_title(title_text, fontsize=12, pad=14)
        footer_bits = [
            "Bubble size ∝ identity to human; color = clade; dark outline; "
            "per-clade connecting line in the same color. "
            "Reference residue letters shown above each column. "
            "Bright amber band + <pos>* = strict (Teleosts≥0.70 AND Rodents≤0.50). "
            "Soft band + <pos>· = mid (Teleosts≥0.55 AND Rodents≤0.65 AND gap≥0.25). "
            "Palest yellow band + <pos> = broad (Teleosts≥0.70 AND gap≥0.15) — "
            "catches \"fish > rodent\" gaps the strict / mid cuts miss."
        ]
        if page_num is not None and total_pages is not None:
            footer_bits.append(f"Page {page_num} of {total_pages}")
        # Footer at the FIGURE level, below all blocks, so it never sits
        # inside the last block's axis (which previously caused overlap with
        # position-tick labels on stacked PNG views).
        fig_local.text(
            0.06, 0.01, "  •  ".join(footer_bits),
            ha="left", va="bottom", fontsize=7, color="#777",
        )
        # tight_layout with smaller padding now that block_h + hspace are
        # tuned to leave just enough breathing room.
        fig_local.tight_layout(pad=0.9, h_pad=0.6, rect=(0, 0.015, 1, 1))
        return fig_local

    # Tall single-page figure (PNG + SVG) — historical V11 view.
    fig = _render_blocks(list(range(n_blocks)))
    png = outdir / png_filename
    svg = outdir / svg_filename
    fig.savefig(png, dpi=170, bbox_inches="tight", pad_inches=0.35)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)

    # Multi-page PDF (PdfPages) — matches the user's MATLAB chart format, with
    # `pdf_blocks_per_page` blocks of `residues_per_block` residues per page.
    if pdf_filename:
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            blocks_per_page = max(1, int(pdf_blocks_per_page))
            page_count = math.ceil(n_blocks / blocks_per_page)
            pdf_path = outdir / pdf_filename
            with PdfPages(pdf_path) as pdf:
                for page_idx in range(page_count):
                    start_b = page_idx * blocks_per_page
                    end_b = min(n_blocks, start_b + blocks_per_page)
                    page_fig = _render_blocks(
                        list(range(start_b, end_b)),
                        page_num=page_idx + 1,
                        total_pages=page_count,
                    )
                    pdf.savefig(page_fig, bbox_inches="tight", pad_inches=0.35)
                    plt.close(page_fig)
                # Append pocket-distance table page(s) when the caller
                # supplied the CSV path. Used by the 9-group bubble grid so
                # the user gets the bubble blocks AND the per-position
                # pocket-distance table in one printable artefact.
                if pocket_distance_csv:
                    try:
                        pd_figs = _v11_render_pocket_distance_table_figures(
                            Path(pocket_distance_csv), gene_label,
                        )
                        for pd_fig in pd_figs:
                            pdf.savefig(pd_fig, bbox_inches="tight", pad_inches=0.35)
                            plt.close(pd_fig)
                    except Exception:  # noqa: BLE001
                        # Table is a bonus; if it fails the bubble pages
                        # above are still the source of truth.
                        pass
                # Pilot allosteric pages — appended after the pocket-distance
                # table when the caller supplies the pilot CSV. Includes the
                # per-highlight allosteric profile + the top-MI coevolution
                # partners (when available).
                if allosteric_summary_csv:
                    try:
                        allo_figs = _v11_render_allosteric_summary_figures(
                            Path(allosteric_summary_csv), gene_label,
                            coevolution_csv=Path(allosteric_coevolution_csv)
                                if allosteric_coevolution_csv else None,
                        )
                        for allo_fig in allo_figs:
                            pdf.savefig(allo_fig, bbox_inches="tight", pad_inches=0.35)
                            plt.close(allo_fig)
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            # PDF is a bonus; if it fails (e.g. PdfPages backend missing) the
            # PNG + SVG above are still the source of truth.
            pass
    return svg


def v11_write_clade_identity_overlay_csv(outdir: Path | str,
                                         alignment: Any,
                                         reference_species: str,
                                         taxonomy_lookup: Optional[Dict[str, Any]] = None,
                                         *,
                                         clade_resolver: Optional[Any] = None,
                                         clade_sort_key: Optional[Any] = None,
                                         output_filename: str = V11_STRUCTURE_OVERLAY_CSV) -> Path:
    """Write the viewer-format per-clade identity CSV. For every reference
    (ungapped) position and every broad clade, reports the fraction of that
    clade's members whose residue matches the human reference (gap-aware),
    the covered-species count, the identical-species count, and the clade's
    total member count. Columns mirror the MATLAB DHRS7 overlay CSV so the
    3Dmol viewer can auto-detect the groups.

    `clade_resolver` overrides the species→clade mapping (defaults to
    v11_resolve_broad_clade via _v11_species_clade_map). Pass a callable that
    takes the record-id string and returns the clade label to use a custom
    grouping (e.g. MATLAB-style Primates/Rodents/OtherMammals).
    """
    outdir = Path(outdir)
    aln_to_ref, ref_record = _v11_ref_position_records(alignment, reference_species)
    if clade_resolver is None:
        species_clade = _v11_species_clade_map(alignment, taxonomy_lookup)
    else:
        species_clade = {str(rec.id): clade_resolver(rec) for rec in alignment}
    ref_seq = str(ref_record.seq).upper()

    # Group records by clade and record total members per clade.
    clade_records: Dict[str, List[Any]] = {}
    for record in alignment:
        clade = species_clade.get(str(record.id), "Unassigned")
        clade_records.setdefault(clade, []).append(record)
    sort_key = clade_sort_key if clade_sort_key is not None else _v11_clade_sort_key
    clades_sorted = sorted(clade_records.keys(), key=sort_key)
    clade_totals = {c: len(recs) for c, recs in clade_records.items()}

    rows: List[Dict[str, Any]] = []
    aln_len = alignment.get_alignment_length()
    for col_idx in range(aln_len):
        ref_pos = aln_to_ref[col_idx]
        if ref_pos is None:
            continue
        ref_aa = ref_seq[col_idx] if col_idx < len(ref_seq) else "-"
        row: Dict[str, Any] = {
            "ReferenceResidueNumber": int(ref_pos),
            "AlignmentColumn": int(col_idx + 1),
            "reference_residue": ref_aa,
        }
        summary_bits: List[str] = []
        for clade in clades_sorted:
            covered = 0
            identical = 0
            for rec in clade_records[clade]:
                seq = str(rec.seq)
                if col_idx >= len(seq):
                    continue
                aa = seq[col_idx].upper()
                if aa in GAP_CHARS:
                    continue
                covered += 1
                if ref_aa not in GAP_CHARS and aa == ref_aa:
                    identical += 1
            frac = (identical / covered) if covered else float("nan")
            row[f"{clade}_IdentityFraction"] = frac
            row[f"{clade}_CoveredSpecies"] = covered
            row[f"{clade}_IdentitySpeciesEquivalent"] = identical
            row[f"{clade}_TotalSpeciesInGroup"] = clade_totals[clade]
            if covered:
                summary_bits.append(f"{clade}={frac:.3f}(n={covered})")
        row["CladeIdentityFractions"] = "; ".join(summary_bits)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = outdir / output_filename
    df.to_csv(out_path, index=False)
    return out_path


def v11_write_structure_overlay(outdir: Path | str,
                                alignment: Any,
                                reference_species: str,
                                gene_label: str,
                                taxonomy_lookup: Optional[Dict[str, Any]] = None,
                                site_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Orchestrate the interactive 3D structure overlay: write the per-clade
    identity CSV, then call the standalone generator (build_structure_overlay.py,
    imported as a module) to produce a self-contained
    <V11_STRUCTURE_OVERLAY_HTML> that colours those clade identities onto the
    human reference AlphaFold model. Requires the human reference PDB +
    metadata to be present in `outdir` (written by the AlphaFold bundle step)."""
    outdir = Path(outdir)
    result: Dict[str, Any] = {"available": False}

    pdb_path = outdir / ALPHAFOLD_MODEL_FILENAME
    meta_path = outdir / ALPHAFOLD_METADATA_FILENAME
    if not pdb_path.exists():
        # Some runs store the model under a metadata-specified filename.
        meta = read_json_artifact(meta_path) if meta_path.exists() else {}
        alt = meta.get("model_filename") if isinstance(meta, dict) else None
        if alt and (outdir / alt).exists():
            pdb_path = outdir / alt
    if not pdb_path.exists():
        result["reason"] = "human reference AlphaFold PDB not found in output dir."
        return result

    csv_path = v11_write_clade_identity_overlay_csv(outdir, alignment, reference_species, taxonomy_lookup)
    result["csv_path"] = str(csv_path)

    # V11 broad-clade bubble grid (13 vertebrate clades, Mammalia as one bucket).
    try:
        bubble = v11_plot_clade_identity_bubble(csv_path, outdir, gene_label, reference_species)
        result["bubble_svg"] = str(bubble) if bubble else None
    except Exception as exc:  # noqa: BLE001
        result["bubble_svg"] = None
        result["bubble_error"] = str(exc)

    # Import the generator module from the script directory. Hoisted ABOVE
    # the 9-group bubble call so we can resolve the gene's active-site
    # residues (KNOWN_SITE_CONFIGS) and pre-generate the pocket-distance
    # CSV BEFORE the 9-group bubble PDF is written — the PDF appends the
    # pocket-distance table after its block pages.
    try:
        import importlib.util
        gen_path = Path(__file__).resolve().with_name("build_structure_overlay.py")
        if not gen_path.exists():
            result["reason"] = f"generator not found: {gen_path}"
            return result
        spec = importlib.util.spec_from_file_location("v11_structure_overlay_gen", gen_path)
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
    except Exception as exc:  # noqa: BLE001
        result["reason"] = f"could not import generator: {exc}"
        return result

    # Resolve the active-site pairs (position + label) once for both the
    # pocket-distance CSV and the later overlay HTMLs.
    effective_site_config = site_config
    if effective_site_config is None and gene_label:
        known = getattr(gen, "KNOWN_SITE_CONFIGS", None)
        if isinstance(known, dict):
            effective_site_config = known.get(str(gene_label).strip().upper())
    active_pairs_for_distance: List[Tuple[int, str]] = []
    if effective_site_config:
        ar = effective_site_config.get("activeResidues") or []
        cl = effective_site_config.get("catalyticLabels") or []
        for i, pos in enumerate(ar):
            label = cl[i] if i < len(cl) else None
            try:
                active_pairs_for_distance.append((int(pos), label or ""))
            except (TypeError, ValueError):
                continue

    # Subdivided 9-group bubble grid (Primates/Rodents/OtherMammals/Teleosts/
    # OtherFish/Birds/Reptiles/Amphibians/OtherVertebrates). This is the
    # Python reimplementation of the MATLAB clade-analysis grouping the user
    # shared — runs alongside the broad-clade view for every gene so the
    # mammal subdivision is visible. Driven from a SEPARATE per-clade CSV so
    # the original 3Dmol structure overlay above keeps its broader-clade
    # granularity.
    grouped_csv: Optional[Path] = None
    try:
        species_clade_broad = _v11_species_clade_map(alignment, taxonomy_lookup)

        def _grouped_resolver(record):
            species, _sym = parse_header_species_symbol(str(record.id))
            broad = species_clade_broad.get(str(record.id), "Unassigned")
            return v11_resolve_grouped_clade(species, broad)

        def _grouped_sort_key(clade: str):
            try:
                return (0, _V11_GROUPED_BUBBLE_CLADE_ORDER.index(clade))
            except ValueError:
                return (1, clade)

        grouped_csv = v11_write_clade_identity_overlay_csv(
            outdir, alignment, reference_species, taxonomy_lookup,
            clade_resolver=_grouped_resolver,
            clade_sort_key=_grouped_sort_key,
            output_filename=V11_GROUPED_CLADE_IDENTITY_CSV,
        )
        result["grouped_csv_path"] = str(grouped_csv)
        # Pre-generate the teleost-rodent highlight-sites CSV and the
        # pocket-distance augmentation BEFORE the 9-group bubble so its
        # multi-page PDF can append the pocket-distance table as final
        # page(s). Both helpers are idempotent — running them again later
        # in this function just overwrites with the same content.
        pocket_dist_csv: Optional[Path] = None
        try:
            if grouped_csv is not None and grouped_csv.exists():
                v11_write_teleost_conserved_rodent_diverged_sites(grouped_csv, outdir)
                pocket_dist_csv = v11_write_highlight_sites_pocket_distance(
                    outdir, gene_label or "GENE",
                    active_site_residues=active_pairs_for_distance or None,
                    alignment=alignment,
                    reference_species=reference_species,
                    taxonomy_lookup=taxonomy_lookup,
                )
        except Exception as _pd_pre_exc:  # noqa: BLE001
            result["pocket_distance_pre_error"] = str(_pd_pre_exc)
            pocket_dist_csv = None
        grouped_bubble = v11_plot_clade_identity_bubble(
            grouped_csv, outdir, gene_label, reference_species,
            clade_order=_V11_GROUPED_BUBBLE_CLADE_ORDER,
            png_filename=V11_GROUPED_CLADE_IDENTITY_BUBBLE_PNG,
            svg_filename=V11_GROUPED_CLADE_IDENTITY_BUBBLE_SVG,
            pdf_filename=V11_GROUPED_CLADE_IDENTITY_BUBBLE_PDF,
            title_suffix="(subdivided 9-group: Primates / Rodents / OtherMammals split)",
            pocket_distance_csv=pocket_dist_csv,
        )
        result["grouped_bubble_svg"] = str(grouped_bubble) if grouped_bubble else None
    except Exception as exc:  # noqa: BLE001
        result["grouped_bubble_svg"] = None
        result["grouped_bubble_error"] = str(exc)

    # Resolve gene / accession / model from the metadata JSON when available.
    meta = read_json_artifact(meta_path) if meta_path.exists() else {}
    uniprot = clean_json_value(meta.get("uniprot_accession")) if isinstance(meta, dict) else None
    af_model = None
    af_version = None
    if isinstance(meta, dict):
        pred = meta.get("alphafold_prediction") or {}
        af_model = clean_json_value(pred.get("modelEntityId")) or (f"AF-{uniprot}-F1" if uniprot else None)
        af_version = pred.get("latestVersion")

    html_path = outdir / V11_STRUCTURE_OVERLAY_HTML
    js_src = outdir / ALPHAFOLD_VIEWER_JS_FILENAME  # 3Dmol-min.js already staged by the report

    # Compute the "fish-conserved / rodent-diverged" highlight residues from
    # the 9-group CSV (if it exists) so they're available as an optional
    # labeled overlay in every viewer variant. Same threshold + math as the
    # bubble-grid highlight band; just packaged for the 3D viewer.
    highlight_sites_payload: List[Dict[str, Any]] = []
    try:
        if grouped_csv is not None and grouped_csv.exists():
            _grouped_df_for_highlight = pd.read_csv(grouped_csv)
            _tiers_for_highlight = _v11_highlight_teleost_rodent_diverged_tiered(
                _grouped_df_for_highlight
            )
            for tier_name in ("strict", "mid", "broad"):
                for pos in _tiers_for_highlight.get(tier_name, []) or []:
                    row_match = _grouped_df_for_highlight[
                        _grouped_df_for_highlight["ReferenceResidueNumber"] == pos
                    ]
                    ref_aa = ""
                    tele = None
                    rod = None
                    if not row_match.empty:
                        ref_aa = str(row_match.iloc[0].get("reference_residue", ""))
                        try:
                            tele = float(row_match.iloc[0]["Teleosts_IdentityFraction"])
                            rod = float(row_match.iloc[0]["Rodents_IdentityFraction"])
                        except (KeyError, ValueError, TypeError):
                            tele, rod = None, None
                    highlight_sites_payload.append({
                        "position": int(pos),
                        "reference_residue": ref_aa,
                        "kind": "fish_conserved_rodent_diverged",
                        "tier": tier_name,  # "strict", "mid", or "broad"
                        "teleost_identity": tele,
                        "rodent_identity": rod,
                    })
    except Exception:  # noqa: BLE001
        highlight_sites_payload = []
    try:
        gen.build_overlay(
            gene=gene_label,
            uniprot_accession=uniprot or "",
            af_model=af_model or "",
            af_version=af_version,
            csv_path=csv_path,
            pdb_path=pdb_path,
            output_html=html_path,
            js_path=js_src if js_src.exists() else None,
            site_config=site_config,
            highlight_sites=highlight_sites_payload,
        )
    except Exception as exc:  # noqa: BLE001
        result["reason"] = f"overlay generation failed: {exc}"
        return result

    result["available"] = html_path.exists()
    result["html_path"] = str(html_path) if html_path.exists() else None

    # Second interactive AlphaFold overlay driven by the subdivided 9-group
    # CSV (Primates / Rodents / OtherMammals split). Same viewer code path
    # auto-detects clade groups from the CSV columns, so the user can pick
    # any of the 9 groups in the viewer to shade the structure.
    if grouped_csv is not None and grouped_csv.exists():
        grouped_html_path = outdir / V11_GROUPED_STRUCTURE_OVERLAY_HTML
        try:
            gen.build_overlay(
                gene=gene_label,
                uniprot_accession=uniprot or "",
                af_model=af_model or "",
                af_version=af_version,
                csv_path=grouped_csv,
                pdb_path=pdb_path,
                output_html=grouped_html_path,
                js_path=js_src if js_src.exists() else None,
                site_config=site_config,
            )
            result["grouped_html_path"] = str(grouped_html_path) if grouped_html_path.exists() else None
        except Exception as exc:  # noqa: BLE001
            result["grouped_html_path"] = None
            result["grouped_html_error"] = f"grouped overlay generation failed: {exc}"

    # "Fish-conserved, rodent-lost" sites CSV — positions where Teleosts is
    # highly conserved with human but Rodents has substantially diverged.
    # Driven by the 9-group CSV (has clean Primates/Rodents/OtherMammals
    # columns). Emitted alongside the highlight band in the bubble grids.
    try:
        if grouped_csv is not None and grouped_csv.exists():
            sites_path = v11_write_teleost_conserved_rodent_diverged_sites(grouped_csv, outdir)
            result["teleost_rodent_sites_csv_path"] = str(sites_path) if sites_path else None
    except Exception as exc:  # noqa: BLE001
        result["teleost_rodent_sites_csv_path"] = None
        result["teleost_rodent_sites_error"] = str(exc)

    # Pocket-distance augmentation of the highlight-sites CSV. Active-site
    # residues come from the caller's site_config OR, as a fallback, from
    # build_structure_overlay.KNOWN_SITE_CONFIGS (the hand-curated catalytic
    # residue table the 3D viewer already uses). When neither is available
    # the helper still emits the file with "n/a" distance class so
    # downstream tooling sees a consistent schema for every gene.
    try:
        effective_site_config = site_config
        if effective_site_config is None and gene_label:
            known = getattr(gen, "KNOWN_SITE_CONFIGS", None)
            if isinstance(known, dict):
                effective_site_config = known.get(str(gene_label).strip().upper())
        active_pairs_for_distance: List[Tuple[int, str]] = []
        if effective_site_config:
            ar = effective_site_config.get("activeResidues") or []
            cl = effective_site_config.get("catalyticLabels") or []
            for i, pos in enumerate(ar):
                label = cl[i] if i < len(cl) else None
                active_pairs_for_distance.append((int(pos), label or ""))
        pocket_dist_path = v11_write_highlight_sites_pocket_distance(
            outdir, gene_label or "GENE",
            active_site_residues=active_pairs_for_distance or None,
            alignment=alignment,
            reference_species=reference_species,
            taxonomy_lookup=taxonomy_lookup,
        )
        result["highlight_pocket_distance_csv_path"] = (
            str(pocket_dist_path) if pocket_dist_path else None
        )
    except Exception as exc:  # noqa: BLE001
        result["highlight_pocket_distance_csv_path"] = None
        result["highlight_pocket_distance_error"] = str(exc)

    # Compact 9-group "mod" CSV + bubble grid + structure overlay. Same buckets
    # as the 9-group view but with the EXACT column names from the user's
    # reference (`Other` instead of `OtherFish`, `x` instead of
    # `OtherVertebrates`). Filename also matches the user's reference
    # (`clade_identity_by_reference_position_mod.csv`) so downstream tooling
    # written against the MATLAB output can consume V11's CSV unchanged.
    mod_csv: Optional[Path] = None
    try:
        def _mod_resolver(record):
            species, _sym = parse_header_species_symbol(str(record.id))
            broad = species_clade_broad.get(str(record.id), "Unassigned")
            return v11_resolve_mod_clade(species, broad)

        def _mod_sort_key(clade: str):
            try:
                return (0, _V11_MOD_BUBBLE_CLADE_ORDER.index(clade))
            except ValueError:
                return (1, clade)

        mod_csv = v11_write_clade_identity_overlay_csv(
            outdir, alignment, reference_species, taxonomy_lookup,
            clade_resolver=_mod_resolver,
            clade_sort_key=_mod_sort_key,
            output_filename=V11_MOD_CLADE_IDENTITY_CSV,
        )
        result["mod_csv_path"] = str(mod_csv)
        mod_bubble = v11_plot_clade_identity_bubble(
            mod_csv, outdir, gene_label, reference_species,
            clade_order=_V11_MOD_BUBBLE_CLADE_ORDER,
            png_filename=V11_MOD_CLADE_IDENTITY_BUBBLE_PNG,
            svg_filename=V11_MOD_CLADE_IDENTITY_BUBBLE_SVG,
            pdf_filename=V11_MOD_CLADE_IDENTITY_BUBBLE_PDF,
            title_suffix="(Compact 9-group naming: Other / x)",
        )
        result["mod_bubble_svg"] = str(mod_bubble) if mod_bubble else None

        mod_html_path = outdir / V11_MOD_STRUCTURE_OVERLAY_HTML
        try:
            gen.build_overlay(
                gene=gene_label,
                uniprot_accession=uniprot or "",
                af_model=af_model or "",
                af_version=af_version,
                csv_path=mod_csv,
                pdb_path=pdb_path,
                output_html=mod_html_path,
                js_path=js_src if js_src.exists() else None,
                site_config=site_config,
            )
            result["mod_html_path"] = str(mod_html_path) if mod_html_path.exists() else None
        except Exception as exc:  # noqa: BLE001
            result["mod_html_path"] = None
            result["mod_html_error"] = f"mod overlay generation failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        result["mod_bubble_svg"] = None
        result["mod_bubble_error"] = str(exc)

    # Third overlay: COMBINED view. Merges broad-clade + 9-group + mod columns
    # into one CSV so a single viewer's clade dropdown exposes every bucket
    # the user might want to shade by (Mammalia OR Primates OR Other-style,
    # without changing files).
    try:
        if grouped_csv is not None and grouped_csv.exists() and csv_path.exists():
            broad_df = pd.read_csv(csv_path)
            grouped_df = pd.read_csv(grouped_csv)
            # Drop the per-row text summary so columns are unambiguous; the
            # viewer just needs the _IdentityFraction / _CoveredSpecies /
            # _IdentitySpeciesEquivalent / _TotalSpeciesInGroup quartets.
            for col in ("CladeIdentityFractions",):
                if col in broad_df.columns:
                    broad_df = broad_df.drop(columns=[col])
                if col in grouped_df.columns:
                    grouped_df = grouped_df.drop(columns=[col])
            # 9-group keeps only the group-specific quartet columns; merge by
            # ReferenceResidueNumber (the row keys are identical).
            grouped_only_cols = [
                c for c in grouped_df.columns
                if c not in {"ReferenceResidueNumber", "AlignmentColumn", "reference_residue"}
            ]
            combined_df = broad_df.merge(
                grouped_df[["ReferenceResidueNumber"] + grouped_only_cols],
                on="ReferenceResidueNumber",
                how="left",
            )
            # Fold in the compact 9-group "mod" columns (Other_*, x_*) too so
            # the combined viewer dropdown lists Other and x as options.
            if mod_csv is not None and mod_csv.exists():
                mod_df_for_combine = pd.read_csv(mod_csv)
                if "CladeIdentityFractions" in mod_df_for_combine.columns:
                    mod_df_for_combine = mod_df_for_combine.drop(columns=["CladeIdentityFractions"])
                mod_only_cols = [
                    c for c in mod_df_for_combine.columns
                    if c not in {"ReferenceResidueNumber", "AlignmentColumn", "reference_residue"}
                ]
                # Only keep compact 9-group buckets whose name is NOT already in
                # the broad/9group columns (e.g. Primates appears in 9-group
                # already; Other and x are new).
                existing = set(combined_df.columns)
                mod_new_cols = [c for c in mod_only_cols if c not in existing]
                if mod_new_cols:
                    combined_df = combined_df.merge(
                        mod_df_for_combine[["ReferenceResidueNumber"] + mod_new_cols],
                        on="ReferenceResidueNumber",
                        how="left",
                    )
            combined_csv = outdir / V11_COMBINED_CLADE_IDENTITY_CSV
            combined_df.to_csv(combined_csv, index=False)
            result["combined_csv_path"] = str(combined_csv)

            combined_html_path = outdir / V11_COMBINED_STRUCTURE_OVERLAY_HTML
            try:
                gen.build_overlay(
                    gene=gene_label,
                    uniprot_accession=uniprot or "",
                    af_model=af_model or "",
                    af_version=af_version,
                    csv_path=combined_csv,
                    pdb_path=pdb_path,
                    output_html=combined_html_path,
                    js_path=js_src if js_src.exists() else None,
                    site_config=site_config,
                )
                result["combined_html_path"] = (
                    str(combined_html_path) if combined_html_path.exists() else None
                )
            except Exception as exc:  # noqa: BLE001
                result["combined_html_path"] = None
                result["combined_html_error"] = f"combined overlay generation failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        result["combined_csv_path"] = None
        result["combined_csv_error"] = f"combined CSV merge failed: {exc}"
    return result
