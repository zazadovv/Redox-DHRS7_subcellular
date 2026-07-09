#!/usr/bin/env python3
"""
gene_phylo_conservation_pipeline.py  -- gene-agnostic ortholog phylogeny & conservation pipeline (representative property tracks, optional tree, focused secondary-structure view)

End-to-end pipeline for a single gene:
1. Retrieve orthologous genes from Ensembl
2. Retrieve one representative protein sequence per species
3. Align sequences with MAFFT, MUSCLE, or an internal Python fallback aligner
4. Optionally build a protein phylogeny with IQ-TREE
5. Retrieve protein domain annotations from UniProt / InterPro xrefs
6. Compute per-position exact and biochemical-property conservation
7. Detect conserved windows
8. Export per-residue conservation scan outputs
9. Export CSV/TSV/SVG/PNG outputs
10. Append the run to a SQLite archive and write an offline interactive HTML report

Environment:
    conda activate phylo
    python gene_phylo_conservation_pipeline.py DHRS7 --source_species homo_sapiens --outdir DHRS7_output

External tools:
    mafft    (optional)
    muscle   (optional)
    iqtree3  (iqtree / iqtree2 also accepted; optional if tree building is skipped)

Examples:
    python gene_phylo_conservation_pipeline.py DHRS7 --source_species homo_sapiens --outdir DHRS7_output
    python gene_phylo_conservation_pipeline.py TP53 --source_species homo_sapiens \
        --target_species mus_musculus rattus_norvegicus danio_rerio bos_taurus gallus_gallus \
        --reference_species homo_sapiens \
        --mafft_exe "C:\\path\\to\\mafft.bat" \
        --iqtree_exe "E:\\path\\to\\iqtree3.exe"

Notes:
- This script chooses one representative translated protein per species by preferring the canonical transcript translation.
- Domain annotations are fetched from UniProt when possible and include InterPro cross-references when available.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import requests
from Bio import AlignIO, Phylo, SeqIO
from Bio.Align import substitution_matrices
from Bio.Align import MultipleSeqAlignment, PairwiseAligner
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from gene_phylo_conservation_archive import (
    DEFAULT_SQLITE_ARCHIVE_PATH,
    INTERACTIVE_REPORT_FILENAME,
    classify_alignment_broad_clade,
    classify_alignment_phylum,
    default_clade_age_mya,
    export_output_archive,
    write_comparative_alphafold_secondary_structure_bundle,
    # V11: representative comparison helpers
    v11_write_representative_comparison_outputs,
    v11_plot_paper_quality_tree_svg,
    V11_DEFAULT_PROPERTY_WINDOW,
    V11_MANDATORY_FOCUS_SPECIES,
    V11_DEFAULT_REPRESENTATIVE_CSV,
    # V11 motif + lineage-stabilization extension
    v11_write_motif_analysis_outputs,
    V11_DEFAULT_ANCESTRAL_CLADES,
    V11_DEFAULT_DERIVED_CLADES,
    # V11: gene-agnostic labelling
    v11_set_active_gene,
    # V11: default per-clade consolidated summary (SS + net charge + domains)
    v11_write_clade_consolidated_outputs,
    # V11: interactive 3D structure overlay (3Dmol.js)
    v11_write_structure_overlay,
)


ENSEMBL = "https://rest.ensembl.org"
UNIPROT = "https://rest.uniprot.org"
ALPHAFOLD = "https://alphafold.ebi.ac.uk"
REQUEST_TIMEOUT = 60
REQUEST_RETRY_ATTEMPTS = 4
REQUEST_RETRY_BASE_DELAY = 1.0
REQUEST_RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504}
SEQUENCE_FETCH_MAX_WORKERS = 4
PYTHON_FALLBACK_MAX_WORKERS = 8
PAIRWISE_REPORT_MAX_WORKERS = 4
MUSCLE_SUPER5_SEQUENCE_THRESHOLD = 200

AA_GROUPS: Dict[str, set[str]] = {
    "aliphatic": set("AVILM"),
    "aromatic": set("FWYH"),
    "hydrophobic": set("AVILMFWYC"),
    "polar": set("STNQ"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "small": set("GAS"),
    "special": set("GPC"),
    "charged": set("KRHDE"),
}

AA_GROUP_SCHEMES: Dict[str, Dict[str, set[str]]] = {
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

VERTEBRATE_KEYWORDS = (
    "vertebrata", "mammalia", "aves", "reptilia", "amphibia",
    "actinopterygii", "sarcopterygii", "teleostei", "chondrichthyes",
    "tetrapoda", "amniota", "eutheria", "theria", "primates", "rodentia",
    "carnivora", "cetartiodactyla", "perissodactyla", "lagomorpha", "glires",
    "sauropsida", "anura", "cypriniformes", "clupeocephala", "otophysi",
)

INVERTEBRATE_KEYWORDS = (
    "arthropoda", "insecta", "nematoda", "mollusca", "annelida", "cnidaria",
    "echinodermata", "urochordata", "cephalochordata", "protostomia", "ecdysozoa",
    "fungi", "ascomycota", "dikarya", "saccharomycetes", "saccharomycetaceae", "saccharomyces",
)

GAP_CHARS = {"-", "."}
TREE_METHOD_CHOICES = ("auto", "iqtree", "python_nj")
TREE_DISTANCE_MODELS = ("blosum62", "identity")
SELECTION_MODE_CHOICES = ("all_filtered", "curated_subset")
METADATA_ENRICHMENT_CHOICES = ("auto", "local", "required")
TREE_NOMENCLATURE_SOURCE_CHOICES = ("auto", "local", "ensembl")
ALPHAFOLD_MODE_CHOICES = ("auto", "off", "required")
TREE_NOMENCLATURE_FILENAME = "tree_nomenclature.json"
PROTEIN_METADATA_FILENAME = "protein_metadata.tsv"
PROTEIN_FEATURES_FILENAME = "protein_features.tsv"
PROTEIN_XREFS_FILENAME = "protein_xrefs.tsv"
ALPHAFOLD_METADATA_FILENAME = "human_reference_alphafold_metadata.json"
ALPHAFOLD_MODEL_FILENAME = "human_reference_alphafold_model.pdb"
ALPHAFOLD_OVERLAY_FILENAME = "human_reference_alphafold_overlay.pml"
ALPHAFOLD_VIEWER_JS_FILENAME = "3Dmol-min.js"
ALPHAFOLD_VIEWER_JS_URL = "https://3Dmol.org/build/3Dmol-min.js"
ALPHAFOLD_VIEWER_JS_URLS = (
    ALPHAFOLD_VIEWER_JS_URL,
    "https://cdn.jsdelivr.net/npm/3dmol/build/3Dmol-min.js",
)
SELECTED_CONSENSUS_CHUNKS_FILENAME = "selected_consensus_chunks.tsv"
SELECTED_CONSENSUS_CHUNKS_MAP_FILENAME = "selected_consensus_chunks_structure_map.tsv"
NOMENCLATURE_TREE_SVG_FILENAME = "phylo_tree_nomenclature.svg"

UNIPROT_DOMAIN_FEATURE_TYPES = {"Domain", "Region", "Repeat", "Zinc finger", "Motif"}
UNIPROT_FEATURE_TYPES = UNIPROT_DOMAIN_FEATURE_TYPES | {
    "Binding site", "Active site", "Modified residue", "Site", "Calcium binding", "Metal binding",
    "Disulfide bond", "Lipidation", "Compositional bias",
}
SELECTED_CONSENSUS_CHUNK_COLUMNS = [
    "chunk_id", "label", "start_reference_position", "end_reference_position",
    "score", "color_hex", "notes",
]
EVOLUTIONARY_ALIGNMENT_WINDOWS_DIRNAME = "evolutionary_alignment_windows"
EVOLUTIONARY_CORE_CLADE_ORDER = [
    "tetrapods",
    "actinistia",
    "holostei",
    "teleosts",
    "other_fish",
    "other_vertebrates",
]
EVOLUTIONARY_FIRST_DIVERGENCE_CLADE_ORDER = [
    "tetrapods",
    "dipnoi",
    "actinistia",
    "holostei",
    "teleosts",
    "other_fish",
    "other_vertebrates",
]
EVOLUTIONARY_CURATED_FEATURE_TYPES = {"Domain", "Region"}
EVOLUTIONARY_SINGLETON_WINDOW_FEATURE_TYPES = {"Active site", "Binding site", "Modified residue"}
EVOLUTIONARY_CLADE_COLORS: Dict[str, str] = {
    "tetrapods": "#0f766e",
    "dipnoi": "#7c3aed",
    "actinistia": "#c2410c",
    "holostei": "#65a30d",
    "teleosts": "#2563eb",
    "other_fish": "#d97706",
    "other_vertebrates": "#6b7280",
}

BLOSUM62 = substitution_matrices.load("BLOSUM62")
AA_COLORS: Dict[str, str] = {
    "A": "#C8C8C8", "V": "#00DCDC", "I": "#00DCDC", "L": "#00DCDC",
    "F": "#00DCDC", "W": "#00DCDC", "Y": "#00DCDC",
    "K": "#FF66CC", "R": "#FF66CC", "H": "#C090FF",
    "D": "#FF6666", "E": "#FF6666",
    "S": "#66FF66", "T": "#66FF66",
    "N": "#66FFAA", "Q": "#66FFAA",
    "C": "#FFFF66", "G": "#FFA94D", "P": "#FFB380",
    "M": "#E6E600", "-": "#FFFFFF", ".": "#FFFFFF", "X": "#F2F2F2", "?": "#F2F2F2",
}

CURATED_BUCKET_QUOTAS: Dict[str, int] = {
    "lower_eukaryote": 1,
    "reptile": 2,
    "fish": 3,
    "bird": 3,
    "mammal": 5,
}

CURATED_BUCKET_PRIORITY: Dict[str, Tuple[str, ...]] = {
    "lower_eukaryote": (
        "saccharomyces_cerevisiae", "schizosaccharomyces_pombe",
        "neurospora_crassa", "aspergillus_nidulans",
    ),
    "reptile": (
        "anolis_carolinensis", "python_bivittatus", "pelodiscus_sinensis",
        "chelonia_mydas", "alligator_mississippiensis",
    ),
    "fish": (
        "danio_rerio", "oryzias_latipes", "gasterosteus_aculeatus",
        "takifugu_rubripes", "tetraodon_nigroviridis", "xiphophorus_maculatus",
    ),
    "bird": (
        "gallus_gallus", "taeniopygia_guttata", "anas_platyrhynchos",
        "meleagris_gallopavo", "coturnix_japonica",
    ),
    "mammal": (
        "homo_sapiens", "mus_musculus", "rattus_norvegicus",
        "macaca_mulatta", "pan_troglodytes", "bos_taurus",
        "canis_lupus_familiaris", "sus_scrofa",
    ),
}


class APIError(RuntimeError):
    pass


ProgressCallback = Optional[Callable[[str], None]]


def emit_log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s"


def get_json(url: str, params: Optional[Dict[str, Any]] = None,
             headers: Optional[Dict[str, str]] = None,
             timeout: int = REQUEST_TIMEOUT) -> Any:
    hdr = headers or {"Accept": "application/json"}
    for attempt in range(1, REQUEST_RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, headers=hdr, timeout=timeout)
        except requests.RequestException as exc:
            if attempt >= REQUEST_RETRY_ATTEMPTS:
                raise APIError(f"Request failed for {url}: {exc}") from exc
            delay = REQUEST_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            emit_log(
                f"Transient request failure for {url}: {exc}. "
                f"Retrying in {delay:.1f}s ({attempt}/{REQUEST_RETRY_ATTEMPTS})."
            )
            time.sleep(delay)
            continue

        if response.ok:
            return response.json()

        if response.status_code in REQUEST_RETRY_STATUS_CODES and attempt < REQUEST_RETRY_ATTEMPTS:
            retry_after = response.headers.get("Retry-After")
            try:
                delay = max(float(retry_after), REQUEST_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
            except (TypeError, ValueError):
                delay = REQUEST_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            emit_log(
                f"HTTP {response.status_code} from {url}. "
                f"Retrying in {delay:.1f}s ({attempt}/{REQUEST_RETRY_ATTEMPTS})."
            )
            time.sleep(delay)
            continue

        raise APIError(f"{response.status_code} error for {url}: {response.text[:500]}")

    raise APIError(f"Request failed for {url} after {REQUEST_RETRY_ATTEMPTS} attempts.")


def get_text(url: str, params: Optional[Dict[str, Any]] = None,
             headers: Optional[Dict[str, str]] = None,
             timeout: int = REQUEST_TIMEOUT) -> str:
    hdr = headers or {"Accept": "text/plain"}
    for attempt in range(1, REQUEST_RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, headers=hdr, timeout=timeout)
        except requests.RequestException as exc:
            if attempt >= REQUEST_RETRY_ATTEMPTS:
                raise APIError(f"Request failed for {url}: {exc}") from exc
            delay = REQUEST_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            emit_log(
                f"Transient request failure for {url}: {exc}. "
                f"Retrying in {delay:.1f}s ({attempt}/{REQUEST_RETRY_ATTEMPTS})."
            )
            time.sleep(delay)
            continue

        if response.ok:
            return response.text

        if response.status_code in REQUEST_RETRY_STATUS_CODES and attempt < REQUEST_RETRY_ATTEMPTS:
            retry_after = response.headers.get("Retry-After")
            try:
                delay = max(float(retry_after), REQUEST_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
            except (TypeError, ValueError):
                delay = REQUEST_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            emit_log(
                f"HTTP {response.status_code} from {url}. "
                f"Retrying in {delay:.1f}s ({attempt}/{REQUEST_RETRY_ATTEMPTS})."
            )
            time.sleep(delay)
            continue

        raise APIError(f"{response.status_code} error for {url}: {response.text[:500]}")

    raise APIError(f"Request failed for {url} after {REQUEST_RETRY_ATTEMPTS} attempts.")


def download_file(url: str,
                  out_path: Path,
                  params: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None,
                  timeout: int = REQUEST_TIMEOUT) -> Path:
    hdr = headers or {"Accept": "*/*"}
    for attempt in range(1, REQUEST_RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(url, params=params, headers=hdr, timeout=timeout)
        except requests.RequestException as exc:
            if attempt >= REQUEST_RETRY_ATTEMPTS:
                raise APIError(f"Request failed for {url}: {exc}") from exc
            delay = REQUEST_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            emit_log(
                f"Transient request failure for {url}: {exc}. "
                f"Retrying in {delay:.1f}s ({attempt}/{REQUEST_RETRY_ATTEMPTS})."
            )
            time.sleep(delay)
            continue

        if response.ok:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(response.content)
            return out_path

        if response.status_code in REQUEST_RETRY_STATUS_CODES and attempt < REQUEST_RETRY_ATTEMPTS:
            retry_after = response.headers.get("Retry-After")
            try:
                delay = max(float(retry_after), REQUEST_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
            except (TypeError, ValueError):
                delay = REQUEST_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            emit_log(
                f"HTTP {response.status_code} from {url}. "
                f"Retrying in {delay:.1f}s ({attempt}/{REQUEST_RETRY_ATTEMPTS})."
            )
            time.sleep(delay)
            continue

        raise APIError(f"{response.status_code} error for {url}: {response.text[:500]}")

    raise APIError(f"Request failed for {url} after {REQUEST_RETRY_ATTEMPTS} attempts.")


def sanitize_filename(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        elif ch in (" ", "|", "/", "\\", ":"):
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "output"


def check_executable(explicit: Optional[str], candidates: Sequence[str], tool_name: str) -> str:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return str(path)
        resolved = shutil.which(explicit)
        if resolved:
            return resolved
        raise FileNotFoundError(f"{tool_name} not found: {explicit}")

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError(
        f"Could not find {tool_name}. Tried: {', '.join(candidates)}. "
        f"Provide the explicit executable path as needed."
    )


def normalize_tree_method(tree_method: Optional[str]) -> str:
    method = (tree_method or "auto").strip().lower()
    if method not in TREE_METHOD_CHOICES:
        raise RuntimeError(
            f"Unsupported tree method: {tree_method}. "
            f"Choose from: {', '.join(TREE_METHOD_CHOICES)}."
        )
    return method


def choose_worker_count(max_workers: int, reserve_one_core: bool = False) -> int:
    cpu_total = os.cpu_count() or 1
    available = max(1, cpu_total - 1) if reserve_one_core else max(1, cpu_total)
    return max(1, min(max_workers, available))


def species_to_scientific_name(species: Optional[str]) -> str:
    text = str(species or "").strip()
    if not text:
        return ""
    return text.replace("_", " ")


def species_to_display_label(species: Optional[str], common_name: Optional[str] = None) -> str:
    common = str(common_name or "").strip()
    if common:
        return common[:1].upper() + common[1:]
    scientific = species_to_scientific_name(species)
    if not scientific:
        return "Unknown"
    if scientific.lower() == "homo sapiens":
        return "Human"
    tokens = scientific.split()
    if len(tokens) == 2:
        genus, species_token = tokens
        if genus[:1].isupper() and species_token.islower():
            return f"{genus[:1]}. {species_token}"
    return scientific[:1].upper() + scientific[1:]


def protein_record_id_for(species: str,
                          translation_id: Optional[str] = None,
                          gene_id: Optional[str] = None,
                          symbol: Optional[str] = None) -> str:
    anchor = str(translation_id or gene_id or symbol or "unknown").strip() or "unknown"
    return sanitize_filename(f"{species}__{anchor}")


def clean_serializable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): clean_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_serializable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if pd.isna(value):
        return None
    return value


def write_json_file(out_path: Path, payload: Any) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(clean_serializable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ensembl_lookup_symbol(species: str, symbol: str, expand: int = 1) -> Dict[str, Any]:
    url = f"{ENSEMBL}/lookup/symbol/{species}/{symbol}"
    return get_json(url, params={"expand": expand})


def ensembl_lookup_id(stable_id: str, expand: int = 1) -> Dict[str, Any]:
    url = f"{ENSEMBL}/lookup/id/{stable_id}"
    return get_json(url, params={"expand": expand})


def ensembl_homologs_by_symbol(species: str, symbol: str,
                               target_species: Optional[List[str]] = None,
                               homology_type: str = "orthologues",
                               response_format: str = "condensed",
                               sequence: Optional[str] = None,
                               aligned: Optional[bool] = None) -> List[Dict[str, Any]]:
    url = f"{ENSEMBL}/homology/symbol/{species}/{symbol}"
    params: Dict[str, Any] = {"type": homology_type, "format": response_format}
    if target_species:
        params["target_species"] = target_species
    if sequence:
        params["sequence"] = sequence
    if aligned is not None:
        params["aligned"] = int(aligned)
    data = get_json(url, params=params)
    if not data.get("data"):
        return []
    return data["data"][0].get("homologies", [])


def extract_best_translation_id(lookup: Dict[str, Any]) -> Optional[str]:
    transcripts = lookup.get("Transcript", [])
    canonical_transcript = lookup.get("canonical_transcript")

    if canonical_transcript:
        for tx in transcripts:
            if tx.get("id") == canonical_transcript:
                tr = tx.get("Translation")
                if tr and tr.get("id"):
                    return tr["id"]

    for tx in transcripts:
        tr = tx.get("Translation")
        if tr and tr.get("id"):
            return tr["id"]

    return None


def ensembl_sequence_by_id(seq_id: str, seq_type: str = "protein") -> str:
    url = f"{ENSEMBL}/sequence/id/{seq_id}"
    data = get_json(url, params={"type": seq_type})
    seq = data.get("seq")
    if not seq:
        raise APIError(f"No sequence returned for {seq_id}")
    return seq


def infer_taxonomy_label(homology_record: Dict[str, Any]) -> str:
    target = homology_record.get("target", {})
    taxonomy = homology_record.get("taxonomy_level")
    if taxonomy:
        return str(taxonomy)
    species = target.get("species") or ""
    return species


def extract_homology_target_fields(homology: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize Ensembl homology records across `full` and `condensed` formats.
    In condensed responses, target fields live at the top level.
    """
    target = homology.get("target", {}) if isinstance(homology.get("target"), dict) else {}
    species = target.get("species") or homology.get("species")
    gene_id = target.get("id") or homology.get("id")
    symbol = target.get("display_id") or homology.get("display_id") or gene_id
    protein_id = target.get("protein_id") or homology.get("protein_id")
    perc_id = target.get("perc_id") or homology.get("perc_id")
    return {
        "species": species,
        "symbol": symbol,
        "ensembl_gene_id": gene_id,
        "ensembl_protein_id": protein_id,
        "perc_id_to_query": perc_id,
        "orthology_type": homology.get("type"),
    }


def maybe_filter_vertebrates(rows: List[Dict[str, Any]], enabled: bool) -> List[Dict[str, Any]]:
    if not enabled:
        return rows

    filtered: List[Dict[str, Any]] = []
    for row in rows:
        taxonomy_text = " ".join(
            [
                str(row.get("taxonomy_level", "")),
                str(row.get("species", "")),
                str(row.get("symbol", "")),
            ]
        ).lower()

        if row.get("species") == "homo_sapiens":
            filtered.append(row)
            continue

        if any(k in taxonomy_text for k in INVERTEBRATE_KEYWORDS):
            continue

        if any(k in taxonomy_text for k in VERTEBRATE_KEYWORDS):
            filtered.append(row)
            continue

        # Be permissive when taxonomy information is sparse or uses an unexpected vertebrate clade.
        # Ensembl homology results often encode vertebrate orthologs under narrower clades.
        filtered.append(row)

    return filtered


def normalize_selection_mode(selection_mode: Optional[str]) -> str:
    mode = (selection_mode or "all_filtered").strip().lower()
    if mode not in SELECTION_MODE_CHOICES:
        raise RuntimeError(
            f"Unsupported selection mode: {selection_mode}. "
            f"Choose from: {', '.join(SELECTION_MODE_CHOICES)}."
        )
    return mode


def classify_species_bucket(species: str, taxonomy_level: Optional[str] = None) -> str:
    species_text = (species or "").strip().lower()
    taxonomy_text = (taxonomy_level or "").strip().lower()
    joined = f"{species_text} {taxonomy_text}"

    if any(k in joined for k in ("saccharomy", "schizo", "fungi", "ascomycota", "yeast", "neurospora", "aspergillus")):
        return "lower_eukaryote"
    if any(k in joined for k in ("mammalia", "eutheria", "theria", "primates", "rodentia")):
        return "mammal"
    if any(k in joined for k in ("aves", "bird", "gallus", "taeniopygia", "anas", "meleagris", "coturnix")):
        return "bird"
    if any(k in joined for k in ("reptilia", "sauropsida", "anolis", "python", "pelodiscus", "alligator", "chelonia")):
        return "reptile"
    if any(k in joined for k in ("actinopterygii", "teleost", "cypriniformes", "clupeocephala", "danio", "oryzias", "gasterosteus", "takifugu", "tetraodon", "xiphophorus")):
        return "fish"

    if species_text in CURATED_BUCKET_PRIORITY["mammal"]:
        return "mammal"
    if species_text in CURATED_BUCKET_PRIORITY["bird"]:
        return "bird"
    if species_text in CURATED_BUCKET_PRIORITY["reptile"]:
        return "reptile"
    if species_text in CURATED_BUCKET_PRIORITY["fish"]:
        return "fish"
    if species_text in CURATED_BUCKET_PRIORITY["lower_eukaryote"]:
        return "lower_eukaryote"

    return "other"


def _curated_priority_key(row: pd.Series, bucket: str, source_species: str) -> Tuple[int, int, float, str]:
    species = str(row.get("species") or "").strip().lower()
    preferred = CURATED_BUCKET_PRIORITY.get(bucket, ())
    if species == source_species.lower():
        rank = -1
    elif species in preferred:
        rank = preferred.index(species)
    else:
        rank = len(preferred) + 100

    is_query = 1 if bool(row.get("is_query")) else 0
    perc_id = row.get("perc_id_to_query")
    try:
        perc_key = -float(perc_id) if pd.notna(perc_id) else 9999.0
    except Exception:
        perc_key = 9999.0
    return (rank, -is_query, perc_key, species)


def select_curated_phylo_subset(ortholog_df: pd.DataFrame, source_species: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if ortholog_df.empty:
        return ortholog_df.copy(), pd.DataFrame(columns=["species", "bucket", "selected", "reason"])

    work = ortholog_df.copy()
    work["bucket"] = work.apply(
        lambda row: classify_species_bucket(str(row.get("species") or ""), row.get("taxonomy_level")),
        axis=1,
    )
    selected_parts: List[pd.DataFrame] = []

    for bucket, quota in CURATED_BUCKET_QUOTAS.items():
        bucket_df = work[work["bucket"] == bucket].copy()
        if bucket_df.empty:
            continue
        bucket_df["_priority"] = bucket_df.apply(lambda row: _curated_priority_key(row, bucket, source_species), axis=1)
        bucket_df = bucket_df.sort_values(by="_priority", kind="stable").drop(columns=["_priority"])
        chosen = bucket_df.head(quota).copy()
        if not chosen.empty:
            chosen["selection_reason"] = f"curated_{bucket}"
            selected_parts.append(chosen)

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=work.columns.tolist() + ["selection_reason"])

    source_rows = work[work["species"].astype(str).str.lower() == source_species.lower()].copy()
    if not source_rows.empty:
        source_rows["selection_reason"] = "source_reference"
        selected = pd.concat([selected, source_rows], ignore_index=True)

    if not selected.empty:
        selected = selected.sort_values(by=["is_query", "species"], ascending=[False, True]).drop_duplicates(subset=["species"], keep="first").reset_index(drop=True)

    manifest = work[["species", "symbol", "taxonomy_level", "perc_id_to_query", "bucket"]].copy()
    selected_species = selected["species"].tolist() if not selected.empty else []
    manifest["selected"] = manifest["species"].isin(selected_species)
    reason_map = dict(zip(selected.get("species", []), selected.get("selection_reason", []))) if not selected.empty else {}
    manifest["reason"] = manifest["species"].map(reason_map).fillna("not_selected")

    if len(selected) < 3:
        fallback = work.copy()
        fallback["_perc"] = pd.to_numeric(fallback["perc_id_to_query"], errors="coerce").fillna(-1.0)
        fallback = fallback.sort_values(by=["is_query", "_perc", "species"], ascending=[False, False, True])
        selected = fallback.head(max(3, min(len(fallback), 6))).drop(columns=["_perc"]).reset_index(drop=True)
        selected["selection_reason"] = "curated_fallback"
        selected_species = selected["species"].tolist()
        manifest["selected"] = manifest["species"].isin(selected_species)
        manifest["reason"] = manifest["species"].apply(lambda s: "curated_fallback" if s in selected_species else "not_selected")

    return selected.reset_index(drop=True), manifest


def collect_ortholog_table(source_species: str,
                           gene_symbol: str,
                           target_species: Optional[List[str]],
                           filter_vertebrates: bool,
                           selection_mode: str = "all_filtered") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    emit_log(f"Looking up query gene {gene_symbol} in Ensembl for {source_species}.")
    query_lookup = ensembl_lookup_symbol(source_species, gene_symbol, expand=1)
    rows.append({
        "species": source_species,
        "symbol": gene_symbol,
        "ensembl_gene_id": query_lookup.get("id"),
        "ensembl_protein_id": extract_best_translation_id(query_lookup),
        "perc_id_to_query": 100.0,
        "orthology_type": "query",
        "taxonomy_level": source_species,
        "is_query": True,
    })

    homologies = ensembl_homologs_by_symbol(source_species, gene_symbol, target_species=target_species)
    emit_log(f"Ensembl returned {len(homologies)} orthology records before filtering.")
    for homology in homologies:
        target_fields = extract_homology_target_fields(homology)
        rows.append({
            "species": target_fields["species"],
            "symbol": target_fields["symbol"],
            "ensembl_gene_id": target_fields["ensembl_gene_id"],
            "ensembl_protein_id": target_fields["ensembl_protein_id"],
            "perc_id_to_query": target_fields["perc_id_to_query"],
            "orthology_type": target_fields["orthology_type"],
            "taxonomy_level": infer_taxonomy_label(homology),
            "is_query": False,
        })

    rows = maybe_filter_vertebrates(rows, filter_vertebrates)

    df = pd.DataFrame(rows)
    df = df[df["species"].notna() & df["ensembl_gene_id"].notna()].copy()
    df = df.drop_duplicates(subset=["species", "ensembl_gene_id"]).reset_index(drop=True)
    emit_log(f"Retained {len(df)} ortholog rows after filtering and deduplication.")

    selection_mode = normalize_selection_mode(selection_mode)
    if selection_mode == "curated_subset":
        selected_df, _ = select_curated_phylo_subset(df, source_species=source_species)
        emit_log(f"Curated subset mode retained {len(selected_df)} species.")
        return selected_df
    emit_log(f"Using all filtered ortholog rows ({len(df)} species).")
    return df




def normalize_text_token(text: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def parse_fasta_object_metadata(fasta_obj_path: Optional[str]) -> pd.DataFrame:
    if not fasta_obj_path:
        return pd.DataFrame()
    path = Path(fasta_obj_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA object metadata file was not found: {fasta_obj_path}")

    rows: List[Dict[str, Any]] = []
    current_header: Optional[str] = None
    seq_chunks: List[str] = []

    def flush_record() -> None:
        nonlocal current_header, seq_chunks
        if not current_header:
            return
        header = current_header[1:].strip() if current_header.startswith('>') else current_header.strip()
        object_id = header
        meta: Dict[str, Any] = {}
        if ' {' in header and header.endswith('}'):
            left, right = header.split(' {', 1)
            object_id = left.strip()
            try:
                meta = json.loads('{' + right)
            except Exception:
                meta = {}
        organism_name = meta.get('organism_name')
        pub_gene_id = meta.get('pub_gene_id')
        description = meta.get('description')
        og_name = meta.get('og_name')
        seq = ''.join(seq_chunks).strip()
        rows.append({
            'object_id': object_id,
            'organism_name': organism_name,
            'species': str(organism_name).strip().lower().replace(' ', '_') if organism_name else None,
            'pub_gene_id': pub_gene_id,
            'description': description,
            'og_name': og_name,
            'pub_og_id': meta.get('pub_og_id'),
            'level_taxid': meta.get('level_taxid'),
            'organism_taxid': meta.get('organism_taxid'),
            'raw_header': header,
            'seq_len': len(seq),
            'sequence': seq,
        })
        current_header = None
        seq_chunks = []

    with path.open('r', encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush_record()
                current_header = line
            else:
                seq_chunks.append(line.strip())
    flush_record()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ['organism_name', 'pub_gene_id', 'description', 'og_name', 'raw_header', 'species']:
        df[f'{col}_norm'] = df[col].map(normalize_text_token)
    return df


def pick_best_fasta_object_metadata(metadata_df: pd.DataFrame,
                                    species: str,
                                    gene_symbol: str,
                                    gene_label: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if metadata_df is None or metadata_df.empty:
        return None
    species_key = species.strip().lower().replace(' ', '_')
    sub = metadata_df[metadata_df['species'] == species_key].copy()
    if sub.empty:
        return None
    symbol_norm = normalize_text_token(gene_symbol)
    label_norm = normalize_text_token(gene_label)

    def score_row(row: pd.Series) -> Tuple[int, int, int, int]:
        hay = ' '.join([
            str(row.get('pub_gene_id_norm') or ''),
            str(row.get('description_norm') or ''),
            str(row.get('og_name_norm') or ''),
            str(row.get('raw_header_norm') or ''),
        ])
        exact_gene = 1 if symbol_norm and symbol_norm in hay else 0
        exact_label = 1 if label_norm and label_norm in hay else 0
        looks_like_phospholipase = 1 if ('phospholipase' in hay or 'pla2' in hay or 'cpla2' in hay) else 0
        seq_len = int(row.get('seq_len') or 0)
        return (exact_gene, exact_label, looks_like_phospholipase, seq_len)

    sub['_score'] = sub.apply(score_row, axis=1)
    sub = sub.sort_values(by=['_score'], ascending=False, kind='stable')
    best = sub.iloc[0].drop(labels=['_score']).to_dict()
    return best


def build_sequence_header(species: str,
                          resolved_symbol: str,
                          gene_id: Optional[str],
                          translation_id: Optional[str],
                          protein_record_id: str,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
    parts = [
        species,
        f'ProteinRecordID={protein_record_id}',
        f'Gene={resolved_symbol}',
        f'EnsemblGene={gene_id}',
        f'Protein={translation_id}',
    ]
    if metadata:
        display = metadata.get('pub_gene_id') or metadata.get('description') or metadata.get('og_name')
        if display:
            parts.append(f'Label={sanitize_filename(str(display))}')
        if metadata.get('organism_name'):
            parts.append(f'Organism={sanitize_filename(str(metadata.get("organism_name")))}')
        if metadata.get('description'):
            parts.append(f'Description={sanitize_filename(str(metadata.get("description")))}')
        if metadata.get('object_id'):
            parts.append(f'MetadataID={sanitize_filename(str(metadata.get("object_id")))}')
    return '|'.join(str(p) for p in parts if p is not None)


def _collect_sequence_worker(row_data: Dict[str, Any],
                             metadata_df: pd.DataFrame,
                             gene_symbol: Optional[str] = None,
                             gene_label: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[SeqRecord], str]:
    species = str(row_data["species"])
    symbol = str(row_data["symbol"])
    ensembl_gene_id = row_data.get("ensembl_gene_id")
    ensembl_protein_id = row_data.get("ensembl_protein_id")

    lookup = None
    gene_id = None
    translation_id = None

    if pd.notna(ensembl_protein_id) and str(ensembl_protein_id).strip():
        translation_id = str(ensembl_protein_id).strip()
        gene_id = str(ensembl_gene_id).strip() if pd.notna(ensembl_gene_id) else None

    if translation_id is None and pd.notna(ensembl_gene_id) and str(ensembl_gene_id).strip():
        try:
            lookup = ensembl_lookup_id(str(ensembl_gene_id), expand=1)
            gene_id = lookup.get("id")
        except Exception:
            lookup = None

    if translation_id is None and lookup is None:
        lookup = ensembl_lookup_symbol(species, symbol, expand=1)
        gene_id = lookup.get("id")

    resolved_symbol = (lookup.get("display_name") if lookup else None) or symbol
    if translation_id is None:
        translation_id = extract_best_translation_id(lookup)
    protein_record_id = protein_record_id_for(
        species=species,
        translation_id=translation_id,
        gene_id=gene_id,
        symbol=resolved_symbol,
    )
    if not translation_id:
        return ({
            "species": species,
            "symbol": resolved_symbol,
            "ensembl_gene_id": gene_id,
            "translation_id": None,
            "protein_record_id": protein_record_id,
            "sequence_header": None,
            "length_aa": None,
            "status": "no_translation",
            "metadata_object_id": None,
            "metadata_organism_name": None,
            "metadata_pub_gene_id": None,
            "metadata_description": None,
            "metadata_og_name": None,
            "metadata_raw_header": None,
        }, None, f"no translated protein was available for {species}")

    protein_seq = ensembl_sequence_by_id(translation_id, seq_type="protein")
    metadata = pick_best_fasta_object_metadata(
        metadata_df,
        species=species,
        gene_symbol=gene_symbol or resolved_symbol,
        gene_label=gene_label,
    )
    header = build_sequence_header(
        species,
        resolved_symbol,
        gene_id,
        translation_id,
        protein_record_id=protein_record_id,
        metadata=metadata,
    )
    record = SeqRecord(Seq(protein_seq), id=header, description="")
    return ({
        "species": species,
        "symbol": resolved_symbol,
        "ensembl_gene_id": gene_id,
        "translation_id": translation_id,
        "protein_record_id": protein_record_id,
        "sequence_header": header,
        "length_aa": len(protein_seq),
        "status": "ok",
        "metadata_object_id": (metadata or {}).get("object_id"),
        "metadata_organism_name": (metadata or {}).get("organism_name"),
        "metadata_pub_gene_id": (metadata or {}).get("pub_gene_id"),
        "metadata_description": (metadata or {}).get("description"),
        "metadata_og_name": (metadata or {}).get("og_name"),
        "metadata_raw_header": (metadata or {}).get("raw_header"),
    }, record, f"recovered {len(protein_seq)} aa for {species}")


def collect_sequences(ortholog_df: pd.DataFrame,
                      gene_symbol: Optional[str] = None,
                      gene_label: Optional[str] = None,
                      fasta_object_metadata_path: Optional[str] = None) -> Tuple[pd.DataFrame, List[SeqRecord]]:
    rows: List[Dict[str, Any]] = []
    records: List[SeqRecord] = []
    metadata_df = parse_fasta_object_metadata(fasta_object_metadata_path)
    total = len(ortholog_df.index)

    if fasta_object_metadata_path:
        emit_log(f"Loading FASTA metadata from {Path(fasta_object_metadata_path).resolve()}.")
        emit_log(f"Loaded {len(metadata_df)} metadata records for sequence label matching.")

    jobs = [(idx, row.to_dict()) for idx, (_, row) in enumerate(ortholog_df.iterrows(), start=1)]
    worker_count = min(total, choose_worker_count(SEQUENCE_FETCH_MAX_WORKERS)) if total else 1
    emit_log(
        f"Fetching {total} protein sequences with up to {worker_count} concurrent "
        f"Ensembl requests."
    )
    ordered_results: Dict[int, Tuple[Dict[str, Any], Optional[SeqRecord]]] = {}

    def store_result(job_index: int, result_row: Dict[str, Any],
                     record: Optional[SeqRecord], message: str) -> None:
        ordered_results[job_index] = (result_row, record)
        emit_log(f"Sequence {job_index}/{total}: {message}.")

    if worker_count <= 1:
        for idx, row_data in jobs:
            try:
                result_row, record, message = _collect_sequence_worker(
                    row_data,
                    metadata_df,
                    gene_symbol=gene_symbol,
                    gene_label=gene_label,
                )
            except Exception as exc:
                result_row = {
                    "species": str(row_data["species"]),
                    "symbol": str(row_data["symbol"]),
                    "ensembl_gene_id": row_data.get("ensembl_gene_id"),
                    "translation_id": None,
                    "protein_record_id": protein_record_id_for(
                        species=str(row_data["species"]),
                        translation_id=row_data.get("ensembl_protein_id"),
                        gene_id=row_data.get("ensembl_gene_id"),
                        symbol=str(row_data["symbol"]),
                    ),
                    "sequence_header": None,
                    "length_aa": None,
                    "status": f"error: {exc}",
                    "metadata_object_id": None,
                    "metadata_organism_name": None,
                    "metadata_pub_gene_id": None,
                    "metadata_description": None,
                    "metadata_og_name": None,
                    "metadata_raw_header": None,
                }
                record = None
                message = f"failed for {row_data['species']}: {exc}"
            store_result(idx, result_row, record, message)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    _collect_sequence_worker,
                    row_data,
                    metadata_df,
                    gene_symbol,
                    gene_label,
                ): (idx, row_data)
                for idx, row_data in jobs
            }
            for future in as_completed(future_map):
                idx, row_data = future_map[future]
                try:
                    result_row, record, message = future.result()
                except Exception as exc:
                    result_row = {
                        "species": str(row_data["species"]),
                        "symbol": str(row_data["symbol"]),
                        "ensembl_gene_id": row_data.get("ensembl_gene_id"),
                        "translation_id": None,
                        "protein_record_id": protein_record_id_for(
                            species=str(row_data["species"]),
                            translation_id=row_data.get("ensembl_protein_id"),
                            gene_id=row_data.get("ensembl_gene_id"),
                            symbol=str(row_data["symbol"]),
                        ),
                        "sequence_header": None,
                        "length_aa": None,
                        "status": f"error: {exc}",
                        "metadata_object_id": None,
                        "metadata_organism_name": None,
                        "metadata_pub_gene_id": None,
                        "metadata_description": None,
                        "metadata_og_name": None,
                        "metadata_raw_header": None,
                    }
                    record = None
                    message = f"failed for {row_data['species']}: {exc}"
                store_result(idx, result_row, record, message)

    for idx in range(1, total + 1):
        result_row, record = ordered_results[idx]
        rows.append(result_row)
        if record is not None:
            records.append(record)

    seq_df = pd.DataFrame(rows)
    ok_count = int((seq_df["status"] == "ok").sum()) if not seq_df.empty else 0
    emit_log(f"Sequence recovery complete: {ok_count}/{total} proteins recovered.")
    return seq_df, records


def _parse_reference_guided_pair(reference_seq: str, aligned_reference: str,
                                 aligned_target: str) -> Tuple[List[str], List[str]]:
    """
    Parse a pairwise alignment against the ungapped reference sequence into:
    - inserted residues at each reference boundary
    - one aligned residue/gap per reference position
    """
    if len(aligned_reference) != len(aligned_target):
        raise RuntimeError("Aligned reference and target sequences have different lengths.")

    insertions = ["" for _ in range(len(reference_seq) + 1)]
    residues = ["-" for _ in range(len(reference_seq))]
    ref_pos = 0
    boundary = 0

    for ref_char, tgt_char in zip(aligned_reference, aligned_target):
        if ref_char == "-":
            if tgt_char != "-":
                insertions[boundary] += tgt_char
            continue

        if ref_pos >= len(reference_seq):
            raise RuntimeError("Aligned reference extends beyond the ungapped reference length.")
        residues[ref_pos] = tgt_char if tgt_char != "-" else "-"
        ref_pos += 1
        boundary = ref_pos

    if ref_pos != len(reference_seq):
        raise RuntimeError("Aligned reference does not cover the full ungapped reference sequence.")
    return insertions, residues


def build_reference_guided_alignment(reference_seq: str,
                                     query_header: str,
                                     pair_rows: List[Dict[str, Any]]) -> MultipleSeqAlignment:
    if not pair_rows:
        raise RuntimeError("No pairwise alignments were available to build a reference-guided alignment.")

    parsed_rows = []
    max_insert_lengths = [0 for _ in range(len(reference_seq) + 1)]

    for row in pair_rows:
        insertions, residues = _parse_reference_guided_pair(
            reference_seq,
            str(row["source_aligned"]),
            str(row["target_aligned"]),
        )
        parsed_rows.append((row, insertions, residues))
        for idx, ins in enumerate(insertions):
            if len(ins) > max_insert_lengths[idx]:
                max_insert_lengths[idx] = len(ins)

    aligned_records: List[SeqRecord] = []

    query_parts: List[str] = []
    for idx, aa in enumerate(reference_seq):
        query_parts.append("-" * max_insert_lengths[idx])
        query_parts.append(aa)
    query_parts.append("-" * max_insert_lengths[len(reference_seq)])
    aligned_records.append(SeqRecord(Seq("".join(query_parts)), id=query_header, description=""))

    for row, insertions, residues in parsed_rows:
        parts: List[str] = []
        for idx, aa in enumerate(residues):
            parts.append(insertions[idx].ljust(max_insert_lengths[idx], "-"))
            parts.append(aa)
        parts.append(insertions[len(reference_seq)].ljust(max_insert_lengths[len(reference_seq)], "-"))
        aligned_records.append(SeqRecord(Seq("".join(parts)), id=str(row["header"]), description=""))

    return MultipleSeqAlignment(aligned_records)


def build_ensembl_reference_guided_alignment(source_species: str,
                                             gene_symbol: str,
                                             ortholog_df: Optional[pd.DataFrame] = None,
                                             target_species: Optional[List[str]] = None,
                                             filter_vertebrates: bool = True,
                                             one2one_only: bool = True) -> MultipleSeqAlignment:
    emit_log("Requesting Ensembl reference-guided protein alignment records.")
    homologies = ensembl_homologs_by_symbol(
        source_species,
        gene_symbol,
        target_species=target_species,
        homology_type="orthologues",
        response_format="full",
        sequence="protein",
        aligned=True,
    )
    if not homologies:
        raise RuntimeError("Ensembl did not return aligned orthology records for this gene.")
    emit_log(f"Ensembl returned {len(homologies)} aligned orthology records.")

    allowed_gene_ids: Optional[set[str]] = None
    symbol_by_gene_id: Dict[str, str] = {}
    if ortholog_df is not None and not ortholog_df.empty:
        allowed_gene_ids = {
            str(gid) for gid in ortholog_df["ensembl_gene_id"].tolist()
            if pd.notna(gid) and str(gid).strip()
        }
        for _, row in ortholog_df.iterrows():
            gid = row.get("ensembl_gene_id")
            symbol = row.get("symbol")
            if pd.notna(gid) and str(gid).strip() and pd.notna(symbol):
                symbol_by_gene_id[str(gid)] = str(symbol)

    pair_rows: List[Dict[str, Any]] = []
    reference_seq = None
    query_header = None

    for homology in homologies:
        target = homology.get("target", {})
        source = homology.get("source", {})

        gene_id = target.get("id")
        species = target.get("species")
        source_aligned = source.get("align_seq")
        target_aligned = target.get("align_seq")
        if not gene_id or not species or not source_aligned or not target_aligned:
            continue

        if one2one_only and homology.get("type") != "ortholog_one2one":
            continue

        if allowed_gene_ids is not None and str(gene_id) not in allowed_gene_ids:
            continue

        synthetic_row = {
            "species": species,
            "symbol": symbol_by_gene_id.get(str(gene_id), str(gene_id)),
            "taxonomy_level": infer_taxonomy_label(homology),
            "is_query": False,
        }
        if filter_vertebrates and not maybe_filter_vertebrates([synthetic_row], enabled=True):
            continue

        if reference_seq is None:
            reference_seq = str(source_aligned).replace("-", "")
            source_gene_id = source.get("id")
            source_protein_id = source.get("protein_id")
            query_header = (
                f"{source_species}|Gene={gene_symbol}|EnsemblGene={source_gene_id}|Protein={source_protein_id}"
            )

        symbol = symbol_by_gene_id.get(str(gene_id), str(gene_id))
        header = f"{species}|Gene={symbol}|EnsemblGene={gene_id}|Protein={target.get('protein_id')}"
        pair_rows.append({
            "header": header,
            "source_aligned": source_aligned,
            "target_aligned": target_aligned,
        })

    if reference_seq is None or query_header is None or not pair_rows:
        raise RuntimeError("No Ensembl aligned ortholog sequences remained after filtering.")

    emit_log(f"Constructed Ensembl reference-guided alignment from {len(pair_rows)} ortholog pairs.")
    return build_reference_guided_alignment(reference_seq, query_header, pair_rows)





def pairwise_aligner_global_strings(reference_seq: str,
                                    target_seq: str,
                                    gap_open: float = -10.0,
                                    gap_extend: float = -0.5) -> Tuple[str, str]:
    """
    Global protein alignment using Bio.Align.PairwiseAligner.

    This replaces the deprecated Bio.pairwise2 fallback path. The returned
    strings are the aligned reference and aligned target sequences, including
    gap characters, and are suitable for reference-guided MSA merging.
    """
    ref = str(reference_seq).replace("-", "").replace(".", "").upper()
    tgt = str(target_seq).replace("-", "").replace(".", "").upper()
    if not ref:
        raise RuntimeError("Fallback alignment failed because the reference sequence is empty.")
    if not tgt:
        raise RuntimeError("Fallback alignment failed because a target sequence is empty.")

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = BLOSUM62
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend

    alignments = aligner.align(ref, tgt)
    if len(alignments) < 1:
        raise RuntimeError("PairwiseAligner returned no alignments.")
    aln = alignments[0]

    target_blocks, query_blocks = aln.aligned
    ref_out: List[str] = []
    tgt_out: List[str] = []
    ref_pos = 0
    tgt_pos = 0

    for (ref_start, ref_end), (tgt_start, tgt_end) in zip(target_blocks, query_blocks):
        ref_start = int(ref_start)
        ref_end = int(ref_end)
        tgt_start = int(tgt_start)
        tgt_end = int(tgt_end)

        if ref_start > ref_pos:
            ref_out.append(ref[ref_pos:ref_start])
            tgt_out.append("-" * (ref_start - ref_pos))
        if tgt_start > tgt_pos:
            ref_out.append("-" * (tgt_start - tgt_pos))
            tgt_out.append(tgt[tgt_pos:tgt_start])

        ref_out.append(ref[ref_start:ref_end])
        tgt_out.append(tgt[tgt_start:tgt_end])
        ref_pos = ref_end
        tgt_pos = tgt_end

    if ref_pos < len(ref):
        ref_out.append(ref[ref_pos:])
        tgt_out.append("-" * (len(ref) - ref_pos))
    if tgt_pos < len(tgt):
        ref_out.append("-" * (len(tgt) - tgt_pos))
        tgt_out.append(tgt[tgt_pos:])

    aligned_ref = "".join(ref_out)
    aligned_tgt = "".join(tgt_out)
    if len(aligned_ref) != len(aligned_tgt):
        raise RuntimeError("PairwiseAligner reconstruction produced unequal aligned string lengths.")
    return aligned_ref, aligned_tgt


def _fallback_alignment_worker(reference_seq: str, record_id: str, target_seq: str) -> Dict[str, Any]:
    aligned_ref, aligned_target = pairwise_aligner_global_strings(reference_seq, target_seq)
    return {
        "header": record_id,
        "source_aligned": aligned_ref,
        "target_aligned": aligned_target,
    }


def fallback_alignment(records: List[SeqRecord]) -> MultipleSeqAlignment:
    """
    Reference-guided progressive fallback MSA using PairwiseAligner.

    The first record is used as the reference. Each other sequence is globally
    aligned to the ungapped reference, then all pairwise alignments are merged
    into one MSA by retaining one column per reference residue plus the maximum
    insertion width observed at each reference boundary. This keeps downstream
    human-reference projection stable and avoids the Biopython pairwise2
    deprecation warning.
    """
    if not records:
        raise RuntimeError("No records were provided to the Python fallback aligner.")
    if len(records) == 1:
        return MultipleSeqAlignment([SeqRecord(Seq(str(records[0].seq)), id=records[0].id, description="")])

    reference_record = records[0]
    reference_seq = str(reference_record.seq).replace("-", "").replace(".", "").upper()
    if not reference_seq:
        raise RuntimeError("The first/reference sequence is empty, so fallback alignment cannot run.")

    pair_rows: List[Dict[str, Any]] = []
    total = len(records) - 1
    worker_count = min(total, choose_worker_count(PYTHON_FALLBACK_MAX_WORKERS, reserve_one_core=True))
    emit_log(
        f"Python fallback aligner is processing {total} pairwise alignments "
        f"with up to {worker_count} worker process{'es' if worker_count != 1 else ''}."
    )

    if worker_count <= 1:
        for idx, record in enumerate(records[1:], start=1):
            emit_log(f"Python fallback pairwise alignment {idx}/{total}: {short_record_label(record.id, 60)}")
            pair_rows.append(_fallback_alignment_worker(reference_seq, record.id, str(record.seq)))
    else:
        ordered_rows: Dict[int, Dict[str, Any]] = {}
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        _fallback_alignment_worker,
                        reference_seq,
                        record.id,
                        str(record.seq),
                    ): (idx, record.id)
                    for idx, record in enumerate(records[1:], start=1)
                }
                for future in as_completed(future_map):
                    idx, record_id = future_map[future]
                    ordered_rows[idx] = future.result()
                    emit_log(
                        f"Python fallback pairwise alignment {idx}/{total}: "
                        f"{short_record_label(record_id, 60)} complete."
                    )
            pair_rows = [ordered_rows[idx] for idx in range(1, total + 1)]
        except Exception as exc:
            emit_log(
                f"Python fallback worker pool was unavailable ({exc}). "
                "Falling back to sequential pairwise alignment."
            )
            pair_rows = []
            for idx, record in enumerate(records[1:], start=1):
                emit_log(f"Python fallback pairwise alignment {idx}/{total}: {short_record_label(record.id, 60)}")
                pair_rows.append(_fallback_alignment_worker(reference_seq, record.id, str(record.seq)))

    return build_reference_guided_alignment(reference_seq, reference_record.id, pair_rows)


def write_fasta(records: List[SeqRecord], fasta_path: Path) -> None:
    if not records:
        raise RuntimeError("No protein sequences were collected.")
    SeqIO.write(records, str(fasta_path), "fasta")


def run_mafft(input_fasta: Path, output_fasta: Path, mafft_exe: str, accurate: bool) -> None:
    if accurate:
        cmd = [mafft_exe, "--thread", "-1", "--threadit", "0", "--localpair", "--maxiterate", "1000", str(input_fasta)]
    else:
        cmd = [mafft_exe, "--thread", "-1", "--auto", str(input_fasta)]

    with output_fasta.open("w", encoding="utf-8") as out_handle:
        proc = subprocess.run(cmd, stdout=out_handle, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"MAFFT failed:\n{proc.stderr}")


def _run_subprocess_capture_stdout(cmd: List[str], output_fasta: Path, tool_name: str) -> None:
    with output_fasta.open("w", encoding="utf-8") as out_handle:
        proc = subprocess.run(cmd, stdout=out_handle, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{tool_name} failed:\n{proc.stderr}")


def run_muscle(input_fasta: Path, output_fasta: Path, muscle_exe: str, sequence_count: int) -> str:
    """
    Support both MUSCLE v5 style CLI (-align/-output) and older v3 style (-in/-out).
    """
    preferred_super5 = sequence_count >= MUSCLE_SUPER5_SEQUENCE_THRESHOLD
    if preferred_super5:
        attempts = [
            ("super5", [muscle_exe, "-super5", str(input_fasta), "-output", str(output_fasta)]),
            ("align", [muscle_exe, "-align", str(input_fasta), "-output", str(output_fasta)]),
            ("legacy", [muscle_exe, "-in", str(input_fasta), "-out", str(output_fasta)]),
        ]
    else:
        attempts = [
            ("align", [muscle_exe, "-align", str(input_fasta), "-output", str(output_fasta)]),
            ("super5", [muscle_exe, "-super5", str(input_fasta), "-output", str(output_fasta)]),
            ("legacy", [muscle_exe, "-in", str(input_fasta), "-out", str(output_fasta)]),
        ]
    errors: List[str] = []
    for mode, cmd in attempts:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0 and output_fasta.exists() and output_fasta.stat().st_size > 0:
            return mode
        errors.append(
            "CMD: " + " ".join(cmd) + "\n" + (proc.stderr or proc.stdout or "No output returned.")
        )
    raise RuntimeError("MUSCLE failed after multiple command styles:\n\n" + "\n\n".join(errors))


def run_alignment(records: List[SeqRecord], input_fasta: Path, output_fasta: Path,
                  alignment_method: str,
                  mafft_exe: Optional[str], muscle_exe: Optional[str], accurate: bool,
                  source_species: Optional[str] = None,
                  gene_symbol: Optional[str] = None,
                  target_species: Optional[List[str]] = None,
                  filter_vertebrates: bool = True,
                  ortholog_df: Optional[pd.DataFrame] = None) -> str:
    """
    Run the selected aligner when available.

    alignment_method options:
        auto, mafft, muscle, ensembl, python
    """
    emit_log(f"Preparing alignment input FASTA with {len(records)} sequences.")
    write_fasta(records, input_fasta)
    method = (alignment_method or "auto").strip().lower()

    if method == "mafft":
        if not mafft_exe:
            raise RuntimeError("Alignment method 'mafft' was selected, but no MAFFT executable was found.")
        emit_log(
            f"Running MAFFT alignment with {len(records)} sequences "
            f"({'accurate localpair' if accurate else 'auto'} mode, threads=auto)."
        )
        run_mafft(input_fasta, output_fasta, mafft_exe=mafft_exe, accurate=accurate)
        emit_log("MAFFT alignment complete.")
        return "mafft"

    if method == "muscle":
        if not muscle_exe:
            raise RuntimeError("Alignment method 'muscle' was selected, but no MUSCLE executable was found.")
        emit_log(
            f"Running MUSCLE alignment with {len(records)} sequences "
            f"(preferred mode={'super5' if len(records) >= MUSCLE_SUPER5_SEQUENCE_THRESHOLD else 'align'})."
        )
        muscle_mode = run_muscle(input_fasta, output_fasta, muscle_exe=muscle_exe, sequence_count=len(records))
        emit_log(f"MUSCLE alignment complete (mode={muscle_mode}).")
        return "muscle"


    if method == "ensembl":
        if not (source_species and gene_symbol):
            raise RuntimeError("Alignment method 'ensembl' requires source species and gene symbol.")
        emit_log("Running Ensembl reference-guided alignment.")
        msa = build_ensembl_reference_guided_alignment(
            source_species=source_species,
            gene_symbol=gene_symbol,
            ortholog_df=ortholog_df,
            target_species=target_species,
            filter_vertebrates=filter_vertebrates,
            one2one_only=True,
        )
        AlignIO.write(msa, str(output_fasta), "fasta")
        emit_log("Ensembl reference-guided alignment complete.")
        return "ensembl_pairwise_merge"

    if method == "python":
        emit_log("Running built-in Python fallback alignment.")
        msa = fallback_alignment(records)
        AlignIO.write(msa, str(output_fasta), "fasta")
        emit_log("Python fallback alignment complete.")
        return "python_fallback"

    if method != "auto":
        raise RuntimeError(f"Unsupported alignment method: {alignment_method}")

    if mafft_exe:
        emit_log("Alignment auto mode selected MAFFT.")
        run_mafft(input_fasta, output_fasta, mafft_exe=mafft_exe, accurate=accurate)
        emit_log("MAFFT alignment complete.")
        return "mafft"

    if muscle_exe:
        emit_log("Alignment auto mode selected MUSCLE.")
        muscle_mode = run_muscle(input_fasta, output_fasta, muscle_exe=muscle_exe, sequence_count=len(records))
        emit_log(f"MUSCLE alignment complete (mode={muscle_mode}).")
        return "muscle"


    if source_species and gene_symbol:
        try:
            emit_log("Alignment auto mode is trying Ensembl reference-guided alignment.")
            msa = build_ensembl_reference_guided_alignment(
                source_species=source_species,
                gene_symbol=gene_symbol,
                ortholog_df=ortholog_df,
                target_species=target_species,
                filter_vertebrates=filter_vertebrates,
                one2one_only=True,
            )
            AlignIO.write(msa, str(output_fasta), "fasta")
            emit_log("Ensembl reference-guided alignment complete.")
            return "ensembl_pairwise_merge"
        except Exception as exc:
            emit_log(f"Ensembl alignment path failed, falling back to Python alignment: {exc}")
            pass

    emit_log("Alignment auto mode is falling back to the built-in Python aligner.")
    msa = fallback_alignment(records)
    AlignIO.write(msa, str(output_fasta), "fasta")
    emit_log("Python fallback alignment complete.")
    return "python_fallback"


def run_iqtree(alignment_fasta: Path, outdir: Path, iqtree_exe: str, bootstrap: int) -> Path:
    prefix = outdir / "phylo"
    # V11: clean stale IQ-TREE artifacts from any prior run in this dir. Without
    # this, IQ-TREE aborts with "Checkpoint indicates a previous run finished;
    # use -redo" when re-running into an existing output directory.
    for stale in outdir.glob("phylo.*"):
        try:
            stale.unlink()
        except OSError:
            pass
    emit_log(f"Running IQ-TREE with {bootstrap} bootstrap replicates.")
    # V11: pin thread count instead of "-nt AUTO". On this 8-core box the AUTO
    # probe mis-detected a single thread as fastest after one glitched 2-thread
    # trial (654s vs 8.7s), forcing ModelFinder to grind single-threaded for
    # hours. A fixed -nt 6 bypasses the probe and uses the cores (2 left free
    # for the OS / parent process).
    cmd = [
        iqtree_exe,
        "-s", str(alignment_fasta),
        "-m", "MFP",
        "-bb", str(bootstrap),
        "-alrt", str(bootstrap),
        "-nt", "6",
        "--prefix", str(prefix),
        "-redo",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"IQ-TREE failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    treefile = outdir / "phylo.treefile"
    if not treefile.exists():
        raise FileNotFoundError("IQ-TREE finished but phylo.treefile was not created.")
    emit_log(f"IQ-TREE completed successfully: {treefile.name}")
    return treefile


def _drop_all_gap_columns(alignment: MultipleSeqAlignment) -> MultipleSeqAlignment:
    keep_indices = [
        idx for idx in range(alignment.get_alignment_length())
        if any(str(record.seq[idx]) not in GAP_CHARS for record in alignment)
    ]
    if not keep_indices:
        raise RuntimeError("Aligned sequences contained only gap columns, so no tree could be built.")

    trimmed_records: List[SeqRecord] = []
    for record in alignment:
        trimmed_seq = "".join(str(record.seq[idx]) for idx in keep_indices)
        trimmed_records.append(SeqRecord(Seq(trimmed_seq), id=record.id, description=""))
    return MultipleSeqAlignment(trimmed_records)


def _sanitize_tree_alignment(alignment: MultipleSeqAlignment) -> MultipleSeqAlignment:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ-*?")
    cleaned_records: List[SeqRecord] = []

    for record in alignment:
        seq = str(record.seq).upper().replace(".", "-")
        cleaned = "".join(ch if ch in allowed else "X" for ch in seq)
        if not cleaned.replace("-", "").replace("?", "").strip():
            raise RuntimeError(f"Sequence {record.id} is empty after gap cleanup and cannot be used for tree building.")
        cleaned_records.append(SeqRecord(Seq(cleaned), id=record.id, description=""))

    return _drop_all_gap_columns(MultipleSeqAlignment(cleaned_records))


def _clamp_negative_branch_lengths(tree: Any) -> None:
    for clade in tree.find_clades():
        if clade.branch_length is not None and clade.branch_length < 0:
            clade.branch_length = 0.0


def run_python_nj_tree(alignment_fasta: Path, outdir: Path) -> Path:
    emit_log("Building phylogeny with the built-in Python neighbor-joining method.")
    alignment = AlignIO.read(str(alignment_fasta), "fasta")
    cleaned_alignment = _sanitize_tree_alignment(alignment)

    constructor = DistanceTreeConstructor()
    errors: List[str] = []
    for model in TREE_DISTANCE_MODELS:
        try:
            emit_log(f"Trying Python NJ distance model: {model}.")
            calculator = DistanceCalculator(model)
            distance_matrix = calculator.get_distance(cleaned_alignment)
            tree = constructor.nj(distance_matrix)
            tree.rooted = False
            _clamp_negative_branch_lengths(tree)
            treefile = outdir / "phylo.treefile"
            Phylo.write(tree, str(treefile), "newick")
            if not treefile.exists():
                raise FileNotFoundError("Python NJ tree finished but phylo.treefile was not created.")
            emit_log(f"Python NJ phylogeny complete using the {model} distance model.")
            return treefile
        except Exception as exc:
            errors.append(f"{model}: {exc}")

    raise RuntimeError(
        "Built-in Python NJ phylogeny failed with all distance models tried:\n" + "\n".join(errors)
    )


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


def ensembl_gene_tree_by_symbol(species: str, symbol: str) -> Dict[str, Any]:
    url = f"{ENSEMBL}/genetree/member/symbol/{species}/{symbol}"
    return get_json(url, params={"sequence": "none"})


def normalize_species_key_from_scientific(name: Optional[str]) -> str:
    text = str(name or "").strip().lower().replace(" ", "_")
    return text


def humanize_taxonomy_label(value: Optional[str]) -> str:
    text = str(value or "").strip().replace("_", " ")
    if not text:
        return ""
    return text[:1].upper() + text[1:]


def extract_uniprot_gene_names(record: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for gene in record.get("genes", []) or []:
        gene_name = gene.get("geneName") or {}
        if gene_name.get("value"):
            names.append(str(gene_name["value"]))
        for synonym in gene.get("synonyms", []) or []:
            if synonym.get("value"):
                names.append(str(synonym["value"]))
    return [name for name in dict.fromkeys(names) if name]


def extract_uniprot_protein_name(record: Dict[str, Any]) -> Optional[str]:
    protein_desc = record.get("proteinDescription")
    if isinstance(protein_desc, dict):
        recommended = protein_desc.get("recommendedName") or {}
        full_name = recommended.get("fullName") or {}
        if full_name.get("value"):
            return str(full_name["value"])
        submission_names = protein_desc.get("submissionNames") or []
        for item in submission_names:
            full_name = item.get("fullName") or {}
            if full_name.get("value"):
                return str(full_name["value"])
    protein_name = record.get("proteinName")
    if isinstance(protein_name, dict):
        value = protein_name.get("value")
        if value:
            return str(value)
    if isinstance(protein_name, str):
        return protein_name
    return None


def extract_uniprot_alt_protein_names(record: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    protein_desc = record.get("proteinDescription") or {}
    for item in protein_desc.get("alternativeNames", []) or []:
        full_name = item.get("fullName") or {}
        if full_name.get("value"):
            names.append(str(full_name["value"]))
    for item in protein_desc.get("submissionNames", []) or []:
        full_name = item.get("fullName") or {}
        if full_name.get("value"):
            names.append(str(full_name["value"]))
    return [name for name in dict.fromkeys(names) if name]


def extract_uniprot_comment_text(entry: Dict[str, Any], comment_type: str) -> List[str]:
    texts: List[str] = []
    for comment in entry.get("comments", []) or []:
        if str(comment.get("commentType") or "").strip().lower() != comment_type.strip().lower():
            continue
        if comment_type.strip().lower() == "subcellular location":
            for location in comment.get("subcellularLocations", []) or []:
                location_value = ((location.get("location") or {}).get("value"))
                if location_value:
                    texts.append(str(location_value))
        else:
            for text_block in comment.get("texts", []) or []:
                if text_block.get("value"):
                    texts.append(str(text_block["value"]))
    return [text for text in dict.fromkeys(texts) if text]


def extract_uniprot_keyword_values(entry: Dict[str, Any]) -> List[str]:
    keywords: List[str] = []
    for keyword in entry.get("keywords", []) or []:
        value = keyword.get("value")
        if value:
            keywords.append(str(value))
    return [value for value in dict.fromkeys(keywords) if value]


def uniprot_entry_reviewed(record: Dict[str, Any]) -> bool:
    entry_type = str(record.get("entryType") or "").strip().lower()
    return "reviewed" in entry_type and "unreviewed" not in entry_type


def uniprot_search_by_gene_species(symbol: str, species_name: str, size: int = 5) -> List[Dict[str, Any]]:
    query = f'(gene:{symbol}) AND (organism_name:"{species_name}")'
    fields = ",".join([
        "accession",
        "id",
        "gene_names",
        "organism_name",
        "protein_name",
        "xref_interpro",
    ])
    url = f"{UNIPROT}/uniprotkb/search"
    data = get_json(url, params={"query": query, "format": "json", "fields": fields, "size": size})
    return data.get("results", [])


def uniprot_fetch_entry(accession: str) -> Dict[str, Any]:
    url = f"{UNIPROT}/uniprotkb/{accession}.json"
    return get_json(url)


def uniprot_search_by_ensembl_protein(ensembl_protein_id: str,
                                      size: int = 3) -> List[Dict[str, Any]]:
    """V9.8c: look up UniProt entries cross-referenced to a given Ensembl
    protein ID. Used to backfill uniprot_accession on protein_metadata_df rows
    that Ensembl-side ortholog retrieval left blank, so the per-species
    AlphaFold fetcher can later download each species' own AF model."""
    query = f"xref:{ensembl_protein_id}"
    fields = ",".join([
        "accession",
        "id",
        "gene_names",
        "organism_name",
        "protein_name",
        "reviewed",
    ])
    url = f"{UNIPROT}/uniprotkb/search"
    data = get_json(url, params={"query": query, "format": "json", "fields": fields, "size": size})
    return data.get("results", [])


def enrich_protein_metadata_with_uniprot_accessions(protein_metadata_df: pd.DataFrame,
                                                     max_workers: int = 4) -> pd.DataFrame:
    """V9.8c: for rows missing uniprot_accession but holding a usable
    ensembl_protein_id, query UniProt's REST search API (xref:<ensembl_id>)
    and write back the best (reviewed-preferred) accession + the canonical
    alphafold_entry_id (AF-{accession}-F1). Non-fatal on per-row failures so
    a single 404 from UniProt does not abort the whole step."""
    if protein_metadata_df is None or protein_metadata_df.empty:
        return protein_metadata_df
    cols = set(protein_metadata_df.columns)
    if "uniprot_accession" not in cols or "ensembl_protein_id" not in cols:
        return protein_metadata_df

    targets: List[Tuple[Any, str]] = []
    for idx, row in protein_metadata_df.iterrows():
        accession_raw = row.get("uniprot_accession")
        try:
            is_blank = pd.isna(accession_raw)
        except Exception:
            is_blank = False
        accession_str = "" if is_blank else str(accession_raw or "").strip()
        if accession_str and accession_str.lower() not in {"nan", "none"}:
            continue
        ensembl_id_raw = row.get("ensembl_protein_id")
        try:
            ensembl_blank = pd.isna(ensembl_id_raw)
        except Exception:
            ensembl_blank = False
        ensembl_id = "" if ensembl_blank else str(ensembl_id_raw or "").strip()
        if not ensembl_id or ensembl_id.lower() in {"nan", "none"}:
            continue
        targets.append((idx, ensembl_id))

    if not targets:
        emit_log("UniProt accession enrichment: nothing to look up.")
        return protein_metadata_df

    emit_log(
        f"UniProt accession enrichment: looking up {len(targets)} Ensembl protein IDs "
        f"(thread pool size {max_workers})."
    )

    def _lookup(target: Tuple[Any, str]) -> Tuple[Any, Optional[str], Optional[str]]:
        row_idx, ensembl_id = target
        try:
            results = uniprot_search_by_ensembl_protein(ensembl_id, size=3)
        except Exception as exc:
            return (row_idx, None, str(exc)[:120])
        if not results:
            return (row_idx, None, None)
        reviewed = [r for r in results if uniprot_entry_reviewed(r)]
        chosen = (reviewed or results)[0]
        accession = str(chosen.get("primaryAccession") or "").strip()
        if not accession:
            return (row_idx, None, "no_primary_accession_in_result")
        return (row_idx, accession, None)

    new_accessions: Dict[Any, str] = {}
    not_found = 0
    errored = 0
    workers = max(1, min(max_workers, len(targets)))
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_target = {executor.submit(_lookup, t): t for t in targets}
        for future in as_completed(future_to_target):
            row_idx, accession, err = future.result()
            completed += 1
            if accession:
                new_accessions[row_idx] = accession
            elif err:
                errored += 1
            else:
                not_found += 1
            if completed % 25 == 0 or completed == len(targets):
                emit_log(
                    f"UniProt lookup: {completed}/{len(targets)} "
                    f"(found={len(new_accessions)}, not_found={not_found}, errored={errored})"
                )

    if not new_accessions:
        emit_log(
            f"UniProt accession enrichment: 0 new accessions found "
            f"(not_found={not_found}, errored={errored})."
        )
        return protein_metadata_df

    df_updated = protein_metadata_df.copy()
    for row_idx, accession in new_accessions.items():
        df_updated.at[row_idx, "uniprot_accession"] = accession
        if "alphafold_accession" in df_updated.columns:
            df_updated.at[row_idx, "alphafold_accession"] = accession
        if "alphafold_entry_id" in df_updated.columns:
            df_updated.at[row_idx, "alphafold_entry_id"] = f"AF-{accession}-F1"
    emit_log(
        f"UniProt accession enrichment: filled {len(new_accessions)} new accessions "
        f"({not_found} ensembl IDs had no UniProt match, {errored} errored)."
    )
    return df_updated


def choose_best_uniprot_search_result(results: Sequence[Dict[str, Any]],
                                      symbol: str,
                                      species_name: str) -> Optional[Dict[str, Any]]:
    if not results:
        return None
    symbol_norm = normalize_text_token(symbol)
    species_norm = normalize_text_token(species_name)

    def score_result(result: Dict[str, Any]) -> Tuple[int, int, int, int, int, str]:
        genes = extract_uniprot_gene_names(result)
        gene_norms = {normalize_text_token(name) for name in genes}
        organism = result.get("organism", {}) or {}
        scientific_name = organism.get("scientificName") or organism.get("scientific_name") or organism.get("organismName")
        scientific_norm = normalize_text_token(scientific_name)
        protein_name = normalize_text_token(extract_uniprot_protein_name(result))
        accession = str(result.get("primaryAccession") or "")
        return (
            1 if symbol_norm and symbol_norm in gene_norms else 0,
            1 if species_norm and species_norm == scientific_norm else 0,
            1 if uniprot_entry_reviewed(result) else 0,
            1 if symbol_norm and symbol_norm in protein_name else 0,
            int(result.get("annotationScore") or 0),
            accession,
        )

    return sorted(results, key=score_result, reverse=True)[0]


def parse_interpro_ids(entry: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") == "InterPro" and xref.get("id"):
            ids.append(xref["id"])
    return sorted(set(ids))


def classify_uniprot_xref_category(database: Optional[str]) -> str:
    text = str(database or "").strip().lower()
    if text == "interpro":
        return "interpro"
    if text == "pfam":
        return "pfam"
    if text in {"go", "gene ontology"}:
        return "go"
    if text == "alphafolddb":
        return "alphafold"
    if text == "pdb":
        return "pdb"
    if text in {"refseq", "embl", "ensembl", "ensemblgenomes"}:
        return "reference"
    if text in {"reactome", "kegg", "biocyc", "wikipathways"}:
        return "pathway"
    if text in {"smart", "prosite", "gene3d", "cdd", "superfamily", "prints"}:
        return "ontology"
    if "structure" in text:
        return "structure"
    return "reference"


def parse_uniprot_feature_rows(entry: Dict[str, Any],
                               protein_record_id: str,
                               species: str,
                               symbol: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    accession = entry.get("primaryAccession")
    for feature in entry.get("features", []) or []:
        feature_type = clean_serializable(feature.get("type"))
        location = feature.get("location") or {}
        start = (location.get("start") or {}).get("value")
        end = (location.get("end") or {}).get("value")
        rows.append({
            "protein_record_id": protein_record_id,
            "species": species,
            "symbol": symbol,
            "uniprot_accession": accession,
            "source_database": "UniProt",
            "feature_type": feature_type,
            "description": clean_serializable(feature.get("description")),
            "start": clean_serializable(start),
            "end": clean_serializable(end),
            "source_feature_id": clean_serializable(feature.get("featureId")),
        })
    return rows


def parse_uniprot_xref_rows(entry: Dict[str, Any],
                            protein_record_id: str,
                            species: str,
                            symbol: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    accession = entry.get("primaryAccession")
    for xref in entry.get("uniProtKBCrossReferences", []) or []:
        database = clean_serializable(xref.get("database"))
        external_id = clean_serializable(xref.get("id"))
        if not database or not external_id:
            continue
        props = xref.get("properties") or []
        label_parts = []
        for prop in props:
            key = clean_serializable(prop.get("key"))
            value = clean_serializable(prop.get("value"))
            if key and value:
                label_parts.append(f"{key}: {value}")
        rows.append({
            "protein_record_id": protein_record_id,
            "species": species,
            "symbol": symbol,
            "uniprot_accession": accession,
            "database": database,
            "external_id": external_id,
            "label": "; ".join(label_parts) or None,
            "category": classify_uniprot_xref_category(database),
        })
    return rows


def extract_ensembl_tree_context(source_species: str,
                                 gene_symbol: str,
                                 valid_species: Sequence[str],
                                 tree_source_mode: str) -> Tuple[str, Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    valid_species_set = {str(species).strip() for species in valid_species if str(species).strip()}
    if tree_source_mode == "local":
        return "local", {}, []

    try:
        data = ensembl_gene_tree_by_symbol(source_species, gene_symbol)
    except Exception as exc:
        if tree_source_mode == "ensembl":
            raise RuntimeError(f"Ensembl tree nomenclature fetch failed for {source_species}/{gene_symbol}: {exc}") from exc
        emit_log(f"Tree nomenclature fallback: Ensembl gene tree fetch failed for {source_species}/{gene_symbol}: {exc}")
        return "local", {}, []

    root = data.get("tree") or {}
    leaf_by_species: Dict[str, Dict[str, Any]] = {}
    groups_by_signature: Dict[str, Dict[str, Any]] = {}
    total_valid = len(valid_species_set)

    def walk(node: Dict[str, Any]) -> List[str]:
        taxonomy = node.get("taxonomy") or {}
        scientific_name = clean_serializable(taxonomy.get("scientific_name"))
        species_key = normalize_species_key_from_scientific(scientific_name)
        common_name = clean_serializable(taxonomy.get("common_name"))
        timetree_mya = clean_serializable(taxonomy.get("timetree_mya"))
        taxonomy_id = clean_serializable(taxonomy.get("id"))
        children = node.get("children") or []
        if not children:
            if species_key and species_key in valid_species_set:
                leaf_by_species[species_key] = {
                    "species": species_key,
                    "scientific_name": scientific_name,
                    "common_name": common_name,
                    "taxonomy_id": taxonomy_id,
                    "timetree_mya": timetree_mya,
                    "ensembl_member_id": clean_serializable(((node.get("id") or {}).get("accession"))),
                }
                return [species_key]
            return []

        child_species: List[str] = []
        for child in children:
            child_species.extend(walk(child))
        child_species = sorted(set(child_species))
        if len(child_species) > 1 and len(child_species) < max(total_valid, 2):
            label = common_name or scientific_name
            signature = "|".join(child_species)
            if label and signature not in groups_by_signature:
                groups_by_signature[signature] = {
                    "signature": signature,
                    "species": child_species,
                    "label": humanize_taxonomy_label(label),
                    "event_type": clean_serializable((node.get("events") or {}).get("type")),
                    "timetree_mya": timetree_mya,
                    "homolog_count": len(child_species),
                    "source": "ensembl_tree",
                }
        return child_species

    walk(root if isinstance(root, dict) else {})
    return "ensembl", leaf_by_species, list(groups_by_signature.values())


def build_local_tree_groups(protein_metadata_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if protein_metadata_df is None or protein_metadata_df.empty:
        return []
    rows: List[Dict[str, Any]] = []
    total_species = len({str(species) for species in protein_metadata_df["species"].dropna().tolist()})
    seen_signatures: set[str] = set()
    for field_name in ("taxonomy_level", "broad_clade", "phylum", "clade"):
        if field_name not in protein_metadata_df.columns:
            continue
        for value, sub in protein_metadata_df.groupby(field_name, dropna=True):
            species_list = sorted({str(species) for species in sub["species"].dropna().tolist() if str(species).strip()})
            if len(species_list) <= 1 or len(species_list) >= max(total_species, 2):
                continue
            signature = "|".join(species_list)
            if not signature or signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            rows.append({
                "signature": signature,
                "species": species_list,
                "label": humanize_taxonomy_label(value),
                "event_type": None,
                "timetree_mya": None,
                "homolog_count": len(species_list),
                "source": f"local_{field_name}",
            })
    return rows


def build_tree_nomenclature_payload(protein_metadata_df: pd.DataFrame,
                                    source_species: str,
                                    gene_symbol: str,
                                    source_label: str,
                                    remote_groups: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    local_groups = build_local_tree_groups(protein_metadata_df)
    groups_by_signature: Dict[str, Dict[str, Any]] = {}
    for group in list(remote_groups) + local_groups:
        signature = str(group.get("signature") or "").strip()
        if not signature:
            continue
        if signature in groups_by_signature and groups_by_signature[signature].get("source") == "ensembl_tree":
            continue
        groups_by_signature.setdefault(signature, group)

    leaf_by_species: Dict[str, Dict[str, Any]] = {}
    leaf_by_protein_record_id: Dict[str, Dict[str, Any]] = {}
    if protein_metadata_df is not None and not protein_metadata_df.empty:
        for _, row in protein_metadata_df.iterrows():
            species = str(row.get("species") or "").strip()
            protein_record_id = str(row.get("protein_record_id") or "").strip()
            leaf_payload = {
                "species": species,
                "protein_record_id": protein_record_id,
                "preferred_label": clean_serializable(row.get("nomenclature_leaf_label")),
                "preferred_gene_label": clean_serializable(row.get("preferred_public_gene_label")),
                "preferred_protein_label": clean_serializable(row.get("preferred_protein_name")),
                "species_display_label": clean_serializable(row.get("species_display_label")),
                "common_name": clean_serializable(row.get("common_name")),
                "scientific_name": clean_serializable(row.get("scientific_name")),
                "uniprot_accession": clean_serializable(row.get("uniprot_accession")),
            }
            if species:
                leaf_by_species[species] = leaf_payload
            if protein_record_id:
                leaf_by_protein_record_id[protein_record_id] = leaf_payload

    event_types = sorted({
        str(group.get("event_type"))
        for group in groups_by_signature.values()
        if group.get("event_type")
    })
    return {
        "source": source_label,
        "query_species": source_species,
        "query_gene_symbol": gene_symbol,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "leaf_by_species": leaf_by_species,
        "leaf_by_protein_record_id": leaf_by_protein_record_id,
        "groups": sorted(groups_by_signature.values(), key=lambda item: (-int(item.get("homolog_count") or 0), str(item.get("label") or ""))),
        "event_types": event_types,
    }


def collect_protein_metadata_bundle(seq_df: pd.DataFrame,
                                    ortholog_df: pd.DataFrame,
                                    source_species: str,
                                    gene_symbol: str,
                                    reference_species: Optional[str],
                                    metadata_enrichment_mode: str = "auto",
                                    tree_nomenclature_source: str = "auto") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    metadata_columns = [
        "species", "protein_record_id", "sequence_header", "symbol", "ensembl_gene_id", "ensembl_protein_id",
        "translation_id", "orthology_type", "taxonomy_level", "clade", "phylum", "broad_clade",
        "is_query", "is_reference", "length_aa", "retrieval_status", "header_label_source",
        "metadata_object_id", "metadata_organism_name", "metadata_pub_gene_id", "metadata_description",
        "metadata_og_name", "metadata_raw_header", "scientific_name", "common_name", "species_display_label",
        "preferred_public_gene_label", "preferred_protein_name", "preferred_public_label", "nomenclature_leaf_label",
        "description", "biotype", "canonical_translation", "reviewed_status", "protein_existence",
        "uniprot_accession", "uniprot_entry_id", "gene_synonyms", "protein_alternative_names",
        "subcellular_locations", "keyword_list", "ec_numbers", "alphafold_accession", "alphafold_entry_id",
        "alphafold_model_url", "alphafold_source_label", "metadata_error",
    ]
    feature_columns = [
        "protein_record_id", "species", "symbol", "uniprot_accession", "source_database",
        "feature_type", "description", "start", "end", "source_feature_id",
    ]
    xref_columns = [
        "protein_record_id", "species", "symbol", "uniprot_accession", "database",
        "external_id", "label", "category",
    ]
    remote_allowed = metadata_enrichment_mode != "local"
    require_remote = metadata_enrichment_mode == "required"
    reference_species_name = reference_species or source_species
    ortholog_lookup = {
        str(row.get("species") or "").strip(): row
        for row in ortholog_df.to_dict(orient="records")
        if str(row.get("species") or "").strip()
    }
    valid_species = [
        str(species).strip()
        for species in seq_df["species"].dropna().tolist()
        if str(species).strip()
    ] if seq_df is not None and not seq_df.empty and "species" in seq_df.columns else []
    tree_source_label, tree_leaf_context, remote_groups = extract_ensembl_tree_context(
        source_species=source_species,
        gene_symbol=gene_symbol,
        valid_species=valid_species,
        tree_source_mode=tree_nomenclature_source,
    )

    metadata_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    xref_rows: List[Dict[str, Any]] = []
    total = len(seq_df.index) if seq_df is not None else 0

    for idx, row in enumerate(seq_df.to_dict(orient="records"), start=1):
        species = str(row.get("species") or "").strip()
        symbol = str(row.get("symbol") or gene_symbol).strip() or gene_symbol
        translation_id = clean_serializable(row.get("translation_id"))
        ensembl_gene_id = clean_serializable(row.get("ensembl_gene_id"))
        protein_record_id = str(row.get("protein_record_id") or protein_record_id_for(species, translation_id, ensembl_gene_id, symbol)).strip()
        ortholog_row = ortholog_lookup.get(species, {})
        taxonomy_level = clean_serializable(ortholog_row.get("taxonomy_level"))
        clade = classify_site_species_clade(species, taxonomy_level)
        phylum = classify_alignment_phylum(species, taxonomy_level)
        broad_clade = classify_alignment_broad_clade(clade, phylum, taxonomy_level)
        length_aa = clean_serializable(row.get("length_aa"))
        scientific_name = species_to_scientific_name(species)
        tree_leaf = tree_leaf_context.get(species, {})
        common_name = clean_serializable(tree_leaf.get("common_name"))
        species_display_label = species_to_display_label(species, common_name=common_name)
        preferred_gene_label = clean_serializable(row.get("metadata_pub_gene_id")) or symbol
        preferred_protein_name = clean_serializable(row.get("metadata_description")) or clean_serializable(row.get("metadata_og_name"))
        metadata_error = None
        uniprot_entry: Optional[Dict[str, Any]] = None
        emit_log(f"Protein metadata {idx}/{total}: resolving annotations for {species} ({symbol}).")

        if remote_allowed and str(row.get("status") or "") == "ok":
            try:
                species_query = species_to_scientific_name(species)
                search_results = uniprot_search_by_gene_species(symbol, species_query, size=5)
                best_result = choose_best_uniprot_search_result(search_results, symbol, species_query)
                if best_result and best_result.get("primaryAccession"):
                    uniprot_entry = uniprot_fetch_entry(str(best_result["primaryAccession"]))
                elif require_remote:
                    raise RuntimeError(f"No UniProt entry matched {species} ({symbol})")
            except Exception as exc:
                metadata_error = str(exc)
                if require_remote:
                    raise RuntimeError(f"Protein metadata enrichment failed for {species} ({symbol}): {exc}") from exc
                emit_log(f"Protein metadata {idx}/{total}: remote enrichment fallback for {species} ({symbol}) -> {exc}")

        if uniprot_entry:
            scientific_name = clean_serializable((uniprot_entry.get("organism") or {}).get("scientificName")) or scientific_name
            common_name = clean_serializable((uniprot_entry.get("organism") or {}).get("commonName")) or common_name
            species_display_label = species_to_display_label(species, common_name=common_name)
            preferred_gene_label = clean_serializable((extract_uniprot_gene_names(uniprot_entry) or [preferred_gene_label])[0]) or preferred_gene_label
            preferred_protein_name = clean_serializable(extract_uniprot_protein_name(uniprot_entry)) or preferred_protein_name
            feature_rows.extend(parse_uniprot_feature_rows(uniprot_entry, protein_record_id, species, symbol))
            xref_rows.extend(parse_uniprot_xref_rows(uniprot_entry, protein_record_id, species, symbol))
            time.sleep(0.05)

        protein_existence = clean_serializable((uniprot_entry or {}).get("proteinExistence"))
        if isinstance(protein_existence, dict):
            protein_existence = clean_serializable(protein_existence.get("value"))
        preferred_public_label = f"{preferred_gene_label}, {species_display_label}".strip(", ")
        metadata_rows.append({
            "species": species,
            "protein_record_id": protein_record_id,
            "sequence_header": clean_serializable(row.get("sequence_header")),
            "symbol": symbol,
            "ensembl_gene_id": ensembl_gene_id,
            "ensembl_protein_id": clean_serializable(ortholog_row.get("ensembl_protein_id")),
            "translation_id": translation_id,
            "orthology_type": clean_serializable(ortholog_row.get("orthology_type")),
            "taxonomy_level": taxonomy_level,
            "clade": clade,
            "phylum": phylum,
            "broad_clade": broad_clade,
            "is_query": bool(ortholog_row.get("is_query")) if pd.notna(ortholog_row.get("is_query")) else (species == source_species),
            "is_reference": species == reference_species_name,
            "length_aa": length_aa,
            "retrieval_status": clean_serializable(row.get("status")),
            "header_label_source": "fasta_object_metadata" if row.get("metadata_pub_gene_id") or row.get("metadata_description") or row.get("metadata_og_name") else "ensembl",
            "metadata_object_id": clean_serializable(row.get("metadata_object_id")),
            "metadata_organism_name": clean_serializable(row.get("metadata_organism_name")),
            "metadata_pub_gene_id": clean_serializable(row.get("metadata_pub_gene_id")),
            "metadata_description": clean_serializable(row.get("metadata_description")),
            "metadata_og_name": clean_serializable(row.get("metadata_og_name")),
            "metadata_raw_header": clean_serializable(row.get("metadata_raw_header")),
            "scientific_name": scientific_name,
            "common_name": common_name,
            "species_display_label": species_display_label,
            "preferred_public_gene_label": preferred_gene_label,
            "preferred_protein_name": preferred_protein_name,
            "preferred_public_label": preferred_public_label,
            "nomenclature_leaf_label": preferred_public_label,
            "description": preferred_protein_name or clean_serializable(row.get("metadata_description")),
            "biotype": clean_serializable((ortholog_row.get("biotype"))),
            "canonical_translation": translation_id,
            "reviewed_status": "reviewed" if uniprot_entry_reviewed(uniprot_entry or {}) else ("unreviewed" if uniprot_entry else None),
            "protein_existence": protein_existence,
            "uniprot_accession": clean_serializable((uniprot_entry or {}).get("primaryAccession")),
            "uniprot_entry_id": clean_serializable((uniprot_entry or {}).get("uniProtkbId")),
            "gene_synonyms": "; ".join(extract_uniprot_gene_names(uniprot_entry or {})[1:]) or None,
            "protein_alternative_names": "; ".join(extract_uniprot_alt_protein_names(uniprot_entry or {})) or None,
            "subcellular_locations": "; ".join(extract_uniprot_comment_text(uniprot_entry or {}, "Subcellular location")) or None,
            "keyword_list": "; ".join(extract_uniprot_keyword_values(uniprot_entry or {})) or None,
            "ec_numbers": "; ".join(extract_uniprot_comment_text(uniprot_entry or {}, "Catalytic activity")) or None,
            "alphafold_accession": None,
            "alphafold_entry_id": None,
            "alphafold_model_url": None,
            "alphafold_source_label": None,
            "metadata_error": metadata_error,
        })

    protein_metadata_df = pd.DataFrame(metadata_rows, columns=metadata_columns)
    protein_features_df = pd.DataFrame(feature_rows, columns=feature_columns).drop_duplicates() if feature_rows else pd.DataFrame(columns=feature_columns)
    protein_xrefs_df = pd.DataFrame(xref_rows, columns=xref_columns).drop_duplicates() if xref_rows else pd.DataFrame(columns=xref_columns)
    tree_payload = build_tree_nomenclature_payload(
        protein_metadata_df=protein_metadata_df,
        source_species=source_species,
        gene_symbol=gene_symbol,
        source_label=tree_source_label,
        remote_groups=remote_groups,
    )
    return protein_metadata_df, protein_features_df, protein_xrefs_df, tree_payload


def collect_domain_annotations(protein_metadata_df: pd.DataFrame,
                               protein_features_df: pd.DataFrame,
                               protein_xrefs_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "protein_record_id", "species", "symbol", "uniprot_accession", "interpro_ids",
        "feature_type", "description", "start", "end",
    ]
    if protein_metadata_df is None or protein_metadata_df.empty:
        return pd.DataFrame(columns=columns)

    interpro_map: Dict[str, str] = {}
    if protein_xrefs_df is not None and not protein_xrefs_df.empty:
        for protein_record_id, sub in protein_xrefs_df[protein_xrefs_df["category"] == "interpro"].groupby("protein_record_id"):
            ids = sorted({str(value) for value in sub["external_id"].dropna().tolist() if str(value).strip()})
            interpro_map[str(protein_record_id)] = ";".join(ids) if ids else None

    feature_df = protein_features_df.copy() if protein_features_df is not None else pd.DataFrame()
    if not feature_df.empty:
        feature_df = feature_df[feature_df["feature_type"].isin(sorted(UNIPROT_DOMAIN_FEATURE_TYPES))].copy()

    rows: List[Dict[str, Any]] = []
    for _, meta_row in protein_metadata_df.iterrows():
        protein_record_id = str(meta_row.get("protein_record_id") or "").strip()
        sub = feature_df[feature_df["protein_record_id"] == protein_record_id].copy() if not feature_df.empty else pd.DataFrame()
        if sub.empty:
            rows.append({
                "protein_record_id": protein_record_id,
                "species": clean_serializable(meta_row.get("species")),
                "symbol": clean_serializable(meta_row.get("symbol")),
                "uniprot_accession": clean_serializable(meta_row.get("uniprot_accession")),
                "interpro_ids": interpro_map.get(protein_record_id),
                "feature_type": None,
                "description": None,
                "start": None,
                "end": None,
            })
            continue
        for _, feat_row in sub.iterrows():
            rows.append({
                "protein_record_id": protein_record_id,
                "species": clean_serializable(meta_row.get("species")),
                "symbol": clean_serializable(meta_row.get("symbol")),
                "uniprot_accession": clean_serializable(meta_row.get("uniprot_accession")),
                "interpro_ids": interpro_map.get(protein_record_id),
                "feature_type": clean_serializable(feat_row.get("feature_type")),
                "description": clean_serializable(feat_row.get("description")),
                "start": clean_serializable(feat_row.get("start")),
                "end": clean_serializable(feat_row.get("end")),
            })
    return pd.DataFrame(rows, columns=columns)


def merge_sequence_retrieval_metadata(seq_df: pd.DataFrame,
                                      protein_metadata_df: pd.DataFrame) -> pd.DataFrame:
    if seq_df is None or seq_df.empty or protein_metadata_df is None or protein_metadata_df.empty:
        return seq_df.copy() if seq_df is not None else pd.DataFrame()
    merge_columns = [
        "protein_record_id",
        "preferred_public_label",
        "preferred_public_gene_label",
        "preferred_protein_name",
        "species_display_label",
        "scientific_name",
        "common_name",
        "uniprot_accession",
        "reviewed_status",
        "alphafold_entry_id",
        "alphafold_source_label",
    ]
    available = [column for column in merge_columns if column in protein_metadata_df.columns]
    meta_view = protein_metadata_df[available].drop_duplicates(subset=["protein_record_id"])
    return seq_df.merge(meta_view, on="protein_record_id", how="left")


def non_gap_residues(column: Iterable[str]) -> List[str]:
    return [aa.upper() for aa in column if aa not in GAP_CHARS and aa != "X" and aa != "?"]


def shannon_entropy(residues: Sequence[str]) -> float:
    if not residues:
        return float("nan")
    counts = Counter(residues)
    total = len(residues)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def residue_group_fraction(residues: Sequence[str], aa_group: set[str]) -> float:
    if not residues:
        return float("nan")
    return sum(1 for aa in residues if aa in aa_group) / len(residues)


def scheme_match_fraction(residues: Sequence[str], scheme_name: str) -> float:
    if not residues:
        return float("nan")
    scheme = AA_GROUP_SCHEMES[scheme_name]
    best = 0.0
    total = len(residues)
    for aa_set in scheme.values():
        frac = sum(1 for aa in residues if aa in aa_set) / total
        if frac > best:
            best = frac
    return best


def build_alignment_matrix(alignment: MultipleSeqAlignment) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    records = list(alignment)
    aln_length = alignment.get_alignment_length()
    ref_indices: Dict[str, int] = {record.id: 0 for record in records}

    for pos in range(aln_length):
        for record in records:
            aa = record.seq[pos]
            if aa not in GAP_CHARS:
                ref_indices[record.id] += 1
                seq_pos = ref_indices[record.id]
            else:
                seq_pos = None

            species, symbol = parse_header_species_symbol(record.id)
            rows.append({
                "alignment_position": pos + 1,
                "species": species,
                "symbol": symbol,
                "record_id": record.id,
                "label": parse_header_field(record.id, "Label"),
                "description": parse_header_field(record.id, "Description"),
                "residue": aa,
                "ungapped_position": seq_pos,
            })
    return pd.DataFrame(rows)


def compute_reference_mapping(reference_record: SeqRecord) -> List[Optional[int]]:
    mapping: List[Optional[int]] = []
    ref_pos = 0
    for aa in str(reference_record.seq):
        if aa not in GAP_CHARS:
            ref_pos += 1
            mapping.append(ref_pos)
        else:
            mapping.append(None)
    return mapping


def find_reference_record(alignment: MultipleSeqAlignment,
                          reference_species: Optional[str]) -> Tuple[Optional[SeqRecord], int]:
    records = list(alignment)
    if not records:
        return None, 0
    ref_idx = 0
    if reference_species:
        for idx, record in enumerate(records):
            species, _ = parse_header_species_symbol(record.id)
            if species == reference_species:
                ref_idx = idx
                break
    return records[ref_idx], ref_idx


def project_alignment_to_reference(alignment: MultipleSeqAlignment,
                                   reference_species: Optional[str] = None) -> Tuple[MultipleSeqAlignment, SeqRecord]:
    """
    Remove alignment columns where the chosen reference sequence has a gap.
    This preserves one column per reference residue, which is the right space for
    peptide / domain conservation mapped onto the reference protein.
    """
    records = list(alignment)
    if not records:
        raise RuntimeError("Alignment is empty.")

    ref_record = None
    if reference_species:
        for record in records:
            species, _ = parse_header_species_symbol(record.id)
            if species == reference_species:
                ref_record = record
                break
    if ref_record is None:
        ref_record = records[0]

    keep_indices = [idx for idx, aa in enumerate(str(ref_record.seq)) if aa not in GAP_CHARS]
    if not keep_indices:
        raise RuntimeError("Reference-projected alignment has no non-gap positions.")

    projected_records: List[SeqRecord] = []
    for record in records:
        projected_seq = "".join(str(record.seq[idx]) for idx in keep_indices)
        projected_records.append(SeqRecord(Seq(projected_seq), id=record.id, description=record.description))

    projected_alignment = MultipleSeqAlignment(projected_records)
    return projected_alignment, ref_record


def compute_conservation(alignment: MultipleSeqAlignment,
                         reference_species: Optional[str] = None) -> Tuple[pd.DataFrame, SeqRecord]:
    records = list(alignment)
    if not records:
        raise RuntimeError("Alignment is empty.")

    ref_record = None
    if reference_species:
        for record in records:
            species, _ = parse_header_species_symbol(record.id)
            if species == reference_species:
                ref_record = record
                break
    if ref_record is None:
        ref_record = records[0]

    ref_map = compute_reference_mapping(ref_record)
    aln_length = alignment.get_alignment_length()
    rows: List[Dict[str, Any]] = []

    for pos in range(aln_length):
        column = [record.seq[pos] for record in records]
        residues = non_gap_residues(column)
        occupancy = len(residues) / len(records) if records else float("nan")

        ref_residue = ref_record.seq[pos]
        counts = Counter(residues)
        most_common_residue, most_common_count = (None, 0)
        if counts:
            most_common_residue, most_common_count = counts.most_common(1)[0]
        identity_max = (most_common_count / len(residues)) if residues else float("nan")

        row = {
            "alignment_position": pos + 1,
            "reference_species": parse_header_species_symbol(ref_record.id)[0],
            "reference_symbol": parse_header_species_symbol(ref_record.id)[1],
            "reference_residue": ref_residue,
            "reference_ungapped_position": ref_map[pos],
            "n_sequences": len(records),
            "n_non_gap": len(residues),
            "occupancy": occupancy,
            "most_common_residue": most_common_residue,
            "identity_max": identity_max,
            "entropy": shannon_entropy(residues),
        }

        for group_name, aa_set in AA_GROUPS.items():
            row[f"{group_name}_fraction"] = residue_group_fraction(residues, aa_set)

        for scheme_name in AA_GROUP_SCHEMES:
            row[f"{scheme_name}_conservation"] = scheme_match_fraction(residues, scheme_name)

        if ref_residue not in GAP_CHARS and residues:
            row["reference_identity_fraction"] = sum(1 for aa in residues if aa == ref_residue) / len(residues)
            for group_name, aa_set in AA_GROUPS.items():
                row[f"reference_{group_name}_agreement"] = (
                    sum(1 for aa in residues if (aa in aa_set) == (ref_residue in aa_set)) / len(residues)
                )
        else:
            row["reference_identity_fraction"] = float("nan")
            for group_name in AA_GROUPS:
                row[f"reference_{group_name}_agreement"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows), ref_record


def smooth_series(values: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    out: List[float] = []
    half = window // 2
    for i in range(len(values)):
        left = max(0, i - half)
        right = min(len(values), i + half + 1)
        window_vals = [v for v in values[left:right] if pd.notna(v)]
        out.append(float(sum(window_vals) / len(window_vals)) if window_vals else float("nan"))
    return out


def detect_conserved_regions(conservation_df: pd.DataFrame,
                             score_column: str,
                             threshold: float,
                             min_len: int) -> pd.DataFrame:
    regions: List[Dict[str, Any]] = []
    in_region = False
    start_idx = None

    for i, val in enumerate(conservation_df[score_column].tolist()):
        good = pd.notna(val) and val >= threshold
        if good and not in_region:
            in_region = True
            start_idx = i
        elif not good and in_region:
            end_idx = i - 1
            if start_idx is not None and (end_idx - start_idx + 1) >= min_len:
                sub = conservation_df.iloc[start_idx:end_idx + 1]
                regions.append({
                    "score_column": score_column,
                    "start_alignment_position": int(sub["alignment_position"].iloc[0]),
                    "end_alignment_position": int(sub["alignment_position"].iloc[-1]),
                    "start_reference_position": sub["reference_ungapped_position"].iloc[0],
                    "end_reference_position": sub["reference_ungapped_position"].iloc[-1],
                    "length_alignment": int(end_idx - start_idx + 1),
                    "mean_score": float(sub[score_column].mean()),
                    "mean_occupancy": float(sub["occupancy"].mean()),
                })
            in_region = False
            start_idx = None

    if in_region and start_idx is not None:
        end_idx = len(conservation_df) - 1
        if (end_idx - start_idx + 1) >= min_len:
            sub = conservation_df.iloc[start_idx:end_idx + 1]
            regions.append({
                "score_column": score_column,
                "start_alignment_position": int(sub["alignment_position"].iloc[0]),
                "end_alignment_position": int(sub["alignment_position"].iloc[-1]),
                "start_reference_position": sub["reference_ungapped_position"].iloc[0],
                "end_reference_position": sub["reference_ungapped_position"].iloc[-1],
                "length_alignment": int(end_idx - start_idx + 1),
                "mean_score": float(sub[score_column].mean()),
                "mean_occupancy": float(sub["occupancy"].mean()),
            })

    return pd.DataFrame(regions)


def save_figure_svg_png(fig: plt.Figure, out_svg: Path) -> None:
    out_png = out_svg.with_suffix(".png")
    fig.savefig(out_svg, format="svg")
    fig.savefig(out_png, format="png", dpi=300)
    plt.close(fig)


def plot_tree_svg(treefile: Path,
                  out_svg: Path,
                  title: str,
                  clade_mya: Optional[Dict[str, float]] = None) -> None:
    tree = Phylo.read(str(treefile), "newick")
    fig = plt.figure(figsize=(12, max(6, len(tree.get_terminals()) * 0.35)))
    ax = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree, axes=ax, do_show=False)
    ax.set_title(title)
    guide = clade_mya or SITE_CLADE_MYA_DEFAULTS
    legend_items: List[str] = []
    for clade_name in ["tetrapods", "dipnoi", "actinistia", "holostei", "teleosts", "other_fish", "other_vertebrates"]:
        age = guide.get(clade_name)
        if age is None:
            continue
        try:
            age_value = float(age)
        except (TypeError, ValueError):
            continue
        if math.isnan(age_value) or math.isinf(age_value):
            continue
        legend_items.append(f"{clade_name.replace('_', ' ')} {age_value:.0f} Ma")
    if legend_items:
        legend_text = "Approx. divergence guide (~Ma)\n" + "\n".join(legend_items) + "\nBranch lengths are inferred protein distances."
        ax.text(
            0.995,
            0.01,
            legend_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#d0d7de", "boxstyle": "round,pad=0.35", "alpha": 0.92},
        )
    fig.tight_layout()
    save_figure_svg_png(fig, out_svg)


def normalize_pipeline_tree_record_key(record_id: str) -> str:
    text = str(record_id or "").strip().strip("'\"")
    for field_name in ("Gene", "EnsemblGene", "Protein", "ProteinRecordID"):
        text = text.replace(f"|{field_name}_", f"|{field_name}=")
    return text


def tree_species_signature(species_values: Sequence[str]) -> str:
    members = sorted({str(species).strip() for species in species_values if str(species).strip()})
    return "|".join(members)


def annotate_tree_with_nomenclature(tree: Any,
                                    tree_nomenclature: Optional[Dict[str, Any]]) -> None:
    payload = tree_nomenclature or {}
    leaf_by_species = payload.get("leaf_by_species") or {}
    leaf_by_record = payload.get("leaf_by_protein_record_id") or {}
    groups = payload.get("groups") or []
    group_by_signature = {
        str(group.get("signature") or ""): group
        for group in groups
        if str(group.get("signature") or "").strip()
    }

    def walk(clade: Any) -> List[str]:
        children = list(getattr(clade, "clades", []) or [])
        raw_name = str(getattr(clade, "name", "") or "")
        normalized_name = normalize_pipeline_tree_record_key(raw_name)
        species, _ = parse_header_species_symbol(normalized_name)
        protein_record_id = parse_header_field(normalized_name, "ProteinRecordID") or parse_header_field(raw_name, "ProteinRecordID")

        if not children:
            leaf_meta = (
                leaf_by_record.get(str(protein_record_id or "").strip())
                or leaf_by_species.get(species)
                or {}
            )
            setattr(clade, "_nomenclature_leaf_label", leaf_meta.get("preferred_label") or species)
            return [species] if species else []

        child_species: List[str] = []
        for child in children:
            child_species.extend(walk(child))
        signature = tree_species_signature(child_species)
        group = group_by_signature.get(signature) or {}
        if group:
            setattr(clade, "_nomenclature_group_label", group.get("label"))
            setattr(clade, "_nomenclature_event_type", group.get("event_type"))
            setattr(clade, "_nomenclature_homolog_count", group.get("homolog_count"))
        return sorted(set(child_species))

    walk(tree.root)


def plot_tree_nomenclature_svg(treefile: Path,
                               out_svg: Path,
                               title: str,
                               tree_nomenclature: Optional[Dict[str, Any]] = None,
                               clade_mya: Optional[Dict[str, float]] = None) -> None:
    tree = Phylo.read(str(treefile), "newick")
    annotate_tree_with_nomenclature(tree, tree_nomenclature)
    fig = plt.figure(figsize=(14, max(7, len(tree.get_terminals()) * 0.36)))
    ax = fig.add_subplot(1, 1, 1)

    def label_func(clade: Any) -> Optional[str]:
        children = list(getattr(clade, "clades", []) or [])
        if children:
            label = getattr(clade, "_nomenclature_group_label", None)
            if label:
                count = int(getattr(clade, "_nomenclature_homolog_count", 0) or 0)
                event_type = str(getattr(clade, "_nomenclature_event_type", "") or "").strip()
                event_suffix = f" [{event_type}]" if event_type else ""
                if count > 1:
                    return f"{label}: {count} homologs{event_suffix}"
                return f"{label}{event_suffix}"
            return None
        return getattr(clade, "_nomenclature_leaf_label", None) or (clade.name or None)

    Phylo.draw(tree, axes=ax, do_show=False, label_func=label_func)
    ax.set_title(title)
    guide = clade_mya or SITE_CLADE_MYA_DEFAULTS
    legend_items: List[str] = []
    for clade_name in ["tetrapods", "dipnoi", "actinistia", "holostei", "teleosts", "other_fish", "other_vertebrates"]:
        age = guide.get(clade_name)
        if age is None:
            continue
        try:
            age_value = float(age)
        except (TypeError, ValueError):
            continue
        if math.isnan(age_value) or math.isinf(age_value):
            continue
        legend_items.append(f"{clade_name.replace('_', ' ')} {age_value:.0f} Ma")
    event_types = sorted({
        str(group.get("event_type"))
        for group in (tree_nomenclature or {}).get("groups", []) or []
        if group.get("event_type")
    })
    if legend_items or event_types:
        legend_lines = []
        if legend_items:
            legend_lines.append("Approx. divergence guide (~Ma)")
            legend_lines.extend(legend_items)
        if event_types:
            legend_lines.append("")
            legend_lines.append("Mapped Ensembl node events")
            legend_lines.extend(str(event_type) for event_type in event_types)
        legend_lines.append("")
        legend_lines.append("Leaf labels prefer public gene naming; internal labels use mapped nomenclature groups when available.")
        ax.text(
            0.995,
            0.01,
            "\n".join(legend_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#d0d7de", "boxstyle": "round,pad=0.35", "alpha": 0.92},
        )
    fig.tight_layout()
    save_figure_svg_png(fig, out_svg)


def plot_conservation_svg(conservation_df: pd.DataFrame, out_svg: Path, title: str,
                          score_columns: Sequence[str]) -> None:
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(1, 1, 1)
    x = conservation_df["alignment_position"]

    for col in score_columns:
        ax.plot(x, conservation_df[col], label=col)

    ax.set_xlabel("Alignment position")
    ax.set_ylabel("Conservation score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save_figure_svg_png(fig, out_svg)


def plot_heatmap_svg(conservation_df: pd.DataFrame, out_svg: Path, title: str,
                     score_columns: Sequence[str]) -> None:
    data = conservation_df.loc[:, score_columns].T.to_numpy()
    fig = plt.figure(figsize=(18, max(4, 0.6 * len(score_columns))))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(data, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(score_columns)))
    ax.set_yticklabels(score_columns)
    ax.set_xlabel("Alignment position")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    save_figure_svg_png(fig, out_svg)


def plot_property_scan_svg(conservation_df: pd.DataFrame, out_svg: Path) -> None:
    columns = [
        "hydrophobicity_conservation",
        "charge_conservation",
        "polarity_conservation",
        "size_conservation",
        "aromaticity_conservation",
    ]
    plot_conservation_svg(
        conservation_df,
        out_svg,
        title="Biochemical property conservation across alignment",
        score_columns=columns,
    )


def plot_reference_conservation_scan_svg(conservation_df: pd.DataFrame, out_svg: Path,
                                         title: str) -> None:
    scan_df = conservation_df[conservation_df["reference_ungapped_position"].notna()].copy()
    if scan_df.empty:
        raise RuntimeError("Reference conservation scan could not be plotted because no reference positions were available.")

    x = scan_df["reference_ungapped_position"].astype(int)
    fig = plt.figure(figsize=(18, 7))
    ax = fig.add_subplot(1, 1, 1)

    for col in [
        "identity_max_smoothed",
        "reference_identity_fraction_smoothed",
        "hydrophobicity_conservation_smoothed",
        "charge_conservation_smoothed",
        "polarity_conservation_smoothed",
        "size_conservation_smoothed",
        "aromaticity_conservation_smoothed",
    ]:
        if col in scan_df.columns:
            ax.plot(x, scan_df[col], label=col.replace("_smoothed", ""))

    ax.set_xlabel("Reference residue position")
    ax.set_ylabel("Smoothed conservation score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    save_figure_svg_png(fig, out_svg)


def merge_domain_summary(conservation_df: pd.DataFrame, domain_df: pd.DataFrame,
                         reference_species: str, reference_symbol: str) -> pd.DataFrame:
    ref_domains = domain_df[
        (domain_df["species"] == reference_species) &
        (domain_df["symbol"] == reference_symbol) &
        (domain_df["start"].notna()) &
        (domain_df["end"].notna())
    ].copy()

    if ref_domains.empty:
        conservation_df = conservation_df.copy()
        conservation_df["reference_domain_labels"] = None
        return conservation_df

    domain_labels = []
    for _, row in conservation_df.iterrows():
        ref_pos = row["reference_ungapped_position"]
        if pd.isna(ref_pos):
            domain_labels.append(None)
            continue
        matches = ref_domains[(ref_domains["start"] <= ref_pos) & (ref_domains["end"] >= ref_pos)]
        if matches.empty:
            domain_labels.append(None)
        else:
            labels = []
            for _, m in matches.iterrows():
                parts = [str(m["feature_type"])] if pd.notna(m["feature_type"]) else []
                if pd.notna(m["description"]):
                    parts.append(str(m["description"]))
                if pd.notna(m["interpro_ids"]):
                    parts.append(str(m["interpro_ids"]))
                labels.append(" | ".join(parts))
            domain_labels.append("; ".join(labels))
    out = conservation_df.copy()
    out["reference_domain_labels"] = domain_labels
    return out




def short_record_label(record_id: str, max_len: int = 26) -> str:
    species, symbol = parse_header_species_symbol(record_id)
    species_short = species.replace("_", " ")
    preferred = parse_header_field(record_id, "Label") or parse_header_field(record_id, "Description")
    if preferred:
        preferred = preferred.replace('_', ' ')
        label = f"{species_short} | {preferred}"
    else:
        label = f"{species_short} ({symbol})" if symbol and symbol != "unknown" else species_short
    return label if len(label) <= max_len else label[: max_len - 1] + "…"


def blosum_positive(a: str, b: str) -> bool:
    a = a.upper()
    b = b.upper()
    if a in GAP_CHARS or b in GAP_CHARS or a in {"X", "?"} or b in {"X", "?"}:
        return False
    try:
        return float(BLOSUM62[(a, b)]) > 0
    except Exception:
        try:
            return float(BLOSUM62[(b, a)]) > 0
        except Exception:
            return False


def clustal_consensus_char(column: Sequence[str], ref_index: int) -> str:
    ref_aa = str(column[ref_index]).upper()
    if ref_aa in GAP_CHARS or ref_aa in {"X", "?"}:
        return " "
    others = [str(aa).upper() for i, aa in enumerate(column) if i != ref_index]
    comparable = [aa for aa in others if aa not in GAP_CHARS and aa not in {"X", "?"}]
    if not comparable:
        return " "
    if all(aa == ref_aa for aa in comparable):
        return "*"
    if all(blosum_positive(ref_aa, aa) for aa in comparable):
        return ":"
    if any(blosum_positive(ref_aa, aa) for aa in comparable):
        return "."
    return " "


def write_pretty_alignment_text(alignment: MultipleSeqAlignment, out_path: Path,
                                reference_species: Optional[str] = None,
                                residues_per_line: int = 120) -> None:
    records = list(alignment)
    if not records:
        return
    ref_idx = 0
    if reference_species:
        for i, record in enumerate(records):
            species, _ = parse_header_species_symbol(record.id)
            if species == reference_species:
                ref_idx = i
                break
    width = max(18, min(32, max(len(short_record_label(r.id, 32)) for r in records)))
    seq_len = alignment.get_alignment_length()
    with out_path.open("w", encoding="utf-8") as handle:
        for start in range(0, seq_len, residues_per_line):
            end = min(seq_len, start + residues_per_line)
            header_ticks = []
            for pos in range(start + 1, end + 1, 10):
                header_ticks.append(f"{pos:<10}")
            handle.write(" " * (width + 4) + "".join(header_ticks).rstrip() + "\n")
            for record in records:
                label = short_record_label(record.id, width).ljust(width)
                seg = str(record.seq[start:end])
                handle.write(f"{label}  {seg}\n")
            column_chars = [clustal_consensus_char([r.seq[pos] for r in records], ref_idx) for pos in range(start, end)]
            handle.write(f"{'Human-ref'.ljust(width)}  {''.join(column_chars)}\n\n")


def plot_alignment_blocks_pdf(alignment: MultipleSeqAlignment, out_pdf: Path,
                              reference_species: Optional[str] = None,
                              residues_per_line: int = 120,
                              blocks_per_page: int = 4) -> None:
    records = list(alignment)
    if not records:
        return
    ref_idx = 0
    if reference_species:
        for i, record in enumerate(records):
            species, _ = parse_header_species_symbol(record.id)
            if species == reference_species:
                ref_idx = i
                break
    seq_len = alignment.get_alignment_length()
    n_blocks = math.ceil(seq_len / residues_per_line)
    label_width = max(18, min(30, max(len(short_record_label(r.id, 30)) for r in records)))
    pages = math.ceil(n_blocks / blocks_per_page)
    with PdfPages(out_pdf) as pdf:
        for page in range(pages):
            start_block = page * blocks_per_page
            end_block = min(n_blocks, start_block + blocks_per_page)
            blocks_on_page = end_block - start_block
            fig_h = 1.4 + blocks_on_page * (len(records) * 0.22 + 1.4)
            fig, axes = plt.subplots(blocks_on_page, 1, figsize=(16, max(4, fig_h)))
            if blocks_on_page == 1:
                axes = [axes]
            for ax, block_index in zip(axes, range(start_block, end_block)):
                start = block_index * residues_per_line
                end = min(seq_len, start + residues_per_line)
                block_len = end - start
                n_rows = len(records) + 1
                ax.set_xlim(-label_width - 1, block_len)
                ax.set_ylim(n_rows + 0.5, -1.4)
                ax.axis("off")
                # position ruler
                for pos in range(0, block_len, 10):
                    abs_pos = start + pos + 1
                    ax.text(pos, -0.6, str(abs_pos), fontsize=7, family='monospace', va='center')
                # rows
                for row_i, record in enumerate(records):
                    y = row_i
                    ax.text(-label_width, y + 0.5, short_record_label(record.id, label_width),
                            fontsize=7.5, family='monospace', va='center')
                    segment = str(record.seq[start:end])
                    for col_i, aa in enumerate(segment):
                        color = AA_COLORS.get(aa.upper(), '#F0F0F0')
                        ax.add_patch(Rectangle((col_i, y), 1, 1, facecolor=color, edgecolor='0.82', linewidth=0.3))
                        ax.text(col_i + 0.5, y + 0.53, aa, ha='center', va='center', fontsize=5.7, family='monospace')
                # consensus row against human ref
                y = len(records)
                ax.text(-label_width, y + 0.5, 'Human-ref identity', fontsize=7.5, family='monospace', va='center')
                for col_i in range(block_len):
                    column = [r.seq[start + col_i] for r in records]
                    ch = clustal_consensus_char(column, ref_idx)
                    bg = '#FFFFFF' if ch == ' ' else '#D9D9D9'
                    ax.add_patch(Rectangle((col_i, y), 1, 1, facecolor=bg, edgecolor='0.85', linewidth=0.3))
                    ax.text(col_i + 0.5, y + 0.53, ch, ha='center', va='center', fontsize=6.2, family='monospace')
                ax.text(block_len + 0.5, y + 0.5, f'{start+1}-{end}', fontsize=7, va='center')
            fig.suptitle(f'Reference-projected protein alignment blocks (page {page + 1} of {pages})', fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            pdf.savefig(fig)
            fig.savefig(out_pdf.with_name(f"{out_pdf.stem}_page_{page+1}.png"), dpi=220, bbox_inches='tight')
            plt.close(fig)


def plot_reference_architecture_svg(conservation_df: pd.DataFrame, domain_df: pd.DataFrame,
                                    conserved_regions: pd.DataFrame, ref_record: SeqRecord,
                                    out_svg: Path, title: str,
                                    identity_threshold: float = 0.85,
                                    annotated_sites_df: Optional[pd.DataFrame] = None) -> None:
    ref_species, ref_symbol = parse_header_species_symbol(ref_record.id)
    ref_len = sum(1 for aa in str(ref_record.seq) if aa not in GAP_CHARS)
    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.2, 2.0], hspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)

    for ax in (ax0, ax1, ax2):
        ax.set_xlim(1, ref_len)

    # backbone
    ax0.plot([1, ref_len], [0.5, 0.5], color='black', linewidth=2)
    ref_domains = domain_df[(domain_df['species'] == ref_species) & (domain_df['symbol'] == ref_symbol) &
                            domain_df['start'].notna() & domain_df['end'].notna()].copy()
    colors = ['#89CFF0', '#FFCC99', '#C7CEEA', '#B5EAD7', '#FFB7B2', '#E2F0CB']
    occupied_levels = []
    for idx, (_, row) in enumerate(ref_domains.sort_values(['start', 'end']).iterrows()):
        start = int(row['start'])
        end = int(row['end'])
        level = 0
        while level < len(occupied_levels) and start <= occupied_levels[level]:
            level += 1
        if level == len(occupied_levels):
            occupied_levels.append(0)
        occupied_levels[level] = end
        y = 0.2 + level * 0.42
        ax0.add_patch(Rectangle((start, y), end - start + 1, 0.25,
                                facecolor=colors[idx % len(colors)], edgecolor='0.3'))
        label = str(row['description']) if pd.notna(row['description']) else str(row['feature_type'])
        ax0.text((start + end) / 2, y + 0.125, label[:30], ha='center', va='center', fontsize=7)
    ax0.set_ylim(0, max(1.0, 0.8 + 0.42 * max(1, len(occupied_levels))))
    ax0.set_ylabel('Domains')
    ax0.set_title(title)

    # conserved windows
    if conserved_regions is not None and not conserved_regions.empty:
        y_map = {}
        y_idx = 0
        for _, row in conserved_regions.sort_values(['score_column', 'start_reference_position']).iterrows():
            start = row.get('start_reference_position')
            end = row.get('end_reference_position')
            if pd.isna(start) or pd.isna(end):
                continue
            score_col = str(row['score_column'])
            if score_col not in y_map:
                y_map[score_col] = y_idx
                y_idx += 1
            y = y_map[score_col]
            ax1.add_patch(Rectangle((int(start), y - 0.35), int(end) - int(start) + 1, 0.7, alpha=0.65))
        ax1.set_yticks(list(y_map.values()))
        ax1.set_yticklabels(list(y_map.keys()), fontsize=7)
        ax1.set_ylim(-0.8, max(0.8, len(y_map)-0.2))
    else:
        ax1.text(ref_len/2, 0, 'No conserved windows passed thresholds', ha='center', va='center')
        ax1.set_yticks([])
        ax1.set_ylim(-1, 1)
    ax1.set_ylabel('Regions')

    # conservation tracks + high sites
    x = conservation_df['reference_ungapped_position'].astype(int)
    ax2.plot(x, conservation_df['identity_max_smoothed'], label='Identity max', linewidth=1.5)
    ax2.plot(x, conservation_df['reference_identity_fraction_smoothed'], label='Identity to human', linewidth=1.5)
    ax2.plot(x, conservation_df['hydrophobicity_conservation_smoothed'], label='Hydrophobicity', linewidth=1.2)
    high = conservation_df[conservation_df['reference_identity_fraction'] >= identity_threshold]
    if not high.empty:
        ax2.vlines(high['reference_ungapped_position'].astype(int), ymin=0, ymax=high['reference_identity_fraction'], alpha=0.18, linewidth=0.8)
        top_sites = high.sort_values(['reference_identity_fraction', 'identity_max'], ascending=False).head(25)
        for _, row in top_sites.iterrows():
            ax2.text(int(row['reference_ungapped_position']), min(1.03, float(row['reference_identity_fraction']) + 0.04),
                     str(int(row['reference_ungapped_position'])), fontsize=6, rotation=90, ha='center', va='bottom')
    if annotated_sites_df is not None and not annotated_sites_df.empty:
        for idx, (_, row) in enumerate(annotated_sites_df.sort_values('position').iterrows()):
            pos = int(row['position'])
            ax0.axvline(pos, color='crimson', alpha=0.35, linewidth=0.9, linestyle='--')
            ax2.axvline(pos, color='crimson', alpha=0.25, linewidth=0.8, linestyle='--')
            label = str(row.get('label', pos))[:28]
            y_text = 0.92 - (idx % 8) * 0.08
            ax2.text(pos, y_text, label, fontsize=6, rotation=90, ha='center', va='top', color='crimson')
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel('Score')
    ax2.set_xlabel('Human-reference residue position')
    ax2.legend(fontsize=8, ncol=3, loc='upper right')
    fig.tight_layout()
    save_figure_svg_png(fig, out_svg)




SITE_CLADE_MYA_DEFAULTS: Dict[str, float] = {
    "tetrapods": 420.0,
    "dipnoi": 430.0,
    "actinistia": 450.0,
    "teleosts": 320.0,
    "holostei": 320.0,
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


def parse_site_clade_mya(site_clade_mya_text: Optional[str]) -> Dict[str, float]:
    values = dict(SITE_CLADE_MYA_DEFAULTS)
    if not site_clade_mya_text:
        return values
    text = str(site_clade_mya_text).strip()
    if not text:
        return values
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key, val in parsed.items():
                try:
                    values[str(key).strip().lower()] = float(val)
                except Exception:
                    continue
            return values
    except Exception:
        pass
    for chunk in text.split(","):
        if ":" not in chunk:
            continue
        key, val = chunk.split(":", 1)
        key = key.strip().lower()
        try:
            values[key] = float(val.strip())
        except Exception:
            continue
    return values


def build_species_taxonomy_lookup(ortholog_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    if ortholog_df is None or ortholog_df.empty or "species" not in ortholog_df.columns:
        return lookup
    for _, row in ortholog_df.iterrows():
        species = str(row.get("species") or "").strip()
        if not species or species in lookup:
            continue
        taxonomy_level = row.get("taxonomy_level")
        lookup[species] = str(taxonomy_level).strip() if pd.notna(taxonomy_level) else ""
    return lookup


def classify_site_species_clade(species: str, taxonomy_level: Optional[str] = None) -> str:
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


def _consensus_string(strings: Sequence[str]) -> str:
    if not strings:
        return ""
    max_len = max(len(s) for s in strings)
    out = []
    for i in range(max_len):
        chars = [s[i] for s in strings if i < len(s) and s[i] not in GAP_CHARS and s[i] not in {"X", "?"}]
        if not chars:
            out.append("-")
            continue
        counts = Counter(chars)
        aa, n = counts.most_common(1)[0]
        out.append(aa if n / len(chars) >= 0.5 else "X")
    return "".join(out)


def compute_site_comparison_table(alignment: MultipleSeqAlignment,
                                  annotated_sites_df: pd.DataFrame,
                                  reference_species: Optional[str],
                                  site_window: int = 3,
                                  site_clade_mya_text: Optional[str] = None,
                                  taxonomy_lookup: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    if annotated_sites_df is None or annotated_sites_df.empty:
        return pd.DataFrame()
    records = list(alignment)
    if not records:
        return pd.DataFrame()
    ref_idx = 0
    if reference_species:
        for i, record in enumerate(records):
            species, _ = parse_header_species_symbol(record.id)
            if species == reference_species:
                ref_idx = i
                break
    ref_record = records[ref_idx]
    clade_mya = parse_site_clade_mya(site_clade_mya_text)
    rows: List[Dict[str, Any]] = []
    aln_len = alignment.get_alignment_length()
    site_window = max(0, int(site_window))
    for _, site_row in annotated_sites_df.iterrows():
        pos = int(site_row["position"])
        if pos < 1 or pos > aln_len:
            continue
        start = max(1, pos - site_window)
        end = min(aln_len, pos + site_window)
        ref_motif = str(ref_record.seq[start - 1:end]).upper()
        ref_residue = str(ref_record.seq[pos - 1]).upper()
        by_clade: Dict[str, List[Tuple[str, str, str]]] = {}
        for record in records:
            species, symbol = parse_header_species_symbol(record.id)
            clade = classify_site_species_clade(species, (taxonomy_lookup or {}).get(species))
            motif = str(record.seq[start - 1:end]).upper()
            residue = str(record.seq[pos - 1]).upper()
            by_clade.setdefault(clade, []).append((species, residue, motif))
        for clade in ["tetrapods", "dipnoi", "actinistia", "teleosts", "holostei", "other_fish", "other_vertebrates"]:
            items = by_clade.get(clade, [])
            if not items:
                continue
            residues = [r for _, r, _ in items if r not in GAP_CHARS and r not in {"X", "?"}]
            motifs = [m for _, _, m in items]
            consensus = _consensus_string(motifs)
            residue_counts = Counter(residues)
            major_residue, major_count = (None, 0)
            if residue_counts:
                major_residue, major_count = residue_counts.most_common(1)[0]
            identical_to_ref = sum(1 for r in residues if r == ref_residue)
            motif_identical = sum(1 for m in motifs if m == ref_motif)
            rows.append({
                "site_position": pos,
                "site_label": site_row.get("label", str(pos)),
                "site_window_start": start,
                "site_window_end": end,
                "reference_species": parse_header_species_symbol(ref_record.id)[0],
                "reference_residue": ref_residue,
                "reference_motif": ref_motif,
                "clade": clade,
                "mya_from_human_lineage": clade_mya.get(clade, float("nan")),
                "n_species": len(items),
                "n_non_gap_at_site": len(residues),
                "major_residue": major_residue,
                "major_residue_fraction": (major_count / len(residues)) if residues else float("nan"),
                "reference_residue_fraction": (identical_to_ref / len(residues)) if residues else float("nan"),
                "consensus_motif": consensus,
                "reference_motif_fraction": (motif_identical / len(motifs)) if motifs else float("nan"),
                "classification": "C" if residues and identical_to_ref == len(residues) else "N",
                "species_residues": "; ".join(f"{sp}:{res}" for sp, res, _ in items),
                "species_motifs": "; ".join(f"{sp}:{motif}" for sp, _, motif in items),
            })
    return pd.DataFrame(rows)


def plot_site_comparison_svg(site_df: pd.DataFrame, out_svg: Path, title: str) -> None:
    if site_df is None or site_df.empty:
        return
    sites = list(site_df["site_position"].drop_duplicates())
    clade_order = ["tetrapods", "dipnoi", "actinistia", "teleosts", "holostei", "other_fish", "other_vertebrates"]
    n_sites = len(sites)
    fig_w = max(12, 6.0 * n_sites)
    fig_h = 5.5
    fig, axes = plt.subplots(1, n_sites, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]
    for ax, site in zip(axes, sites):
        sub = site_df[site_df["site_position"] == site].copy()
        present = [c for c in clade_order if c in set(sub["clade"])]
        y_map = {c: i for i, c in enumerate(present)}
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.8, max(0.8, len(present) - 0.2))
        ax.invert_yaxis()
        ax.axis("off")
        label = str(sub["site_label"].iloc[0])
        ref = str(sub["reference_motif"].iloc[0])
        ax.text(0.02, -0.55, f"Site {site}: {label}", fontsize=10, fontweight="bold", va="center")
        ax.text(0.02, -0.25, f"Human/ref motif: {ref}", fontsize=8, va="center")
        if len(present) > 1:
            ax.plot([0.24, 0.24], [0, len(present)-1], color="black", linewidth=1.2)
        for _, row in sub.iterrows():
            clade = row["clade"]
            y = y_map.get(clade)
            if y is None:
                continue
            ax.plot([0.24, 0.36], [y, y], color="black", linewidth=1.2)
            frac = row.get("reference_residue_fraction")
            try:
                frac_f = float(frac)
            except Exception:
                frac_f = float("nan")
            marker = "C" if str(row.get("classification")) == "C" else "N"
            mya = row.get("mya_from_human_lineage")
            mya_txt = ""
            if pd.notna(mya):
                mya_txt = f"~{float(mya):.0f} Ma"
            motif = str(row.get("consensus_motif") or "")
            residue = str(row.get("major_residue") or "-")
            kd = frac_f if pd.notna(frac_f) else 0.0
            ax.text(0.02, y, mya_txt, fontsize=8, va="center")
            ax.text(0.38, y - 0.14, clade.replace("_", " ").title(), fontsize=8, va="center")
            ax.text(0.58, y - 0.14, motif, fontsize=8.5, color="green", va="center", family="monospace")
            ax.text(0.58, y + 0.12, f"major {residue}; ref-frac {kd:.2f}; n={int(row['n_species'])}", fontsize=7.2, va="center")
            ax.text(0.93, y, marker, fontsize=12, fontweight="bold", va="center", ha="center")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure_svg_png(fig, out_svg)


# -----------------------------------------------------------------------------
# Length QC + clade/Fourier conservation analysis helpers (V9.6)
# -----------------------------------------------------------------------------

DEFAULT_LENGTH_FILTER_KEEP_SPECIES: Tuple[str, ...] = (
    # V9.8c: Ensembl PLA2G4A model for lamprey (Petromyzon marinus) is annotated
    # at 302 aa, a truncated gene model that fails the default 30% length window.
    # Whitelist by default so the jawless-vertebrate clade keeps both cyclostomes.
    "petromyzon_marinus",
)


# V12: gene-gated per-species sequence-source overrides. Ensembl's canonical
# translation for some species is a long isoform that the +/-30 aa reference
# length filter then rejects (e.g. Bos taurus DHRS7 ENSBTAP00000086519 = 373 aa
# vs the 339 aa human reference). Swapping in a UniProt isoform of the SAME gene
# keeps the species in the analysis and gives it a real, reference-length row.
# Keyed by upper-cased gene symbol; each entry is additionally gated by the
# record's Ensembl protein id so it never fires for the wrong gene/assembly.
V12_SEQUENCE_SOURCE_OVERRIDES: Dict[str, List[Dict[str, Any]]] = {
    "DHRS7": [
        {
            "species": "bos_taurus",
            "match_protein_ids": ["ENSBTAP00000086519"],
            "uniprot_accession": "Q24K14",
            "alphafold_entry_id": "AF-Q24K14-F1",
            "note": "Ensembl ENSBTAP00000086519 (373 aa) -> UniProt Q24K14 (339 aa) Bos taurus DHRS7 isoform.",
        },
    ],
}


def apply_v12_sequence_source_overrides(records: List[SeqRecord],
                                        seq_df: pd.DataFrame,
                                        gene_symbol: Optional[str]) -> Tuple[List[SeqRecord], pd.DataFrame]:
    """V12: replace a species' Ensembl translation with a UniProt isoform of the
    same gene so it survives the reference-length filter and carries a real
    structure. Gated by gene symbol AND the record's Ensembl protein id so it
    never fires for the wrong gene/assembly. Non-fatal: on any fetch failure the
    original Ensembl sequence is kept."""
    overrides = V12_SEQUENCE_SOURCE_OVERRIDES.get(str(gene_symbol or "").strip().upper())
    if not overrides or not records:
        return records, seq_df
    if seq_df is not None and not seq_df.empty:
        seq_df = seq_df.copy()
    for override in overrides:
        target_species = str(override.get("species") or "").strip().lower()
        match_ids = [str(p).strip() for p in (override.get("match_protein_ids") or []) if str(p).strip()]
        accession = str(override.get("uniprot_accession") or "").strip()
        if not target_species or not accession:
            continue
        matched_record = None
        for record in records:
            species, _ = parse_header_species_symbol(record.id)
            if str(species).strip().lower() != target_species:
                continue
            if match_ids and not any(mid in str(record.id) for mid in match_ids):
                continue
            matched_record = record
            break
        if matched_record is None:
            emit_log(
                f"V12 sequence override [{gene_symbol}]: no {target_species} record matched "
                f"{match_ids or 'species'}; leaving Ensembl sequence in place."
            )
            continue
        try:
            entry = uniprot_fetch_entry(accession)
            new_seq = str(((entry or {}).get("sequence") or {}).get("value") or "").strip().upper()
        except Exception as exc:  # noqa: BLE001
            emit_log(
                f"V12 sequence override [{gene_symbol}]: UniProt fetch for {accession} failed "
                f"({exc}); keeping Ensembl sequence."
            )
            continue
        if not new_seq:
            emit_log(
                f"V12 sequence override [{gene_symbol}]: UniProt {accession} returned no sequence; "
                f"keeping Ensembl sequence."
            )
            continue
        old_len = len(str(matched_record.seq))
        matched_record.seq = Seq(new_seq)
        emit_log(
            f"V12 sequence override [{gene_symbol}]: {target_species} {old_len} aa -> "
            f"UniProt {accession} {len(new_seq)} aa. {override.get('note', '')}".strip()
        )
        if seq_df is not None and not seq_df.empty and "species" in seq_df.columns:
            mask = seq_df["species"].astype(str).str.strip().str.lower() == target_species
            if match_ids:
                id_cols = [c for c in ("protein_record_id", "translation_id", "sequence_header") if c in seq_df.columns]
                if id_cols:
                    id_mask = seq_df[id_cols].astype(str).apply(
                        lambda r: any(mid in " ".join(r.values) for mid in match_ids), axis=1
                    )
                    if (mask & id_mask).any():
                        mask = mask & id_mask
            if "length_aa" in seq_df.columns:
                seq_df.loc[mask, "length_aa"] = len(new_seq)
    return records, seq_df


def filter_records_by_reference_length(records: List[SeqRecord],
                                       seq_df: pd.DataFrame,
                                       reference_species: Optional[str],
                                       source_species: str,
                                       max_deviation: Optional[int],
                                       keep_species_whitelist: Optional[Sequence[str]] = None) -> Tuple[List[SeqRecord], pd.DataFrame]:
    """Reject protein sequences whose ungapped length differs too much from the reference.

    For PLA2G4A/cPLA2alpha this keeps proteins within 749 +/- 30 aa by default.
    The reference sequence itself is always kept.

    `keep_species_whitelist` is an optional iterable of species names that
    bypass the length filter (V9.8c addition). Useful for retaining truncated
    Ensembl gene models on lineages where the annotation is partial but the
    species is phylogenetically important (e.g. lamprey).
    """
    if not max_deviation or int(max_deviation) <= 0 or not records:
        report = seq_df.copy()
        report["length_filter_status"] = report.get("status", "unknown")
        report["reference_length_for_filter"] = None
        report["length_delta_from_reference"] = None
        return records, report

    ref_species = reference_species or source_species
    ref_record = None
    for record in records:
        species, _ = parse_header_species_symbol(record.id)
        if species == ref_species:
            ref_record = record
            break
    if ref_record is None:
        ref_record = records[0]
        ref_species = parse_header_species_symbol(ref_record.id)[0]

    ref_len = len(str(ref_record.seq).replace('-', '').replace('.', ''))
    max_dev = int(max_deviation)
    whitelist = {str(s).strip().lower() for s in (keep_species_whitelist or ()) if str(s).strip()}

    # Short-isoform auto-fallback: if Ensembl returned a reference that is
    # markedly shorter than the typical ortholog length (e.g. TP53 285 aa
    # vs the ~393 aa canonical), accept records that fall within max_dev of
    # the MEDIAN ortholog length as well. When the reference is at or above
    # the median (the normal case for canonical Ensembl picks, e.g. PLA2G4A
    # 749 aa) this branch never activates and the filter behaves identically
    # to the original absolute-window form.
    ortholog_lengths = [
        len(str(rec.seq).replace('-', '').replace('.', '')) for rec in records
    ]
    median_len = int(sorted(ortholog_lengths)[len(ortholog_lengths) // 2]) if ortholog_lengths else ref_len
    short_ref_threshold = 0.85
    use_median_window = ref_len < median_len * short_ref_threshold
    # When the reference is short, broaden to a single inclusive span from
    # (ref - max_dev) through (median + max_dev) so canonical-length orthologs
    # that fall between the two centers aren't dropped in the "gap".
    span_lower = ref_len - max_dev
    span_upper = (median_len if use_median_window else ref_len) + max_dev
    if use_median_window:
        emit_log(
            f"Length filter: reference length {ref_len} aa is well below the "
            f"median ortholog length {median_len} aa (ratio {ref_len/median_len:.2f} "
            f"< {short_ref_threshold}); engaging short-ref fallback so the kept "
            f"window spans [{span_lower}, {span_upper}] aa."
        )

    kept: List[SeqRecord] = []
    keep_species: set[str] = set()
    report_rows: List[Dict[str, Any]] = []

    for record in records:
        species, symbol = parse_header_species_symbol(record.id)
        length = len(str(record.seq).replace('-', '').replace('.', ''))
        delta = length - ref_len
        is_reference = species == ref_species
        is_whitelisted = species.lower() in whitelist
        within_ref_window = abs(delta) <= max_dev
        within_median_window = use_median_window and span_lower <= length <= span_upper
        keep = is_reference or is_whitelisted or within_ref_window or within_median_window
        if keep:
            kept.append(record)
            keep_species.add(species)
        if is_reference:
            status = "kept_reference"
        elif is_whitelisted and not within_ref_window and not within_median_window:
            status = "kept_length_whitelist"
        elif within_median_window and not within_ref_window:
            status = "kept_median_window"
        elif keep:
            status = "kept"
        else:
            status = "rejected_length_outlier"
        report_rows.append({
            "species": species,
            "symbol": symbol,
            "length_aa_from_record": length,
            "reference_species_for_filter": ref_species,
            "reference_length_for_filter": ref_len,
            "length_delta_from_reference": delta,
            "max_allowed_abs_delta": max_dev,
            "length_filter_status": status,
        })

    report = seq_df.copy()
    if not report.empty and "species" in report.columns:
        status_map = {r["species"]: r for r in report_rows}
        report["reference_length_for_filter"] = ref_len
        report["length_delta_from_reference"] = report["species"].map(lambda x: status_map.get(str(x), {}).get("length_delta_from_reference"))
        report["max_allowed_abs_delta"] = max_dev
        report["length_filter_status"] = report["species"].map(lambda x: status_map.get(str(x), {}).get("length_filter_status", "not_recovered"))
    extra = pd.DataFrame(report_rows)
    if report.empty:
        report = extra
    else:
        # The sequence table can contain duplicate species for paralog-like Ensembl rows; keep a complete record-level table appended as needed.
        report = pd.concat([report, extra.add_prefix("record_")], axis=1)
    return kept, report


def _same_property_as_reference(ref_aa: str, aa: str, scheme_name: str) -> Optional[bool]:
    if ref_aa in GAP_CHARS or aa in GAP_CHARS or ref_aa in {"X", "?"} or aa in {"X", "?"}:
        return None
    scheme = AA_GROUP_SCHEMES.get(scheme_name, {})
    for aa_set in scheme.values():
        if ref_aa in aa_set:
            return aa in aa_set
    return False


def parse_pipeline_tree_leaf_order(outdir: Path) -> Tuple[Dict[str, int], Dict[str, int], str]:
    for candidate in ("phylo.treefile", "phylo.contree", "phylo.bionj"):
        tree_path = outdir / candidate
        if not tree_path.exists():
            continue
        try:
            tree = Phylo.read(str(tree_path), "newick")
        except Exception:
            continue
        record_order: Dict[str, int] = {}
        species_order: Dict[str, int] = {}
        for order, terminal in enumerate(tree.get_terminals(), start=1):
            raw_name = str(terminal.name or "").strip()
            normalized_name = normalize_pipeline_tree_record_key(raw_name)
            species, _ = parse_header_species_symbol(normalized_name or raw_name)
            for key in {raw_name, normalized_name}:
                if key:
                    record_order.setdefault(key, order)
            if species:
                species_order.setdefault(species, order)
        return record_order, species_order, candidate
    return {}, {}, "clade_then_species"


def build_species_display_lookup(protein_metadata_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    if protein_metadata_df is None or protein_metadata_df.empty or "species" not in protein_metadata_df.columns:
        return lookup
    preferred_columns = [
        "preferred_public_label",
        "species_display_label",
        "common_name",
        "scientific_name",
        "preferred_public_gene_label",
    ]
    for _, row in protein_metadata_df.iterrows():
        species = str(row.get("species") or "").strip()
        if not species or species in lookup:
            continue
        for column in preferred_columns:
            value = str(row.get(column) or "").strip()
            if value:
                lookup[species] = value
                break
    return lookup


def segment_metric_fraction(reference_sequence: str,
                            target_sequence: str,
                            start: int,
                            end: int,
                            metric_name: str,
                            column_indices: Optional[Sequence[int]] = None) -> Tuple[float, int]:
    if column_indices is None:
        start_idx = max(0, int(start) - 1)
        end_idx = min(len(reference_sequence), int(end))
        iter_indices: Iterable[int] = range(start_idx, end_idx)
    else:
        max_len = min(len(reference_sequence), len(target_sequence))
        iter_indices = [idx for idx in column_indices if 0 <= int(idx) < max_len]
    matches = 0
    comparable = 0
    for idx in iter_indices:
        ref_aa = str(reference_sequence[idx]).upper()
        aa = str(target_sequence[idx]).upper()
        if ref_aa in GAP_CHARS or aa in GAP_CHARS or ref_aa in {"X", "?"} or aa in {"X", "?"}:
            continue
        comparable += 1
        if metric_name == "identity":
            matched = aa == ref_aa
        elif metric_name == "similarity":
            matched = blosum_positive(ref_aa, aa)
        elif metric_name == "polarity":
            matched = bool(_same_property_as_reference(ref_aa, aa, "polarity"))
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")
        if matched:
            matches += 1
    return ((matches / comparable) if comparable else float("nan")), comparable


def reference_window_column_indices(reference_mapping: Sequence[Optional[int]],
                                    start: int,
                                    end: int) -> List[int]:
    start = int(start)
    end = int(end)
    return [
        idx
        for idx, ref_pos in enumerate(reference_mapping)
        if ref_pos is not None and start <= int(ref_pos) <= end
    ]


def collapse_alignment_to_species_representatives(alignment: MultipleSeqAlignment,
                                                  reference_species: Optional[str]) -> Tuple[List[SeqRecord], int]:
    records = list(alignment)
    if not records:
        return [], 0
    ref_record, ref_idx = find_reference_record(alignment, reference_species)
    if ref_record is None:
        return [], 0
    reference_sequence = str(ref_record.seq)
    best_by_species: Dict[str, Dict[str, Any]] = {}
    for idx, record in enumerate(records):
        if idx == ref_idx:
            continue
        species, _ = parse_header_species_symbol(record.id)
        value, comparable = segment_metric_fraction(
            reference_sequence,
            str(record.seq),
            1,
            len(reference_sequence),
            "identity",
        )
        score = -1.0 if pd.isna(value) else float(value)
        current = best_by_species.get(species)
        candidate = {
            "record": record,
            "score": score,
            "comparable": comparable,
            "index": idx,
        }
        if current is None or candidate["score"] > current["score"] or (
            candidate["score"] == current["score"] and (
                candidate["comparable"] > current["comparable"] or (
                    candidate["comparable"] == current["comparable"] and candidate["index"] < current["index"]
                )
            )
        ):
            best_by_species[species] = candidate
    selected = [ref_record]
    selected.extend(best_by_species[species]["record"] for species in sorted(best_by_species))
    return selected, 0


def sort_evolutionary_records(alignment: MultipleSeqAlignment,
                              reference_species: Optional[str],
                              outdir: Path,
                              taxonomy_lookup: Optional[Dict[str, str]] = None,
                              protein_metadata_df: Optional[pd.DataFrame] = None) -> Tuple[List[SeqRecord], List[Dict[str, Any]], str]:
    representative_records, ref_idx = collapse_alignment_to_species_representatives(alignment, reference_species)
    if not representative_records:
        return [], [], "clade_then_species"

    record_order_map, species_order_map, tree_order_source = parse_pipeline_tree_leaf_order(outdir)
    display_lookup = build_species_display_lookup(protein_metadata_df)
    clade_rank_lookup = {clade: idx for idx, clade in enumerate(EVOLUTIONARY_FIRST_DIVERGENCE_CLADE_ORDER)}

    info_rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(representative_records):
        species, symbol = parse_header_species_symbol(record.id)
        normalized_id = normalize_pipeline_tree_record_key(record.id)
        clade = classify_site_species_clade(species, (taxonomy_lookup or {}).get(species))
        tree_order = species_order_map.get(species)
        if tree_order is None:
            tree_order = record_order_map.get(record.id)
        if tree_order is None:
            tree_order = record_order_map.get(normalized_id)
        display_label = display_lookup.get(species) or species.replace("_", " ")
        info_rows.append({
            "record": record,
            "species": species,
            "symbol": symbol,
            "clade": clade,
            "is_reference": idx == ref_idx,
            "tree_order": tree_order,
            "display_label": display_label,
            "clade_rank": clade_rank_lookup.get(clade, len(clade_rank_lookup) + 1),
        })

    def sort_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
        if item["is_reference"]:
            return (0, 0, 0, "", "")
        tree_order = item.get("tree_order")
        if tree_order is not None:
            return (1, 0, int(tree_order), item["clade_rank"], str(item["display_label"]).lower())
        return (1, 1, item["clade_rank"], str(item["display_label"]).lower(), str(item["species"]).lower())

    info_rows.sort(key=sort_key)
    ordered_records = [row["record"] for row in info_rows]
    ordered_info: List[Dict[str, Any]] = []
    for sort_order, row in enumerate(info_rows):
        row_copy = {key: value for key, value in row.items() if key != "record"}
        row_copy["sort_order"] = sort_order
        ordered_info.append(row_copy)
    return ordered_records, ordered_info, tree_order_source


def polarity_consensus_char(column: Sequence[str], ref_index: int) -> str:
    ref_aa = str(column[ref_index]).upper()
    if ref_aa in GAP_CHARS or ref_aa in {"X", "?"}:
        return " "
    comparable = [
        str(aa).upper()
        for idx, aa in enumerate(column)
        if idx != ref_index and str(aa).upper() not in GAP_CHARS and str(aa).upper() not in {"X", "?"}
    ]
    if not comparable:
        return " "
    agreements = [bool(_same_property_as_reference(ref_aa, aa, "polarity")) for aa in comparable]
    if all(agreements):
        return "|"
    if any(agreements):
        return "+"
    return " "


def clip_segment_bounds(start_value: Any, end_value: Any, reference_length: int) -> Tuple[Optional[int], Optional[int]]:
    try:
        start = int(float(start_value))
        end = int(float(end_value))
    except (TypeError, ValueError):
        return None, None
    start = max(1, start)
    end = min(reference_length, end)
    if start > end:
        return None, None
    return start, end


def format_evolutionary_segment_label(segment_source: str,
                                      feature_type: Optional[str],
                                      description: Optional[str],
                                      start: int,
                                      end: int,
                                      region_type: Optional[str] = None,
                                      clade: Optional[str] = None,
                                      site_position: Optional[int] = None) -> Tuple[str, str]:
    description_text = "" if pd.isna(description) else str(description).strip()
    feature_text = "" if pd.isna(feature_type) else str(feature_type).strip()
    if segment_source == "human_feature":
        base = description_text or feature_text or "Human feature"
    elif segment_source == "human_feature_window":
        if feature_text == "Binding site" and site_position is not None and not description_text:
            base = f"Binding site {site_position} motif"
        elif feature_text == "Modified residue" and site_position is not None:
            base = f"{description_text or 'Modified residue'} motif"
        elif feature_text == "Active site" and description_text:
            base = f"{description_text} motif"
        else:
            base = f"{description_text or feature_text or 'Feature'} motif"
    else:
        clade_text = str(clade or "").replace("_", " ").strip()
        region_text = str(region_type or "").replace("_", " ").strip()
        base = " ".join(part for part in [clade_text, region_text] if part) or "Clade window"
    plot_label = f"{base} [{start}-{end}]"
    return base, plot_label


def build_evolutionary_segment_catalog(domain_df: pd.DataFrame,
                                       protein_features_df: Optional[pd.DataFrame],
                                       reference_species: str,
                                       reference_symbol: str,
                                       reference_length: int,
                                       site_window: int,
                                       clade_fourier_regions_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    site_window = max(0, int(site_window))
    feature_source_df = protein_features_df if protein_features_df is not None and not protein_features_df.empty else domain_df
    ref_domains = feature_source_df[
        (feature_source_df["species"] == reference_species)
        & (feature_source_df["symbol"] == reference_symbol)
        & feature_source_df["start"].notna()
        & feature_source_df["end"].notna()
    ].copy()

    if not ref_domains.empty:
        curated = ref_domains[ref_domains["feature_type"].isin(EVOLUTIONARY_CURATED_FEATURE_TYPES)].copy()
        curated = curated.sort_values(["start", "end", "feature_type", "description"], kind="stable")
        for _, row in curated.iterrows():
            start, end = clip_segment_bounds(row.get("start"), row.get("end"), reference_length)
            if start is None or end is None:
                continue
            segment_label, plot_label = format_evolutionary_segment_label(
                segment_source="human_feature",
                feature_type=row.get("feature_type"),
                description=row.get("description"),
                start=start,
                end=end,
            )
            rows.append({
                "segment_source": "human_feature",
                "feature_type": row.get("feature_type"),
                "feature_description": row.get("description"),
                "region_type": None,
                "clade": None,
                "site_position": None,
                "start": start,
                "end": end,
                "length": end - start + 1,
                "segment_label": segment_label,
                "segment_plot_label": plot_label,
                "include_in_bar_plot": True,
                "export_alignment_window": True,
                "window_export_group": "curated_feature",
            })

        singleton = ref_domains[ref_domains["feature_type"].isin(EVOLUTIONARY_SINGLETON_WINDOW_FEATURE_TYPES)].copy()
        singleton = singleton.sort_values(["start", "feature_type", "description"], kind="stable")
        for _, row in singleton.iterrows():
            site_position, _ = clip_segment_bounds(row.get("start"), row.get("start"), reference_length)
            if site_position is None:
                continue
            start = max(1, site_position - site_window)
            end = min(reference_length, site_position + site_window)
            segment_label, plot_label = format_evolutionary_segment_label(
                segment_source="human_feature_window",
                feature_type=row.get("feature_type"),
                description=row.get("description"),
                start=start,
                end=end,
                site_position=site_position,
            )
            rows.append({
                "segment_source": "human_feature_window",
                "feature_type": row.get("feature_type"),
                "feature_description": row.get("description"),
                "region_type": "motif_window",
                "clade": None,
                "site_position": site_position,
                "start": start,
                "end": end,
                "length": end - start + 1,
                "segment_label": segment_label,
                "segment_plot_label": plot_label,
                "include_in_bar_plot": True,
                "export_alignment_window": True,
                "window_export_group": "curated_feature",
            })

    if clade_fourier_regions_df is not None and not clade_fourier_regions_df.empty:
        region_rows = clade_fourier_regions_df.sort_values(
            ["clade", "start_reference_position", "end_reference_position"],
            kind="stable",
        )
        for _, row in region_rows.iterrows():
            start, end = clip_segment_bounds(
                row.get("start_reference_position"),
                row.get("end_reference_position"),
                reference_length,
            )
            if start is None or end is None:
                continue
            segment_label, plot_label = format_evolutionary_segment_label(
                segment_source="clade_fourier_region",
                feature_type=None,
                description=None,
                start=start,
                end=end,
                region_type=row.get("region_type"),
                clade=row.get("clade"),
            )
            region_type = str(row.get("region_type") or "").strip()
            export_alignment_window = region_type == "clade_divergent_from_global"
            rows.append({
                "segment_source": "clade_fourier_region",
                "feature_type": None,
                "feature_description": None,
                "region_type": region_type,
                "clade": row.get("clade"),
                "site_position": None,
                "start": start,
                "end": end,
                "length": end - start + 1,
                "segment_label": segment_label,
                "segment_plot_label": plot_label,
                "include_in_bar_plot": False,
                "export_alignment_window": export_alignment_window,
                "window_export_group": "clade_divergent_window" if export_alignment_window else None,
            })

    if not rows:
        return pd.DataFrame()

    segment_df = pd.DataFrame(rows).drop_duplicates(
        subset=["segment_source", "feature_type", "feature_description", "region_type", "clade", "site_position", "start", "end"],
        keep="first",
    ).reset_index(drop=True)
    segment_df.insert(0, "segment_sort_order", range(1, len(segment_df) + 1))

    seen_ids: Dict[str, int] = {}
    segment_ids: List[str] = []
    for _, row in segment_df.iterrows():
        base = sanitize_filename(
            f"{row['segment_source']}_{row['segment_label']}_{row['start']}_{row['end']}"
        ).strip("_").lower() or "segment"
        count = seen_ids.get(base, 0) + 1
        seen_ids[base] = count
        segment_ids.append(base if count == 1 else f"{base}_{count}")
    segment_df.insert(1, "segment_id", segment_ids)
    return segment_df


def summarize_evolutionary_segments(alignment: MultipleSeqAlignment,
                                    outdir: Path,
                                    segment_df: pd.DataFrame,
                                    reference_species: str,
                                    identity_threshold: float,
                                    property_threshold: float,
                                    reference_mapping: Optional[Sequence[Optional[int]]] = None,
                                    alignment_scope: str = "aligned_reference_projected",
                                    scope_label: str = "Reference-projected",
                                    taxonomy_lookup: Optional[Dict[str, str]] = None,
                                    protein_metadata_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[SeqRecord], List[Dict[str, Any]], str]:
    ordered_records, ordered_info, tree_order_source = sort_evolutionary_records(
        alignment,
        reference_species=reference_species,
        outdir=outdir,
        taxonomy_lookup=taxonomy_lookup,
        protein_metadata_df=protein_metadata_df,
    )
    if not ordered_records or segment_df.empty:
        return segment_df.copy(), pd.DataFrame(), ordered_records, ordered_info, tree_order_source

    reference_record = ordered_records[0]
    reference_sequence = str(reference_record.seq)
    clade_appearance = [
        row["clade"]
        for row in ordered_info
        if not row["is_reference"] and str(row.get("clade") or "").strip()
    ]
    clade_sort_lookup = {clade: idx for idx, clade in enumerate(dict.fromkeys(clade_appearance))}
    fallback_clade_lookup = {clade: idx for idx, clade in enumerate(EVOLUTIONARY_FIRST_DIVERGENCE_CLADE_ORDER)}

    metric_thresholds = {
        "identity": float(identity_threshold),
        "similarity": float(property_threshold),
        "polarity": float(property_threshold),
    }
    metric_columns = {
        "identity": "identity_to_human_mean",
        "similarity": "similarity_to_human_mean",
        "polarity": "polarity_agreement_to_human_mean",
    }
    species_metric_rows: List[Dict[str, Any]] = []
    clade_metric_rows: List[Dict[str, Any]] = []
    segment_rows: List[Dict[str, Any]] = []
    segment_column_cache: Dict[Tuple[int, int], List[int]] = {}

    for _, segment in segment_df.iterrows():
        start = int(segment["start"])
        end = int(segment["end"])
        segment_columns = None
        if reference_mapping is not None:
            window_key = (start, end)
            if window_key not in segment_column_cache:
                segment_column_cache[window_key] = reference_window_column_indices(reference_mapping, start, end)
            segment_columns = segment_column_cache[window_key]
        segment_species_rows: List[Dict[str, Any]] = []
        for record, info in zip(ordered_records, ordered_info):
            if info["is_reference"]:
                continue
            row = {
                "segment_id": segment["segment_id"],
                "segment_label": segment["segment_label"],
                "segment_plot_label": segment["segment_plot_label"],
                "segment_source": segment["segment_source"],
                "feature_type": segment.get("feature_type"),
                "region_type": segment.get("region_type"),
                "segment_clade": segment.get("clade"),
                "start": start,
                "end": end,
                "length": int(segment["length"]),
                "species": info["species"],
                "species_display_label": info["display_label"],
                "clade": info["clade"],
                "sort_order": info["sort_order"],
                "alignment_scope": alignment_scope,
                "alignment_scope_label": scope_label,
            }
            for metric_name in ("identity", "similarity", "polarity"):
                value, comparable = segment_metric_fraction(
                    reference_sequence,
                    str(record.seq),
                    start,
                    end,
                    metric_name,
                    column_indices=segment_columns,
                )
                row[metric_columns[metric_name]] = value
                row[f"{metric_name}_informative_positions"] = comparable
            segment_species_rows.append(row)
            species_metric_rows.append(row.copy())

        species_df = pd.DataFrame(segment_species_rows)
        segment_out = segment.to_dict()
        segment_out["alignment_scope"] = alignment_scope
        segment_out["alignment_scope_label"] = scope_label
        for metric_name in ("identity", "similarity", "polarity"):
            threshold = metric_thresholds[metric_name]
            metric_column = metric_columns[metric_name]
            first_species = None
            if not species_df.empty:
                hit = species_df[
                    species_df[metric_column].notna()
                    & (species_df[metric_column] < threshold)
                ].sort_values(["sort_order", "species"], kind="stable")
                if not hit.empty:
                    first_species = str(hit.iloc[0]["species"])
            segment_out[f"first_divergent_species_{metric_name}"] = first_species

        present_clades = [
            clade
            for clade in dict.fromkeys(species_df["clade"].tolist())
            if str(clade or "").strip()
        ]
        for clade in present_clades:
            sub = species_df[species_df["clade"] == clade].copy()
            if sub.empty:
                continue
            clade_row = {
                "segment_id": segment["segment_id"],
                "segment_label": segment["segment_label"],
                "segment_plot_label": segment["segment_plot_label"],
                "segment_source": segment["segment_source"],
                "feature_type": segment.get("feature_type"),
                "region_type": segment.get("region_type"),
                "segment_clade": segment.get("clade"),
                "clade": clade,
                "start": start,
                "end": end,
                "length": int(segment["length"]),
                "species_count": int(len(sub)),
                "informative_species_count": int(sub["identity_to_human_mean"].notna().sum()),
                "clade_sort_order": clade_sort_lookup.get(
                    clade,
                    len(clade_sort_lookup) + fallback_clade_lookup.get(str(clade), len(fallback_clade_lookup) + 1),
                ),
                "alignment_scope": alignment_scope,
                "alignment_scope_label": scope_label,
            }
            for metric_name in ("identity", "similarity", "polarity"):
                metric_column = metric_columns[metric_name]
                threshold = metric_thresholds[metric_name]
                value = float(sub[metric_column].mean()) if sub[metric_column].notna().any() else float("nan")
                clade_row[metric_column] = value
                clade_row[f"{metric_name}_is_conserved"] = bool(pd.notna(value) and value >= threshold)
                clade_row[f"{metric_name}_is_diverged"] = bool(pd.notna(value) and value < threshold)
            clade_metric_rows.append(clade_row)

        segment_clade_df = pd.DataFrame(
            [row for row in clade_metric_rows if row["segment_id"] == segment["segment_id"]]
        ).sort_values(["clade_sort_order", "clade"], kind="stable")
        for metric_name in ("identity", "similarity", "polarity"):
            metric_column = metric_columns[metric_name]
            threshold = metric_thresholds[metric_name]
            first_clade = None
            if not segment_clade_df.empty:
                hit = segment_clade_df[
                    segment_clade_df[metric_column].notna()
                    & (segment_clade_df[metric_column] < threshold)
                ]
                if not hit.empty:
                    first_clade = str(hit.iloc[0]["clade"])
            segment_out[f"first_divergent_clade_{metric_name}"] = first_clade
        segment_out["tree_order_source"] = tree_order_source
        segment_rows.append(segment_out)

    segment_out_df = pd.DataFrame(segment_rows)
    clade_out_df = pd.DataFrame(clade_metric_rows)
    if not clade_out_df.empty:
        clade_out_df = clade_out_df.sort_values(
            ["segment_label", "clade_sort_order", "clade"],
            kind="stable",
        ).reset_index(drop=True)
    return segment_out_df, clade_out_df, ordered_records, ordered_info, tree_order_source


def write_evolutionary_alignment_window_text(records: Sequence[SeqRecord],
                                             record_info: Sequence[Dict[str, Any]],
                                             out_path: Path,
                                             segment_row: Dict[str, Any],
                                             tree_order_source: str,
                                             alignment_scope_label: str = "Reference-projected",
                                             residues_per_line: int = 80) -> None:
    if not records:
        return
    start = int(segment_row["start"])
    end = int(segment_row["end"])
    highlight_species = {
        str(segment_row.get("first_divergent_species_identity") or "").strip(),
        str(segment_row.get("first_divergent_species_similarity") or "").strip(),
        str(segment_row.get("first_divergent_species_polarity") or "").strip(),
    }
    highlight_species.discard("")
    ref_idx = 0
    labels: List[str] = []
    for record, info in zip(records, record_info):
        prefix = ">> " if (not info.get("is_reference") and info.get("species") in highlight_species) else "   "
        display = str(info.get("display_label") or short_record_label(record.id, 32)).replace("_", " ")
        labels.append(prefix + display)
    width = max(18, min(42, max(len(label) for label in labels + ["Human-ref", "Polarity"])))

    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Segment: {segment_row['segment_plot_label']}\n")
        handle.write(
            f"Source: {segment_row.get('segment_source')}"
            + (f"; feature_type={segment_row.get('feature_type')}" if segment_row.get("feature_type") else "")
            + (f"; region_type={segment_row.get('region_type')}" if segment_row.get("region_type") else "")
            + (f"; clade={segment_row.get('clade')}" if segment_row.get("clade") else "")
            + "\n"
        )
        handle.write(f"Reference window: {start}-{end} ({segment_row['length']} aa)\n")
        handle.write(f"Alignment basis: {alignment_scope_label}\n")
        handle.write(f"Species order: {tree_order_source}\n")
        handle.write(
            "First divergence (identity): "
            f"clade={segment_row.get('first_divergent_clade_identity') or 'none'}; "
            f"species={segment_row.get('first_divergent_species_identity') or 'none'}\n"
        )
        handle.write(
            "First divergence (similarity): "
            f"clade={segment_row.get('first_divergent_clade_similarity') or 'none'}; "
            f"species={segment_row.get('first_divergent_species_similarity') or 'none'}\n"
        )
        handle.write(
            "First divergence (polarity): "
            f"clade={segment_row.get('first_divergent_clade_polarity') or 'none'}; "
            f"species={segment_row.get('first_divergent_species_polarity') or 'none'}\n"
        )
        handle.write(
            "Legend: Human-ref uses CLUSTAL * : . relative to the human/reference residue. "
            "Polarity uses | when all informative residues share human polarity, + when polarity agreement is mixed.\n"
        )
        handle.write(
            f"Representative rule: one highest-identity record per species from the {alignment_scope_label.lower()} alignment.\n\n"
        )

        for block_start in range(start - 1, end, residues_per_line):
            block_end = min(end, block_start + residues_per_line)
            header_ticks = []
            for pos in range(block_start + 1, block_end + 1, 10):
                header_ticks.append(f"{pos:<10}")
            handle.write(" " * (width + 4) + "".join(header_ticks).rstrip() + "\n")
            for label, record in zip(labels, records):
                segment = str(record.seq[block_start:block_end])
                handle.write(f"{label.ljust(width)}  {segment}\n")
            clustal_chars = [
                clustal_consensus_char([record.seq[pos] for record in records], ref_idx)
                for pos in range(block_start, block_end)
            ]
            polarity_chars = [
                polarity_consensus_char([record.seq[pos] for record in records], ref_idx)
                for pos in range(block_start, block_end)
            ]
            handle.write(f"{'Human-ref'.ljust(width)}  {''.join(clustal_chars)}\n")
            handle.write(f"{'Polarity'.ljust(width)}  {''.join(polarity_chars)}\n\n")


def plot_evolutionary_segment_bars(segment_df: pd.DataFrame,
                                   clade_metrics_df: pd.DataFrame,
                                   out_svg: Path,
                                   identity_threshold: float,
                                   property_threshold: float,
                                   scope_label: str = "Reference-projected") -> None:
    if segment_df.empty or clade_metrics_df.empty:
        return
    curated = segment_df[segment_df["include_in_bar_plot"].astype(bool)].copy()
    if curated.empty:
        return
    plot_clades = [
        clade
        for clade in EVOLUTIONARY_CORE_CLADE_ORDER
        if clade in set(clade_metrics_df["clade"].dropna().tolist())
    ]
    if not plot_clades:
        return

    curated = curated.sort_values(["start", "end", "segment_label"], kind="stable").reset_index(drop=True)
    metric_specs = [
        ("identity_to_human_mean", float(identity_threshold), "Identity to human"),
        ("similarity_to_human_mean", float(property_threshold), "Similarity to human"),
        ("polarity_agreement_to_human_mean", float(property_threshold), "Polarity agreement"),
    ]
    fig_height = max(12.5, 0.56 * len(curated) + 4.8)
    fig, axes = plt.subplots(1, 3, figsize=(27.5, fig_height), sharey=True)
    y = np.arange(len(curated))
    total_height = 0.78
    bar_height = total_height / max(1, len(plot_clades))
    offsets = np.linspace(
        -total_height / 2 + bar_height / 2,
        total_height / 2 - bar_height / 2,
        num=len(plot_clades),
    )

    for ax, (metric_column, threshold, title) in zip(axes, metric_specs):
        for clade, offset in zip(plot_clades, offsets):
            values: List[float] = []
            failures: List[bool] = []
            for _, segment in curated.iterrows():
                sub = clade_metrics_df[
                    (clade_metrics_df["segment_id"] == segment["segment_id"])
                    & (clade_metrics_df["clade"] == clade)
                ]
                value = float(sub.iloc[0][metric_column]) if not sub.empty and pd.notna(sub.iloc[0][metric_column]) else 0.0
                values.append(value)
                failures.append((not sub.empty) and pd.notna(sub.iloc[0][metric_column]) and float(sub.iloc[0][metric_column]) < threshold)
            bars = ax.barh(
                y + offset,
                values,
                height=bar_height * 0.92,
                color=EVOLUTIONARY_CLADE_COLORS.get(clade, "#64748b"),
                alpha=0.86,
                edgecolor="#14213d",
                linewidth=0.4,
                label=clade.replace("_", " "),
            )
            for bar, failed in zip(bars, failures):
                if failed:
                    bar.set_hatch("///")
                    bar.set_edgecolor("#991b1b")
                    bar.set_linewidth(1.2)
        ax.axvline(threshold, color="#991b1b", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Mean fraction", fontsize=10)
        ax.set_title(title, fontsize=13)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="x", alpha=0.2, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(curated["segment_plot_label"].tolist(), fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Human domains, regions, and motif windows", fontsize=11)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)

    clade_handles = [
        Rectangle((0, 0), 1, 1, facecolor=EVOLUTIONARY_CLADE_COLORS.get(clade, "#64748b"), edgecolor="#14213d", linewidth=0.4)
        for clade in plot_clades
    ]
    clade_labels = [clade.replace("_", " ") for clade in plot_clades]
    clade_legend = fig.legend(
        clade_handles,
        clade_labels,
        ncol=min(3, max(1, len(clade_labels))),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        frameon=False,
        fontsize=9,
        columnspacing=1.3,
        handletextpad=0.5,
    )
    fig.add_artist(clade_legend)
    meaning_handles = [
        Line2D([0], [0], color="#991b1b", linestyle="--", linewidth=1.2),
        Rectangle((0, 0), 1, 1, facecolor="#ffffff", edgecolor="#991b1b", hatch="///", linewidth=1.2),
    ]
    meaning_labels = [
        "Dashed red line = divergence cutoff",
        "Hatched bar = clade mean below cutoff",
    ]
    fig.legend(
        meaning_handles,
        meaning_labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.914),
        frameon=False,
        fontsize=9,
        columnspacing=1.8,
        handlelength=2.4,
        handletextpad=0.7,
    )
    fig.suptitle(
        "Human-centric evolutionary divergence across domains, regions, and motif windows",
        fontsize=14,
        y=0.992,
    )
    fig.text(
        0.5,
        0.972,
        f"Alignment basis: {scope_label}. Earliest recovered lineage in tree order is used for first-divergence reporting.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#334155",
    )
    fig.tight_layout(rect=[0.03, 0.02, 0.995, 0.84])
    save_figure_svg_png(fig, out_svg)


def export_evolutionary_segment_analysis(alignment: MultipleSeqAlignment,
                                         outdir: Path,
                                         domain_df: pd.DataFrame,
                                         protein_features_df: Optional[pd.DataFrame],
                                         annotated_sites_df: pd.DataFrame,
                                         clade_fourier_regions_df: pd.DataFrame,
                                         reference_species: str,
                                         reference_symbol: str,
                                         identity_threshold: float,
                                         property_threshold: float,
                                         site_window: int,
                                         alignment_scope: str = "aligned_reference_projected",
                                         scope_label: str = "Reference-projected",
                                         taxonomy_lookup: Optional[Dict[str, str]] = None,
                                         protein_metadata_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_record, _ = find_reference_record(alignment, reference_species)
    reference_mapping = compute_reference_mapping(reference_record) if reference_record is not None else []
    reference_length = max((pos for pos in reference_mapping if pos is not None), default=alignment.get_alignment_length())
    segment_df = build_evolutionary_segment_catalog(
        domain_df=domain_df,
        protein_features_df=protein_features_df,
        reference_species=reference_species,
        reference_symbol=reference_symbol,
        reference_length=reference_length,
        site_window=site_window,
        clade_fourier_regions_df=clade_fourier_regions_df,
    )
    if segment_df.empty:
        empty_manifest = pd.DataFrame()
        segment_df.to_csv(outdir / "evolutionary_segments.csv", index=False)
        pd.DataFrame().to_csv(outdir / "evolutionary_segment_metrics_by_clade.csv", index=False)
        empty_manifest.to_csv(outdir / "evolutionary_alignment_windows_manifest.csv", index=False)
        return segment_df, pd.DataFrame(), empty_manifest

    segment_summary_df, clade_metrics_df, ordered_records, ordered_info, tree_order_source = summarize_evolutionary_segments(
        alignment=alignment,
        outdir=outdir,
        segment_df=segment_df,
        reference_species=reference_species,
        identity_threshold=identity_threshold,
        property_threshold=property_threshold,
        reference_mapping=reference_mapping,
        alignment_scope=alignment_scope,
        scope_label=scope_label,
        taxonomy_lookup=taxonomy_lookup,
        protein_metadata_df=protein_metadata_df,
    )
    segment_summary_df.to_csv(outdir / "evolutionary_segments.csv", index=False)
    clade_metrics_df.to_csv(outdir / "evolutionary_segment_metrics_by_clade.csv", index=False)

    window_dir = outdir / EVOLUTIONARY_ALIGNMENT_WINDOWS_DIRNAME
    window_dir.mkdir(parents=True, exist_ok=True)
    for existing_txt in window_dir.glob("*.txt"):
        existing_txt.unlink()

    manifest_rows: List[Dict[str, Any]] = []
    export_rows = segment_summary_df[segment_summary_df["export_alignment_window"].astype(bool)].copy()
    export_rows = export_rows.sort_values(["include_in_bar_plot", "start", "end", "segment_label"], ascending=[False, True, True, True], kind="stable")
    for _, row in export_rows.iterrows():
        file_name = f"{row['segment_id']}.txt"
        relative_path = f"{EVOLUTIONARY_ALIGNMENT_WINDOWS_DIRNAME}/{file_name}"
        write_evolutionary_alignment_window_text(
            records=ordered_records,
            record_info=ordered_info,
            out_path=window_dir / file_name,
            segment_row=row.to_dict(),
            tree_order_source=tree_order_source,
            alignment_scope_label=scope_label,
            residues_per_line=min(90, max(40, int(row["length"]))),
        )
        manifest_rows.append({
            "segment_id": row["segment_id"],
            "segment_label": row["segment_label"],
            "segment_source": row["segment_source"],
            "feature_type": row.get("feature_type"),
            "region_type": row.get("region_type"),
            "clade": row.get("clade"),
            "window_export_group": row.get("window_export_group"),
            "start": row["start"],
            "end": row["end"],
            "length": row["length"],
            "text_report_path": relative_path,
            "tree_order_source": tree_order_source,
            "first_divergent_clade_identity": row.get("first_divergent_clade_identity"),
            "first_divergent_species_identity": row.get("first_divergent_species_identity"),
            "first_divergent_clade_similarity": row.get("first_divergent_clade_similarity"),
            "first_divergent_species_similarity": row.get("first_divergent_species_similarity"),
            "first_divergent_clade_polarity": row.get("first_divergent_clade_polarity"),
            "first_divergent_species_polarity": row.get("first_divergent_species_polarity"),
        })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(outdir / "evolutionary_alignment_windows_manifest.csv", index=False)

    plot_evolutionary_segment_bars(
        segment_df=segment_summary_df,
        clade_metrics_df=clade_metrics_df,
        out_svg=outdir / "evolutionary_segment_bars.svg",
        identity_threshold=identity_threshold,
        property_threshold=property_threshold,
        scope_label=scope_label,
    )
    scoped_svg = outdir / f"evolutionary_segment_bars_{alignment_scope}.svg"
    if scoped_svg.name != "evolutionary_segment_bars.svg":
        plot_evolutionary_segment_bars(
            segment_df=segment_summary_df,
            clade_metrics_df=clade_metrics_df,
            out_svg=scoped_svg,
            identity_threshold=identity_threshold,
            property_threshold=property_threshold,
            scope_label=scope_label,
        )
    return segment_summary_df, clade_metrics_df, manifest_df


def export_evolutionary_scope_bar_figure(alignment: MultipleSeqAlignment,
                                         outdir: Path,
                                         domain_df: pd.DataFrame,
                                         protein_features_df: Optional[pd.DataFrame],
                                         clade_fourier_regions_df: pd.DataFrame,
                                         reference_species: str,
                                         reference_symbol: str,
                                         identity_threshold: float,
                                         property_threshold: float,
                                         site_window: int,
                                         alignment_scope: str,
                                         scope_label: str,
                                         taxonomy_lookup: Optional[Dict[str, str]] = None,
                                         protein_metadata_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reference_record, _ = find_reference_record(alignment, reference_species)
    if reference_record is None:
        return pd.DataFrame(), pd.DataFrame()
    reference_mapping = compute_reference_mapping(reference_record)
    reference_length = max((pos for pos in reference_mapping if pos is not None), default=0)
    if reference_length <= 0:
        return pd.DataFrame(), pd.DataFrame()

    segment_df = build_evolutionary_segment_catalog(
        domain_df=domain_df,
        protein_features_df=protein_features_df,
        reference_species=reference_species,
        reference_symbol=reference_symbol,
        reference_length=reference_length,
        site_window=site_window,
        clade_fourier_regions_df=clade_fourier_regions_df,
    )
    if segment_df.empty:
        return segment_df, pd.DataFrame()

    segment_summary_df, clade_metrics_df, _, _, _ = summarize_evolutionary_segments(
        alignment=alignment,
        outdir=outdir,
        segment_df=segment_df,
        reference_species=reference_species,
        identity_threshold=identity_threshold,
        property_threshold=property_threshold,
        reference_mapping=reference_mapping,
        alignment_scope=alignment_scope,
        scope_label=scope_label,
        taxonomy_lookup=taxonomy_lookup,
        protein_metadata_df=protein_metadata_df,
    )
    if clade_metrics_df.empty:
        return segment_summary_df, clade_metrics_df
    plot_evolutionary_segment_bars(
        segment_df=segment_summary_df,
        clade_metrics_df=clade_metrics_df,
        out_svg=outdir / f"evolutionary_segment_bars_{alignment_scope}.svg",
        identity_threshold=identity_threshold,
        property_threshold=property_threshold,
        scope_label=scope_label,
    )
    return segment_summary_df, clade_metrics_df


def _nan_interp(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    x = np.arange(arr.size)
    arr[~finite] = np.interp(x[~finite], x[finite], arr[finite])
    return arr


def _fft_lowpass(values: Sequence[float], keep_terms: int = 18) -> np.ndarray:
    arr = _nan_interp(values)
    if arr.size == 0:
        return arr
    keep_terms = max(1, min(int(keep_terms), arr.size // 2 if arr.size > 2 else 1))
    centered = arr - np.nanmean(arr)
    coeff = np.fft.rfft(centered)
    filt = np.zeros_like(coeff)
    filt[:keep_terms + 1] = coeff[:keep_terms + 1]
    recon = np.fft.irfft(filt, n=arr.size) + np.nanmean(arr)
    return np.clip(recon, 0.0, 1.0)


def _fft_spectrum(values: Sequence[float], clade: str) -> pd.DataFrame:
    arr = _nan_interp(values)
    if arr.size < 3:
        return pd.DataFrame()
    centered = arr - np.nanmean(arr)
    coeff = np.fft.rfft(centered)
    freq = np.fft.rfftfreq(arr.size, d=1.0)
    amp = np.abs(coeff) / arr.size
    rows = []
    for i in range(1, len(freq)):
        period = (1.0 / freq[i]) if freq[i] > 0 else float("nan")
        rows.append({"clade": clade, "frequency_cycles_per_residue": freq[i], "period_residues": period, "amplitude": amp[i]})
    return pd.DataFrame(rows)


def compute_clade_fourier_conservation(alignment: MultipleSeqAlignment,
                                        reference_species: Optional[str],
                                        smoothing_window: int,
                                        fourier_terms: int,
                                        clade_min_species: int = 1,
                                        taxonomy_lookup: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = list(alignment)
    if not records:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ref_idx = 0
    if reference_species:
        for i, record in enumerate(records):
            species, _ = parse_header_species_symbol(record.id)
            if species == reference_species:
                ref_idx = i
                break
    ref_record = records[ref_idx]
    ref_species, _ = parse_header_species_symbol(ref_record.id)
    aln_len = alignment.get_alignment_length()

    clade_to_indices: Dict[str, List[int]] = {}
    for i, record in enumerate(records):
        if i == ref_idx:
            continue
        species, _ = parse_header_species_symbol(record.id)
        clade = classify_site_species_clade(species, (taxonomy_lookup or {}).get(species))
        clade_to_indices.setdefault(clade, []).append(i)
    clade_order = [c for c in ["tetrapods", "dipnoi", "actinistia", "holostei", "teleosts", "other_fish", "other_vertebrates"] if len(clade_to_indices.get(c, [])) >= clade_min_species]

    rows: List[Dict[str, Any]] = []
    for pos in range(aln_len):
        ref_aa = str(ref_record.seq[pos]).upper()
        row: Dict[str, Any] = {
            "reference_position": pos + 1,
            "reference_residue": ref_aa,
        }
        global_residues = []
        global_identity = []
        global_hydro = []
        for i, record in enumerate(records):
            if i == ref_idx:
                continue
            aa = str(record.seq[pos]).upper()
            if aa in GAP_CHARS or aa in {"X", "?"}:
                continue
            global_residues.append(aa)
            global_identity.append(1.0 if aa == ref_aa else 0.0)
            prop = _same_property_as_reference(ref_aa, aa, "hydrophobicity")
            if prop is not None:
                global_hydro.append(1.0 if prop else 0.0)
        row["global_identity_to_human"] = float(np.mean(global_identity)) if global_identity else float("nan")
        row["global_hydrophobicity_agreement_to_human"] = float(np.mean(global_hydro)) if global_hydro else float("nan")
        row["global_n_non_gap"] = len(global_residues)
        for clade in clade_order:
            idxs = clade_to_indices.get(clade, [])
            residues = []
            ident = []
            hydro = []
            for i in idxs:
                aa = str(records[i].seq[pos]).upper()
                if aa in GAP_CHARS or aa in {"X", "?"}:
                    continue
                residues.append(aa)
                ident.append(1.0 if aa == ref_aa else 0.0)
                prop = _same_property_as_reference(ref_aa, aa, "hydrophobicity")
                if prop is not None:
                    hydro.append(1.0 if prop else 0.0)
            row[f"{clade}_n_non_gap"] = len(residues)
            row[f"{clade}_identity_to_human"] = float(np.mean(ident)) if ident else float("nan")
            row[f"{clade}_hydrophobicity_agreement_to_human"] = float(np.mean(hydro)) if hydro else float("nan")
            row[f"{clade}_major_residue"] = Counter(residues).most_common(1)[0][0] if residues else None
        rows.append(row)

    profile_df = pd.DataFrame(rows)
    spectrum_parts = []
    recon_df = profile_df[["reference_position", "reference_residue"]].copy()
    diff_df = profile_df[["reference_position", "reference_residue"]].copy()
    for name in ["global"] + clade_order:
        col = "global_identity_to_human" if name == "global" else f"{name}_identity_to_human"
        if col not in profile_df.columns:
            continue
        smoothed = smooth_series(profile_df[col].tolist(), smoothing_window)
        profile_df[f"{name}_identity_to_human_smoothed"] = smoothed
        recon = _fft_lowpass(profile_df[col].tolist(), keep_terms=fourier_terms)
        recon_df[f"{name}_identity_fft_lowpass"] = recon
        spec = _fft_spectrum(profile_df[col].tolist(), name)
        if not spec.empty:
            spectrum_parts.append(spec)
        if name != "global":
            diff_df[f"{name}_minus_global_identity"] = profile_df[col] - profile_df["global_identity_to_human"]
            diff_df[f"{name}_minus_global_identity_fft_lowpass"] = recon - recon_df["global_identity_fft_lowpass"]
    spectrum_df = pd.concat(spectrum_parts, ignore_index=True) if spectrum_parts else pd.DataFrame()
    return profile_df, spectrum_df, recon_df, diff_df


def detect_clade_regions_from_profiles(profile_df: pd.DataFrame,
                                       recon_df: pd.DataFrame,
                                       min_len: int,
                                       conserved_threshold: float,
                                       divergent_threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if profile_df.empty:
        return pd.DataFrame()
    clades = [c.replace("_identity_to_human", "") for c in profile_df.columns if c.endswith("_identity_to_human") and not c.startswith("global")]
    for clade in clades:
        col = f"{clade}_identity_to_human"
        fft_col = f"{clade}_identity_fft_lowpass"
        global_fft = "global_identity_fft_lowpass"
        values = recon_df[fft_col].to_numpy(dtype=float) if fft_col in recon_df.columns else _nan_interp(profile_df[col])
        global_values = recon_df[global_fft].to_numpy(dtype=float) if global_fft in recon_df.columns else _nan_interp(profile_df["global_identity_to_human"])
        modes = [
            ("clade_conserved_vs_human", values >= conserved_threshold),
            ("clade_divergent_from_global", (values - global_values) <= -abs(divergent_threshold)),
        ]
        for mode, mask in modes:
            start = None
            for i, good in enumerate(list(mask) + [False]):
                if good and start is None:
                    start = i
                elif (not good) and start is not None:
                    end = i - 1
                    if end - start + 1 >= min_len:
                        pos_start = int(profile_df["reference_position"].iloc[start])
                        pos_end = int(profile_df["reference_position"].iloc[end])
                        sub = profile_df.iloc[start:end+1]
                        rows.append({
                            "clade": clade,
                            "region_type": mode,
                            "start_reference_position": pos_start,
                            "end_reference_position": pos_end,
                            "length": pos_end - pos_start + 1,
                            "mean_identity_to_human": float(sub[col].mean()),
                            "mean_global_identity_to_human": float(sub["global_identity_to_human"].mean()),
                            "mean_delta_vs_global": float((sub[col] - sub["global_identity_to_human"]).mean()),
                        })
                    start = None
    return pd.DataFrame(rows)


def summarize_domains_by_clade(profile_df: pd.DataFrame, domain_df: pd.DataFrame,
                               reference_species: str, reference_symbol: str) -> pd.DataFrame:
    if profile_df.empty or domain_df.empty:
        return pd.DataFrame()
    ref_domains = domain_df[
        (domain_df["species"] == reference_species) &
        (domain_df["symbol"] == reference_symbol) &
        domain_df["start"].notna() & domain_df["end"].notna()
    ].copy()
    rows: List[Dict[str, Any]] = []
    if ref_domains.empty:
        # fallback: summarize coarse thirds if UniProt domain lookup fails
        max_pos = int(profile_df["reference_position"].max())
        ref_domains = pd.DataFrame([
            {"description": "N-terminal third", "feature_type": "fallback_region", "start": 1, "end": max_pos // 3},
            {"description": "Middle third", "feature_type": "fallback_region", "start": max_pos // 3 + 1, "end": 2 * max_pos // 3},
            {"description": "C-terminal third", "feature_type": "fallback_region", "start": 2 * max_pos // 3 + 1, "end": max_pos},
        ])
    clade_cols = [c for c in profile_df.columns if c.endswith("_identity_to_human")]
    for _, dom in ref_domains.iterrows():
        start = int(float(dom["start"])); end = int(float(dom["end"]))
        sub = profile_df[(profile_df["reference_position"] >= start) & (profile_df["reference_position"] <= end)]
        if sub.empty:
            continue
        desc = str(dom.get("description") or dom.get("feature_type") or "domain")
        for col in clade_cols:
            clade = col.replace("_identity_to_human", "")
            vals = pd.to_numeric(sub[col], errors="coerce")
            rows.append({
                "domain_label": desc,
                "feature_type": dom.get("feature_type"),
                "start": start,
                "end": end,
                "length": end - start + 1,
                "clade": clade,
                "mean_identity_to_human": float(vals.mean()) if vals.notna().any() else float("nan"),
                "min_identity_to_human": float(vals.min()) if vals.notna().any() else float("nan"),
                "fraction_positions_identity_ge_0_85": float((vals >= 0.85).mean()) if vals.notna().any() else float("nan"),
            })
    return pd.DataFrame(rows)


def plot_clade_fourier_profiles(profile_df: pd.DataFrame, recon_df: pd.DataFrame, out_svg: Path, title: str) -> None:
    if profile_df.empty or recon_df.empty:
        return
    fig = plt.figure(figsize=(18, 7))
    ax = fig.add_subplot(1, 1, 1)
    x = profile_df["reference_position"]
    cols = [c for c in recon_df.columns if c.endswith("_identity_fft_lowpass")]
    for col in cols:
        label = col.replace("_identity_fft_lowpass", "").replace("_", " ")
        ax.plot(x, recon_df[col], linewidth=1.5, label=label)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Human-reference residue position")
    ax.set_ylabel("FFT-smoothed identity to human")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=4, loc="lower left")
    fig.tight_layout()
    save_figure_svg_png(fig, out_svg)


def plot_clade_difference_heatmap(diff_df: pd.DataFrame, out_svg: Path, title: str) -> None:
    if diff_df.empty:
        return
    cols = [c for c in diff_df.columns if c.endswith("_minus_global_identity_fft_lowpass")]
    if not cols:
        return
    data = diff_df[cols].T.to_numpy(dtype=float)
    fig = plt.figure(figsize=(18, max(4, 0.55 * len(cols))))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(data, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([c.replace("_minus_global_identity_fft_lowpass", "").replace("_", " ") for c in cols], fontsize=8)
    ax.set_xlabel("Human-reference residue position")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="clade identity - global identity")
    fig.tight_layout()
    save_figure_svg_png(fig, out_svg)


def plot_fourier_spectrum(spectrum_df: pd.DataFrame, out_svg: Path, title: str, max_period: float = 250.0) -> None:
    if spectrum_df.empty:
        return
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    sub = spectrum_df[(spectrum_df["period_residues"] <= max_period) & (spectrum_df["period_residues"] >= 2)].copy()
    for clade, clade_df in sub.groupby("clade"):
        top = clade_df.sort_values("amplitude", ascending=False).head(60).sort_values("period_residues")
        ax.plot(top["period_residues"], top["amplitude"], marker="o", markersize=2.5, linewidth=1, label=clade.replace("_", " "))
    ax.set_xlabel("Fourier period length (residues)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    save_figure_svg_png(fig, out_svg)


def plot_domain_clade_heatmap(domain_summary_df: pd.DataFrame, out_svg: Path, title: str) -> None:
    if domain_summary_df.empty:
        return
    pivot = domain_summary_df.pivot_table(index="domain_label", columns="clade", values="mean_identity_to_human", aggfunc="mean")
    if pivot.empty:
        return
    fig = plt.figure(figsize=(max(10, 1.2 * len(pivot.columns)), max(4, 0.5 * len(pivot.index))))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("_", " ") for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i)[:55] for i in pivot.index], fontsize=8)
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            val = pivot.iloc[y, x]
            if pd.notna(val):
                ax.text(x, y, f"{val:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="mean identity to human")
    fig.tight_layout()
    save_figure_svg_png(fig, out_svg)

def filter_ortholog_table_by_patterns(ortholog_df: pd.DataFrame, patterns: Optional[Sequence[str]], source_species: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if ortholog_df.empty:
        return ortholog_df.copy(), pd.DataFrame()
    work = ortholog_df.copy()
    work["manual_pattern_selected"] = False
    work["manual_pattern_match"] = None
    if not patterns:
        return work, work[["species", "symbol", "manual_pattern_selected", "manual_pattern_match"]].copy()

    lowered = [(p, p.lower()) for p in patterns if str(p).strip()]
    selected_idx = []
    matched = []
    for idx, row in work.iterrows():
        species_text = str(row.get("species", "")).lower()
        symbol_text = str(row.get("symbol", "")).lower()
        if str(row.get("species", "")) == source_species:
            selected_idx.append(idx)
            matched.append((idx, "source_species"))
            continue
        for original, pat in lowered:
            if pat in species_text or pat in symbol_text:
                selected_idx.append(idx)
                matched.append((idx, original))
                break
    if not selected_idx:
        return work.iloc[0:0].copy(), work[["species", "symbol", "manual_pattern_selected", "manual_pattern_match"]].copy()
    out = work.loc[sorted(set(selected_idx))].copy()
    for idx, reason in matched:
        if idx in out.index:
            out.at[idx, "manual_pattern_selected"] = True
            out.at[idx, "manual_pattern_match"] = reason
            work.at[idx, "manual_pattern_selected"] = True
            work.at[idx, "manual_pattern_match"] = reason
    manifest = work[["species", "symbol", "manual_pattern_selected", "manual_pattern_match"]].copy()
    return out.reset_index(drop=True), manifest


def parse_annotated_sites(site_text: Optional[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not site_text:
        return pd.DataFrame(columns=["position", "label"])
    for chunk in str(site_text).split(","):
        item = chunk.strip()
        if not item:
            continue
        if ":" in item:
            pos_text, label = item.split(":", 1)
        else:
            pos_text, label = item, item
        pos_text = pos_text.strip()
        label = label.strip() or pos_text
        try:
            pos = int(pos_text)
        except ValueError:
            continue
        rows.append({"position": pos, "label": label})
    df = pd.DataFrame(rows).drop_duplicates(subset=["position", "label"]).sort_values("position") if rows else pd.DataFrame(columns=["position", "label"])
    return df


def annotate_functional_sites_against_conservation(conservation_df: pd.DataFrame, annotated_sites_df: pd.DataFrame) -> pd.DataFrame:
    if annotated_sites_df is None or annotated_sites_df.empty:
        return pd.DataFrame(columns=["position", "label", "reference_residue", "identity_max", "reference_identity_fraction", "occupancy", "reference_domain_labels"])
    scan = conservation_df[conservation_df["reference_ungapped_position"].notna()].copy()
    scan["reference_ungapped_position"] = scan["reference_ungapped_position"].astype(int)
    merged = annotated_sites_df.merge(scan, left_on="position", right_on="reference_ungapped_position", how="left")
    keep = ["position", "label", "reference_residue", "identity_max", "reference_identity_fraction", "occupancy", "reference_domain_labels"]
    for col in keep:
        if col not in merged.columns:
            merged[col] = None
    return merged[keep].copy()


def pymol_conservation_commands(conservation_df: pd.DataFrame, object_name: str = "human_ref") -> List[str]:
    scan = conservation_df[conservation_df["reference_ungapped_position"].notna()].copy()
    if scan.empty:
        return []
    scan["reference_ungapped_position"] = scan["reference_ungapped_position"].astype(int)
    lines = [
        f"hide everything, {object_name}",
        f"show cartoon, {object_name}",
        "set cartoon_fancy_helices, 1",
        f"color grey80, {object_name}",
    ]
    for _, row in scan.iterrows():
        pos = int(row["reference_ungapped_position"])
        score = float(row.get("reference_identity_fraction", 0.0) or 0.0)
        if score >= 0.95:
            color = "red"
        elif score >= 0.80:
            color = "orange"
        elif score >= 0.60:
            color = "yellow"
        elif score >= 0.40:
            color = "tv_green"
        else:
            color = "cyan"
        lines.append(f"color {color}, {object_name} and resi {pos}")
    lines.append(f"spectrum b, blue_white_red, {object_name}")
    return lines


def write_pymol_coloring_script(conservation_df: pd.DataFrame, out_path: Path, object_name: str = "human_ref") -> None:
    commands = pymol_conservation_commands(conservation_df, object_name=object_name)
    if not commands:
        return
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# PyMOL coloring script for {object_name}\n")
        handle.write("\n".join(commands) + "\n")


def pymol_secondary_structure_commands(object_name: str = "human_ref") -> List[str]:
    return [
        f"hide everything, {object_name}",
        f"show cartoon, {object_name}",
        "set cartoon_fancy_helices, 1",
        f"dss {object_name}",
        f"color gray70, {object_name}",
        f"color orange, {object_name} and ss h",
        f"color purple, {object_name} and ss s",
        f"color gray70, {object_name} and not (ss h+s)",
    ]


def normalize_color_hex(value: Optional[str]) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", text):
        return text
    return "#d97706"


def load_selected_consensus_chunks(consensus_chunk_table: Optional[str]) -> pd.DataFrame:
    columns = list(SELECTED_CONSENSUS_CHUNK_COLUMNS)
    if not consensus_chunk_table:
        return pd.DataFrame(columns=columns)
    path = Path(consensus_chunk_table)
    if not path.exists():
        raise FileNotFoundError(f"Consensus chunk table does not exist: {path}")
    df = pd.read_csv(path, sep=None, engine="python")
    for column in columns:
        if column not in df.columns:
            df[column] = None
    df = df[columns].copy()
    for column in ("start_reference_position", "end_reference_position"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["color_hex"] = df["color_hex"].map(normalize_color_hex)
    df["chunk_id"] = df["chunk_id"].fillna("").map(str)
    df["label"] = df["label"].fillna("").map(str)
    df["notes"] = df["notes"].fillna("").map(str)
    return df


def ensure_selected_consensus_chunk_template(outdir: Path,
                                             consensus_chunk_table: Optional[str] = None) -> pd.DataFrame:
    df = load_selected_consensus_chunks(consensus_chunk_table)
    out_path = outdir / SELECTED_CONSENSUS_CHUNKS_FILENAME
    if df.empty:
        pd.DataFrame(columns=SELECTED_CONSENSUS_CHUNK_COLUMNS).to_csv(out_path, sep="\t", index=False)
        return pd.DataFrame(columns=SELECTED_CONSENSUS_CHUNK_COLUMNS)
    df.to_csv(out_path, sep="\t", index=False)
    return df


def fetch_alphafold_prediction(accession: str) -> Dict[str, Any]:
    url = f"{ALPHAFOLD}/api/prediction/{accession}"
    data = get_json(url)
    if isinstance(data, list):
        if not data:
            raise APIError(f"AlphaFold returned no models for {accession}")
        return data[0]
    if isinstance(data, dict):
        return data
    raise APIError(f"AlphaFold response for {accession} was not understood.")


def stage_reference_and_override_alphafold_models(outdir: Path,
                                                  reference_accession: Optional[str] = None,
                                                  reference_model_filename: Optional[str] = None) -> Dict[str, Any]:
    """V9.8c durability fix: write_comparative_alphafold_secondary_structure_bundle
    only looks inside outdir/comparative_alphafold_models/ when deciding whether
    a species has its own AF PDB. Two edge cases would otherwise silently fall
    back to projection on every fresh pipeline run:

    1. The human reference PDB is downloaded as human_reference_alphafold_model.pdb
       in the main outdir. Copy it into the comparative cache under the canonical
       AF-{accession}-F1-model_v6.pdb naming so the human row escapes projection.
    2. COMPARATIVE_ALPHAFOLD_ACCESSION_OVERRIDES re-points species (currently just
       danio_rerio) at alternative accessions (currently P50392, the canonical
       human PLA2G4A) that no row in protein_metadata.tsv owns. Download those
       so the override-targeted PDB exists in the cache.
    """
    from gene_phylo_conservation_archive import (
        COMPARATIVE_ALPHAFOLD_MODEL_DIRNAME,
        COMPARATIVE_ALPHAFOLD_ACCESSION_OVERRIDES,
    )
    summary = {"copied_reference": 0, "fetched_overrides": 0, "cached_overrides": 0, "errored_overrides": 0}

    comp_dir = outdir / COMPARATIVE_ALPHAFOLD_MODEL_DIRNAME
    comp_dir.mkdir(parents=True, exist_ok=True)

    # (1) Mirror the human reference PDB into the comparative cache.
    ref_accession = str(reference_accession or "").strip().upper()
    if ref_accession:
        existing = list(comp_dir.glob(f"AF-{ref_accession}-F1-model_v*.pdb"))
        if not existing:
            ref_source_filename = str(reference_model_filename or "human_reference_alphafold_model.pdb").strip()
            ref_source_path = outdir / ref_source_filename
            if ref_source_path.exists():
                shutil.copyfile(ref_source_path, comp_dir / f"AF-{ref_accession}-F1-model_v6.pdb")
                summary["copied_reference"] = 1
                emit_log(f"Comparative AF: copied reference AF model into cache as AF-{ref_accession}-F1-model_v6.pdb")
            else:
                emit_log(f"Comparative AF: reference model file not found at {ref_source_path}; skipping copy")

    # (2) Fetch any override-targeted accessions (e.g. P50392 for danio_rerio).
    seen_override_accessions: set = set()
    for override in COMPARATIVE_ALPHAFOLD_ACCESSION_OVERRIDES.values():
        accession = str(override.get("uniprot_accession") or "").strip().upper()
        if not accession or accession in seen_override_accessions:
            continue
        if accession == ref_accession:
            continue
        seen_override_accessions.add(accession)
        existing = list(comp_dir.glob(f"AF-{accession}-F1-model_v*.pdb"))
        if existing and any(p.stat().st_size > 1000 for p in existing):
            summary["cached_overrides"] += 1
            continue
        try:
            prediction = fetch_alphafold_prediction(accession)
            pdb_url = clean_serializable(prediction.get("pdbUrl"))
            if not pdb_url:
                emit_log(f"Comparative AF override: AlphaFold returned no pdbUrl for {accession}")
                summary["errored_overrides"] += 1
                continue
            url_filename = pdb_url.rsplit("/", 1)[-1] or f"AF-{accession}-F1-model_v4.pdb"
            download_file(pdb_url, comp_dir / url_filename)
            summary["fetched_overrides"] += 1
            emit_log(f"Comparative AF override: fetched {url_filename}")
        except Exception as exc:
            summary["errored_overrides"] += 1
            emit_log(f"Comparative AF override: error fetching {accession}: {exc}")

    return summary


def fetch_comparative_alphafold_models(outdir: Path,
                                       protein_metadata_df: pd.DataFrame,
                                       reference_accession: Optional[str] = None,
                                       max_workers: int = 4) -> Dict[str, Any]:
    """V9.8c: Download per-species AlphaFold PDBs into the comparative cache so
    write_comparative_alphafold_secondary_structure_bundle can produce real
    per-species SS rather than the reference projection. Skips already-cached
    files, blank/missing UniProt accessions, and the reference accession (the
    primary human model is downloaded separately). Errors on individual species
    are non-fatal -- they fall back to the projection in the SS bundle. Returns
    a small summary dict for the step log."""
    from gene_phylo_conservation_archive import COMPARATIVE_ALPHAFOLD_MODEL_DIRNAME
    summary = {"fetched": 0, "cached": 0, "skipped": 0, "errored": 0, "no_accession": 0}
    if protein_metadata_df is None or protein_metadata_df.empty:
        return summary
    if "uniprot_accession" not in protein_metadata_df.columns:
        summary["no_accession"] = int(len(protein_metadata_df))
        return summary

    model_dir = outdir / COMPARATIVE_ALPHAFOLD_MODEL_DIRNAME
    model_dir.mkdir(parents=True, exist_ok=True)

    ref_accession = str(reference_accession or "").strip().upper()
    seen: set = set()
    targets: List[Tuple[str, str]] = []  # (species, accession)
    for _, row in protein_metadata_df.iterrows():
        accession = str(row.get("uniprot_accession") or "").strip()
        if not accession or accession.lower() in {"nan", "none"}:
            summary["no_accession"] += 1
            continue
        accession_upper = accession.upper()
        if accession_upper == ref_accession:
            continue
        if accession_upper in seen:
            continue
        seen.add(accession_upper)
        species = str(row.get("species") or "").strip() or accession_upper
        targets.append((species, accession_upper))

    remaining: List[Tuple[str, str]] = []
    for species, accession in targets:
        existing = list(model_dir.glob(f"AF-{accession}-F1-model_v*.pdb"))
        if existing and any(p.stat().st_size > 1000 for p in existing):
            summary["cached"] += 1
        else:
            remaining.append((species, accession))

    if not remaining:
        emit_log(
            f"Comparative AlphaFold models: {summary['cached']} cached, nothing new to fetch."
        )
        return summary

    emit_log(
        f"Comparative AlphaFold models: {summary['cached']} cached, fetching {len(remaining)} "
        f"new (thread pool size {max_workers})."
    )

    def _fetch_one(species_accession: Tuple[str, str]) -> Tuple[str, str, str, Optional[str]]:
        species, accession = species_accession
        try:
            prediction = fetch_alphafold_prediction(accession)
        except Exception as exc:
            return (species, accession, "no_prediction", str(exc))
        pdb_url = clean_serializable(prediction.get("pdbUrl"))
        if not pdb_url:
            return (species, accession, "no_pdb_url", "AlphaFold prediction had no pdbUrl")
        url_filename = pdb_url.rsplit("/", 1)[-1] or f"AF-{accession}-F1-model_v4.pdb"
        out_path = model_dir / url_filename
        try:
            download_file(pdb_url, out_path)
        except Exception as exc:
            return (species, accession, "download_failed", str(exc))
        return (species, accession, "ok", None)

    workers = max(1, min(max_workers, len(remaining)))
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_target = {executor.submit(_fetch_one, t): t for t in remaining}
        for future in as_completed(future_to_target):
            species, accession, status, message = future.result()
            completed += 1
            if status == "ok":
                summary["fetched"] += 1
            elif status in {"no_prediction", "no_pdb_url"}:
                summary["skipped"] += 1
            else:
                summary["errored"] += 1
            if completed % 25 == 0 or completed == len(remaining):
                emit_log(
                    f"Comparative AlphaFold: {completed}/{len(remaining)}; "
                    f"fetched={summary['fetched']}, skipped={summary['skipped']}, errored={summary['errored']}"
                )

    return summary


def ensure_alphafold_viewer_asset(outdir: Path, alphafold_mode: str = "auto") -> Optional[Path]:
    viewer_path = outdir / ALPHAFOLD_VIEWER_JS_FILENAME
    if viewer_path.exists() and viewer_path.stat().st_size > 100_000:
        return viewer_path
    errors: List[str] = []
    for url in ALPHAFOLD_VIEWER_JS_URLS:
        try:
            return download_file(url, viewer_path, headers={"Accept": "application/javascript,*/*"})
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            continue
    message = "; ".join(errors) if errors else "no download URL was attempted"
    if alphafold_mode == "required":
        raise RuntimeError(f"3Dmol.js viewer asset could not be cached: {message}")
    emit_log(f"AlphaFold embedded viewer fallback: could not cache 3Dmol.js ({message})")
    return None


def build_selected_consensus_chunk_map(chunk_df: pd.DataFrame,
                                       protein_length: Optional[int]) -> pd.DataFrame:
    columns = [
        "chunk_id", "label", "start_reference_position", "end_reference_position",
        "start_structure_residue", "end_structure_residue", "score", "color_hex",
        "notes", "mapping_status",
    ]
    if chunk_df is None or chunk_df.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, Any]] = []
    for _, row in chunk_df.iterrows():
        start = row.get("start_reference_position")
        end = row.get("end_reference_position")
        status = "ok"
        if pd.isna(start) or pd.isna(end):
            status = "missing_range"
        else:
            start = int(start)
            end = int(end)
            if start > end:
                start, end = end, start
                status = "range_swapped"
            if protein_length and (start < 1 or end > int(protein_length)):
                status = "out_of_bounds"
        rows.append({
            "chunk_id": clean_serializable(row.get("chunk_id")),
            "label": clean_serializable(row.get("label")),
            "start_reference_position": clean_serializable(start),
            "end_reference_position": clean_serializable(end),
            "start_structure_residue": clean_serializable(start),
            "end_structure_residue": clean_serializable(end),
            "score": clean_serializable(row.get("score")),
            "color_hex": normalize_color_hex(row.get("color_hex")),
            "notes": clean_serializable(row.get("notes")),
            "mapping_status": status,
        })
    return pd.DataFrame(rows, columns=columns)


def write_alphafold_overlay_script(conservation_df: pd.DataFrame,
                                   model_filename: str,
                                   out_path: Path,
                                   chunk_map_df: Optional[pd.DataFrame] = None,
                                   object_name: str = "human_ref") -> None:
    commands = [
        f"# AlphaFold overlay script for {object_name}",
        f"load {model_filename}, {object_name}",
        f"remove not polymer.protein and {object_name}",
    ]
    commands.extend(pymol_secondary_structure_commands(object_name=object_name))
    commands.append("# Secondary-structure cartoon colors: alpha helix orange, beta sheet purple, loops gray.")
    commands.append("# Conservation coloring remains available in human_reference_conservation_coloring.pml.")
    chunk_rows = chunk_map_df.to_dict(orient="records") if chunk_map_df is not None and not chunk_map_df.empty else []
    if chunk_rows:
        commands.append(f"set cartoon_transparency, 0.15, {object_name}")
        for row in chunk_rows:
            if row.get("mapping_status") == "missing_range":
                continue
            chunk_id = sanitize_filename(str(row.get("chunk_id") or row.get("label") or "chunk"))
            label = str(row.get("label") or chunk_id)
            color_hex = normalize_color_hex(row.get("color_hex"))
            start = int(row.get("start_structure_residue") or row.get("start_reference_position") or 0)
            end = int(row.get("end_structure_residue") or row.get("end_reference_position") or 0)
            red = int(color_hex[1:3], 16) / 255.0
            green = int(color_hex[3:5], 16) / 255.0
            blue = int(color_hex[5:7], 16) / 255.0
            commands.append(f"set_color {chunk_id}_color, [{red:.3f}, {green:.3f}, {blue:.3f}]")
            commands.append(f"select {chunk_id}, {object_name} and resi {start}-{end}")
            commands.append(f"show sticks, {chunk_id}")
            commands.append(f"color {chunk_id}_color, {chunk_id}")
            commands.append(f"label ({chunk_id} and name CA and resi {start}), \"{label}\"")
    out_path.write_text("\n".join(commands) + "\n", encoding="utf-8")


def prepare_human_reference_alphafold_bundle(outdir: Path,
                                             protein_metadata_df: pd.DataFrame,
                                             protein_xrefs_df: pd.DataFrame,
                                             conservation_df: pd.DataFrame,
                                             reference_species: Optional[str],
                                             alphafold_mode: str = "auto",
                                             consensus_chunk_table: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    chunk_df = ensure_selected_consensus_chunk_template(outdir, consensus_chunk_table=consensus_chunk_table)
    metadata_path = outdir / ALPHAFOLD_METADATA_FILENAME
    empty_chunk_map = pd.DataFrame(columns=[
        "chunk_id", "label", "start_reference_position", "end_reference_position",
        "start_structure_residue", "end_structure_residue", "score", "color_hex", "notes", "mapping_status",
    ])
    if alphafold_mode == "off":
        payload = {
            "available": False,
            "mode": alphafold_mode,
            "reason": "AlphaFold bundle generation was disabled.",
        }
        write_json_file(metadata_path, payload)
        return protein_metadata_df, protein_xrefs_df, payload, empty_chunk_map

    human_mask = (
        protein_metadata_df["species"].astype(str).eq(str(reference_species or "").strip())
        & protein_metadata_df["is_reference"].astype(bool)
    ) if not protein_metadata_df.empty else pd.Series(dtype=bool)
    if protein_metadata_df.empty or not human_mask.any() or str(reference_species or "").strip() != "homo_sapiens":
        payload = {
            "available": False,
            "mode": alphafold_mode,
            "reason": "Human reference protein metadata was not available for AlphaFold export.",
        }
        write_json_file(metadata_path, payload)
        if alphafold_mode == "required":
            raise RuntimeError(payload["reason"])
        return protein_metadata_df, protein_xrefs_df, payload, empty_chunk_map

    human_index = protein_metadata_df[human_mask].index[0]
    human_row = protein_metadata_df.loc[human_index].to_dict()
    accession = str(human_row.get("uniprot_accession") or "").strip()
    if not accession:
        payload = {
            "available": False,
            "mode": alphafold_mode,
            "reason": "No human UniProt accession was available for AlphaFold export.",
            "protein_record_id": clean_serializable(human_row.get("protein_record_id")),
        }
        write_json_file(metadata_path, payload)
        if alphafold_mode == "required":
            raise RuntimeError(payload["reason"])
        return protein_metadata_df, protein_xrefs_df, payload, empty_chunk_map

    try:
        alphafold_prediction = fetch_alphafold_prediction(accession)
        model_url = clean_serializable(alphafold_prediction.get("pdbUrl"))
        if not model_url:
            raise RuntimeError(f"AlphaFold prediction for {accession} did not expose a PDB URL.")
        model_path = download_file(model_url, outdir / ALPHAFOLD_MODEL_FILENAME)
        viewer_asset_path = ensure_alphafold_viewer_asset(outdir, alphafold_mode=alphafold_mode)
        chunk_map_df = build_selected_consensus_chunk_map(chunk_df, protein_length=human_row.get("length_aa"))
        if not chunk_map_df.empty:
            chunk_map_df.to_csv(outdir / SELECTED_CONSENSUS_CHUNKS_MAP_FILENAME, sep="\t", index=False)
        write_alphafold_overlay_script(
            conservation_df=conservation_df,
            model_filename=model_path.name,
            out_path=outdir / ALPHAFOLD_OVERLAY_FILENAME,
            chunk_map_df=chunk_map_df,
            object_name="human_ref",
        )
        payload = {
            "available": True,
            "mode": alphafold_mode,
            "protein_record_id": clean_serializable(human_row.get("protein_record_id")),
            "uniprot_accession": accession,
            "alphafold_prediction": copy.deepcopy(alphafold_prediction),
            "model_filename": model_path.name,
            "overlay_filename": ALPHAFOLD_OVERLAY_FILENAME,
            "viewer_js_filename": viewer_asset_path.name if viewer_asset_path else None,
            "selected_consensus_chunks_filename": SELECTED_CONSENSUS_CHUNKS_FILENAME,
            "selected_consensus_chunks_map_filename": SELECTED_CONSENSUS_CHUNKS_MAP_FILENAME if not chunk_map_df.empty else None,
        }
        write_json_file(metadata_path, payload)
        protein_metadata_df = protein_metadata_df.copy()
        for column in ("alphafold_accession", "alphafold_entry_id", "alphafold_model_url", "alphafold_source_label"):
            if column in protein_metadata_df.columns:
                protein_metadata_df[column] = protein_metadata_df[column].astype(object)
        protein_metadata_df.loc[human_index, "alphafold_accession"] = clean_serializable(alphafold_prediction.get("uniprotAccession")) or accession
        protein_metadata_df.loc[human_index, "alphafold_entry_id"] = clean_serializable(alphafold_prediction.get("entryId"))
        protein_metadata_df.loc[human_index, "alphafold_model_url"] = model_url
        protein_metadata_df.loc[human_index, "alphafold_source_label"] = f"AlphaFold DB v{clean_serializable(alphafold_prediction.get('latestVersion'))}"
        protein_xrefs_df = protein_xrefs_df.copy()
        new_xref = pd.DataFrame([{
            "protein_record_id": clean_serializable(human_row.get("protein_record_id")),
            "species": clean_serializable(human_row.get("species")),
            "symbol": clean_serializable(human_row.get("symbol")),
            "uniprot_accession": accession,
            "database": "AlphaFoldDB",
            "external_id": clean_serializable(alphafold_prediction.get("entryId")),
            "label": clean_serializable(alphafold_prediction.get("uniprotDescription")),
            "category": "alphafold",
        }])
        protein_xrefs_df = pd.concat([protein_xrefs_df, new_xref], ignore_index=True).drop_duplicates()
        return protein_metadata_df, protein_xrefs_df, payload, chunk_map_df
    except Exception as exc:
        payload = {
            "available": False,
            "mode": alphafold_mode,
            "protein_record_id": clean_serializable(human_row.get("protein_record_id")),
            "uniprot_accession": accession,
            "reason": str(exc),
        }
        write_json_file(metadata_path, payload)
        if alphafold_mode == "required":
            raise RuntimeError(f"AlphaFold bundle generation failed for {accession}: {exc}") from exc
        emit_log(f"AlphaFold bundle fallback: {exc}")
        return protein_metadata_df, protein_xrefs_df, payload, empty_chunk_map


def plot_alignment_colored_pages(alignment: MultipleSeqAlignment, out_pdf: Path,
                                 reference_species: Optional[str] = None,
                                 residues_per_line: int = 100,
                                 blocks_per_page: int = 6) -> None:
    records = list(alignment)
    if not records:
        return
    seq_len = alignment.get_alignment_length()
    label_width = max(18, min(30, max(len(short_record_label(r.id, 30)) for r in records)))
    n_blocks = math.ceil(seq_len / residues_per_line)
    pages = math.ceil(n_blocks / blocks_per_page)
    with PdfPages(out_pdf) as pdf:
        for page in range(pages):
            start_block = page * blocks_per_page
            end_block = min(n_blocks, start_block + blocks_per_page)
            blocks_on_page = end_block - start_block
            fig_h = 0.8 + blocks_on_page * (len(records) * 0.32 + 1.4)
            fig, axes = plt.subplots(blocks_on_page, 1, figsize=(18, max(4.5, fig_h)))
            if blocks_on_page == 1:
                axes = [axes]
            for ax, block_index in zip(axes, range(start_block, end_block)):
                start = block_index * residues_per_line
                end = min(seq_len, start + residues_per_line)
                block_len = end - start
                ax.set_xlim(-label_width - 1.5, block_len)
                ax.set_ylim(len(records)+0.35, -1.15)
                ax.axis('off')
                for pos in range(0, block_len, 5):
                    abs_pos = start + pos + 1
                    ax.text(pos, -0.55, str(abs_pos), fontsize=6.5, family='monospace', color='0.25')
                for row_i, record in enumerate(records):
                    y = row_i
                    ax.text(-label_width, y + 0.5, short_record_label(record.id, label_width), fontsize=7.2, family='monospace', va='center')
                    seg = str(record.seq[start:end])
                    for col_i, aa in enumerate(seg):
                        color = AA_COLORS.get(aa.upper(), '#F0F0F0')
                        ax.add_patch(Rectangle((col_i, y), 1, 1, facecolor=color, edgecolor='0.8', linewidth=0.25))
                        ax.text(col_i + 0.5, y + 0.53, aa, ha='center', va='center', fontsize=5.3, family='monospace')
                ax.set_title(f'{start+1}-{end}', fontsize=8, loc='left', pad=2)
            fig.suptitle(f'Colored reference-projected protein alignment (page {page + 1} of {pages})', fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            fig.savefig(out_pdf.with_name(f"{out_pdf.stem}_page_{page+1}.png"), dpi=240, bbox_inches='tight')
            plt.close(fig)


def _export_pairwise_reference_report_worker(reference_record_id: str,
                                             reference_record_seq: str,
                                             record_id: str,
                                             record_seq: str,
                                             pair_dir: str,
                                             reference_species: Optional[str],
                                             residues_per_line: int) -> str:
    pair_alignment = MultipleSeqAlignment([
        SeqRecord(Seq(reference_record_seq), id=reference_record_id, description=""),
        SeqRecord(Seq(record_seq), id=record_id, description=""),
    ])
    species, _ = parse_header_species_symbol(record_id)
    stem = sanitize_filename(f'human_vs_{species}')
    pair_dir_path = Path(pair_dir)
    write_pretty_alignment_text(
        pair_alignment,
        pair_dir_path / f'{stem}.txt',
        reference_species=reference_species,
        residues_per_line=residues_per_line,
    )
    plot_alignment_blocks_pdf(
        pair_alignment,
        pair_dir_path / f'{stem}.pdf',
        reference_species=reference_species,
        residues_per_line=residues_per_line,
        blocks_per_page=8,
    )
    return species


def export_pairwise_reference_reports(alignment: MultipleSeqAlignment, outdir: Path,
                                     reference_species: Optional[str] = None,
                                     residues_per_line: int = 70) -> None:
    records = list(alignment)
    if len(records) < 2:
        return
    ref_idx = 0
    if reference_species:
        for i, record in enumerate(records):
            species, _ = parse_header_species_symbol(record.id)
            if species == reference_species:
                ref_idx = i
                break
    ref_record = records[ref_idx]
    pair_dir = outdir / 'pairwise_human_reference_alignments'
    pair_dir.mkdir(parents=True, exist_ok=True)
    tasks = [record for i, record in enumerate(records) if i != ref_idx]
    total = len(tasks)
    worker_count = min(total, choose_worker_count(PAIRWISE_REPORT_MAX_WORKERS, reserve_one_core=True)) if total else 1
    emit_log(
        f"Exporting {total} pairwise reference report sets with up to {worker_count} "
        f"worker process{'es' if worker_count != 1 else ''}."
    )

    if worker_count <= 1:
        for idx, record in enumerate(tasks, start=1):
            species = _export_pairwise_reference_report_worker(
                ref_record.id,
                str(ref_record.seq),
                record.id,
                str(record.seq),
                str(pair_dir),
                reference_species,
                residues_per_line,
            )
            emit_log(f"Pairwise reference report {idx}/{total}: completed for {species}.")
        return

    try:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    _export_pairwise_reference_report_worker,
                    ref_record.id,
                    str(ref_record.seq),
                    record.id,
                    str(record.seq),
                    str(pair_dir),
                    reference_species,
                    residues_per_line,
                ): idx
                for idx, record in enumerate(tasks, start=1)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                species = future.result()
                emit_log(f"Pairwise reference report {idx}/{total}: completed for {species}.")
    except Exception as exc:
        emit_log(
            f"Pairwise report worker pool was unavailable ({exc}). "
            "Falling back to sequential export."
        )
        for idx, record in enumerate(tasks, start=1):
            species = _export_pairwise_reference_report_worker(
                ref_record.id,
                str(ref_record.seq),
                record.id,
                str(record.seq),
                str(pair_dir),
                reference_species,
                residues_per_line,
            )
            emit_log(f"Pairwise reference report {idx}/{total}: completed for {species}.")


def write_summary_text(out_path: Path,
                       gene_symbol: str,
                       source_species: str,
                       seq_df: pd.DataFrame,
                       conserved_regions: pd.DataFrame,
                       alignment_method: Optional[str] = None,
                       tree_method: Optional[str] = None,
                       tree_built: Optional[bool] = None,
                       full_alignment_length: Optional[int] = None,
                       reference_projected_length: Optional[int] = None,
                       selection_mode: Optional[str] = None,
                       manual_species_patterns: Optional[Sequence[str]] = None,
                       annotated_sites_text: Optional[str] = None,
                       fasta_object_metadata_path: Optional[str] = None,
                       site_window: Optional[int] = None,
                       site_clade_mya_text: Optional[str] = None,
                       metadata_enrichment_mode: Optional[str] = None,
                       tree_nomenclature_source: Optional[str] = None,
                       alphafold_mode: Optional[str] = None,
                       consensus_chunk_table: Optional[str] = None) -> None:
    ok_df = seq_df[seq_df["status"] == "ok"]
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Gene: {gene_symbol}\n")
        handle.write(f"Source species: {source_species}\n")
        handle.write(f"Sequences recovered: {len(ok_df)} / {len(seq_df)}\n")
        if alignment_method is not None:
            handle.write(f"Alignment method: {alignment_method}\n")
        if tree_method is not None:
            handle.write(f"Tree method: {tree_method}\n")
        if tree_built is not None:
            handle.write(f"Phylogeny built: {'yes' if tree_built else 'no'}\n")
        if full_alignment_length is not None:
            handle.write(f"Full alignment length: {full_alignment_length}\n")
        if reference_projected_length is not None:
            handle.write(f"Reference-projected alignment length: {reference_projected_length}\n")
        if selection_mode is not None:
            handle.write(f"Selection mode: {selection_mode}\n")
        if manual_species_patterns:
            handle.write(f"Manual species filters: {', '.join(manual_species_patterns)}\n")
        if annotated_sites_text:
            handle.write(f"Annotated sites: {annotated_sites_text}\n")
        if fasta_object_metadata_path:
            handle.write(f"FASTA object metadata file: {fasta_object_metadata_path}\n")
        if site_window is not None:
            handle.write(f"Annotated-site comparison window: +/-{site_window} residues\n")
        if site_clade_mya_text:
            handle.write(f"Annotated-site clade MyA overrides: {site_clade_mya_text}\n")
        if metadata_enrichment_mode:
            handle.write(f"Protein metadata enrichment mode: {metadata_enrichment_mode}\n")
        if tree_nomenclature_source:
            handle.write(f"Tree nomenclature source: {tree_nomenclature_source}\n")
        if alphafold_mode:
            handle.write(f"AlphaFold bundle mode: {alphafold_mode}\n")
        if consensus_chunk_table:
            handle.write(f"Consensus chunk table: {consensus_chunk_table}\n")
        handle.write("\n")
        handle.write("Recovered species:\n")
        for _, row in ok_df.iterrows():
            handle.write(f"  - {row['species']} ({row['symbol']}) len={row['length_aa']}\n")

        handle.write("\nTop conserved regions:\n")
        if conserved_regions.empty:
            handle.write("  None found with the current thresholds.\n")
        else:
            top = conserved_regions.sort_values(["mean_score", "length_alignment"], ascending=[False, False]).head(20)
            for _, row in top.iterrows():
                handle.write(
                    f"  - {row['score_column']}: aln {row['start_alignment_position']}-{row['end_alignment_position']}, "
                    f"ref {row['start_reference_position']}-{row['end_reference_position']}, "
                    f"len={row['length_alignment']}, mean={row['mean_score']:.3f}\n"
                )


def run_pipeline(args: argparse.Namespace, progress_callback: ProgressCallback = None) -> None:
    pipeline_started = time.time()
    # V11: +1 step for representative-property comparison,
    # +1 step for motif evolution + lineage stabilization.
    total_steps = 14 + int(bool(getattr(args, "run_phylogeny", False)))

    def update_status(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    def start_step(step_number: int, label: str) -> float:
        message = f"Step {step_number}/{total_steps}: {label}"
        update_status(message)
        emit_log(message)
        return time.time()

    def finish_step(step_started: float, detail: str) -> None:
        emit_log(f"{detail} ({format_elapsed(time.time() - step_started)}).")

    step_number = 1
    step_started = start_step(step_number, "Preparing output folder and checking tools")
    # V11: register the active gene symbol so all figure/report labels are
    # gene-agnostic (replaces V9.9's hard-coded "PLA2G4A" strings).
    v11_set_active_gene(getattr(args, "gene_symbol", None))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        mafft_exe = check_executable(args.mafft_exe, ["mafft"], "MAFFT")
    except FileNotFoundError:
        mafft_exe = None

    try:
        muscle_exe = check_executable(args.muscle_exe, ["muscle", "muscle5"], "MUSCLE")
    except FileNotFoundError:
        muscle_exe = None

    requested_tree_method = normalize_tree_method(getattr(args, "tree_method", "auto"))
    iqtree_exe = None
    if getattr(args, "run_phylogeny", False) and requested_tree_method in {"auto", "iqtree"}:
        try:
            iqtree_exe = check_executable(args.iqtree_exe, ["iqtree3", "iqtree2", "iqtree"], "IQ-TREE")
        except FileNotFoundError:
            if requested_tree_method == "iqtree":
                raise RuntimeError(
                    "Tree method 'iqtree' was selected, but IQ-TREE was not found. "
                    "Provide --iqtree_exe or switch --tree_method to auto or python_nj."
                )
            iqtree_exe = None
    emit_log(f"Output directory: {outdir.resolve()}")
    emit_log(
        "Executable availability: "
        f"MAFFT={'yes' if mafft_exe else 'no'}, "
        f"MUSCLE={'yes' if muscle_exe else 'no'}, "
        f"IQ-TREE={'yes' if iqtree_exe else 'no'}"
    )
    emit_log(
        "Worker configuration: "
        f"sequence_fetch={choose_worker_count(SEQUENCE_FETCH_MAX_WORKERS)}, "
        f"python_fallback={choose_worker_count(PYTHON_FALLBACK_MAX_WORKERS, reserve_one_core=True)}, "
        f"pairwise_reports={choose_worker_count(PAIRWISE_REPORT_MAX_WORKERS, reserve_one_core=True)}"
    )
    finish_step(step_started, "Preparation complete")

    step_number += 1
    step_started = start_step(step_number, "Retrieving orthologs from Ensembl")
    ortholog_df = collect_ortholog_table(
        source_species=args.source_species,
        gene_symbol=args.gene_symbol,
        target_species=args.target_species,
        filter_vertebrates=args.filter_vertebrates,
        selection_mode=getattr(args, "selection_mode", "all_filtered"),
    )
    ortholog_df.to_csv(outdir / "orthologs.tsv", sep="\t", index=False)
    finish_step(step_started, f"Ortholog table written with {len(ortholog_df)} rows")

    step_number += 1
    step_started = start_step(step_number, "Fetching protein sequences")
    seq_df, records = collect_sequences(
        ortholog_df,
        gene_symbol=args.gene_symbol,
        gene_label=getattr(args, "gene_label", None),
        fasta_object_metadata_path=getattr(args, "fasta_object_metadata_path", None),
    )
    # V12: swap configured Ensembl translations for a same-gene UniProt isoform
    # (e.g. Bos taurus DHRS7 373 aa -> Q24K14 339 aa) so they clear the reference
    # length filter and gain a real structure, before anything downstream reads
    # the sequences or builds metadata.
    records, seq_df = apply_v12_sequence_source_overrides(records, seq_df, args.gene_symbol)
    seq_df.to_csv(outdir / "sequence_retrieval.tsv", sep="\t", index=False)
    ok_count = int((seq_df["status"] == "ok").sum()) if not seq_df.empty else 0
    finish_step(step_started, f"Sequence retrieval written with {ok_count} successful proteins")
    reference_species = args.reference_species or args.source_species

    step_number += 1
    step_started = start_step(step_number, "Enriching protein metadata and tree nomenclature")
    protein_metadata_df, protein_features_df, protein_xrefs_df, tree_nomenclature_payload = collect_protein_metadata_bundle(
        seq_df=seq_df,
        ortholog_df=ortholog_df,
        source_species=args.source_species,
        gene_symbol=args.gene_symbol,
        reference_species=reference_species,
        metadata_enrichment_mode=getattr(args, "metadata_enrichment_mode", "auto"),
        tree_nomenclature_source=getattr(args, "tree_nomenclature_source", "auto"),
    )
    seq_df = merge_sequence_retrieval_metadata(seq_df, protein_metadata_df)
    seq_df.to_csv(outdir / "sequence_retrieval.tsv", sep="\t", index=False)
    protein_metadata_df.to_csv(outdir / PROTEIN_METADATA_FILENAME, sep="\t", index=False)
    protein_features_df.to_csv(outdir / PROTEIN_FEATURES_FILENAME, sep="\t", index=False)
    protein_xrefs_df.to_csv(outdir / PROTEIN_XREFS_FILENAME, sep="\t", index=False)
    write_json_file(outdir / TREE_NOMENCLATURE_FILENAME, tree_nomenclature_payload)
    finish_step(
        step_started,
        f"Protein metadata tables written with {len(protein_metadata_df)} proteins, "
        f"{len(protein_features_df)} features, and {len(protein_xrefs_df)} cross-references",
    )

    if getattr(args, "max_length_deviation", 0):
        before_len_filter = len(records)
        whitelist_arg = getattr(args, "length_filter_keep_species", None)
        whitelist = list(whitelist_arg) if whitelist_arg else list(DEFAULT_LENGTH_FILTER_KEEP_SPECIES)
        records, length_filter_report_df = filter_records_by_reference_length(
            records,
            seq_df,
            reference_species=reference_species,
            source_species=args.source_species,
            max_deviation=getattr(args, "max_length_deviation", 30),
            keep_species_whitelist=whitelist,
        )
        length_filter_report_df.to_csv(outdir / "length_filter_report.csv", index=False)
        whitelist_kept = int((length_filter_report_df.get("length_filter_status") == "kept_length_whitelist").sum()) if "length_filter_status" in length_filter_report_df.columns else 0
        emit_log(
            f"Length filter retained {len(records)}/{before_len_filter} proteins "
            f"within reference length +/- {getattr(args, 'max_length_deviation', 30)} aa "
            f"(whitelist rescued {whitelist_kept})."
        )

    if len(records) < args.min_sequences:
        emit_log(
            f"Stopping because only {len(records)} sequences were recovered; "
            f"the pipeline requires at least {args.min_sequences}."
        )
        raise RuntimeError(
            f"Only {len(records)} protein sequences were recovered. "
            f"Need at least {args.min_sequences} for this pipeline."
        )

    step_number += 1
    step_started = start_step(step_number, "Aligning protein sequences")
    proteins_fasta = outdir / "proteins.fasta"
    aligned_fasta = outdir / "aligned.fasta"
    alignment_method = run_alignment(
        records,
        proteins_fasta,
        aligned_fasta,
        alignment_method=args.alignment_method,
        mafft_exe=mafft_exe,
        muscle_exe=muscle_exe,
        accurate=args.accurate_mafft,
        source_species=args.source_species,
        gene_symbol=args.gene_symbol,
        target_species=args.target_species,
        filter_vertebrates=args.filter_vertebrates,
        ortholog_df=ortholog_df,
    )
    alignment = AlignIO.read(str(aligned_fasta), "fasta")
    finish_step(
        step_started,
        f"Alignment completed with {len(records)} sequences, {alignment.get_alignment_length()} columns, method={alignment_method}",
    )

    step_number += 1
    step_started = start_step(step_number, "Projecting alignment to the reference sequence")
    reference_projected_alignment, projected_ref_record = project_alignment_to_reference(
        alignment,
        reference_species=reference_species,
    )
    projected_fasta = outdir / "aligned_reference_projected.fasta"
    AlignIO.write(reference_projected_alignment, str(projected_fasta), "fasta")
    finish_step(
        step_started,
        f"Reference projection complete with {reference_projected_alignment.get_alignment_length()} columns",
    )

    step_number += 1
    step_started = start_step(step_number, "Exporting full-alignment visuals")
    write_pretty_alignment_text(
        reference_projected_alignment,
        outdir / "alignment_reference_projected_pretty.txt",
        reference_species=reference_species,
        residues_per_line=getattr(args, "alignment_residues_per_line", 120),
    )
    if not getattr(args, "figure_only", False):
        plot_alignment_blocks_pdf(
            reference_projected_alignment,
            outdir / "alignment_reference_projected_blocks.pdf",
            reference_species=reference_species,
            residues_per_line=getattr(args, "alignment_residues_per_line", 120),
            blocks_per_page=getattr(args, "alignment_blocks_per_page", 4),
        )
        plot_alignment_colored_pages(
            reference_projected_alignment,
            outdir / "alignment_reference_projected_colored.pdf",
            reference_species=reference_species,
            residues_per_line=max(50, min(100, getattr(args, "alignment_residues_per_line", 120))),
            blocks_per_page=max(3, getattr(args, "alignment_blocks_per_page", 4) + 1),
        )
    finish_step(
        step_started,
        "Reference-projected alignment text, PDF, and PNG pages exported",
    )

    step_number += 1
    step_started = start_step(step_number, "Exporting pairwise reference reports")
    export_pairwise_reference_reports(
        reference_projected_alignment,
        outdir,
        reference_species=reference_species,
        residues_per_line=min(80, getattr(args, "alignment_residues_per_line", 120)),
    )
    finish_step(
        step_started,
        f"Pairwise reference reports exported for {len(reference_projected_alignment) - 1} sequences",
    )

    treefile = None
    tree_method_used = None
    if getattr(args, "run_phylogeny", False):
        step_number += 1
        step_started = start_step(step_number, "Building the phylogeny")
        if requested_tree_method == "python_nj":
            treefile = run_python_nj_tree(aligned_fasta, outdir=outdir)
            tree_method_used = "python_neighbor_joining"
        elif iqtree_exe:
            treefile = run_iqtree(aligned_fasta, outdir=outdir, iqtree_exe=iqtree_exe, bootstrap=args.bootstrap)
            tree_method_used = "iqtree_mfp"
        else:
            emit_log("IQ-TREE is unavailable; falling back to the built-in Python NJ tree builder.")
            treefile = run_python_nj_tree(aligned_fasta, outdir=outdir)
            tree_method_used = "python_neighbor_joining"
        finish_step(step_started, f"Phylogeny written with method={tree_method_used}")

    step_number += 1
    step_started = start_step(step_number, "Computing conservation scores and domain annotations")
    if not getattr(args, "figure_only", False):
        alignment_matrix_df = build_alignment_matrix(reference_projected_alignment)
        alignment_matrix_df.to_csv(outdir / "alignment_matrix.csv", index=False)

        full_alignment_matrix_df = build_alignment_matrix(alignment)
        full_alignment_matrix_df.to_csv(outdir / "alignment_matrix_full.csv", index=False)

    conservation_df, ref_record = compute_conservation(reference_projected_alignment, reference_species=reference_species)

    domain_df = collect_domain_annotations(protein_metadata_df, protein_features_df, protein_xrefs_df)
    domain_df.to_csv(outdir / "domains.tsv", sep="\t", index=False)

    ref_species, ref_symbol = parse_header_species_symbol(ref_record.id)
    conservation_df = merge_domain_summary(conservation_df, domain_df, ref_species, ref_symbol)

    for col in [
        "identity_max",
        "reference_identity_fraction",
        "hydrophobicity_conservation",
        "charge_conservation",
        "polarity_conservation",
        "size_conservation",
        "aromaticity_conservation",
    ]:
        conservation_df[f"{col}_smoothed"] = smooth_series(conservation_df[col].tolist(), args.smoothing_window)

    conservation_df.to_csv(outdir / "conservation_per_position.csv", index=False)

    annotated_sites_df = parse_annotated_sites(getattr(args, "annotated_sites", None))
    annotated_site_summary_df = annotate_functional_sites_against_conservation(conservation_df, annotated_sites_df)
    annotated_site_summary_df.to_csv(outdir / "annotated_functional_sites.csv", index=False)
    taxonomy_lookup = build_species_taxonomy_lookup(ortholog_df)

    site_comparison_df = compute_site_comparison_table(
        reference_projected_alignment,
        annotated_sites_df,
        reference_species=reference_species,
        site_window=getattr(args, "site_window", 3),
        site_clade_mya_text=getattr(args, "site_clade_mya", None),
        taxonomy_lookup=taxonomy_lookup,
    )
    site_comparison_df.to_csv(outdir / "annotated_site_clade_comparison.csv", index=False)

    clade_profile_df, clade_spectrum_df, clade_recon_df, clade_diff_df = compute_clade_fourier_conservation(
        reference_projected_alignment,
        reference_species=reference_species,
        smoothing_window=args.smoothing_window,
        fourier_terms=getattr(args, "fourier_terms", 18),
        clade_min_species=getattr(args, "clade_min_species", 1),
        taxonomy_lookup=taxonomy_lookup,
    )
    clade_profile_df.to_csv(outdir / "clade_identity_profiles.csv", index=False)
    clade_spectrum_df.to_csv(outdir / "clade_fourier_spectrum.csv", index=False)
    clade_recon_df.to_csv(outdir / "clade_fourier_lowpass_profiles.csv", index=False)
    clade_diff_df.to_csv(outdir / "clade_difference_from_global.csv", index=False)
    clade_fourier_regions_df = detect_clade_regions_from_profiles(
        clade_profile_df,
        clade_recon_df,
        min_len=getattr(args, "clade_min_region_len", args.min_region_len),
        conserved_threshold=getattr(args, "clade_conserved_threshold", 0.85),
        divergent_threshold=getattr(args, "clade_divergence_delta", 0.20),
    )
    clade_fourier_regions_df.to_csv(outdir / "clade_fourier_conserved_and_divergent_regions.csv", index=False)
    domain_clade_summary_df = summarize_domains_by_clade(clade_profile_df, domain_df, ref_species, ref_symbol)
    if not getattr(args, "figure_only", False):
        domain_clade_summary_df.to_csv(outdir / "domain_clade_conservation_summary.csv", index=False)
    evolutionary_segments_df, evolutionary_segment_metrics_df, evolutionary_alignment_manifest_df = export_evolutionary_segment_analysis(
        alignment=reference_projected_alignment,
        outdir=outdir,
        domain_df=domain_df,
        protein_features_df=protein_features_df,
        annotated_sites_df=annotated_sites_df,
        clade_fourier_regions_df=clade_fourier_regions_df,
        reference_species=ref_species,
        reference_symbol=ref_symbol,
        identity_threshold=args.identity_threshold,
        property_threshold=args.property_threshold,
        site_window=getattr(args, "site_window", 3),
        taxonomy_lookup=taxonomy_lookup,
        protein_metadata_df=protein_metadata_df,
    )
    export_evolutionary_scope_bar_figure(
        alignment=alignment,
        outdir=outdir,
        domain_df=domain_df,
        protein_features_df=protein_features_df,
        clade_fourier_regions_df=clade_fourier_regions_df,
        reference_species=ref_species,
        reference_symbol=ref_symbol,
        identity_threshold=args.identity_threshold,
        property_threshold=args.property_threshold,
        site_window=getattr(args, "site_window", 3),
        alignment_scope="aligned_full",
        scope_label="Full alignment",
        taxonomy_lookup=taxonomy_lookup,
        protein_metadata_df=protein_metadata_df,
    )

    conservation_scan_df = conservation_df[conservation_df["reference_ungapped_position"].notna()].copy()
    conservation_scan_df["reference_ungapped_position"] = conservation_scan_df["reference_ungapped_position"].astype(int)
    conservation_scan_df = conservation_scan_df[[
        "reference_ungapped_position",
        "reference_residue",
        "identity_max",
        "identity_max_smoothed",
        "reference_identity_fraction",
        "reference_identity_fraction_smoothed",
        "hydrophobicity_conservation",
        "hydrophobicity_conservation_smoothed",
        "charge_conservation",
        "charge_conservation_smoothed",
        "polarity_conservation",
        "polarity_conservation_smoothed",
        "size_conservation",
        "size_conservation_smoothed",
        "aromaticity_conservation",
        "aromaticity_conservation_smoothed",
        "occupancy",
        "reference_domain_labels",
    ]].copy()
    conservation_scan_df.to_csv(outdir / "conservation_scan.csv", index=False)
    if getattr(args, "export_pymol", False):
        write_pymol_coloring_script(conservation_df, outdir / "human_reference_conservation_coloring.pml", object_name="human_ref")

    property_cols = [c for c in conservation_df.columns if c.endswith("_conservation") and not c.endswith("_smoothed")]
    property_df = conservation_df[
        ["alignment_position", "reference_ungapped_position", "reference_residue"] + property_cols
    ].copy()
    property_df.to_csv(outdir / "property_conservation.csv", index=False)

    region_tables = []
    region_targets = [
        ("identity_max", args.identity_threshold),
        ("reference_identity_fraction", args.reference_identity_threshold),
        ("hydrophobicity_conservation", args.property_threshold),
        ("charge_conservation", args.property_threshold),
        ("polarity_conservation", args.property_threshold),
        ("size_conservation", args.property_threshold),
        ("aromaticity_conservation", args.property_threshold),
    ]
    for col, threshold in region_targets:
        reg = detect_conserved_regions(conservation_df, col, threshold=threshold, min_len=args.min_region_len)
        if not reg.empty:
            region_tables.append(reg)

    conserved_regions = pd.concat(region_tables, ignore_index=True) if region_tables else pd.DataFrame()
    conserved_regions.to_csv(outdir / "conserved_regions.csv", index=False)
    finish_step(
        step_started,
        "Conservation analysis produced "
        f"{len(conservation_df)} positions, {len(domain_df)} domain rows, {len(conserved_regions)} conserved regions, "
        f"{len(evolutionary_segments_df)} evolutionary segments, {len(evolutionary_alignment_manifest_df)} alignment-window excerpts, "
        "and scope-aware divergence bar figures",
    )

    step_number += 1
    step_started = start_step(step_number, "Preparing the human AlphaFold bundle")
    protein_metadata_df, protein_xrefs_df, alphafold_bundle_payload, selected_consensus_chunk_map_df = prepare_human_reference_alphafold_bundle(
        outdir=outdir,
        protein_metadata_df=protein_metadata_df,
        protein_xrefs_df=protein_xrefs_df,
        conservation_df=conservation_df,
        reference_species=reference_species,
        alphafold_mode=getattr(args, "alphafold_mode", "auto"),
        consensus_chunk_table=getattr(args, "consensus_chunk_table", None),
    )
    seq_df = merge_sequence_retrieval_metadata(seq_df, protein_metadata_df)
    seq_df.to_csv(outdir / "sequence_retrieval.tsv", sep="\t", index=False)
    # V9.8c: enrich protein_metadata_df with UniProt accessions discovered via
    # xref:{ensembl_protein_id} lookup so the per-species AlphaFold fetcher
    # below has accessions to fetch for non-human species (Ensembl ortholog
    # retrieval leaves uniprot_accession blank for most species).
    protein_metadata_df = enrich_protein_metadata_with_uniprot_accessions(
        protein_metadata_df,
        max_workers=4,
    )
    protein_metadata_df.to_csv(outdir / PROTEIN_METADATA_FILENAME, sep="\t", index=False)
    protein_xrefs_df.to_csv(outdir / PROTEIN_XREFS_FILENAME, sep="\t", index=False)
    # V9.8c: fetch each species' own AlphaFold PDB so the comparative SS bundle
    # reflects per-species structure (real helix boundaries) rather than the
    # reference-projection fallback. V9.7 silently relied on a pre-populated
    # cache; this populates it explicitly.
    reference_accession = (
        alphafold_bundle_payload.get("uniprot_accession")
        if isinstance(alphafold_bundle_payload, dict) else None
    )
    comparative_af_fetch_summary = fetch_comparative_alphafold_models(
        outdir=outdir,
        protein_metadata_df=protein_metadata_df,
        reference_accession=reference_accession,
        max_workers=4,
    )
    # V9.8c durability: copy the human reference PDB into the comparative cache
    # and fetch override-targeted accessions (e.g. P50392 for the danio_rerio
    # override) so the SS bundle does NOT fall back to projection for the human
    # or danio rows on a fresh pipeline run. Without this both rows look
    # identical, which is the recurring "danio == human SS" symptom.
    reference_model_filename = (
        alphafold_bundle_payload.get("model_filename")
        if isinstance(alphafold_bundle_payload, dict) else None
    )
    comparative_af_stage_summary = stage_reference_and_override_alphafold_models(
        outdir=outdir,
        reference_accession=reference_accession,
        reference_model_filename=reference_model_filename,
    )
    comparative_ss_payload = write_comparative_alphafold_secondary_structure_bundle(
        outdir=outdir,
        reference_species=reference_species,
    )
    alphafold_step_message = (
        "AlphaFold bundle prepared"
        if alphafold_bundle_payload.get("available")
        else f"AlphaFold bundle skipped: {alphafold_bundle_payload.get('reason') or 'not available'}"
    )
    if comparative_ss_payload.get("available"):
        alphafold_step_message += (
            f"; comparative AF SS maps built for {len(comparative_ss_payload.get('records') or [])} aligned proteins"
            f" (per-species fetched={comparative_af_fetch_summary.get('fetched')}, "
            f"cached={comparative_af_fetch_summary.get('cached')}, "
            f"skipped={comparative_af_fetch_summary.get('skipped')}, "
            f"errored={comparative_af_fetch_summary.get('errored')}; "
            f"reference_copied={comparative_af_stage_summary.get('copied_reference')}, "
            f"overrides_fetched={comparative_af_stage_summary.get('fetched_overrides')}, "
            f"overrides_cached={comparative_af_stage_summary.get('cached_overrides')})"
        )
    else:
        alphafold_step_message += (
            f"; comparative AF SS skipped: {comparative_ss_payload.get('reason') or 'not available'}"
        )
    finish_step(
        step_started,
        alphafold_step_message,
    )

    # ------------------------------------------------------------------ #
    # V11 new step: representative property comparison vs human         #
    # ------------------------------------------------------------------ #
    step_number += 1
    step_started = start_step(
        step_number,
        "Picking clade representatives and computing net-charge / aromaticity tracks",
    )
    v11_property_window = getattr(args, "v11_property_window", V11_DEFAULT_PROPERTY_WINDOW)
    v11_mandatory_focus = tuple(getattr(args, "v11_focus_species", V11_MANDATORY_FOCUS_SPECIES) or V11_MANDATORY_FOCUS_SPECIES)
    v11_summary = v11_write_representative_comparison_outputs(
        outdir=outdir,
        reference_projected_alignment=reference_projected_alignment,
        reference_species=reference_species,
        taxonomy_lookup=taxonomy_lookup,
        smoothing_window=v11_property_window,
        always_include=v11_mandatory_focus,
    )
    rep_count = v11_summary.get("representative_count", 0)
    focused_ss = v11_summary.get("focused_ss_payload_summary") or {}
    focused_ss_count = focused_ss.get("focused_species_count")
    msg_parts = [f"V11 representative comparison written: {rep_count} representative species"]
    if focused_ss_count is not None:
        msg_parts.append(f"focused SS bundle covers {focused_ss_count} species")
    msg_parts.append(f"smoothing_window={v11_summary.get('smoothing_window')}aa")
    finish_step(step_started, "; ".join(msg_parts))

    # ------------------------------------------------------------------ #
    # V11 new step: motif evolution + lineage stabilization              #
    # ------------------------------------------------------------------ #
    step_number += 1
    step_started = start_step(
        step_number,
        "Computing motif evolution and lineage stabilization",
    )
    try:
        reps_df_for_motifs = pd.read_csv(outdir / V11_DEFAULT_REPRESENTATIVE_CSV, sep="\t")
    except Exception:
        reps_df_for_motifs = None
    v11_motif_summary = v11_write_motif_analysis_outputs(
        outdir=outdir,
        alignment=reference_projected_alignment,
        reference_species=reference_species,
        taxonomy_lookup=taxonomy_lookup,
        annotated_motifs_text=getattr(args, "annotated_motifs", None),
        extra_motif_regex_text=getattr(args, "v11_extra_motif_regex", None),
        ancestral_clades=tuple(getattr(args, "v11_stabilization_ancestral_clades", V11_DEFAULT_ANCESTRAL_CLADES) or V11_DEFAULT_ANCESTRAL_CLADES),
        derived_clades=tuple(getattr(args, "v11_stabilization_derived_clades", V11_DEFAULT_DERIVED_CLADES) or V11_DEFAULT_DERIVED_CLADES),
        representatives_df=reps_df_for_motifs,
    )
    motif_msg = (
        f"V11 motif analysis written: {v11_motif_summary.get('motif_total', 0)} motifs "
        f"(user={v11_motif_summary.get('user_motif_count', 0)}, library={v11_motif_summary.get('library_motif_count', 0)}); "
        f"stabilization={v11_motif_summary.get('stabilization_rows', 0)} positions "
        f"(ancestral={'|'.join(v11_motif_summary.get('ancestral_clades', []))}, "
        f"derived={'|'.join(v11_motif_summary.get('derived_clades', []))}); "
        f"figures={v11_motif_summary.get('motif_figure_count', 0)}"
    )
    finish_step(step_started, motif_msg)

    # V11 default per-clade consolidated summary: consensus AlphaFold SS +
    # mean net charge + domain architecture for every represented clade.
    try:
        v11_clade_summary = v11_write_clade_consolidated_outputs(
            outdir=outdir,
            alignment=reference_projected_alignment,
            reference_species=reference_species,
            gene_label=args.gene_symbol,
            taxonomy_lookup=taxonomy_lookup,
            smoothing_window=v11_property_window,
        )
        emit_log(
            f"V11 per-clade consolidated summary: {v11_clade_summary.get('clade_count', 0)} clades, "
            f"{v11_clade_summary.get('per_clade_ss_rows', 0)} SS rows, "
            f"{v11_clade_summary.get('domain_span_count', 0)} domain spans."
        )
    except Exception as exc:
        emit_log(f"V11 per-clade consolidated summary failed: {exc}")

    # V11 interactive 3D structure overlay: paint per-clade identity onto the
    # human reference AlphaFold model (3Dmol.js viewer).
    try:
        v11_overlay = v11_write_structure_overlay(
            outdir=outdir,
            alignment=reference_projected_alignment,
            reference_species=reference_species,
            gene_label=args.gene_symbol,
            taxonomy_lookup=taxonomy_lookup,
        )
        if v11_overlay.get("available"):
            emit_log(f"V11 structure overlay written: {v11_overlay.get('html_path')}")
        else:
            emit_log(f"V11 structure overlay skipped: {v11_overlay.get('reason')}")
    except Exception as exc:
        emit_log(f"V11 structure overlay failed: {exc}")

    step_number += 1
    step_started = start_step(step_number, "Rendering figures and writing summary files")
    if treefile:
        plot_tree_svg(
            treefile,
            outdir / "phylo_tree.svg",
            title=f"{args.gene_symbol} protein phylogeny",
            clade_mya=parse_site_clade_mya(getattr(args, "site_clade_mya", None)),
        )
        plot_tree_nomenclature_svg(
            treefile,
            outdir / NOMENCLATURE_TREE_SVG_FILENAME,
            title=f"{args.gene_symbol} protein phylogeny (nomenclature view)",
            tree_nomenclature=tree_nomenclature_payload,
            clade_mya=parse_site_clade_mya(getattr(args, "site_clade_mya", None)),
        )
        # V11 paper-quality tree: clade-colored tips, representatives bolded.
        v11_reps_path = outdir / V11_DEFAULT_REPRESENTATIVE_CSV
        v11_reps_df = pd.read_csv(v11_reps_path, sep="\t") if v11_reps_path.exists() else None
        try:
            v11_plot_paper_quality_tree_svg(
                treefile,
                outdir / "v11_phylo_tree_paper_quality.svg",
                representatives_df=v11_reps_df,
                taxonomy_lookup=taxonomy_lookup,
                title=f"{args.gene_symbol} protein phylogeny — paper-quality (V11)",
                show_bootstrap_threshold=70.0,
            )
        except Exception as exc:
            emit_log(f"V11 paper-quality tree render failed: {exc}")
    if not getattr(args, "figure_only", False):
        plot_conservation_svg(
            conservation_df,
            outdir / "exact_conservation.svg",
            title=f"{args.gene_symbol} exact conservation",
            score_columns=["identity_max", "reference_identity_fraction"],
        )
    plot_property_scan_svg(conservation_df, outdir / "property_conservation.svg")
    plot_reference_conservation_scan_svg(
        conservation_df,
        outdir / "conservation_scan.svg",
        title=f"{args.gene_symbol} reference-mapped conservation scan",
    )
    if not getattr(args, "figure_only", False):
        plot_heatmap_svg(
            conservation_df,
            outdir / "property_heatmap.svg",
            title=f"{args.gene_symbol} property conservation heatmap",
            score_columns=[
                "hydrophobicity_conservation",
                "charge_conservation",
                "polarity_conservation",
                "size_conservation",
                "aromaticity_conservation",
            ],
        )
    plot_reference_architecture_svg(
        conservation_df,
        domain_df,
        conserved_regions,
        projected_ref_record,
        outdir / "reference_domain_architecture.svg",
        title=f"{args.gene_symbol} human-reference domain architecture and conserved sites",
        identity_threshold=args.reference_identity_threshold,
        annotated_sites_df=annotated_sites_df,
    )
    if 'site_comparison_df' in locals() and site_comparison_df is not None and not site_comparison_df.empty:
        plot_site_comparison_svg(
            site_comparison_df,
            outdir / "annotated_site_clade_comparison.svg",
            title=f"{args.gene_symbol} annotated-site clade comparison with approximate divergence times",
        )
    if 'clade_profile_df' in locals() and clade_profile_df is not None and not clade_profile_df.empty:
        plot_clade_fourier_profiles(
            clade_profile_df, clade_recon_df,
            outdir / "clade_fourier_conservation_profiles.svg",
            title=f"{args.gene_symbol} clade identity-to-human profiles, FFT low-pass smoothed",
        )
        plot_clade_difference_heatmap(
            clade_diff_df,
            outdir / "clade_difference_from_global_heatmap.svg",
            title=f"{args.gene_symbol} clade-specific conservation differences versus global profile",
        )
        plot_fourier_spectrum(
            clade_spectrum_df,
            outdir / "clade_fourier_spectrum.svg",
            title=f"{args.gene_symbol} Fourier spectrum of clade conservation profiles",
        )
    if ('domain_clade_summary_df' in locals() and domain_clade_summary_df is not None
            and not domain_clade_summary_df.empty and not getattr(args, "figure_only", False)):
        plot_domain_clade_heatmap(
            domain_clade_summary_df,
            outdir / "domain_clade_conservation_heatmap.svg",
            title=f"{args.gene_symbol} domain-level conservation by clade",
        )

    write_summary_text(
        outdir / "run_summary.txt",
        gene_symbol=args.gene_symbol,
        source_species=args.source_species,
        seq_df=seq_df,
        conserved_regions=conserved_regions,
        alignment_method=alignment_method,
        tree_method=tree_method_used,
        tree_built=bool(treefile),
        full_alignment_length=alignment.get_alignment_length(),
        reference_projected_length=reference_projected_alignment.get_alignment_length(),
        selection_mode=getattr(args, "selection_mode", "all_filtered"),
        manual_species_patterns=getattr(args, "manual_species_patterns", None),
        annotated_sites_text=getattr(args, "annotated_sites", None),
        fasta_object_metadata_path=getattr(args, "fasta_object_metadata_path", None),
        site_window=getattr(args, "site_window", None),
        site_clade_mya_text=getattr(args, "site_clade_mya", None),
        metadata_enrichment_mode=getattr(args, "metadata_enrichment_mode", None),
        tree_nomenclature_source=getattr(args, "tree_nomenclature_source", None),
        alphafold_mode=getattr(args, "alphafold_mode", None),
        consensus_chunk_table=getattr(args, "consensus_chunk_table", None),
    )
    finish_step(step_started, "Final figures and summary text written")

    step_number += 1
    step_started = start_step(step_number, "Archiving outputs and building the interactive report")
    archive_result = export_output_archive(outdir)
    finish_step(
        step_started,
        f"SQLite archive updated at {archive_result['database_path']} and "
        f"interactive report written to {archive_result['html_path']}",
    )

    # V11 functional-divergence pilot. Best-effort: any missing optional
    # package (umap / openmm / prody / scipy) only triggers a per-sub-script
    # skip — the pilot still produces V11_summary.html with whatever
    # artefacts succeeded. Disabled by --skip_v11_pilot.
    if not getattr(args, "skip_v11_pilot", False):
        try:
            import subprocess
            pilot_path = Path(__file__).resolve().parent / "_v11_pilot.py"
            if pilot_path.exists():
                emit_log("Running V11 functional-divergence pilot...")
                subprocess.run(
                    [sys.executable, str(pilot_path), str(outdir)],
                    cwd=str(pilot_path.parent),
                )
        except Exception as exc:  # noqa: BLE001
            emit_log(f"V11 pilot warning (continuing): {exc}")

    update_status("Finished")
    emit_log(f"Pipeline completed in {format_elapsed(time.time() - pipeline_started)}.")
    emit_log(f"SQLite archive: {Path(DEFAULT_SQLITE_ARCHIVE_PATH).resolve()}")
    emit_log(f"Interactive report: {(outdir / INTERACTIVE_REPORT_FILENAME).resolve()}")
    emit_log(f"Done. Outputs written to: {outdir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve ortholog proteins, build a phylogeny, and scan exact/property conservation."
    )
    parser.add_argument("gene_symbol", help="Gene symbol, e.g. DHRS7 or TP53")
    parser.add_argument("--source_species", default="homo_sapiens", help="Ensembl species name for the query gene")
    parser.add_argument("--reference_species", default=None, help="Reference species for residue numbering and reference-based scoring")
    parser.add_argument("--target_species", nargs="*", default=None, help="Optional Ensembl target species list")
    parser.add_argument("--outdir", default=None, help="Output directory")
    parser.add_argument("--filter_vertebrates", action="store_true", help="Retain only likely vertebrate homologs")
    parser.add_argument(
        "--selection_mode",
        default="all_filtered",
        choices=list(SELECTION_MODE_CHOICES),
        help="Use all filtered orthologues or a curated subset spanning major phyla / model organisms",
    )
    parser.add_argument("--alignment_method", default="auto", choices=["auto", "mafft", "muscle", "ensembl", "python"], help="Alignment backend to use")
    parser.add_argument("--mafft_exe", default=None, help="Path to MAFFT executable if not on PATH")
    parser.add_argument("--muscle_exe", default=None, help="Path to MUSCLE executable if not on PATH")
    parser.add_argument("--iqtree_exe", default=None, help="Path to IQ-TREE executable if not on PATH")
    parser.add_argument("--run_phylogeny", action="store_true", help="Build a phylogenetic tree with the selected tree backend")
    parser.add_argument("--figure_only", action="store_true",
                        help="Skip out-of-scope heavy exports not needed for the species-snapshot figure "
                             "(full alignment matrices, block/colored PDF+PNG renders, exact-conservation "
                             "and property/domain-clade heatmaps). Conservation scan, property conservation, "
                             "domain architecture, pairwise reports and the alignment browser are still written.")
    parser.add_argument(
        "--tree_method",
        default="auto",
        choices=list(TREE_METHOD_CHOICES),
        help="Phylogeny backend: auto prefers IQ-TREE and falls back to Python neighbor-joining",
    )
    parser.add_argument("--bootstrap", type=int, default=1000, help="IQ-TREE ultrafast bootstrap / SH-aLRT replicates")
    parser.add_argument("--accurate_mafft", action="store_true", help="Use MAFFT L-INS-i-like settings")
    parser.add_argument("--min_sequences", type=int, default=3, help="Minimum recovered protein sequences required")
    parser.add_argument("--identity_threshold", type=float, default=0.85, help="Threshold for identity_max region calls")
    parser.add_argument("--reference_identity_threshold", type=float, default=0.85, help="Threshold for reference_identity_fraction region calls")
    parser.add_argument("--property_threshold", type=float, default=0.90, help="Threshold for property-based region calls")
    parser.add_argument("--min_region_len", type=int, default=6, help="Minimum contiguous alignment length for a conserved region")
    parser.add_argument("--smoothing_window", type=int, default=9, help="Window size for smoothed conservation tracks")
    parser.add_argument("--alignment_residues_per_line", type=int, default=120, help="Residues per displayed line in the alignment block export")
    parser.add_argument("--alignment_blocks_per_page", type=int, default=4, help="Number of alignment blocks per PDF page")
    parser.add_argument("--manual_species_patterns", nargs="*", default=None, help="Optional manual species/header substring filter applied after ortholog recovery")
    parser.add_argument("--gene_label", default=None, help="Optional gene label or alias used for output labeling and metadata matching")
    parser.add_argument("--fasta_object_metadata_path", default=None, help="Optional path to a FASTA object text file with richer per-species gene labels/descriptions")
    parser.add_argument(
        "--metadata_enrichment_mode",
        default="auto",
        choices=list(METADATA_ENRICHMENT_CHOICES),
        help="Protein metadata enrichment mode: auto fetches remote annotations when possible, local disables them, required treats missing remote metadata as fatal",
    )
    parser.add_argument(
        "--tree_nomenclature_source",
        default="auto",
        choices=list(TREE_NOMENCLATURE_SOURCE_CHOICES),
        help="Tree nomenclature metadata source: auto prefers Ensembl and falls back locally, local disables Ensembl tree naming, ensembl requires the richer remote tree metadata",
    )
    parser.add_argument(
        "--alphafold_mode",
        default="auto",
        choices=list(ALPHAFOLD_MODE_CHOICES),
        help="Human-reference AlphaFold bundle mode: auto fetches it when possible, off disables it, required treats missing AlphaFold metadata as fatal",
    )
    parser.add_argument(
        "--consensus_chunk_table",
        default=None,
        help="Optional TSV/CSV table of selected consensus chunks to overlay onto the human AlphaFold model",
    )
    parser.add_argument("--annotated_sites", default=None, help="Comma-separated annotated human-reference sites, e.g. 228:Ser catalytic,505:phosphosite")
    parser.add_argument("--site_window", type=int, default=3, help="Residues on each side of an annotated site for clade motif comparison")
    parser.add_argument("--site_clade_mya", default=None, help="Optional clade divergence overrides, e.g. tetrapods:420,teleosts:320 or JSON dict")
    parser.add_argument("--max_length_deviation", type=int, default=30, help="Reject recovered proteins whose length differs from the reference protein by more than this many amino acids; use 0 to disable")
    parser.add_argument(
        "--length_filter_keep_species",
        nargs="*",
        default=None,
        help=(
            "Species names that always pass the length filter even when their "
            "Ensembl gene model is truncated. Defaults to "
            f"{', '.join(DEFAULT_LENGTH_FILTER_KEEP_SPECIES)} (lamprey's PLA2G4A "
            "model is 302 aa vs human 749 aa). Pass an empty list to disable."
        ),
    )
    parser.add_argument("--fourier_terms", type=int, default=18, help="Number of low-frequency Fourier terms retained for smoothed clade conservation profiles")
    parser.add_argument("--clade_min_species", type=int, default=1, help="Minimum species per clade required for clade conservation/Fourier output")
    parser.add_argument("--clade_min_region_len", type=int, default=8, help="Minimum length for clade conserved/divergent regions from FFT-smoothed profiles")
    parser.add_argument("--clade_conserved_threshold", type=float, default=0.85, help="Clade identity-to-human threshold used for clade conserved region calls")
    parser.add_argument("--clade_divergence_delta", type=float, default=0.20, help="Minimum negative difference from the global identity profile for clade-specific divergent region calls")
    parser.add_argument("--export_pymol", action="store_true", help="Write a PyMOL coloring script using human-reference conservation")
    parser.add_argument(
        "--skip_v11_pilot",
        action="store_true",
        help=(
            "V11: skip the functional-divergence pilot at end of pipeline "
            "(catalytic integrity, localization signals, ANM, UMAP, radial tree, MD proxy). "
            "Pilot otherwise runs by default and writes V11_summary.html + V11_*.csv into outdir."
        ),
    )
    # V11: representative property comparison knobs.
    parser.add_argument(
        "--v11_property_window",
        type=int,
        default=V11_DEFAULT_PROPERTY_WINDOW,
        help="V11: sliding-window size (residues) for the per-species net-charge and aromaticity tracks (default 5).",
    )
    parser.add_argument(
        "--v11_focus_species",
        nargs="*",
        default=list(V11_MANDATORY_FOCUS_SPECIES),
        help=(
            "V11: species that should always be retained in the focused comparative view "
            "(net-charge, aromaticity, SS) even if they are not the most-conserved representative "
            "of their broad clade. Default: homo_sapiens danio_rerio."
        ),
    )
    # V11 motif & lineage-stabilization flags.
    parser.add_argument(
        "--annotated_motifs",
        dest="annotated_motifs",
        default=None,
        help=(
            "V11: comma-separated motif ranges to investigate at the reference "
            "(human) ungapped coordinates, e.g. '263-269:cPLA2_PL_rich_263_269,500-505:label_b'. "
            "No function is assumed; the label is whatever you write. Pure inspection."
        ),
    )
    parser.add_argument(
        "--v11_extra_motif_regex",
        default=None,
        help=(
            "V11: extra regex patterns to add to the curated motif library, "
            "comma-separated as 'name1:regex1,name2:regex2'. Applied to the upper-case "
            "ungapped human reference sequence."
        ),
    )
    parser.add_argument(
        "--v11_stabilization_ancestral_clades",
        nargs="*",
        default=None,
        help=(
            "V11: broad clades treated as the 'ancestral' bucket when computing "
            "the lineage stabilization score H_ancestral - H_derived. "
            "Default: Cyclostomata Tunicata Chondrichthyes."
        ),
    )
    parser.add_argument(
        "--v11_stabilization_derived_clades",
        nargs="*",
        default=None,
        help=(
            "V11: broad clades treated as the 'derived' bucket for the "
            "stabilization score. Default: Mammalia Aves Reptilia."
        ),
    )
    # --- V11.1 Functional Divergence & Regulatable Signal Module --------- #
    parser.add_argument(
        "--subgroup_a",
        default=None,
        help=(
            "V11.1: comma-separated species list for subgroup A in the "
            "Subgroup-Discriminating Position (SDP) analyzer "
            "(e.g. 'homo_sapiens,danio_rerio'). Pair with --subgroup_b."
        ),
    )
    parser.add_argument(
        "--subgroup_b",
        default=None,
        help="V11.1: comma-separated species list for subgroup B.",
    )
    parser.add_argument(
        "--subgroup_label_a",
        default="A",
        help="V11.1: short label for subgroup A, used in output filenames.",
    )
    parser.add_argument(
        "--subgroup_label_b",
        default="B",
        help="V11.1: short label for subgroup B, used in output filenames.",
    )
    parser.add_argument(
        "--diff_species",
        default=None,
        help=(
            "V11.1: pair of species 'A,B' for the pairwise diff report "
            "(e.g. 'mus_musculus,homo_sapiens'). Emits v11_species_diff_<A>_vs_<B>.csv "
            "+ HTML."
        ),
    )
    parser.add_argument(
        "--v11_detect_idr",
        action="store_true",
        help="V11.1: emit per-species IDR predictions (Wootton-Federhen+composition).",
    )
    parser.add_argument(
        "--v11_detect_pocket",
        action="store_true",
        help="V11.1: emit AlphaFold substrate-pocket residues CSV (default ON when active site is known).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.outdir is None:
        tag = sanitize_filename(f"{args.gene_symbol}_{args.source_species}_phylo")
        args.outdir = tag

    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
