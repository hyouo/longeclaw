"""
Prediction pipeline: ingest user data (CSV/TSV of beta values), run all applicable
clocks, compute feature importance across clocks, and produce a scientific interpretation.
"""

import math
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from .clock_db import get_db, ClockModel
from .annotations import (
    get_gene_hallmarks, map_genes_to_hallmarks,
    GRIMAGE_COMPONENTS, HALLMARKS,
)

# Progress callback for real-time status updates during clock computation
_progress_cb = None

def set_progress_callback(cb):
    global _progress_cb
    _progress_cb = cb


def load_beta_values(path: str | Path) -> dict[str, float]:
    """Load a single sample's CpG beta values from CSV/TSV.

    Expects either:
      - Two columns: cpg_id, beta_value  (no header or with header)
      - Or a wide format with cpg IDs as column headers and one data row
    """
    path = Path(path)
    sep = "\t" if path.suffix in (".tsv", ".txt") else ","

    df = pd.read_csv(path, sep=sep)

    # Case 1: Two-column format (feature_id, value) — works for CpGs, genes, proteins
    if df.shape[1] == 2:
        cols = df.columns.tolist()
        # Check if first row is actually data (no header)
        if cols[0].startswith("cg") or cols[0].startswith("ENSG"):
            # No header -- re-read
            df = pd.read_csv(path, sep=sep, header=None, names=["feature_id", "value"])
        else:
            df.columns = ["feature_id", "value"]
        return dict(zip(df["feature_id"].astype(str), df["value"].astype(float)))

    # Case 2: Wide format (columns are CpG IDs, one or more sample rows)
    cpg_cols = [c for c in df.columns if str(c).startswith("cg")]
    if cpg_cols:
        # Take first data row
        row = df.iloc[0]
        return {c: float(row[c]) for c in cpg_cols if pd.notna(row[c])}

    # Case 3: First column is sample_id, rest are features
    # Check if second column onward look like CpG IDs
    if len(df.columns) > 10:
        potential_cpgs = [c for c in df.columns[1:] if str(c).startswith("cg")]
        if potential_cpgs:
            row = df.iloc[0]
            return {c: float(row[c]) for c in potential_cpgs if pd.notna(row[c])}

    # Case 4: Gene expression or proteomics data (columns are gene symbols / protein IDs)
    non_meta = [c for c in df.columns if c not in ("SAMPLE", "sample_id", "TISSUE", "AGE", "SPLIT")]
    if len(non_meta) > 50:
        row = df.iloc[0]
        return {c: float(row[c]) for c in non_meta if pd.notna(row[c])}

    raise ValueError(
        f"Could not parse values from {path}. Expected either:\n"
        "  - Two columns: feature_id, value\n"
        "  - Wide format with CpG IDs or gene symbols as column headers"
    )


def load_blood_panel(path: str | Path) -> dict[str, float]:
    """Load blood panel from CSV (marker, value) or JSON-like format."""
    path = Path(path)
    sep = "\t" if path.suffix in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep)

    if df.shape[1] == 2:
        df.columns = ["marker", "value"]
        return dict(zip(df["marker"].astype(str), df["value"].astype(float)))

    # Single row with marker names as columns
    return {col: float(df.iloc[0][col]) for col in df.columns if col != "sample_id"}


def _load_ensembl_to_gene() -> dict[str, str]:
    """Load Ensembl -> gene symbol mapping for transcriptomic clocks."""
    mapping_path = Path(__file__).resolve().parent.parent / "data" / "transcriptomic_population" / "ensembl_to_gene.tsv"
    if not mapping_path.exists():
        return {}
    result = {}
    with open(mapping_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


_ensembl_map: dict[str, str] | None = None


def _get_ensembl_map() -> dict[str, str]:
    global _ensembl_map
    if _ensembl_map is None:
        _ensembl_map = _load_ensembl_to_gene()
    return _ensembl_map


def run_all_applicable_clocks(
    beta_values: dict[str, float],
    min_coverage: float = 0.95,
) -> list[dict]:
    """Run all applicable clocks on the provided values (CpG betas or gene expression).

    For transcriptomic clocks using Ensembl IDs (PASTA, REG), automatically maps
    gene symbols to Ensembl IDs using the precomputed mapping.

    Returns list of clock results sorted by coverage, each with:
      - clock_name, description, raw_score, coverage
      - top_contributors (features driving the score)
    """
    db = get_db()
    available = set(beta_values.keys())

    # Build reverse gene->value mapping for Ensembl-keyed clocks
    # (if input uses gene symbols but clock uses ENSG IDs)
    ensembl_map = _get_ensembl_map()
    gene_to_ensembl = {v: k for k, v in ensembl_map.items()}
    has_ensembl_data = any(k.startswith("ENSG") for k in available)

    # If input is gene symbols, expand available set with Ensembl IDs
    ensembl_values: dict[str, float] = {}
    if not has_ensembl_data and gene_to_ensembl:
        for gene_sym, val in beta_values.items():
            eid = gene_to_ensembl.get(gene_sym)
            if eid:
                ensembl_values[eid] = val
        available = available | set(ensembl_values.keys())

    # Merged lookup: original values + ensembl-mapped values
    all_values = {**beta_values, **ensembl_values}

    applicable = db.find_applicable_clocks(available)

    # Filter to clocks meeting coverage threshold
    filtered = [e for e in applicable if e["coverage"] >= min_coverage]

    results = []
    for i, entry in enumerate(filtered):
        clock_name = entry["clock"]

        if _progress_cb:
            _progress_cb(f"{clock_name} ({i+1}/{len(filtered)})")
        model = db.models[clock_name]

        # Compute
        score = model.intercept
        contributions = []
        n_used = 0

        for fid, coeff in model.coefficients.items():
            val = all_values.get(fid)
            if val is not None:
                contrib = coeff * val
                score += contrib
                n_used += 1
                contributions.append({
                    "feature_id": fid,
                    "gene": model.gene_map.get(fid, ""),
                    "beta_value": round(val, 4),
                    "coefficient": round(coeff, 6),
                    "contribution": round(contrib, 6),
                    "abs_contribution": abs(contrib),
                })

        contributions.sort(key=lambda x: x["abs_contribution"], reverse=True)

        result = {
            "clock_name": clock_name,
            "description": entry.get("description", ""),
            "modality": entry["modality"],
            "raw_score": round(score, 6),
            "coverage": entry["coverage"],
            "n_features_used": n_used,
            "n_features_total": entry["n_required"],
            "top_contributors": contributions[:20],
        }

        # Special transforms for known clocks
        if clock_name == "horvath2013":
            if score < 0:
                result["predicted_age"] = round(21 * math.exp(score) - 1, 2)
            else:
                result["predicted_age"] = round(21 * score + 20, 2)
            result["clock_type"] = "chronological_age"

        elif clock_name == "hannum":
            # Hannum is also a direct age predictor (linear)
            result["predicted_age"] = round(score, 2)
            result["clock_type"] = "chronological_age"

        elif "dunedinpace" in clock_name.lower():
            result["pace_of_aging"] = round(score, 4)
            result["clock_type"] = "pace_of_aging"
            result["interpretation"] = (
                f"Aging at {score:.2f} years per calendar year. "
                f"{'Faster' if score > 1.0 else 'Slower'} than average (1.0)."
            )

        elif "phenoage" in clock_name.lower() or "dnamphenoage" in clock_name.lower():
            result["clock_type"] = "phenotypic_age"

        elif "grimage" in clock_name.lower():
            result["clock_type"] = "mortality_risk"

        elif "epitoc" in clock_name.lower() or "stemtoc" in clock_name.lower():
            result["clock_type"] = "mitotic_age"

        elif "damage" in clock_name.lower() or "damag" in clock_name.lower():
            result["clock_type"] = "damage_component"

        elif "adapt" in clock_name.lower():
            result["clock_type"] = "adaptive_component"

        else:
            result["clock_type"] = "age_predictor"

        results.append(result)

    return results


def compute_cross_clock_feature_importance(
    clock_results: list[dict],
    top_n: int = 30,
) -> list[dict]:
    """Aggregate feature importance across all computed clocks.

    A CpG that appears with high weight in many clocks is more robustly
    associated with aging than one that appears in a single clock.
    """
    feature_scores: dict[str, dict] = {}

    for cr in clock_results:
        clock_name = cr["clock_name"]
        for contrib in cr.get("top_contributors", []):
            fid = contrib["feature_id"]
            if fid not in feature_scores:
                feature_scores[fid] = {
                    "feature_id": fid,
                    "gene": contrib.get("gene", ""),
                    "clocks": [],
                    "total_abs_contribution": 0.0,
                    "mean_coefficient": 0.0,
                    "directions": [],
                }
            feature_scores[fid]["clocks"].append(clock_name)
            feature_scores[fid]["total_abs_contribution"] += contrib["abs_contribution"]
            feature_scores[fid]["directions"].append(
                "+" if contrib["coefficient"] > 0 else "-"
            )

    # Compute summary stats
    for fid, fs in feature_scores.items():
        n_clocks = len(fs["clocks"])
        fs["n_clocks"] = n_clocks
        fs["mean_abs_contribution"] = round(fs["total_abs_contribution"] / n_clocks, 6)
        fs["total_abs_contribution"] = round(fs["total_abs_contribution"], 6)
        # Consensus direction
        pos = fs["directions"].count("+")
        neg = fs["directions"].count("-")
        if pos > neg:
            fs["consensus_direction"] = "accelerates aging"
        elif neg > pos:
            fs["consensus_direction"] = "decelerates aging"
        else:
            fs["consensus_direction"] = "mixed"
        fs["direction_agreement"] = round(max(pos, neg) / n_clocks, 2)
        del fs["directions"]

    # Rank by cross-clock importance (n_clocks * mean_abs_contribution)
    ranked = sorted(
        feature_scores.values(),
        key=lambda x: x["n_clocks"] * x["mean_abs_contribution"],
        reverse=True,
    )[:top_n]

    # Add hallmark annotations
    for feat in ranked:
        gene = feat.get("gene", "")
        if gene and gene != "NA":
            feat["hallmarks"] = get_gene_hallmarks(gene)
        else:
            feat["hallmarks"] = []

    return ranked


def interpret_clock_results(
    clock_results: list[dict],
    chronological_age: float | None = None,
) -> dict:
    """Produce a scientific interpretation of multi-clock results.

    Returns:
      - age_summary: predicted ages from each clock type
      - hallmark_profile: which hallmarks are implicated
      - cross_clock_features: features that matter across clocks
      - grimage_components: if GrimAge was computed, component-level interpretation
    """
    db = get_db()

    # 1. Summarize ages by clock type
    age_summary = {
        "chronological_age": chronological_age,
        "clocks": [],
    }
    for cr in clock_results:
        entry = {
            "clock": cr["clock_name"],
            "type": cr.get("clock_type", "unknown"),
            "raw_score": cr["raw_score"],
        }
        if "predicted_age" in cr:
            entry["predicted_age"] = cr["predicted_age"]
            if chronological_age:
                entry["age_gap"] = round(cr["predicted_age"] - chronological_age, 2)
        if "pace_of_aging" in cr:
            entry["pace"] = cr["pace_of_aging"]
        age_summary["clocks"].append(entry)

    # 2. Collect all genes from top contributors across clocks
    all_top_genes = set()
    for cr in clock_results:
        for contrib in cr.get("top_contributors", []):
            gene = contrib.get("gene", "")
            if gene and gene != "NA":
                all_top_genes.add(gene)

    hallmark_profile = map_genes_to_hallmarks(list(all_top_genes))

    # 3. Cross-clock feature importance
    cross_features = compute_cross_clock_feature_importance(clock_results)

    # 4. GrimAge component interpretation
    grimage_interp = []
    for cr in clock_results:
        cn = cr["clock_name"]
        # Match GrimAge sub-models
        for comp_key, comp_info in GRIMAGE_COMPONENTS.items():
            if comp_key.lower() in cn.lower():
                grimage_interp.append({
                    "component": comp_key,
                    "clock_name": cn,
                    "raw_score": cr["raw_score"],
                    **comp_info,
                })

    return {
        "age_summary": age_summary,
        "hallmark_profile": hallmark_profile,
        "cross_clock_features": cross_features[:20],
        "grimage_components": grimage_interp,
        "n_clocks_computed": len(clock_results),
    }
