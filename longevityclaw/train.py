"""
On-the-fly training of hallmark-specific aging models.

Given a set of features (CpGs or genes from an MSigDB hallmark pathway),
trains an ElasticNet regression model predicting chronological age from
population data, then computes SHAP values for interpretability.

Supports three modalities:
  - methylation: 6,599 samples × 23K CpGs (from population_betas.npz)
  - transcriptomics: 12,453 GTEx samples × 12K genes (from feather)
  - proteomics: 316 samples × 2.9K proteins (from population_npx.npz)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from .annotations import HALLMARKS, GENE_TO_HALLMARKS, ALL_HALLMARK_GENES
from .clock_db import get_db

# Progress callback
_progress_cb = None

def set_progress_callback(cb):
    global _progress_cb
    _progress_cb = cb


# ── Data loaders (cached) ────────────────────────────────────────────

_meth_data = None
_trans_data = None
_prot_data = None


def _load_methylation_training_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load methylation betas matrix + ages. Returns (X, y, feature_names)."""
    global _meth_data
    if _meth_data is not None:
        return _meth_data

    pop_dir = Path(__file__).resolve().parent.parent / "data" / "population"

    # Load CpG index
    cpg_list = []
    with open(pop_dir / "cpg_index.tsv") as f:
        next(f)
        for line in f:
            cpg_list.append(line.strip().split("\t")[1])

    # Load betas and ages
    betas_data = np.load(pop_dir / "population_betas.npz")
    X = betas_data["betas"]  # (n_samples, n_cpgs)

    stats = np.load(pop_dir / "population_stats.npz")
    y = stats["ages"]

    _meth_data = (X, y, cpg_list)
    return _meth_data


def _load_transcriptomic_training_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load GTEx gene expression matrix + ages. Returns (X, y, feature_names)."""
    global _trans_data
    if _trans_data is not None:
        return _trans_data

    gtex_path = Path(__file__).resolve().parent.parent / "data" / "transcriptomic_population" / "gtex_training.feather"
    if not gtex_path.exists():
        gtex_path = Path(__file__).resolve().parent.parent.parent / "llmregress" / "data" / "gtex_x_y_full_categorial_toptissues.feather"
    if not gtex_path.exists():
        raise FileNotFoundError("GTEx training data not found. Place gtex_training.feather in data/transcriptomic_population/")

    df = pd.read_feather(str(gtex_path))
    gene_cols = [c for c in df.columns if c not in ("SAMPLE", "TISSUE", "AGE", "SPLIT")]

    X = df[gene_cols].values.astype(np.float32)
    y = df["AGE"].values.astype(np.float32)

    _trans_data = (X, y, gene_cols)
    return _trans_data


def _load_proteomics_training_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load proteomics NPX matrix + ages. Returns (X, y, feature_names)."""
    global _prot_data
    if _prot_data is not None:
        return _prot_data

    pop_dir = Path(__file__).resolve().parent.parent / "data" / "proteomics_population"

    # Load protein index
    protein_list = []
    with open(pop_dir / "protein_index.tsv") as f:
        next(f)
        for line in f:
            protein_list.append(line.strip().split("\t")[1])

    # Load NPX matrix and ages
    npx_data = np.load(pop_dir / "population_npx.npz")
    X = npx_data["npx"]  # (n_samples, n_proteins)

    stats = np.load(pop_dir / "population_stats.npz")
    y = stats["ages"]

    _prot_data = (X, y, protein_list)
    return _prot_data


# ── CpG-to-gene mapping for methylation ──────────────────────────────

_cpg_to_gene: dict[str, str] | None = None


def _get_cpg_to_gene() -> dict[str, str]:
    """Build CpG -> gene symbol mapping from clock database."""
    global _cpg_to_gene
    if _cpg_to_gene is not None:
        return _cpg_to_gene

    db = get_db()
    _cpg_to_gene = {}
    for model in db.models.values():
        for fid, gene in model.gene_map.items():
            if fid.startswith("cg") and gene and gene != "NA":
                _cpg_to_gene[fid] = gene.upper()
    return _cpg_to_gene


# ── Core training function ────────────────────────────────────────────

def train_hallmark_model(
    hallmark_id: str,
    modality: str = "auto",
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    cv_folds: int = 5,
    compute_shap: bool = True,
    tissue: str | None = None,
) -> dict:
    """Train an ElasticNet age-prediction model on features from an MSigDB hallmark.

    Args:
        hallmark_id: MSigDB hallmark pathway ID (e.g. "inflammatory_response")
        modality: "methylation", "transcriptomics", or "auto" (tries both)
        alpha: ElasticNet regularization strength
        l1_ratio: L1/L2 mix (1.0 = Lasso, 0.0 = Ridge)
        cv_folds: Number of cross-validation folds
        compute_shap: Whether to compute SHAP values (slower but interpretable)
        tissue: For transcriptomics, filter to specific tissue (e.g. "Blood")

    Returns:
        Dict with model performance, coefficients, and SHAP analysis.
    """
    if hallmark_id not in HALLMARKS:
        return {"error": f"Unknown hallmark: {hallmark_id}. Available: {list(HALLMARKS.keys())[:10]}..."}

    hallmark_genes = HALLMARKS[hallmark_id]["genes"]
    hallmark_name = HALLMARKS[hallmark_id]["name"]

    results = {}

    if modality in ("auto", "transcriptomics"):
        try:
            r = _train_on_modality("transcriptomics", hallmark_id, hallmark_genes,
                                    alpha, l1_ratio, cv_folds, compute_shap, tissue)
            results["transcriptomics"] = r
        except Exception as e:
            results["transcriptomics"] = {"error": str(e)}

    if modality in ("auto", "proteomics"):
        try:
            r = _train_on_modality("proteomics", hallmark_id, hallmark_genes,
                                    alpha, l1_ratio, cv_folds, compute_shap, None)
            results["proteomics"] = r
        except Exception as e:
            results["proteomics"] = {"error": str(e)}

    if modality in ("auto", "methylation"):
        try:
            r = _train_on_modality("methylation", hallmark_id, hallmark_genes,
                                    alpha, l1_ratio, cv_folds, compute_shap, None)
            results["methylation"] = r
        except Exception as e:
            results["methylation"] = {"error": str(e)}

    return {
        "hallmark_id": hallmark_id,
        "hallmark_name": hallmark_name,
        "n_hallmark_genes": len(hallmark_genes),
        "modality_results": results,
    }


def _train_on_modality(
    modality: str,
    hallmark_id: str,
    hallmark_genes: set[str],
    alpha: float,
    l1_ratio: float,
    cv_folds: int,
    compute_shap: bool,
    tissue: str | None,
) -> dict:
    """Train model for a single modality."""
    if _progress_cb:
        _progress_cb(f"loading {modality} data...")

    if modality == "proteomics":
        X_full, y_full, feature_names = _load_proteomics_training_data()
    elif modality == "transcriptomics":
        X_full, y_full, feature_names = _load_transcriptomic_training_data()
    else:
        X_full, y_full, feature_names = _load_methylation_training_data()

    # Filter tissue for transcriptomics
    if tissue and modality == "transcriptomics":
        gtex_path = Path(__file__).resolve().parent.parent / "data" / "transcriptomic_population" / "gtex_training.feather"
        if not gtex_path.exists():
            gtex_path = Path(__file__).resolve().parent.parent.parent / "llmregress" / "data" / "gtex_x_y_full_categorial_toptissues.feather"
        df = pd.read_feather(str(gtex_path))
        tissue_mask = df["TISSUE"].str.lower() == tissue.lower()
        X_full = X_full[tissue_mask]
        y_full = y_full[tissue_mask]
        if len(y_full) < 50:
            return {"error": f"Too few samples for tissue '{tissue}': {len(y_full)}"}

    # Map hallmark genes to available features
    if modality in ("transcriptomics", "proteomics"):
        # Features ARE gene symbols
        feature_to_idx = {f: i for i, f in enumerate(feature_names)}
        matched = [(g, feature_to_idx[g]) for g in hallmark_genes if g in feature_to_idx]
    else:
        # Features are CpG IDs — need gene mapping
        cpg_to_gene = _get_cpg_to_gene()
        feature_to_idx = {f: i for i, f in enumerate(feature_names)}
        matched = []
        for i, cpg_id in enumerate(feature_names):
            gene = cpg_to_gene.get(cpg_id, "")
            if gene in hallmark_genes and cpg_id in feature_to_idx:
                matched.append((cpg_id, feature_to_idx[cpg_id]))

    if len(matched) < 3:
        return {
            "error": f"Only {len(matched)} features found for this hallmark in {modality} data. Need at least 3.",
            "matched_features": len(matched),
        }

    if _progress_cb:
        _progress_cb(f"{modality}: {len(matched)} features, training...")

    # Build feature matrix
    selected_names = [m[0] for m in matched]
    selected_idx = [m[1] for m in matched]
    X = X_full[:, selected_idx]

    # Handle NaN
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.sum() > 0:
        X = X[~nan_mask]
        y = y_full[~nan_mask]
    else:
        y = y_full

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train ElasticNet
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)

    if _progress_cb:
        _progress_cb(f"{modality}: cross-validating ({cv_folds} folds)...")

    # Cross-validated predictions
    y_pred_cv = cross_val_predict(model, X_scaled, y, cv=cv_folds)
    mae_cv = mean_absolute_error(y, y_pred_cv)
    r2_cv = r2_score(y, y_pred_cv)

    # Fit final model on all data
    model.fit(X_scaled, y)

    # Coefficients
    coef_data = []
    for i, (fname, coef) in enumerate(zip(selected_names, model.coef_)):
        if modality == "methylation":
            gene = _get_cpg_to_gene().get(fname, "")
        else:  # transcriptomics and proteomics: feature IS gene symbol
            gene = fname
        coef_data.append({
            "feature": fname,
            "gene": gene,
            "coefficient": round(float(coef), 6),
            "abs_coefficient": abs(float(coef)),
            "direction": "ages" if coef > 0 else "protects",
        })
    coef_data.sort(key=lambda x: x["abs_coefficient"], reverse=True)

    n_nonzero = sum(1 for c in model.coef_ if abs(c) > 1e-8)

    result = {
        "n_samples": len(y),
        "n_features": len(selected_names),
        "n_nonzero_coefficients": n_nonzero,
        "cv_mae": round(mae_cv, 2),
        "cv_r2": round(r2_cv, 4),
        "intercept": round(float(model.intercept_), 4),
        "top_features": coef_data[:20],
        "all_coefficients": coef_data,
    }

    # SHAP values
    if compute_shap and n_nonzero > 0:
        if _progress_cb:
            _progress_cb(f"{modality}: computing SHAP values...")
        try:
            import shap
            # Use a subsample for speed
            n_bg = min(500, len(X_scaled))
            bg_idx = np.random.RandomState(42).choice(len(X_scaled), n_bg, replace=False)
            explainer = shap.LinearExplainer(model, X_scaled[bg_idx])
            shap_values = explainer.shap_values(X_scaled)

            # Mean absolute SHAP per feature
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            shap_data = []
            for i, (fname, shap_val) in enumerate(zip(selected_names, mean_abs_shap)):
                if modality == "methylation":
                    gene = _get_cpg_to_gene().get(fname, "")
                else:  # transcriptomics and proteomics
                    gene = fname
                shap_data.append({
                    "feature": fname,
                    "gene": gene,
                    "mean_abs_shap": round(float(shap_val), 6),
                })
            shap_data.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
            result["shap_importance"] = shap_data[:20]
        except Exception as e:
            result["shap_error"] = str(e)

    if _progress_cb:
        _progress_cb(f"{modality}: done (MAE={mae_cv:.1f}y, R²={r2_cv:.3f})")

    return result


def train_custom_model(
    feature_list: list[str],
    modality: str = "auto",
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    cv_folds: int = 5,
    compute_shap: bool = True,
    tissue: str | None = None,
) -> dict:
    """Train a model on an arbitrary list of genes/CpGs (not tied to a hallmark)."""
    # Convert gene list to a set for matching
    gene_set = {g.upper() for g in feature_list}

    results = {}

    if modality in ("auto", "transcriptomics"):
        try:
            r = _train_on_modality("transcriptomics", "custom", gene_set,
                                    alpha, l1_ratio, cv_folds, compute_shap, tissue)
            results["transcriptomics"] = r
        except Exception as e:
            results["transcriptomics"] = {"error": str(e)}

    if modality in ("auto", "proteomics"):
        try:
            r = _train_on_modality("proteomics", "custom", gene_set,
                                    alpha, l1_ratio, cv_folds, compute_shap, None)
            results["proteomics"] = r
        except Exception as e:
            results["proteomics"] = {"error": str(e)}

    if modality in ("auto", "methylation"):
        try:
            r = _train_on_modality("methylation", "custom", gene_set,
                                    alpha, l1_ratio, cv_folds, compute_shap, None)
            results["methylation"] = r
        except Exception as e:
            results["methylation"] = {"error": str(e)}

    return {
        "feature_list": feature_list[:20],
        "n_features_requested": len(feature_list),
        "modality_results": results,
    }
