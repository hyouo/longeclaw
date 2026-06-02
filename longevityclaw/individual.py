"""
Individual interpretation engine.

Given a single person's CpG beta values and a population reference,
this module computes:

1. Per-CpG deviation from population (z-score, percentile)
2. Clock-weighted impact: coefficient * (individual - population_mean)
   This is the SHAP-like attribution -- how much does THIS person's value
   at THIS CpG push their biological age up or down vs the average person.
3. Age-stratified comparison (vs same-age peers, not just global population)
4. Aggregated gene-level and hallmark-level interpretation

The key insight: a CpG with a large clock coefficient matters, but only if
this person's value DEVIATES from the population. coeff * deviation = personal impact.
"""

import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from .clock_db import get_db, ClockModel
from .annotations import get_gene_hallmarks, HALLMARKS, GENE_TO_HALLMARKS

# Progress callback for real-time status updates
_progress_cb = None

def set_progress_callback(cb):
    global _progress_cb
    _progress_cb = cb


class PopulationReference:
    """Pre-computed population statistics for CpG beta values."""

    def __init__(self):
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.percentiles: dict[str, np.ndarray] = {}  # sorted values for percentile lookup
        self.n_samples: int = 0
        self.age_bins: dict[str, dict] = {}  # age_bin -> {feature -> {mean, std}}
        self.tissue_bins: dict[str, dict] = {}  # tissue -> {feature -> {mean, std}}
        self._loaded = False

    def load_from_tsv(
        self,
        features_path: str | Path,
        meta_path: str | Path | None = None,
        cpg_mapping_path: str | Path | None = None,
    ):
        """Load population stats from the prepared_353 dataset."""
        features = pd.read_csv(features_path, sep="\t")
        feat_cols = [c for c in features.columns if c != "sample_id"]

        # If we have a CpG mapping, translate short names to CpG IDs
        cpg_map = None
        if cpg_mapping_path and Path(cpg_mapping_path).exists():
            cpg_map = pd.read_csv(cpg_mapping_path, sep="\t")
            short_to_cpg = dict(zip(cpg_map["short_name"], cpg_map["cpg_id"]))
        else:
            short_to_cpg = None

        # Compute global stats
        for col in feat_cols:
            cpg_id = short_to_cpg[col] if short_to_cpg and col in short_to_cpg else col
            vals = features[col].dropna().values
            self.means[cpg_id] = float(np.mean(vals))
            self.stds[cpg_id] = float(np.std(vals))
            self.percentiles[cpg_id] = np.sort(vals)

        self.n_samples = len(features)

        # Compute age-stratified stats if metadata available
        if meta_path and Path(meta_path).exists():
            meta = pd.read_csv(meta_path, sep="\t")
            merged = meta[["sample_id", "age"]].merge(features, on="sample_id")

            bins = [(0, 20, "0-20"), (20, 40, "20-40"), (40, 60, "40-60"),
                    (60, 80, "60-80"), (80, 120, "80+")]

            for lo, hi, label in bins:
                subset = merged[(merged["age"] >= lo) & (merged["age"] < hi)]
                if len(subset) < 10:
                    continue
                bin_stats = {}
                for col in feat_cols:
                    cpg_id = short_to_cpg[col] if short_to_cpg and col in short_to_cpg else col
                    vals = subset[col].dropna().values
                    bin_stats[cpg_id] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "n": len(vals),
                    }
                self.age_bins[label] = bin_stats

        self._loaded = True
        return self

    def get_percentile(self, cpg_id: str, value: float) -> float | None:
        """Get percentile of a value within the population distribution."""
        if cpg_id not in self.percentiles:
            return None
        sorted_vals = self.percentiles[cpg_id]
        return float(np.searchsorted(sorted_vals, value) / len(sorted_vals) * 100)

    def get_zscore(self, cpg_id: str, value: float) -> float | None:
        """Get z-score of a value relative to population."""
        if cpg_id not in self.means or cpg_id not in self.stds:
            return None
        std = self.stds[cpg_id]
        if std < 1e-10:
            return 0.0
        return float((value - self.means[cpg_id]) / std)

    def load_from_npz(self, pop_dir: str | Path):
        """Load from pre-computed .npz files. Works for methylation, transcriptomic, and proteomics refs."""
        pop_dir = Path(pop_dir)

        # Load feature index — try various index files
        feature_list = []
        for index_name in ("gene_index.tsv", "cpg_index.tsv", "protein_index.tsv"):
            index_path = pop_dir / index_name
            if index_path.exists():
                with open(index_path) as f:
                    next(f)  # skip header
                    for line in f:
                        parts = line.strip().split("\t")
                        feature_list.append(parts[1])
                break

        if not feature_list:
            raise FileNotFoundError(f"No feature index found in {pop_dir}")

        # Load stats
        stats = np.load(pop_dir / "population_stats.npz")
        means = stats["means"]
        stds = stats["stds"]
        self.n_samples = int(stats["n_samples"][0])

        for i, fid in enumerate(feature_list):
            self.means[fid] = float(means[i])
            self.stds[fid] = float(stds[i])

        # Load sorted values for exact percentile lookup (methylation only)
        sorted_path = pop_dir / "population_sorted_betas.npz"
        if sorted_path.exists():
            sorted_data = np.load(sorted_path)
            sorted_betas = sorted_data["sorted_betas"]
            for i, fid in enumerate(feature_list):
                self.percentiles[fid] = sorted_betas[:, i]

        # Auto-discover age bins from npz keys (bin_*_means)
        import re
        bin_keys = sorted(k for k in stats.keys() if k.startswith("bin_") and k.endswith("_means"))
        for key_m in bin_keys:
            # Extract label: bin_0_20_means -> 0_20, bin_70plus_means -> 70plus
            label = key_m[4:-6]  # strip "bin_" and "_means"
            key_s = f"bin_{label}_stds"
            key_n = f"bin_{label}_n"

            if key_s not in stats:
                continue

            # Convert to display label: 0_20 -> 0-20, 70plus -> 70+
            display = label.replace("plus", "+").replace("_", "-", 1)

            bin_means = stats[key_m]
            bin_stds = stats[key_s]
            n = int(stats[key_n][0]) if key_n in stats else 0
            bin_stats = {}
            for i, fid in enumerate(feature_list):
                bin_stats[fid] = {
                    "mean": float(bin_means[i]),
                    "std": float(bin_stds[i]),
                    "n": n,
                }
            self.age_bins[display] = bin_stats

        # Load tissue-binned stats (transcriptomic)
        tissue_keys = sorted(k for k in stats.keys() if k.startswith("tissue_") and k.endswith("_means"))
        for key_m in tissue_keys:
            tissue_label = key_m[7:-6]  # strip "tissue_" and "_means"
            key_s = f"tissue_{tissue_label}_stds"
            key_n = f"tissue_{tissue_label}_n"
            if key_s not in stats:
                continue
            t_means = stats[key_m]
            t_stds = stats[key_s]
            n = int(stats[key_n][0]) if key_n in stats else 0
            tissue_stats = {}
            for i, fid in enumerate(feature_list):
                tissue_stats[fid] = {
                    "mean": float(t_means[i]),
                    "std": float(t_stds[i]),
                    "n": n,
                }
            self.tissue_bins[tissue_label] = tissue_stats

        self._loaded = True
        return self

    def get_age_bin(self, age: float) -> str | None:
        """Get the age bin label for a given age."""
        for label in self.age_bins:
            parts = label.replace("+", "-120").split("-")
            lo, hi = float(parts[0]), float(parts[1])
            if lo <= age < hi:
                return label
        return None


def compute_individual_profile(
    beta_values: dict[str, float],
    chronological_age: float,
    population: PopulationReference,
    clock_names: list[str] | None = None,
    top_n: int = 30,
) -> dict:
    """Compute a full individual interpretation profile.

    For each CpG used by applicable clocks:
      - z-score and percentile vs global population
      - z-score vs age-matched peers
      - clock-weighted impact: coeff * (value - pop_mean)
        This tells us: "this CpG is pushing your biological age X years
        higher/lower than the average person"

    Returns a rich interpretation dict with:
      - per_cpg_impacts: ranked list of CpGs by personal impact
      - gene_level_impacts: aggregated by gene
      - hallmark_level_impacts: aggregated by aging hallmark
      - clock_summaries: per-clock age with top personal drivers
    """
    db = get_db()

    # Determine which clocks to analyze
    if clock_names is None:
        applicable = db.find_applicable_clocks(set(beta_values.keys()))
        clock_names = [a["clock"] for a in applicable if a["coverage"] >= 0.95]

    # Age bin for peer comparison
    age_bin = population.get_age_bin(chronological_age)
    peer_stats = population.age_bins.get(age_bin, {}) if age_bin else {}

    # Compute per-CpG impacts across all clocks
    cpg_impacts: dict[str, dict] = {}

    clock_summaries = []

    for i, clock_name in enumerate(clock_names):
        if _progress_cb:
            _progress_cb(f"profiling {clock_name} ({i+1}/{len(clock_names)})")

        model = db.get_clock_model(clock_name)
        if not model:
            continue

        clock_total_impact = 0.0
        clock_cpg_impacts = []

        for fid, coeff in model.coefficients.items():
            if fid not in beta_values:
                continue

            val = beta_values[fid]
            pop_mean = population.means.get(fid)
            pop_std = population.stds.get(fid)

            if pop_mean is None:
                continue

            # Personal impact: how much does this person deviate from average
            deviation = val - pop_mean
            impact = coeff * deviation  # in clock score units

            # Population stats
            zscore = population.get_zscore(fid, val)
            percentile = population.get_percentile(fid, val)

            # Age-peer comparison
            peer = peer_stats.get(fid)
            peer_zscore = None
            if peer and peer["std"] > 1e-10:
                peer_zscore = (val - peer["mean"]) / peer["std"]

            gene = model.gene_map.get(fid, "")

            # Accumulate per-CpG
            if fid not in cpg_impacts:
                cpg_impacts[fid] = {
                    "cpg_id": fid,
                    "gene": gene,
                    "beta_value": round(val, 4),
                    "pop_mean": round(pop_mean, 4),
                    "pop_std": round(pop_std, 4) if pop_std else None,
                    "zscore": round(zscore, 2) if zscore is not None else None,
                    "percentile": round(percentile, 1) if percentile is not None else None,
                    "peer_zscore": round(peer_zscore, 2) if peer_zscore is not None else None,
                    "clocks": {},
                    "total_impact": 0.0,
                    "n_clocks": 0,
                }

            cpg_impacts[fid]["clocks"][clock_name] = {
                "coefficient": round(coeff, 6),
                "impact": round(impact, 6),
                "direction": "ages you" if impact > 0 else "protects you",
            }
            cpg_impacts[fid]["total_impact"] += abs(impact)
            cpg_impacts[fid]["n_clocks"] += 1

            clock_total_impact += impact
            clock_cpg_impacts.append({
                "cpg_id": fid,
                "gene": gene,
                "impact": round(impact, 6),
                "direction": "ages you" if impact > 0 else "protects you",
                "beta": round(val, 4),
                "pop_mean": round(pop_mean, 4),
                "zscore": round(zscore, 2) if zscore is not None else None,
            })

        clock_cpg_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        clock_summaries.append({
            "clock": clock_name,
            "total_personal_deviation": round(clock_total_impact, 4),
            "top_personal_drivers": clock_cpg_impacts[:10],
        })

    # Rank CpGs by total cross-clock impact
    ranked_cpgs = sorted(cpg_impacts.values(), key=lambda x: x["total_impact"], reverse=True)[:top_n]

    # Aggregate by gene
    gene_impacts: dict[str, dict] = {}
    for cpg in ranked_cpgs:
        gene = cpg["gene"]
        if not gene or gene == "NA":
            continue
        if gene not in gene_impacts:
            gene_impacts[gene] = {
                "gene": gene,
                "cpgs": [],
                "total_impact": 0.0,
                "net_direction": 0.0,
                "hallmarks": get_gene_hallmarks(gene),
            }
        gene_impacts[gene]["cpgs"].append(cpg["cpg_id"])
        gene_impacts[gene]["total_impact"] += cpg["total_impact"]
        # Net direction from largest clock
        biggest_clock_impact = max(cpg["clocks"].values(), key=lambda x: abs(x["impact"]))
        gene_impacts[gene]["net_direction"] += biggest_clock_impact["impact"]

    ranked_genes = sorted(gene_impacts.values(), key=lambda x: x["total_impact"], reverse=True)
    for g in ranked_genes:
        g["net_direction_label"] = "ages you" if g["net_direction"] > 0 else "protects you"
        g["total_impact"] = round(g["total_impact"], 4)
        g["net_direction"] = round(g["net_direction"], 6)

    # Aggregate by hallmark
    hallmark_impacts: dict[str, dict] = {}
    for g in ranked_genes:
        for h in g["hallmarks"]:
            hid = h["hallmark_id"]
            if hid not in hallmark_impacts:
                hallmark_impacts[hid] = {
                    "hallmark_id": hid,
                    "name": h["name"],
                    "genes": [],
                    "total_impact": 0.0,
                    "net_direction": 0.0,
                }
            hallmark_impacts[hid]["genes"].append(g["gene"])
            hallmark_impacts[hid]["total_impact"] += g["total_impact"]
            hallmark_impacts[hid]["net_direction"] += g["net_direction"]

    ranked_hallmarks = sorted(hallmark_impacts.values(), key=lambda x: x["total_impact"], reverse=True)
    for h in ranked_hallmarks:
        h["net_direction_label"] = "accelerated" if h["net_direction"] > 0 else "protected"
        h["total_impact"] = round(h["total_impact"], 4)
        h["net_direction"] = round(h["net_direction"], 6)

    return {
        "chronological_age": chronological_age,
        "age_bin": age_bin,
        "n_clocks_analyzed": len(clock_summaries),
        "population_size": population.n_samples,
        "per_cpg_impacts": ranked_cpgs,
        "gene_level_impacts": ranked_genes[:20],
        "hallmark_level_impacts": ranked_hallmarks,
        "clock_summaries": clock_summaries,
        "interpretation_notes": {
            "impact_meaning": "Impact = coefficient * (your_value - population_mean). "
                              "Positive = ages you faster, negative = protects you.",
            "zscore_meaning": "Z-score shows how many standard deviations your value is from the population mean. "
                              ">2 or <-2 is notably unusual.",
            "peer_zscore_meaning": f"Peer z-score compares you to others aged {age_bin} (n={peer_stats.get(list(peer_stats.keys())[0] if peer_stats else '', {}).get('n', '?')})." if peer_stats else "No peer group data available.",
        },
    }


# Module-level singletons for population references
_pop_ref: PopulationReference | None = None
_transcriptomic_pop_ref: PopulationReference | None = None


def get_population_reference() -> PopulationReference:
    """Load methylation population reference. Prefers full 23K CpG npz, falls back to 353 TSV."""
    global _pop_ref
    if _pop_ref is None:
        # Try full population reference first (23K CpGs, 95 clocks)
        full_dir = Path(__file__).resolve().parent.parent / "data" / "population"
        if (full_dir / "population_stats.npz").exists():
            _pop_ref = PopulationReference().load_from_npz(full_dir)
            return _pop_ref

        # Fall back to 353 CpG reference
        data_dir = Path(__file__).resolve().parent.parent / "data" / "prepared_353"
        if not data_dir.exists():
            data_dir = Path(__file__).resolve().parent.parent.parent / "llmregress" / "data" / "prepared_353"
        _pop_ref = PopulationReference().load_from_tsv(
            features_path=data_dir / "methylation_features.tsv",
            meta_path=data_dir / "methylation_meta.tsv",
            cpg_mapping_path=data_dir / "cpg_mapping.tsv",
        )
    return _pop_ref


_proteomics_pop_ref: PopulationReference | None = None


def get_proteomics_population_reference() -> PopulationReference:
    """Load proteomics population reference (Allen Institute, 2.9K proteins, 316 subjects)."""
    global _proteomics_pop_ref
    if _proteomics_pop_ref is None:
        pop_dir = Path(__file__).resolve().parent.parent / "data" / "proteomics_population"
        if not (pop_dir / "population_stats.npz").exists():
            raise FileNotFoundError(
                f"Proteomics population reference not found at {pop_dir}. "
                "Run prepare_proteomics_reference.py first."
            )
        _proteomics_pop_ref = PopulationReference().load_from_npz(pop_dir)
    return _proteomics_pop_ref


def get_transcriptomic_population_reference() -> PopulationReference:
    """Load transcriptomic population reference (GTEx, 12K genes, 12K samples, 11 tissues)."""
    global _transcriptomic_pop_ref
    if _transcriptomic_pop_ref is None:
        pop_dir = Path(__file__).resolve().parent.parent / "data" / "transcriptomic_population"
        if not (pop_dir / "population_stats.npz").exists():
            raise FileNotFoundError(
                f"Transcriptomic population reference not found at {pop_dir}. "
                "Run prepare_transcriptomic_reference.py first."
            )
        _transcriptomic_pop_ref = PopulationReference().load_from_npz(pop_dir)
    return _transcriptomic_pop_ref
