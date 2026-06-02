"""
Pathway Generator: LLM-assisted discovery of aging-relevant pathways.

Uses cross-clock coefficient analysis to:
1. Score pathways by aggregate aging clock evidence
2. Rank all MSigDB hallmarks by aging relevance
3. Find synergistic pathway combinations
4. Generate novel pathway hypotheses from seed genes

Integrates with annotations.py (MSigDB hallmarks, Lopez-Otin mapping)
and clock_db.py (233 clocks, 429K coefficients).
"""

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from scipy import stats

from .annotations import (
    AGING_HALLMARK_MAPPING,
    ALL_HALLMARK_GENES,
    GENE_TO_HALLMARKS,
    HALLMARKS,
)
from .clock_db import get_db


@dataclass(frozen=True)
class PathwayEvidence:
    """Evidence summary for a pathway's aging relevance."""

    pathway_id: str
    pathway_name: str
    n_genes: int
    n_genes_in_clocks: int
    coverage: float
    mean_abs_coefficient: float
    coefficient_consistency: float
    clock_count: int
    top_genes: tuple[tuple[str, float], ...]
    lopez_otin_hallmarks: tuple[str, ...]
    aging_score: float


@dataclass
class PathwayCombination:
    """Synergistic pathway combination."""

    pathway_ids: tuple[str, ...]
    shared_genes: frozenset[str]
    combined_coverage: float
    synergy_score: float
    mechanism_hypothesis: str = ""


def get_gene_clock_coefficients(gene: str) -> dict[str, list[float]]:
    """Collect all clock coefficients for a gene across all clocks."""
    db = get_db()
    coefficients: dict[str, list[float]] = {}

    clock_ids = db.find_clocks_by_gene(gene)
    for clock_id in clock_ids:
        model = db.get_clock_model(clock_id)
        if model is None:
            continue
        coef = model.coefficients.get(gene)
        if coef is not None:
            modality = model.modality or "unknown"
            if modality not in coefficients:
                coefficients[modality] = []
            coefficients[modality].append(coef)

    return coefficients


def score_pathway(pathway_id: str) -> PathwayEvidence | None:
    """Score a pathway using cross-clock coefficient analysis."""
    if pathway_id not in HALLMARKS:
        return None

    pathway = HALLMARKS[pathway_id]
    genes = pathway["genes"]
    n_genes = len(genes)

    if n_genes == 0:
        return None

    db = get_db()
    gene_scores: list[tuple[str, float]] = []
    all_coefficients: list[float] = []
    clocks_seen: set[str] = set()

    for gene in genes:
        clock_ids = db.find_clocks_by_gene(gene)
        if not clock_ids:
            continue

        gene_coeffs: list[float] = []
        for clock_id in clock_ids:
            model = db.get_clock_model(clock_id)
            if model is None:
                continue
            coef = model.coefficients.get(gene)
            if coef is not None:
                gene_coeffs.append(coef)
                all_coefficients.append(coef)
                clocks_seen.add(clock_id)

        if gene_coeffs:
            mean_coef = float(np.mean(np.abs(gene_coeffs)))
            gene_scores.append((gene, mean_coef))

    n_genes_in_clocks = len(gene_scores)
    if n_genes_in_clocks == 0:
        return PathwayEvidence(
            pathway_id=pathway_id,
            pathway_name=pathway["name"],
            n_genes=n_genes,
            n_genes_in_clocks=0,
            coverage=0.0,
            mean_abs_coefficient=0.0,
            coefficient_consistency=0.0,
            clock_count=0,
            top_genes=(),
            lopez_otin_hallmarks=(),
            aging_score=0.0,
        )

    coverage = n_genes_in_clocks / n_genes
    mean_abs_coef = float(np.mean(np.abs(all_coefficients)))

    signs = np.sign(all_coefficients)
    consistency = float(np.abs(np.mean(signs)))

    gene_scores.sort(key=lambda x: x[1], reverse=True)
    top_genes = tuple(gene_scores[:10])

    lopez_otin = []
    for hallmark, pathways in AGING_HALLMARK_MAPPING.items():
        if pathway_id in pathways:
            lopez_otin.append(hallmark)

    aging_score = (
        0.4 * coverage
        + 0.3 * min(mean_abs_coef * 10, 1.0)
        + 0.2 * consistency
        + 0.1 * min(len(lopez_otin) / 3, 1.0)
    )

    return PathwayEvidence(
        pathway_id=pathway_id,
        pathway_name=pathway["name"],
        n_genes=n_genes,
        n_genes_in_clocks=n_genes_in_clocks,
        coverage=coverage,
        mean_abs_coefficient=mean_abs_coef,
        coefficient_consistency=consistency,
        clock_count=len(clocks_seen),
        top_genes=top_genes,
        lopez_otin_hallmarks=tuple(lopez_otin),
        aging_score=aging_score,
    )


def rank_all_pathways(min_genes: int = 5) -> list[PathwayEvidence]:
    """Rank all MSigDB hallmark pathways by aging clock evidence."""
    results: list[PathwayEvidence] = []

    for pathway_id in HALLMARKS:
        evidence = score_pathway(pathway_id)
        if evidence is None:
            continue
        if evidence.n_genes < min_genes:
            continue
        results.append(evidence)

    results.sort(key=lambda x: x.aging_score, reverse=True)
    return results


def find_pathway_synergies(
    pathway_ids: list[str], min_shared: int = 3
) -> list[PathwayCombination]:
    """Find synergistic combinations among given pathways."""
    if len(pathway_ids) < 2:
        return []

    combinations: list[PathwayCombination] = []

    for i, pid1 in enumerate(pathway_ids):
        if pid1 not in HALLMARKS:
            continue
        genes1 = HALLMARKS[pid1]["genes"]

        for pid2 in pathway_ids[i + 1 :]:
            if pid2 not in HALLMARKS:
                continue
            genes2 = HALLMARKS[pid2]["genes"]

            shared = frozenset(genes1 & genes2)
            if len(shared) < min_shared:
                continue

            combined = genes1 | genes2
            combined_coverage = _compute_clock_coverage(combined)

            cov1 = _compute_clock_coverage(genes1)
            cov2 = _compute_clock_coverage(genes2)
            expected_combined = cov1 + cov2 - (cov1 * cov2)
            synergy = combined_coverage - expected_combined if expected_combined > 0 else 0.0

            combinations.append(
                PathwayCombination(
                    pathway_ids=(pid1, pid2),
                    shared_genes=shared,
                    combined_coverage=combined_coverage,
                    synergy_score=synergy,
                )
            )

    combinations.sort(key=lambda x: x.synergy_score, reverse=True)
    return combinations


def _compute_clock_coverage(genes: set[str]) -> float:
    """Compute fraction of clocks that include at least one gene from the set."""
    db = get_db()
    clocks_covered: set[str] = set()

    for gene in genes:
        clock_ids = db.find_clocks_by_gene(gene)
        clocks_covered.update(clock_ids)

    total_clocks = len(db.clocks)
    if total_clocks == 0:
        return 0.0
    return len(clocks_covered) / total_clocks


def generate_pathway_hypothesis(
    seed_genes: list[str], expansion_depth: int = 1
) -> dict:
    """Generate a novel pathway hypothesis from seed genes.

    Uses clock co-occurrence to expand seed genes into a coherent module,
    then characterizes the resulting pathway.
    """
    db = get_db()

    seed_set = set(seed_genes)
    expanded = set(seed_genes)

    for _ in range(expansion_depth):
        seed_clocks: set[str] = set()
        for gene in expanded:
            seed_clocks.update(db.find_clocks_by_gene(gene))

        if not seed_clocks:
            break

        candidates: dict[str, int] = {}
        for clock_id in seed_clocks:
            model = db.get_clock_model(clock_id)
            if model is None or len(model.coefficients) > 10000:
                continue
            for other_gene in model.coefficients:
                if other_gene not in expanded:
                    candidates[other_gene] = candidates.get(other_gene, 0) + 1

        if not candidates:
            break

        threshold = max(3, int(len(seed_clocks) * 0.5))
        added = 0
        max_add = 100
        for gene, count in sorted(candidates.items(), key=lambda x: x[1], reverse=True):
            if count >= threshold and added < max_add:
                expanded.add(gene)
                added += 1

    hallmark_counts: dict[str, int] = {}
    for gene in expanded:
        for hallmark_id in GENE_TO_HALLMARKS.get(gene, []):
            hallmark_counts[hallmark_id] = hallmark_counts.get(hallmark_id, 0) + 1

    enriched_hallmarks = sorted(hallmark_counts.items(), key=lambda x: x[1], reverse=True)

    lopez_otin_hits: dict[str, int] = {}
    for hallmark_id, _ in enriched_hallmarks[:5]:
        for aging_hallmark, pathways in AGING_HALLMARK_MAPPING.items():
            if hallmark_id in pathways:
                lopez_otin_hits[aging_hallmark] = lopez_otin_hits.get(aging_hallmark, 0) + 1

    clock_coverage = _compute_clock_coverage(expanded)

    coefficients: list[float] = []
    for gene in expanded:
        clock_ids = db.find_clocks_by_gene(gene)
        for clock_id in clock_ids:
            model = db.get_clock_model(clock_id)
            if model is None:
                continue
            coef = model.coefficients.get(gene)
            if coef is not None:
                coefficients.append(coef)

    direction = "mixed"
    if coefficients:
        mean_sign = np.mean(np.sign(coefficients))
        if mean_sign > 0.5:
            direction = "pro-aging"
        elif mean_sign < -0.5:
            direction = "anti-aging"

    return {
        "seed_genes": seed_genes,
        "expanded_genes": sorted(expanded),
        "n_genes": len(expanded),
        "expansion_ratio": len(expanded) / len(seed_genes) if seed_genes else 0,
        "clock_coverage": clock_coverage,
        "direction": direction,
        "enriched_hallmarks": enriched_hallmarks[:10],
        "lopez_otin_hallmarks": sorted(lopez_otin_hits.items(), key=lambda x: x[1], reverse=True),
        "mean_coefficient": float(np.mean(coefficients)) if coefficients else 0.0,
    }


def discover_aging_modules(
    min_genes: int = 10, min_coverage: float = 0.1
) -> list[dict]:
    """Discover coherent aging-related gene modules from clock data.

    Clusters genes by clock co-occurrence patterns to find
    functionally coherent modules.
    """
    db = get_db()

    gene_clock_matrix: dict[str, set[str]] = {}
    for clock_id in db.clocks:
        model = db.get_clock_model(clock_id)
        if model is None:
            continue
        for gene in model.coefficients:
            if gene not in gene_clock_matrix:
                gene_clock_matrix[gene] = set()
            gene_clock_matrix[gene].add(clock_id)

    genes_with_clocks = [g for g, clocks in gene_clock_matrix.items() if len(clocks) >= 3]

    if len(genes_with_clocks) < min_genes:
        return []

    modules: list[dict] = []
    used_genes: set[str] = set()

    for seed_gene in genes_with_clocks:
        if seed_gene in used_genes:
            continue

        seed_clocks = gene_clock_matrix[seed_gene]
        module_genes = {seed_gene}

        for other_gene in genes_with_clocks:
            if other_gene in used_genes or other_gene == seed_gene:
                continue
            other_clocks = gene_clock_matrix[other_gene]

            overlap = len(seed_clocks & other_clocks)
            union = len(seed_clocks | other_clocks)
            jaccard = overlap / union if union > 0 else 0

            if jaccard > 0.5:
                module_genes.add(other_gene)

        if len(module_genes) >= min_genes:
            coverage = _compute_clock_coverage(module_genes)
            if coverage >= min_coverage:
                hypothesis = generate_pathway_hypothesis(list(module_genes)[:5], expansion_depth=0)

                modules.append({
                    "genes": sorted(module_genes),
                    "n_genes": len(module_genes),
                    "coverage": coverage,
                    "direction": hypothesis["direction"],
                    "enriched_hallmarks": hypothesis["enriched_hallmarks"][:5],
                    "lopez_otin": hypothesis["lopez_otin_hallmarks"][:3],
                })

                used_genes.update(module_genes)

    modules.sort(key=lambda x: x["coverage"], reverse=True)
    return modules[:20]


def get_pathway_intervention_targets(pathway_id: str) -> list[dict]:
    """Get potential intervention targets for a pathway based on clock coefficients."""
    if pathway_id not in HALLMARKS:
        return []

    pathway = HALLMARKS[pathway_id]
    genes = pathway["genes"]
    db = get_db()

    targets: list[dict] = []

    for gene in genes:
        coefficients = get_gene_clock_coefficients(gene)
        if not coefficients:
            continue

        all_coefs: list[float] = []
        for modality_coefs in coefficients.values():
            all_coefs.extend(modality_coefs)

        if not all_coefs:
            continue

        mean_coef = float(np.mean(all_coefs))
        abs_mean = abs(mean_coef)
        consistency = float(np.abs(np.mean(np.sign(all_coefs))))

        if abs_mean > 0.01 and consistency > 0.6:
            targets.append({
                "gene": gene,
                "mean_coefficient": mean_coef,
                "direction": "pro-aging" if mean_coef > 0 else "anti-aging",
                "consistency": consistency,
                "n_clocks": len(all_coefs),
                "modalities": list(coefficients.keys()),
                "intervention": "inhibit" if mean_coef > 0 else "activate",
                "confidence": min(consistency * abs_mean * 10, 1.0),
            })

    targets.sort(key=lambda x: x["confidence"], reverse=True)
    return targets[:20]
