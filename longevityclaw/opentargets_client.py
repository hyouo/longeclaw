"""
OpenTargets GraphQL API client for target novelty validation.

Checks if targets are well-known (already in pharma pipelines) vs novel.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OPENTARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"

# GraphQL query for target info and drug associations
TARGET_QUERY = """
query TargetInfo($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    biotype
    drugAndClinicalCandidates {
      count
      rows {
        drug {
          id
          name
          maximumClinicalStage
        }
        maxClinicalStage
      }
    }
    associatedDiseases {
      count
    }
  }
}
"""

# Query to search by gene symbol
SYMBOL_SEARCH_QUERY = """
query SearchTarget($symbol: String!) {
  search(queryString: $symbol, entityNames: ["target"], page: {size: 5, index: 0}) {
    hits {
      id
      entity
      name
      description
    }
  }
}
"""

# Phase string to numeric mapping
PHASE_MAP = {
    "PHASE_0": 0,
    "PHASE_1": 1,
    "PHASE_1_2": 1,
    "PHASE_2": 2,
    "PHASE_2_3": 2,
    "PHASE_3": 3,
    "PHASE_4": 4,
    "APPROVED": 4,
}


@dataclass(frozen=True)
class TargetNoveltyScore:
    """Novelty assessment for a target gene."""

    gene: str
    ensembl_id: str | None
    approved_name: str | None
    known_drugs_count: int
    max_clinical_phase: int
    associated_diseases_count: int
    tractability_sm: bool  # Small molecule tractable
    tractability_ab: bool  # Antibody tractable
    novelty_tier: str  # "novel", "emerging", "established", "well_known"
    novelty_score: float  # 0-10, higher = more novel
    raw_data: dict[str, Any]


def query_graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """Execute GraphQL query against OpenTargets API."""
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            OPENTARGETS_API,
            json={"query": query, "variables": variables},
        )
        response.raise_for_status()
        return response.json()


def resolve_gene_symbol(symbol: str) -> str | None:
    """Resolve gene symbol to Ensembl ID via OpenTargets search."""
    try:
        result = query_graphql(SYMBOL_SEARCH_QUERY, {"symbol": symbol})
        hits = result.get("data", {}).get("search", {}).get("hits", [])

        for hit in hits:
            if hit.get("entity") == "target":
                # Prefer exact match
                if hit.get("name", "").upper() == symbol.upper():
                    return hit.get("id")

        # Return first target hit if no exact match
        for hit in hits:
            if hit.get("entity") == "target":
                return hit.get("id")

    except Exception as e:
        logger.warning(f"Failed to resolve {symbol}: {e}")

    return None


def get_target_info(ensembl_id: str) -> dict[str, Any] | None:
    """Get target information from OpenTargets."""
    try:
        result = query_graphql(TARGET_QUERY, {"ensemblId": ensembl_id})
        return result.get("data", {}).get("target")
    except Exception as e:
        logger.warning(f"Failed to get target info for {ensembl_id}: {e}")
        return None


def parse_phase(phase_str: str | None) -> int:
    """Convert phase string like 'PHASE_3' to integer."""
    if not phase_str:
        return 0
    return PHASE_MAP.get(phase_str.upper(), 0)


def calculate_novelty_score(
    known_drugs: int,
    max_phase: int,
    diseases: int,
) -> tuple[str, float]:
    """
    Calculate novelty tier and score.

    Novel = Not in any drug pipeline, few disease associations
    Emerging = Early pipeline (phase 1-2) or moderate associations
    Established = Late pipeline (phase 3+) or many associations
    Well-known = Approved drugs, major therapeutic target

    Returns:
        Tuple of (tier, score) where score is 0-10 (higher = more novel)
    """
    if known_drugs == 0 and diseases < 50:
        return "novel", 9.0 + (1.0 - min(diseases, 50) / 50)
    elif known_drugs == 0 and diseases >= 50:
        return "novel", 7.0 + 2.0 * (1.0 - min(diseases, 200) / 200)
    elif max_phase <= 2 and known_drugs <= 3:
        return "emerging", 5.0 + 2.0 * (1.0 - max_phase / 4)
    elif max_phase <= 3 or known_drugs <= 10:
        return "established", 3.0 + 2.0 * (1.0 - min(known_drugs, 10) / 10)
    else:
        return "well_known", max(0.0, 3.0 - known_drugs / 20)


def assess_novelty(gene: str) -> TargetNoveltyScore:
    """
    Assess novelty of a gene as a therapeutic target.

    Args:
        gene: Gene symbol.

    Returns:
        TargetNoveltyScore with novelty assessment.
    """
    # Resolve symbol to Ensembl ID
    ensembl_id = resolve_gene_symbol(gene)

    if not ensembl_id:
        return TargetNoveltyScore(
            gene=gene,
            ensembl_id=None,
            approved_name=None,
            known_drugs_count=0,
            max_clinical_phase=0,
            associated_diseases_count=0,
            tractability_sm=False,
            tractability_ab=False,
            novelty_tier="novel",
            novelty_score=10.0,  # Unknown = potentially novel
            raw_data={"error": "Could not resolve gene symbol"},
        )

    # Get target info
    target = get_target_info(ensembl_id)
    if not target:
        return TargetNoveltyScore(
            gene=gene,
            ensembl_id=ensembl_id,
            approved_name=None,
            known_drugs_count=0,
            max_clinical_phase=0,
            associated_diseases_count=0,
            tractability_sm=False,
            tractability_ab=False,
            novelty_tier="novel",
            novelty_score=9.5,
            raw_data={"ensembl_id": ensembl_id, "error": "No target data"},
        )

    # Extract metrics
    drug_data = target.get("drugAndClinicalCandidates", {})
    drugs_count = drug_data.get("count", 0) or 0
    diseases_count = target.get("associatedDiseases", {}).get("count", 0) or 0

    # Get max clinical phase from drugs
    max_phase = 0
    for row in drug_data.get("rows", []):
        row_phase = parse_phase(row.get("maxClinicalStage"))
        drug_phase = parse_phase(row.get("drug", {}).get("maximumClinicalStage"))
        max_phase = max(max_phase, row_phase, drug_phase)

    # Tractability not available in simplified query
    sm_tract = False
    ab_tract = False

    # Calculate novelty
    tier, score = calculate_novelty_score(drugs_count, max_phase, diseases_count)

    return TargetNoveltyScore(
        gene=gene,
        ensembl_id=ensembl_id,
        approved_name=target.get("approvedName"),
        known_drugs_count=drugs_count,
        max_clinical_phase=max_phase,
        associated_diseases_count=diseases_count,
        tractability_sm=sm_tract,
        tractability_ab=ab_tract,
        novelty_tier=tier,
        novelty_score=score,
        raw_data=target,
    )


def batch_assess_novelty(
    genes: list[str],
    progress_callback: Any = None,
) -> list[TargetNoveltyScore]:
    """
    Assess novelty for a batch of genes.

    Args:
        genes: List of gene symbols.
        progress_callback: Optional callback(current, total, gene).

    Returns:
        List of TargetNoveltyScore sorted by novelty_score descending.
    """
    results = []
    for i, gene in enumerate(genes):
        if progress_callback:
            progress_callback(i + 1, len(genes), gene)
        logger.info(f"Assessing novelty for {gene} ({i + 1}/{len(genes)})")
        results.append(assess_novelty(gene))

    # Sort by novelty score descending
    results.sort(key=lambda x: x.novelty_score, reverse=True)
    return results


def save_novelty_results(
    results: list[TargetNoveltyScore],
    output_path: Path | str | None = None,
) -> Path:
    """Save novelty results to JSON."""
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "target_novelty.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "gene": r.gene,
            "ensembl_id": r.ensembl_id,
            "approved_name": r.approved_name,
            "known_drugs_count": r.known_drugs_count,
            "max_clinical_phase": r.max_clinical_phase,
            "associated_diseases_count": r.associated_diseases_count,
            "tractability_sm": r.tractability_sm,
            "tractability_ab": r.tractability_ab,
            "novelty_tier": r.novelty_tier,
            "novelty_score": r.novelty_score,
        }
        for r in results
    ]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(data)} novelty assessments to {output_path}")
    return output_path


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Test with a few genes
    test_genes = sys.argv[1:] if len(sys.argv) > 1 else ["TP53", "SIRT1", "TERT"]

    for gene in test_genes:
        result = assess_novelty(gene)
        print(f"\n{gene}:")
        print(f"  Ensembl: {result.ensembl_id}")
        print(f"  Known drugs: {result.known_drugs_count}")
        print(f"  Max phase: {result.max_clinical_phase}")
        print(f"  Diseases: {result.associated_diseases_count}")
        print(f"  Novelty: {result.novelty_tier} ({result.novelty_score:.1f}/10)")
