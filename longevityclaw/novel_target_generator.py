"""
Novel Aging Target Generator with 6-Dimension Scoring.

Single-pass generation: Ask L-LLM to generate NOVEL + DRUGGABLE aging targets
with 6-dimension scores directly, emphasizing underexplored targets.

User's 6-dimension framework:
- Confidence (0-100): Strength of evidence
- Druggability (0-100): Feasibility as drug target
- Safety (0-100): Expected safety profile
- Novelty (0-100): How underexplored/novel the target is
- Commercial (0-100): Commercial viability
- Mechanism (0-100): Mechanistic understanding

Score bands: 76-100 High, 51-75 Moderate, 26-50 Emerging, 0-25 Low

Filter criteria for novel druggable targets: Novelty >= 76 AND Druggability >= 51
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .llm_client import query_llm

logger = logging.getLogger(__name__)


# 12 Hallmarks + 2 additional categories
HALLMARK_CATEGORIES = [
    "inflammation",
    "genomic_instability",
    "altered_intercellular_communication",
    "mitochondrial_dysfunction",
    "impaired_proteostasis",
    "ecm_stiffness",
    "cellular_senescence",
    "deregulated_nutrient_signaling",
    "epigenetic_shift",
    "stem_cell_exhaustion",
    "telomere_attrition",
    "retrotranspositions",
    "overall_longevity",
    "druggable_aging",
]


# Hallmark-specific context for the prompts
HALLMARK_CONTEXT = {
    "inflammation": "chronic inflammation, inflammaging, NF-κB pathway, inflammatory cytokines (IL-6, TNF-α, IL-1β), SASP, immune system aging",
    "genomic_instability": "DNA damage repair, telomere maintenance, replication fidelity, chromosomal stability, ATM/ATR pathway",
    "altered_intercellular_communication": "cell signaling dysregulation, hormonal changes, neurohormonal signaling, paracrine/endocrine communication",
    "mitochondrial_dysfunction": "oxidative phosphorylation, ROS production, mitochondrial biogenesis, mitophagy, mtDNA integrity, PGC-1α",
    "impaired_proteostasis": "protein folding, autophagy, ubiquitin-proteasome system, chaperone function, protein aggregation, HSPs",
    "ecm_stiffness": "collagen crosslinking, elastin degradation, matrix metalloproteinases, tissue stiffening, fibrosis",
    "cellular_senescence": "cell cycle arrest (p16, p21), SASP factors, senolytic sensitivity, BCL-2 family",
    "deregulated_nutrient_signaling": "insulin/IGF-1 pathway, mTOR signaling, AMPK activation, sirtuins, caloric restriction mimetics",
    "epigenetic_shift": "DNA methylation changes, histone modifications, chromatin remodeling, epigenetic clock, DNMTs, TETs",
    "stem_cell_exhaustion": "stem cell self-renewal, niche signaling, regenerative capacity, tissue homeostasis",
    "telomere_attrition": "telomerase activity, shelterin complex, telomere length maintenance, TERT, TERC",
    "retrotranspositions": "LINE-1 element regulation, transposable element silencing, cGAS-STING pathway",
    "overall_longevity": "cross-hallmark integration, lifespan extension, healthspan, geroscience validated targets",
    "druggable_aging": "small molecule modulators, approved drugs for repurposing, clear therapeutic potential",
}


@dataclass
class GeneScore:
    """6-dimension score for a single gene."""
    gene: str
    confidence: int
    druggability: int
    safety: int
    novelty: int
    commercial: int
    mechanism: int
    rationale: str = ""
    hallmark: str = ""

    @property
    def is_novel_druggable(self) -> bool:
        """Check if gene passes the novel+druggable filter."""
        return self.novelty >= 76 and self.druggability >= 51

    @property
    def total_score(self) -> int:
        """Sum of all 6 dimensions."""
        return (self.confidence + self.druggability + self.safety +
                self.novelty + self.commercial + self.mechanism)

    @property
    def novelty_druggability_score(self) -> int:
        """Combined novelty + druggability score for ranking."""
        return self.novelty + self.druggability


@dataclass
class HallmarkResult:
    """Result of novel target generation for one hallmark."""
    hallmark: str
    genes: list[GeneScore]
    raw_response: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def novel_druggable_genes(self) -> list[GeneScore]:
        """Genes passing the novel+druggable filter."""
        return [g for g in self.genes if g.is_novel_druggable]


def _build_generation_prompt(hallmark: str) -> str:
    """Build the single-pass generation prompt for a hallmark."""
    context = HALLMARK_CONTEXT.get(hallmark, hallmark)

    return f"""You are an expert in aging biology and drug discovery. Your task is to identify NOVEL and DRUGGABLE therapeutic targets for aging, specifically for: {hallmark}

Context: {context}

CRITICAL REQUIREMENTS:
1. Focus on NOVEL targets - genes that are UNDEREXPLORED in aging research
2. AVOID well-known canonical aging genes like: SIRT1, MTOR, FOXO3, TP53, TERT, IGF1, INS, CDKN2A, LMNA, APOE, KLOTHO, GDF11
3. Look for recently discovered, understudied, or unexpected genes with aging connections
4. Prioritize genes with clear druggability (enzymes, receptors, ion channels, kinases)
5. Consider targets from recent publications (2020-2025) that haven't yet been validated

For each gene, provide 6-dimension scores (0-100 scale):
- Confidence: Strength of evidence for aging role (76-100 High, 51-75 Moderate, 26-50 Emerging, 0-25 Low)
- Druggability: Feasibility as drug target - enzymes/kinases/GPCRs score higher
- Safety: Expected safety profile based on biology and knockdown studies
- Novelty: How UNDEREXPLORED/NOVEL the target is - canonical aging genes score LOW
- Commercial: Commercial potential and competitive landscape
- Mechanism: Depth of mechanistic understanding

Return EXACTLY 30 genes in this JSON format:
```json
[
  {{"gene": "SYMBOL", "confidence": 65, "druggability": 80, "safety": 70, "novelty": 85, "commercial": 60, "mechanism": 55, "rationale": "Brief 1-2 sentence justification for why this is a novel aging target"}}
]
```

IMPORTANT: We want HIGH NOVELTY scores (>=76) for underexplored targets. Well-known aging genes should have LOW novelty scores.
Return ONLY valid JSON array, no other text."""


SYSTEM_PROMPT = """You are a world-class expert in:
1. Aging biology and geroscience
2. Drug target discovery and validation
3. Recent aging research literature (2020-2025)

Your specialty is identifying NOVEL, underexplored therapeutic targets that:
- Have emerging evidence for aging roles
- Are druggable with small molecules or biologics
- Have NOT been extensively studied as aging targets
- Come from unexpected pathways or recent discoveries

You prioritize novelty and druggability over established validation.
You are familiar with the distinction between canonical aging genes (well-known, extensively studied)
and novel emerging targets (recently discovered, understudied, unexpected connections).

Return structured JSON responses only."""


def _parse_gene_scores(response_content: str, hallmark: str) -> list[GeneScore]:
    """Parse gene scores from LLM JSON response."""
    genes = []

    # Try to extract JSON from the response
    content = response_content.strip()

    # Find JSON array in response
    json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
    if not json_match:
        logger.warning(f"No JSON array found in response for {hallmark}")
        return genes

    try:
        data = json.loads(json_match.group())
        for item in data:
            if isinstance(item, dict) and "gene" in item:
                try:
                    gene = GeneScore(
                        gene=item["gene"].upper().strip(),
                        confidence=int(item.get("confidence", 50)),
                        druggability=int(item.get("druggability", 50)),
                        safety=int(item.get("safety", 50)),
                        novelty=int(item.get("novelty", 50)),
                        commercial=int(item.get("commercial", 50)),
                        mechanism=int(item.get("mechanism", 50)),
                        rationale=item.get("rationale", ""),
                        hallmark=hallmark,
                    )
                    genes.append(gene)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse gene entry: {e}")
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error for {hallmark}: {e}")
        # Try line-by-line fallback
        for line in content.split('\n'):
            if '"gene"' in line:
                match = re.search(r'"gene"\s*:\s*"([A-Z][A-Z0-9]+)"', line)
                if match:
                    genes.append(GeneScore(
                        gene=match.group(1),
                        confidence=50, druggability=50, safety=50,
                        novelty=50, commercial=50, mechanism=50,
                        hallmark=hallmark,
                    ))

    return genes


def generate_novel_targets_for_hallmark(hallmark: str) -> HallmarkResult:
    """Generate novel aging targets with 6-dim scores for one hallmark."""
    prompt = _build_generation_prompt(hallmark)

    try:
        response = query_llm(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.7,  # Higher temp for diversity
            max_tokens=4096,
        )
        genes = _parse_gene_scores(response.content, hallmark)
        raw_response = response.content
    except Exception as e:
        logger.error(f"Failed to generate for {hallmark}: {e}")
        genes = []
        raw_response = f"Error: {e}"

    return HallmarkResult(
        hallmark=hallmark,
        genes=genes,
        raw_response=raw_response,
    )


def run_novel_target_generation(
    hallmarks: list[str] | None = None,
    progress_callback: Any | None = None,
    parallel: bool = False,
    max_workers: int = 8,
) -> list[HallmarkResult]:
    """
    Run novel target generation across all hallmarks.

    Args:
        hallmarks: List of hallmarks to process (default: all 14)
        progress_callback: Optional callback(hallmark, current, total)
        parallel: If True, run hallmarks in parallel using ThreadPoolExecutor
        max_workers: Max parallel workers (default: 8)

    Returns:
        List of HallmarkResult objects.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if hallmarks is None:
        hallmarks = HALLMARK_CATEGORIES

    total = len(hallmarks)

    if parallel:
        logger.info(f"Running {total} hallmarks in parallel (max_workers={max_workers})...")
        results = []
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_hallmark = {
                executor.submit(generate_novel_targets_for_hallmark, h): h
                for h in hallmarks
            }

            for future in as_completed(future_to_hallmark):
                hallmark = future_to_hallmark[future]
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    novel_count = len(result.novel_druggable_genes)
                    logger.info(f"  [{completed}/{total}] {hallmark}: {len(result.genes)} genes, {novel_count} novel+druggable")
                    if progress_callback:
                        progress_callback(hallmark, completed, total)
                except Exception as e:
                    logger.error(f"  [{completed}/{total}] {hallmark} failed: {e}")
                    results.append(HallmarkResult(hallmark=hallmark, genes=[], raw_response=f"Error: {e}"))

        return results
    else:
        # Sequential execution
        results = []
        for i, hallmark in enumerate(hallmarks):
            if progress_callback:
                progress_callback(hallmark, i + 1, total)

            logger.info(f"Generating novel targets for {hallmark} ({i+1}/{total})...")
            result = generate_novel_targets_for_hallmark(hallmark)
            results.append(result)

            novel_count = len(result.novel_druggable_genes)
            logger.info(f"  Got {len(result.genes)} genes, {novel_count} pass novel+druggable filter")

        return results


def run_multiple_generations(
    num_runs: int = 3,
    parallel: bool = True,
    max_workers: int = 8,
) -> dict[str, Any]:
    """
    Run multiple generation passes and aggregate results.

    Args:
        num_runs: Number of generation runs
        parallel: If True, run hallmarks in parallel
        max_workers: Max parallel workers per run

    Returns:
        Aggregated analysis dict with gene frequencies.
    """
    all_results: list[list[HallmarkResult]] = []

    for run_idx in range(num_runs):
        logger.info(f"\n{'='*60}")
        logger.info(f"GENERATION RUN {run_idx + 1}/{num_runs}")
        logger.info(f"{'='*60}")
        results = run_novel_target_generation(parallel=parallel, max_workers=max_workers)
        all_results.append(results)

    # Aggregate across runs
    gene_counts: dict[str, int] = {}  # gene -> how many runs it appeared in
    gene_best_score: dict[str, GeneScore] = {}  # gene -> best score across runs

    for run_results in all_results:
        run_genes: set[str] = set()
        for hallmark_result in run_results:
            for gene in hallmark_result.genes:
                run_genes.add(gene.gene)
                if gene.gene not in gene_best_score or gene.novelty_druggability_score > gene_best_score[gene.gene].novelty_druggability_score:
                    gene_best_score[gene.gene] = gene

        for g in run_genes:
            gene_counts[g] = gene_counts.get(g, 0) + 1

    # Filter for genes that appear in multiple runs
    consistent_genes = {g: cnt for g, cnt in gene_counts.items() if cnt >= 2}

    # Sort by frequency * score
    ranked_genes = sorted(
        [(g, cnt, gene_best_score[g]) for g, cnt in gene_counts.items()],
        key=lambda x: (x[1], x[2].novelty_druggability_score),
        reverse=True,
    )

    return {
        "num_runs": num_runs,
        "total_unique_genes": len(gene_counts),
        "genes_in_2plus_runs": len(consistent_genes),
        "top_50_by_frequency": [
            {
                "gene": g,
                "frequency": cnt,
                "novelty": score.novelty,
                "druggability": score.druggability,
                "total": score.total_score,
                "hallmark": score.hallmark,
                "rationale": score.rationale[:100] if score.rationale else "",
            }
            for g, cnt, score in ranked_genes[:50]
        ],
        "all_runs_results": [[
            {
                "hallmark": r.hallmark,
                "gene_count": len(r.genes),
                "novel_druggable_count": len(r.novel_druggable_genes),
                "genes": [
                    {
                        "gene": g.gene,
                        "novelty": g.novelty,
                        "druggability": g.druggability,
                        "confidence": g.confidence,
                        "safety": g.safety,
                        "commercial": g.commercial,
                        "mechanism": g.mechanism,
                        "rationale": g.rationale[:200] if g.rationale else "",
                    }
                    for g in r.genes
                ],
            }
            for r in run_results
        ] for run_results in all_results],
    }


def analyze_generation_results(results: list[HallmarkResult]) -> dict[str, Any]:
    """Analyze results: aggregate genes, find top novel+druggable targets."""

    # Collect all genes with their scores
    all_genes: dict[str, GeneScore] = {}
    hallmark_genes: dict[str, list[str]] = {}

    for result in results:
        hallmark_genes[result.hallmark] = []
        for gene in result.genes:
            # Keep the score with highest novelty+druggability
            if gene.gene not in all_genes or gene.novelty_druggability_score > all_genes[gene.gene].novelty_druggability_score:
                all_genes[gene.gene] = gene
            hallmark_genes[result.hallmark].append(gene.gene)

    # Filter for novel + druggable
    novel_druggable = [g for g in all_genes.values() if g.is_novel_druggable]
    novel_druggable.sort(key=lambda g: g.novelty_druggability_score, reverse=True)

    # All genes sorted by novelty+druggability
    all_sorted = sorted(all_genes.values(), key=lambda g: g.novelty_druggability_score, reverse=True)

    # Paper targets comparison (from target_scoring.py)
    paper_targets = {"CXCL12", "SPP1", "ITGB5", "PPM1A", "RAB7B", "ADAMTS14", "KDM7A", "MYSM1", "MTMR4"}
    all_gene_set = set(all_genes.keys())
    paper_overlap = paper_targets & all_gene_set

    return {
        "total_genes": len(all_genes),
        "novel_druggable_count": len(novel_druggable),
        "hallmark_counts": {h: len(genes) for h, genes in hallmark_genes.items()},
        "top_50_novel_druggable": [
            {
                "gene": g.gene,
                "novelty": g.novelty,
                "druggability": g.druggability,
                "confidence": g.confidence,
                "safety": g.safety,
                "commercial": g.commercial,
                "mechanism": g.mechanism,
                "total": g.total_score,
                "hallmark": g.hallmark,
                "rationale": g.rationale,
            }
            for g in novel_druggable[:50]
        ],
        "all_genes_ranked": [
            {
                "gene": g.gene,
                "novelty": g.novelty,
                "druggability": g.druggability,
                "total": g.total_score,
                "hallmark": g.hallmark,
            }
            for g in all_sorted
        ],
        "paper_targets": list(paper_targets),
        "paper_overlap": list(paper_overlap),
        "paper_overlap_count": len(paper_overlap),
    }


def save_results(
    output_path: Path,
    results: list[HallmarkResult],
    analysis: dict[str, Any],
) -> None:
    """Save results to JSON."""
    data = {
        "generated": datetime.now().isoformat(),
        "method": "novel_target_generator_6dim_single_pass",
        "filter_criteria": "novelty >= 76 AND druggability >= 51",
        "hallmark_results": [
            {
                "hallmark": r.hallmark,
                "timestamp": r.timestamp,
                "genes": [asdict(g) for g in r.genes],
                "novel_druggable_count": len(r.novel_druggable_genes),
            }
            for r in results
        ],
        "analysis": analysis,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved results to {output_path}")


def generate_report(analysis: dict[str, Any]) -> str:
    """Generate text report."""
    lines = [
        "=" * 80,
        "NOVEL AGING TARGETS - 6-DIMENSION SCORING REPORT",
        f"Generated: {datetime.now().isoformat()}",
        "Filter: Novelty >= 76 AND Druggability >= 51",
        "=" * 80,
        "",
        f"Total genes generated: {analysis['total_genes']}",
        f"Genes passing novel+druggable filter: {analysis['novel_druggable_count']}",
        "",
        "Paper targets (PMC9004567):",
        f"  {', '.join(sorted(analysis['paper_targets']))}",
        f"  Overlap: {analysis['paper_overlap_count']}/9 - {', '.join(analysis['paper_overlap']) or 'None'}",
        "",
        "-" * 80,
        "TOP 30 NOVEL + DRUGGABLE TARGETS",
        "-" * 80,
    ]

    for i, gene in enumerate(analysis["top_50_novel_druggable"][:30], 1):
        lines.append(
            f"{i:2}. {gene['gene']:12} | Nov:{gene['novelty']:3} Drug:{gene['druggability']:3} "
            f"Conf:{gene['confidence']:3} Safe:{gene['safety']:3} | Total:{gene['total']:3} | {gene['hallmark']}"
        )
        if gene.get('rationale'):
            lines.append(f"    {gene['rationale'][:100]}...")

    lines.extend([
        "",
        "-" * 80,
        "GENES PER HALLMARK",
        "-" * 80,
    ])

    for hallmark, count in sorted(analysis["hallmark_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"  {hallmark}: {count} genes")

    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate novel aging targets with 6-dimension scoring"
    )
    parser.add_argument(
        "--hallmarks",
        nargs="+",
        default=None,
        help="Specific hallmarks to process (default: all 14)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/novel_targets_6dim.json"),
        help="Output JSON path",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting novel target generation with 6-dimension scoring...")

    results = run_novel_target_generation(hallmarks=args.hallmarks)
    analysis = analyze_generation_results(results)

    report = generate_report(analysis)
    print(report)

    save_results(args.output, results, analysis)

    logger.info(f"\nResults saved to {args.output}")
    logger.info(f"Novel+druggable targets: {analysis['novel_druggable_count']}")


if __name__ == "__main__":
    main()
