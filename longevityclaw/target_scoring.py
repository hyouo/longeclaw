"""
Target scoring module using 12 Hallmarks of Aging methodology.

Based on PMC9004567 (PandaOmics paper) scoring approach, adapted for L-LLM.
Supports two distinct modes:
1. generation - Ask L-LLM to GENERATE lists of top aging target genes
2. hallmarks - Score genes using 12 Hallmarks of Aging criteria from the paper

Compares results with paper's 9 validated targets using Jaccard similarity.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .llm_client import query_llm

logger = logging.getLogger(__name__)


# Paper's 9 validated targets (PMC9004567)
PAPER_VALIDATED_TARGETS = {
    "CXCL12", "SPP1", "ITGB5", "PPM1A", "RAB7B",
    "ADAMTS14", "KDM7A", "MYSM1", "MTMR4",
}


# 12 Hallmarks of Aging (for scoring mode)
HALLMARKS_PROMPTS: dict[str, dict[str, str]] = {
    "inflammation": {
        "prompt": "Rate the gene {gene} for its role in chronic inflammation and inflammaging. Consider: inflammatory cytokine regulation, NF-κB pathway involvement, senescence-associated secretory phenotype (SASP), and immune system aging. Score 0-10.",
        "direction": "negative",
    },
    "genomic_instability": {
        "prompt": "Rate the gene {gene} for its involvement in genomic instability. Consider: DNA damage repair mechanisms, telomere maintenance, replication fidelity, and chromosomal stability. Score 0-10.",
        "direction": "negative",
    },
    "altered_intercellular_communication": {
        "prompt": "Rate the gene {gene} for its role in altered intercellular communication during aging. Consider: cell signaling dysregulation, hormonal changes, neurohormonal signaling, and paracrine/endocrine communication. Score 0-10.",
        "direction": "negative",
    },
    "mitochondrial_dysfunction": {
        "prompt": "Rate the gene {gene} for its involvement in mitochondrial dysfunction. Consider: oxidative phosphorylation, ROS production, mitochondrial biogenesis, mitophagy, and mtDNA integrity. Score 0-10.",
        "direction": "negative",
    },
    "impaired_proteostasis": {
        "prompt": "Rate the gene {gene} for its role in impaired proteostasis. Consider: protein folding, autophagy, ubiquitin-proteasome system, chaperone function, and protein aggregation. Score 0-10.",
        "direction": "negative",
    },
    "ecm_stiffness": {
        "prompt": "Rate the gene {gene} for its involvement in extracellular matrix stiffness and remodeling. Consider: collagen crosslinking, elastin degradation, matrix metalloproteinases, and tissue stiffening. Score 0-10.",
        "direction": "negative",
    },
    "cellular_senescence": {
        "prompt": "Rate the gene {gene} for its role in cellular senescence. Consider: cell cycle arrest (p16, p21), SASP factors, senolytic sensitivity, and accumulation of senescent cells. Score 0-10.",
        "direction": "negative",
    },
    "deregulated_nutrient_signaling": {
        "prompt": "Rate the gene {gene} for its involvement in deregulated nutrient signaling. Consider: insulin/IGF-1 pathway, mTOR signaling, AMPK activation, sirtuins, and metabolic regulation. Score 0-10.",
        "direction": "mixed",
    },
    "epigenetic_shift": {
        "prompt": "Rate the gene {gene} for its role in epigenetic alterations during aging. Consider: DNA methylation changes, histone modifications, chromatin remodeling, and epigenetic clock associations. Score 0-10.",
        "direction": "negative",
    },
    "stem_cell_exhaustion": {
        "prompt": "Rate the gene {gene} for its involvement in stem cell exhaustion. Consider: stem cell self-renewal, niche signaling, regenerative capacity, and tissue homeostasis maintenance. Score 0-10.",
        "direction": "negative",
    },
    "telomere_attrition": {
        "prompt": "Rate the gene {gene} for its role in telomere attrition. Consider: telomerase activity, shelterin complex, telomere length maintenance, and replicative senescence. Score 0-10.",
        "direction": "negative",
    },
    "retrotranspositions": {
        "prompt": "Rate the gene {gene} for its involvement in retrotransposon activity. Consider: LINE-1 element regulation, transposable element silencing, and genomic stability from mobile elements. Score 0-10.",
        "direction": "negative",
    },
}


# Generation mode prompts - ask L-LLM to GENERATE gene lists
GENERATION_PROMPTS: dict[str, str] = {
    "inflammation": """List the top 30 human genes that are the most promising therapeutic targets for reducing chronic inflammation and inflammaging.
Consider genes involved in: NF-κB pathway, inflammatory cytokines (IL-6, TNF-α, IL-1β), SASP regulation, and immune system aging.
Return ONLY gene symbols, one per line, no explanations.""",

    "genomic_instability": """List the top 30 human genes that are the most promising therapeutic targets for addressing genomic instability in aging.
Consider genes involved in: DNA damage repair (BRCA1/2, ATM, ATR), telomere maintenance, replication fidelity, and chromosomal stability.
Return ONLY gene symbols, one per line, no explanations.""",

    "altered_intercellular_communication": """List the top 30 human genes that are the most promising therapeutic targets for improving intercellular communication in aging.
Consider genes involved in: cell signaling, growth factors, hormonal regulation, and paracrine/endocrine communication.
Return ONLY gene symbols, one per line, no explanations.""",

    "mitochondrial_dysfunction": """List the top 30 human genes that are the most promising therapeutic targets for addressing mitochondrial dysfunction in aging.
Consider genes involved in: oxidative phosphorylation, mitochondrial biogenesis (PGC-1α), mitophagy, and ROS regulation.
Return ONLY gene symbols, one per line, no explanations.""",

    "impaired_proteostasis": """List the top 30 human genes that are the most promising therapeutic targets for improving proteostasis in aging.
Consider genes involved in: autophagy, ubiquitin-proteasome system, chaperones (HSP), and protein quality control.
Return ONLY gene symbols, one per line, no explanations.""",

    "ecm_stiffness": """List the top 30 human genes that are the most promising therapeutic targets for addressing ECM stiffness and aging.
Consider genes involved in: collagen metabolism, elastin, matrix metalloproteinases, and tissue remodeling.
Return ONLY gene symbols, one per line, no explanations.""",

    "cellular_senescence": """List the top 30 human genes that are the most promising therapeutic targets for eliminating senescent cells (senolytics).
Consider genes involved in: cell cycle arrest (p16, p21, p53), SASP factors, anti-apoptotic pathways (BCL-2 family), and senescence markers.
Return ONLY gene symbols, one per line, no explanations.""",

    "deregulated_nutrient_signaling": """List the top 30 human genes that are the most promising therapeutic targets for modulating nutrient signaling in aging.
Consider genes involved in: insulin/IGF-1 pathway, mTOR, AMPK, sirtuins, and caloric restriction mimetics.
Return ONLY gene symbols, one per line, no explanations.""",

    "epigenetic_shift": """List the top 30 human genes that are the most promising therapeutic targets for epigenetic rejuvenation.
Consider genes involved in: DNA methylation (DNMTs, TETs), histone modification, chromatin remodeling, and epigenetic reprogramming.
Return ONLY gene symbols, one per line, no explanations.""",

    "stem_cell_exhaustion": """List the top 30 human genes that are the most promising therapeutic targets for reversing stem cell exhaustion.
Consider genes involved in: stem cell self-renewal, niche signaling, regenerative capacity, and tissue homeostasis.
Return ONLY gene symbols, one per line, no explanations.""",

    "telomere_attrition": """List the top 30 human genes that are the most promising therapeutic targets for addressing telomere attrition.
Consider genes involved in: telomerase (TERT, TERC), shelterin complex, telomere maintenance, and replicative lifespan.
Return ONLY gene symbols, one per line, no explanations.""",

    "retrotranspositions": """List the top 30 human genes that are the most promising therapeutic targets for controlling retrotransposon activity in aging.
Consider genes involved in: LINE-1 silencing, transposable element regulation, and cGAS-STING pathway.
Return ONLY gene symbols, one per line, no explanations.""",

    # Additional general prompts
    "overall_longevity": """List the top 50 human genes that are the most promising therapeutic targets for extending healthy lifespan.
Consider all known longevity pathways, geroscience targets, and genes validated in model organisms or human studies.
Return ONLY gene symbols, one per line, no explanations.""",

    "druggable_aging": """List the top 50 druggable human genes that could be targeted for anti-aging interventions.
Focus on genes with known small molecule modulators, approved drugs that could be repurposed, or clear therapeutic potential.
Return ONLY gene symbols, one per line, no explanations.""",
}


@dataclass(frozen=True)
class HallmarkScore:
    """Score for a single hallmark category."""
    hallmark: str
    score: float
    direction: str
    reasoning: str
    raw_response: str


@dataclass
class GeneHallmarksProfile:
    """Complete hallmarks profile for a gene."""
    gene: str
    mode: str
    scores: list[HallmarkScore] = field(default_factory=list)
    total_score: float = 0.0
    weighted_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def compute_totals(self) -> None:
        """Compute total and weighted scores."""
        if not self.scores:
            return
        self.total_score = sum(s.score for s in self.scores)
        weighted = 0.0
        for s in self.scores:
            if s.direction == "positive":
                weighted += s.score
            elif s.direction == "negative":
                weighted -= s.score
            else:
                weighted += s.score * 0.5
        self.weighted_score = weighted


@dataclass
class GenerationResult:
    """Result of a generation run for one hallmark."""
    hallmark: str
    genes: list[str]
    raw_response: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class GenerationRun:
    """Complete generation run across all hallmarks."""
    run_id: int
    results: list[GenerationResult] = field(default_factory=list)
    all_genes: set[str] = field(default_factory=set)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def collect_genes(self) -> None:
        """Collect all unique genes from all results."""
        self.all_genes = set()
        for r in self.results:
            self.all_genes.update(r.genes)


def _parse_score(response_content: str) -> tuple[float, str]:
    """Parse score and reasoning from LLM response."""
    content = response_content.strip()
    patterns = [
        r"[Ss]core[:\s]*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*/\s*10",
        r"^(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*(?:out of|/)\s*10",
    ]
    score = 5.0
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            try:
                val = float(match.group(1))
                if 0 <= val <= 10:
                    score = val
                    break
            except ValueError:
                pass
    reasoning = content
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            end_pos = match.end()
            remainder = content[end_pos:].strip()
            if remainder:
                reasoning = remainder
            break
    return score, reasoning


def _parse_gene_list(response_content: str, valid_genes: set[str] | None = None) -> list[str]:
    """Parse gene symbols from LLM response."""
    content = response_content.strip()
    genes = []

    # Try line-by-line first
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove common prefixes like "1.", "- ", etc.
        line = re.sub(r"^[\d]+[\.\)]\s*", "", line)
        line = re.sub(r"^[-•*]\s*", "", line)
        # Extract gene symbol (uppercase letters and numbers, 1-15 chars)
        match = re.match(r"^([A-Z][A-Z0-9]{0,14})(?:\s|$|,|-|:)", line)
        if match:
            gene = match.group(1)
            if valid_genes is None or gene in valid_genes:
                genes.append(gene)

    # If we got very few genes, try extracting all uppercase words
    if len(genes) < 5:
        all_caps = re.findall(r"\b([A-Z][A-Z0-9]{1,14})\b", content)
        for gene in all_caps:
            if gene not in genes:
                if valid_genes is None or gene in valid_genes:
                    genes.append(gene)

    return genes


def load_hgnc_genes(path: Path | None = None) -> set[str]:
    """Load HGNC protein-coding gene symbols."""
    if path is None:
        path = Path(__file__).parent.parent / "data" / "hgnc_protein_coding_genes.txt"
    if not path.exists():
        logger.warning(f"HGNC genes file not found: {path}")
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def overlap_coefficient(set1: set[str], set2: set[str]) -> float:
    """Compute overlap coefficient (intersection / min size)."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    min_size = min(len(set1), len(set2))
    return intersection / min_size if min_size > 0 else 0.0


# ============================================================================
# GENERATION MODE: Ask L-LLM to generate gene lists
# ============================================================================

def generate_genes_for_hallmark(
    hallmark: str,
    prompt: str,
    valid_genes: set[str] | None = None,
) -> GenerationResult:
    """Ask L-LLM to generate a list of top genes for a hallmark."""
    system_prompt = """You are an expert in aging biology, longevity research, and drug target discovery.
Your task is to identify the most promising therapeutic target genes for addressing aging hallmarks.
Focus on genes that are:
1. Validated in aging research (human or model organisms)
2. Druggable or have therapeutic potential
3. Specific to the aging hallmark being addressed
Return ONLY valid human gene symbols (HGNC approved symbols), one per line."""

    try:
        response = query_llm(
            prompt,
            system_prompt=system_prompt,
            temperature=0.7,  # Higher temp for diversity across runs
            max_tokens=1024,
        )
        genes = _parse_gene_list(response.content, valid_genes)
        raw_response = response.content
    except Exception as e:
        logger.error(f"Failed to generate genes for {hallmark}: {e}")
        genes = []
        raw_response = f"Error: {e}"

    return GenerationResult(
        hallmark=hallmark,
        genes=genes,
        raw_response=raw_response,
    )


def run_generation_mode(
    n_runs: int = 5,
    valid_genes: set[str] | None = None,
    progress_callback: Any | None = None,
) -> list[GenerationRun]:
    """
    Run generation mode N times, collecting gene lists from L-LLM.

    Args:
        n_runs: Number of generation runs.
        valid_genes: Set of valid HGNC genes to filter results.
        progress_callback: Optional callback(run_id, hallmark, current, total).

    Returns:
        List of GenerationRun objects.
    """
    runs = []
    total_prompts = len(GENERATION_PROMPTS)

    for run_id in range(1, n_runs + 1):
        logger.info(f"Generation run {run_id}/{n_runs}")
        run = GenerationRun(run_id=run_id)

        for i, (hallmark, prompt) in enumerate(GENERATION_PROMPTS.items()):
            if progress_callback:
                progress_callback(run_id, hallmark, i + 1, total_prompts)

            logger.info(f"  Generating genes for {hallmark}...")
            result = generate_genes_for_hallmark(hallmark, prompt, valid_genes)
            run.results.append(result)
            logger.info(f"    Got {len(result.genes)} genes")

        run.collect_genes()
        runs.append(run)
        logger.info(f"Run {run_id} complete: {len(run.all_genes)} unique genes")

    return runs


def analyze_generation_results(
    runs: list[GenerationRun],
    top_n: int = 50,
) -> dict[str, Any]:
    """
    Analyze generation results: find consensus genes, compare with paper targets.

    Args:
        runs: List of GenerationRun objects.
        top_n: Number of top genes to extract based on frequency.

    Returns:
        Analysis dict with gene frequencies, top genes, and paper comparison.
    """
    # Count gene frequencies across all runs
    gene_counter: Counter[str] = Counter()
    for run in runs:
        for result in run.results:
            gene_counter.update(result.genes)

    # Get top genes by frequency
    top_genes = [gene for gene, _ in gene_counter.most_common(top_n)]
    top_genes_set = set(top_genes)

    # Compare with paper targets
    paper_overlap = PAPER_VALIDATED_TARGETS & top_genes_set
    jaccard = jaccard_similarity(top_genes_set, PAPER_VALIDATED_TARGETS)
    overlap_coef = overlap_coefficient(top_genes_set, PAPER_VALIDATED_TARGETS)

    # Per-hallmark analysis
    hallmark_genes: dict[str, Counter[str]] = {}
    for run in runs:
        for result in run.results:
            if result.hallmark not in hallmark_genes:
                hallmark_genes[result.hallmark] = Counter()
            hallmark_genes[result.hallmark].update(result.genes)

    return {
        "total_runs": len(runs),
        "total_unique_genes": len(gene_counter),
        "gene_frequencies": dict(gene_counter.most_common(100)),
        "top_genes": top_genes,
        "paper_targets": list(PAPER_VALIDATED_TARGETS),
        "paper_overlap": list(paper_overlap),
        "paper_overlap_count": len(paper_overlap),
        "jaccard_similarity": jaccard,
        "overlap_coefficient": overlap_coef,
        "hallmark_top_genes": {
            h: [g for g, _ in c.most_common(20)]
            for h, c in hallmark_genes.items()
        },
    }


# ============================================================================
# SCORING MODE: Score genes using 12 Hallmarks prompts
# ============================================================================

def score_gene_hallmark(
    gene: str,
    hallmark: str,
    prompt_template: str,
    direction: str,
) -> HallmarkScore:
    """Score a gene for a specific hallmark."""
    prompt = prompt_template.format(gene=gene)
    system_prompt = """You are an expert in aging biology and longevity research.
Score the gene's relevance to the specified aging hallmark on a scale of 0-10.
Start your response with "Score: X" where X is the numeric score.
Then provide a brief explanation of your reasoning."""

    try:
        response = query_llm(
            prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=512,
        )
        score, reasoning = _parse_score(response.content)
        response_content = response.content
    except Exception as e:
        logger.warning(f"Failed to score {gene} for {hallmark}: {e}")
        score = 5.0
        reasoning = f"Error: {e}"
        response_content = ""

    return HallmarkScore(
        hallmark=hallmark,
        score=score,
        direction=direction,
        reasoning=reasoning,
        raw_response=response_content,
    )


def score_gene(
    gene: str,
    progress_callback: Any | None = None,
) -> GeneHallmarksProfile:
    """Score a gene using 12 Hallmarks of Aging criteria."""
    profile = GeneHallmarksProfile(gene=gene, mode="hallmarks")
    total = len(HALLMARKS_PROMPTS)

    for i, (hallmark, config) in enumerate(HALLMARKS_PROMPTS.items()):
        if progress_callback:
            progress_callback(hallmark, i + 1, total)

        score = score_gene_hallmark(
            gene=gene,
            hallmark=hallmark,
            prompt_template=config["prompt"],
            direction=config["direction"],
        )
        profile.scores.append(score)

    profile.compute_totals()
    return profile


def run_hallmarks_scoring(
    genes: list[str],
    progress_callback: Any | None = None,
) -> list[GeneHallmarksProfile]:
    """
    Run hallmarks scoring for multiple genes.

    Args:
        genes: List of gene symbols to score.
        progress_callback: Optional callback(gene, gene_idx, total_genes, hallmark, h_idx, h_total).

    Returns:
        List of GeneHallmarksProfile sorted by weighted_score (descending).
    """
    profiles = []
    total_genes = len(genes)

    for gene_idx, gene in enumerate(genes):
        logger.info(f"Scoring {gene} ({gene_idx + 1}/{total_genes})")

        def gene_progress(hallmark: str, h_idx: int, h_total: int) -> None:
            if progress_callback:
                progress_callback(gene, gene_idx + 1, total_genes, hallmark, h_idx, h_total)

        profile = score_gene(gene, progress_callback=gene_progress)
        profiles.append(profile)

    profiles.sort(key=lambda p: p.weighted_score, reverse=True)
    return profiles


def analyze_scoring_results(
    profiles: list[GeneHallmarksProfile],
    top_n: int = 50,
) -> dict[str, Any]:
    """
    Analyze scoring results and compare with paper targets.

    Args:
        profiles: Scored gene profiles (should be sorted by weighted_score).
        top_n: Number of top genes to compare.

    Returns:
        Analysis dict with rankings and paper comparison.
    """
    top_genes = [p.gene for p in profiles[:top_n]]
    top_genes_set = set(top_genes)

    paper_overlap = PAPER_VALIDATED_TARGETS & top_genes_set
    jaccard = jaccard_similarity(top_genes_set, PAPER_VALIDATED_TARGETS)
    overlap_coef = overlap_coefficient(top_genes_set, PAPER_VALIDATED_TARGETS)

    # Find ranks of paper targets
    gene_to_rank = {p.gene: i + 1 for i, p in enumerate(profiles)}
    paper_ranks = {
        gene: gene_to_rank.get(gene, -1)
        for gene in PAPER_VALIDATED_TARGETS
    }

    return {
        "total_genes_scored": len(profiles),
        "top_genes": top_genes,
        "paper_targets": list(PAPER_VALIDATED_TARGETS),
        "paper_overlap": list(paper_overlap),
        "paper_overlap_count": len(paper_overlap),
        "jaccard_similarity": jaccard,
        "overlap_coefficient": overlap_coef,
        "paper_target_ranks": paper_ranks,
        "top_10_details": [
            {
                "rank": i + 1,
                "gene": p.gene,
                "weighted_score": p.weighted_score,
                "total_score": p.total_score,
            }
            for i, p in enumerate(profiles[:10])
        ],
    }


# ============================================================================
# COMPARISON AND REPORTING
# ============================================================================

def compare_methods(
    generation_analysis: dict[str, Any],
    scoring_analysis: dict[str, Any],
) -> dict[str, Any]:
    """Compare generation and scoring methods."""
    gen_top = set(generation_analysis["top_genes"])
    score_top = set(scoring_analysis["top_genes"])

    return {
        "generation_vs_scoring": {
            "jaccard": jaccard_similarity(gen_top, score_top),
            "overlap_coef": overlap_coefficient(gen_top, score_top),
            "common_genes": list(gen_top & score_top),
            "only_generation": list(gen_top - score_top),
            "only_scoring": list(score_top - gen_top),
        },
        "vs_paper": {
            "generation_jaccard": generation_analysis["jaccard_similarity"],
            "scoring_jaccard": scoring_analysis["jaccard_similarity"],
            "generation_overlap": generation_analysis["paper_overlap"],
            "scoring_overlap": scoring_analysis["paper_overlap"],
        },
    }


def generate_report(
    generation_analysis: dict[str, Any] | None,
    scoring_analysis: dict[str, Any] | None,
    comparison: dict[str, Any] | None = None,
) -> str:
    """Generate a text report."""
    lines = [
        "=" * 80,
        "L-LLM TARGET SCORING REPORT",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 80,
        "",
        "Paper's 9 Validated Targets (PMC9004567):",
        f"  {', '.join(sorted(PAPER_VALIDATED_TARGETS))}",
        "",
    ]

    if generation_analysis:
        lines.extend([
            "-" * 80,
            "GENERATION MODE (L-LLM generates gene lists)",
            "-" * 80,
            f"Total runs: {generation_analysis['total_runs']}",
            f"Unique genes generated: {generation_analysis['total_unique_genes']}",
            "",
            "Top 20 genes by frequency:",
        ])
        for i, gene in enumerate(generation_analysis["top_genes"][:20], 1):
            freq = generation_analysis["gene_frequencies"].get(gene, 0)
            marker = " <-- PAPER TARGET" if gene in PAPER_VALIDATED_TARGETS else ""
            lines.append(f"  {i:2}. {gene} (freq: {freq}){marker}")

        lines.extend([
            "",
            f"Paper targets in top-50: {generation_analysis['paper_overlap_count']}/9",
            f"  Found: {', '.join(generation_analysis['paper_overlap']) or 'None'}",
            f"  Missing: {', '.join(set(PAPER_VALIDATED_TARGETS) - set(generation_analysis['paper_overlap']))}",
            f"Jaccard similarity with paper: {generation_analysis['jaccard_similarity']:.4f}",
            f"Overlap coefficient: {generation_analysis['overlap_coefficient']:.4f}",
            "",
        ])

    if scoring_analysis:
        lines.extend([
            "-" * 80,
            "SCORING MODE (12 Hallmarks of Aging)",
            "-" * 80,
            f"Total genes scored: {scoring_analysis['total_genes_scored']}",
            "",
            "Top 20 genes by weighted score:",
        ])
        for detail in scoring_analysis.get("top_10_details", [])[:20]:
            gene = detail["gene"]
            marker = " <-- PAPER TARGET" if gene in PAPER_VALIDATED_TARGETS else ""
            lines.append(
                f"  {detail['rank']:2}. {gene} (weighted: {detail['weighted_score']:.2f}){marker}"
            )

        lines.extend([
            "",
            f"Paper targets in top-50: {scoring_analysis['paper_overlap_count']}/9",
            f"  Found: {', '.join(scoring_analysis['paper_overlap']) or 'None'}",
            f"  Missing: {', '.join(set(PAPER_VALIDATED_TARGETS) - set(scoring_analysis['paper_overlap']))}",
            f"Jaccard similarity with paper: {scoring_analysis['jaccard_similarity']:.4f}",
            f"Overlap coefficient: {scoring_analysis['overlap_coefficient']:.4f}",
            "",
            "Paper target ranks:",
        ])
        for gene, rank in sorted(scoring_analysis["paper_target_ranks"].items(), key=lambda x: x[1] if x[1] > 0 else 99999):
            status = f"#{rank}" if rank > 0 else "NOT SCORED"
            lines.append(f"  {gene}: {status}")
        lines.append("")

    if comparison:
        lines.extend([
            "-" * 80,
            "METHOD COMPARISON",
            "-" * 80,
            f"Generation vs Scoring Jaccard: {comparison['generation_vs_scoring']['jaccard']:.4f}",
            f"Common genes in both top-50: {len(comparison['generation_vs_scoring']['common_genes'])}",
            "",
            "Paper similarity comparison:",
            f"  Generation Jaccard: {comparison['vs_paper']['generation_jaccard']:.4f}",
            f"  Scoring Jaccard: {comparison['vs_paper']['scoring_jaccard']:.4f}",
            "",
        ])

    lines.append("=" * 80)
    return "\n".join(lines)


def save_results(
    output_path: Path,
    generation_runs: list[GenerationRun] | None = None,
    generation_analysis: dict[str, Any] | None = None,
    scoring_profiles: list[GeneHallmarksProfile] | None = None,
    scoring_analysis: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> None:
    """Save all results to JSON."""
    data: dict[str, Any] = {
        "generated": datetime.now().isoformat(),
        "paper_targets": list(PAPER_VALIDATED_TARGETS),
    }

    if generation_runs:
        data["generation_runs"] = [
            {
                "run_id": r.run_id,
                "timestamp": r.timestamp,
                "total_genes": len(r.all_genes),
                "results": [
                    {
                        "hallmark": res.hallmark,
                        "genes": res.genes,
                        "gene_count": len(res.genes),
                    }
                    for res in r.results
                ],
            }
            for r in generation_runs
        ]

    if generation_analysis:
        data["generation_analysis"] = generation_analysis

    if scoring_profiles:
        data["scoring_profiles"] = [
            {
                "gene": p.gene,
                "weighted_score": p.weighted_score,
                "total_score": p.total_score,
                "scores": [asdict(s) for s in p.scores],
            }
            for p in scoring_profiles[:100]  # Save top 100
        ]

    if scoring_analysis:
        data["scoring_analysis"] = scoring_analysis

    if comparison:
        data["comparison"] = comparison

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved results to {output_path}")


def main() -> None:
    """CLI entry point for target scoring."""
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Score/generate aging targets using L-LLM"
    )
    parser.add_argument(
        "--mode",
        choices=["generation", "scoring", "both"],
        default="both",
        help="Mode: generation (L-LLM generates gene lists), scoring (score genes), or both",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of generation runs (for generation mode)",
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        default=None,
        help="Genes to score (for scoring mode). If not provided, uses HGNC genes.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of genes to score",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/target_scoring_results.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top genes to compare with paper targets",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load HGNC genes for validation
    hgnc_genes = load_hgnc_genes()
    logger.info(f"Loaded {len(hgnc_genes)} HGNC protein-coding genes")

    generation_runs = None
    generation_analysis = None
    scoring_profiles = None
    scoring_analysis = None
    comparison = None

    # Generation mode
    if args.mode in ("generation", "both"):
        logger.info(f"Running generation mode ({args.n_runs} runs)...")
        generation_runs = run_generation_mode(
            n_runs=args.n_runs,
            valid_genes=hgnc_genes,
        )
        generation_analysis = analyze_generation_results(
            generation_runs,
            top_n=args.top_n,
        )

    # Scoring mode
    if args.mode in ("scoring", "both"):
        if args.genes:
            genes_to_score = args.genes
        else:
            # Use all HGNC genes or limit
            genes_to_score = sorted(hgnc_genes)

        if args.limit:
            genes_to_score = genes_to_score[: args.limit]

        logger.info(f"Running scoring mode on {len(genes_to_score)} genes...")
        scoring_profiles = run_hallmarks_scoring(genes_to_score)
        scoring_analysis = analyze_scoring_results(
            scoring_profiles,
            top_n=args.top_n,
        )

    # Compare methods
    if generation_analysis and scoring_analysis:
        comparison = compare_methods(generation_analysis, scoring_analysis)

    # Generate and print report
    report = generate_report(generation_analysis, scoring_analysis, comparison)
    print(report)

    # Save results
    save_results(
        args.output,
        generation_runs=generation_runs,
        generation_analysis=generation_analysis,
        scoring_profiles=scoring_profiles,
        scoring_analysis=scoring_analysis,
        comparison=comparison,
    )


if __name__ == "__main__":
    main()
