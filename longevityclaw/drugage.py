"""
DrugAge compound scoring: ranks longevity interventions by cross-species efficacy,
consistency, and clock-gene relevance.

DrugAge database: ~1000 compounds tested across multiple species for lifespan effects.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "drugage"


@dataclass
class DrugStudy:
    """Single study entry from DrugAge."""
    compound: str
    species: str
    strain: str
    dosage: str
    avg_lifespan_change: float | None
    max_lifespan_change: float | None
    avg_significance: str
    max_significance: str
    gender: str
    is_itp: bool
    pubmed_id: str


@dataclass
class CompoundScore:
    """Aggregated score for a compound across all studies."""
    compound: str
    n_studies: int
    n_species: int
    species_list: list[str]
    mean_lifespan_change: float
    max_lifespan_change: float
    consistency: float  # fraction of positive studies
    itp_validated: bool
    longevity_score: float  # composite score
    top_studies: list[DrugStudy] = field(default_factory=list)


def _parse_float(val: str) -> float | None:
    """Parse float from CSV, handling empty strings."""
    if not val or val.strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def load_drugage() -> list[DrugStudy]:
    """Load all DrugAge study entries."""
    path = DATA_DIR / "drugage.csv"
    if not path.exists():
        return []

    studies = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            avg_change = _parse_float(row.get("avg_lifespan_change_percent", ""))
            max_change = _parse_float(row.get("max_lifespan_change_percent", ""))

            study = DrugStudy(
                compound=row.get("compound_name", "").strip(),
                species=row.get("species", "").strip(),
                strain=row.get("strain", "").strip(),
                dosage=row.get("dosage", "").strip(),
                avg_lifespan_change=avg_change,
                max_lifespan_change=max_change,
                avg_significance=row.get("avg_lifespan_significance", "").strip(),
                max_significance=row.get("max_lifespan_significance", "").strip(),
                gender=row.get("gender", "").strip(),
                is_itp=row.get("ITP", "").strip().lower() == "yes",
                pubmed_id=row.get("pubmed_id", "").strip(),
            )
            if study.compound:
                studies.append(study)

    return studies


def _compute_consistency(changes: list[float]) -> float:
    """Compute consistency as fraction of positive lifespan changes."""
    if not changes:
        return 0.0
    positive = sum(1 for c in changes if c > 0)
    return positive / len(changes)


def score_compound(compound: str, studies: list[DrugStudy] | None = None) -> CompoundScore | None:
    """Score a single compound by aggregating all its studies."""
    if studies is None:
        studies = load_drugage()

    compound_studies = [s for s in studies if s.compound.lower() == compound.lower()]
    if not compound_studies:
        return None

    # Collect all lifespan changes
    avg_changes = [s.avg_lifespan_change for s in compound_studies if s.avg_lifespan_change is not None]
    max_changes = [s.max_lifespan_change for s in compound_studies if s.max_lifespan_change is not None]

    mean_avg = sum(avg_changes) / len(avg_changes) if avg_changes else 0.0
    best_max = max(max_changes) if max_changes else 0.0

    # Species diversity
    species_set = set(s.species for s in compound_studies)

    # ITP validation (rigorous multi-site testing)
    itp_validated = any(s.is_itp for s in compound_studies)

    # Consistency across studies
    consistency = _compute_consistency(avg_changes)

    # Composite longevity score:
    # - base: mean lifespan change (normalized to 0-1 range, assuming max realistic is 100%)
    # - bonus for species diversity
    # - bonus for ITP validation
    # - bonus for consistency
    base_score = min(mean_avg / 100.0, 1.0) if mean_avg > 0 else max(mean_avg / 50.0, -1.0)
    species_bonus = min(len(species_set) * 0.05, 0.2)  # up to 0.2 for 4+ species
    itp_bonus = 0.15 if itp_validated else 0.0
    consistency_bonus = consistency * 0.1

    longevity_score = base_score + species_bonus + itp_bonus + consistency_bonus
    longevity_score = max(0.0, min(1.0, longevity_score))  # clamp to [0, 1]

    # Top studies by effect size
    sorted_studies = sorted(
        [s for s in compound_studies if s.avg_lifespan_change is not None],
        key=lambda s: s.avg_lifespan_change or 0,
        reverse=True
    )

    return CompoundScore(
        compound=compound,
        n_studies=len(compound_studies),
        n_species=len(species_set),
        species_list=sorted(species_set),
        mean_lifespan_change=round(mean_avg, 2),
        max_lifespan_change=round(best_max, 2),
        consistency=round(consistency, 3),
        itp_validated=itp_validated,
        longevity_score=round(longevity_score, 4),
        top_studies=sorted_studies[:5],
    )


def rank_all_compounds(min_studies: int = 2) -> list[CompoundScore]:
    """Rank all compounds by longevity score."""
    studies = load_drugage()

    # Group by compound
    compound_names = set(s.compound for s in studies)

    scores = []
    for compound in compound_names:
        score = score_compound(compound, studies)
        if score and score.n_studies >= min_studies:
            scores.append(score)

    # Sort by longevity score descending
    scores.sort(key=lambda s: s.longevity_score, reverse=True)
    return scores


def get_itp_compounds() -> list[CompoundScore]:
    """Get all ITP-validated compounds (gold standard interventions)."""
    all_scores = rank_all_compounds(min_studies=1)
    return [s for s in all_scores if s.itp_validated]


def search_compounds(query: str, limit: int = 20) -> list[CompoundScore]:
    """Search compounds by name substring."""
    all_scores = rank_all_compounds(min_studies=1)
    query_lower = query.lower()
    matches = [s for s in all_scores if query_lower in s.compound.lower()]
    return matches[:limit]


def get_top_by_species(species: str, top_n: int = 20) -> list[CompoundScore]:
    """Get top compounds for a specific species."""
    studies = load_drugage()
    species_lower = species.lower()

    # Filter studies by species
    species_studies = [s for s in studies if species_lower in s.species.lower()]

    # Get unique compounds in these studies
    compounds = set(s.compound for s in species_studies)

    scores = []
    for compound in compounds:
        # Score only within this species
        compound_species_studies = [s for s in species_studies if s.compound == compound]
        avg_changes = [s.avg_lifespan_change for s in compound_species_studies if s.avg_lifespan_change]
        if not avg_changes:
            continue

        mean_change = sum(avg_changes) / len(avg_changes)

        # Create simplified score
        score = CompoundScore(
            compound=compound,
            n_studies=len(compound_species_studies),
            n_species=1,
            species_list=[species],
            mean_lifespan_change=round(mean_change, 2),
            max_lifespan_change=round(max(avg_changes), 2),
            consistency=_compute_consistency(avg_changes),
            itp_validated=any(s.is_itp for s in compound_species_studies),
            longevity_score=round(min(mean_change / 100.0, 1.0), 4) if mean_change > 0 else 0.0,
            top_studies=compound_species_studies[:3],
        )
        scores.append(score)

    scores.sort(key=lambda s: s.mean_lifespan_change, reverse=True)
    return scores[:top_n]


def get_species_list() -> list[tuple[str, int]]:
    """Get list of species with study counts."""
    studies = load_drugage()
    species_counts: dict[str, int] = {}
    for s in studies:
        species_counts[s.species] = species_counts.get(s.species, 0) + 1

    return sorted(species_counts.items(), key=lambda x: x[1], reverse=True)


# ── Cross-link with aging clocks ────────────────────────────────────────


def link_compound_to_pathways(compound: str) -> dict:
    """
    Link compound to aging pathways via known gene targets.

    Uses MSigDB drug-gene links and returns overlapping pathways.
    """
    from longevityclaw.annotations import get_gene_hallmarks
    from longevityclaw.clock_db import get_db

    db = get_db()

    # Known compound -> gene target mappings (curated subset)
    COMPOUND_TARGETS = {
        "rapamycin": ["MTOR", "RPTOR", "RICTOR", "RPS6KB1", "EIF4EBP1"],
        "metformin": ["PRKAA1", "PRKAA2", "AMPK", "PPARGC1A", "SIRT1"],
        "resveratrol": ["SIRT1", "SIRT3", "PPARGC1A", "NRF2", "NFE2L2"],
        "nicotinamide": ["SIRT1", "SIRT2", "SIRT3", "NAMPT", "PARP1"],
        "spermidine": ["ATG5", "ATG7", "BECN1", "MAP1LC3A", "EP300"],
        "quercetin": ["SIRT1", "FOXO3", "CDKN2A", "BCL2L1", "NFKB1"],
        "fisetin": ["SIRT1", "CDKN2A", "BCL2", "FOXO3", "NFKB1"],
        "dasatinib": ["SRC", "ABL1", "PDGFRB", "EPHA2", "FYN"],
        "navitoclax": ["BCL2", "BCL2L1", "BCL2L2", "MCL1"],
        "acarbose": ["MGAM", "SI", "GAA", "AMY2A"],
        "aspirin": ["PTGS1", "PTGS2", "NFKB1", "ALOX5"],
        "lithium": ["GSK3B", "GSK3A", "BDNF", "INPP1"],
        "curcumin": ["NFKB1", "TNF", "IL6", "STAT3", "SIRT1"],
        "vitamin d": ["VDR", "CYP27B1", "CYP24A1", "GC"],
        "vitamin e": ["TTPA", "SCARB1", "NR1I2", "CYP4F2"],
    }

    compound_lower = compound.lower()
    target_genes = []

    for drug, genes in COMPOUND_TARGETS.items():
        if drug in compound_lower or compound_lower in drug:
            target_genes = genes
            break

    if not target_genes:
        return {
            "compound": compound,
            "known_targets": False,
            "target_genes": [],
            "pathways": [],
            "clock_genes": [],
        }

    # Find which targets are in aging clocks
    clock_genes = []
    for gene in target_genes:
        gene_clocks = db.find_clocks_by_gene(gene)
        if gene_clocks:
            clock_genes.append({
                "gene": gene,
                "clock_count": len(gene_clocks),
                "clocks": gene_clocks[:5],
            })

    # Find overlapping pathways
    pathways = []
    for gene in target_genes:
        gene_hallmarks = get_gene_hallmarks(gene)
        for hallmark in gene_hallmarks:
            if hallmark not in [p["pathway"] for p in pathways]:
                pathways.append({
                    "pathway": hallmark,
                    "genes": [gene],
                })
            else:
                for p in pathways:
                    if p["pathway"] == hallmark and gene not in p["genes"]:
                        p["genes"].append(gene)

    return {
        "compound": compound,
        "known_targets": True,
        "target_genes": target_genes,
        "pathways": pathways,
        "clock_genes": clock_genes,
    }
