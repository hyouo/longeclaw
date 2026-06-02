"""
CpG and gene annotation layer.
Provides pathway mapping via MSigDB Hallmark gene sets (50 pathways, ~7K genes)
and aging-specific interpretation via GrimAge component knowledge.
Preranked GSEA against MSigDB v2025.1 collections (hallmarks, Reactome, KEGG, cancer).

Gene sets loaded from: data/msigdb_hallmarks.gmt (Broad Institute, CC-BY-4.0)
MSigDB JSON collections in: data/msigdb/ (v2025.1)
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

logger = logging.getLogger(__name__)

GMT_PATH = Path(__file__).resolve().parent.parent / "data" / "msigdb_hallmarks.gmt"

# ── Load MSigDB Hallmarks from GMT ─────────────────────────────────────

HALLMARKS: dict[str, dict] = {}
ALL_HALLMARK_GENES: set[str] = set()
GENE_TO_HALLMARKS: dict[str, list[str]] = {}


def _load_gmt():
    """Parse GMT file: each line is pathway_name<tab>url<tab>gene1<tab>gene2..."""
    global HALLMARKS, ALL_HALLMARK_GENES, GENE_TO_HALLMARKS

    if not GMT_PATH.exists():
        return

    with open(GMT_PATH) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            raw_name = parts[0]  # e.g. HALLMARK_INFLAMMATORY_RESPONSE
            url = parts[1]
            genes = set(parts[2:])

            # Clean up name: HALLMARK_INFLAMMATORY_RESPONSE -> inflammatory_response
            hallmark_id = raw_name.replace("HALLMARK_", "").lower()
            display_name = hallmark_id.replace("_", " ").title()

            HALLMARKS[hallmark_id] = {
                "name": display_name,
                "description": f"MSigDB Hallmark: {display_name}",
                "genes": genes,
                "url": url,
            }

            ALL_HALLMARK_GENES |= genes
            for g in genes:
                if g not in GENE_TO_HALLMARKS:
                    GENE_TO_HALLMARKS[g] = []
                GENE_TO_HALLMARKS[g].append(hallmark_id)


# Load on import
_load_gmt()


# ── Mapping Lopez-Otin aging hallmarks to relevant MSigDB sets ─────────

# The 12 Lopez-Otin hallmarks of aging map to these MSigDB pathway clusters.
# This lets us bridge from MSigDB enrichment to aging biology language.
AGING_HALLMARK_MAPPING = {
    "genomic_instability": ["dna_repair", "uv_response_up", "uv_response_dn", "p53_pathway"],
    "telomere_attrition": ["e2f_targets", "g2m_checkpoint"],
    "epigenetic_alterations": ["e2f_targets", "myc_targets_v1", "myc_targets_v2"],
    "loss_of_proteostasis": ["unfolded_protein_response", "protein_secretion"],
    "disabled_macroautophagy": ["mtorc1_signaling", "pi3k_akt_mtor_signaling"],
    "deregulated_nutrient_sensing": [
        "mtorc1_signaling", "pi3k_akt_mtor_signaling", "glycolysis",
        "fatty_acid_metabolism", "cholesterol_homeostasis", "adipogenesis",
        "oxidative_phosphorylation",
    ],
    "mitochondrial_dysfunction": [
        "oxidative_phosphorylation", "reactive_oxygen_species_pathway",
        "fatty_acid_metabolism", "heme_metabolism",
    ],
    "cellular_senescence": [
        "p53_pathway", "tnfa_signaling_via_nfkb", "inflammatory_response",
        "il6_jak_stat3_signaling", "apoptosis",
    ],
    "stem_cell_exhaustion": [
        "notch_signaling", "wnt_beta_catenin_signaling", "hedgehog_signaling",
        "myogenesis",
    ],
    "altered_intercellular_communication": [
        "tgf_beta_signaling", "notch_signaling", "wnt_beta_catenin_signaling",
        "angiogenesis", "coagulation",
    ],
    "chronic_inflammation": [
        "inflammatory_response", "tnfa_signaling_via_nfkb",
        "il6_jak_stat3_signaling", "il2_stat5_signaling",
        "interferon_alpha_response", "interferon_gamma_response",
        "complement", "allograft_rejection",
    ],
    "dysbiosis": ["xenobiotic_metabolism", "bile_acid_metabolism"],
}


def get_aging_hallmark_for_pathway(msigdb_id: str) -> list[str]:
    """Map an MSigDB pathway ID to Lopez-Otin aging hallmark(s)."""
    results = []
    for aging_hk, pathways in AGING_HALLMARK_MAPPING.items():
        if msigdb_id in pathways:
            results.append(aging_hk)
    return results


# ── Public API ─────────────────────────────────────────────────────────

def get_gene_hallmarks(gene: str) -> list[dict]:
    """Get MSigDB hallmark pathways associated with a gene."""
    hallmark_ids = GENE_TO_HALLMARKS.get(gene.upper(), [])
    return [
        {
            "hallmark_id": hid,
            "name": HALLMARKS[hid]["name"],
            "aging_hallmarks": get_aging_hallmark_for_pathway(hid),
        }
        for hid in hallmark_ids
    ]


def get_hallmark_info(hallmark_id: str) -> dict | None:
    """Get full info about an MSigDB hallmark pathway."""
    h = HALLMARKS.get(hallmark_id)
    if not h:
        return None
    return {
        "id": hallmark_id,
        "name": h["name"],
        "n_genes": len(h["genes"]),
        "genes": sorted(h["genes"]),
        "url": h.get("url", ""),
        "aging_hallmarks": get_aging_hallmark_for_pathway(hallmark_id),
    }


def map_genes_to_hallmarks(genes: list[str], background_size: int | None = None) -> dict:
    """Map a list of genes to MSigDB hallmark pathways with statistical enrichment.

    Uses Fisher's exact test (one-sided, over-representation) against the MSigDB
    hallmark gene universe. Reports p-value and fold enrichment for each pathway.

    Args:
        genes: list of gene symbols from clock features
        background_size: total number of genes in the background (default: MSigDB universe)
    """
    gene_set = {g.upper() for g in genes}
    n_query = len(gene_set)
    N = background_size or len(ALL_HALLMARK_GENES)  # universe size

    hallmark_hits: dict[str, list[str]] = {}
    unmapped = []

    for gene in gene_set:
        hids = GENE_TO_HALLMARKS.get(gene, [])
        if hids:
            for hid in hids:
                if hid not in hallmark_hits:
                    hallmark_hits[hid] = []
                hallmark_hits[hid].append(gene)
        else:
            unmapped.append(gene)

    results = []
    for hid, hit_genes in hallmark_hits.items():
        k = len(hit_genes)                       # hits in query
        K = len(HALLMARKS[hid]["genes"])          # pathway size
        n = n_query                              # query size

        # Fisher's exact test (2x2 contingency table)
        # [[k, K-k], [n-k, N-n-K+k]]
        a = k
        b = K - k
        c = n - k
        d = N - n - K + k
        if d < 0:
            d = 0
        _, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")

        # Fold enrichment: (k/n) / (K/N)
        expected = (K / N) * n if N > 0 else 0
        fold_enrichment = k / expected if expected > 0 else 0

        results.append({
            "hallmark_id": hid,
            "name": HALLMARKS[hid]["name"],
            "n_hits": k,
            "genes": sorted(hit_genes),
            "pathway_size": K,
            "fold_enrichment": round(fold_enrichment, 2),
            "pvalue": round(pvalue, 6),
            "significant": pvalue < 0.05,
            "aging_hallmarks": get_aging_hallmark_for_pathway(hid),
        })

    # Sort by p-value (most significant first)
    results.sort(key=lambda x: x["pvalue"])

    return {
        "total_genes_queried": n_query,
        "genes_mapped_to_hallmarks": n_query - len(unmapped),
        "genes_unmapped": len(unmapped),
        "universe_size": N,
        "hallmark_enrichment": results,
    }


# ── GrimAge component interpretation ──────────────────────────────────

GRIMAGE_COMPONENTS = {
    "DNAmPACKYRS": {
        "name": "DNAm Pack-Years",
        "measures": "Smoking exposure (pack-years) estimated from DNA methylation",
        "biological_meaning": "Chronic exposure to tobacco carcinogens, oxidative stress, inflammation",
        "msigdb_pathways": ["inflammatory_response", "reactive_oxygen_species_pathway", "p53_pathway"],
        "aging_hallmarks": ["genomic_instability", "chronic_inflammation", "cellular_senescence"],
        "interventions": ["Smoking cessation is the single most impactful intervention"],
    },
    "DNAmADM": {
        "name": "DNAm Adrenomedullin",
        "measures": "Plasma adrenomedullin levels estimated from DNA methylation",
        "biological_meaning": "Cardiovascular stress, blood pressure regulation, vasodilation",
        "msigdb_pathways": ["angiogenesis", "coagulation", "hypoxia"],
        "aging_hallmarks": ["altered_intercellular_communication", "chronic_inflammation"],
        "interventions": ["Cardiovascular exercise", "blood pressure management", "stress reduction"],
    },
    "DNAmB2M": {
        "name": "DNAm Beta-2-Microglobulin",
        "measures": "Plasma B2M levels -- a marker of immune activation and kidney function",
        "biological_meaning": "Immune system aging (immunosenescence), renal function decline",
        "msigdb_pathways": ["interferon_gamma_response", "allograft_rejection", "complement"],
        "aging_hallmarks": ["chronic_inflammation", "cellular_senescence", "stem_cell_exhaustion"],
        "interventions": ["Anti-inflammatory diet", "exercise", "kidney health monitoring"],
    },
    "DNAmCystatinC": {
        "name": "DNAm Cystatin C",
        "measures": "Plasma cystatin C -- a marker of kidney filtration rate (GFR)",
        "biological_meaning": "Kidney aging, cardiovascular risk",
        "msigdb_pathways": ["coagulation"],
        "aging_hallmarks": ["altered_intercellular_communication"],
        "interventions": ["Hydration", "blood pressure control", "nephrotoxin avoidance"],
    },
    "DNAmGDF15": {
        "name": "DNAm GDF-15",
        "measures": "Growth Differentiation Factor 15 -- a mitokine and stress response marker",
        "biological_meaning": "Mitochondrial stress, cellular senescence, inflammation. GDF15 is one of the strongest aging biomarkers.",
        "msigdb_pathways": ["tgf_beta_signaling", "oxidative_phosphorylation", "p53_pathway"],
        "aging_hallmarks": ["mitochondrial_dysfunction", "cellular_senescence", "chronic_inflammation"],
        "interventions": ["Exercise (reduces GDF15)", "NAD+ precursors", "mitochondrial support"],
    },
    "DNAmLeptin": {
        "name": "DNAm Leptin",
        "measures": "Plasma leptin -- adiposity hormone",
        "biological_meaning": "Metabolic health, body fat regulation, leptin resistance",
        "msigdb_pathways": ["adipogenesis", "mtorc1_signaling", "inflammatory_response"],
        "aging_hallmarks": ["deregulated_nutrient_sensing", "chronic_inflammation"],
        "interventions": ["Weight management", "exercise", "anti-inflammatory diet", "sleep optimization"],
    },
    "DNAmPAI1": {
        "name": "DNAm PAI-1",
        "measures": "Plasminogen Activator Inhibitor-1 -- blood clotting regulator",
        "biological_meaning": "Thrombotic risk, senescence-associated secretory phenotype (SASP), metabolic syndrome",
        "msigdb_pathways": ["coagulation", "tnfa_signaling_via_nfkb", "inflammatory_response"],
        "aging_hallmarks": ["cellular_senescence", "chronic_inflammation", "altered_intercellular_communication"],
        "interventions": ["Exercise", "weight loss", "omega-3 fatty acids", "senolytic discussion with physician"],
    },
    "DNAmTIMP1": {
        "name": "DNAm TIMP-1",
        "measures": "Tissue Inhibitor of Metalloproteinase 1 -- extracellular matrix regulator",
        "biological_meaning": "Tissue remodeling, fibrosis, liver health",
        "msigdb_pathways": ["epithelial_mesenchymal_transition", "tgf_beta_signaling"],
        "aging_hallmarks": ["altered_intercellular_communication", "loss_of_proteostasis"],
        "interventions": ["Liver health (reduce alcohol, NAFLD management)", "anti-fibrotic strategies"],
    },
    "DNAmlogCRP": {
        "name": "DNAm log(CRP)",
        "measures": "C-reactive protein -- acute-phase inflammatory marker (GrimAge v2)",
        "biological_meaning": "Systemic inflammation, cardiovascular risk",
        "msigdb_pathways": ["inflammatory_response", "il6_jak_stat3_signaling", "complement"],
        "aging_hallmarks": ["chronic_inflammation"],
        "interventions": ["Anti-inflammatory diet (Mediterranean)", "exercise", "omega-3", "sleep"],
    },
    "DNAmlogA1C": {
        "name": "DNAm log(HbA1c)",
        "measures": "Glycated hemoglobin -- 3-month average blood glucose (GrimAge v2)",
        "biological_meaning": "Glycemic control, metabolic health, diabetes risk",
        "msigdb_pathways": ["glycolysis", "mtorc1_signaling", "pi3k_akt_mtor_signaling"],
        "aging_hallmarks": ["deregulated_nutrient_sensing"],
        "interventions": ["Dietary carbohydrate management", "exercise", "metformin (discuss with physician)"],
    },
}


# ── MSigDB v2025.1 JSON collections for GSEA ────────────────────────────

MSIGDB_DIR = Path(__file__).resolve().parent.parent / "data" / "msigdb"

MSIGDB_COLLECTIONS = {
    "hallmarks": "h.all.v2025.1.Hs.json",
    "reactome": "c2.cp.reactome.v2025.1.Hs.json",
    "kegg": "c2.cp.kegg_medicus.v2025.1.Hs.json",
    "cancer": ["c4.cgn.v2025.1.Hs.json", "c4.cm.v2025.1.Hs.json", "c4.3ca.v2025.1.Hs.json"],
}

_msigdb_cache: dict[str, dict[str, list[str]]] = {}


def load_msigdb_json(collection: str) -> dict[str, list[str]]:
    if collection in _msigdb_cache:
        return _msigdb_cache[collection]

    if collection == "all":
        merged: dict[str, list[str]] = {}
        for coll in MSIGDB_COLLECTIONS:
            merged.update(load_msigdb_json(coll))
        _msigdb_cache["all"] = merged
        return merged

    filenames = MSIGDB_COLLECTIONS.get(collection)
    if filenames is None:
        raise ValueError(f"Unknown MSigDB collection: {collection}. Available: {list(MSIGDB_COLLECTIONS.keys()) + ['all']}")

    if isinstance(filenames, str):
        filenames = [filenames]

    gene_sets: dict[str, list[str]] = {}
    for fn in filenames:
        path = MSIGDB_DIR / fn
        if not path.exists():
            logger.warning(f"MSigDB file not found: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        for name, info in data.items():
            gene_sets[name] = info.get("geneSymbols", [])

    _msigdb_cache[collection] = gene_sets
    return gene_sets


def modernize_gene_symbols(ranked_genes: dict[str, float]) -> dict[str, float]:
    """Remap deprecated gene symbols to current HGNC names via the CpG annotation DB.

    For CpG-based ranked lists where genes come from clock_db (which may use old symbols
    like FLJ21839, C19orf30, hfl-B5), resolve to modern names (AGBL5, MIR7-3HG, EIF3M)
    using the genomic annotations in cpg_annotations.sqlite.

    Only remaps genes that look like deprecated identifiers (starting with C\dorf, FLJ,
    LOC, hfl, KIAA, or FAM) to avoid unnecessary lookups.
    """
    import re
    try:
        from .cpg_db import get_cpg_db
        from .clock_db import get_db
    except Exception:
        return ranked_genes

    deprecated_pattern = re.compile(
        r"^(C\d+orf\d+|FLJ\d+|LOC\d+|hfl-?\w+|KIAA\d+|FAM\d+\w*|ENSG\d+)$", re.IGNORECASE
    )

    needs_remap = {g for g in ranked_genes if deprecated_pattern.match(g)}
    if not needs_remap:
        return ranked_genes

    clock_db = get_db()
    cpg_db = get_cpg_db()

    gene_to_cpgs: dict[str, list[str]] = {}
    for clock_name in clock_db.list_clocks():
        model = clock_db.get_clock_model(clock_name.name)
        if not model:
            continue
        for fid, gene in model.gene_map.items():
            if gene in needs_remap and fid.startswith("cg"):
                gene_to_cpgs.setdefault(gene, []).append(fid)

    remap: dict[str, str] = {}
    for old_gene, cpg_ids in gene_to_cpgs.items():
        results = cpg_db.lookup(cpg_ids[:1])
        if results and results[0].get("found"):
            for g in results[0].get("genes", []):
                new_sym = g["gene"]
                if (new_sym != old_gene
                        and not new_sym.startswith("ENSG")
                        and g.get("biotype") == "protein_coding"):
                    remap[old_gene] = new_sym
                    break

    if not remap:
        return ranked_genes

    remapped: dict[str, float] = {}
    for gene, value in ranked_genes.items():
        new_gene = remap.get(gene, gene)
        if new_gene in remapped:
            remapped[new_gene] = (remapped[new_gene] + value) / 2
        else:
            remapped[new_gene] = value
    return remapped


def _silence_gseapy():
    """Remove all handlers gseapy attaches to its loggers."""
    for name in list(logging.Logger.manager.loggerDict) + ["gseapy"]:
        if "gseapy" in name.lower():
            lg = logging.getLogger(name)
            lg.handlers.clear()
            lg.setLevel(logging.CRITICAL)


def run_preranked_gsea(
    ranked_genes: dict[str, float],
    database: str = "hallmarks",
    min_size: int = 5,
    max_size: int = 500,
    permutation_num: int = 1000,
) -> list[dict]:
    """Run preranked GSEA using gseapy on a ranked gene list.

    Args:
        ranked_genes: gene_symbol → statistic (fold change, z-score, coefficient, etc.)
        database: MSigDB collection — "hallmarks", "reactome", "kegg", "cancer", or "all"
        min_size: minimum gene set size to test (default 5 to handle small lists)
        max_size: maximum gene set size to test
        permutation_num: number of permutations for p-value estimation

    Returns:
        list of dicts sorted by |NES|, each with: name, nes, pval, fdr, n_genes,
        leading_edge (top 10), aging_hallmark mapping
    """
    import gseapy
    import pandas as pd

    gene_sets = load_msigdb_json(database)
    if not gene_sets:
        return []

    ranked_genes = modernize_gene_symbols(ranked_genes)

    input_genes = set(ranked_genes.keys())
    all_gs_genes = {g for genes in gene_sets.values() for g in genes}
    overlap = input_genes & all_gs_genes
    if len(overlap) < 5:
        raise ValueError(
            f"Only {len(overlap)}/{len(input_genes)} input genes match {database} gene sets. "
            f"GSEA needs standard HGNC symbols (e.g. TP53, FOXO3). "
            f"Non-matching examples: {sorted(input_genes - all_gs_genes)[:5]}"
        )

    rnk = pd.Series(ranked_genes, dtype=float)
    # Collapse duplicate gene symbols: mean of their statistics
    if rnk.index.duplicated().any():
        rnk = rnk.groupby(level=0).mean()
    # Break tied values with tiny seeded noise so GSEA ranking is deterministic
    if rnk.duplicated().any():
        rng = np.random.default_rng(42)
        noise = rng.normal(0, abs(rnk).median() * 1e-6, size=len(rnk))
        rnk = rnk + noise
    rnk = rnk.sort_values(ascending=False)

    # gseapy creates its own StreamHandler on each call — suppress it
    _silence_gseapy()
    pre_res = gseapy.prerank(
        rnk=rnk,
        gene_sets=gene_sets,
        min_size=min_size,
        max_size=max_size,
        permutation_num=permutation_num,
        outdir=None,
        no_plot=True,
        verbose=False,
        seed=42,
    )
    _silence_gseapy()

    df = pre_res.res2d
    if df is None or df.empty:
        return []

    results = []
    for _, row in df.iterrows():
        name = row["Term"]
        lead_genes_str = row.get("Lead_genes", "")
        lead_genes = [g for g in str(lead_genes_str).split(";") if g] if lead_genes_str else []

        msigdb_id = name.replace("HALLMARK_", "").lower() if name.startswith("HALLMARK_") else ""
        aging_hks = get_aging_hallmark_for_pathway(msigdb_id) if msigdb_id else []

        results.append({
            "name": name,
            "nes": round(float(row["NES"]), 3),
            "pval": float(row["NOM p-val"]),
            "fdr": float(row["FDR q-val"]),
            "n_genes": int(row.get("Tag %", "0/0").split("/")[1]) if "/" in str(row.get("Tag %", "")) else 0,
            "n_hits": int(row.get("Tag %", "0/0").split("/")[0]) if "/" in str(row.get("Tag %", "")) else 0,
            "leading_edge": lead_genes[:10],
            "aging_hallmarks": aging_hks,
        })

    results.sort(key=lambda x: abs(x["nes"]), reverse=True)
    return results
