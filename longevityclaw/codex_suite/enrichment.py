"""Offline pathway tests with an explicit measured-gene background."""
from __future__ import annotations
import math
from pathlib import Path
from .io import SuiteError, fingerprint, gene_list, number, table, unique_ids

def read_gmt(path: Path) -> dict[str, set[str]]:
    sets = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 3 or not fields[0] or fields[0] in sets:
                raise SuiteError("Malformed GMT or duplicate pathway name.")
            genes = {gene.strip() for gene in fields[2:] if gene.strip()}
            if genes:
                sets[fields[0]] = genes
    if not sets:
        raise SuiteError("GMT contains no non-empty gene sets.")
    return sets

def hypergeom_tail(n: int, k: int, draws: int, hits: int) -> float:
    """One-sided Fisher/hypergeometric P[X >= hits], in log space."""
    low, high = max(0, draws - (n-k)), min(k, draws)
    if hits <= low:
        return 1.0
    if hits > high:
        return 0.0
    def log_choose(a, b):
        return math.lgamma(a+1) - math.lgamma(b+1) - math.lgamma(a-b+1)
    denom = log_choose(n, draws)
    terms = [log_choose(k, x) + log_choose(n-k, draws-x) - denom for x in range(hits, high+1)]
    peak = max(terms)
    return min(1.0, math.exp(peak) * math.fsum(math.exp(v-peak) for v in terms))

def bh_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted, running = [1.0] * len(p_values), 1.0
    for rank in range(len(order), 0, -1):
        idx = order[rank-1]
        running = min(running, p_values[idx] * len(order)/rank)
        adjusted[idx] = running
    return adjusted

def enrich(foreground_path: Path, universe_path: Path, gmt: Path) -> dict:
    foreground, universe = gene_list(foreground_path), gene_list(universe_path)
    if foreground - universe:
        raise SuiteError(f"Foreground must be within universe; unmatched: {sorted(foreground-universe)[:10]}")
    rows = []
    for name, genes in read_gmt(gmt).items():
        eligible = genes & universe
        if not eligible:
            continue
        hits = foreground & eligible
        rows.append({"pathway": name, "overlap": len(hits), "genes": sorted(hits),
                     "pathway_in_universe": len(eligible), "foreground_size": len(foreground), "universe_size": len(universe),
                     "p_value": hypergeom_tail(len(universe), len(eligible), len(foreground), len(hits))})
    if not rows:
        raise SuiteError("No pathway genes match the universe; check species and identifiers.")
    for row, value in zip(rows, bh_adjust([r["p_value"] for r in rows])):
        row["fdr_bh"] = value
    rows.sort(key=lambda r: (r["fdr_bh"], r["p_value"], r["pathway"]))
    return {"summary": {"pathways_tested": len(rows), "significant_fdr_005": sum(r["fdr_bh"] < .05 for r in rows)},
            "results": rows, "sources": [fingerprint(p) for p in (foreground_path, universe_path, gmt)],
            "warnings": ["Over-representation tests association, not causal aging mechanisms.",
                         "Background must reflect genes that could have been selected in this experiment."]}

def gsea(ranked_path: Path, gmt: Path, seed: int = 42, permutations: int = 1000) -> dict:
    header, rows = table(ranked_path)
    if header != ["gene", "score"]:
        raise SuiteError("Ranked table requires exactly gene,score (CSV) or gene<TAB>score (TSV).")
    unique_ids([row[0] for row in rows], "Ranked genes")
    ranks = [(row[0], number(row[1], row[0])) for row in rows]
    if len(ranks) < 10:
        raise SuiteError("GSEA requires at least 10 ranked genes; use the full tested-gene ranking.")
    ranks.sort(key=lambda p: (-p[1], p[0]))
    try:
        import pandas as pd
        import gseapy as gp
    except ImportError as exc:
        raise SuiteError("GSEA requires pandas and gseapy. Install codex-requirements-analysis.txt.") from exc
    result = gp.prerank(rnk=pd.DataFrame(ranks), gene_sets={k: sorted(v) for k, v in read_gmt(gmt).items()},
                        min_size=5, max_size=5000, permutation_num=permutations, seed=seed, threads=1,
                        outdir=None, no_plot=True, verbose=False)
    import json
    records = json.loads(result.res2d.to_json(orient="records"))
    return {"summary": {"pathways_tested": len(records), "ranked_genes": len(ranks), "seed": seed, "permutations": permutations},
            "results": records, "sources": [fingerprint(ranked_path), fingerprint(gmt)],
            "warnings": ["Supply a signed contrast-specific statistic, not expression levels.",
                         "This tool performs neither normalization nor differential-expression analysis."]}
