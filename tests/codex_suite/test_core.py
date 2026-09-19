import csv
import hashlib
import importlib.util
import json
import math
import pytest
from longevityclaw.codex_suite.enrichment import bh_adjust, hypergeom_tail
from longevityclaw.codex_suite.io import SuiteError, read_json
from longevityclaw.codex_suite.registry import TOOLS

def test_every_sample_scored(suite, score_args):
    data = suite.call("score_file", score_args)["data"]
    assert data["summary"]["samples"] == 2
    assert [(r["sample_id"], r["raw_score"]) for r in data["results"]] == [("A", 5), ("B", 7)]
    assert all(r["score_unit"] == "uncalibrated_model_score" and "predicted_age" not in r for r in data["results"])

def test_orientations_equal(suite, score_args):
    first = suite.call("score_file", score_args)["data"]["results"]
    second = suite.call("score_file", {**score_args, "input_path": "features.tsv", "layout": "features_by_samples"})["data"]["results"]
    assert first == second

def test_long(suite, score_args):
    assert suite.call("score_file", {**score_args, "input_path": "long.csv", "layout": "long"})["data"]["results"][0]["raw_score"] == 5

def test_no_cross_modality(suite, score_args):
    out = suite.call("score_file", {k: v for k, v in score_args.items() if k != "clocks"})
    assert all(r["modality"] == "transcriptomics" for r in out["data"]["results"])
    with pytest.raises(SuiteError, match="does not belong"):
        suite.call("score_file", {**score_args, "clocks": ["toy_protein"]})

def test_per_sample_missing_coverage(repo, suite, score_args):
    (repo / "samples.csv").write_text("sample_id,G1,G2\nA,1,NA\nB,3,4\n")
    rows = suite.call("score_file", score_args)["data"]["results"]
    assert rows[0]["coverage"] == .5 and rows[0]["status"] == "insufficient_coverage" and rows[0]["raw_score"] is None
    assert rows[1]["raw_score"] == 7

def test_partial_coverage_explicit(repo, suite, score_args):
    (repo / "samples.csv").write_text("sample_id,G1,G2\nA,1,\n")
    with pytest.raises(SuiteError, match="explicit"):
        suite.call("score_file", {**score_args, "min_coverage": .5})
    row = suite.call("score_file", {**score_args, "min_coverage": .5, "missing_policy": "zero"})["data"]["results"][0]
    assert row["raw_score"] == 7 and row["missing_policy"] == "zero"

@pytest.mark.parametrize("extra", [{"scale": "counts"}, {"min_coverage": 0}, {"min_coverage": 1.1}, {"min_coverage": True},
                                    {"top_n": -1}, {"surprise": 1}, {"clocks": []}, {"clocks": ["toy_rna", "toy_rna"]}])
def test_invalid_args(suite, score_args, extra):
    with pytest.raises(SuiteError):
        suite.call("score_file", {**score_args, **extra})

@pytest.mark.parametrize("text", ["sample_id,G1,G1\nA,1,2\n", "sample_id,G1\nA,1\nA,2\n", "sample_id,G1,G2\nA,1\n",
                                 "sample_id,G1,G2\nA,no,2\n", "sample_id,G1,G2\nA,inf,2\n", "sample_id,G1,G2\nA,NA,NA\n",
                                 "sample_id,G1,G2\n,1,2\n", "id,G1,G2\nA,1,2\n"])
def test_bad_input(repo, suite, score_args, text):
    (repo / "samples.csv").write_text(text)
    with pytest.raises(SuiteError):
        suite.call("score_file", score_args)

def test_bad_weights_not_filled(suite, score_args):
    rows = suite.call("score_file", {**score_args, "clocks": ["bad_coeff", "duplicate"]})["data"]["results"]
    assert all(r["status"] == "invalid_coefficients" and r["raw_score"] is None for r in rows)

def test_beta_range(suite, score_args):
    with pytest.raises(SuiteError, match="beta"):
        suite.call("score_file", {**score_args, "modality": "methylation", "scale": "beta", "clocks": ["toy_beta"]})

def test_scoped_paths(repo, suite, score_args):
    outside = repo.parent / "outside.csv"
    outside.write_text("sample_id,G1,G2\nA,1,2\n")
    (repo / "escape.csv").symlink_to(outside)
    for path in ["../outside.csv", str(outside), "escape.csv"]:
        with pytest.raises(SuiteError, match="escapes"):
            suite.call("score_file", {**score_args, "input_path": path})

def test_export_and_no_overwrite(repo, suite, score_args):
    original = (repo / "samples.csv").read_bytes()
    out = suite.call("score_file", {**score_args, "output_path": "out.json", "output_csv": "out.csv"})
    saved = json.loads((repo / "out.json").read_text())
    with (repo / "out.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(saved["data"]["results"]) == 2 and rows[1]["sample_id"] == "B" and float(rows[1]["raw_score"]) == 7
    assert (repo / "samples.csv").read_bytes() == original
    assert saved["data"]["input"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert out["summary"]["rows"] == 2
    with pytest.raises(SuiteError):
        suite.call("score_file", {**score_args, "output_path": "out.json"})

def test_output_preflight(repo, suite, score_args):
    for extra in [{"output_path": "new.json", "output_csv": "samples.csv"}, {"output_path": "new.json", "output_csv": "new.json"}, {"output_path": "../out.json"}]:
        with pytest.raises(SuiteError):
            suite.call("score_file", {**score_args, **extra})
    assert not (repo / "new.json").exists()

def test_catalog(suite):
    assert suite.call("list_clocks", {})["data"]["total"] == 5
    detail = suite.call("clock_details", {"clock_name": "toy_rna"})["data"]
    assert detail["intercept"] == 5 and detail["n_features"] == 2
    assert suite.call("search_gene", {"gene": "g1"})["data"]["total"] >= 3

@pytest.mark.parametrize("n,k,draws,hits", [(6,2,2,2), (20,7,5,3), (100,20,10,5), (100,20,10,0), (10,1,2,3)])
def test_fisher_independent_exact(n, k, draws, hits):
    exact = sum(math.comb(k,x)*math.comb(n-k,draws-x) for x in range(max(hits,0,draws-(n-k)),min(k,draws)+1))/math.comb(n,draws)
    assert hypergeom_tail(n,k,draws,hits) == pytest.approx(exact, rel=1e-10, abs=1e-14)

def test_full_family_bh(suite):
    data = suite.call("enrich_genes", {"foreground_path": "foreground.txt", "universe_path": "universe.txt"})["data"]
    assert data["summary"]["pathways_tested"] == 3 and data["results"][0]["pathway"] == "P1"
    assert data["results"][0]["p_value"] == pytest.approx(1/15) and data["results"][0]["fdr_bh"] == pytest.approx(.2)
    assert any(r["overlap"] == 0 and r["p_value"] == 1 for r in data["results"])
    assert bh_adjust([.01,.04,.03]) == pytest.approx([.03,.04,.04])

def test_background_required(repo, suite):
    (repo / "foreground.txt").write_text("G999\n")
    with pytest.raises(SuiteError, match="within"):
        suite.call("enrich_genes", {"foreground_path": "foreground.txt", "universe_path": "universe.txt"})

@pytest.mark.parametrize("text", ['{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'])
def test_strict_json(text):
    with pytest.raises(SuiteError):
        read_json(text)

def test_tool_contracts(suite):
    json.dumps(TOOLS, allow_nan=False)
    with pytest.raises(SuiteError):
        suite.call("__import__", {})

def test_drugage_raw(suite):
    data = suite.call("search_drugage", {"query": "compoundX"})["data"]
    assert data["total"] == 1 and data["records"][0]["species"] == "mouse" and "predicted_lifespan" not in data

def test_gsea_dependency_error(repo, suite):
    if importlib.util.find_spec("gseapy"):
        pytest.skip("gseapy installed; tested by optional integration")
    (repo / "ranked.tsv").write_text("gene\tscore\n"+"".join(f"G{i}\t{i}\n" for i in range(20)))
    with pytest.raises(SuiteError, match="gseapy"):
        suite.call("gsea", {"ranked_path": "ranked.tsv"})

def test_optional_real_gsea(repo, suite):
    pytest.importorskip("gseapy", reason="Optional gseapy dependency not installed")
    (repo / "ranked.tsv").write_text("gene\tscore\n"+"".join(f"G{i}\t{20-i}\n" for i in range(20)))
    (repo / "custom.gmt").write_text("set1\ttoy\t"+"\t".join(f"G{i}" for i in range(8))+"\n")
    out = suite.call("gsea", {"ranked_path": "ranked.tsv", "gmt_path": "custom.gmt", "permutations": 10})
    assert out["ok"] and out["data"]["summary"]["pathways_tested"] == 1

def test_large_results_not_silently_lost(repo, suite, score_args):
    (repo / "samples.csv").write_text("sample_id,G1,G2\n"+"".join(f"S{i},1,2\n" for i in range(205)))
    data = suite.call("score_file", score_args)["data"]
    assert data["preview_only"] and data["total_results"] == 205 and len(data["results"]) == 200
    suite.call("score_file", {**score_args, "output_path": "full.json"})
    assert len(json.loads((repo / "full.json").read_text())["data"]["results"]) == 205
