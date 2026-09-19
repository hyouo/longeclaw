"""All numeric fixtures are synthetic, not biological models."""
import csv
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "CLOCKSdata").mkdir(parents=True)
    (root / "data/drugage").mkdir(parents=True)
    entries = [("toy_rna", "transcriptomics"), ("toy_protein", "proteomics"), ("toy_beta", "methylation"),
               ("bad_coeff", "transcriptomics"), ("duplicate", "transcriptomics")]
    with (root / "CLOCKSdata/clock_descriptions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clock_name", "description", "modality", "year", "n_features", "citation", "doi"])
        for name, modality in entries:
            writer.writerow([name, "Synthetic test fixture only", modality, 2020, 2, "test", ""])
    with (root / "CLOCKSdata/unified_aging_clocks.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clock_name", "feature_id", "coefficient", "gene_symbol", "modality", "data_type", "year"])
        for name, modality in entries[:3]:
            for fid, weight, gene in [("intercept", 5, "INTERCEPT"), ("G1", 2, "G1"), ("G2", -1, "G2")]:
                writer.writerow([name, fid, weight, gene, modality, "test_only", 2020])
        writer.writerow(["bad_coeff", "G1", "", "G1", "transcriptomics", "test_only", 2020])
        for weight in [2, 3]:
            writer.writerow(["duplicate", "G1", weight, "G1", "transcriptomics", "test_only", 2020])
    (root / "samples.csv").write_text("sample_id,G1,G2\nA,1,2\nB,3,4\n")
    (root / "features.tsv").write_text("feature_id\tA\tB\nG1\t1\t3\nG2\t2\t4\n")
    (root / "long.csv").write_text("feature_id,value\nG1,1\nG2,2\n")
    (root / "data/msigdb_hallmarks.gmt").write_text("P1\ttoy\tG1\tG2\nP2\ttoy\tG3\tG4\nP3\ttoy\tG1\tG3\n")
    (root / "foreground.txt").write_text("G1\nG2\n")
    (root / "universe.txt").write_text("G1\nG2\nG3\nG4\nG5\nG6\n")
    (root / "data/drugage/drugage.csv").write_text("compound,species,effect\ncompoundX,mouse,10\ncompoundY,worm,2\n")
    return root

@pytest.fixture
def suite(repo):
    from longevityclaw.codex_suite.registry import Suite
    return Suite(repo, repo)

@pytest.fixture
def score_args():
    return {"input_path": "samples.csv", "layout": "samples_by_features", "modality": "transcriptomics", "scale": "tpm", "clocks": ["toy_rna"]}
