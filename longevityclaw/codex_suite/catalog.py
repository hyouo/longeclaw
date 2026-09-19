"""Read upstream coefficient CSVs; no agent, pickle cache or age calibration."""
from __future__ import annotations
import csv
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from .io import SuiteError, fingerprint, number

@dataclass
class Model:
    name: str
    modality: str
    intercept: float = 0.0
    has_intercept: bool = False
    weights: dict[str, float] = field(default_factory=dict)
    genes: dict[str, str] = field(default_factory=dict)
    data_types: set[str] = field(default_factory=set)
    issues: set[str] = field(default_factory=set)

class Catalog:
    def __init__(self, repo: Path):
        self.repo, self.metadata, self.models, self.sources = repo, {}, {}, []

    def load(self) -> Catalog:
        desc = self.repo / "CLOCKSdata/clock_descriptions.csv"
        coeff = self.repo / "CLOCKSdata/unified_aging_clocks.csv"
        for path in (desc, coeff):
            if not path.is_file():
                raise SuiteError(f"Missing upstream data: {path}. Install the add-on in a complete clone.")
        with desc.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"clock_name", "description", "modality", "citation", "doi"} <= set(reader.fieldnames or []):
                raise SuiteError("Unsupported clock_descriptions.csv schema.")
            for row in reader:
                name = (row.get("clock_name") or "").strip()
                if not name or name in self.metadata:
                    raise SuiteError("Empty or duplicate clock name in metadata.")
                self.metadata[name] = {key: value for key, value in row.items() if key is not None}
        with coeff.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"clock_name", "feature_id", "coefficient", "gene_symbol", "modality", "data_type"}
            if not required <= set(reader.fieldnames or []):
                raise SuiteError("Unsupported unified_aging_clocks.csv schema.")
            for row in reader:
                name, mod = (row.get("clock_name") or "").strip(), (row.get("modality") or "").strip()
                if not name or not mod:
                    raise SuiteError("Missing clock name or modality in coefficient table.")
                model = self.models.setdefault(name, Model(name, mod))
                if model.modality != mod:
                    model.issues.add("inconsistent_modality")
                model.data_types.add((row.get("data_type") or "").strip())
                feature, gene = (row.get("feature_id") or "").strip(), (row.get("gene_symbol") or "").strip()
                try:
                    weight = number(row.get("coefficient"), f"coefficient {name}/{feature}")
                except SuiteError:
                    model.issues.add("missing_or_nonfinite_coefficient")
                    continue
                if feature.lower() == "intercept" or gene.upper() == "INTERCEPT":
                    if model.has_intercept:
                        model.issues.add("duplicate_intercept")
                    model.intercept, model.has_intercept = weight, True
                else:
                    if not feature or feature in model.weights:
                        model.issues.add("empty_or_duplicate_feature")
                    model.weights[feature] = weight
                    if gene and gene.lower() != "nan":
                        model.genes[feature] = gene
        for name, model in self.models.items():
            if not model.weights:
                model.issues.add("no_numeric_features")
            if name in self.metadata and self.metadata[name]["modality"] != model.modality:
                model.issues.add("metadata_modality_mismatch")
        if not self.models:
            raise SuiteError("Coefficient table contains no models.")
        self.sources = [fingerprint(desc), fingerprint(coeff)]
        return self

    def list_clocks(self, modality: str | None = None) -> dict:
        rows = []
        for name in sorted(set(self.metadata) | set(self.models)):
            meta, model = self.metadata.get(name, {}), self.models.get(name)
            mod = model.modality if model else meta.get("modality", "")
            if modality is not None and modality != mod:
                continue
            rows.append({"name": name, "modality": mod, "description": meta.get("description", ""),
                         "coefficient_features": len(model.weights) if model else 0,
                         "numeric_coefficients_usable": bool(model and not model.issues),
                         "issues": sorted(model.issues) if model else ["no_coefficient_model"], "doi": meta.get("doi", "")})
        return {"total": len(rows), "clocks": rows, "modalities": dict(Counter(r["modality"] for r in rows)),
                "sources": self.sources, "warning": "Catalog membership is not validation of an age predictor."}

    def details(self, clock_name: str, top_n: int = 20) -> dict:
        if clock_name not in self.metadata and clock_name not in self.models:
            raise SuiteError("Unknown clock. Call list_clocks first.")
        model = self.models.get(clock_name)
        data = {"metadata": self.metadata.get(clock_name, {}), "sources": self.sources}
        if model:
            data.update({"intercept": model.intercept, "intercept_recorded": model.has_intercept,
                         "data_types": sorted(model.data_types), "n_features": len(model.weights),
                         "issues": sorted(model.issues), "top_coefficients": [
                             {"feature_id": fid, "gene": model.genes.get(fid, ""), "coefficient": weight}
                             for fid, weight in sorted(model.weights.items(), key=lambda p: (-abs(p[1]), p[0]))[:top_n]]})
        return data

    def search_gene(self, gene: str, limit: int = 50) -> dict:
        rows = [{"clock": name, "feature_id": fid, "gene": symbol, "modality": model.modality,
                 "coefficient": model.weights[fid]}
                for name, model in sorted(self.models.items()) for fid, symbol in model.genes.items()
                if symbol.upper() == gene.upper()]
        return {"gene": gene, "total": len(rows), "matches": rows[:limit], "truncated": len(rows) > limit,
                "warning": "Coefficient signs describe score contributions, not causal aging effects."}

    def score(self, samples: dict[str, dict[str, float]], *, modality: str, scale: str,
              clocks: list[str] | None = None, min_coverage: float = 1.0,
              missing_policy: str = "reject", top_n: int = 10) -> dict:
        if not 0 < min_coverage <= 1:
            raise SuiteError("min_coverage must be greater than zero and at most one.")
        scales = {"methylation": {"beta"}, "transcriptomics": {"tpm", "log2_tpm", "native"},
                  "proteomics": {"npx", "native"}}
        if scale not in scales.get(modality, {"native"}):
            raise SuiteError(f"Unsupported scale {scale!r} for {modality}; raw counts are not accepted.")
        if missing_policy not in ("reject", "zero"):
            raise SuiteError("missing_policy must be reject or zero.")
        if min_coverage < 1 and missing_policy != "zero":
            raise SuiteError("Partial coverage requires explicit missing_policy=zero; exploratory use only.")
        if modality not in {m.modality for m in self.models.values()}:
            raise SuiteError("Unknown modality. Call list_clocks first.")
        names = sorted(clocks) if clocks is not None else sorted(n for n, m in self.models.items() if m.modality == modality)
        if not names or len(names) != len(set(names)):
            raise SuiteError("clocks must contain distinct, existing names.")
        for name in names:
            if name not in self.models or self.models[name].modality != modality:
                raise SuiteError(f"Clock {name!r} does not belong to modality {modality!r}.")
        rows = []
        for sample_id, values in samples.items():
            for fid, value in values.items():
                if not math.isfinite(value):
                    raise SuiteError(f"{sample_id}/{fid}: non-finite measurement.")
                if scale == "beta" and not 0 <= value <= 1:
                    raise SuiteError(f"{sample_id}/{fid}: methylation beta must be between 0 and 1.")
                if scale in ("tpm", "log2_tpm") and value < 0:
                    raise SuiteError("TPM/log2(TPM+1) cannot be negative.")
            for name in names:
                model = self.models[name]
                present, missing = model.weights.keys() & values.keys(), model.weights.keys() - values.keys()
                coverage = len(present) / len(model.weights) if model.weights else 0.0
                row = {"sample_id": sample_id, "clock": name, "modality": modality, "input_scale": scale,
                       "coverage": coverage, "n_used": len(present), "n_required": len(model.weights),
                       "n_missing": len(missing), "missing_features_preview": sorted(missing)[:20],
                       "raw_score": None, "score_unit": "uncalibrated_model_score", "status": "scored",
                       "doi": self.metadata.get(name, {}).get("doi", "")}
                if model.issues:
                    row.update(status="invalid_coefficients", issues=sorted(model.issues))
                elif coverage < min_coverage:
                    row["status"] = "insufficient_coverage"
                else:
                    contrib = [(fid, model.weights[fid] * values[fid]) for fid in present]
                    try:
                        score = math.fsum([model.intercept] + [val for _, val in contrib])
                    except (ValueError, OverflowError):
                        score = math.inf
                    if not math.isfinite(score):
                        row["status"] = "nonfinite_score"
                    else:
                        row.update(raw_score=score, intercept=model.intercept, intercept_recorded=model.has_intercept,
                                   missing_policy=missing_policy, top_contributors=[
                            {"feature_id": fid, "gene": model.genes.get(fid, ""), "value": values[fid],
                             "coefficient": model.weights[fid], "contribution": val}
                            for fid, val in sorted(contrib, key=lambda p: (-abs(p[1]), p[0]))[:top_n]])
                rows.append(row)
        return {"summary": {"samples": len(samples), "clocks_requested": len(names), "rows": len(rows),
                            "statuses": dict(Counter(r["status"] for r in rows))}, "results": rows, "sources": self.sources,
                "warnings": ["Exploratory coefficient scores, NOT validated biological ages or clinical predictions.",
                             "No automatic normalization, identifier conversion, imputation or age calibration.",
                             "Confirm tissue, assay, identifiers and each original model's preprocessing.",
                             "Missing data reduce coverage per sample; invalid coefficients are not filled with 1."]}
