"""
Clock database: loads CLOCKSdata/ and provides lookup, computation, and querying tools.
This is the agent's deep knowledge layer -- it knows every clock, every coefficient,
every gene, and can compute biological age from raw input data.
"""

import os
import math
import pickle
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "CLOCKSdata"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


@dataclass
class ClockInfo:
    name: str
    description: str
    modality: str
    year: int
    n_features: int
    citation: str
    doi: str


@dataclass
class ClockModel:
    """A computable clock: intercept + feature coefficients."""
    name: str
    modality: str
    intercept: float
    coefficients: dict[str, float]  # feature_id -> coefficient
    gene_map: dict[str, str]        # feature_id -> gene_symbol
    data_type: str
    year: int


class ClockDatabase:
    """Loads and indexes all 239 clocks and 430K coefficients."""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.clocks: dict[str, ClockInfo] = {}
        self.models: dict[str, ClockModel] = {}
        self._feature_to_clocks: dict[str, list[str]] = {}
        self._gene_to_clocks: dict[str, list[str]] = {}
        self._modality_index: dict[str, list[str]] = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return self
        if not self._load_from_cache():
            self._load_descriptions()
            self._load_coefficients()
            self._build_indices()
            self._save_cache()
        self._loaded = True
        return self

    def _source_hash(self) -> str:
        h = hashlib.md5()
        for name in ("clock_descriptions.parquet", "unified_aging_clocks.parquet",
                      "clock_descriptions.csv", "unified_aging_clocks.csv"):
            p = self.data_dir / name
            if p.exists():
                h.update(p.name.encode())
                h.update(str(p.stat().st_mtime_ns).encode())
        return h.hexdigest()

    def _cache_path(self) -> Path:
        return CACHE_DIR / "clock_db.pkl"

    def _load_from_cache(self) -> bool:
        cp = self._cache_path()
        if not cp.exists():
            return False
        try:
            with open(cp, "rb") as f:
                data = pickle.load(f)
            if data.get("hash") != self._source_hash():
                return False
            self.clocks = data["clocks"]
            self.models = data["models"]
            self._feature_to_clocks = data["feature_to_clocks"]
            self._gene_to_clocks = data["gene_to_clocks"]
            self._modality_index = data["modality_index"]
            return True
        except Exception:
            return False

    def _save_cache(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "hash": self._source_hash(),
            "clocks": self.clocks,
            "models": self.models,
            "feature_to_clocks": self._feature_to_clocks,
            "gene_to_clocks": self._gene_to_clocks,
            "modality_index": self._modality_index,
        }
        with open(self._cache_path(), "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_descriptions(self):
        pq = self.data_dir / "clock_descriptions.parquet"
        if pq.exists():
            df = pd.read_parquet(pq)
        else:
            df = pd.read_csv(self.data_dir / "clock_descriptions.csv")
        for row in df.itertuples(index=False):
            self.clocks[row.clock_name] = ClockInfo(
                name=row.clock_name,
                description=row.description,
                modality=row.modality,
                year=int(row.year),
                n_features=int(row.n_features),
                citation=row.citation,
                doi=row.doi,
            )

    def _load_coefficients(self):
        pq = self.data_dir / "unified_aging_clocks.parquet"
        if pq.exists():
            df = pd.read_parquet(pq)
        else:
            df = pd.read_csv(self.data_dir / "unified_aging_clocks.csv", low_memory=False)
            df["coefficient"] = pd.to_numeric(df["coefficient"], errors="coerce").fillna(1.0)

        clock_names = df["clock_name"].values
        feature_ids = df["feature_id"].values
        coefficients = df["coefficient"].values
        genes = df["gene_symbol"].values
        modalities = df["modality"].values
        data_types = df["data_type"].values
        years = df["year"].values

        clock_rows: dict[str, tuple[float, dict, dict, str, str, int]] = {}

        for i in range(len(clock_names)):
            cname = clock_names[i]
            fid = feature_ids[i]
            coeff = coefficients[i]
            gene = genes[i]

            if cname not in clock_rows:
                clock_rows[cname] = (0.0, {}, {}, modalities[i], data_types[i], int(years[i]))

            intercept, coeff_dict, gene_map, mod, dt, yr = clock_rows[cname]

            if fid == "intercept" or gene == "INTERCEPT":
                clock_rows[cname] = (coeff, coeff_dict, gene_map, mod, dt, yr)
            else:
                coeff_dict[fid] = coeff
                if isinstance(gene, str) and gene and gene != "INTERCEPT":
                    gene_map[fid] = gene

        for cname, (intercept, coeff_dict, gene_map, mod, dt, yr) in clock_rows.items():
            self.models[cname] = ClockModel(
                name=cname,
                modality=mod,
                intercept=float(intercept),
                coefficients=coeff_dict,
                gene_map=gene_map,
                data_type=dt,
                year=yr,
            )

    def _build_indices(self):
        self._modality_index = defaultdict(list)
        self._feature_to_clocks = defaultdict(list)
        self._gene_to_clocks = defaultdict(list)

        for cname, model in self.models.items():
            self._modality_index[model.modality].append(cname)
            for fid in model.coefficients:
                self._feature_to_clocks[fid].append(cname)
            seen_genes = set()
            for fid, gene in model.gene_map.items():
                key = gene.upper()
                if key not in seen_genes:
                    self._gene_to_clocks[key].append(cname)
                    seen_genes.add(key)

        self._modality_index = dict(self._modality_index)
        self._feature_to_clocks = dict(self._feature_to_clocks)
        self._gene_to_clocks = dict(self._gene_to_clocks)

    # ── Query methods (used as agent tools) ────────────────────────────

    def list_clocks(self, modality: str | None = None) -> list[ClockInfo]:
        if modality:
            names = self._modality_index.get(modality, [])
            return [self.clocks[n] for n in names if n in self.clocks]
        return list(self.clocks.values())

    def get_clock_info(self, clock_name: str) -> ClockInfo | None:
        return self.clocks.get(clock_name)

    def get_clock_model(self, clock_name: str) -> ClockModel | None:
        return self.models.get(clock_name)

    def list_modalities(self) -> dict[str, int]:
        return {mod: len(names) for mod, names in self._modality_index.items()}

    def find_clocks_by_gene(self, gene: str) -> list[str]:
        return self._gene_to_clocks.get(gene.upper(), [])

    def find_clocks_by_feature(self, feature_id: str) -> list[str]:
        return self._feature_to_clocks.get(feature_id, [])

    def get_top_features(self, clock_name: str, n: int = 20) -> list[dict]:
        model = self.models.get(clock_name)
        if not model:
            return []
        sorted_feats = sorted(model.coefficients.items(),
                              key=lambda x: abs(x[1]), reverse=True)[:n]
        return [
            {
                "feature_id": fid,
                "coefficient": coeff,
                "gene": model.gene_map.get(fid, ""),
                "direction": "accelerates aging" if coeff > 0 else "decelerates aging",
            }
            for fid, coeff in sorted_feats
        ]

    def find_applicable_clocks(self, available_features: set[str]) -> list[dict]:
        """Given user's available feature IDs, find which clocks can be computed."""
        results = []
        for cname, model in self.models.items():
            required = set(model.coefficients.keys())
            if not required:
                continue
            overlap = required & available_features
            coverage = len(overlap) / len(required)
            if coverage >= 0.9:  # allow up to 10% missing (treat as 0)
                results.append({
                    "clock": cname,
                    "modality": model.modality,
                    "coverage": coverage,
                    "n_matched": len(overlap),
                    "n_required": len(required),
                    "description": self.clocks[cname].description if cname in self.clocks else "",
                })
        results.sort(key=lambda x: (-x["coverage"], x["clock"]))
        return results

    # ── Compute methods ────────────────────────────────────────────────

    def compute_clock(self, clock_name: str, feature_values: dict[str, float]) -> dict:
        """Compute a clock's raw score from feature values. Returns score + contribution breakdown."""
        model = self.models.get(clock_name)
        if not model:
            return {"error": f"Unknown clock: {clock_name}"}

        score = model.intercept
        contributions = []
        missing = []

        for fid, coeff in model.coefficients.items():
            if fid in feature_values:
                val = feature_values[fid]
                contrib = coeff * val
                score += contrib
                contributions.append({
                    "feature_id": fid,
                    "gene": model.gene_map.get(fid, ""),
                    "value": val,
                    "coefficient": coeff,
                    "contribution": contrib,
                })
            else:
                missing.append(fid)

        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return {
            "clock": clock_name,
            "raw_score": score,
            "intercept": model.intercept,
            "n_features_used": len(model.coefficients) - len(missing),
            "n_features_missing": len(missing),
            "top_contributors": contributions[:20],
            "missing_features": missing[:10] if missing else [],
        }

    def compute_phenoage_blood(self, blood_markers: dict[str, float]) -> dict:
        """Compute Levine PhenoAge from 9 blood biomarkers + chronological age.

        Units per Levine 2018 (NHANES III):
          albumin_g_dl        Albumin (g/dL)
          creatinine_umol_l   Creatinine (umol/L)
          glucose_mmol_l      Glucose (mmol/L)
          crp_mg_dl           C-reactive protein (mg/dL) -- log-transformed internally
          lymphocyte_pct      Lymphocyte (%)
          mcv_fl              Mean cell volume (fL)
          rdw_pct             Red cell distribution width (%)
          alkaline_phosphatase_u_l  Alkaline phosphatase (U/L)
          wbc_10e3_ul         White blood cell count (1000 cells/uL)
          age                 Chronological age (years)
        """
        required = ["albumin_g_dl", "creatinine_umol_l", "glucose_mmol_l",
                     "crp_mg_dl", "lymphocyte_pct", "mcv_fl", "rdw_pct",
                     "alkaline_phosphatase_u_l", "wbc_10e3_ul", "age"]
        missing = [k for k in required if k not in blood_markers]
        if missing:
            return {"error": f"Missing blood markers: {missing}",
                    "required_keys": required,
                    "unit_help": {
                        "albumin_g_dl": "Albumin in g/dL (typical: 3.5-5.5)",
                        "creatinine_umol_l": "Creatinine in umol/L (typical: 60-110). To convert mg/dL->umol/L: multiply by 88.4",
                        "glucose_mmol_l": "Glucose in mmol/L (typical: 3.9-6.1). To convert mg/dL->mmol/L: divide by 18.016",
                        "crp_mg_dl": "CRP in mg/dL (typical: 0.01-0.5). To convert mg/L->mg/dL: divide by 10",
                        "lymphocyte_pct": "Lymphocyte percentage (typical: 20-40)",
                        "mcv_fl": "Mean cell volume in fL (typical: 80-100)",
                        "rdw_pct": "Red cell distribution width % (typical: 11.5-14.5)",
                        "alkaline_phosphatase_u_l": "Alkaline phosphatase in U/L (typical: 44-147)",
                        "wbc_10e3_ul": "White blood cells in 1000/uL (typical: 4.5-11.0)",
                        "age": "Chronological age in years",
                    }}

        m = blood_markers
        ln_crp = math.log(max(m["crp_mg_dl"], 0.001))

        # Step 1: Linear predictor (xb)
        # Coefficients from Levine 2018 Supplementary Table S6
        xb = (-19.9067
              - 0.0336 * m["albumin_g_dl"]
              + 0.0095 * m["creatinine_umol_l"]
              + 0.1953 * m["glucose_mmol_l"]
              + 0.0954 * ln_crp
              - 0.0120 * m["lymphocyte_pct"]
              + 0.0268 * m["mcv_fl"]
              + 0.3306 * m["rdw_pct"]
              + 0.0019 * m["alkaline_phosphatase_u_l"]
              + 0.0554 * m["wbc_10e3_ul"]
              + 0.0804 * m["age"])

        # Step 2: 10-year mortality probability via Gompertz model
        gamma = 0.0076927
        t_months = 120
        gompertz_h0 = (math.exp(gamma * t_months) - 1) / gamma
        mortality_score = 1.0 - math.exp(-math.exp(xb) * gompertz_h0)

        # Step 3: Convert mortality to PhenoAge
        # Numerically stable: when mortality_score ≈ 1, use log-space
        if mortality_score > 0.9999999:
            log_1_minus_m = -math.exp(xb) * gompertz_h0
        else:
            log_1_minus_m = math.log(1.0 - mortality_score)

        # log_1_minus_m is negative (since 0 < 1-M < 1)
        # Formula: PhenoAge = 141.50225 + ln(-0.00553 * ln(1-M)) / 0.090165
        phenoage = 141.50225 + math.log(-0.00553 * log_1_minus_m) / 0.090165

        age_gap = phenoage - m["age"]

        # Component contributions for interpretation
        components = [
            ("Albumin", m["albumin_g_dl"], -0.0336, "g/dL", "lower is worse (inflammation/malnutrition)"),
            ("Creatinine", m["creatinine_umol_l"], 0.0095, "umol/L", "higher means kidney decline"),
            ("Glucose", m["glucose_mmol_l"], 0.1953, "mmol/L", "higher means metabolic dysregulation"),
            ("ln(CRP)", ln_crp, 0.0954, "ln(mg/dL)", "higher means chronic inflammation"),
            ("Lymphocyte %", m["lymphocyte_pct"], -0.0120, "%", "lower means immune aging"),
            ("MCV", m["mcv_fl"], 0.0268, "fL", "higher may indicate B12/folate deficiency"),
            ("RDW", m["rdw_pct"], 0.3306, "%", "higher means red cell size variability (inflammation)"),
            ("Alk Phos", m["alkaline_phosphatase_u_l"], 0.0019, "U/L", "higher may indicate liver/bone issues"),
            ("WBC", m["wbc_10e3_ul"], 0.0554, "10^3/uL", "higher means immune activation/inflammation"),
            ("Chron. Age", m["age"], 0.0804, "years", "baseline contribution"),
        ]
        contributions = []
        for name, val, coeff, unit, interp in components:
            contributions.append({
                "marker": name,
                "value": round(val, 3),
                "unit": unit,
                "coefficient": coeff,
                "contribution": round(coeff * val, 3),
                "interpretation": interp,
            })
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return {
            "clock": "PhenoAge (blood)",
            "phenoage": round(phenoage, 2),
            "chronological_age": m["age"],
            "age_gap": round(age_gap, 2),
            "direction": "older" if age_gap > 0 else "younger",
            "mortality_score": round(mortality_score, 6),
            "xb": round(xb, 4),
            "component_contributions": contributions,
        }

    def compute_horvath_age(self, beta_values: dict[str, float]) -> dict:
        """Compute Horvath 353 clock. Input: dict of CpG ID -> beta value."""
        result = self.compute_clock("horvath2013", beta_values)
        if "error" in result:
            return result

        # Horvath uses anti-log transformation: age = inverse_F(score)
        # F(age) = log(age+1) - log(21+1) for adult calibration
        # For simplicity, raw score is already in transformed space
        # The intercept encodes the calibration
        raw = result["raw_score"]

        # Horvath's transform: if score < 0 -> age = 21*exp(score)-1, else age = 21*score + 20
        # Actually the standard transform from the paper:
        if raw < 0:
            age = 21 * math.exp(raw) - 1
        else:
            age = 21 * raw + 20

        result["horvath_age"] = round(age, 2)
        result["note"] = "Age derived via Horvath anti-log transform of raw elastic net score"
        return result


# Singleton for convenience
_db: ClockDatabase | None = None


def get_db() -> ClockDatabase:
    global _db
    if _db is None:
        _db = ClockDatabase().load()
    return _db
