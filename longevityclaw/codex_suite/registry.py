"""Tool registry shared by the JSON CLI and tools-only stdio MCP server."""
from __future__ import annotations
import csv
import importlib
import importlib.util
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from . import __version__
from .catalog import Catalog
from .enrichment import enrich, gsea
from .io import SuiteError, fingerprint, json_text, measurements, scoped_path
from .schema import validate

S = {"type": "string", "minLength": 1}
OUTPUT = {"output_path": S}
UPSTREAM_POLICIES = {
    "get_hallmark_info": set(), "map_genes_to_hallmarks": set(), "get_gene_aging_role": set(),
    "explain_grimage_component": set(), "interpret_individual": set(), "lookup_cpg_annotations": set(),
    "rank_pathways": set(), "score_pathway": set(), "find_pathway_synergies": set(),
    "generate_pathway_hypothesis": set(), "discover_aging_modules": set(), "get_pathway_targets": set(),
    "analyze_control_laws": set(), "list_interventions": set(),
    "pubmed_search": {"network"}, "annotate_gene": {"network"}, "validate_targets": {"network"}, "run_gsea": {"network"},
    "query_longevity_llm": {"network", "llm"}, "discover_novel_targets": {"network", "llm"},
    "train_hallmark_model": {"upstream_writes"}, "train_custom_model": {"upstream_writes"},
}

def definition(name: str, description: str, properties: dict, required=(), *, writes=False, network=False) -> dict:
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False},
            "annotations": {"readOnlyHint": not writes, "destructiveHint": name == "upstream_call",
                            "idempotentHint": not writes, "openWorldHint": network}}

TOOLS = [
    definition("doctor", "Check local assets and optional packages; does not read keys or contact services.", {}),
    definition("list_clocks", "List actual clocks and coefficient readiness; not a validation of age prediction.", {"modality": S}),
    definition("clock_details", "Get clock metadata, DOI and top coefficients; contribution is not causality.",
               {"clock_name": S, "top_n": {"type": "integer", "minimum": 1, "maximum": 100}}, ["clock_name"]),
    definition("search_gene", "Find an exact gene symbol in clock coefficients.",
               {"gene": S, "limit": {"type": "integer", "minimum": 1, "maximum": 500}}, ["gene"]),
    definition("score_file", "Score EVERY sample in CSV/TSV. Explicit modality, scale and layout required. Returns uncalibrated scores, NOT ages. Export full cohorts with output_path/output_csv.",
               {"input_path": S, "layout": {"type": "string", "enum": ["long", "samples_by_features", "features_by_samples"]},
                "modality": S, "scale": {"type": "string", "enum": ["beta", "tpm", "log2_tpm", "npx", "native"]},
                "clocks": {"type": "array", "items": S, "minItems": 1, "maxItems": 1000},
                "min_coverage": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                "missing_policy": {"type": "string", "enum": ["reject", "zero"]},
                "top_n": {"type": "integer", "minimum": 0, "maximum": 20}, "output_csv": S, **OUTPUT},
               ["input_path", "layout", "modality", "scale"], writes=True),
    definition("enrich_genes", "Fisher over-representation and BH FDR against local GMT; measured-gene universe required.",
               {"foreground_path": S, "universe_path": S, "gmt_path": S, **OUTPUT}, ["foreground_path", "universe_path"], writes=True),
    definition("gsea", "Preranked GSEA against local GMT; requires pandas/gseapy and signed gene statistics. No network.",
               {"ranked_path": S, "gmt_path": S, "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                "permutations": {"type": "integer", "minimum": 10, "maximum": 10000}, **OUTPUT}, ["ranked_path"], writes=True),
    definition("search_drugage", "Search local DrugAge raw experimental records, not human treatment effects.",
               {"query": S, "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, ["query"]),
    definition("upstream_tools", "List optional upstream Python tool contracts and permissions. Original dependencies required; no Claude agent.", {}),
    definition("upstream_call", "Call an allowlisted original Python tool. Results are scientifically unvalidated. Network, L-LLM and training writes require startup opt-ins; original code can maintain local caches.",
               {"tool_name": {"type": "string", "enum": sorted(UPSTREAM_POLICIES)},
                "arguments": {"type": "object", "additionalProperties": True}, **OUTPUT}, ["tool_name", "arguments"], writes=True, network=True),
]
TOOL_MAP = {tool["name"]: tool for tool in TOOLS}
CSV_FIELDS = ["sample_id", "clock", "modality", "input_scale", "raw_score", "score_unit", "status", "coverage", "n_used", "n_required", "n_missing", "doi"]

def plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, set):
        return [plain(v) for v in sorted(value)]
    if type(value).__module__.startswith("numpy"):
        return plain(value.tolist() if hasattr(value, "tolist") else value.item())
    if isinstance(value, Path):
        return str(value)
    return value

class Suite:
    def __init__(self, repo: Path, workspace: Path, *, allow_network=False, allow_llm=False, allow_upstream_writes=False):
        self.repo, self.workspace = repo.resolve(), workspace.resolve()
        if not self.repo.is_dir() or not self.workspace.is_dir():
            raise SuiteError("repo and workspace must be existing directories.")
        self.permissions = {"network": allow_network, "llm": allow_llm, "upstream_writes": allow_upstream_writes}
        self._catalog = None

    @property
    def catalog(self) -> Catalog:
        if self._catalog is None:
            self._catalog = Catalog(self.repo).load()
        return self._catalog

    def input(self, value: str) -> Path:
        return scoped_path(self.workspace, value)

    def gmt(self, value: str | None) -> Path:
        path = self.input(value) if value else self.repo / "data/msigdb_hallmarks.gmt"
        if not path.is_file():
            raise SuiteError("MSigDB GMT missing; restore data/msigdb_hallmarks.gmt or pass gmt_path.")
        return path

    def doctor(self) -> dict:
        files = {"clock_metadata": "CLOCKSdata/clock_descriptions.csv", "clock_coefficients": "CLOCKSdata/unified_aging_clocks.csv",
                 "msigdb_hallmarks": "data/msigdb_hallmarks.gmt", "drugage": "data/drugage/drugage.csv",
                 "methylation_stats": "data/population/population_stats.npz",
                 "transcriptomic_stats": "data/transcriptomic_population/population_stats.npz",
                 "proteomics_stats": "data/proteomics_population/population_stats.npz",
                 "cpg_annotations": "data/cpg_annotations.sqlite", "mygene_cache": "data/mygene_cache.sqlite"}
        assets = {key: {"path": path, "exists": (self.repo/path).is_file()} for key, path in files.items()}
        packages = {name: importlib.util.find_spec(name) is not None for name in
                    ("numpy", "pandas", "scipy", "pyarrow", "gseapy", "sklearn", "shap", "anthropic")}
        return {"version": __version__, "repo": str(self.repo), "workspace": str(self.workspace),
                "offline_core_files_present": all(assets[k]["exists"] for k in ("clock_metadata", "clock_coefficients")),
                "assets": assets, "optional_packages": packages, "permissions": self.permissions,
                "warnings": ["Stats files do not establish the presence of individual-level training matrices.",
                             "This check does not validate any external service or scientific model."]}

    def upstream(self):
        try:
            module = importlib.import_module("longevityclaw.tools")
        except ImportError as exc:
            raise SuiteError(f"Optional upstream tools cannot load ({exc.name}); run uv sync in a complete clone.") from exc
        if not Path(module.__file__).resolve().is_relative_to(self.repo):
            raise SuiteError("Loaded upstream module is outside repo. Install add-on in that checkout.")
        return module

    def _upstream_call(self, tool_name: str, arguments: dict) -> dict:
        for permission in UPSTREAM_POLICIES[tool_name]:
            if not self.permissions[permission]:
                raise SuiteError(f"Tool requires explicit startup --allow-{permission.replace('_', '-')}.")
        module = self.upstream()
        schemas = {t["name"]: t["input_schema"] for t in module.get_tool_definitions()}
        handlers = module.get_tool_handlers()
        if tool_name not in schemas or tool_name not in handlers:
            raise SuiteError("This upstream checkout does not expose the tool.")
        validate(arguments, {**schemas[tool_name], "additionalProperties": False})
        args = dict(arguments)
        for key, value in args.items():
            if key.endswith("_path") and isinstance(value, str):
                args[key] = str(self.input(value))
        result = plain(handlers[tool_name](**args))
        if isinstance(result, dict) and "error" in result:
            raise SuiteError(f"Upstream tool failed: {result['error']}")
        return {"upstream_tool": tool_name, "result": result,
                "warnings": ["Original upstream result: this adapter does not scientifically validate it.",
                             "LLM scores, control-law rankings and coefficient signs are not treatment evidence."]}

    def _execute(self, name: str, args: dict) -> dict:
        if name == "doctor":
            return self.doctor()
        if name == "list_clocks":
            return self.catalog.list_clocks(**args)
        if name == "clock_details":
            return self.catalog.details(**args)
        if name == "search_gene":
            return self.catalog.search_gene(**args)
        if name == "score_file":
            path, layout = self.input(args.pop("input_path")), args.pop("layout")
            result = self.catalog.score(measurements(path, layout), **args)
            result.update(input=fingerprint(path), parameters={"layout": layout, **args})
            return result
        if name == "enrich_genes":
            return enrich(self.input(args["foreground_path"]), self.input(args["universe_path"]), self.gmt(args.get("gmt_path")))
        if name == "gsea":
            return gsea(self.input(args["ranked_path"]), self.gmt(args.get("gmt_path")), args.get("seed", 42), args.get("permutations", 1000))
        if name == "search_drugage":
            path = self.repo / "data/drugage/drugage.csv"
            if not path.is_file():
                raise SuiteError("Local DrugAge CSV is missing.")
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = [row for row in csv.DictReader(stream) if args["query"].casefold() in " ".join(str(v) for v in row.values()).casefold()]
            limit = args.get("limit", 20)
            return {"query": args["query"], "total": len(rows), "records": rows[:limit], "truncated": len(rows) > limit,
                    "source": fingerprint(path), "warning": "Raw experimental records; check species/dose/study quality. Not human efficacy."}
        if name == "upstream_tools":
            return {"tools": [{**t, "required_permissions": sorted(UPSTREAM_POLICIES[t["name"]])}
                               for t in self.upstream().get_tool_definitions() if t["name"] in UPSTREAM_POLICIES]}
        if name == "upstream_call":
            return self._upstream_call(**args)
        raise SuiteError(f"Unknown tool: {name}")

    def call(self, name: str, arguments: dict) -> dict:
        if name not in TOOL_MAP:
            raise SuiteError(f"Unknown tool: {name}; use tools to inspect available contracts.")
        validate(arguments, TOOL_MAP[name]["inputSchema"])
        args, outputs = dict(arguments), {}
        for key in ("output_path", "output_csv"):
            value = args.pop(key, None)
            if value is not None:
                path = scoped_path(self.workspace, value, exists=False)
                if path.exists() or not path.parent.is_dir():
                    raise SuiteError("Output must be a new file in an existing workspace directory.")
                outputs[key] = path
        if len(set(outputs.values())) != len(outputs):
            raise SuiteError("JSON and CSV output paths must differ.")
        payload = {"ok": True, "tool": name, "suite_version": __version__, "data": self._execute(name, args)}
        encoded = json_text(payload) + "\n"
        created = []
        try:
            for key, path in outputs.items():
                with path.open("x", encoding="utf-8", newline="") as stream:
                    created.append(path)
                    if key == "output_csv":
                        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
                        writer.writeheader()
                        writer.writerows(payload["data"]["results"])
                    else:
                        stream.write(encoded)
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        if outputs:
            return {"ok": True, "tool": name, "suite_version": __version__,
                    "artifacts": {key: fingerprint(path) for key, path in outputs.items()},
                    "summary": payload["data"].get("summary", {}), "warnings": payload["data"].get("warnings", [])}
        data = payload["data"]
        if isinstance(data.get("results"), list) and len(data["results"]) > 200:
            data["total_results"], data["results"], data["preview_only"] = len(data["results"]), data["results"][:200], True
            data.setdefault("warnings", []).append("Only 200 preview rows returned. Use output_path to save all rows.")
        return payload
