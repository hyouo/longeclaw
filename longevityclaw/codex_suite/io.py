"""Strict JSON/tabular input and workspace-scoped paths. Standard library only."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

class SuiteError(ValueError):
    """An actionable input, data or capability error."""

def read_json(text: str) -> Any:
    def reject(value):
        raise SuiteError(f"Non-finite JSON number: {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SuiteError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(text, parse_constant=reject, object_pairs_hook=unique)

def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}

def scoped_path(root: Path, value: str, *, exists: bool = True) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SuiteError("A non-empty local path is required.")
    path = Path(value).expanduser()
    path = (root / path if not path.is_absolute() else path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise SuiteError("Path escapes the configured workspace (including symlinks).")
    if exists and not path.is_file():
        raise SuiteError(f"File does not exist: {value}")
    return path

def table(path: Path) -> tuple[list[str], list[list[str]]]:
    delimiter = "\t" if path.suffix.lower() in (".tsv", ".txt", ".rnk") else ","
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=delimiter)
        try:
            header = [cell.strip() for cell in next(reader)]
        except StopIteration:
            raise SuiteError("Empty input table.") from None
        if not header or any(not cell for cell in header) or len(set(header)) != len(header):
            raise SuiteError("Table headers must be non-empty and unique.")
        rows = []
        for index, row in enumerate(reader, 2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                raise SuiteError(f"Row {index}: expected {len(header)} columns, found {len(row)}.")
            rows.append([cell.strip() for cell in row])
    if not rows:
        raise SuiteError("Table contains no data rows.")
    return header, rows

def number(value: Any, label: str, *, missing: bool = False) -> float | None:
    if missing and (value is None or str(value).strip().lower() in ("", "na", "nan", "null", "none")):
        return None
    if isinstance(value, bool):
        raise SuiteError(f"{label}: boolean is not a numeric measurement.")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        raise SuiteError(f"{label}: expected a finite number.") from None
    if not math.isfinite(result):
        raise SuiteError(f"{label}: infinite or non-finite number.")
    return result

def unique_ids(values: list[str], label: str) -> None:
    if any(not value for value in values) or len(values) != len(set(values)):
        raise SuiteError(f"{label}: empty or duplicate IDs; resolve them explicitly upstream.")

def measurements(path: Path, layout: str) -> dict[str, dict[str, float]]:
    """Read every sample; NA is omitted per sample and reduces coverage."""
    header, rows = table(path)
    samples = {}
    if layout == "long":
        if header != ["feature_id", "value"]:
            raise SuiteError("long format requires exactly feature_id,value.")
        unique_ids([row[0] for row in rows], "Features")
        samples["sample_1"] = {row[0]: value for row in rows
                               if (value := number(row[1], row[0], missing=True)) is not None}
    elif layout == "samples_by_features":
        if header[0] != "sample_id" or len(header) < 2:
            raise SuiteError("samples_by_features requires sample_id as the first column.")
        unique_ids([row[0] for row in rows], "Samples")
        for row in rows:
            samples[row[0]] = {feature: value for feature, text in zip(header[1:], row[1:])
                               if (value := number(text, f"{row[0]}/{feature}", missing=True)) is not None}
    elif layout == "features_by_samples":
        if header[0] != "feature_id" or len(header) < 2:
            raise SuiteError("features_by_samples requires feature_id as the first column.")
        unique_ids([row[0] for row in rows], "Features")
        samples = {sample: {} for sample in header[1:]}
        for row in rows:
            for sample, text in zip(header[1:], row[1:]):
                value = number(text, f"{sample}/{row[0]}", missing=True)
                if value is not None:
                    samples[sample][row[0]] = value
    else:
        raise SuiteError("Unknown layout; specify long, samples_by_features or features_by_samples.")
    if any(not values for values in samples.values()):
        raise SuiteError("Every sample needs at least one finite measurement.")
    return samples

def gene_list(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig") as stream:
        values = [line.strip() for line in stream if line.strip() and not line.lstrip().startswith("#")]
    if values and values[0].lower() in ("gene", "gene_symbol", "feature_id"):
        values = values[1:]
    if not values or any(any(char in value for char in (",", "\t", " ")) for value in values):
        raise SuiteError("Provide one gene symbol per line, optionally with a gene header.")
    return set(values)
