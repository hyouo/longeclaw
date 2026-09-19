"""Validate the JSON Schema subset used by the native tool contracts."""
from __future__ import annotations
import math
from typing import Any
from .io import SuiteError

def validate(value: Any, schema: dict, path: str = "arguments", depth: int = 0) -> None:
    if depth > 30:
        raise SuiteError("Arguments exceed supported nesting depth.")
    kind = schema.get("type")
    types = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "boolean": isinstance(value, bool),
             "number": type(value) in (int, float), "integer": type(value) is int, "null": value is None}
    if kind and not types.get(kind, False):
        raise SuiteError(f"{path}: expected {kind}.")
    if "enum" in schema and value not in schema["enum"]:
        raise SuiteError(f"{path}: expected one of {schema['enum']}.")
    if kind in ("number", "integer"):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise SuiteError(f"{path}: number must be finite.")
        for key, comparison in (("minimum", lambda a, b: a < b), ("maximum", lambda a, b: a > b),
                                ("exclusiveMinimum", lambda a, b: a <= b)):
            if key in schema and comparison(value, schema[key]):
                raise SuiteError(f"{path}: violates {key}={schema[key]}.")
    elif kind == "string" and len(value) < schema.get("minLength", 0):
        raise SuiteError(f"{path}: empty string is not permitted.")
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", float("inf")):
            raise SuiteError(f"{path}: array length out of bounds.")
        for i, item in enumerate(value):
            validate(item, schema.get("items", {}), f"{path}[{i}]", depth + 1)
    elif kind == "object":
        missing = set(schema.get("required", [])) - value.keys()
        if missing:
            raise SuiteError(f"{path}: missing required fields {sorted(missing)}.")
        properties, extra = schema.get("properties", {}), schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key], f"{path}.{key}", depth + 1)
            elif extra is False:
                raise SuiteError(f"{path}: unknown field {key}.")
            elif isinstance(extra, dict):
                validate(item, extra, f"{path}.{key}", depth + 1)
