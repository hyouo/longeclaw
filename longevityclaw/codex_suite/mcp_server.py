"""Tools-only MCP stdio, protocol 2025-06-18. No network listener.

Sequential initialize, notifications/initialized, ping, tools/list, tools/call.
No background jobs, resources, sampling or HTTP transport. Use CLI for long jobs.
"""
from __future__ import annotations
import contextlib
import json
import sys
from typing import Any, BinaryIO, TextIO
from . import __version__
from .io import SuiteError, json_text, read_json
from .registry import Suite, TOOLS, TOOL_MAP

PROTOCOL_VERSION = "2025-06-18"
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
INSTRUCTIONS = (
    "Run doctor first. Specify bulk modality, scale and layout explicitly. score_file returns UNCALIBRATED "
    "coefficient scores, not biological ages. Use output paths for all cohort results. No clinical inference. "
    "User-data paths are limited to workspace. Optional upstream code can maintain caches; network, L-LLM "
    "and model-training writes need operator opt-ins. Do not upload sample data without user authorization."
)

def rpc_error(identifier: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": identifier, "error": {"code": code, "message": message}}

class Server:
    def __init__(self, suite: Suite):
        self.suite, self.initialized, self.ready = suite, False, False

    def handle(self, request: Any) -> dict | None:
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or not isinstance(request.get("method"), str):
            return rpc_error(None, -32600, "Invalid JSON-RPC request")
        method = request["method"]
        # Notifications never execute tools or write files.
        if "id" not in request:
            if method == "notifications/initialized" and self.initialized:
                self.ready = True
            return None
        identifier, params = request["id"], request.get("params", {})
        if type(identifier) not in (int, str):
            return rpc_error(None, -32600, "Request id must be an integer or string")
        if not isinstance(params, dict):
            return rpc_error(identifier, -32602, "params must be an object")
        if method == "initialize":
            if self.initialized:
                return rpc_error(identifier, -32600, "Already initialized")
            client = params.get("clientInfo")
            if (not isinstance(params.get("protocolVersion"), str) or not isinstance(params.get("capabilities"), dict)
                or not isinstance(client, dict) or not isinstance(client.get("name"), str)
                or not isinstance(client.get("version"), str)):
                return rpc_error(identifier, -32602, "initialize requires protocolVersion, capabilities and clientInfo")
            self.initialized = True
            result = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": "longeclaw-codex", "version": __version__}, "instructions": INSTRUCTIONS}
        elif method == "ping":
            result = {}
        elif not self.ready:
            return rpc_error(identifier, -32002, "Initialize and send notifications/initialized first")
        elif method == "tools/list":
            if params.get("cursor") is not None:
                return rpc_error(identifier, -32602, "No further pages")
            result = {"tools": TOOLS}
        elif method == "tools/call":
            name, args = params.get("name"), params.get("arguments", {})
            if not isinstance(name, str) or name not in TOOL_MAP or not isinstance(args, dict):
                return rpc_error(identifier, -32602, "Unknown tool or invalid arguments object")
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    payload = self.suite.call(name, args)
            except Exception as exc:
                payload = {"ok": False, "tool": name, "error": {"type": type(exc).__name__, "message": str(exc)}}
            result = {"content": [{"type": "text", "text": json_text(payload)}],
                      "structuredContent": payload, "isError": not payload["ok"]}
        else:
            return rpc_error(identifier, -32601, "Method not found")
        return {"jsonrpc": "2.0", "id": identifier, "result": result}

    def serve(self, source: BinaryIO, sink: TextIO) -> None:
        while True:
            raw = source.readline(MAX_MESSAGE_BYTES + 1)
            if not raw:
                return
            if len(raw) > MAX_MESSAGE_BYTES:
                while raw and not raw.endswith(b"\n"):
                    raw = source.readline(MAX_MESSAGE_BYTES + 1)
                response = rpc_error(None, -32600, "Message exceeds 16 MiB limit")
            else:
                try:
                    request = read_json(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError, SuiteError, RecursionError, ValueError):
                    response = rpc_error(None, -32700, "Invalid UTF-8 JSON message")
                else:
                    response = self.handle(request)
            if response is not None:
                sink.write(json_text(response) + "\n")
                sink.flush()
