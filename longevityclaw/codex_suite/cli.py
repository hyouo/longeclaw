"""Non-interactive JSON CLI. No Claude/OpenAI API client is instantiated."""
from __future__ import annotations
import argparse
import contextlib
import sys
from pathlib import Path
from .io import json_text, read_json
from .registry import Suite, TOOLS

def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="LongevityClaw tools for Codex (offline-first)")
    result.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    result.add_argument("--workspace", type=Path, default=Path.cwd())
    result.add_argument("--allow-network", action="store_true")
    result.add_argument("--allow-llm", action="store_true")
    result.add_argument("--allow-upstream-writes", action="store_true")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("doctor", "tools", "mcp"):
        commands.add_parser(name)
    call = commands.add_parser("call")
    call.add_argument("name")
    args = call.add_mutually_exclusive_group()
    args.add_argument("--args", default="{}")
    args.add_argument("--args-file")
    bulk = commands.add_parser("bulk", help="Score every sample; no automatic age calibration")
    bulk.add_argument("--input", required=True, dest="input_path")
    bulk.add_argument("--layout", required=True, choices=["long", "samples_by_features", "features_by_samples"])
    bulk.add_argument("--modality", required=True)
    bulk.add_argument("--scale", required=True, choices=["beta", "tpm", "log2_tpm", "npx", "native"])
    bulk.add_argument("--clocks", nargs="+")
    bulk.add_argument("--min-coverage", type=float, default=1.0)
    bulk.add_argument("--missing-policy", choices=["reject", "zero"], default="reject")
    bulk.add_argument("--top-n", type=int, default=10)
    bulk.add_argument("--output", dest="output_path")
    bulk.add_argument("--output-csv")
    return result

def main(argv: list[str] | None = None) -> int:
    options = parser().parse_args(argv)
    try:
        suite = Suite(options.repo, options.workspace, allow_network=options.allow_network,
                      allow_llm=options.allow_llm, allow_upstream_writes=options.allow_upstream_writes)
        if options.command == "mcp":
            from .mcp_server import Server
            Server(suite).serve(sys.stdin.buffer, sys.stdout)
            return 0
        with contextlib.redirect_stdout(sys.stderr):
            if options.command == "tools":
                payload = {"ok": True, "tools": TOOLS}
            elif options.command == "doctor":
                payload = suite.call("doctor", {})
            elif options.command == "call":
                raw = suite.input(options.args_file).read_text(encoding="utf-8") if options.args_file else options.args
                payload = suite.call(options.name, read_json(raw))
            else:
                keys = ("input_path", "layout", "modality", "scale", "clocks", "min_coverage", "missing_policy", "top_n", "output_path", "output_csv")
                payload = suite.call("score_file", {key: getattr(options, key) for key in keys if getattr(options, key) is not None})
        print(json_text(payload))
        return 0
    except (KeyboardInterrupt, BrokenPipeError):
        return 130
    except Exception as exc:
        print(json_text({"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
