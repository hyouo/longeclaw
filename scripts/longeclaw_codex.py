#!/usr/bin/env python3
"""Repository-local Codex entry point; no installation of the Claude app needed."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longevityclaw.codex_suite.cli import main
if __name__ == "__main__":
    raise SystemExit(main())
