# LongevityClaw for Codex

This fork adds a **local, offline-first analysis toolkit** that Codex can call
through a repository Skill, a JSON CLI, or a stdio MCP server. Core tools require
Python 3.11+ only; no Anthropic key or Claude agent is needed.

## Get started

Until the integration PR is merged, use the `codex-toolkit` branch:

```bash
git clone --branch codex-toolkit https://github.com/hyouo/longeclaw.git
cd longeclaw
python3 scripts/longeclaw_codex.py doctor
python3 scripts/longeclaw_codex.py tools
```

Start Codex in this checkout and invoke `$longeclaw`. To register MCP tools for
the local Codex CLI or IDE:

```bash
codex mcp add longeclaw -- python3 "$PWD/scripts/longeclaw_codex.py" --workspace "$PWD" mcp
codex mcp list
codex
```

These commands configure **your local Codex environment**, not the hosted ChatGPT
web environment. No credentials or global Codex configuration are committed.

## Bulk analysis

For a preprocessed TPM table whose first header is `feature_id` and subsequent
columns are samples:

```bash
python3 scripts/longeclaw_codex.py bulk \
  --input bulk_tpm.tsv --layout features_by_samples \
  --modality transcriptomics --scale tpm \
  --output bulk_scores.json --output-csv bulk_scores.csv
```

All samples are processed. Scores are **uncalibrated coefficient scores, not
validated biological ages**. Confirm assay, tissue, identifiers and preprocessing
against the selected clocks. No automatic count normalization is performed.

## Included

- Clock catalog, coefficient details, exact gene lookup and bulk model scoring.
- Fisher enrichment with measured-gene background and BH FDR; optional local GSEA.
- DrugAge experimental records and an opt-in bridge to allowlisted upstream tools.
- Workspace-scoped file access, no-overwrite exports, explicit network/LLM permissions.
- Regression tests and GitHub Actions checks for Python 3.11/3.13 and optional GSEA.

[Full guide (中文)](docs/CODEX.md) · [Codex instructions](AGENTS.md) ·
[Repository Skill](.agents/skills/longeclaw/SKILL.md)

The original application, databases, dependency lockfile and scientific algorithms
are retained unchanged. This adapter does not certify their scientific accuracy.
