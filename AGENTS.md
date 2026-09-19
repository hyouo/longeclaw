# LongevityClaw — Codex tools

Read `docs/CODEX.md` before using the new adapter. This file adds Codex-specific
instructions; the original application and its scientific code are unchanged.

## Entry points

From the repository root:

```bash
python3 scripts/longeclaw_codex.py doctor
python3 scripts/longeclaw_codex.py tools
python3 scripts/longeclaw_codex.py call list_clocks --args '{"modality":"transcriptomics"}'
python3 -m pytest -q tests/codex_suite
```

The core adapter needs Python >=3.11, but no Anthropic key or third-party Python
packages. Native GSEA is optional; its dependencies are in
`codex-requirements-analysis.txt`. The optional upstream bridge needs the original
project environment (`uv sync`) and all data required by the selected function.
Do not start `longevityclaw` or instantiate its Claude agent for ordinary analysis.

## Analysis rules

- Run `doctor`; report missing assets rather than claiming all reference data are available.
- Inspect the actual table. Require explicit modality, expression scale and matrix orientation.
  Do not infer RNA versus protein from gene symbols alone. Do not pass raw counts as TPM.
- Use `bulk` / `score_file` for cohorts; never silently take the first sample only.
- Native scores are **uncalibrated coefficient scores**, not validated biological ages.
  Coverage does not validate assay, tissue, preprocessing, normalization or age transforms.
- Preserve per-sample status, coverage, missing-feature counts, input hashes and source hashes.
  Never convert coefficient signs into causal aging or treatment claims.
- ORA requires a measured/tested-gene background. GSEA requires a full signed contrast statistic.
  Do not fabricate a contrast, infer differential expression from one sample, or substitute raw
  expression levels for differential-expression statistics without a justified analysis design.
- Keep raw inputs unchanged. Export complete cohorts using new JSON/CSV filenames.
  A response with `preview_only` contains only a preview, not the complete result.
- Native input/output paths are confined to `--workspace` (default: current directory).
  Configure the intended workspace instead of bypassing it with symlinks.
- Network, external L-LLM and original training writes are opt-in. Obtain authorization before
  enabling them or transmitting data. These flags are application policy, not an OS sandbox;
  original Python functions may maintain local caches.
- DrugAge records, LLM-generated scores and control-law rankings are research evidence or
  hypotheses; do not present them as established human efficacy or medical advice.

## Development and verification

The adapter lives in `longevityclaw/codex_suite/`; CLI and MCP share one registry.
Keep stdout machine-readable. Third-party diagnostics must go to stderr.
Maintain regression tests for bulk orientation, missing data, modality isolation,
permission gates, JSON contracts, output protection and MCP lifecycle.
The adapter tests do not certify the original clocks, clinical accuracy or Codex client integration.
Do not edit upstream scientific algorithms or existing user data to make adapter tests pass.
