---
name: longeclaw
description: 使用本地 LongevityClaw 数据库分析预处理后的 bulk RNA-seq、甲基化或蛋白组数据，批量计算模型原始分数、检查覆盖率、检索基因和 DrugAge、做通路富集。用于本仓库的研究分析；不用于原始 FASTQ 处理、临床诊断或未经验证的生物年龄推断。
---

# LongevityClaw

Read `AGENTS.md` and `docs/CODEX.md` at the repository root.
Use the configured `longeclaw` MCP tools when available. Otherwise call the JSON CLI:

```bash
python3 scripts/longeclaw_codex.py doctor
python3 scripts/longeclaw_codex.py tools
```

Do not start the original Claude chat agent. The native adapter does not need an Anthropic key.

## Workflow

1. Inspect asset availability and the user's actual file. Establish the assay, tissue/species,
   identifiers, units/normalization and matrix layout. Ask only for material missing information.
   Do not transform raw counts into TPM without gene lengths and a justified preprocessing plan.
2. Inspect applicable clock metadata/DOIs. Run all intended samples using `score_file` / `bulk`,
   with explicit modality and scale. Default to complete feature coverage and no imputation.
3. For selected genes, use `enrich_genes` only with the measured/tested-gene universe. For GSEA,
   use full signed, contrast-specific gene statistics and a local GMT. Missing optional dependencies
   or reference matrices are blockers for that analysis, not a reason to fabricate results.
4. Export complete results with `output_path` and, for bulk, `output_csv`. Use fresh filenames.
   Summarize sample count, scoring/skipped/error statuses, coverage, source hashes and limitations.

Bulk example, only after verifying the file is a features-by-samples TPM table:

```bash
python3 scripts/longeclaw_codex.py bulk \
  --input bulk_tpm.tsv --layout features_by_samples \
  --modality transcriptomics --scale tpm \
  --output bulk_scores.json --output-csv bulk_scores.csv
```

The first header in that table must be `feature_id`. A samples-by-features table instead
uses `sample_id` as its first header. Metadata belongs in a separate file.

## Interpretation and privacy

`raw_score` is an uncalibrated model score; do not rename it biological age or derive age gaps.
No automatic normalization, Ensembl mapping or model-specific age conversion is performed.
Clock coefficients describe score contributions, not causal mechanisms.

Network, L-LLM and original training writes require separate explicit startup permissions.
Do not enable these, upload data, or send data to external models without authorization.
Do not expose credentials in prompts, logs or committed files. The original bridge is optional
and its outputs are not scientifically revalidated by this adapter.
