# LongevityClaw Codex 适配层

版本：0.1.0。审阅的上游基线：`dd6861cb280d5ffba0c81766a405a702d0b3acef`。
本适配层以新增文件的方式安装，不替换原有 `agent.py`、`tools.py`、模型算法、依赖锁文件或数据库。

当前目标仓库：`hyouo/longeclaw`；适配分支：`codex-toolkit`。
初次使用参见 [README_CODEX.md](../README_CODEX.md)。

## 1. 先检查本地环境

在已安装适配层的完整仓库根目录执行：

```bash
python3 scripts/longeclaw_codex.py doctor
python3 scripts/longeclaw_codex.py tools
```

原生目录查询、系数评分、Fisher 富集、DrugAge 检索、JSON CLI 和 MCP 仅依赖 Python 3.11+ 标准库。
无需启动原始聊天界面，无需 Anthropic API key。Codex 自身的登录和使用权限另行配置。
`doctor` 报告的是文件存在情况，不是模型准确性认证；有人群统计文件也不保证存在完整训练矩阵。

## 2. 连接 Codex

### 仓库 Skill

在本仓库打开 Codex。Skill 位于 `.agents/skills/longeclaw/SKILL.md`，可使用 `$longeclaw` 指定：

```text
$longeclaw 检查 bulk_tpm.tsv 的结构和可用时钟，对所有样本计算转录组原始模型分数，
输出 JSON 和 CSV，报告覆盖率和跳过原因。不要联网，也不要把原始分数称为生物年龄。
```

### MCP 工具

在仓库根目录注册本地 stdio 服务：

```bash
codex mcp add longeclaw -- python3 "$PWD/scripts/longeclaw_codex.py" --workspace "$PWD" mcp
codex mcp list
```

也可以把 `python3` 换成安装了可选分析依赖的虚拟环境解释器绝对路径。
注册后使用新的 Codex 会话确认工具可见。本实现提供 MCP `2025-06-18` 的 tools-only stdio 子集：
初始化、ping、工具列表、工具调用和结构化结果。请求串行执行；不提供 HTTP 服务、资源订阅、
采样或后台任务。长时间训练和大规模队列应走 CLI，避免客户端工具超时。
本次构建验证了协议子进程交互，未在真实 Codex 客户端内联调。

## 3. 工具清单

| 工具 | 分析内容 | 额外要求 |
|---|---|---|
| `doctor` | 数据文件、包和权限检查 | 无 |
| `list_clocks` | 实际时钟目录及系数可用性 | 上游 CSV |
| `clock_details` | 系数、截距、文献信息 | 上游 CSV |
| `search_gene` | 基因在时钟中的系数记录 | 上游 CSV |
| `score_file` | 单样本或全部 bulk 样本的原始分数、贡献和覆盖率 | 已预处理的 CSV/TSV |
| `enrich_genes` | Fisher 单侧富集和 BH FDR | 前景、实测背景、GMT |
| `gsea` | 基于有符号排序统计量的 GSEA | pandas、gseapy、GMT |
| `search_drugage` | 原始化合物实验记录检索 | 本地 DrugAge CSV |
| `upstream_tools` | 可选原函数及参数目录 | 原项目依赖 |
| `upstream_call` | 直接调用白名单内的原 Python 工具 | 依赖、参考数据及相应授权 |

通用调用使用 JSON 参数：

```bash
python3 scripts/longeclaw_codex.py call list_clocks --args '{"modality":"transcriptomics"}'
python3 scripts/longeclaw_codex.py call search_gene --args '{"gene":"FOXO3"}'
```

参数也可用 `--args-file params.json` 传入，文件必须位于工作目录范围内。
stdout 只输出机器可读 JSON，诊断信息发往 stderr；分析错误退出码为 2。

## 4. Bulk 输入与输出

`features_by_samples`（TSV，首列名必须为 `feature_id`）：

```text
feature_id\tsample_A\tsample_B
GENE1\t1.2\t2.3
GENE2\t3.4\t4.5
```

这里的 `\t` 表示实际制表符。示例数值是格式示意，不是生物学数据。
`samples_by_features` 则要求首列名为 `sample_id`，每行一个样本。
`long` 为单样本两列表，表头固定 `feature_id,value`。样本元数据请放在独立文件中。

```bash
python3 scripts/longeclaw_codex.py bulk \
  --input bulk_tpm.tsv --layout features_by_samples \
  --modality transcriptomics --scale tpm \
  --output bulk_scores.json --output-csv bulk_scores.csv
```

可加 `--clocks CLOCK_NAME ...` 限定时钟；先从 `list_clocks` 获取实际名字。
甲基化使用 `--modality methylation --scale beta`；Olink NPX 数据可使用
`--modality proteomics --scale npx`。选择输入尺度并不意味着该尺度适用于所有匹配的时钟，
仍需按原论文确认平台、组织和预处理。基因名称与 ENSG、蛋白标识不会自动映射。

默认要求完整特征覆盖，不填补缺失值。若用户明确选择探索性的缺失置零计算，必须同时指定：

```bash
--min-coverage 0.95 --missing-policy zero
```

空值会降低对应样本的覆盖率；非法数字、重复样本或重复特征会明确报错。
原始 counts 不在支持尺度中；这里不做测序定量、归一化、差异表达或年龄校准。
不需要强行把不适用的输入改成可计算格式。

每个样本/时钟都有 `status`、`coverage`、特征数和 `raw_score`。
无效系数、覆盖不足、非有限结果会保留状态，分数为 null。
原生计算不把缺失的模型系数替换成 1，也不加载原来的 pickle 缓存。

JSON 保存完整贡献、参数、输入和数据库 SHA256；CSV 保存汇总行。
输出文件不得已存在，父目录须先创建。未指定文件时，超过 200 行只返回明确标记的预览；
批量任务应始终导出文件。MCP 服务复用首次载入的数据库快照；数据变更后重启服务。

## 5. 通路分析

前景与背景文件每行一个基因符号，背景必须对应本实验中可能被选中的实测/检验基因：

```bash
python3 scripts/longeclaw_codex.py call enrich_genes --args \
  '{"foreground_path":"selected_genes.txt","universe_path":"tested_genes.txt","output_path":"ora.json"}'
```

默认读取 `data/msigdb_hallmarks.gmt`；自定义集合通过 `gmt_path` 提供。
FDR 校正在所有有背景基因覆盖的集合上进行，包括零命中的集合。

GSEA 可选依赖：

```bash
python3 -m pip install -r codex-requirements-analysis.txt
python3 scripts/longeclaw_codex.py call gsea --args \
  '{"ranked_path":"contrast_ranked.tsv","seed":42,"permutations":1000,"output_path":"gsea.json"}'
```

排序文件首行是 `gene<TAB>score`，包含完整有符号对比统计量，不能只给显著基因或未经解释的表达值。
GSEA 使用本地 GMT，不自动下载集合。缺少依赖时返回明确错误。

## 6. 可选上游工具

安装原项目环境后，可直接使用原 Python 函数，不经过 Claude：

```bash
uv sync
uv run python scripts/longeclaw_codex.py call upstream_tools
uv run python scripts/longeclaw_codex.py call upstream_call --args \
  '{"tool_name":"get_hallmark_info","arguments":{"hallmark_id":"cellular_senescence"}}'
```

网络工具需要 `--allow-network`；外部 longevity LLM 还需要 `--allow-llm`；
原训练工具需要 `--allow-upstream-writes`。这些全局选项必须置于 `call` 或 `mcp` 前。
例如已获授权的文献检索：

```bash
uv run python scripts/longeclaw_codex.py --allow-network call upstream_call --args \
  '{"tool_name":"pubmed_search","arguments":{"query":"epigenetic aging clock","max_results":3}}'
```

本适配层没有修复或认证原模型的单位、校准、L-LLM 后端配置和科学假设。
原函数所需的参考矩阵、SQLite 数据库或服务凭据不一定包含在仓库中。
原调用可能维护缓存；权限开关是调用策略，不是操作系统级隔离。

## 7. 验证范围

```bash
python3 -m pip install 'pytest>=8,<10'
python3 -m pytest -q tests/codex_suite -ra
```

测试覆盖合成矩阵上的逐样本计算、方向转换、数据异常、覆盖率、模态隔离、
Fisher 独立数值对照、输出保护、CLI 和 MCP 协议子进程。
可选 GSEA 与真实上游资产测试在环境缺失时明确跳过。
这不代表所有时钟、临床准确性、外部服务或实际 Codex 客户端已经验证。

## 参考

- 上游源码：https://github.com/Insilico-org/longeclaw/tree/dd6861cb280d5ffba0c81766a405a702d0b3acef
- Codex Skills：https://developers.openai.com/codex/skills/
- Codex MCP：https://developers.openai.com/codex/mcp/
- MCP 生命周期：https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle
