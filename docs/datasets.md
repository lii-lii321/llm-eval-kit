# 评测数据集指南：JSONL 格式规范与人工标注指南

llm-eval-kit 支持两种构造评测集的方式：规则合成（`generate_eval_set`，见 README）与
**JSONL 数据集文件**（本文档）。后者是人工标注查询集的标准载体——当你准备为真实语料
构造一条条人工标注的查询（例如 200 条主标注战役）时，从第一条标注起就用这个格式，
后续的加载、校验、评测、回归门禁通道全部现成。

种子示例：[`examples/dataset_demo.jsonl`](../examples/dataset_demo.jsonl)
（**合成 demo 数据，非人工标注**，`_meta.provenance` 已明确标注）。

## 文件格式规范

- 编码 UTF-8，扩展名建议 `.jsonl`；每行一个独立 JSON 对象，行间无逗号；
- 空行跳过；行号从 1 起计（含空行），所有校验错误都会带行号；
- **元信息行**：首个非空行允许是 `{"_meta": {...}}`，全文件最多出现一次，
  且不得混入查询字段。`_meta` 内容自由，建议包含：

| 键 | 含义 |
|---|---|
| `corpus` | 语料描述（来源、规模、doc_id 规则） |
| `annotated_at` | 标注日期（ISO 格式，如 2026-10-03） |
| `annotator` | 标注人 / 标注小组 |
| `version` | 数据集版本号（标注修订后递增） |
| `provenance` | 数据来源；合成数据必须写明（如 `synthetic demo, not human-annotated`） |

### 查询记录

```jsonl
{"query_id": "q001", "query": "什么是 RAG 检索增强生成？", "relevant_doc_ids": ["d01"]}
{"query_id": "q010", "query": "如何缓解大模型的幻觉问题？", "relevant_doc_ids": ["d01", "d12"], "grades": {"d12": 2, "d01": 1}, "notes": "跨文档多相关"}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `query_id` | 是 | 非空字符串，全文件唯一 |
| `query` | 是 | 非空字符串，用户的真实问法 |
| `relevant_doc_ids` | 是 | 非空字符串数组，语料中的相关文档 ID，行内不得重复 |
| `grades` | 否 | 对象 `{doc_id: 0\|1\|2}`，键必须落在该行 `relevant_doc_ids` 内，值取 0/1/2 整数（JSON 布尔值不合法） |
| `notes` | 否 | 字符串，标注备注（边界情况说明、标注依据等） |

未知字段一律报错（防 `relevant_doc_id` 这类拼写笔误静默丢标注）。

### 校验规则（`load_dataset` 严格模式）

以下任一问题抛 `DatasetError`（`ValueError` 子类），message 带行号，`line` 属性可编程取用：

1. 行不是合法 JSON / 不是 JSON 对象；
2. 缺少必填字段；字段类型不对（含 grades 里的 bool 陷阱）；
3. `query_id` 重复（报出两处行号）；
4. `relevant_doc_ids` 为空、含空/非字符串元素、行内重复；
5. `grades` 键不在该行 `relevant_doc_ids` 内、值超出 0/1/2；
6. `_meta` 重复、不在首个非空行、混入查询字段；
7. 整个文件没有任何查询记录。

## 使用方式

### CLI（批量评测 + 回归门禁）

```bash
# 用评测集跑内置玩具语料上的三路检索对比
python -m llm_eval_kit.cli --dataset examples/dataset_demo.jsonl --out reports

# 与 --fail-under 组合，直接当 CI 门禁用
python -m llm_eval_kit.cli --dataset my_annotations.jsonl --out reports \
  --fail-under recall_at_5=0.85 --fail-under mrr=0.8
```

- 加载失败（格式/校验错误）退出码 2，stderr 给出行号；门禁不达标退出码 1；
- `--num-cases` / `--paraphrase-ratio` 是合成评测集专用参数，与 `--dataset` 互斥（退出码 2）；
- `--seed` 仍可用：只影响 bootstrap 重采样，不影响从文件读哪些查询；
- 语料沿用 demo 内置玩具语料。若评测集引用了语料中不存在的 `doc_id`
  （永远检索不到，会静默拉低召回指标），CLI 会在 stderr 打印警告明细——标注与语料
  允许分阶段演进，故警告不阻断。

### Python 库

```python
from llm_eval_kit import BM25Retriever, load_corpus_from_dir, load_dataset, run_evaluation

dataset = load_dataset("my_annotations.jsonl")       # 校验失败在此抛 DatasetError（带行号）
print(dataset.stats().describe())                    # 条数 / 平均相关文档数 / 查询长度分布
cases = dataset.to_eval_cases()                      # → run_evaluation 直接可用

docs = load_corpus_from_dir("path/to/txt_docs")
data = run_evaluation(cases, {"bm25": BM25Retriever(docs)}, top_k=10)
```

`Dataset.unknown_doc_ids(corpus_doc_ids)` 返回 `{query_id: [语料缺失的 doc_id]}`，
评测前自查标注与语料的一致性。

### DatasetStats

`dataset.stats()` 返回条数、平均相关文档数、查询长度 min/中位/均值/max 与长度分桶
分布（1-10 / 11-20 / 21-30 / 31-50 / >50 字符）。标注完成后先看分布：查询长度
过度集中、平均相关文档数异常，往往提示标注指南需要修订。

## 人工标注指南（骨架）

为真实语料构造高质量标注集（如 200 条）的推荐流程：

### 1. 从真实语料构造查询

- **查询来源优先级**：真实用户查询日志 > 业务方访谈收集 > LLM 辅助生成 + 人工逐条复核 >
  标注员自拟。自拟查询容易不自觉复用文档原词，会高估词面检索、低估语义 gap；
- **覆盖度**：按主题/文档分桶分配配额，避免 200 条全压在少数热门主题；
- **难度分布**：刻意保留一部分词面不匹配（用户不会用文档术语提问）、
  多相关文档、以及少量语料确实答不了的查询（考察拒答/归因），并记进 `notes`；
- **doc_id 对齐**：标注前先确定语料切块方案（`load_corpus_from_dir` 的
  `max_chars`），标注引用的 `doc_id` 必须与最终索引的切块一一对应；切块方案变了，
  标注要复标（`_meta.version` 递增）。

### 2. 相关性判定标准

- 写一页标注指南，给出**可操作的判定规则**而非"感觉相关"：例如
  "文档包含能直接回答查询的事实 → 相关；只共享主题词但答不了 → 不相关"；
- 分级相关度建议：`2` = 直接完整回答；`1` = 部分相关/需要与其他文档拼合；
  `0` = 仅主题相近。`grades` 里给 0 的意义是显式记录边界判断，便于仲裁；
- 每条查询的判定依据写进 `notes`，仲裁时省一半扯皮。

### 3. 标注一致性

- **双人独立标注**同一批查询，对 `relevant_doc_ids` 不一致的条目引入第三人仲裁；
- 抽样计算标注间一致性（如 Cohen's kappa），低于约定阈值（常见 ≥ 0.8）先修指南再继续；
- 定期抽样复标已标条目，监控漂移；
- 全程版本化：任何指南修订、语料切块变更都体现在 `_meta.version` 与 `annotated_at`。

### 4. 质检清单（提交前）

1. `load_dataset` 通过（0 个校验错误）；
2. `unknown_doc_ids` 为空（所有 doc_id 都在语料里）；
3. `stats()` 分布合理（无超长查询堆积、平均相关文档数符合设计）；
4. `query_id` 稳定可追溯（不要重排序后重编号，下游对比依赖 ID 稳定）；
5. 抽 10 条人工回读 `query` 是否像真实用户问法。

## 当前限制

- **grades 只校验不参与指标**：分级相关度经校验后透传进 `EvalCase.meta`，但
  Recall/MRR/MAP/NDCG 仍按二元相关度计算（与 README 已知限制"NDCG 使用二元相关度"同源）。
  分级相关度指标（如加权 NDCG）是规划方向，格式先行是为了标注数据不用返工；
- **数据集通道暂无参考答案字段**：JSONL 不含 reference answer，`--dataset` 跑的
  评测中 LLM-as-judge 段会静默跳过（judge 需要参考答案）；需要 judge 时可暂用
  合成评测集，或在代码里给 `EvalCase.answer` 赋值后走库接口；
- **CLI `--dataset` 的语料是内置玩具语料**：对外部语料 + 人工标注集的完整组合，
  用上面的 Python 库用法（`load_corpus_from_dir` + `load_dataset` + `run_evaluation`）；
- **非行级错误不带行号**：文件不存在、整体为空等错误没有对应行，`DatasetError.line`
  为 None。
