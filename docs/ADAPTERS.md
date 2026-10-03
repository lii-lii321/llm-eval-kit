# 外部检索器接入指南

llm-eval-kit 的评测管线不绑定内置检索器。任何检索系统——向量数据库、
Elasticsearch、自研混合检索、另一个仓库里的 RAG 模块——只要能
"给定查询返回 top-K 文档 ID"，就能用同一套指标（Recall@K / MRR / MAP@10 /
NDCG@10 / 延迟）、badcase 归因和报告输出评测。

## 1. Retriever 协议

协议是 duck typing 的，不要求继承任何类：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float | None = None  # score 可选，仅用于排序

class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 10) -> list[Hit]: ...
```

`retrieve` 返回的每个元素宽容归一化，以下形态都接受：

| 形态 | 示例 |
|---|---|
| `Hit` | `Hit(doc_id="d01", score=0.87)` |
| `(doc_id, score)` 元组 | `("d01", 0.87)` |
| 裸 doc_id 字符串 | `"d01"` |
| 任何带 `doc_id` 属性的对象 | 本库的 `RetrievalResult`、ORM 行等 |

## 2. 最短路径：CallableRetriever（约 10 行）

把已有检索函数包一层即可参与评测：

```python
from llm_eval_kit import CallableRetriever, Hit, generate_eval_set, load_corpus_from_dir, run_evaluation

def my_search(query: str, k: int) -> list[Hit]:        # 你现有的任意检索逻辑
    return [Hit(doc_id=d["id"], score=d["score"])
            for d in my_engine.search(query, size=k)]   # 换成你的引擎调用

docs = load_corpus_from_dir("path/to/txt_docs")
cases = generate_eval_set(docs, num_cases=50, seed=42)
report = run_evaluation(cases, {"my-engine": CallableRetriever(my_search, name="my-engine")})
```

`run_evaluation` / `run_retrieval` 的检索器字典值可以直接传：
内置 `BaseRetriever`（BM25/TF/混合）、`CallableRetriever`、或任何自己实现了
`retrieve(query, top_k)` 方法的对象（内部自动适配）。

## 3. 自定义类 + ensure_retriever 自检

```python
from llm_eval_kit import ensure_retriever, run_evaluation

class MyRetriever:
    name = "my-retriever"

    def __init__(self, engine):
        self._engine = engine

    def retrieve(self, query: str, top_k: int = 10) -> list[Hit]:
        return self._engine.top_k(query, top_k)

retriever = ensure_retriever(MyRetriever(engine))   # 不满足协议时抛可诊断错误
```

`ensure_retriever` 校验两件事：存在可调用的 `retrieve` 方法；签名能接受
`(query, top_k)` 两个参数（`top_k` 位置或关键字传递均可）。不满足时抛出
`RetrieverProtocolError`，错误信息包含：缺什么、期望签名、最快的接入方式与本文档路径。
`top_k` 只接受关键字参数（`def retrieve(self, query, *, top_k)`）的检索器也被接受。

## 4. doc_texts 与 badcase 归因

badcase 三类归因（关键词不匹配 / 语义漂移 / 语料缺失）需要 `doc_id → 原文`
的映射（`doc_texts`）。适配层按以下优先级获取：

1. 显式传入：`RetrieverAdapter(my_retriever, doc_texts={...})`；
2. 被包装对象自带同名属性 `doc_texts`（duck typing 自动识别）；
3. 都没有 → 空字典。

**已知限制**：`doc_texts` 为空时，所有未命中的查询会被归因为"语料缺失"，
与真实失败机理不符。如果你的检索系统拿不到原文，建议显式传入
`doc_texts`（哪怕只覆盖评测集涉及的相关文档），归因才有参考价值。

## 5. 完整示例：把 Math_Tutor_RAG 的 QuestionVectorStore 包成 Retriever

以下是面向文档的适配示例，展示如何评测另一个仓库的检索器；
**不需要改动 Math_Tutor_RAG 仓库本身**，在评测脚本一侧完成包装即可。
`QuestionVectorStore.semantic_search(query, *, user_ids, top_k, min_similarity)`
返回 `RagHit(question_id, distance, tags, snippet)`，其中 `distance` 是
cosine 距离（越小越相关），先换算成相似度再交给评测器：

```python
from llm_eval_kit import CallableRetriever, Hit, run_evaluation

def wrap_question_vector_store(store, user_ids: list[int], name: str = "math-tutor-vector"):
    """把 Math_Tutor_RAG 的 QuestionVectorStore 包成 llm-eval-kit 检索器。"""
    def retrieve(query: str, top_k: int) -> list[Hit]:
        hits = store.semantic_search(query, user_ids=user_ids, top_k=top_k)
        return [Hit(doc_id=str(h.question_id), score=max(0.0, min(1.0, 1.0 - h.distance)))
                for h in hits]
    return CallableRetriever(retrieve, name=name)

report = run_evaluation(eval_cases, {"vector": wrap_question_vector_store(store, user_ids=[1])})
```

要点：

- 评测集的 `relevant_ids` 必须与包装层的 `doc_id` 同一口径（这里是题目 `question_id`
  的字符串形式）；建议直接用真实题库构造人工标注评测集，而不是用本库的规则合成器；
- `user_ids` 是 Math_Tutor_RAG 的租户隔离参数，评测时固定成评测集所属用户；
- score 用 `1 - cosine_distance` 换算，只影响排序展示，指标本身只用 `doc_id` 顺序。

## 6. 在 CI 里守住指标底线

评测脚本建议加回归门禁，指标跌破阈值时以非零码退出，阻断合并（完整
GitHub Actions 示例见 README「评测回归门禁」一节）：

```bash
python -m llm_eval_kit.cli demo --out reports --fail-under recall_at_5=0.85
```
