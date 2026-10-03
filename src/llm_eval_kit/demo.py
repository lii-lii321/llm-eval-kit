"""玩具语料端到端 demo：构建语料 → 合成评测集 → 多路检索对比 → judge → 归因 → 报告。

语料是 24 段关于 RAG 与 LLM 评测的中文短文，纯代码内置，离线可跑。
demo 里的应答器是抽取式的（直接返回 top-1 文档原文），
只为把检索、judge 串成闭环，不代表真实生成质量。
"""

from __future__ import annotations

from pathlib import Path

from .attribution import REASON_LABELS_ZH
from .corpus import Doc
from .judge import DIMENSIONS_ZH, JudgeProvider, MockJudge
from .pipeline import run_evaluation
from .report import ReportData, write_reports
from .retrieval import BaseRetriever, BM25Retriever, HybridRetriever, TFRetriever
from .synth import EvalCase, generate_eval_set

# (doc_id, 正文)。刻意让主题词以短中文串开头，规则合成器能提出自然短语。
DEMO_DOCS: list[tuple[str, str]] = [
    ("d01", "RAG：检索增强生成。RAG 先从知识库检索相关片段，再拼接进提示词交给大模型生成答案。"
            "RAG 让模型引用外部知识，显著缓解幻觉，是企业知识问答的主流方案。"),
    ("d02", "向量数据库：存储文本向量，支持近似最近邻检索。常见引擎有 Faiss、Milvus 与 Qdrant，"
            "配合 HNSW 索引实现亚秒级语义检索。"),
    ("d03", "BM25：经典稀疏检索算法，基于词频与逆文档频率对文档打分。"
            "BM25 无需训练模型，在关键词命中场景仍是强基线。"),
    ("d04", "Embedding：把文本映射为稠密向量。语义相近的文本在向量空间中距离更近，"
            "选型要关注领域、维度与召回效果。"),
    ("d05", "混合检索：融合稠密检索与稀疏检索。常用倒数排序融合 RRF，或按权重加权求和，"
            "兼顾关键词命中与语义泛化。"),
    ("d06", "重排序：在粗排召回的基础上用交叉编码器精排。重排序只处理少量候选，却能显著提升排序质量。"),
    ("d07", "分块策略：把文档切分为检索片段。常见做法有固定长度滑窗、句子边界切分与语义分块，"
            "一般叠加 10% 到 20% 的窗口重叠。"),
    ("d08", "NDCG：归一化折损累计增益。NDCG 考虑相关文档的位置，越靠前贡献越大，是搜索评测的常用排序指标。"),
    ("d09", "MRR：平均倒数排名。MRR 看第一条相关结果出现的位置，适合关注首条命中质量的场景。"),
    ("d10", "Recall：召回率衡量前 K 条结果覆盖的相关文档比例。"
            "评测检索管线时常同时报告 Recall@1、Recall@5 与 Recall@10。"),
    ("d11", "LLM-as-judge：用大模型给待测答案打分。常见维度有正确性、相关性、可操作性与清晰度，"
            "提示词要给出明确评分标准。"),
    ("d12", "幻觉：模型一本正经地编造事实。检索增强、引用来源与事后校验是缓解幻觉的常用手段。"),
    ("d13", "评测集：量化系统表现的基础。评测集可由人工标注、真实查询采样或规则合成生成，"
            "要控制难度分布并剔除退化样本。"),
    ("d14", "Prompt 提示词工程：通过指令设计、少样本示例与思维链提升输出质量。"
            "稳定的提示词模板是线上可复现的前提。"),
    ("d15", "微调：把领域数据继续训练进模型参数。LoRA 等参数高效微调适合固定风格与格式任务，"
            "知识更新不如检索灵活。"),
    ("d16", "RAG 与微调的取舍取决于知识更新频率、成本与可解释性。"
            "知识频繁变化优先 RAG，风格需要深度定制再考虑微调。"),
    ("d17", "上下文窗口：限制单次推理可处理的 token 数量。长上下文模型可扩展到数十万 token，"
            "窗口越大成本越高，常配合压缩与摘要。"),
    ("d18", "延迟优化：流式输出、结果缓存、模型量化与并行检索都是常用手段。"
            "端到端延迟通常用 P50 与 P95 分位数度量。"),
    ("d19", "Agent：让大模型按 ReAct 模式循环思考并调用工具完成任务。"
            "function calling 是工具调用的核心机制，Agent 评测要覆盖任务成功率与工具选择准确性。"),
    ("d20", "护栏：在模型输入输出两侧过滤敏感内容与越权请求。企业级应用还需配套审计日志与人工兜底流程。"),
    ("d21", "评测报告：把指标表格、badcase 归因与改进建议沉淀成文档。定期产出评测报告让优化效果可追踪。"),
    ("d22", "数据隐私：对训练与检索语料做脱敏。金融与医疗场景常选私有化部署，权限隔离与访问审计是隐私工程的基础。"),
    ("d23", "成本控制：token 用量监控、小模型分流与结果缓存。按请求复杂度路由到不同规格模型能显著降低账单。"),
    ("d24", "流式输出：通过 SSE 把生成内容逐段推给前端，带来打字机效果并降低首 token 延迟。"
            "流式场景要处理中断与重连。"),
]

# 改写词典：把查询里的主题词换成零词面重叠的表达，构造词面不匹配的困难样本。
DEMO_SYNONYMS = {
    "向量数据库": "卡片盒子",
    "幻觉": "梦话",
    "召回率": "捞回条数",
    "重排序": "二次排队",
    "流式输出": "水龙头",
}


def build_demo_corpus() -> list[Doc]:
    return [Doc(doc_id=doc_id, text=text, source="demo") for doc_id, text in DEMO_DOCS]


class ExtractiveAnswerer:
    """演示用抽取式应答器：直接返回 top-1 检索文档原文。"""

    def __init__(self, retriever: BaseRetriever):
        self._retriever = retriever

    def answer(self, query: str, *, top_k: int = 1) -> str:
        if not query.strip():
            return ""
        results = self._retriever.retrieve(query, top_k=top_k)
        texts = self._retriever.doc_texts
        if not results:
            return ""
        return texts.get(results[0].doc_id, "")


def run_demo(
    out_dir: str | Path = "reports",
    *,
    seed: int = 42,
    num_cases: int = 24,
    paraphrase_ratio: float = 1.0,
    judge: JudgeProvider | None = None,
    cases: list[EvalCase] | None = None,
    notes_prefix: list[str] | None = None,
) -> ReportData:
    """跑通完整评测闭环并写出报告，返回 ReportData。

    paraphrase_ratio=1.0：对所有含改写词典键的查询都尝试改写，
    构造词面不匹配的困难样本，让 badcase 归因链路在 demo 里可见。
    cases 显式传入时（如 CLI --dataset 从 JSONL 加载的查询集）跳过合成生成，
    此时 seed/num_cases/paraphrase_ratio 不参与用例构造（seed 仅影响 bootstrap）。
    notes_prefix 会写进报告 notes 的最前面，用于记录评测集来源与统计。
    """
    docs = build_demo_corpus()
    if cases is None:
        cases = generate_eval_set(
            docs,
            num_cases=num_cases,
            seed=seed,
            paraphrase_ratio=paraphrase_ratio,
            synonym_map=DEMO_SYNONYMS,
        )
    retrievers: dict[str, BaseRetriever] = {
        "bm25": BM25Retriever(docs),
        "tf": TFRetriever(docs),
        "hybrid": HybridRetriever(docs, alpha=0.6),
    }
    answerer = ExtractiveAnswerer(retrievers["bm25"])
    answers = {case.qid: answerer.answer(case.query) for case in cases}
    data = run_evaluation(
        cases,
        retrievers,
        top_k=10,
        judge=judge if judge is not None else MockJudge(),
        answers=answers,
    )
    data.notes.extend(notes_prefix or [])
    data.notes.append("数字由内置 24 段玩具语料实测得出，仅代表 demo 规模，机器与运行环境不同会有差异。")
    md_path, html_path = write_reports(data, out_dir)
    data.notes.append(f"报告已写入：{md_path} 与 {html_path}")
    return data


def print_summary(data: ReportData) -> None:
    """把报告数据打印成控制台摘要（demo 输出）。"""
    print("== llm-eval-kit demo ==")
    print(f"语料文档数: {data.corpus_size}  评测集规模: {data.eval_size}  top-K: {data.top_k}")
    ks = data.pipelines[0].ks if data.pipelines else ()
    header = (
        "管线".ljust(8)
        + "".join(f"Recall@{k}".ljust(12) for k in ks)
        + "MRR".ljust(8)
        + "MAP@10".ljust(8)
        + "NDCG@10".ljust(10)
        + "P50(ms)".ljust(10)
    )
    print(header)
    for metrics in data.pipelines:
        row = metrics.name.ljust(8)
        row += "".join(f"{metrics.recall.get(k, 0.0):<12.4f}" for k in ks)
        row += f"{metrics.mrr:<8.4f}{metrics.map:<8.4f}{metrics.ndcg:<10.4f}{metrics.latency.p50:<10.3f}"
        print(row)
    if data.attribution:
        primary_name = next(iter(data.attribution))
        counts = data.attribution[primary_name]
        parts = [f"{REASON_LABELS_ZH[reason]}={count}" for reason, count in counts.items() if count]
        print(f"badcase 归因（{primary_name}）: " + (", ".join(parts) if parts else "无"))
    if data.judge.n_scored:
        means = "  ".join(f"{DIMENSIONS_ZH[dim]}={data.judge.dim_means.get(dim, 0.0):.2f}" for dim in DIMENSIONS_ZH)
        print(f"LLM-as-judge（{data.judge.provider}，n={data.judge.n_scored}）: {means}")
    elif data.judge.skipped:
        print(f"LLM-as-judge: 已跳过（{data.judge.skip_reason}）")
    for note in data.notes:
        print(f"> {note}")
