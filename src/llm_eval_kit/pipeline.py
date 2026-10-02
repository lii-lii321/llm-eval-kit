"""端到端评测管线：多路检索对比 + 指标聚合 + judge + badcase 归因 → 报告数据。"""

from __future__ import annotations

import time
from datetime import datetime

from .attribution import Badcase, attribute_badcases, summarize_attribution
from .judge import JudgeProvider, JudgeScore, JudgeSummary, LLMJudgeError, judge_summary
from .metrics import PipelineMetrics, latency_stats, mrr, ndcg_at_k, recall_at_k
from .report import ReportData
from .retrieval import BaseRetriever
from .synth import EvalCase


def run_retrieval(
    cases: list[EvalCase],
    retriever: BaseRetriever,
    *,
    top_k: int = 10,
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> PipelineMetrics:
    """在评测集上运行一条检索管线，返回聚合指标与真实计时延迟。"""
    ranked_by_qid: dict[str, list[str]] = {}
    durations_ms: list[float] = []
    for case in cases:
        start = time.perf_counter()
        results = retriever.retrieve(case.query, top_k=top_k)
        durations_ms.append((time.perf_counter() - start) * 1000)
        ranked_by_qid[case.qid] = [r.doc_id for r in results]
    n = len(cases) or 1
    metrics = PipelineMetrics(name=retriever.name, ks=tuple(ks))
    for k in ks:
        metrics.recall[k] = sum(recall_at_k(ranked_by_qid[c.qid], c.relevant_ids, k) for c in cases) / n
    metrics.mrr = sum(mrr(ranked_by_qid[c.qid], c.relevant_ids) for c in cases) / n
    metrics.ndcg = sum(ndcg_at_k(ranked_by_qid[c.qid], c.relevant_ids, 10) for c in cases) / n
    metrics.hit_rate = sum(
        1 for c in cases if set(ranked_by_qid[c.qid][:top_k]) & set(c.relevant_ids)
    ) / n
    metrics.latency = latency_stats(durations_ms)
    return metrics


def run_evaluation(
    cases: list[EvalCase],
    retrievers: dict[str, BaseRetriever],
    *,
    top_k: int = 10,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    judge: JudgeProvider | None = None,
    answers: dict[str, str] | None = None,
) -> ReportData:
    """多路检索管线对比评测，产出报告数据。

    - retrievers：管线名 → 检索器，第一个视为主管线（badcase 示例取自主管线）；
    - judge + answers：可选。answers[qid] 是待评系统答案，
      与用例自带的参考答案一起交给裁判评分；裁判不可用时整段优雅跳过。
    """
    if not retrievers:
        raise ValueError("至少需要一个检索管线")
    answers = answers or {}
    retriever_list = list(retrievers.values())

    pipelines = [run_retrieval(cases, r, top_k=top_k, ks=ks) for r in retriever_list]

    attribution: dict[str, dict[str, int]] = {}
    badcases_by_pipeline: dict[str, list[Badcase]] = {}
    for name, retriever in retrievers.items():
        ranked_by_qid = {
            c.qid: [x.doc_id for x in retriever.retrieve(c.query, top_k=top_k)] for c in cases
        }
        badcases = attribute_badcases(cases, ranked_by_qid, retriever.doc_texts, top_k=top_k)
        attribution[name] = summarize_attribution(badcases)
        badcases_by_pipeline[name] = badcases

    summary = JudgeSummary()
    if judge is not None:
        try:
            primary = retriever_list[0]
            texts = primary.doc_texts
            scores: list[JudgeScore] = []
            for case in cases:
                system_answer = answers.get(case.qid, "")
                if not system_answer or not case.answer:
                    continue
                context = ""
                ranked = [x.doc_id for x in primary.retrieve(case.query, top_k=1)]
                if ranked and ranked[0] in texts:
                    context = texts[ranked[0]]
                scores.append(
                    judge.score(query=case.query, answer=system_answer, reference=case.answer, context=context)
                )
            summary = judge_summary(scores)
        except LLMJudgeError as exc:
            summary = JudgeSummary(
                provider=getattr(judge, "name", "unknown"), skipped=True, skip_reason=str(exc)
            )

    primary_name = next(iter(retrievers))
    return ReportData(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        corpus_size=len(retriever_list[0].doc_texts),
        eval_size=len(cases),
        top_k=top_k,
        pipelines=pipelines,
        judge=summary,
        attribution=attribution,
        primary_badcases=badcases_by_pipeline[primary_name][:5],
        notes=[],
    )
