"""端到端评测管线：多路检索对比 + 指标聚合 + judge + badcase 归因 → 报告数据。"""

from __future__ import annotations

import time
from datetime import datetime

from .attribution import Badcase, attribute_badcases, summarize_attribution
from .judge import JudgeProvider, JudgeScore, JudgeSummary, LLMJudgeError, judge_summary
from .metrics import (
    PipelineMetrics,
    average_precision,
    bootstrap_ci,
    latency_stats,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from .report import ReportData
from .retrieval import BaseRetriever, RetrievalResult
from .synth import EvalCase


def _collect_rankings(
    cases: list[EvalCase],
    retriever: BaseRetriever,
    top_k: int,
) -> tuple[dict[str, list[RetrievalResult]], list[float]]:
    """对每条用例检索一次，返回 {qid: 检索结果} 与每条查询的真实耗时（毫秒）。

    结果在管线内复用于指标、归因与 judge 上下文，避免重复检索。
    """
    ranked: dict[str, list[RetrievalResult]] = {}
    durations_ms: list[float] = []
    for case in cases:
        start = time.perf_counter()
        results = retriever.retrieve(case.query, top_k=top_k)
        durations_ms.append((time.perf_counter() - start) * 1000)
        ranked[case.qid] = results
    return ranked, durations_ms


def _build_metrics(
    name: str,
    cases: list[EvalCase],
    ranked: dict[str, list[RetrievalResult]],
    durations_ms: list[float],
    *,
    top_k: int,
    ks: tuple[int, ...],
    seed: int,
    n_boot: int,
    confidence: float,
) -> PipelineMetrics:
    """基于已收集的检索结果聚合指标（不做任何检索），并给出 bootstrap 置信区间。"""
    ranked_ids = {qid: [r.doc_id for r in results] for qid, results in ranked.items()}
    n = len(cases) or 1
    per_query_recall: dict[int, list[float]] = {k: [] for k in ks}
    per_query_mrr: list[float] = []
    per_query_map: list[float] = []
    per_query_ndcg: list[float] = []
    per_query_hit: list[float] = []
    for case in cases:
        ids = ranked_ids[case.qid]
        rel = set(case.relevant_ids)
        for k in ks:
            per_query_recall[k].append(recall_at_k(ids, rel, k))
        per_query_mrr.append(mrr(ids, rel))
        per_query_map.append(average_precision(ids, rel, 10))
        per_query_ndcg.append(ndcg_at_k(ids, rel, 10))
        per_query_hit.append(1.0 if rel & set(ids[:top_k]) else 0.0)

    metrics = PipelineMetrics(name=name, ks=tuple(ks))
    for k in ks:
        metrics.recall[k] = sum(per_query_recall[k]) / n
    metrics.mrr = sum(per_query_mrr) / n
    metrics.map = sum(per_query_map) / n
    metrics.ndcg = sum(per_query_ndcg) / n
    metrics.hit_rate = sum(per_query_hit) / n
    if cases and n_boot > 0:
        metrics.recall_ci = {
            k: bootstrap_ci(per_query_recall[k], n_boot=n_boot, confidence=confidence, seed=seed)
            for k in ks
        }
        metrics.mrr_ci = bootstrap_ci(per_query_mrr, n_boot=n_boot, confidence=confidence, seed=seed)
        metrics.map_ci = bootstrap_ci(per_query_map, n_boot=n_boot, confidence=confidence, seed=seed)
        metrics.ndcg_ci = bootstrap_ci(per_query_ndcg, n_boot=n_boot, confidence=confidence, seed=seed)
        metrics.hit_rate_ci = bootstrap_ci(per_query_hit, n_boot=n_boot, confidence=confidence, seed=seed)
    metrics.latency = latency_stats(durations_ms)
    return metrics


def run_retrieval(
    cases: list[EvalCase],
    retriever: BaseRetriever,
    *,
    top_k: int = 10,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    seed: int = 42,
    n_boot: int = 1000,
    confidence: float = 0.95,
) -> PipelineMetrics:
    """在评测集上运行一条检索管线，返回聚合指标（含 bootstrap 置信区间）与真实计时延迟。

    - seed/n_boot/confidence 控制 bootstrap 重采样；n_boot=0 关闭置信区间；
    - 固定 seed 保证区间可复现。
    """
    ranked, durations_ms = _collect_rankings(cases, retriever, top_k)
    return _build_metrics(
        retriever.name,
        cases,
        ranked,
        durations_ms,
        top_k=top_k,
        ks=ks,
        seed=seed,
        n_boot=n_boot,
        confidence=confidence,
    )


def run_evaluation(
    cases: list[EvalCase],
    retrievers: dict[str, BaseRetriever],
    *,
    top_k: int = 10,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    judge: JudgeProvider | None = None,
    answers: dict[str, str] | None = None,
    seed: int = 42,
    n_boot: int = 1000,
    confidence: float = 0.95,
) -> ReportData:
    """多路检索管线对比评测，产出报告数据。

    - retrievers：管线名 → 检索器，第一个视为主管线（badcase 示例取自主管线）；
    - judge + answers：可选。answers[qid] 是待评系统答案，
      与用例自带的参考答案一起交给裁判评分；裁判不可用时整段优雅跳过；
    - seed/n_boot/confidence 控制 bootstrap 置信区间，n_boot=0 关闭；
      所有管线共用同一 seed 与评测集规模，重采样索引一致，区间可直接横向对比。
    - 每条查询对每个检索器只检索一次，指标/归因/judge 上下文共用同一批结果。
    """
    if not retrievers:
        raise ValueError("至少需要一个检索管线")
    answers = answers or {}

    pipelines: list[PipelineMetrics] = []
    rankings: dict[str, dict[str, list[RetrievalResult]]] = {}
    for name, retriever in retrievers.items():
        ranked, durations_ms = _collect_rankings(cases, retriever, top_k)
        rankings[name] = ranked
        pipelines.append(
            _build_metrics(
                name,
                cases,
                ranked,
                durations_ms,
                top_k=top_k,
                ks=ks,
                seed=seed,
                n_boot=n_boot,
                confidence=confidence,
            )
        )

    attribution: dict[str, dict[str, int]] = {}
    badcases_by_pipeline: dict[str, list[Badcase]] = {}
    for name, ranked in rankings.items():
        ranked_ids = {qid: [r.doc_id for r in results] for qid, results in ranked.items()}
        badcases = attribute_badcases(
            cases, ranked_ids, retrievers[name].doc_texts, top_k=top_k
        )
        attribution[name] = summarize_attribution(badcases)
        badcases_by_pipeline[name] = badcases

    summary = JudgeSummary()
    if judge is not None:
        primary_name = next(iter(retrievers))
        primary_ranked = rankings[primary_name]
        texts = retrievers[primary_name].doc_texts
        try:
            scores: list[JudgeScore] = []
            for case in cases:
                system_answer = answers.get(case.qid, "")
                if not system_answer or not case.answer:
                    continue
                context = ""
                top1 = primary_ranked.get(case.qid, [])[:1]
                if top1 and top1[0].doc_id in texts:
                    context = texts[top1[0].doc_id]
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
        corpus_size=len(retrievers[primary_name].doc_texts),
        eval_size=len(cases),
        top_k=top_k,
        pipelines=pipelines,
        judge=summary,
        attribution=attribution,
        primary_badcases=badcases_by_pipeline[primary_name][:5],
        notes=[],
    )
