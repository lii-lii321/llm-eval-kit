"""检索评测指标：Recall@K、MRR、NDCG@10 与延迟分位数统计。

相关度采用二元定义（命中/未命中），NDCG 因此是 binary NDCG。
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """前 K 条结果覆盖的相关文档比例。relevant 为空时返回 0。"""
    if k <= 0:
        raise ValueError("k 必须为正整数")
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(set(ranked[:k]) & rel) / len(rel)


def mrr(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """第一条相关结果的倒数排名；排序列表通常只有 top-K，K 之外记 0。"""
    rel = set(relevant)
    for i, doc_id in enumerate(ranked):
        if doc_id in rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """binary NDCG@K：位置越靠前的相关文档贡献越大。"""
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, doc_id in enumerate(ranked[:k]) if doc_id in rel)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数（与 numpy 默认策略一致）。"""
    if not values:
        raise ValueError("values 不能为空")
    if not 0 <= p <= 100:
        raise ValueError("p 取值范围 [0, 100]")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


@dataclass
class LatencyStats:
    """单条查询耗时的分布统计（单位毫秒）。"""

    p50: float = 0.0
    p95: float = 0.0
    mean: float = 0.0
    max: float = 0.0
    n: int = 0


def latency_stats(durations_ms: Sequence[float]) -> LatencyStats:
    """汇总一组延迟样本；空列表返回全 0 统计。"""
    values = [float(v) for v in durations_ms]
    if not values:
        return LatencyStats()
    return LatencyStats(
        p50=percentile(values, 50),
        p95=percentile(values, 95),
        mean=sum(values) / len(values),
        max=max(values),
        n=len(values),
    )


@dataclass
class PipelineMetrics:
    """一条检索管线在评测集上的聚合指标。"""

    name: str
    ks: tuple[int, ...] = (1, 3, 5, 10)
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: float = 0.0  # NDCG@10
    hit_rate: float = 0.0  # top-K 内至少命中一条相关文档的查询占比
    latency: LatencyStats = field(default_factory=LatencyStats)
