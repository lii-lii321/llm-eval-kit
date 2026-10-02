"""检索评测指标：Recall@K、MRR、MAP@10、NDCG@10、bootstrap 置信区间与延迟分位数统计。

相关度采用二元定义（命中/未命中），NDCG 因此是 binary NDCG。
"""

import math
import random
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


def average_precision(ranked: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    """AP@K：每条相关文档命中位置的精度均值，按 min(相关文档数, K) 归一。

    与 ndcg_at_k 一致使用二元相关度；relevant 为空返回 0。
    """
    if k <= 0:
        raise ValueError("k 必须为正整数")
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, doc_id in enumerate(ranked[:k]):
        if doc_id in rel:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / min(len(rel), k)


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


@dataclass(frozen=True)
class ConfidenceInterval:
    """bootstrap 百分位置信区间（对样本均值）。"""

    low: float
    high: float
    level: float = 0.95
    n_boot: int = 1000


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> ConfidenceInterval:
    """对样本均值做 bootstrap 重采样，返回百分位置信区间。

    相同 values + 相同 seed 产出完全相同的区间（可复现）；
    不同检索管线传入相同 seed 时重采样索引一致，区间可直接横向对比。
    """
    if not values:
        raise ValueError("values 不能为空")
    if n_boot <= 0:
        raise ValueError("n_boot 必须为正整数")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence 取值范围 (0, 1)")

    data = [float(v) for v in values]
    rng = random.Random(seed)
    n = len(data)
    means: list[float] = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += data[rng.randrange(n)]
        means.append(total / n)
    alpha = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        low=percentile(means, alpha * 100),
        high=percentile(means, (1.0 - alpha) * 100),
        level=confidence,
        n_boot=n_boot,
    )


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
    map: float = 0.0
    ndcg: float = 0.0  # NDCG@10
    hit_rate: float = 0.0  # top-K 内至少命中一条相关文档的查询占比
    recall_ci: dict[int, ConfidenceInterval] = field(default_factory=dict)
    mrr_ci: ConfidenceInterval | None = None
    map_ci: ConfidenceInterval | None = None
    ndcg_ci: ConfidenceInterval | None = None
    hit_rate_ci: ConfidenceInterval | None = None
    latency: LatencyStats = field(default_factory=LatencyStats)
    # *_ci：对应聚合指标的 bootstrap 置信区间；评测集为空或关闭重采样时为空/None
    # map / map_ci 是 MAP@10（各查询 AP@10 的均值），与 ndcg 同为 @10 口径
