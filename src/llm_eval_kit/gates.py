"""评测回归门禁：给检索指标设定最低阈值，任一未达标即判定失败。

与 CLI 的 ``--fail-under 指标名=阈值``（可多次）配合，把评测当作回归测试：
CI 中指标跌破阈值时进程以退出码 1 结束，阻断合并或发布。
门禁作用于主管线（评测结果里的第一个管线），阈值取值范围 [0, 1]。

支持的指标名：
- ``recall_at_<K>``：Recall@K（K 取决于本次评测的 ks，默认 1/3/5/10）
- ``mrr``：MRR
- ``map`` / ``map_at_10``：MAP@10
- ``ndcg`` / ``ndcg_at_10``：NDCG@10
- ``hit_rate``：top-K 命中率
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .metrics import PipelineMetrics

_FIXED_METRICS: dict[str, str] = {
    "mrr": "mrr",
    "map": "map",
    "map_at_10": "map",
    "ndcg": "ndcg",
    "ndcg_at_10": "ndcg",
    "hit_rate": "hit_rate",
}


def supported_metric_names(ks: Sequence[int] = (1, 3, 5, 10)) -> list[str]:
    """列出当前评测口径下全部门禁可用指标名。"""
    return [f"recall_at_{k}" for k in ks] + list(_FIXED_METRICS)


def parse_fail_under(spec: str) -> tuple[str, float]:
    """解析 ``指标名=阈值``（如 recall_at_5=0.85），格式非法抛 ValueError。"""
    text = spec.strip()
    name, sep, raw_threshold = text.partition("=")
    if not sep:
        raise ValueError(
            f"--fail-under 需要 '指标名=阈值' 格式，例如 --fail-under recall_at_5=0.85，收到：{spec!r}"
        )
    name = name.strip()
    if not name:
        raise ValueError(f"--fail-under 指标名不能为空，收到：{spec!r}")
    try:
        threshold = float(raw_threshold.strip())
    except ValueError:
        raise ValueError(f"--fail-under 阈值必须是数字，收到：{spec!r}") from None
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"--fail-under 阈值取值范围 [0, 1]，收到：{spec!r}")
    return name, threshold


def metric_value(metrics: PipelineMetrics, name: str) -> float:
    """按名取主管线指标值；名字未知或该 K 未评测时抛带支持列表的 ValueError。"""
    if name.startswith("recall_at_"):
        try:
            k = int(name.removeprefix("recall_at_"))
        except ValueError:
            k = -1
        if k in metrics.recall:
            return metrics.recall[k]
        available_ks = sorted(metrics.recall)
        raise ValueError(
            f"指标 {name!r} 不可用：本次评测未计算该 Recall@K（可用 K：{available_ks}）。"
            f"支持的全部指标名：{supported_metric_names(available_ks)}"
        )
    attr = _FIXED_METRICS.get(name)
    if attr is not None:
        return float(getattr(metrics, attr))
    available_ks = sorted(metrics.recall)
    raise ValueError(
        f"未知指标名 {name!r}。支持的全部指标名：{supported_metric_names(available_ks)}"
    )


@dataclass(frozen=True)
class GateFailure:
    """一条未达标的门禁：记录指标名、阈值与实测值。"""

    metric: str
    threshold: float
    actual: float

    @property
    def gap(self) -> float:
        """实测值距阈值还差多少（正数 = 缺口大小）。"""
        return self.threshold - self.actual

    @property
    def message(self) -> str:
        return (
            f"门禁未达标：{self.metric} 实测 {self.actual:.4f} < 阈值 {self.threshold:.4f}"
            f"（差 {self.gap:.4f}）"
        )


def evaluate_gates(
    metrics: PipelineMetrics, gates: Sequence[tuple[str, float]]
) -> list[GateFailure]:
    """逐项比对指标与阈值，返回未达标列表（空列表 = 全部达标）。"""
    failures: list[GateFailure] = []
    for name, threshold in gates:
        actual = metric_value(metrics, name)
        if actual < threshold:
            failures.append(GateFailure(metric=name, threshold=threshold, actual=actual))
    return failures
