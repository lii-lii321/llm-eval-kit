"""JSONL 评测集加载与校验：人工标注查询集的统一数据通道。

格式规范（完整版见 docs/datasets.md）：
- 文件为 UTF-8 JSONL，每行一个 JSON 对象，空行跳过；
- 首个非空行允许是元信息行 ``{"_meta": {...}}``（语料描述、标注日期、标注人、版本等），
  全文件最多出现一次，且不得混入查询字段；
- 其余每行一条查询记录：query_id / query / relevant_doc_ids 必填，
  grades（分级相关度 0|1|2）与 notes 可选；
- 任何格式/内容问题抛 DatasetError，错误信息带行号。

设计取舍：仓库核心零第三方依赖，故用 dataclass + 手写校验而非 pydantic。
分级相关度 grades 校验后透传（存入 EvalCase.meta）：评测集任一查询带 grades 时，
管线自动产出分级相关度 Weighted NDCG@10（见 metrics.ndcg_at_k_weighted）；
Recall/MRR/MAP 仍按二元相关度——见 README 已知限制。
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .synth import EvalCase

META_KEY = "_meta"
REQUIRED_KEYS: tuple[str, ...] = ("query_id", "query", "relevant_doc_ids")
ALLOWED_KEYS: frozenset[str] = frozenset({"query_id", "query", "relevant_doc_ids", "grades", "notes"})
GRADE_VALUES: frozenset[int] = frozenset({0, 1, 2})

# 查询长度（字符数）分布分桶，上界含端点
_LENGTH_BUCKETS: tuple[tuple[int, str], ...] = ((10, "1-10"), (20, "11-20"), (30, "21-30"), (50, "31-50"))


class DatasetError(ValueError):
    """评测集格式或内容非法。message 含行号；line 属性可供编程取用（非行级问题为 None）。"""

    def __init__(self, message: str, *, line: int | None = None) -> None:
        self.line = line
        super().__init__(message)


@dataclass(frozen=True)
class DatasetRecord:
    """一条通过校验的查询记录。line 是它在文件中的行号（1 起，含空行计数）。"""

    query_id: str
    query: str
    relevant_doc_ids: list[str]
    grades: dict[str, int] = field(default_factory=dict)
    notes: str = ""
    line: int = 0


@dataclass(frozen=True)
class DatasetStats:
    """评测集统计摘要：条数、平均相关文档数与查询长度分布。"""

    n_cases: int
    avg_relevant_docs: float
    query_len_min: int
    query_len_median: float
    query_len_mean: float
    query_len_max: int
    query_len_buckets: dict[str, int]

    @classmethod
    def from_records(cls, records: list[DatasetRecord]) -> DatasetStats:
        lengths = [len(r.query) for r in records]
        buckets = {label: 0 for _, label in _LENGTH_BUCKETS}
        buckets[">50"] = 0
        for n in lengths:
            buckets[_length_bucket(n)] += 1
        return cls(
            n_cases=len(records),
            avg_relevant_docs=sum(len(r.relevant_doc_ids) for r in records) / len(records) if records else 0.0,
            query_len_min=min(lengths) if lengths else 0,
            query_len_median=float(statistics.median(lengths)) if lengths else 0.0,
            query_len_mean=sum(lengths) / len(lengths) if lengths else 0.0,
            query_len_max=max(lengths) if lengths else 0,
            query_len_buckets=buckets,
        )

    def describe(self) -> str:
        """单行人类可读摘要，CLI 加载评测集后打印。"""
        buckets = "，".join(f"{label} {count} 条" for label, count in self.query_len_buckets.items() if count)
        text = (
            f"共 {self.n_cases} 条，平均相关文档 {self.avg_relevant_docs:.2f} 篇；"
            f"查询长度 min/中位/均值/max = {self.query_len_min}/{self.query_len_median:g}/"
            f"{self.query_len_mean:.1f}/{self.query_len_max} 字符"
        )
        return f"{text}；长度分布：{buckets}" if buckets else text


@dataclass
class Dataset:
    """一份加载并校验完成的评测集：记录列表 + _meta 元信息 + 来源路径。"""

    records: list[DatasetRecord]
    meta: dict = field(default_factory=dict)
    path: Path | None = None

    @property
    def n_cases(self) -> int:
        return len(self.records)

    def stats(self) -> DatasetStats:
        return DatasetStats.from_records(self.records)

    def to_eval_cases(self) -> list[EvalCase]:
        """转成评测管线可直接消费的 EvalCase 列表（grades/notes/行号收进 meta）。"""
        return [
            EvalCase(
                qid=r.query_id,
                query=r.query,
                relevant_ids=list(r.relevant_doc_ids),
                answer="",
                meta={"grades": dict(r.grades), "notes": r.notes, "line": r.line},
            )
            for r in self.records
        ]

    def unknown_doc_ids(self, corpus_doc_ids: Iterable[str]) -> dict[str, list[str]]:
        """返回 {query_id: [语料中不存在的 doc_id]}，供调用方提示标注与语料不一致。

        不存在的相关文档永远检索不到，会静默拉低召回类指标，因此在 CLI
        加载后显式警告（不阻断：评测集与语料允许分阶段演进）。
        """
        known = set(corpus_doc_ids)
        return {
            r.query_id: missing
            for r in self.records
            if (missing := [doc_id for doc_id in r.relevant_doc_ids if doc_id not in known])
        }


def _length_bucket(n: int) -> str:
    for upper, label in _LENGTH_BUCKETS:
        if n <= upper:
            return label
    return ">50"


def _require_non_empty_str(line_no: int, obj: dict, key: str) -> str:
    value = obj[key]
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"第 {line_no} 行 {key} 必须是非空字符串，收到：{value!r}", line=line_no)
    return value


def _parse_relevant(line_no: int, value: object) -> list[str]:
    if not isinstance(value, list):
        raise DatasetError(
            f"第 {line_no} 行 relevant_doc_ids 必须是字符串数组，收到 {type(value).__name__}", line=line_no
        )
    if not value:
        raise DatasetError(f"第 {line_no} 行 relevant_doc_ids 不能为空：至少标注一篇相关文档", line=line_no)
    ids: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise DatasetError(f"第 {line_no} 行 relevant_doc_ids 含非字符串或空元素：{item!r}", line=line_no)
        ids.append(item)
    duplicated = sorted({doc_id for doc_id in ids if ids.count(doc_id) > 1})
    if duplicated:
        raise DatasetError(f"第 {line_no} 行 relevant_doc_ids 含重复 doc_id：{', '.join(duplicated)}", line=line_no)
    return ids


def _parse_grades(line_no: int, value: object, relevant: list[str]) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise DatasetError(
            f"第 {line_no} 行 grades 必须是对象（{{doc_id: 0|1|2}}），收到 {type(value).__name__}", line=line_no
        )
    relevant_set = set(relevant)
    grades: dict[str, int] = {}
    for key, grade in value.items():
        if not isinstance(key, str) or key not in relevant_set:
            raise DatasetError(f"第 {line_no} 行 grades 的键 {key!r} 不在该行 relevant_doc_ids 中", line=line_no)
        if isinstance(grade, bool) or not isinstance(grade, int) or grade not in GRADE_VALUES:
            raise DatasetError(f"第 {line_no} 行 grades[{key!r}] 的值必须是 0/1/2 整数，收到：{grade!r}", line=line_no)
        grades[key] = grade
    return grades


def _parse_record(line_no: int, obj: dict, seen_ids: dict[str, int]) -> DatasetRecord:
    missing = [key for key in REQUIRED_KEYS if key not in obj]
    if missing:
        raise DatasetError(
            f"第 {line_no} 行缺少必填字段：{', '.join(missing)}（必填：{', '.join(REQUIRED_KEYS)}）", line=line_no
        )
    unknown = sorted(set(obj) - ALLOWED_KEYS)
    if unknown:
        raise DatasetError(
            f"第 {line_no} 行存在未知字段：{', '.join(unknown)}"
            f"（允许的字段：{', '.join(sorted(ALLOWED_KEYS))}，字段名是否拼写有误？）",
            line=line_no,
        )
    query_id = _require_non_empty_str(line_no, obj, "query_id")
    query = _require_non_empty_str(line_no, obj, "query")
    relevant = _parse_relevant(line_no, obj["relevant_doc_ids"])
    grades = _parse_grades(line_no, obj.get("grades"), relevant)
    notes = obj.get("notes", "")
    if not isinstance(notes, str):
        raise DatasetError(f"第 {line_no} 行 notes 必须是字符串，收到 {type(notes).__name__}", line=line_no)
    if query_id in seen_ids:
        raise DatasetError(
            f"第 {line_no} 行 query_id {query_id!r} 与第 {seen_ids[query_id]} 行重复；query_id 必须全局唯一",
            line=line_no,
        )
    seen_ids[query_id] = line_no
    return DatasetRecord(
        query_id=query_id, query=query, relevant_doc_ids=relevant, grades=grades, notes=notes, line=line_no
    )


def _parse_lines(text: str) -> tuple[list[DatasetRecord], dict]:
    records: list[DatasetRecord] = []
    meta: dict = {}
    meta_line: int | None = None
    seen_ids: dict[str, int] = {}
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"第 {line_no} 行不是合法 JSON：{exc.msg}（列 {exc.colno}）", line=line_no) from None
        if not isinstance(obj, dict):
            raise DatasetError(f"第 {line_no} 行必须是 JSON 对象，收到 {type(obj).__name__}", line=line_no)
        if META_KEY in obj:
            if meta_line is not None:
                raise DatasetError(
                    f"第 {line_no} 行出现重复的 {META_KEY} 元信息（首见于第 {meta_line} 行）；"
                    f"{META_KEY} 只允许出现在首个非空行",
                    line=line_no,
                )
            if records:
                raise DatasetError(
                    f"第 {line_no} 行的 {META_KEY} 元信息必须出现在首个非空行（此前已有查询记录）", line=line_no
                )
            extra = sorted(set(obj) - {META_KEY})
            if extra:
                raise DatasetError(
                    f"第 {line_no} 行的 {META_KEY} 元信息行不得混入查询字段：{', '.join(extra)}", line=line_no
                )
            meta_value = obj[META_KEY]
            if not isinstance(meta_value, dict):
                raise DatasetError(f"第 {line_no} 行 {META_KEY} 的值必须是 JSON 对象", line=line_no)
            meta = meta_value
            meta_line = line_no
            continue
        records.append(_parse_record(line_no, obj, seen_ids))
    if not records:
        raise DatasetError("评测集为空：未解析到任何查询记录（除 _meta 外至少需要一行查询记录）")
    return records, meta


def load_dataset(path: str | Path) -> Dataset:
    """加载并严格校验 JSONL 评测集，返回 Dataset。

    校验规则：逐行 JSON 对象；必填字段齐全且类型正确；query_id 全局唯一；
    relevant_doc_ids 非空且行内不重复；grades 键必须落在 relevant_doc_ids
    内、值取 0/1/2 整数；_meta 只允许出现在首个非空行。任何违规抛
    DatasetError，message 带行号。
    """
    file = Path(path)
    if not file.is_file():
        raise DatasetError(f"评测集文件不存在：{file}")
    try:
        text = file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"评测集文件读取失败：{file}（{exc}）") from None
    records, meta = _parse_lines(text)
    return Dataset(records=records, meta=meta, path=file)
