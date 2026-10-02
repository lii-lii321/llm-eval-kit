"""badcase 归因：对 top-K 未命中的查询按启发式规则分类。

三类原因（判定顺序即优先级）：
1. corpus_missing 语料缺失：相关文档根本不在检索器语料里；
2. keyword_mismatch 关键词不匹配：查询与相关文档零词面重叠（改写/同义表达）；
3. semantic_drift 语义漂移：有词面重叠但相关文档仍被挤出 top-K
   （查询词在语料中分布过广，词面打分被其他文档压制）。

规则纯词面统计、可解释，但只是启发式，不等于真实失败机理。
"""

from __future__ import annotations

from dataclasses import dataclass

from .synth import EvalCase
from .tokenize import tokenize

REASON_KEYWORD_MISMATCH = "keyword_mismatch"
REASON_SEMANTIC_DRIFT = "semantic_drift"
REASON_CORPUS_MISSING = "corpus_missing"

REASON_LABELS_ZH = {
    REASON_KEYWORD_MISMATCH: "关键词不匹配",
    REASON_SEMANTIC_DRIFT: "语义漂移",
    REASON_CORPUS_MISSING: "语料缺失",
}


@dataclass
class Badcase:
    """一条未命中用例及其归因。"""

    qid: str
    query: str
    reason: str
    detail: str = ""


def attribute_badcases(
    cases: list[EvalCase],
    ranked_by_qid: dict[str, list[str]],
    doc_texts: dict[str, str],
    *,
    top_k: int = 10,
) -> list[Badcase]:
    """对每条用例判定是否未命中并归类；命中用例不出现在结果中。"""
    badcases: list[Badcase] = []
    for case in cases:
        ranked = ranked_by_qid.get(case.qid, [])
        if set(ranked[:top_k]) & set(case.relevant_ids):
            continue
        missing = [rid for rid in case.relevant_ids if rid not in doc_texts]
        if missing:
            badcases.append(
                Badcase(
                    qid=case.qid,
                    query=case.query,
                    reason=REASON_CORPUS_MISSING,
                    detail=f"相关文档不在语料中：{', '.join(missing)}",
                )
            )
            continue
        q_tokens = set(tokenize(case.query))
        best_overlap = 0
        best_doc = ""
        for rid in case.relevant_ids:
            overlap = q_tokens & set(tokenize(doc_texts[rid]))
            if len(overlap) > best_overlap:
                best_overlap = len(overlap)
                best_doc = rid
        if best_overlap == 0:
            badcases.append(
                Badcase(
                    qid=case.qid,
                    query=case.query,
                    reason=REASON_KEYWORD_MISMATCH,
                    detail="查询与相关文档零词面重叠，多为同义改写或换词表达",
                )
            )
        else:
            badcases.append(
                Badcase(
                    qid=case.qid,
                    query=case.query,
                    reason=REASON_SEMANTIC_DRIFT,
                    detail=f"查询与 {best_doc} 有 {best_overlap} 个重叠词元，但仍被其他文档挤出 top-{top_k}",
                )
            )
    return badcases


def summarize_attribution(badcases: list[Badcase]) -> dict[str, int]:
    """按原因计数，保证三类原因键始终存在。"""
    counts = {
        REASON_KEYWORD_MISMATCH: 0,
        REASON_SEMANTIC_DRIFT: 0,
        REASON_CORPUS_MISSING: 0,
    }
    for badcase in badcases:
        counts[badcase.reason] = counts.get(badcase.reason, 0) + 1
    return counts
