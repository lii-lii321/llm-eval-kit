"""RRFHybridRetriever：融合公式、语料一致性校验与边界行为。"""

from __future__ import annotations

import pytest

from llm_eval_kit.adapters import ensure_retriever
from llm_eval_kit.corpus import Doc
from llm_eval_kit.retrieval import BaseRetriever, RetrievalResult, RRFHybridRetriever


class _FixedRankRetriever(BaseRetriever):
    """按给定 doc_id 顺序返回固定名次的桩检索器（score = 1/rank，仅占位）。"""

    def __init__(self, name: str, docs: list[Doc], order: list[str]):
        super().__init__(docs)
        self.name = name
        self._order = order

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        self._check_top_k(top_k)
        return [
            RetrievalResult(doc_id=doc_id, score=1.0 / rank)
            for rank, doc_id in enumerate(self._order, start=1)
        ][:top_k]


def _docs() -> list[Doc]:
    return [Doc(doc_id=d, text=f"文本 {d}") for d in ("d1", "d2", "d3")]


def test_rrf_formula_hand_computed() -> None:
    docs = _docs()
    a = _FixedRankRetriever("a", docs, ["d1", "d2", "d3"])  # d1 rank1, d2 rank2
    b = _FixedRankRetriever("b", docs, ["d2", "d1", "d3"])  # d2 rank1, d1 rank2
    rrf = RRFHybridRetriever(docs, k=1, retrievers=[a, b])
    hits = rrf.retrieve("任意查询", top_k=3)
    # d1: 1/(1+1) + 1/(1+2) = 0.5 + 0.3333 = 0.8333
    # d2: 1/(1+2) + 1/(1+1) = 0.3333 + 0.5   = 0.8333（与 d1 并列）
    # d3: 1/(1+3) + 1/(1+3) = 0.25
    assert [h.doc_id for h in hits] == ["d1", "d2", "d3"]  # 并列按 doc_id 字典序
    assert abs(hits[0].score - (1 / 2 + 1 / 3)) < 1e-12
    assert abs(hits[2].score - (1 / 4 + 1 / 4)) < 1e-12


def test_rrf_prefers_consensually_ranked_document() -> None:
    docs = _docs()
    a = _FixedRankRetriever("a", docs, ["d1", "d2", "d3"])
    b = _FixedRankRetriever("b", docs, ["d1", "d3", "d2"])
    rrf = RRFHybridRetriever(docs, k=60, retrievers=[a, b])
    assert rrf.retrieve("q", top_k=1)[0].doc_id == "d1"  # 两路都把 d1 排第一


def test_k_and_retriever_validation() -> None:
    docs = _docs()
    with pytest.raises(ValueError, match="k 必须为正整数"):
        RRFHybridRetriever(docs, k=0)
    with pytest.raises(ValueError, match="至少需要一路"):
        RRFHybridRetriever(docs, retrievers=[])


def test_foreign_corpus_rejected() -> None:
    docs = _docs()
    foreign = _FixedRankRetriever("x", [Doc(doc_id="z9", text="别的语料")], ["z9"])
    with pytest.raises(ValueError, match="不一致"):
        RRFHybridRetriever(docs, retrievers=[foreign])


def test_protocol_compliance_and_empty_query() -> None:
    docs = _docs()
    rrf = RRFHybridRetriever(docs)
    ensure_retriever(rrf)
    assert rrf.retrieve("", top_k=5) == []
