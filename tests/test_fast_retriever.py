"""FastBM25Retriever：协议一致性、与玩具 BM25 的口径一致性与边界行为。"""

from __future__ import annotations

import pytest

from llm_eval_kit.adapters import ensure_retriever
from llm_eval_kit.corpus import Doc
from llm_eval_kit.fast_retriever import FastBM25Retriever
from llm_eval_kit.retrieval import BM25Retriever


def _corpus() -> list[Doc]:
    return [
        Doc(doc_id="d01", text="机器学习 模型 训练 数据 集过拟合 正则化"),
        Doc(doc_id="d02", text="深度学习 神经网络 梯度 下降 反向 传播"),
        Doc(doc_id="d03", text="检索 增强 生成 向量 数据 库语义 匹配"),
        Doc(doc_id="d04", text="机器学习 梯度 优化 数据 增强技巧"),
        Doc(doc_id="d05", text="异常 检测 多元 高斯 分布模型"),
    ]


def test_protocol_compliance() -> None:
    retriever = FastBM25Retriever(_corpus())
    ensure_retriever(retriever)  # 不抛即合规


def test_matches_toy_bm25_topk_and_scores() -> None:
    docs = _corpus()
    queries = ["机器学习 数据", "梯度 下降", "向量 检索 匹配", "增强", "模型 训练"]
    old = BM25Retriever(docs)
    fast = FastBM25Retriever(docs)
    for q in queries:
        a = old.retrieve(q, top_k=5)
        b = fast.retrieve(q, top_k=5)
        assert [r.doc_id for r in a] == [r.doc_id for r in b]
        assert all(abs(x.score - y.score) < 1e-9 for x, y in zip(a, b, strict=True))


def test_zero_overlap_returns_empty() -> None:
    fast = FastBM25Retriever(_corpus())
    assert fast.retrieve("完全无关词汇表", top_k=5) == []


def test_empty_corpus_and_query() -> None:
    fast = FastBM25Retriever([])
    assert fast.retrieve("任何查询", top_k=5) == []
    fast2 = FastBM25Retriever(_corpus())
    assert fast2.retrieve("", top_k=5) == []


def test_top_k_must_be_positive() -> None:
    fast = FastBM25Retriever(_corpus())
    with pytest.raises(ValueError, match="top_k"):
        fast.retrieve("机器学习", top_k=0)


def test_duplicate_doc_id_rejected() -> None:
    docs = [Doc(doc_id="d01", text="重复 id"), Doc(doc_id="d01", text="重复 id 之二")]
    with pytest.raises(ValueError, match="重复"):
        FastBM25Retriever(docs)


def test_document_texts_property() -> None:
    docs = _corpus()
    fast = FastBM25Retriever(docs)
    assert fast.doc_texts == {d.doc_id: d.text for d in docs}
