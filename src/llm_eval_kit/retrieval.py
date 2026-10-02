"""玩具检索器：BM25、词频 TF 与加权混合，全部纯 Python 离线实现。

仅适合演示与小语料（几百块以内）评测，未做任何性能优化；
生产环境请换 Elasticsearch / 向量数据库等真实检索引擎。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass

from .corpus import Doc
from .tokenize import tokenize


@dataclass(frozen=True)
class RetrievalResult:
    """单条检索结果：文档 ID 与得分（仅用于排序，不保证绝对数值含义）。"""

    doc_id: str
    score: float


class BaseRetriever(ABC):
    """检索器接口：给定查询返回按得分降序的 top-K 结果。

    约定：得分为 0（与查询零词面重叠）的文档不作为结果返回——
    没有词面证据的文档排进结果纯属 doc_id 平局噪声，还会掩盖真实的未命中。
    """

    name: str = "retriever"

    def __init__(self, docs: list[Doc]):
        doc_ids = [d.doc_id for d in docs]
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("语料中存在重复 doc_id")
        self._texts = {d.doc_id: d.text for d in docs}

    @property
    def doc_texts(self) -> dict[str, str]:
        """doc_id 到原文的映射，供 badcase 归因使用。"""
        return dict(self._texts)

    @staticmethod
    def _check_top_k(top_k: int) -> None:
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """返回 top-K 检索结果；并列得分按 doc_id 字典序稳定排序。"""


class BM25Retriever(BaseRetriever):
    """经典 BM25（Okapi）打分的稀疏检索器。"""

    name = "bm25"

    def __init__(self, docs: list[Doc], *, k1: float = 1.5, b: float = 0.75):
        super().__init__(docs)
        self.k1 = k1
        self.b = b
        self._doc_ids = [d.doc_id for d in docs]
        self._tfs = {d.doc_id: Counter(tokenize(d.text)) for d in docs}
        self._doc_len = {d.doc_id: sum(self._tfs[d.doc_id].values()) for d in docs}
        df: Counter = Counter()
        for tf in self._tfs.values():
            df.update(tf.keys())
        self._df = df
        self._n_docs = len(docs)
        self._avgdl = (sum(self._doc_len.values()) / len(docs)) if docs else 0.0

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        self._check_top_k(top_k)
        if not self._doc_ids:
            return []
        q_tokens = set(tokenize(query))
        scores: list[tuple[float, str]] = []
        for doc_id in self._doc_ids:
            tf = self._tfs[doc_id]
            doc_len = self._doc_len[doc_id]
            total = 0.0
            for token in q_tokens:
                freq = tf.get(token, 0)
                if freq == 0:
                    continue
                idf = math.log(1 + (self._n_docs - self._df[token] + 0.5) / (self._df[token] + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)
                total += idf * freq * (self.k1 + 1) / denom
            scores.append((total, doc_id))
        scores = [(score, doc_id) for score, doc_id in scores if score > 0]
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [RetrievalResult(doc_id=doc_id, score=score) for score, doc_id in scores[:top_k]]


class TFRetriever(BaseRetriever):
    """朴素词频检索器：查询词在文档中的出现次数，按文档长度平方根归一。"""

    name = "tf"

    def __init__(self, docs: list[Doc]):
        super().__init__(docs)
        self._doc_ids = [d.doc_id for d in docs]
        self._tfs = {d.doc_id: Counter(tokenize(d.text)) for d in docs}
        self._doc_len = {d.doc_id: sum(self._tfs[d.doc_id].values()) for d in docs}

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        self._check_top_k(top_k)
        if not self._doc_ids:
            return []
        q_tokens = set(tokenize(query))
        scores: list[tuple[float, str]] = []
        for doc_id in self._doc_ids:
            tf = self._tfs[doc_id]
            raw = sum(tf[t] for t in q_tokens)
            doc_len = self._doc_len[doc_id]
            normalized = raw / (doc_len**0.5) if doc_len else 0.0
            scores.append((normalized, doc_id))
        scores = [(score, doc_id) for score, doc_id in scores if score > 0]
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [RetrievalResult(doc_id=doc_id, score=score) for score, doc_id in scores[:top_k]]


class HybridRetriever(BaseRetriever):
    """混合检索：BM25 与 TF 的归一化得分加权融合，alpha 为 BM25 权重。"""

    name = "hybrid"

    def __init__(self, docs: list[Doc], *, alpha: float = 0.5):
        super().__init__(docs)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha 取值范围 [0, 1]")
        self.alpha = alpha
        self._bm25 = BM25Retriever(docs)
        self._tf = TFRetriever(docs)
        self._doc_ids = [d.doc_id for d in docs]

    @staticmethod
    def _normalize(results: list[RetrievalResult]) -> dict[str, float]:
        if not results:
            return {}
        max_score = max(r.score for r in results)
        if max_score <= 0:
            return {r.doc_id: 0.0 for r in results}
        return {r.doc_id: r.score / max_score for r in results}

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        self._check_top_k(top_k)
        if not self._doc_ids:
            return []
        full_k = len(self._doc_ids)
        bm25_scores = self._normalize(self._bm25.retrieve(query, top_k=full_k))
        tf_scores = self._normalize(self._tf.retrieve(query, top_k=full_k))
        combined: dict[str, float] = {}
        for doc_id in self._doc_ids:
            combined[doc_id] = self.alpha * bm25_scores.get(doc_id, 0.0) + (1 - self.alpha) * tf_scores.get(doc_id, 0.0)
        ranked = sorted(
            ((doc_id, score) for doc_id, score in combined.items() if score > 0),
            key=lambda x: (-x[1], x[0]),
        )
        return [RetrievalResult(doc_id=doc_id, score=score) for doc_id, score in ranked[:top_k]]
