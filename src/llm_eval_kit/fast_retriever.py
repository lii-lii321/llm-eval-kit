"""大语料快速 BM25：倒排索引 + 按候选集累加打分，公式与 BM25Retriever 完全一致。

与玩具实现的唯一差异在候选集：玩具版每次查询全量扫描全部文档（O(N×查询词数)），
快速版只遍历查询词元的倒排表（含任一查询词元的文档）。中文二元组分词下查询词元
普遍稀有，语料越大、查询越具体，跳过的文档比例越高，加速越明显。

打分公式（Okapi BM25）、零分不返回、并列按 doc_id 字典序稳定排序等口径
与 BM25Retriever 保持一致；查询词元按排序后顺序累加（消除 set 迭代序带来的
浮点累加次序波动）。仍为单机内存索引，非生产级检索引擎。
"""

from __future__ import annotations

import math
from collections import Counter

from .corpus import Doc
from .retrieval import BaseRetriever, RetrievalResult
from .tokenize import tokenize


class FastBM25Retriever(BaseRetriever):
    """倒排索引版 BM25（Okapi），打分口径与 BM25Retriever 相同，适合千段以上语料。"""

    name = "bm25_fast"

    def __init__(self, docs: list[Doc], *, k1: float = 1.5, b: float = 0.75):
        super().__init__(docs)
        self.k1 = k1
        self.b = b
        self._doc_ids = [d.doc_id for d in docs]
        self._n_docs = len(docs)
        total_len = 0
        postings: dict[str, list[tuple[int, int]]] = {}
        doc_len: list[int] = [0] * self._n_docs
        for idx, doc in enumerate(docs):
            tfs = Counter(tokenize(doc.text))
            dl = sum(tfs.values())
            doc_len[idx] = dl
            total_len += dl
            for token, freq in tfs.items():
                postings.setdefault(token, []).append((idx, freq))
        self._doc_len = doc_len
        self._postings = postings
        self._avgdl = (total_len / self._n_docs) if self._n_docs else 0.0
        # 每个词元的 idf 只依赖 df（倒排表长度），构建期一次算好
        self._idf = {
            token: math.log(1 + (self._n_docs - len(pl) + 0.5) / (len(pl) + 0.5)) for token, pl in postings.items()
        }

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        self._check_top_k(top_k)
        if not self._doc_ids:
            return []
        q_tokens = sorted(set(tokenize(query)))
        scores: dict[int, float] = {}
        for token in q_tokens:
            pl = self._postings.get(token)
            if not pl:
                continue
            idf = self._idf[token]
            k1, b, avgdl = self.k1, self.b, self._avgdl
            doc_len = self._doc_len
            for idx, freq in pl:
                denom = freq + k1 * (1 - b + b * doc_len[idx] / avgdl)
                scores[idx] = scores.get(idx, 0.0) + idf * freq * (k1 + 1) / denom
        ranked = sorted(
            ((self._doc_ids[idx], score) for idx, score in scores.items()),
            key=lambda x: (-x[1], x[0]),
        )
        return [RetrievalResult(doc_id=doc_id, score=score) for doc_id, score in ranked[:top_k]]
