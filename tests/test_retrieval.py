import pytest

from llm_eval_kit.corpus import Doc
from llm_eval_kit.retrieval import BM25Retriever, HybridRetriever, RetrievalResult, TFRetriever


class TestBM25Retriever:
    def test_ranks_relevant_doc_first(self, toy_docs):
        retriever = BM25Retriever(toy_docs)
        results = retriever.retrieve("BM25 词频 稀疏检索", top_k=3)
        assert results[0].doc_id == "d01"

    def test_top_k_respected(self, toy_docs):
        results = BM25Retriever(toy_docs).retrieve("检索", top_k=2)
        assert len(results) == 2

    def test_scores_descending(self, toy_docs):
        results = BM25Retriever(toy_docs).retrieve("向量数据库 最近邻", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_corpus_returns_empty(self):
        assert BM25Retriever([]).retrieve("查询", top_k=3) == []

    def test_invalid_top_k_raises(self, toy_docs):
        with pytest.raises(ValueError):
            BM25Retriever(toy_docs).retrieve("查询", top_k=0)

    def test_duplicate_doc_id_raises(self):
        docs = [Doc(doc_id="d1", text="甲"), Doc(doc_id="d1", text="乙")]
        with pytest.raises(ValueError):
            BM25Retriever(docs)

    def test_deterministic_across_calls(self, toy_docs):
        retriever = BM25Retriever(toy_docs)
        first = [r.doc_id for r in retriever.retrieve("微调 LoRA", top_k=3)]
        second = [r.doc_id for r in retriever.retrieve("微调 LoRA", top_k=3)]
        assert first == second

    def test_doc_texts_exposed(self, toy_docs):
        assert set(BM25Retriever(toy_docs).doc_texts) == {"d01", "d02", "d03"}

    def test_zero_overlap_returns_empty(self, toy_docs):
        # 零词面重叠的文档得分为 0，不作为结果返回（见 BaseRetriever 约定）
        assert BM25Retriever(toy_docs).retrieve("完全不相关词", top_k=3) == []

    def test_zero_score_docs_filtered_from_partial_match(self, toy_docs):
        # 查询词只命中 d01/d02 时，d03（零重叠）不应出现在结果里
        results = BM25Retriever(toy_docs).retrieve("BM25 词频 检索", top_k=3)
        assert all(r.score > 0 for r in results)


class TestTFRetriever:
    def test_ranks_relevant_doc_first(self, toy_docs):
        results = TFRetriever(toy_docs).retrieve("LoRA 参数高效微调", top_k=3)
        assert results[0].doc_id == "d03"

    def test_returns_retrieval_result(self, toy_docs):
        results = TFRetriever(toy_docs).retrieve("向量", top_k=2)
        assert results
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_invalid_top_k_raises(self, toy_docs):
        with pytest.raises(ValueError):
            TFRetriever(toy_docs).retrieve("查询", top_k=-1)


class TestHybridRetriever:
    def test_alpha_one_matches_bm25_order(self, toy_docs):
        query = "向量数据库 最近邻"
        bm25 = [r.doc_id for r in BM25Retriever(toy_docs).retrieve(query, top_k=3)]
        hybrid = [r.doc_id for r in HybridRetriever(toy_docs, alpha=1.0).retrieve(query, top_k=3)]
        assert hybrid == bm25

    def test_alpha_zero_matches_tf_order(self, toy_docs):
        query = "LoRA 参数高效微调"
        tf = [r.doc_id for r in TFRetriever(toy_docs).retrieve(query, top_k=3)]
        hybrid = [r.doc_id for r in HybridRetriever(toy_docs, alpha=0.0).retrieve(query, top_k=3)]
        assert hybrid == tf

    def test_deterministic(self, toy_docs):
        retriever = HybridRetriever(toy_docs, alpha=0.5)
        first = [r.doc_id for r in retriever.retrieve("检索", top_k=3)]
        second = [r.doc_id for r in retriever.retrieve("检索", top_k=3)]
        assert first == second

    def test_results_within_corpus(self, toy_docs):
        results = HybridRetriever(toy_docs, alpha=0.5).retrieve("向量数据库 微调", top_k=3)
        assert {r.doc_id for r in results} <= {"d01", "d02", "d03"}

    def test_zero_overlap_returns_empty(self, toy_docs):
        assert HybridRetriever(toy_docs, alpha=0.5).retrieve("完全不相关词", top_k=3) == []

    def test_invalid_alpha_raises(self, toy_docs):
        with pytest.raises(ValueError):
            HybridRetriever(toy_docs, alpha=1.5)
