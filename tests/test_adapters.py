"""外部检索器适配层测试：协议校验、归一化、错误可读性与端到端小评测。"""

import pytest

from llm_eval_kit import (
    CallableRetriever,
    Hit,
    RetrieverAdapter,
    RetrieverProtocolError,
    as_eval_retriever,
    ensure_retriever,
    normalize_hit,
    normalize_hits,
    run_evaluation,
    run_retrieval,
)
from llm_eval_kit.demo import build_demo_corpus
from llm_eval_kit.retrieval import BM25Retriever, RetrievalResult
from llm_eval_kit.synth import EvalCase
from llm_eval_kit.tokenize import tokenize


class TokenOverlapRetriever:
    """模拟外部检索系统：按查询/文档词元重叠数打分，返回 Hit。"""

    name = "external-overlap"

    def __init__(self, docs):
        self.doc_texts = {d.doc_id: d.text for d in docs}

    def retrieve(self, query: str, top_k: int = 10) -> list[Hit]:
        q_tokens = set(tokenize(query))
        scored = []
        for doc_id, text in self.doc_texts.items():
            overlap = len(q_tokens & set(tokenize(text)))
            if overlap:
                scored.append((overlap, doc_id))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [Hit(doc_id=doc_id, score=float(overlap)) for overlap, doc_id in scored[:top_k]]


class KeywordOnlyTopKRetriever:
    """top_k 只接受关键字参数的外部检索器（协议宽容这类签名）。"""

    name = "keyword-only"

    def __init__(self, doc_ids: list[str]):
        self._doc_ids = doc_ids

    def retrieve(self, query: str, *, top_k: int = 10) -> list[Hit]:
        return [Hit(doc_id=doc_id) for doc_id in self._doc_ids[:top_k]]


def _two_cases() -> list[EvalCase]:
    return [
        EvalCase(qid="q001", query="什么是RAG？", relevant_ids=["d01"], answer="RAG 是检索增强生成。"),
        EvalCase(qid="q002", query="什么是BM25？", relevant_ids=["d03"], answer="BM25 是经典稀疏检索算法。"),
    ]


class TestHit:
    def test_defaults(self):
        hit = Hit(doc_id="d01")
        assert hit.doc_id == "d01"
        assert hit.score is None

    def test_frozen(self):
        hit = Hit(doc_id="d01", score=0.5)
        with pytest.raises(AttributeError):
            hit.doc_id = "d02"  # type: ignore[misc]


class TestNormalizeHit:
    def test_hit_passthrough(self):
        hit = Hit(doc_id="d1", score=0.7)
        assert normalize_hit(hit) is hit

    def test_bare_doc_id_string(self):
        assert normalize_hit("d1") == Hit(doc_id="d1")

    def test_pair_tuple(self):
        assert normalize_hit(("d1", 0.9)) == Hit(doc_id="d1", score=0.9)

    def test_score_coerced_to_float(self):
        hit = normalize_hit(("d1", 2))
        assert hit.score == 2.0
        assert isinstance(hit.score, float)

    def test_single_element_tuple(self):
        assert normalize_hit(("d1",)) == Hit(doc_id="d1")

    def test_object_with_doc_id_attribute(self):
        results = normalize_hit(RetrievalResult(doc_id="d1", score=0.3))
        assert results == Hit(doc_id="d1", score=0.3)

    def test_object_without_score_attribute(self):
        class Row:
            doc_id = "r1"

        assert normalize_hit(Row()) == Hit(doc_id="r1")

    def test_unknown_shape_raises_readable_error(self):
        with pytest.raises(TypeError) as excinfo:
            normalize_hit(42)
        assert "Hit" in str(excinfo.value)
        assert "int" in str(excinfo.value)

    def test_oversized_tuple_raises(self):
        with pytest.raises(TypeError) as excinfo:
            normalize_hit(("d1", 0.5, "extra"))
        assert "(doc_id, score)" in str(excinfo.value)

    def test_normalize_hits_batch(self):
        hits = normalize_hits([("a", 0.5), "b", Hit("c")])
        assert [h.doc_id for h in hits] == ["a", "b", "c"]
        assert hits[0].score == 0.5
        assert hits[1].score is None


class TestCallableRetriever:
    def test_wraps_callable_and_normalizes(self):
        retriever = CallableRetriever(lambda query, top_k: [("d1", 0.9), "d2"][:top_k], name="mock")
        assert retriever.name == "mock"
        hits = retriever.retrieve("任意查询", top_k=2)
        assert hits == [Hit(doc_id="d1", score=0.9), Hit(doc_id="d2")]

    def test_non_callable_rejected(self):
        with pytest.raises(TypeError) as excinfo:
            CallableRetriever("not-callable")
        assert "可调用" in str(excinfo.value)

    def test_top_k_must_be_positive(self):
        retriever = CallableRetriever(lambda query, top_k: [])
        with pytest.raises(ValueError, match="top_k"):
            retriever.retrieve("q", top_k=0)

    def test_satisfies_protocol_and_ensure_retriever(self):
        retriever = CallableRetriever(lambda query, top_k: [])
        assert ensure_retriever(retriever) is retriever


class TestEnsureRetriever:
    def test_valid_object_passes_through(self):
        retriever = TokenOverlapRetriever(build_demo_corpus())
        assert ensure_retriever(retriever) is retriever

    def test_builtin_retriever_passes(self):
        retriever = BM25Retriever(build_demo_corpus())
        assert ensure_retriever(retriever) is retriever

    def test_missing_method_error_is_readable(self):
        class NotARetriever:
            pass

        with pytest.raises(RetrieverProtocolError) as excinfo:
            ensure_retriever(NotARetriever())
        message = str(excinfo.value)
        assert "retrieve" in message
        assert "NotARetriever" in message
        assert "CallableRetriever" in message
        assert "docs/ADAPTERS.md" in message

    def test_non_callable_attribute_rejected(self):
        class BadRetriever:
            retrieve = "i-am-not-callable"

        with pytest.raises(RetrieverProtocolError) as excinfo:
            ensure_retriever(BadRetriever())
        assert "缺少可调用的 retrieve 方法" in str(excinfo.value)

    def test_wrong_signature_rejected_with_hint(self):
        class NoTopK:
            def retrieve(self, query):
                return []

        with pytest.raises(RetrieverProtocolError) as excinfo:
            ensure_retriever(NoTopK())
        message = str(excinfo.value)
        assert "签名不兼容" in message
        assert "top_k" in message

    def test_keyword_only_top_k_accepted(self):
        retriever = KeywordOnlyTopKRetriever(["d1"])
        assert ensure_retriever(retriever) is retriever

    def test_is_type_error(self):
        with pytest.raises(TypeError):
            ensure_retriever(object())


class TestRetrieverAdapter:
    def test_wraps_protocol_object_with_duck_typed_doc_texts(self):
        docs = build_demo_corpus()
        adapter = RetrieverAdapter(TokenOverlapRetriever(docs))
        assert adapter.name == "external-overlap"
        assert "d01" in adapter.doc_texts
        results = adapter.retrieve("什么是RAG？", top_k=3)
        assert 1 <= len(results) <= 3
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_explicit_name_and_doc_texts_win(self):
        adapter = RetrieverAdapter(
            TokenOverlapRetriever([]), name="custom", doc_texts={"x1": "文本"}
        )
        assert adapter.name == "custom"
        assert adapter.doc_texts == {"x1": "文本"}

    def test_without_doc_texts_attribution_falls_back_to_empty(self):
        adapter = RetrieverAdapter(CallableRetriever(lambda q, k: ["d1"]))
        assert adapter.doc_texts == {}

    def test_top_k_must_be_positive(self):
        adapter = RetrieverAdapter(TokenOverlapRetriever(build_demo_corpus()))
        with pytest.raises(ValueError, match="top_k"):
            adapter.retrieve("q", top_k=-1)

    def test_invalid_inner_object_rejected_at_construction(self):
        with pytest.raises(RetrieverProtocolError):
            RetrieverAdapter(object())

    def test_keyword_only_signature_called_with_keyword(self):
        adapter = RetrieverAdapter(KeywordOnlyTopKRetriever(["d1", "d2", "d3"]))
        hits = adapter.retrieve("q", top_k=2)
        assert [r.doc_id for r in hits] == ["d1", "d2"]

    def test_as_eval_retriever_passthrough_for_base_retriever(self):
        retriever = BM25Retriever(build_demo_corpus())
        assert as_eval_retriever(retriever) is retriever

    def test_as_eval_retriever_wraps_external(self):
        wrapped = as_eval_retriever(TokenOverlapRetriever(build_demo_corpus()))
        assert isinstance(wrapped, RetrieverAdapter)


class TestExternalRetrieverEndToEnd:
    def test_run_evaluation_with_external_retriever(self):
        docs = build_demo_corpus()
        cases = _two_cases()
        data = run_evaluation(
            cases, {"ext": TokenOverlapRetriever(docs), "bm25": BM25Retriever(docs)}, top_k=5, n_boot=0
        )
        assert [p.name for p in data.pipelines] == ["ext", "bm25"]
        assert data.pipelines[0].recall[1] == 1.0
        assert data.pipelines[0].mrr == 1.0
        assert data.corpus_size == len(docs)

    def test_run_evaluation_accepts_callable_retriever_directly(self):
        data = run_evaluation(
            _two_cases(),
            {"fixed": CallableRetriever(lambda q, k: [("d01", 1.0), ("d02", 0.5)][:k])},
            top_k=2,
            n_boot=0,
        )
        assert data.pipelines[0].name == "fixed"
        assert data.pipelines[0].recall[1] == 0.5

    def test_callable_without_doc_texts_classifies_miss_as_corpus_missing(self):
        data = run_evaluation(
            _two_cases(),
            {"fixed": CallableRetriever(lambda q, k: [("d01", 1.0), ("d02", 0.5)][:k])},
            top_k=2,
            n_boot=0,
        )
        assert data.attribution["fixed"]["corpus_missing"] == 1

    def test_adapter_with_doc_texts_attributes_semantic_drift(self):
        data = run_evaluation(
            _two_cases(),
            {"ext": RetrieverAdapter(
                CallableRetriever(lambda q, k: [("d01", 1.0), ("d02", 0.5)][:k]),
                doc_texts={"d01": "RAG 检索增强生成。", "d02": "向量数据库。", "d03": "BM25 稀疏检索算法。"},
            )},
            top_k=2,
            n_boot=0,
        )
        assert data.attribution["ext"]["semantic_drift"] == 1
        assert data.primary_badcases[0].qid == "q002"

    def test_run_retrieval_with_external_retriever(self):
        metrics = run_retrieval(
            _two_cases(), TokenOverlapRetriever(build_demo_corpus()), top_k=5, n_boot=0
        )
        assert metrics.name == "external-overlap"
        assert metrics.recall[1] == 1.0

    def test_object_missing_retrieve_gets_diagnostic_error(self):
        with pytest.raises(RetrieverProtocolError) as excinfo:
            run_evaluation(_two_cases(), {"broken": object()}, top_k=2)
        assert "docs/ADAPTERS.md" in str(excinfo.value)
