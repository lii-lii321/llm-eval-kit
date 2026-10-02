from llm_eval_kit.attribution import (
    REASON_CORPUS_MISSING,
    REASON_KEYWORD_MISMATCH,
    REASON_SEMANTIC_DRIFT,
    Badcase,
    attribute_badcases,
    summarize_attribution,
)
from llm_eval_kit.synth import EvalCase

DOCS = {
    "d01": "BM25 基于词频与逆文档频率对文档打分。",
    "d02": "向量数据库支持近似最近邻检索。",
    "d03": "LoRA 是参数高效微调方法。",
}


def _cases() -> list[EvalCase]:
    return [
        EvalCase(qid="q1", query="BM25 怎么打分", relevant_ids=["d01"]),
        EvalCase(qid="q2", query="什么是卡片盒子", relevant_ids=["d02"]),
        EvalCase(qid="q3", query="微调方法", relevant_ids=["d99"]),
    ]


class TestAttributeBadcases:
    def test_all_hit_no_badcases(self):
        cases = [
            EvalCase(qid="q1", query="BM25 打分", relevant_ids=["d01"]),
            EvalCase(qid="q2", query="向量数据库", relevant_ids=["d02"]),
        ]
        ranked = {"q1": ["d01", "d02"], "q2": ["d02", "d01"]}
        assert attribute_badcases(cases, ranked, DOCS, top_k=2) == []

    def test_hit_within_top_k_counts(self):
        cases = [EvalCase(qid="q1", query="BM25 打分", relevant_ids=["d01"])]
        ranked = {"q1": ["d02", "d01"]}
        assert attribute_badcases(cases, ranked, DOCS, top_k=2) == []
        assert attribute_badcases(cases, ranked, DOCS, top_k=1) != []

    def test_corpus_missing(self):
        ranked = {"q1": ["d01"], "q2": ["d02"], "q3": ["d01"]}
        badcases = attribute_badcases(_cases(), ranked, DOCS, top_k=1)
        by_qid = {b.qid: b.reason for b in badcases}
        assert by_qid["q3"] == REASON_CORPUS_MISSING

    def test_keyword_mismatch(self):
        ranked = {"q1": ["d01"], "q2": ["d03"], "q3": ["d01"]}
        badcases = attribute_badcases(_cases(), ranked, DOCS, top_k=1)
        by_qid = {b.qid: b.reason for b in badcases}
        assert by_qid["q2"] == REASON_KEYWORD_MISMATCH

    def test_semantic_drift(self):
        ranked = {"q1": ["d03"], "q2": ["d02"], "q3": ["d01"]}
        badcases = attribute_badcases(_cases(), ranked, DOCS, top_k=1)
        by_qid = {b.qid: b.reason for b in badcases}
        assert by_qid["q1"] == REASON_SEMANTIC_DRIFT

    def test_badcase_carries_query_and_detail(self):
        ranked = {"q1": ["d01"], "q2": ["d03"], "q3": ["d01"]}
        badcases = attribute_badcases(_cases(), ranked, DOCS, top_k=1)
        for b in badcases:
            assert isinstance(b, Badcase)
            assert b.query
            assert b.detail


class TestSummarizeAttribution:
    def test_counts_by_reason(self):
        badcases = [
            Badcase("q1", "a", REASON_KEYWORD_MISMATCH),
            Badcase("q2", "b", REASON_KEYWORD_MISMATCH),
            Badcase("q3", "c", REASON_SEMANTIC_DRIFT),
            Badcase("q4", "d", REASON_CORPUS_MISSING),
        ]
        counts = summarize_attribution(badcases)
        assert counts[REASON_KEYWORD_MISMATCH] == 2
        assert counts[REASON_SEMANTIC_DRIFT] == 1
        assert counts[REASON_CORPUS_MISSING] == 1

    def test_empty_has_all_keys(self):
        counts = summarize_attribution([])
        assert counts == {
            REASON_KEYWORD_MISMATCH: 0,
            REASON_SEMANTIC_DRIFT: 0,
            REASON_CORPUS_MISSING: 0,
        }
