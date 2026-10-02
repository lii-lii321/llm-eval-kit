import pytest

from llm_eval_kit.judge import LLMUnavailable, MockJudge
from llm_eval_kit.pipeline import run_evaluation, run_retrieval
from llm_eval_kit.report import render_markdown
from llm_eval_kit.retrieval import BM25Retriever, TFRetriever
from llm_eval_kit.synth import EvalCase, generate_eval_set


class BadJudge:
    """score 必抛 LLMUnavailable 的假裁判，用于测试优雅跳过。"""

    name = "bad"

    def score(self, **kwargs):
        raise LLMUnavailable("EVAL_LLM_API_KEY 未设置（测试）")


class TestRunRetrieval:
    def test_perfect_metrics_on_easy_case(self, toy_docs):
        cases = [EvalCase(qid="q1", query="BM25 词频", relevant_ids=["d01"])]
        metrics = run_retrieval(cases, BM25Retriever(toy_docs), top_k=3)
        assert metrics.name == "bm25"
        assert metrics.recall[1] == 1.0
        assert metrics.mrr == 1.0
        assert metrics.ndcg == 1.0
        assert metrics.hit_rate == 1.0

    def test_miss_gives_zero(self, toy_docs):
        # 零重叠查询所有文档得分并列 0，按 doc_id 字典序 d01 排最前；
        # 相关文档设为 d03 且 top_k=2，确保确实未命中。
        cases = [EvalCase(qid="q1", query="完全不相关的查询词", relevant_ids=["d03"])]
        metrics = run_retrieval(cases, BM25Retriever(toy_docs), top_k=2, ks=(1, 2))
        assert metrics.recall[1] == 0.0
        assert metrics.recall[2] == 0.0
        assert metrics.mrr == 0.0
        assert metrics.hit_rate == 0.0

    def test_latency_measured_and_positive(self, toy_docs):
        cases = [EvalCase(qid="q1", query="BM25", relevant_ids=["d01"])]
        metrics = run_retrieval(cases, BM25Retriever(toy_docs))
        assert metrics.latency.n == 1
        assert metrics.latency.p50 > 0


class TestRunEvaluation:
    def test_multi_pipeline_comparison(self, toy_docs):
        cases = generate_eval_set(toy_docs, num_cases=3, seed=5)
        data = run_evaluation(
            cases,
            {"bm25": BM25Retriever(toy_docs), "tf": TFRetriever(toy_docs)},
            top_k=3,
        )
        assert [p.name for p in data.pipelines] == ["bm25", "tf"]
        assert data.eval_size == len(cases)
        assert data.corpus_size == 3
        assert all(0.0 <= p.recall[3] <= 1.0 for p in data.pipelines)
        assert set(data.attribution) == {"bm25", "tf"}

    def test_requires_at_least_one_pipeline(self, toy_docs):
        with pytest.raises(ValueError):
            run_evaluation([], {})

    def test_mock_judge_summary(self, toy_docs):
        cases = generate_eval_set(toy_docs, num_cases=2, seed=5)
        answers = {c.qid: c.answer for c in cases}
        data = run_evaluation(
            cases,
            {"bm25": BM25Retriever(toy_docs)},
            judge=MockJudge(),
            answers=answers,
        )
        assert data.judge.n_scored == 2
        assert data.judge.provider == "mock"
        assert 1 <= data.judge.dim_means["correctness"] <= 5
        assert not data.judge.skipped

    def test_judge_unavailable_skips_gracefully(self, toy_docs):
        cases = generate_eval_set(toy_docs, num_cases=2, seed=5)
        answers = {c.qid: "某个系统答案" for c in cases}
        data = run_evaluation(
            cases,
            {"bm25": BM25Retriever(toy_docs)},
            judge=BadJudge(),
            answers=answers,
        )
        assert data.judge.skipped
        assert "EVAL_LLM_API_KEY" in data.judge.skip_reason

    def test_no_judge_by_default(self, toy_docs):
        cases = generate_eval_set(toy_docs, num_cases=2, seed=5)
        data = run_evaluation(cases, {"bm25": BM25Retriever(toy_docs)})
        assert data.judge.n_scored == 0
        assert not data.judge.skipped

    def test_retriever_called_once_per_case(self, toy_docs):
        """同一查询对同一检索器只检索一次，指标/归因/judge 上下文复用结果。"""
        from llm_eval_kit.retrieval import BM25Retriever

        class CountingRetriever:
            def __init__(self, inner):
                self._inner = inner
                self.calls = 0

            @property
            def name(self):
                return self._inner.name

            @property
            def doc_texts(self):
                return self._inner.doc_texts

            def retrieve(self, query, top_k=10):
                self.calls += 1
                return self._inner.retrieve(query, top_k=top_k)

        cases = generate_eval_set(toy_docs, num_cases=3, seed=5)
        answers = {c.qid: c.answer for c in cases}
        spy = CountingRetriever(BM25Retriever(toy_docs))
        run_evaluation(cases, {"bm25": spy}, top_k=3, judge=MockJudge(), answers=answers)
        assert spy.calls == len(cases)

    def test_report_data_renders(self, toy_docs):
        cases = generate_eval_set(toy_docs, num_cases=3, seed=5)
        data = run_evaluation(
            cases,
            {"bm25": BM25Retriever(toy_docs), "tf": TFRetriever(toy_docs)},
            top_k=3,
        )
        md = render_markdown(data)
        assert "## 检索指标" in md
        assert "badcase" in md


class TestBootstrapConfidence:
    def test_ci_fields_populated(self, toy_docs):
        cases = generate_eval_set(toy_docs, num_cases=5, seed=5)
        metrics = run_retrieval(cases, BM25Retriever(toy_docs), top_k=3)
        assert set(metrics.recall_ci) == set(metrics.ks)
        assert metrics.mrr_ci is not None
        assert metrics.ndcg_ci is not None
        assert metrics.hit_rate_ci is not None
        for ci in [*metrics.recall_ci.values(), metrics.mrr_ci, metrics.ndcg_ci, metrics.hit_rate_ci]:
            assert 0.0 <= ci.low <= ci.high <= 1.0

    def test_point_estimate_unchanged_by_bootstrap(self, toy_docs):
        """聚合指标本身不因引入置信区间而改变。"""
        cases = generate_eval_set(toy_docs, num_cases=5, seed=5)
        metrics = run_retrieval(cases, BM25Retriever(toy_docs), top_k=3, n_boot=0)
        again = run_retrieval(cases, BM25Retriever(toy_docs), top_k=3)
        assert metrics.recall == again.recall
        assert metrics.mrr == again.mrr
        assert metrics.ndcg == again.ndcg
        assert metrics.hit_rate == again.hit_rate

    def test_perfect_case_degenerate_ci(self, toy_docs):
        cases = [EvalCase(qid="q1", query="BM25 词频", relevant_ids=["d01"])]
        metrics = run_retrieval(cases, BM25Retriever(toy_docs), top_k=3)
        assert metrics.mrr_ci is not None
        assert metrics.mrr_ci.low == pytest.approx(1.0)
        assert metrics.mrr_ci.high == pytest.approx(1.0)

    def test_same_seed_reproducible_ci(self, toy_docs):
        cases = generate_eval_set(toy_docs, num_cases=6, seed=5)
        first = run_retrieval(cases, BM25Retriever(toy_docs), top_k=3, seed=9)
        second = run_retrieval(cases, BM25Retriever(toy_docs), top_k=3, seed=9)
        assert first.mrr_ci == second.mrr_ci
        assert first.recall_ci == second.recall_ci

    def test_pipelines_share_resamples_for_same_seed(self, toy_docs):
        """所有管线共用同一 seed 与评测集规模，置信区间可直接横向对比。"""
        cases = generate_eval_set(toy_docs, num_cases=6, seed=5)
        data = run_evaluation(cases, {"bm25": BM25Retriever(toy_docs), "tf": TFRetriever(toy_docs)}, top_k=3)
        levels = {p.mrr_ci.level for p in data.pipelines if p.mrr_ci is not None}
        boots = {p.mrr_ci.n_boot for p in data.pipelines if p.mrr_ci is not None}
        assert levels == {0.95}
        assert boots == {1000}

    def test_n_boot_zero_disables_ci(self, toy_docs):
        cases = generate_eval_set(toy_docs, num_cases=3, seed=5)
        metrics = run_retrieval(cases, BM25Retriever(toy_docs), top_k=3, n_boot=0)
        assert metrics.recall_ci == {}
        assert metrics.mrr_ci is None
        assert metrics.ndcg_ci is None
        assert metrics.hit_rate_ci is None

    def test_empty_cases_skip_ci(self, toy_docs):
        metrics = run_retrieval([], BM25Retriever(toy_docs), top_k=3)
        assert metrics.recall_ci == {}
        assert metrics.mrr_ci is None

    def test_markdown_renders_ci_and_legend(self, toy_docs):
        cases = generate_eval_set(toy_docs, num_cases=4, seed=5)
        data = run_evaluation(cases, {"bm25": BM25Retriever(toy_docs)}, top_k=3)
        md = render_markdown(data)
        assert "bootstrap 置信区间" in md
        assert "[" in md and "]" in md

    def test_markdown_without_ci_has_no_legend(self, toy_docs):
        cases = generate_eval_set(toy_docs, num_cases=4, seed=5)
        data = run_evaluation(cases, {"bm25": BM25Retriever(toy_docs)}, top_k=3, n_boot=0)
        md = render_markdown(data)
        assert "bootstrap 置信区间" not in md
        assert "[" not in md.split("## 检索指标")[1].split("##")[0]
