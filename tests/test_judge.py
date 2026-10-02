import pytest

import llm_eval_kit.judge as judge_module
from llm_eval_kit.judge import (
    DIMENSIONS,
    LLMJudgeError,
    LLMUnavailable,
    MockJudge,
    OpenAICompatibleJudge,
    _parse_judge_content,
    judge_summary,
)


class TestMockJudge:
    def test_deterministic(self):
        kwargs = {"query": "什么是 BM25", "answer": "BM25 是打分算法。", "reference": "BM25 基于词频打分。"}
        first = MockJudge().score(**kwargs)
        second = MockJudge().score(**kwargs)
        assert first.scores == second.scores
        assert first.rationale == second.rationale

    def test_scores_in_range(self):
        result = MockJudge().score(query="q", answer="答案内容", reference="参考内容")
        for dim in DIMENSIONS:
            assert 1 <= result.scores[dim] <= 5

    def test_overlap_improves_correctness(self):
        reference = "向量数据库存储文本向量并支持近似最近邻检索"
        good = "向量数据库存储文本向量，支持近似最近邻检索。"
        bad = "今天天气不错，适合出门散步。"
        judge = MockJudge()
        good_score = judge.score(query="q", answer=good, reference=reference).scores["correctness"]
        bad_score = judge.score(query="q", answer=bad, reference=reference).scores["correctness"]
        assert good_score > bad_score

    def test_actionability_rewards_steps(self):
        judge = MockJudge()
        steps = "第一步：明确目标。第二步：准备 3 个候选方案。最后：建议先小流量验证再全量。"
        vague = "这个问题需要综合考量多方面的因素影响才能得到比较全面的认识和理解。"
        assert judge.score(query="q", answer=steps, reference="").scores["actionability"] >= 4
        assert judge.score(query="q", answer=vague, reference="").scores["actionability"] <= 2

    def test_clarity_prefers_short_sentences(self):
        judge = MockJudge()
        crisp = "第一步：明确目标。第二步：准备方案。"
        rambling = (
            "这个问题其实需要从多个不同的角度出发进行综合性的考量与分析梳理，"
            "然后才有可能得到一个相对全面且兼顾各方利益诉求的结论性认识。" + "补" * 60 + "。"
        )
        assert judge.score(query="q", answer=crisp, reference="").scores["clarity"] > judge.score(
            query="q", answer=rambling, reference=""
        ).scores["clarity"]

    def test_provides_rationale(self):
        result = MockJudge().score(query="q", answer="答案", reference="参考")
        assert result.rationale
        assert result.provider == "mock"


class TestOpenAICompatibleJudgeUrlValidation:
    def test_rejects_non_http_scheme(self):
        with pytest.raises(ValueError):
            OpenAICompatibleJudge(base_url="ftp://api.example.com", api_key="k", model="m")

    def test_rejects_localhost(self):
        with pytest.raises(ValueError):
            OpenAICompatibleJudge(base_url="http://localhost:11434/v1", api_key="k", model="m")

    def test_rejects_loopback_ip(self):
        with pytest.raises(ValueError):
            OpenAICompatibleJudge(base_url="http://127.0.0.1:8000/v1", api_key="k", model="m")

    def test_rejects_private_ip(self):
        with pytest.raises(ValueError):
            OpenAICompatibleJudge(base_url="http://10.0.0.5/v1", api_key="k", model="m")
        with pytest.raises(ValueError):
            OpenAICompatibleJudge(base_url="http://192.168.1.1/v1", api_key="k", model="m")

    def test_allow_local_optin(self):
        judge = OpenAICompatibleJudge(
            base_url="http://localhost:11434/v1", api_key="k", model="m", allow_local=True
        )
        assert judge.base_url == "http://localhost:11434/v1"

    def test_public_https_url_accepted(self):
        judge = OpenAICompatibleJudge(base_url="https://api.example.com/v1", api_key="k", model="m")
        assert judge.base_url == "https://api.example.com/v1"

    def test_missing_host_raises(self):
        with pytest.raises(ValueError):
            OpenAICompatibleJudge(base_url="https:///v1", api_key="k", model="m")


class TestOpenAICompatibleJudgeBehavior:
    def test_missing_key_raises_unavailable(self):
        judge = OpenAICompatibleJudge(base_url="https://api.example.com/v1", api_key="", model="m")
        assert not judge.available
        with pytest.raises(LLMUnavailable):
            judge.score(query="q", answer="a")

    def test_from_env_reads_variables(self, monkeypatch):
        monkeypatch.setenv("EVAL_LLM_API_KEY", "test-placeholder-key")
        monkeypatch.setenv("EVAL_LLM_BASE_URL", "https://api.example.com/v1")
        monkeypatch.setenv("EVAL_LLM_MODEL", "test-model")
        judge = OpenAICompatibleJudge.from_env()
        assert judge.available
        assert judge.model == "test-model"
        assert judge.base_url == "https://api.example.com/v1"

    def test_from_env_empty_environ_defaults(self):
        judge = OpenAICompatibleJudge.from_env(environ={})
        assert not judge.available
        assert judge.model == judge_module.DEFAULT_MODEL

    def test_score_builds_request_and_parses(self, monkeypatch):
        judge = OpenAICompatibleJudge(base_url="https://api.example.com/v1", api_key="k", model="test-model")
        captured = {}

        def fake_post(url, payload, *, api_key, timeout):
            captured["url"] = url
            captured["payload"] = payload
            captured["api_key"] = api_key
            content = '{"correctness": 4, "relevance": 5, "actionability": 3, "clarity": 4, "rationale": "ok"}'
            return {"choices": [{"message": {"content": content}}]}

        monkeypatch.setattr(judge_module, "_http_post_json", fake_post)
        result = judge.score(query="问题", answer="答案", reference="参考")
        assert result.scores == {"correctness": 4, "relevance": 5, "actionability": 3, "clarity": 4}
        assert captured["url"] == "https://api.example.com/v1/chat/completions"
        assert captured["payload"]["model"] == "test-model"
        assert captured["api_key"] == "k"
        assert "问题" in captured["payload"]["messages"][0]["content"]

    def test_score_clamps_out_of_range(self, monkeypatch):
        judge = OpenAICompatibleJudge(base_url="https://api.example.com/v1", api_key="k", model="m")
        monkeypatch.setattr(
            judge_module,
            "_http_post_json",
            lambda *a, **kw: {
                "choices": [
                    {"message": {"content": '{"correctness": 9, "relevance": 0, "actionability": 3, "clarity": 4}'}}
                ]
            },
        )
        result = judge.score(query="q", answer="a")
        assert result.scores["correctness"] == 5
        assert result.scores["relevance"] == 1

    def test_missing_dimension_raises(self, monkeypatch):
        judge = OpenAICompatibleJudge(base_url="https://api.example.com/v1", api_key="k", model="m")
        monkeypatch.setattr(
            judge_module,
            "_http_post_json",
            lambda *a, **kw: {"choices": [{"message": {"content": '{"correctness": 4}'}}]},
        )
        with pytest.raises(LLMJudgeError):
            judge.score(query="q", answer="a")

    def test_non_json_output_raises(self, monkeypatch):
        judge = OpenAICompatibleJudge(base_url="https://api.example.com/v1", api_key="k", model="m")
        monkeypatch.setattr(
            judge_module,
            "_http_post_json",
            lambda *a, **kw: {"choices": [{"message": {"content": "抱歉，我无法评分"}}]},
        )
        with pytest.raises(LLMJudgeError):
            judge.score(query="q", answer="a")

    def test_code_fence_stripped(self, monkeypatch):
        judge = OpenAICompatibleJudge(base_url="https://api.example.com/v1", api_key="k", model="m")
        content = '```json\n{"correctness": 4, "relevance": 4, "actionability": 4, "clarity": 4}\n```'
        monkeypatch.setattr(
            judge_module, "_http_post_json", lambda *a, **kw: {"choices": [{"message": {"content": content}}]}
        )
        result = judge.score(query="q", answer="a")
        assert result.scores["correctness"] == 4


class TestNetworkRobustness:
    """网络与响应异常必须包装成 LLMJudgeError，保证管线可优雅跳过。"""

    class _FakeResponse:
        def __init__(self, body: bytes):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return self._body

    def _judge(self) -> OpenAICompatibleJudge:
        return OpenAICompatibleJudge(base_url="https://api.example.com/v1", api_key="k", model="m")

    def test_urlopen_error_wrapped(self, monkeypatch):
        import urllib.error

        def fake_urlopen(request, timeout):
            raise urllib.error.URLError("<urlopen error timed out>")

        monkeypatch.setattr(judge_module.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(LLMJudgeError):
            self._judge().score(query="q", answer="a")

    def test_timeout_wrapped(self, monkeypatch):
        def fake_urlopen(request, timeout):
            raise TimeoutError("timed out")

        monkeypatch.setattr(judge_module.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(LLMJudgeError):
            self._judge().score(query="q", answer="a")

    def test_http_error_wrapped(self, monkeypatch):
        import urllib.error

        def fake_urlopen(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 502, "Bad Gateway", {}, None)

        monkeypatch.setattr(judge_module.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(LLMJudgeError, match="502"):
            self._judge().score(query="q", answer="a")

    def test_html_error_page_wrapped(self, monkeypatch):
        def fake_urlopen(request, timeout):
            return TestNetworkRobustness._FakeResponse(b"<html><body>502 Bad Gateway</body></html>")

        monkeypatch.setattr(judge_module.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(LLMJudgeError, match="不是 JSON"):
            self._judge().score(query="q", answer="a")

    def test_run_evaluation_skips_on_connection_failure(self, toy_docs):
        """指向 127.0.0.1:9（端口必拒绝）的裁判应让评测优雅跳过而不是崩溃。"""
        from llm_eval_kit.pipeline import run_evaluation
        from llm_eval_kit.retrieval import BM25Retriever
        from llm_eval_kit.synth import generate_eval_set

        judge = OpenAICompatibleJudge(
            base_url="http://127.0.0.1:9/v1", api_key="test-placeholder-key", model="m",
            timeout=1.0, allow_local=True,
        )
        cases = generate_eval_set(toy_docs, num_cases=2, seed=5)
        answers = {c.qid: "某个系统答案" for c in cases}
        data = run_evaluation(cases, {"bm25": BM25Retriever(toy_docs)}, judge=judge, answers=answers)
        assert data.judge.skipped
        assert data.judge.skip_reason
        assert data.eval_size == 2  # 检索评测本身不受影响


class TestParseAndSummary:
    def test_parse_non_dict_raises(self):
        with pytest.raises(LLMJudgeError):
            _parse_judge_content("[1, 2, 3]", provider="p")

    def test_judge_summary_means(self):
        scores = [
            judge_module.JudgeScore(provider="mock", scores={d: 4 for d in DIMENSIONS}),
            judge_module.JudgeScore(provider="mock", scores={d: 2 for d in DIMENSIONS}),
        ]
        summary = judge_summary(scores)
        assert summary.n_scored == 2
        assert summary.provider == "mock"
        assert all(summary.dim_means[d] == 3.0 for d in DIMENSIONS)

    def test_judge_summary_empty(self):
        summary = judge_summary([])
        assert summary.n_scored == 0
        assert summary.dim_means == {}
