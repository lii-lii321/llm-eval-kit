import os

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

    def test_json_with_leading_text_extracted(self, monkeypatch):
        """真实小模型常在 JSON 前后加说明文字，应能提取出 JSON 对象。"""
        judge = OpenAICompatibleJudge(base_url="https://api.example.com/v1", api_key="k", model="m")
        content = (
            '好的，评分如下：{"correctness": 3, "relevance": 4, "actionability": 2, "clarity": 5} '
            "希望对您有帮助。"
        )
        monkeypatch.setattr(
            judge_module, "_http_post_json", lambda *a, **kw: {"choices": [{"message": {"content": content}}]}
        )
        result = judge.score(query="q", answer="a")
        assert result.scores == {"correctness": 3, "relevance": 4, "actionability": 2, "clarity": 5}

    def test_json_with_nested_braces_extracted(self, monkeypatch):
        judge = OpenAICompatibleJudge(base_url="https://api.example.com/v1", api_key="k", model="m")
        content = '结论 {"correctness": 5, "relevance": 5, "actionability": 5, "clarity": 5, "note": "含 } 花括号"}'
        monkeypatch.setattr(
            judge_module, "_http_post_json", lambda *a, **kw: {"choices": [{"message": {"content": content}}]}
        )
        result = judge.score(query="q", answer="a")
        assert result.scores["clarity"] == 5

    def test_text_without_any_json_raises(self, monkeypatch):
        judge = OpenAICompatibleJudge(base_url="https://api.example.com/v1", api_key="k", model="m")
        monkeypatch.setattr(
            judge_module,
            "_http_post_json",
            lambda *a, **kw: {"choices": [{"message": {"content": "抱歉，我无法完成评分"}}]},
        )
        with pytest.raises(LLMJudgeError, match="不是合法 JSON"):
            judge.score(query="q", answer="a")


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


class TestRealProviderSmoke:
    """真实 LLM 服务冒烟测试：默认跳过，显式设置 EVAL_REAL_LLM_SMOKE=1 才运行。

    运行条件（会产生真实网络请求与少量费用，不进常规离线套件）：
        EVAL_REAL_LLM_SMOKE=1
        EVAL_LLM_API_KEY=<真实 key>          # 只从环境变量读取，绝不入库
        EVAL_LLM_BASE_URL=<OpenAI 兼容地址>   # 例如阿里云 DashScope compatible-mode
        EVAL_LLM_MODEL=<模型名>
    """

    @pytest.mark.skipif(
        os.environ.get("EVAL_REAL_LLM_SMOKE", "") != "1" or not os.environ.get("EVAL_LLM_API_KEY"),
        reason="仅当 EVAL_REAL_LLM_SMOKE=1 且提供真实端点环境变量时运行（会产生真实网络请求）",
    )
    def test_real_provider_scores_within_range(self):
        judge = OpenAICompatibleJudge.from_env()
        result = judge.score(
            query="什么是召回率（Recall@K）？",
            answer="召回率衡量前 K 条检索结果覆盖了多少相关文档，等于命中数除以相关文档总数。",
            reference="Recall@K 等于前 K 条结果中命中的相关文档数除以相关文档总数。",
        )
        for dim in DIMENSIONS:
            assert 1 <= result.scores[dim] <= 5


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
