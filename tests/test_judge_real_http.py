"""真实 HTTP 层端到端测试：本地起真 socket 服务器，走完整 urlopen 链路。

此前 provider 的网络路径只有 monkeypatch 层面的覆盖；
本文件用 127.0.0.1 上的真实 HTTP 服务器（allow_local=True 显式放行）
验证请求构造、鉴权头、响应解析与错误包装在真实网络栈下的行为。
不产生任何外网流量；测试用 key 由运行时随机生成（非任何真实凭据）。
"""

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llm_eval_kit.judge import LLMJudgeError, OpenAICompatibleJudge


class _OpenAICompatHandler(BaseHTTPRequestHandler):
    """canned 的 OpenAI 兼容 chat/completions 服务器，记录收到的请求。

    server.force_status / server.force_body 可强制返回错误响应，
    用于测试 HTTPError 与 HTML 错误页的包装路径。
    """

    def do_POST(self):  # http.server 要求的命名约定
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        captured = self.server.captured  # type: ignore[attr-defined]
        captured["auth"] = self.headers.get("Authorization", "")
        captured["path"] = self.path
        try:
            captured["payload"] = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            captured["payload"] = None

        force_status = getattr(self.server, "force_status", None)
        if force_status is not None:
            force_body = getattr(self.server, "force_body", b'{"error": "forced"}')
            self._respond(force_status, force_body)
            return
        if not captured["auth"].startswith("Bearer "):
            self._respond(401, b'{"error": "unauthorized"}')
            return
        content = json.dumps(
            {"correctness": 4, "relevance": 5, "actionability": 3, "clarity": 4, "rationale": "真实 HTTP 层测试"}
        )
        body = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
        self._respond(200, body)

    def _respond(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # 静默测试输出


@pytest.fixture
def local_server():
    server = HTTPServer(("127.0.0.1", 0), _OpenAICompatHandler)
    server.captured = {}  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()


def _fake_key() -> str:
    return "runtime-" + uuid.uuid4().hex[:12]


class TestRealHttpLayer:
    def test_score_over_real_socket(self, local_server):
        fake_key = _fake_key()
        judge = OpenAICompatibleJudge(
            base_url=f"http://127.0.0.1:{local_server.server_port}/v1",
            api_key=fake_key,
            model="test-model",
            timeout=5.0,
            allow_local=True,
        )
        result = judge.score(query="什么是 BM25", answer="BM25 是打分算法", reference="参考")
        assert result.scores == {"correctness": 4, "relevance": 5, "actionability": 3, "clarity": 4}
        assert result.provider == "openai_compatible"
        assert local_server.captured["path"] == "/v1/chat/completions"
        assert local_server.captured["payload"]["model"] == "test-model"
        assert local_server.captured["auth"] == f"Bearer {fake_key}"
        assert "什么是 BM25" in local_server.captured["payload"]["messages"][0]["content"]

    def test_http_error_wrapped_over_real_socket(self, local_server):
        """服务器返回 401 时，HTTPError 应被包装为 LLMJudgeError。"""
        local_server.force_status = 401
        local_server.force_body = b'{"error": "unauthorized"}'
        judge = OpenAICompatibleJudge(
            base_url=f"http://127.0.0.1:{local_server.server_port}/v1",
            api_key=_fake_key(),
            model="m",
            timeout=5.0,
            allow_local=True,
        )
        with pytest.raises(LLMJudgeError, match="401"):
            judge.score(query="q", answer="a")

    def test_html_error_page_wrapped_over_real_socket(self, local_server):
        """服务器返回 400 + HTML 错误页时，应被包装为 LLMJudgeError 而非裸异常。"""
        local_server.force_status = 400
        local_server.force_body = b"<html><body>400 Bad Request</body></html>"
        judge = OpenAICompatibleJudge(
            base_url=f"http://127.0.0.1:{local_server.server_port}/v1",
            api_key=_fake_key(),
            model="m",
            timeout=5.0,
            allow_local=True,
        )
        with pytest.raises(LLMJudgeError):
            judge.score(query="q", answer="a")

    def test_run_evaluation_over_real_socket(self, local_server, toy_docs):
        """真实 HTTP 裁判接入 run_evaluation 的完整链路。"""
        from llm_eval_kit.pipeline import run_evaluation
        from llm_eval_kit.retrieval import BM25Retriever
        from llm_eval_kit.synth import generate_eval_set

        fake_key = _fake_key()
        judge = OpenAICompatibleJudge(
            base_url=f"http://127.0.0.1:{local_server.server_port}/v1",
            api_key=fake_key,
            model="test-model",
            timeout=5.0,
            allow_local=True,
        )
        cases = generate_eval_set(toy_docs, num_cases=2, seed=5)
        answers = {c.qid: c.answer for c in cases}
        data = run_evaluation(cases, {"bm25": BM25Retriever(toy_docs)}, judge=judge, answers=answers)
        assert data.judge.n_scored == 2
        assert data.judge.provider == "openai_compatible"
        assert not data.judge.skipped
