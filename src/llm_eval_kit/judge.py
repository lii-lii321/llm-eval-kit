"""LLM-as-judge 四维评分：provider 可插拔。

四个维度：correctness 正确性 / relevance 相关性 /
actionability 可操作性 / clarity 清晰度，各 1-5 整数分。

- MockJudge：确定性规则评分（词面重叠等启发式），离线可跑，供测试与 demo；
- OpenAICompatibleJudge：读环境变量调用 OpenAI 兼容接口，
  未配置 key 时抛 LLMUnavailable，由管线捕获后优雅跳过；
- SelfConsistencyJudge：包装任意 judge 采样 n 次按维度取多数票，压单次评分噪声。

凭据只从环境变量读取；代码、示例与测试中不出现任何真实密钥。
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from .tokenize import tokenize

DIMENSIONS = ("correctness", "relevance", "actionability", "clarity")
DIMENSIONS_ZH = {
    "correctness": "正确性",
    "relevance": "相关性",
    "actionability": "可操作性",
    "clarity": "清晰度",
}


class LLMJudgeError(Exception):
    """LLM 评分通用错误（网络、响应解析失败等）。"""


class LLMUnavailable(LLMJudgeError):
    """LLM 评分服务不可用（典型：未配置 API key），管线应优雅跳过。"""


@dataclass
class JudgeScore:
    """单条样本的裁判评分。"""

    provider: str
    scores: dict[str, int]
    rationale: str = ""


class JudgeProvider(Protocol):
    """裁判 provider 协议：实现 name 与 score 即可接入管线。"""

    name: str

    def score(self, *, query: str, answer: str, reference: str = "", context: str = "") -> JudgeScore: ...


def _overlap_ratio(answer_tokens: list[str], baseline_tokens: list[str]) -> float:
    baseline = set(baseline_tokens)
    if not baseline:
        return 0.0
    return len(set(answer_tokens) & baseline) / len(baseline)


def _ratio_to_score(ratio: float) -> int:
    if ratio >= 0.6:
        return 5
    if ratio >= 0.4:
        return 4
    if ratio >= 0.25:
        return 3
    if ratio >= 0.1:
        return 2
    return 1


_ACTION_WORDS = ("步骤", "首先", "然后", "接着", "最后", "建议", "第一步", "第二步", "方案", "可以尝试")
_BULLET_RE = re.compile(r"(^|\n)\s*([-•*]|\d+[.、])")
_DIGIT_RE = re.compile(r"\d")


def _actionability_score(answer: str) -> int:
    """可操作性：数字、步骤词、列表标记越多分越高（1-5）。"""
    markers = 0
    if _DIGIT_RE.search(answer):
        markers += 1
    markers += sum(1 for word in _ACTION_WORDS if word in answer)
    if _BULLET_RE.search(answer):
        markers += 1
    return max(1, min(5, 1 + markers))


def _clarity_score(answer: str) -> int:
    """清晰度：按平均句长评分，句子越冗长分数越低（1-5）。"""
    sentences = [s for s in re.split(r"[。！？!?\n]", answer) if s.strip()]
    if not sentences:
        return 1
    avg_len = sum(len(s.strip()) for s in sentences) / len(sentences)
    if avg_len <= 25:
        return 5
    if avg_len <= 45:
        return 4
    if avg_len <= 70:
        return 3
    if avg_len <= 100:
        return 2
    return 1


class MockJudge:
    """确定性规则裁判：完全离线、可复现，不代表真实模型评分分布。"""

    name = "mock"

    def score(self, *, query: str, answer: str, reference: str = "", context: str = "") -> JudgeScore:
        answer_tokens = tokenize(answer)
        ref_ratio = _overlap_ratio(answer_tokens, tokenize(reference))
        ctx_ratio = _overlap_ratio(answer_tokens, tokenize(query) + tokenize(context))
        scores = {
            "correctness": _ratio_to_score(ref_ratio),
            "relevance": _ratio_to_score(ctx_ratio),
            "actionability": _actionability_score(answer),
            "clarity": _clarity_score(answer),
        }
        rationale = f"词面重叠：与参考答案 {ref_ratio:.2f}，与问题/上下文 {ctx_ratio:.2f}（确定性规则评分）"
        return JudgeScore(provider=self.name, scores=scores, rationale=rationale)


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

_JUDGE_PROMPT = """你是一名严格的中文 LLM 评测裁判。请依据以下标准对待评答案打分（1-5 整数）：
- correctness 正确性：与参考答案的事实一致程度；
- relevance 相关性：是否紧扣用户问题；
- actionability 可操作性：是否给出可执行的建议或步骤；
- clarity 清晰度：表达是否简洁清楚。
只输出 JSON，不要输出其他内容：
{{"correctness": 1, "relevance": 1, "actionability": 1, "clarity": 1, "rationale": "一句话理由"}}

用户问题：{query}
参考答案：{reference}
检索上下文：{context}
待评答案：{answer}"""


def _validate_base_url(url: str, *, allow_local: bool) -> str:
    """校验 base_url：仅 http/https，默认拒绝 localhost/回环/私有/保留地址。

    自建模型服务（Ollama 等）需显式设置环境变量 EVAL_LLM_ALLOW_LOCAL=1 放行。
    仅校验 URL 字面量，不做 DNS 解析后的二次校验（见 README 已知限制）。
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"base_url 仅支持 http/https，收到：{url!r}")
    host = parsed.hostname or ""
    if not host:
        raise ValueError(f"base_url 缺少主机名：{url!r}")
    if allow_local:
        return url
    lowered = host.lower()
    if lowered == "localhost" or lowered.endswith(".localhost") or lowered.endswith(".local"):
        raise ValueError(f"拒绝本地地址（自建服务请设置 EVAL_LLM_ALLOW_LOCAL=1）：{url!r}")
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return url
    if (
        addr.is_loopback
        or addr.is_private
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    ):
        raise ValueError(f"拒绝回环/私有/保留地址（自建服务请设置 EVAL_LLM_ALLOW_LOCAL=1）：{url!r}")
    return url


def _http_post_json(url: str, payload: dict, *, api_key: str, timeout: float) -> dict:
    """向 OpenAI 兼容接口发 POST；模块级函数，便于测试替换。

    所有网络与响应解析失败（URLError/超时/HTTP 错误码/响应非 JSON）
    都包装成 LLMJudgeError 抛出，保证上层“优雅跳过”承诺成立。
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    # base_url 已通过 _validate_base_url 校验协议与目标地址
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # 尽力读取错误响应体（如 OpenAI 的 invalid_api_key），帮助排障 provider 配置
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:200].strip()
        except Exception:  # 诊断读取是尽力而为，任何失败都不掩盖原始错误
            detail = ""
        suffix = f"：{detail}" if detail else ""
        raise LLMJudgeError(f"LLM 服务返回 HTTP {exc.code}{suffix}") from exc
    except OSError as exc:  # URLError/socket 超时/连接拒绝均为 OSError 子类
        raise LLMJudgeError(f"LLM 服务连接失败：{exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMJudgeError(f"LLM 服务响应不是 JSON（可能是网关错误页）：{raw[:200]!r}") from exc


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")


def _extract_json_object(text: str) -> dict | None:
    """从混有说明文字的输出中提取第一个平衡的 JSON 对象；找不到返回 None。

    真实小模型常无视“只输出 JSON”的指令，在 JSON 前后加说明文字，
    该回退让 provider 对这类输出保持健壮。
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


def _parse_judge_content(content: str, *, provider: str) -> JudgeScore:
    text = content.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        extracted = _extract_json_object(text)
        if extracted is None:
            raise LLMJudgeError(f"裁判输出不是合法 JSON：{content[:200]!r}") from None
        raw = extracted
    if not isinstance(raw, dict):
        raise LLMJudgeError(f"裁判输出应为 JSON 对象，收到：{type(raw).__name__}")
    scores: dict[str, int] = {}
    for dim in DIMENSIONS:
        if dim not in raw:
            raise LLMJudgeError(f"裁判输出缺少维度 {dim}")
        try:
            value = int(round(float(raw[dim])))
        except (TypeError, ValueError) as exc:
            raise LLMJudgeError(f"维度 {dim} 的分值不是数字：{raw[dim]!r}") from exc
        scores[dim] = max(1, min(5, value))
    return JudgeScore(provider=provider, scores=scores, rationale=str(raw.get("rationale", "")))


class OpenAICompatibleJudge:
    """OpenAI 兼容 chat/completions 裁判。

    环境变量（全部可选，均有默认值）：
    - EVAL_LLM_API_KEY：API key，未设置时评分跳过；
    - EVAL_LLM_BASE_URL：默认 https://api.openai.com/v1；
    - EVAL_LLM_MODEL：默认 gpt-4o-mini；
    - EVAL_LLM_TIMEOUT：请求超时秒数，默认 30；
    - EVAL_LLM_ALLOW_LOCAL：置 1 放行 localhost/内网地址（自建模型服务）；
    - EVAL_LLM_TEMPERATURE：采样温度，默认 0（确定性）；配 SelfConsistencyJudge
      建议设 0.5-0.9，让多次采样产生差异，多数票才有降噪空间。
    """

    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        allow_local: bool = False,
        temperature: float = 0.0,
    ):
        self.base_url = _validate_base_url(base_url, allow_local=allow_local)
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"temperature 需在 [0, 2] 区间，收到：{temperature}")
        self.temperature = temperature

    @classmethod
    def from_env(cls, environ: dict | None = None) -> OpenAICompatibleJudge:
        env = os.environ if environ is None else environ
        return cls(
            base_url=env.get("EVAL_LLM_BASE_URL", DEFAULT_BASE_URL),
            api_key=env.get("EVAL_LLM_API_KEY", ""),
            model=env.get("EVAL_LLM_MODEL", DEFAULT_MODEL),
            timeout=float(env.get("EVAL_LLM_TIMEOUT", "30")),
            allow_local=env.get("EVAL_LLM_ALLOW_LOCAL", "") in ("1", "true", "yes"),
            temperature=float(env.get("EVAL_LLM_TEMPERATURE", "0")),
        )

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def score(self, *, query: str, answer: str, reference: str = "", context: str = "") -> JudgeScore:
        if not self.api_key:
            raise LLMUnavailable("EVAL_LLM_API_KEY 未设置，LLM 评分跳过（离线场景可改用 MockJudge）")
        prompt = _JUDGE_PROMPT.format(query=query, reference=reference, context=context, answer=answer)
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        data = _http_post_json(url, payload, api_key=self.api_key, timeout=self.timeout)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMJudgeError(f"响应缺少 choices[0].message.content：{str(data)[:200]}") from exc
        return _parse_judge_content(content, provider=self.name)


@dataclass
class JudgeSummary:
    """裁判评分聚合结果，供报告使用。"""

    provider: str = ""
    n_scored: int = 0
    skipped: bool = False
    skip_reason: str = ""
    dim_means: dict[str, float] = field(default_factory=dict)


def judge_summary(scores: list[JudgeScore]) -> JudgeSummary:
    """按维度求平均分；空列表返回未评分摘要。"""
    if not scores:
        return JudgeSummary()
    means: dict[str, float] = {}
    for dim in DIMENSIONS:
        values = [s.scores[dim] for s in scores if dim in s.scores]
        means[dim] = round(sum(values) / len(values), 2) if values else 0.0
    return JudgeSummary(provider=scores[0].provider, n_scored=len(scores), dim_means=means)


def _majority_vote(values: list[int]) -> int:
    """多数票；并列时取最接近样本均值的分值，仍并列取较小者（确定性、保守）。"""
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    top = max(counts.values())
    candidates = sorted(value for value, count in counts.items() if count == top)
    if len(candidates) == 1:
        return candidates[0]
    mean = sum(values) / len(values)
    return min(candidates, key=lambda value: (abs(value - mean), value))


class SelfConsistencyJudge:
    """自一致裁判：同一输入采样 n 次，各维度取多数票，压单次评分噪声。

    - provider 无关：包装任意 JudgeProvider（MockJudge / OpenAICompatibleJudge 均可），
      实现 JudgeProvider 协议，可直接传给 run_evaluation；
    - 容错语义：只要有效票数仍能构成对 n 的严格多数（>= n//2 + 1）就继续，
      超出容错时抛出第一次失败的原始异常——LLMUnavailable 等"管线跳过"语义因此保留；
    - 并列处理：取最接近样本均值的分值，仍并列取较小者（确定性）；
    - 用法提示：真实模型请把内层 judge 的 temperature 设为 0.5-0.9（见
      OpenAICompatibleJudge 的 EVAL_LLM_TEMPERATURE），采样高度一致时多数票收益有限。

    >>> judge = SelfConsistencyJudge(OpenAICompatibleJudge.from_env(), n=5)
    """

    def __init__(self, inner: JudgeProvider, *, n: int = 5):
        if n < 1:
            raise ValueError(f"采样次数 n 必须 >= 1，收到：{n}")
        self.inner = inner
        self.n = n

    @property
    def name(self) -> str:
        return f"{self.inner.name}_sc{self.n}"

    def score(self, *, query: str, answer: str, reference: str = "", context: str = "") -> JudgeScore:
        needed = self.n // 2 + 1  # 对 n 的严格多数
        votes: list[JudgeScore] = []
        first_error: LLMJudgeError | None = None
        for attempt in range(1, self.n + 1):
            try:
                votes.append(self.inner.score(query=query, answer=answer, reference=reference, context=context))
            except LLMJudgeError as exc:
                if first_error is None:
                    first_error = exc
                if len(votes) + (self.n - attempt) < needed:
                    raise first_error from exc
        scores = {dim: _majority_vote([v.scores[dim] for v in votes]) for dim in DIMENSIONS}
        rationale = next((v.rationale for v in votes if v.rationale), "")
        prefix = f"自一致采样 n={self.n}（有效票 {len(votes)}），按维度多数票；"
        return JudgeScore(provider=self.name, scores=scores, rationale=prefix + rationale)
