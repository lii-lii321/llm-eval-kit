"""外部检索器适配层：让任意检索系统以最小代价接入评测管线。

核心是 Retriever 协议（duck typing，不要求继承）：只要实现
``retrieve(query, top_k) -> list[Hit]`` 就能被评测管线接受。
结果宽容归一化：Hit、(doc_id, score) 元组、裸 doc_id 字符串、
以及任何带 ``doc_id`` 属性的对象（如本库的 RetrievalResult）。

接入路径：
- 最短路径：``CallableRetriever(lambda query, top_k: my_search(query, k=top_k))``；
- 自定义类直接实现 ``retrieve`` 方法，用 ``ensure_retriever`` 自检；
- ``run_evaluation`` / ``run_retrieval`` 的 retrievers 字典值既可以传
  内置 ``BaseRetriever``，也可以传任何满足协议的对象（内部自动包一层适配）。

完整指南见 docs/ADAPTERS.md。
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .retrieval import BaseRetriever, RetrievalResult

DOCS_GUIDE = "docs/ADAPTERS.md"


class RetrieverProtocolError(TypeError):
    """对象不满足 Retriever 协议时抛出，错误信息包含缺失项、期望签名与文档指引。"""


@dataclass(frozen=True)
class Hit:
    """一条检索结果：文档 ID 与可选得分（仅用于排序，不保证绝对数值含义）。"""

    doc_id: str
    score: float | None = None


@runtime_checkable
class Retriever(Protocol):
    """外部检索器协议：给定查询返回按相关性降序的 top-K 结果。

    与内置 BaseRetriever 的区别：不要求提供 doc_texts，也不要求继承任何类；
    badcase 归因需要 doc_texts 时见 docs/ADAPTERS.md 的说明。
    """

    def retrieve(self, query: str, top_k: int = 10) -> list[Hit]:
        """返回 top-K 检索结果。"""
        ...


def _protocol_error(obj: Any, problem: str) -> RetrieverProtocolError:
    return RetrieverProtocolError(
        f"对象 {type(obj).__module__}.{type(obj).__qualname__} 不满足 llm_eval_kit.Retriever 协议：{problem}\n"
        "期望接口：retrieve(query: str, top_k: int) -> list[Hit]，每个结果可以是 "
        "Hit(doc_id, score)、(doc_id, score) 元组、裸 doc_id 字符串，或任何带 doc_id 属性的对象。\n"
        '最快接入方式：CallableRetriever(lambda query, top_k: my_search(query, k=top_k), name="my-retriever")\n'
        f"完整接入指南见 {DOCS_GUIDE}。"
    )


def ensure_retriever(retriever: Any) -> Retriever:
    """duck-typing 校验对象是否满足 Retriever 协议，满足则原样返回。

    校验两件事：存在可调用的 ``retrieve`` 方法；其签名能接受 (query, top_k)
    两个参数（top_k 位置或关键字传递均可）。不满足时抛出 RetrieverProtocolError，
    错误信息说明缺什么、期望签名是什么、去哪里看接入指南。
    """
    retrieve = getattr(retriever, "retrieve", None)
    if not callable(retrieve):
        raise _protocol_error(retriever, "缺少可调用的 retrieve 方法")
    try:
        signature = inspect.signature(retrieve)
    except (TypeError, ValueError):
        return retriever  # 无法内省（内置对象等），按 duck typing 信任
    try:
        signature.bind("查询", 5)
        return retriever
    except TypeError:
        pass
    try:
        signature.bind("查询", top_k=5)
    except TypeError as exc:
        raise _protocol_error(
            retriever,
            f"retrieve 方法签名不兼容：{signature}。期望能以 (query, top_k) 两个参数调用",
        ) from exc
    return retriever


def normalize_hit(item: Any) -> Hit:
    """把单个检索结果归一化为 Hit。

    宽容输入：Hit、(doc_id,)、(doc_id, score)、裸 doc_id 字符串，
    以及任何带 doc_id 属性的对象（RetrievalResult、ORM 行等）。
    无法识别的形状抛出带示例的 TypeError。
    """
    if isinstance(item, Hit):
        return item
    if isinstance(item, str):
        return Hit(doc_id=item)
    doc_id = getattr(item, "doc_id", None)
    if doc_id is not None:
        score = getattr(item, "score", None)
        return Hit(doc_id=str(doc_id), score=None if score is None else float(score))
    if isinstance(item, (tuple, list)):
        if len(item) == 1:
            return Hit(doc_id=str(item[0]))
        if len(item) == 2:
            return Hit(doc_id=str(item[0]), score=float(item[1]))
        raise TypeError(
            f"检索结果元组只支持 (doc_id,) 或 (doc_id, score)，收到长度 {len(item)}：{item!r}"
        )
    raise TypeError(
        f"无法把检索结果归一化为 Hit：{item!r}（类型 {type(item).__name__}）；"
        "支持 Hit、(doc_id, score) 元组、裸 doc_id 字符串，或带 doc_id 属性的对象"
    )


def normalize_hits(raw: Iterable[Any]) -> list[Hit]:
    """把检索器返回的任意可迭代结果逐条归一化为 Hit 列表。"""
    return [normalize_hit(item) for item in raw]


class CallableRetriever:
    """把 ``callable(query, top_k) -> 结果列表`` 包装成 Retriever 协议实现。

    结果列表的每个元素会经 normalize_hit 宽容归一化，返回类型统一为 list[Hit]。
    """

    def __init__(self, fn: Callable[[str, int], Any], *, name: str = "callable"):
        if not callable(fn):
            raise TypeError(f"CallableRetriever 需要可调用对象，收到 {type(fn).__name__}")
        self._fn = fn
        self.name = name

    def retrieve(self, query: str, top_k: int = 10) -> list[Hit]:
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        return normalize_hits(self._fn(query, top_k))


class RetrieverAdapter(BaseRetriever):
    """把任意 Retriever 协议实现包成评测管线直接可用的 BaseRetriever。

    doc_texts（badcase 归因与 judge 上下文用）按以下优先级获取：
    显式传入 > 被包装对象同名属性（duck typing）> 空字典。
    注意：doc_texts 为空时，归因会把所有未命中归为"语料缺失"，见已知限制。
    """

    def __init__(
        self,
        retriever: Any,
        *,
        name: str | None = None,
        doc_texts: dict[str, str] | None = None,
    ):
        ensure_retriever(retriever)
        super().__init__([])
        self._inner = retriever
        self.name = name or str(getattr(retriever, "name", "external"))
        texts = doc_texts
        if texts is None:
            ducked = getattr(retriever, "doc_texts", None)
            if isinstance(ducked, dict):
                texts = ducked
        self._texts = {str(k): str(v) for k, v in (texts or {}).items()}
        try:
            inspect.signature(retriever.retrieve).bind("查询", 5)
            self._top_k_positional = True
        except TypeError:
            self._top_k_positional = False
        except ValueError:
            self._top_k_positional = True

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        self._check_top_k(top_k)
        if self._top_k_positional:
            raw = self._inner.retrieve(query, top_k)
        else:
            raw = self._inner.retrieve(query, top_k=top_k)
        return [
            RetrievalResult(doc_id=hit.doc_id, score=0.0 if hit.score is None else hit.score)
            for hit in normalize_hits(raw)
        ]


def as_eval_retriever(retriever: BaseRetriever | Retriever) -> BaseRetriever:
    """把内置检索器或任意协议实现统一成管线可用的 BaseRetriever。

    内置 BaseRetriever 原样返回；其余对象经协议校验后包一层 RetrieverAdapter。
    """
    if isinstance(retriever, BaseRetriever):
        return retriever
    return RetrieverAdapter(retriever)
