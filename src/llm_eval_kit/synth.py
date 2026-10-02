"""规则式合成评测集生成器：从纯文本文档构造 query-doc-answer 三元组。

不依赖任何外部 LLM：关键词提取基于词频 × 逆文档频率，查询模板固定，
全程可用 seed 复现。这是最朴素的合成策略，产出查询偏词面化，
质量天花板明显低于 LLM 合成（见 README 已知限制）。

安全说明：这里的 random.Random(seed) 是刻意为之——合成评测集必须
同一 seed 得到同一批用例（测试有断言），不属于加密/安全用途，
不能也不应替换为 secrets 等密码学随机源。
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field

from .corpus import Doc

# 常见虚词/单字，用于把长中文串切分成更短的候选短语
STOP_CHARS = "的了是在和与并或把被对从到能可要会中也不就而及等上都为以自若则很还比各个种将"

_SIMPLE_TEMPLATES = (
    "什么是{kw}？",
    "介绍一下{kw}",
    "{kw}的作用是什么？",
    "如何理解{kw}？",
)
_PAIR_TEMPLATES = (
    "{kw1}和{kw2}有什么关系？",
    "{kw1}与{kw2}的区别是什么？",
)

_ASCII_WORD_RE = re.compile(r"[0-9A-Za-z]{2,}")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_STOP_SPLIT_RE = re.compile("[" + re.escape(STOP_CHARS) + "]")


@dataclass
class EvalCase:
    """一条评测用例：查询 + 相关文档 + 参考答案。"""

    qid: str
    query: str
    relevant_ids: list[str]
    answer: str = ""
    meta: dict = field(default_factory=dict)


def extract_phrases(text: str, *, max_len: int = 6) -> list[str]:
    """提取候选关键词：ASCII 词 + 中文串（虚词切分、超长截断），保留重复出现与出现顺序。

    调用方用 Counter 统计词频、用 set 统计文档频率；
    纯数字词元（如“10%”的 10）信息量低，直接丢弃。
    对无分词工具的纯文本这是近似做法：长中文串截断可能产生不自然短语。
    """
    phrases: list[str] = []
    for word in _ASCII_WORD_RE.findall(text.lower()):
        if not word.isdigit():
            phrases.append(word)
    for run in _CJK_RUN_RE.findall(text):
        for seg in _STOP_SPLIT_RE.split(run):
            if seg:
                phrases.append(seg if len(seg) <= max_len else seg[:max_len])
    return phrases


def _score_phrases(docs: list[Doc]) -> tuple[dict[str, Counter], Counter]:
    """统计每个文档的短语词频与全语料文档频率。"""
    doc_tf: dict[str, Counter] = {}
    df: Counter = Counter()
    for doc in docs:
        phrases = extract_phrases(doc.text)
        tf = Counter(phrases)
        doc_tf[doc.doc_id] = tf
        df.update(set(phrases))
    return doc_tf, df


def _rank_phrases(tf: Counter, df: Counter, n_docs: int) -> list[str]:
    """按 tf * idf 降序排短语；同分按文档内首次出现顺序（主题词通常开篇即现）。"""

    def score(phrase: str) -> float:
        return tf[phrase] * math.log(1 + n_docs / max(df[phrase], 1))

    # Counter 迭代顺序即短语首次出现顺序，sorted 稳定排序保留该顺序
    return sorted(tf, key=lambda p: -score(p))


def _reference_answer(text: str, keyword: str) -> str:
    """参考答案：优先取包含关键词的前两句，否则取开头两句。"""
    sentences = [s.strip() for s in re.split(r"(?<=[。！？!?])", text) if s.strip()]
    hits = [s for s in sentences if keyword.lower() in s.lower()]
    chosen = hits[:2] if hits else sentences[:2]
    return "".join(chosen).strip() or text.strip()


def generate_eval_set(
    docs: list[Doc],
    *,
    num_cases: int = 20,
    seed: int = 42,
    paraphrase_ratio: float = 0.0,
    synonym_map: dict[str, str] | None = None,
) -> list[EvalCase]:
    """从语料合成评测集。

    - 每个文档挑 tf*idf 最高的短语构造一条查询；
    - paraphrase_ratio 比例的查询会按 synonym_map 做词面替换，
      用来构造“零词面重叠”的困难样本（考察检索器与 badcase 归因）；
    - 相同查询只保留一条。
    """
    if not docs:
        raise ValueError("语料为空，无法生成评测集")
    if num_cases <= 0:
        raise ValueError("num_cases 必须为正整数")
    rng = random.Random(seed)
    doc_tf, df = _score_phrases(docs)
    n_docs = len(docs)
    ranked_by_doc = {d.doc_id: _rank_phrases(doc_tf[d.doc_id], df, n_docs) for d in docs}

    shuffled = list(docs)
    rng.shuffle(shuffled)
    synonyms = synonym_map or {}
    cases: list[EvalCase] = []
    used_queries: set[str] = set()
    for doc in shuffled:
        if len(cases) >= num_cases:
            break
        ranked = [p for p in ranked_by_doc[doc.doc_id] if len(p) >= 2]
        if not ranked:
            continue
        kw1 = ranked[0]
        kw2 = ranked[1] if len(ranked) > 1 else None
        if kw2 and rng.random() < 0.3:
            query = rng.choice(_PAIR_TEMPLATES).format(kw1=kw1, kw2=kw2)
        else:
            query = rng.choice(_SIMPLE_TEMPLATES).format(kw=kw1)
        paraphrased = False
        if paraphrase_ratio > 0 and synonyms and rng.random() < paraphrase_ratio:
            for key, value in synonyms.items():
                if key in query:
                    query = query.replace(key, value)
                    paraphrased = True
        if query in used_queries:
            continue
        used_queries.add(query)
        cases.append(
            EvalCase(
                qid=f"q{len(cases):03d}",
                query=query,
                relevant_ids=[doc.doc_id],
                answer=_reference_answer(doc.text, kw1),
                meta={"paraphrased": paraphrased, "keywords": [kw1, kw2] if kw2 else [kw1]},
            )
        )
    return cases
