"""中英混合分词：ASCII 词元 + CJK 字符二元组。

零依赖的朴素分词，满足离线检索与评测需求。中文没有真正的分词，
用字符二元组近似；生产环境建议替换为 jieba 等分词器（见 README 已知限制）。
"""

import re

# ASCII 词（字母数字连串）或连续 CJK 字符段
_TOKEN_RE = re.compile(r"[0-9A-Za-z]+|[\u3400-\u4dbf\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """把文本切成词元列表：英文取整词并小写，中文取字符二元组。"""
    tokens: list[str] = []
    for chunk in _TOKEN_RE.findall(text.lower()):
        if chunk.isascii():
            # 单个英文字母信息量低，丢弃；数字（含单数字）保留
            if len(chunk) >= 2 or chunk.isdigit():
                tokens.append(chunk)
        elif len(chunk) == 1:
            tokens.append(chunk)
        else:
            tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return tokens
