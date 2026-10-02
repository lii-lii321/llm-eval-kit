"""语料数据结构与纯文本文档加载。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Doc:
    """语料中的最小检索单元（一个文本块）。"""

    doc_id: str
    text: str
    source: str = ""


def split_text(text: str, *, max_chars: int = 300) -> list[str]:
    """按空行切段；超长段再按句读二次切分，尽量贴近 max_chars。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        sentences = [s for s in re.split(r"(?<=[。！？!?])|\n", para) if s.strip()]
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) > max_chars:
                chunks.append(current)
                current = sentence
            else:
                current += sentence
        if current:
            chunks.append(current)
    return chunks


def load_corpus_from_dir(path: str | Path, *, max_chars: int = 300) -> list[Doc]:
    """读取目录下所有 .txt/.md 文件并切块，doc_id 形如 `readme#000`。"""
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"语料目录不存在：{root}")
    docs: list[Doc] = []
    for file in sorted(p for p in root.iterdir() if p.suffix.lower() in {".txt", ".md"}):
        text = file.read_text(encoding="utf-8")
        for i, chunk in enumerate(split_text(text, max_chars=max_chars)):
            docs.append(Doc(doc_id=f"{file.stem}#{i:03d}", text=chunk, source=file.name))
    return docs
