"""从 Math_Tutor_RAG 只读导出：金标语料 + 用户错题 → llm-eval-kit 语料与标注集。

只读保证：SQLite 以 mode=ro URI 打开；金标语料直接读其测试文件常量（ast.literal_eval
受限解析，不执行任何代码）；本脚本不写 Math_Tutor_RAG 仓库的任何文件，产物全部落在
llm-eval-kit 工作区。
"""
from __future__ import annotations

import ast
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

MM_ROOT = Path(r"D:\My_Project\Math_Tutor_RAG")
OUT_DIR = Path(__file__).resolve().parent.parent / "examples"
CORPUS_DIR = OUT_DIR / "corpus_math_tutor"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# 1) 金标语料（12 条：content, tags, kw, sem）——字面量受限解析，不执行任何代码
src = (MM_ROOT / "tests" / "test_retrieval_quality.py").read_text(encoding="utf-8")
m = re.search(r"_GOLDEN: list\[tuple\[str, list\[str\], str, str\]\] = (\[.*?\n\])", src, re.S)
assert m, "未找到 _GOLDEN 定义"
golden = ast.literal_eval(m.group(1))
print(f"golden entries: {len(golden)}")

# 2) 只读导出 SQLite 中的真实错题（所有用户），作为语料干扰项
con = sqlite3.connect(f"file:{(MM_ROOT / 'data' / 'math_tutor.db').as_posix()}?mode=ro", uri=True)
cur = con.cursor()
tables = [r[0] for r in cur.execute("select name from sqlite_master where type='table'")]
print("tables:", tables)
rows = []
if "questions" in tables:
    cols = [r[1] for r in cur.execute("pragma table_info(questions)")]
    print("question cols:", cols)
    content_col = "content_markdown" if "content_markdown" in cols else "content"
    for row in cur.execute(
        f"select id, {content_col} from questions where {content_col} is not null"  # noqa: S608 - 列名来自 pragma 白名单分支
    ):
        rows.append(row)
con.close()
print(f"questions in db: {len(rows)}")

# 3) 落盘语料：金标 12 条命名 golden_XX.md（可被查询引用），其余为 distractor_NN.md
golden_contents = [str(content).strip() for content, _tags, _kw, _sem in golden]
golden_files = []
for i, (content, tags, _kw, _sem) in enumerate(golden, 1):
    path = CORPUS_DIR / f"golden_{i:02d}.md"
    path.write_text(str(content), encoding="utf-8")
    golden_files.append((path.name, str(content), list(tags)))

distractors = 0
for _qid, content in rows:
    text = str(content).strip()
    if len(text) < 15 or text in golden_contents:
        continue
    distractors += 1
    (CORPUS_DIR / f"distractor_{distractors:03d}.md").write_text(text, encoding="utf-8")
print(f"corpus: {len(golden)} golden + {distractors} distractors")

# 4) 标注集：12 条语义问法查询 + 12 条关键词问法查询，相关文档 = 对应 golden_XX.md
lines = [
    json.dumps(
        {
            "_meta": {
                "corpus": "Math_Tutor_RAG 智能错题本真实语料：12 条金标考点条目（tests/test_retrieval_quality.py _GOLDEN）+ 从其 SQLite 只读导出的真实错题作干扰项（doc_id=文件名#000，单文件单条）",
                "annotated_at": "2026-10-25",
                "annotator": "llm-eval-kit 维护者（金标查询来自 Math_Tutor_RAG 检索质量门禁语料；语料经只读导出）",
                "version": "0.1.0",
                "provenance": "real user data, read-only export from Math_Tutor_RAG (SQLite mode=ro + test constants); queries are its golden student-style paraphrases and keyword probes",
            }
        },
        ensure_ascii=False,
    )
]
for i, (fname, _content, tags) in enumerate(golden_files, 1):
    _c, _t, _kw, sem = golden[i - 1]
    lines.append(
        json.dumps(
            {
                "query_id": f"sem_{i:02d}",
                "query": str(sem),
                "relevant_doc_ids": [f"{Path(fname).stem}#000"],
                "notes": f"金标语料 G{i:02d}，学生口语句（语义改写组口径），tags={tags}",
            },
            ensure_ascii=False,
        )
    )
for i, (fname, _content, _tags) in enumerate(golden_files, 1):
    _c, _t, kw, _sem = golden[i - 1]
    lines.append(
        json.dumps(
            {
                "query_id": f"kw_{i:02d}",
                "query": str(kw),
                "relevant_doc_ids": [f"{Path(fname).stem}#000"],
                "notes": f"金标语料 G{i:02d}，关键词查询（目标题独有词，词面口径）",
            },
            ensure_ascii=False,
        )
    )
out_jsonl = OUT_DIR / "dataset_math_tutor.jsonl"
out_jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"dataset written: {out_jsonl} ({len(lines) - 1} queries)")

# 5) 分组拆分文件：语义问法组 / 关键词问法组（口径分开报告，与 Math_Tutor_RAG 内部评测一致）
sem_lines = [lines[0]] + [l for l in lines[1:] if '"sem_' in l]
kw_lines = [lines[0]] + [l for l in lines[1:] if '"kw_' in l]
(OUT_DIR / "dataset_math_tutor_semantic.jsonl").write_text(
    "\n".join(sem_lines) + "\n", encoding="utf-8"
)
(OUT_DIR / "dataset_math_tutor_keyword.jsonl").write_text(
    "\n".join(kw_lines) + "\n", encoding="utf-8"
)
print("split datasets written: _semantic.jsonl / _keyword.jsonl")
