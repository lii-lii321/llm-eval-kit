"""提交前隐私检查：打印全部 distractor 首行，人工确认无个人信息。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
for p in sorted(Path("examples/corpus_math_tutor").glob("distractor_*.md")):
    text = p.read_text(encoding="utf-8").strip().replace("\n", " ")
    print(f"{p.name}: {text[:100]}")
