"""命令行入口：`python -m llm_eval_kit.cli demo` 或安装后 `llm-eval-kit demo`。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .demo import print_summary, run_demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-eval-kit", description="RAG/LLM 评测工具库：跑内置端到端 demo")
    parser.add_argument("command", nargs="?", default="demo", choices=["demo"], help="子命令（默认 demo）")
    parser.add_argument("--out", default="reports", help="报告输出目录（默认 ./reports）")
    parser.add_argument("--seed", type=int, default=42, help="合成评测集随机种子（默认 42）")
    parser.add_argument("--num-cases", type=int, default=24, help="评测集规模（默认 24）")
    parser.add_argument("--paraphrase-ratio", type=float, default=1.0, help="查询改写比例（默认 1.0，对含同义词键的查询全部尝试改写）")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    data = run_demo(
        out_dir=Path(args.out),
        seed=args.seed,
        num_cases=args.num_cases,
        paraphrase_ratio=args.paraphrase_ratio,
    )
    print_summary(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
