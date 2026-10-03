"""命令行入口：`python -m llm_eval_kit.cli demo` 或安装后 `llm-eval-kit demo`。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dataset import DatasetError, load_dataset
from .demo import build_demo_corpus, print_summary, run_demo
from .gates import evaluate_gates, metric_value, parse_fail_under


def _positive_int(value: str) -> int:
    """argparse 类型：正整数（--num-cases 用）。"""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"需要正整数，收到：{value!r}") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"需要正整数，收到：{value!r}")
    return parsed


def _ratio(value: str) -> float:
    """argparse 类型：[0, 1] 内的比例（--paraphrase-ratio 用）。"""
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"需要 [0, 1] 内的数字，收到：{value!r}") from None
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(f"改写比例取值范围 [0, 1]，收到：{value!r}")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-eval-kit", description="RAG/LLM 评测工具库：跑内置端到端 demo")
    parser.add_argument("command", nargs="?", default="demo", choices=["demo"], help="子命令（默认 demo）")
    parser.add_argument("--out", default="reports", help="报告输出目录（默认 ./reports）")
    parser.add_argument("--seed", type=int, default=42, help="合成评测集随机种子（默认 42）")
    parser.add_argument("--num-cases", type=_positive_int, default=24, help="评测集规模（正整数，默认 24）")
    parser.add_argument(
        "--paraphrase-ratio", type=_ratio, default=1.0,
        help="查询改写比例，取值 [0, 1]（默认 1.0，对含同义词键的查询全部尝试改写）",
    )
    parser.add_argument(
        "--dataset", default=None, metavar="PATH.jsonl",
        help=(
            "JSONL 评测集路径（首个非空行可为 {\"_meta\": {...}}）。提供后跳过合成评测集，"
            "改为从文件加载查询批量评测；语料沿用内置玩具语料，与 --num-cases/--paraphrase-ratio 互斥"
        ),
    )
    parser.add_argument(
        "--fail-under", action="append", default=[], metavar="指标=阈值",
        help=(
            "评测回归门禁：主管线指标低于阈值则退出码 1。可多次传入，"
            "如 --fail-under recall_at_5=0.85 --fail-under mrr=0.8"
        ),
    )
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    try:
        gates = [parse_fail_under(spec) for spec in args.fail_under]
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2

    cases = None
    notes_prefix: list[str] = []
    if args.dataset:
        if args.num_cases != parser.get_default("num_cases") or args.paraphrase_ratio != parser.get_default(
            "paraphrase_ratio"
        ):
            print("参数错误：--num-cases / --paraphrase-ratio 仅用于合成评测集，与 --dataset 互斥", file=sys.stderr)
            return 2
        try:
            dataset = load_dataset(args.dataset)
        except DatasetError as exc:
            print(f"评测集加载失败：{exc}", file=sys.stderr)
            return 2
        cases = dataset.to_eval_cases()
        stats = dataset.stats()
        notes_prefix.append(f"评测集：{dataset.path}")
        notes_prefix.append(f"评测集统计：{stats.describe()}")
        if dataset.meta:
            notes_prefix.append(f"评测集 _meta：{json.dumps(dataset.meta, ensure_ascii=False, sort_keys=True)}")
        print(f"已加载评测集：{dataset.path}")
        print(f"  {stats.describe()}")
        unknown = dataset.unknown_doc_ids(doc.doc_id for doc in build_demo_corpus())
        if unknown:
            total = sum(len(ids) for ids in unknown.values())
            sample = "；".join(f"{qid}: {ids}" for qid, ids in list(unknown.items())[:3])
            print(
                f"警告：{len(unknown)} 条查询引用了玩具语料中不存在的 doc_id（共 {total} 个），"
                f"相关指标会被拉低：{sample}",
                file=sys.stderr,
            )

    data = run_demo(
        out_dir=Path(args.out),
        seed=args.seed,
        num_cases=args.num_cases,
        paraphrase_ratio=args.paraphrase_ratio,
        cases=cases,
        notes_prefix=notes_prefix,
    )
    print_summary(data)

    if gates:
        primary = data.primary
        try:
            failures = evaluate_gates(primary, gates)
        except ValueError as exc:
            print(f"参数错误：{exc}", file=sys.stderr)
            return 2
        if failures:
            print(f"评测回归门禁未通过（主管线：{primary.name}）", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure.message}", file=sys.stderr)
            return 1
        passed = ", ".join(
            f"{name}={metric_value(primary, name):.4f}≥{threshold:.4f}" for name, threshold in gates
        )
        print(f"评测回归门禁通过（主管线：{primary.name}）：{passed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
