"""大语料基准：玩具 BM25（全量扫描） vs FastBM25Retriever（倒排索引）。

生成合成中文语料（Zipf 词频、seed 可复现），对比索引构建耗时与单查询延迟。

用法（在仓库根目录）：
    python scripts/bench_large_corpus.py --n-docs 20000 --n-queries 100
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_eval_kit.corpus import Doc  # noqa: E402
from llm_eval_kit.fast_retriever import FastBM25Retriever  # noqa: E402
from llm_eval_kit.retrieval import BM25Retriever  # noqa: E402
from llm_eval_kit.tokenize import tokenize  # noqa: E402

# 200 个基础汉字 → 相邻二元组词元池，Zipf 权重采样让词频分布接近真实。
# 非加密用途：基准需要固定 seed 的可复现伪随机序列，故用 random 而非 secrets
_CHARS = (
    "的一是了我不人在他有这上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多"
    "然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它"
    "头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"
)


def _build_vocab(rng: random.Random) -> list[str]:
    """5000 个二元词元（汉字两两组合），接近真实中文语料的词元规模。"""
    chars = list(_CHARS)
    pairs: set[str] = set()
    while len(pairs) < 5000:
        pairs.add(rng.choice(chars) + rng.choice(chars))
    return sorted(pairs)


def _sample_token(vocab: list[str], rng: random.Random) -> str:
    # Zipf：index 越小越常见
    i = min(int(rng.paretovariate(1.2)) - 1, len(vocab) - 1)
    return vocab[i]


def make_corpus(n_docs: int, rng: random.Random) -> list[Doc]:
    vocab = _build_vocab(rng)
    docs = []
    for i in range(n_docs):
        n_tokens = rng.randint(40, 90)
        text = " ".join(_sample_token(vocab, rng) for _ in range(n_tokens))
        docs.append(Doc(doc_id=f"d{i:05d}", text=text))
    return docs


def make_queries(
    vocab: list[str], n_queries: int, rng: random.Random, *, min_rank: int = 50, max_rank: int = 2000
) -> list[str]:
    """查询词元取自中低词频带（秩 min_rank~max_rank 的 Zipf 采样）。

    真实关键词查询很少由语料中最高频的词元构成（那些近似停用词），
    限制秩区间模拟这一行为，避免基准被头部高频词元主导而失真。
    """
    queries = []
    for _ in range(n_queries):
        n_tokens = rng.randint(6, 14)
        tokens: list[str] = []
        while len(tokens) < n_tokens:
            r = min(int(rng.paretovariate(1.2)) - 1, len(vocab) - 1)
            if min_rank <= r <= max_rank:
                tokens.append(vocab[r])
        queries.append(" ".join(tokens))
    return queries


def bench_build(cls, docs):  # noqa: ANN001, ANN201
    t0 = time.perf_counter()
    inst = cls(docs)
    return inst, (time.perf_counter() - t0) * 1000


def bench_queries(inst, queries):  # noqa: ANN001, ANN201
    lat = []
    for q in queries:
        t0 = time.perf_counter()
        inst.retrieve(q, top_k=10)
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    p50 = statistics.median(lat)
    p95 = lat[max(0, int(len(lat) * 0.95) - 1)]
    return p50, p95, sum(lat)


def main() -> None:
    ap = argparse.ArgumentParser(description="大语料 BM25 基准（玩具全量扫描 vs 倒排索引）")
    ap.add_argument("--n-docs", type=int, default=20000)
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-rank", type=int, default=50, help="查询词元最小秩（越大越避开高频词元）")
    ap.add_argument("--max-rank", type=int, default=2000, help="查询词元最大秩")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    docs = make_corpus(args.n_docs, rng)
    total_tokens = sum(len(tokenize(d.text)) for d in docs)
    queries_rng = random.Random(args.seed + 1)
    queries = make_queries(
        _build_vocab(random.Random(args.seed)),
        args.n_queries,
        queries_rng,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
    )
    print(f"语料：{args.n_docs} 段 / {total_tokens} 词元 · 查询：{args.n_queries} 条 · seed={args.seed}")
    print(f"查询词元秩带 {args.min_rank}~{args.max_rank}（模拟避开近似停用词的关键词查询）")

    old, old_build = bench_build(BM25Retriever, docs)
    fast, fast_build = bench_build(FastBM25Retriever, docs)
    scan_ratio = statistics.mean(
        sum(len(fast._postings.get(t, [])) for t in set(tokenize(q))) / len(docs)  # noqa: SLF001
        for q in queries
    )
    old_p50, old_p95, old_sum = bench_queries(old, queries)
    fast_p50, fast_p95, fast_sum = bench_queries(fast, queries)
    print(f"倒排扫描量占比均值：{scan_ratio:.1%}")

    print(f"{'实现':<22}{'构建 ms':>10}{'P50 ms':>10}{'P95 ms':>10}{'总耗时 ms':>12}")
    print(f"{'BM25Retriever(全量)':<22}{old_build:>10.0f}{old_p50:>10.1f}{old_p95:>10.1f}{old_sum:>12.0f}")
    print(f"{'FastBM25Retriever':<22}{fast_build:>10.0f}{fast_p50:>10.1f}{fast_p95:>10.1f}{fast_sum:>12.0f}")
    print(f"查询加速比：{old_p50 / fast_p50:.1f}x (P50) / {old_p95 / fast_p95:.1f}x (P95)")

    # 抽样一致性：top-10 文档集合与得分应一致（浮点累加次序不同，容差 1e-9）
    sample = queries[:5]
    for q in sample:
        a = old.retrieve(q, top_k=10)
        b = fast.retrieve(q, top_k=10)
        ids_a = [r.doc_id for r in a]
        ids_b = [r.doc_id for r in b]
        assert ids_a == ids_b, f"top-10 文档序列不一致：{q!r}\n{ids_a}\n{ids_b}"
        max_diff = max(abs(x.score - y.score) for x, y in zip(a, b, strict=True))
        assert max_diff < 1e-9, f"得分偏差超容差：{max_diff}"
    print(f"一致性抽检：{len(sample)} 条查询 top-10 序列与得分（容差 1e-9）全部一致")


if __name__ == "__main__":
    main()
