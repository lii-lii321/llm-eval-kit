"""评测报告输出：Markdown + HTML 各一份，均含指标表格。

HTML 使用克制的配色（深藏青标题 / 白底 / 石板灰正文 / 蓝色强调）。
"""

from __future__ import annotations

import html as html_mod
from dataclasses import dataclass, field
from pathlib import Path

from .attribution import REASON_LABELS_ZH, Badcase
from .judge import DIMENSIONS, DIMENSIONS_ZH, JudgeSummary
from .metrics import PipelineMetrics


@dataclass
class ReportData:
    """一次评测运行的完整结果，是渲染 Markdown/HTML 报告的唯一输入。"""

    generated_at: str
    corpus_size: int
    eval_size: int
    top_k: int
    pipelines: list[PipelineMetrics]
    judge: JudgeSummary = field(default_factory=JudgeSummary)
    attribution: dict[str, dict[str, int]] = field(default_factory=dict)
    primary_badcases: list[Badcase] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def primary(self) -> PipelineMetrics:
        if not self.pipelines:
            raise ValueError("报告中没有任何检索管线")
        return self.pipelines[0]


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _fmt_ms(value: float) -> str:
    return f"{value:.2f}"


def render_markdown(data: ReportData) -> str:
    """渲染 Markdown 报告。"""
    lines: list[str] = ["# LLM/RAG 评测报告", ""]
    lines.append(f"- 生成时间：{data.generated_at}")
    lines.append(f"- 语料文档数：{data.corpus_size}")
    lines.append(f"- 评测集规模：{data.eval_size}")
    lines.append(f"- 检索深度 top-K：{data.top_k}")
    lines.append("")

    if data.pipelines:
        ks = data.primary.ks
        lines.append("## 检索指标")
        lines.append("")
        header = "| 管线 | " + " | ".join(f"Recall@{k}" for k in ks) + " | MRR | NDCG@10 | 命中率 | P50(ms) | P95(ms) |"
        lines.append(header)
        lines.append("|" + "---|" * (len(ks) + 6))
        for p in data.pipelines:
            cells = [p.name] + [_fmt(p.recall.get(k, 0.0)) for k in ks]
            cells += [_fmt(p.mrr), _fmt(p.ndcg), _fmt(p.hit_rate), _fmt_ms(p.latency.p50), _fmt_ms(p.latency.p95)]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## badcase 归因")
    lines.append("")
    if data.attribution:
        lines.append("| 管线 | 关键词不匹配 | 语义漂移 | 语料缺失 | 未命中合计 |")
        lines.append("|---|---|---|---|---|")
        for name, counts in data.attribution.items():
            total = sum(counts.values())
            lines.append(
                f"| {name} | {counts.get('keyword_mismatch', 0)} | {counts.get('semantic_drift', 0)} "
                f"| {counts.get('corpus_missing', 0)} | {total} |"
            )
    else:
        lines.append("所有查询均在 top-K 内命中，无 badcase。")
    if data.primary_badcases:
        lines.append("")
        lines.append(f"### 未命中示例（主管线：{data.primary.name}，最多 5 条）")
        lines.append("")
        for b in data.primary_badcases:
            lines.append(f"- `{b.qid}`【{REASON_LABELS_ZH[b.reason]}】{b.query} —— {b.detail}")
    lines.append("")

    lines.append("## LLM-as-judge")
    lines.append("")
    if data.judge.skipped:
        lines.append(f"> 评分已跳过：{data.judge.skip_reason}")
    elif data.judge.n_scored == 0:
        lines.append("> 无可评分样本（评测集未提供参考答案，或未传入系统答案）。")
    else:
        lines.append(f"- 评分器：`{data.judge.provider}`（样本数 {data.judge.n_scored}）")
        lines.append("")
        lines.append("| 维度 | 平均分 |")
        lines.append("|---|---|")
        for dim in DIMENSIONS:
            lines.append(f"| {DIMENSIONS_ZH[dim]} ({dim}) | {_fmt(data.judge.dim_means.get(dim, 0.0))} |")
    lines.append("")

    for note in data.notes:
        lines.append(f"> {note}")
    return "\n".join(lines) + "\n"


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html_mod.escape(h)}</th>" for h in headers)
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{html_mod.escape(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body_rows}</tbody></table>"


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM/RAG 评测报告</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         color: #334155; background: #fff; margin: 0; }}
  main {{ max-width: 960px; margin: 0 auto; padding: 32px 24px 64px; line-height: 1.7; }}
  h1, h2, h3 {{ color: #1a365d; font-weight: 600; }}
  h1 {{ font-size: 1.6rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
  table {{ border-collapse: collapse; margin: 16px 0; width: 100%; font-size: 0.92rem; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }}
  th {{ background: #f8fafc; color: #1a365d; }}
  code {{ background: #f1f5f9; padding: 1px 5px; border-radius: 4px; font-size: 0.88em; }}
  .meta {{ color: #64748b; font-size: 0.9rem; }}
  .skipped {{ border-left: 3px solid #2563eb; background: #f8fafc; padding: 10px 14px; }}
  .note {{ color: #64748b; font-size: 0.9rem; }}
</style>
</head>
<body>
<main>
<h1>LLM/RAG 评测报告</h1>
<p class="meta">生成时间 {generated_at} ｜ 语料 {corpus_size} 篇 ｜ 评测集 {eval_size} 条 ｜ 检索深度 top-K {top_k}</p>
{sections}
</main>
</body>
</html>
"""


def render_html(data: ReportData) -> str:
    """渲染自包含 HTML 报告（无外部资源）。"""
    sections: list[str] = []

    if data.pipelines:
        ks = data.primary.ks
        headers = ["管线"] + [f"Recall@{k}" for k in ks] + ["MRR", "NDCG@10", "命中率", "P50(ms)", "P95(ms)"]
        rows = []
        for p in data.pipelines:
            cells = [p.name] + [_fmt(p.recall.get(k, 0.0)) for k in ks]
            cells += [_fmt(p.mrr), _fmt(p.ndcg), _fmt(p.hit_rate), _fmt_ms(p.latency.p50), _fmt_ms(p.latency.p95)]
            rows.append(cells)
        sections.append("<h2>检索指标</h2>" + _html_table(headers, rows))

    if data.attribution:
        headers = ["管线", "关键词不匹配", "语义漂移", "语料缺失", "未命中合计"]
        rows = [
            [
                name,
                str(counts.get("keyword_mismatch", 0)),
                str(counts.get("semantic_drift", 0)),
                str(counts.get("corpus_missing", 0)),
                str(sum(counts.values())),
            ]
            for name, counts in data.attribution.items()
        ]
        sections.append("<h2>badcase 归因</h2>" + _html_table(headers, rows))
    else:
        sections.append("<h2>badcase 归因</h2><p>所有查询均在 top-K 内命中，无 badcase。</p>")

    if data.primary_badcases:
        items = "".join(
            f"<li><code>{html_mod.escape(b.qid)}</code>【{html_mod.escape(REASON_LABELS_ZH[b.reason])}】"
            f"{html_mod.escape(b.query)} —— {html_mod.escape(b.detail)}</li>"
            for b in data.primary_badcases
        )
        heading = f"<h3>未命中示例（主管线：{html_mod.escape(data.primary.name)}，最多 5 条）</h3>"
        sections.append(f"{heading}<ul>{items}</ul>")

    if data.judge.skipped:
        sections.append(
            f'<h2>LLM-as-judge</h2><p class="skipped">评分已跳过：{html_mod.escape(data.judge.skip_reason)}</p>'
        )
    elif data.judge.n_scored == 0:
        sections.append("<h2>LLM-as-judge</h2><p>无可评分样本（评测集未提供参考答案，或未传入系统答案）。</p>")
    else:
        rows = [[f"{DIMENSIONS_ZH[dim]} ({dim})", _fmt(data.judge.dim_means.get(dim, 0.0))] for dim in DIMENSIONS]
        sections.append(
            f"<h2>LLM-as-judge</h2><p>评分器：<code>{html_mod.escape(data.judge.provider)}</code>"
            f"（样本数 {data.judge.n_scored}）</p>" + _html_table(["维度", "平均分"], rows)
        )

    for note in data.notes:
        sections.append(f'<p class="note">{html_mod.escape(note)}</p>')

    return _HTML_TEMPLATE.format(
        generated_at=html_mod.escape(data.generated_at),
        corpus_size=data.corpus_size,
        eval_size=data.eval_size,
        top_k=data.top_k,
        sections="\n".join(sections),
    )


def write_reports(data: ReportData, out_dir: str | Path) -> tuple[Path, Path]:
    """写出 Markdown 与 HTML 报告，返回两个文件路径。"""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    md_path = directory / "eval_report.md"
    html_path = directory / "eval_report.html"
    md_path.write_text(render_markdown(data), encoding="utf-8")
    html_path.write_text(render_html(data), encoding="utf-8")
    return md_path, html_path
