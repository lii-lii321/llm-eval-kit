from llm_eval_kit.attribution import REASON_KEYWORD_MISMATCH, Badcase
from llm_eval_kit.judge import DIMENSIONS, JudgeSummary
from llm_eval_kit.metrics import ConfidenceInterval, LatencyStats, PipelineMetrics
from llm_eval_kit.report import ReportData, render_html, render_markdown, write_reports


def make_report(**overrides) -> ReportData:
    p1 = PipelineMetrics(
        name="bm25",
        ks=(1, 5, 10),
        recall={1: 0.5, 5: 0.8, 10: 0.9},
        mrr=0.7,
        ndcg=0.6,
        hit_rate=0.85,
        latency=LatencyStats(p50=1.23, p95=2.34, mean=1.5, max=3.0, n=20),
    )
    p2 = PipelineMetrics(
        name="tf",
        ks=(1, 5, 10),
        recall={1: 0.4, 5: 0.7, 10: 0.85},
        mrr=0.6,
        ndcg=0.55,
        hit_rate=0.8,
        latency=LatencyStats(p50=1.1, p95=2.0, mean=1.3, max=2.5, n=20),
    )
    data = ReportData(
        generated_at="2026-10-02T00:00:00",
        corpus_size=24,
        eval_size=24,
        top_k=10,
        pipelines=[p1, p2],
        judge=JudgeSummary(
            provider="mock",
            n_scored=24,
            dim_means={dim: 4.2 for dim in DIMENSIONS},
        ),
        attribution={
            "bm25": {"keyword_mismatch": 2, "semantic_drift": 1, "corpus_missing": 0},
            "tf": {"keyword_mismatch": 3, "semantic_drift": 2, "corpus_missing": 1},
        },
        primary_badcases=[Badcase("q001", "查询<script>alert(1)</script>", REASON_KEYWORD_MISMATCH, "零重叠")],
    )
    for key, value in overrides.items():
        setattr(data, key, value)
    return data


class TestRenderMarkdown:
    def test_contains_metric_table(self):
        md = render_markdown(make_report())
        assert "## 检索指标" in md
        assert "| bm25 |" in md
        assert "| tf |" in md
        assert "0.7000" in md  # MRR 0.7

    def test_contains_recall_columns(self):
        md = render_markdown(make_report())
        assert "Recall@1" in md
        assert "Recall@5" in md
        assert "Recall@10" in md
        assert "NDCG@10" in md
        assert "P50(ms)" in md

    def test_contains_attribution_table(self):
        md = render_markdown(make_report())
        assert "关键词不匹配" in md
        assert "语义漂移" in md
        assert "语料缺失" in md

    def test_contains_judge_section(self):
        md = render_markdown(make_report())
        assert "LLM-as-judge" in md
        assert "正确性" in md
        assert "4.2000" in md

    def test_judge_skipped_rendered(self):
        data = make_report(judge=JudgeSummary(skipped=True, skip_reason="EVAL_LLM_API_KEY 未设置"))
        md = render_markdown(data)
        assert "评分已跳过" in md
        assert "EVAL_LLM_API_KEY 未设置" in md

    def test_no_scored_samples_rendered(self):
        data = make_report(judge=JudgeSummary())
        assert "无可评分样本" in render_markdown(data)

    def test_badcase_example_listed(self):
        md = render_markdown(make_report())
        assert "q001" in md
        assert "关键词不匹配" in md


class TestRenderHtml:
    def test_doctype_and_table(self):
        html = render_html(make_report())
        assert html.startswith("<!DOCTYPE html>")
        assert "<table>" in html
        assert "检索指标" in html

    def test_escapes_user_content(self):
        html = render_html(make_report())
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_judge_skipped_rendered(self):
        data = make_report(judge=JudgeSummary(skipped=True, skip_reason="无 key"))
        assert "评分已跳过" in render_html(data)


class TestWriteReports:
    def test_writes_both_files(self, tmp_path):
        data = make_report()
        md_path, html_path = write_reports(data, tmp_path)
        assert md_path.exists() and html_path.exists()
        assert md_path.read_text(encoding="utf-8") == render_markdown(data)
        assert html_path.read_text(encoding="utf-8") == render_html(data)

    def test_creates_directory(self, tmp_path):
        out = tmp_path / "nested" / "reports"
        md_path, html_path = write_reports(make_report(), out)
        assert md_path.exists() and html_path.exists()


class TestWeightedNDCGColumn:
    """Weighted NDCG@10 列：无 grades 显示 —（诚实降级），有 grades 出数值与口径说明。"""

    def test_markdown_unavailable_renders_dash_and_note(self):
        md = render_markdown(make_report())
        assert "Weighted NDCG@10" in md
        metrics_section = md.split("## 检索指标")[1].split("##")[0]
        assert "| — |" in metrics_section
        assert "未提供 grades 分级标注" in metrics_section
        assert "不输出 0 冒充" in metrics_section

    def test_markdown_available_renders_value_with_ci_and_graded_note(self):
        data = make_report()
        data.pipelines[0].weighted_ndcg = 0.8
        data.pipelines[0].weighted_ndcg_ci = ConfidenceInterval(low=0.7, high=0.9)
        md = render_markdown(data)
        assert "0.8000 [0.7000, 0.9000]" in md
        assert "分级相关度指标" in md

    def test_html_unavailable_renders_dash(self):
        html = render_html(make_report())
        assert "Weighted NDCG@10" in html
        assert "<td>—</td>" in html
        assert "未提供 grades 分级标注" in html

    def test_html_available_renders_value(self):
        data = make_report()
        data.pipelines[0].weighted_ndcg = 0.8
        html = render_html(data)
        assert "<td>0.8000</td>" in html
        assert "分级相关度指标" in html
