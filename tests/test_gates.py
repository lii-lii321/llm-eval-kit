"""评测回归门禁测试：解析、指标取值、判定逻辑与 CLI 两条退出分支。"""

import pytest

from llm_eval_kit.cli import main
from llm_eval_kit.gates import (
    GateFailure,
    evaluate_gates,
    metric_value,
    parse_fail_under,
    supported_metric_names,
)
from llm_eval_kit.metrics import PipelineMetrics


def _metrics() -> PipelineMetrics:
    metrics = PipelineMetrics(name="primary", ks=(1, 3, 5, 10))
    metrics.recall = {1: 0.5, 3: 0.75, 5: 0.9, 10: 1.0}
    metrics.mrr = 0.8
    metrics.map = 0.7
    metrics.ndcg = 0.6
    metrics.hit_rate = 0.4
    return metrics


class TestParseFailUnder:
    def test_valid_spec(self):
        assert parse_fail_under("recall_at_5=0.85") == ("recall_at_5", 0.85)

    def test_whitespace_tolerated(self):
        assert parse_fail_under("  mrr = 0.8 ") == ("mrr", 0.8)

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError) as excinfo:
            parse_fail_under("recall_at_5")
        assert "指标名=阈值" in str(excinfo.value)
        assert "recall_at_5=0.85" in str(excinfo.value)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="指标名不能为空"):
            parse_fail_under("=0.5")

    def test_non_numeric_threshold_raises(self):
        with pytest.raises(ValueError, match="阈值必须是数字"):
            parse_fail_under("mrr=high")

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            parse_fail_under("mrr=1.5")
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            parse_fail_under("mrr=-0.1")


class TestMetricValue:
    def test_recall_at_k(self):
        assert metric_value(_metrics(), "recall_at_5") == 0.9

    def test_fixed_metrics_and_aliases(self):
        metrics = _metrics()
        assert metric_value(metrics, "mrr") == 0.8
        assert metric_value(metrics, "map") == 0.7
        assert metric_value(metrics, "map_at_10") == 0.7
        assert metric_value(metrics, "ndcg") == 0.6
        assert metric_value(metrics, "ndcg_at_10") == 0.6
        assert metric_value(metrics, "hit_rate") == 0.4

    def test_unknown_name_lists_supported(self):
        with pytest.raises(ValueError) as excinfo:
            metric_value(_metrics(), "precision_at_5")
        message = str(excinfo.value)
        assert "precision_at_5" in message
        assert "recall_at_1" in message
        assert "ndcg_at_10" in message

    def test_uncached_k_reports_available(self):
        with pytest.raises(ValueError) as excinfo:
            metric_value(_metrics(), "recall_at_7")
        message = str(excinfo.value)
        assert "recall_at_7" in message
        assert "7" in message

    def test_supported_metric_names_cover_all_fixtures(self):
        names = supported_metric_names((1, 3, 5, 10))
        for expected in ("recall_at_1", "recall_at_10", "mrr", "map", "map_at_10", "ndcg", "ndcg_at_10", "hit_rate"):
            assert expected in names

    def test_weighted_ndcg_value_and_aliases(self):
        metrics = _metrics()
        metrics.weighted_ndcg = 0.66
        assert metric_value(metrics, "weighted_ndcg") == 0.66
        assert metric_value(metrics, "weighted_ndcg_at_10") == 0.66

    def test_weighted_ndcg_unavailable_raises_not_zero(self):
        # 评测集无 grades 时 weighted_ndcg 为 None：报错说明不可用，而非按 0 判定
        with pytest.raises(ValueError) as excinfo:
            metric_value(_metrics(), "weighted_ndcg_at_10")
        message = str(excinfo.value)
        assert "weighted_ndcg_at_10" in message
        assert "grades" in message
        assert "不可用" in message

    def test_supported_names_include_weighted_ndcg(self):
        names = supported_metric_names((1, 3, 5, 10))
        assert "weighted_ndcg" in names
        assert "weighted_ndcg_at_10" in names


class TestEvaluateGates:
    def test_all_pass_returns_empty(self):
        assert evaluate_gates(_metrics(), [("recall_at_5", 0.85), ("mrr", 0.8)]) == []

    def test_boundary_equal_passes(self):
        assert evaluate_gates(_metrics(), [("mrr", 0.8)]) == []

    def test_failure_captures_actual_threshold_gap(self):
        failures = evaluate_gates(_metrics(), [("recall_at_1", 0.9), ("mrr", 0.99)])
        assert len(failures) == 2
        first = failures[0]
        assert isinstance(first, GateFailure)
        assert first.metric == "recall_at_1"
        assert first.actual == 0.5
        assert first.threshold == 0.9
        assert first.gap == pytest.approx(0.4)
        assert "recall_at_1" in first.message
        assert "0.5000" in first.message
        assert "0.9000" in first.message


class TestCliGate:
    def test_passing_gate_exits_zero(self, tmp_path, capsys):
        rc = main([
            "demo", "--out", str(tmp_path), "--num-cases", "8",
            "--fail-under", "recall_at_5=0.5",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "评测回归门禁通过" in out
        assert "recall_at_5" in out

    def test_failing_gate_exits_one_with_details(self, tmp_path, capsys):
        rc = main([
            "demo", "--out", str(tmp_path), "--num-cases", "8",
            "--fail-under", "recall_at_1=0.9",
        ])
        assert rc == 1
        captured = capsys.readouterr()
        assert "评测回归门禁未通过" in captured.err
        assert "recall_at_1" in captured.err
        assert "0.7500" in captured.err
        assert "0.9000" in captured.err

    def test_multiple_gates_any_failure_fails(self, tmp_path, capsys):
        rc = main([
            "demo", "--out", str(tmp_path), "--num-cases", "8",
            "--fail-under", "recall_at_5=0.5",
            "--fail-under", "recall_at_1=0.99",
        ])
        assert rc == 1
        captured = capsys.readouterr()
        assert "recall_at_1" in captured.err
        assert "recall_at_5" not in captured.err

    def test_malformed_spec_rejected_before_running(self, tmp_path, capsys):
        rc = main(["demo", "--out", str(tmp_path), "--fail-under", "nonsense"])
        assert rc == 2
        assert "指标名=阈值" in capsys.readouterr().err
        assert not (tmp_path / "eval_report.md").exists()

    def test_unknown_metric_exits_two(self, tmp_path, capsys):
        rc = main([
            "demo", "--out", str(tmp_path), "--num-cases", "6",
            "--fail-under", "precision_at_5=0.5",
        ])
        assert rc == 2
        assert "precision_at_5" in capsys.readouterr().err

    def test_weighted_ndcg_gate_without_grades_exits_two(self, tmp_path, capsys):
        # demo（合成评测集）无 grades：weighted_ndcg 门禁报不可用，退出码 2
        rc = main([
            "demo", "--out", str(tmp_path), "--num-cases", "6",
            "--fail-under", "weighted_ndcg_at_10=0.5",
        ])
        assert rc == 2
        err = capsys.readouterr().err
        assert "weighted_ndcg_at_10" in err
        assert "不可用" in err

    def test_no_gate_keeps_default_behavior(self, tmp_path, capsys):
        rc = main(["demo", "--out", str(tmp_path), "--num-cases", "6"])
        assert rc == 0
        assert "门禁" not in capsys.readouterr().out
