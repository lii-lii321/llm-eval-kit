import math

import pytest

from llm_eval_kit.metrics import (
    ConfidenceInterval,
    average_precision,
    bootstrap_ci,
    dcg_at_k,
    latency_stats,
    mrr,
    ndcg_at_k,
    ndcg_at_k_weighted,
    percentile,
    recall_at_k,
)


class TestRecallAtK:
    def test_perfect_hit(self):
        assert recall_at_k(["a", "b"], {"a"}, 1) == 1.0

    def test_partial_hit(self):
        assert recall_at_k(["a", "b"], {"a", "c"}, 2) == 0.5

    def test_no_hit(self):
        assert recall_at_k(["x", "y"], {"a"}, 5) == 0.0

    def test_empty_relevant_is_zero(self):
        assert recall_at_k(["a"], set(), 3) == 0.0

    def test_k_beyond_list_length(self):
        assert recall_at_k(["a"], {"a"}, 10) == 1.0

    def test_k_cuts_off_results(self):
        assert recall_at_k(["x", "a"], {"a"}, 1) == 0.0

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            recall_at_k(["a"], {"a"}, 0)


class TestMRR:
    def test_first_position(self):
        assert mrr(["a", "b"], {"a"}) == 1.0

    def test_second_position(self):
        assert mrr(["x", "a"], {"a"}) == 0.5

    def test_no_hit(self):
        assert mrr(["x", "y"], {"a"}) == 0.0

    def test_uses_first_relevant(self):
        assert mrr(["x", "a", "b"], {"a", "b"}) == 0.5

    def test_empty_ranked(self):
        assert mrr([], {"a"}) == 0.0


class TestNDCGAtK:
    def test_perfect_ordering(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, 10) == pytest.approx(1.0)

    def test_position_matters_for_single_relevant(self):
        perfect = ndcg_at_k(["a", "x"], {"a"}, 10)
        swapped = ndcg_at_k(["x", "a"], {"a"}, 10)
        assert perfect == pytest.approx(1.0)
        assert swapped < perfect

    def test_no_relevant_returns_zero(self):
        assert ndcg_at_k(["a"], set(), 10) == 0.0

    def test_relevant_beyond_k_ignored(self):
        assert ndcg_at_k(["x", "a"], {"a"}, 1) == 0.0

    def test_single_relevant_at_second_position(self):
        assert ndcg_at_k(["x", "a"], {"a"}, 10) == pytest.approx(1 / math.log2(3))

    def test_k_smaller_than_relevant_count(self):
        value = ndcg_at_k(["a", "b", "x"], {"a", "b", "c"}, 2)
        assert value == pytest.approx(1.0)


class TestDCGAtK:
    def test_hand_computed_exponential_gains(self):
        # gains [2,1,0] → 3/log2(2) + 1/log2(3) + 0/log2(4)
        assert dcg_at_k([2, 1, 0], 3) == pytest.approx(3 + 1 / math.log2(3))

    def test_zero_gain_contributes_nothing(self):
        assert dcg_at_k([2, 1, 0], 2) == pytest.approx(3 + 1 / math.log2(3))

    def test_k_truncates_positions(self):
        # 增益 3 = 2^3 - 1，只取第 1 位
        assert dcg_at_k([3, 2, 1], 1) == pytest.approx(7.0)

    def test_position_discount(self):
        # 高增益排后：3/log2(3)
        assert dcg_at_k([0, 2], 2) == pytest.approx(3 / math.log2(3))

    def test_empty_gains(self):
        assert dcg_at_k([], 5) == 0.0

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            dcg_at_k([1], 0)


class TestNDCGWeighted:
    GRADES = {"a": 2, "b": 1, "c": 0}

    def test_perfect_ordering_is_one(self):
        # 按增益降序 a(3) > b(1) > c(0) 排列，DCG == IDCG
        assert ndcg_at_k_weighted(["a", "b", "c"], {"a", "b", "c"}, self.GRADES, 3) == pytest.approx(1.0)

    def test_hand_computed_imperfect_order(self):
        # ranked [b,a]：DCG = 1/log2(2) + 3/log2(3)；IDCG = 3/log2(2) + 1/log2(3)
        value = ndcg_at_k_weighted(["b", "a"], {"a", "b"}, {"a": 2, "b": 1}, 2)
        expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
        assert value == pytest.approx(expected)
        assert 0.79 < value < 0.80

    def test_higher_grade_should_rank_first(self):
        good = ndcg_at_k_weighted(["a", "b"], {"a", "b"}, {"a": 2, "b": 1}, 2)
        bad = ndcg_at_k_weighted(["b", "a"], {"a", "b"}, {"a": 2, "b": 1}, 2)
        assert good == pytest.approx(1.0)
        assert bad < good

    def test_ideal_dcg_sorts_all_gains_not_document_order(self):
        # K=1 时理想 DCG 取全部增益排序后的第 1 位（b 的增益 3），而非排首位的 a（增益 1）
        value = ndcg_at_k_weighted(["a", "b"], {"a", "b"}, {"a": 1, "b": 2}, 1)
        assert value == pytest.approx(1 / 3)

    def test_ungraded_relevant_defaults_to_gain_one(self):
        # b 未标注分级 → 增益 1，与显式标 b:1 同值
        value = ndcg_at_k_weighted(["b", "a"], {"a", "b"}, {"a": 2}, 2)
        expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
        assert value == pytest.approx(expected)

    def test_empty_grades_reduces_to_binary_ndcg(self):
        ranked, rel = ["x", "a", "b"], {"a", "b"}
        assert ndcg_at_k_weighted(ranked, rel, {}, 10) == pytest.approx(ndcg_at_k(ranked, rel, 10))
        assert ndcg_at_k_weighted(ranked, rel, None, 10) == pytest.approx(ndcg_at_k(ranked, rel, 10))

    def test_grade_zero_doc_contributes_nothing(self):
        # 只标了 grade 0 → 全部增益为 0，IDCG 为 0，返回 0
        assert ndcg_at_k_weighted(["a"], {"a"}, {"a": 0}, 10) == 0.0

    def test_relevant_beyond_k_ignored(self):
        assert ndcg_at_k_weighted(["x", "a"], {"a"}, {"a": 2}, 1) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert ndcg_at_k_weighted(["a"], set(), {"a": 2}, 10) == 0.0

    def test_result_bounded_in_unit_interval(self):
        value = ndcg_at_k_weighted(["x", "y", "a"], {"a"}, {"a": 2}, 10)
        assert 0.0 < value <= 1.0


class TestPercentile:
    def test_median_even_count(self):
        assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)

    def test_min_and_max(self):
        assert percentile([3, 1, 2], 0) == pytest.approx(1.0)
        assert percentile([3, 1, 2], 100) == pytest.approx(3.0)

    def test_single_value(self):
        assert percentile([7], 95) == pytest.approx(7.0)

    def test_matches_known_interpolation(self):
        assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            percentile([], 50)

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            percentile([1], 101)


class TestAveragePrecision:
    def test_perfect_ordering(self):
        assert average_precision(["a", "b", "x"], {"a", "b"}, 10) == pytest.approx(1.0)

    def test_order_penalty(self):
        good = average_precision(["a", "x", "b"], {"a", "b"}, 10)
        bad = average_precision(["x", "a", "b"], {"a", "b"}, 10)
        assert good == pytest.approx(5 / 6)
        assert bad == pytest.approx(7 / 12)
        assert good > bad

    def test_single_relevant_at_third_position(self):
        assert average_precision(["x", "y", "a"], {"a"}, 10) == pytest.approx(1 / 3)

    def test_all_relevant_found_sum_over_total(self):
        value = average_precision(["a", "x", "b", "c"], {"a", "b", "c"}, 10)
        assert value == pytest.approx((1 + 2 / 3 + 3 / 4) / 3)

    def test_no_hit(self):
        assert average_precision(["x", "y"], {"a"}, 10) == 0.0

    def test_empty_relevant_is_zero(self):
        assert average_precision(["a"], set(), 10) == 0.0

    def test_k_cuts_off_results(self):
        # 「a」在 K 之外被截断，「b」不在结果里，top-1 未命中
        assert average_precision(["x", "a"], {"a", "b"}, 1) == 0.0

    def test_k_smaller_than_relevant_count_normalizes_by_k(self):
        # top-1 命中即满分：归一分母是 min(相关文档数, K)，与 ndcg_at_k 口径一致
        assert average_precision(["a"], {"a", "z"}, 1) == pytest.approx(1.0)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            average_precision(["a"], {"a"}, 0)


class TestBootstrapCI:
    SPREAD = [0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.35]

    def test_constant_values_degenerate_interval(self):
        ci = bootstrap_ci([0.5] * 10)
        assert ci.low == pytest.approx(0.5)
        assert ci.high == pytest.approx(0.5)
        assert ci.level == 0.95
        assert ci.n_boot == 1000

    def test_interval_within_sample_range(self):
        ci = bootstrap_ci(self.SPREAD, seed=1)
        assert 0.1 <= ci.low <= ci.high <= 0.9

    def test_same_seed_reproducible(self):
        assert bootstrap_ci(self.SPREAD, seed=7) == bootstrap_ci(self.SPREAD, seed=7)

    def test_variable_data_wider_than_constant(self):
        variable = bootstrap_ci([0.0, 1.0] * 10, seed=3)
        constant = bootstrap_ci([0.5] * 20, seed=3)
        assert (variable.high - variable.low) > (constant.high - constant.low)

    def test_custom_n_boot_and_level_recorded(self):
        ci = bootstrap_ci([0.2, 0.8], n_boot=50, confidence=0.9, seed=11)
        assert ci.n_boot == 50
        assert ci.level == 0.9
        assert isinstance(ci, ConfidenceInterval)

    def test_interval_brackets_sample_mean(self):
        values = [0.2] * 30 + [0.8] * 30
        ci = bootstrap_ci(values, seed=5)
        mean = sum(values) / len(values)
        assert ci.low <= mean <= ci.high

    def test_empty_values_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([])

    def test_non_positive_n_boot_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([1.0], n_boot=0)

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([1.0], confidence=1.0)
        with pytest.raises(ValueError):
            bootstrap_ci([1.0], confidence=0.0)


class TestLatencyStats:
    def test_empty_values(self):
        stats = latency_stats([])
        assert stats.n == 0
        assert stats.p50 == 0.0
        assert stats.mean == 0.0

    def test_basic_statistics(self):
        stats = latency_stats([1.0, 2.0, 3.0, 4.0])
        assert stats.n == 4
        assert stats.p50 == pytest.approx(2.5)
        assert stats.mean == pytest.approx(2.5)
        assert stats.max == pytest.approx(4.0)
        assert stats.p95 == pytest.approx(3.85)

    def test_single_sample(self):
        stats = latency_stats([5.0])
        assert stats.n == 1
        assert stats.p50 == pytest.approx(5.0)
        assert stats.p95 == pytest.approx(5.0)
