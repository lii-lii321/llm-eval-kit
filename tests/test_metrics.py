import math

import pytest

from llm_eval_kit.metrics import latency_stats, mrr, ndcg_at_k, percentile, recall_at_k


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
