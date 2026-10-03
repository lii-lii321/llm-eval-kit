import json
from pathlib import Path

import pytest

from llm_eval_kit.cli import main
from llm_eval_kit.dataset import DatasetError, load_dataset
from llm_eval_kit.demo import build_demo_corpus

EXAMPLES_DATASET = Path(__file__).resolve().parent.parent / "examples" / "dataset_demo.jsonl"


def write_dataset(tmp_path: Path, lines: list[str], name: str = "ds.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def record_line(qid: str = "q1", **overrides) -> str:
    record = {"query_id": qid, "query": "什么是 BM25？", "relevant_doc_ids": ["d01"]}
    record.update(overrides)
    return json.dumps(record, ensure_ascii=False)


VALID_LINES = [
    '{"_meta": {"corpus": "toy", "annotator": "tester"}}',
    record_line("q1", grades={"d01": 2}, notes="核心文档"),
    record_line("q2", query="如何缓解幻觉？", relevant_doc_ids=["d01", "d12"]),
    record_line("q3", query="012345678901234567890123456789012345678901234567890123456789"),
]


class TestLoadDataset:
    def test_loads_valid_dataset(self, tmp_path):
        path = write_dataset(tmp_path, VALID_LINES)
        dataset = load_dataset(path)
        assert dataset.n_cases == 3
        assert dataset.path == path
        assert dataset.meta == {"corpus": "toy", "annotator": "tester"}
        first = dataset.records[0]
        assert first.query_id == "q1"
        assert first.query == "什么是 BM25？"
        assert first.relevant_doc_ids == ["d01"]
        assert first.grades == {"d01": 2}
        assert first.notes == "核心文档"
        assert first.line == 2

    def test_skips_blank_lines_and_counts_line_numbers(self, tmp_path):
        lines = ["", record_line("q1"), "", "", record_line("q2")]
        path = write_dataset(tmp_path, lines)
        dataset = load_dataset(path)
        assert dataset.n_cases == 2
        assert [r.line for r in dataset.records] == [2, 5]

    def test_works_without_meta_line(self, tmp_path):
        path = write_dataset(tmp_path, [record_line("q1")])
        dataset = load_dataset(path)
        assert dataset.n_cases == 1
        assert dataset.meta == {}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(DatasetError, match="不存在"):
            load_dataset(tmp_path / "nope.jsonl")

    def test_empty_file_raises(self, tmp_path):
        path = write_dataset(tmp_path, [])
        with pytest.raises(DatasetError, match="评测集为空"):
            load_dataset(path)

    def test_meta_only_file_raises(self, tmp_path):
        path = write_dataset(tmp_path, ['{"_meta": {"corpus": "toy"}}'])
        with pytest.raises(DatasetError, match="评测集为空"):
            load_dataset(path)

    def test_invalid_json_reports_line_number(self, tmp_path):
        path = write_dataset(tmp_path, [record_line("q1"), '{"query_id": "q2", oops}'])
        with pytest.raises(DatasetError, match="第 2 行不是合法 JSON"):
            load_dataset(path)

    def test_non_object_line_reports_line_number(self, tmp_path):
        path = write_dataset(tmp_path, [record_line("q1"), '["q2"]'])
        with pytest.raises(DatasetError, match="第 2 行必须是 JSON 对象"):
            load_dataset(path)

    def test_missing_required_field_reports_line_number(self, tmp_path):
        line = '{"query_id": "q1", "query": "什么是 BM25？"}'
        path = write_dataset(tmp_path, [line])
        with pytest.raises(DatasetError, match="第 1 行缺少必填字段：relevant_doc_ids"):
            load_dataset(path)

    def test_unknown_field_rejected_with_hint(self, tmp_path):
        line = record_line("q1", relevant_doc_id=["d01"])
        path = write_dataset(tmp_path, [line])
        with pytest.raises(DatasetError, match="未知字段：relevant_doc_id"):
            load_dataset(path)

    def test_blank_query_rejected(self, tmp_path):
        path = write_dataset(tmp_path, [record_line("q1", query="   ")])
        with pytest.raises(DatasetError, match="第 1 行 query 必须是非空字符串"):
            load_dataset(path)

    def test_empty_relevant_ids_rejected(self, tmp_path):
        path = write_dataset(tmp_path, [record_line("q1", relevant_doc_ids=[])])
        with pytest.raises(DatasetError, match="第 1 行 relevant_doc_ids 不能为空"):
            load_dataset(path)

    def test_non_string_relevant_id_rejected(self, tmp_path):
        path = write_dataset(tmp_path, [record_line("q1", relevant_doc_ids=["d01", 3])])
        with pytest.raises(DatasetError, match="含非字符串或空元素"):
            load_dataset(path)

    def test_duplicate_doc_id_within_record_rejected(self, tmp_path):
        path = write_dataset(tmp_path, [record_line("q1", relevant_doc_ids=["d01", "d01"])])
        with pytest.raises(DatasetError, match="重复 doc_id：d01"):
            load_dataset(path)

    def test_duplicate_query_id_reports_both_lines(self, tmp_path):
        path = write_dataset(tmp_path, [record_line("q1"), record_line("q1")])
        with pytest.raises(DatasetError, match=r"第 2 行 query_id 'q1' 与第 1 行重复"):
            load_dataset(path)

    def test_meta_after_records_rejected(self, tmp_path):
        lines = [record_line("q1"), '{"_meta": {"corpus": "toy"}}']
        path = write_dataset(tmp_path, lines)
        with pytest.raises(DatasetError, match="必须出现在首个非空行"):
            load_dataset(path)

    def test_duplicate_meta_rejected(self, tmp_path):
        lines = ['{"_meta": {}}', '{"_meta": {}}', record_line("q1")]
        path = write_dataset(tmp_path, lines)
        with pytest.raises(DatasetError, match=r"重复的 _meta 元信息（首见于第 1 行）"):
            load_dataset(path)

    def test_meta_mixed_with_query_fields_rejected(self, tmp_path):
        line = '{"_meta": {}, "query_id": "q1", "query": "什么是 BM25？", "relevant_doc_ids": ["d01"]}'
        path = write_dataset(tmp_path, [line])
        with pytest.raises(DatasetError, match="不得混入查询字段：query, query_id, relevant_doc_ids"):
            load_dataset(path)

    def test_meta_must_be_object(self, tmp_path):
        path = write_dataset(tmp_path, ['{"_meta": "toy"}', record_line("q1")])
        with pytest.raises(DatasetError, match="_meta 的值必须是 JSON 对象"):
            load_dataset(path)

    def test_grades_value_out_of_range_rejected(self, tmp_path):
        path = write_dataset(tmp_path, [record_line("q1", grades={"d01": 3})])
        with pytest.raises(DatasetError, match="必须是 0/1/2 整数"):
            load_dataset(path)

    def test_grades_bool_value_rejected(self, tmp_path):
        path = write_dataset(tmp_path, [record_line("q1", grades={"d01": True})])
        with pytest.raises(DatasetError, match="必须是 0/1/2 整数"):
            load_dataset(path)

    def test_grades_key_not_in_relevant_ids_rejected(self, tmp_path):
        path = write_dataset(tmp_path, [record_line("q1", grades={"d99": 1})])
        with pytest.raises(DatasetError, match="grades 的键 'd99' 不在该行 relevant_doc_ids 中"):
            load_dataset(path)

    def test_grades_must_be_object(self, tmp_path):
        path = write_dataset(tmp_path, [record_line("q1", grades=[1])])
        with pytest.raises(DatasetError, match="grades 必须是对象"):
            load_dataset(path)

    def test_notes_must_be_string(self, tmp_path):
        path = write_dataset(tmp_path, [record_line("q1", notes=42)])
        with pytest.raises(DatasetError, match="notes 必须是字符串"):
            load_dataset(path)

    def test_error_line_attribute_exposed(self, tmp_path):
        path = write_dataset(tmp_path, [record_line("q1"), record_line("q2", relevant_doc_ids=[])])
        with pytest.raises(DatasetError) as excinfo:
            load_dataset(path)
        assert excinfo.value.line == 2


class TestDatasetStats:
    def test_stats_values(self, tmp_path):
        path = write_dataset(tmp_path, VALID_LINES)
        dataset = load_dataset(path)
        stats = dataset.stats()
        lengths = [len(r.query) for r in dataset.records]
        assert stats.n_cases == 3
        assert stats.avg_relevant_docs == pytest.approx(4 / 3)
        assert stats.query_len_min == min(lengths)
        assert stats.query_len_max == max(lengths)
        assert stats.query_len_median == pytest.approx(float(sorted(lengths)[1]))
        assert stats.query_len_mean == pytest.approx(sum(lengths) / 3)
        assert sum(stats.query_len_buckets.values()) == 3
        assert stats.query_len_buckets[">50"] == 1
        assert stats.query_len_buckets["1-10"] == 2

    def test_describe_contains_key_numbers(self, tmp_path):
        path = write_dataset(tmp_path, VALID_LINES)
        text = load_dataset(path).stats().describe()
        assert "共 3 条" in text
        assert "1.33" in text
        assert "长度分布" in text


class TestDatasetConversions:
    def test_to_eval_cases_mapping(self, tmp_path):
        path = write_dataset(tmp_path, VALID_LINES)
        cases = load_dataset(path).to_eval_cases()
        assert [c.qid for c in cases] == ["q1", "q2", "q3"]
        assert cases[1].relevant_ids == ["d01", "d12"]
        assert all(c.answer == "" for c in cases)
        assert cases[0].meta == {"grades": {"d01": 2}, "notes": "核心文档", "line": 2}
        assert cases[1].meta == {"grades": {}, "notes": "", "line": 3}

    def test_unknown_doc_ids_flags_missing_corpus_docs(self, tmp_path):
        path = write_dataset(
            tmp_path,
            [
                record_line("q1", relevant_doc_ids=["d01", "d99"]),
                record_line("q2", relevant_doc_ids=["d01"]),
                record_line("q3", relevant_doc_ids=["d98", "d97"]),
            ],
        )
        unknown = load_dataset(path).unknown_doc_ids(doc.doc_id for doc in build_demo_corpus())
        assert unknown == {"q1": ["d99"], "q3": ["d98", "d97"]}


class TestSeedExampleDataset:
    def test_seed_file_loads_and_is_synthetic_demo(self):
        dataset = load_dataset(EXAMPLES_DATASET)
        assert 10 <= dataset.n_cases <= 15
        assert "synthetic demo, not human-annotated" in dataset.meta["provenance"]
        assert dataset.meta["annotated_at"] and dataset.meta["version"]

    def test_seed_file_doc_ids_all_in_demo_corpus(self):
        dataset = load_dataset(EXAMPLES_DATASET)
        assert dataset.unknown_doc_ids(doc.doc_id for doc in build_demo_corpus()) == {}

    def test_seed_file_ids_unique_and_queries_nonempty(self):
        dataset = load_dataset(EXAMPLES_DATASET)
        assert len({r.query_id for r in dataset.records}) == dataset.n_cases
        assert all(r.query.strip() and r.relevant_doc_ids for r in dataset.records)


class TestCliDataset:
    def test_end_to_end_with_reports_and_gate(self, tmp_path, capsys):
        out_dir = tmp_path / "reports"
        rc = main(["--dataset", str(EXAMPLES_DATASET), "--out", str(out_dir), "--fail-under", "recall_at_5=0.5"])
        assert rc == 0
        assert (out_dir / "eval_report.md").exists()
        assert (out_dir / "eval_report.html").exists()
        out = capsys.readouterr().out
        assert "已加载评测集" in out
        assert "bm25" in out
        assert "评测回归门禁通过" in out

    def test_meta_and_stats_go_into_report(self, tmp_path):
        out_dir = tmp_path / "reports"
        main(["--dataset", str(EXAMPLES_DATASET), "--out", str(out_dir)])
        report = (out_dir / "eval_report.md").read_text(encoding="utf-8")
        assert "评测集统计" in report
        assert "synthetic demo, not human-annotated" in report

    def test_fail_under_failure_exits_1(self, tmp_path, capsys):
        rc = main(["--dataset", str(EXAMPLES_DATASET), "--out", str(tmp_path), "--fail-under", "recall_at_1=1.0"])
        assert rc == 1
        assert "门禁未达标" in capsys.readouterr().err

    def test_missing_dataset_file_exits_2(self, tmp_path, capsys):
        rc = main(["--dataset", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path)])
        assert rc == 2
        assert "评测集加载失败" in capsys.readouterr().err

    def test_invalid_dataset_line_number_in_stderr(self, tmp_path, capsys):
        path = write_dataset(tmp_path, [record_line("q1"), record_line("q1")])
        rc = main(["--dataset", str(path), "--out", str(tmp_path)])
        assert rc == 2
        assert "第 2 行" in capsys.readouterr().err

    def test_unknown_doc_id_warns_but_evaluates(self, tmp_path, capsys):
        lines = [record_line("q1", relevant_doc_ids=["d01", "d99"])]
        path = write_dataset(tmp_path, lines)
        rc = main(["--dataset", str(path), "--out", str(tmp_path)])
        captured = capsys.readouterr()
        assert rc == 0
        assert "警告" in captured.err
        assert "d99" in captured.err
        assert "bm25" in captured.out

    def test_num_cases_conflicts_with_dataset(self, tmp_path, capsys):
        rc = main(["--dataset", str(EXAMPLES_DATASET), "--out", str(tmp_path), "--num-cases", "8"])
        assert rc == 2
        assert "互斥" in capsys.readouterr().err

    def test_paraphrase_ratio_conflicts_with_dataset(self, tmp_path, capsys):
        rc = main(["--dataset", str(EXAMPLES_DATASET), "--out", str(tmp_path), "--paraphrase-ratio", "0.5"])
        assert rc == 2
        assert "互斥" in capsys.readouterr().err

    def test_seed_is_allowed_with_dataset(self, tmp_path):
        rc = main(["--dataset", str(EXAMPLES_DATASET), "--out", str(tmp_path), "--seed", "7"])
        assert rc == 0
