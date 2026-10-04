import json
from pathlib import Path

import pytest

from llm_eval_kit.cli import main
from llm_eval_kit.demo import DEMO_DOCS, DEMO_SYNONYMS, ExtractiveAnswerer, build_demo_corpus, run_demo
from llm_eval_kit.retrieval import BM25Retriever

EXAMPLES_DATASET = Path(__file__).resolve().parent.parent / "examples" / "dataset_demo.jsonl"


def write_external_corpus(tmp_path: Path, name: str = "docs") -> Path:
    """构造小型外部语料目录：2 个文件切出 3 个文本块（rag#000, rag#001, eval#000）。"""
    corpus = tmp_path / name
    corpus.mkdir()
    (corpus / "rag.md").write_text(
        "检索增强生成：先检索外部知识再生成答案，可显著缓解模型幻觉。\n\n"
        "BM25：经典稀疏检索算法，基于词频与逆文档频率打分，无需训练。",
        encoding="utf-8",
    )
    (corpus / "eval.txt").write_text(
        "重排序：用交叉编码器对粗排候选精排，只处理少量候选却能显著提升排序质量。",
        encoding="utf-8",
    )
    return corpus


def write_jsonl_dataset(tmp_path: Path, records: list[dict], name: str = "ds.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


class TestDemoCorpus:
    def test_corpus_size(self):
        assert len(DEMO_DOCS) == 24

    def test_docs_unique_and_nonempty(self):
        docs = build_demo_corpus()
        assert len({d.doc_id for d in docs}) == 24
        assert all(d.text.strip() for d in docs)
        assert all(d.source == "demo" for d in docs)

    def test_synonyms_have_no_lexical_overlap(self):
        """改写词与原词必须零词面重叠，且改写词不出现在语料中。"""
        from llm_eval_kit.tokenize import tokenize

        corpus_tokens = set()
        for d in build_demo_corpus():
            corpus_tokens.update(tokenize(d.text))
        for original, rewritten in DEMO_SYNONYMS.items():
            assert not (set(tokenize(original)) & set(tokenize(rewritten))), f"{original} 与 {rewritten} 有重叠"
            assert not (set(tokenize(rewritten)) & corpus_tokens), f"改写词 {rewritten} 出现在语料中"


class TestExtractiveAnswerer:
    def test_returns_top1_document_text(self):
        docs = build_demo_corpus()
        answerer = ExtractiveAnswerer(BM25Retriever(docs))
        answer = answerer.answer("BM25 词频 打分")
        assert answer
        assert answer in {d.text for d in docs}

    def test_empty_query_returns_empty(self):
        answerer = ExtractiveAnswerer(BM25Retriever(build_demo_corpus()))
        assert answerer.answer("") == ""


class TestRunDemo:
    def test_end_to_end_writes_reports(self, tmp_path):
        data = run_demo(out_dir=tmp_path, num_cases=10, paraphrase_ratio=0.1)
        assert (tmp_path / "eval_report.md").exists()
        assert (tmp_path / "eval_report.html").exists()
        assert len(data.pipelines) == 3
        assert [p.name for p in data.pipelines] == ["bm25", "tf", "hybrid"]
        assert all(0.0 <= p.recall[10] <= 1.0 for p in data.pipelines)
        assert data.corpus_size == 24

    def test_demo_is_deterministic(self, tmp_path):
        first = run_demo(out_dir=tmp_path / "a", num_cases=8, seed=11)
        second = run_demo(out_dir=tmp_path / "b", num_cases=8, seed=11)
        metrics_first = [(p.name, p.mrr, p.ndcg) for p in first.pipelines]
        metrics_second = [(p.name, p.mrr, p.ndcg) for p in second.pipelines]
        assert metrics_first == metrics_second


class TestCli:
    def test_demo_command(self, tmp_path, capsys):
        rc = main(["demo", "--out", str(tmp_path), "--num-cases", "8"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "bm25" in out
        assert "eval_report.md" in out

    def test_default_command_is_demo(self, tmp_path, capsys):
        rc = main(["--out", str(tmp_path), "--num-cases", "6"])
        assert rc == 0
        assert (tmp_path / "eval_report.md").exists()

    def test_num_cases_zero_rejected_with_friendly_error(self, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            main(["demo", "--out", str(tmp_path), "--num-cases", "0"])
        assert excinfo.value.code == 2

    def test_num_cases_negative_rejected(self, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
            main(["demo", "--out", str(tmp_path), "--num-cases", "-3"])
        assert excinfo.value.code == 2

    def test_num_cases_non_integer_rejected(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["demo", "--out", str(tmp_path), "--num-cases", "abc"])
        assert excinfo.value.code == 2
        assert "正整数" in capsys.readouterr().err

    def test_paraphrase_ratio_out_of_range_rejected(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["demo", "--out", str(tmp_path), "--paraphrase-ratio", "1.5"])
        assert excinfo.value.code == 2
        assert "[0, 1]" in capsys.readouterr().err


class TestDatasetWeightedNDCG:
    """--dataset 带 grades 的评测集：weighted NDCG 进入控制台与报告；纯 demo 输出不变。"""

    def test_dataset_with_grades_produces_weighted_ndcg(self, tmp_path, capsys):
        rc = main(["--dataset", str(EXAMPLES_DATASET), "--out", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Weighted NDCG@10" in out
        md = (tmp_path / "eval_report.md").read_text(encoding="utf-8")
        html = (tmp_path / "eval_report.html").read_text(encoding="utf-8")
        assert "Weighted NDCG@10" in md
        assert "分级相关度指标" in md
        assert "Weighted NDCG@10" in html
        assert "未提供 grades 分级标注" not in md

    def test_demo_console_output_has_no_weighted_column(self, tmp_path, capsys):
        # 内置玩具 demo 无 grades：控制台不出现 Weighted NDCG@10 列，行为与既往一致
        rc = main(["demo", "--out", str(tmp_path), "--num-cases", "6"])
        assert rc == 0
        assert "Weighted NDCG@10" not in capsys.readouterr().out


class TestCorpusDir:
    """--corpus-dir 外部语料：与 --dataset/--fail-under 组合、报错口径与报告溯源。"""

    def test_corpus_dir_with_dataset_end_to_end(self, tmp_path, capsys):
        corpus = write_external_corpus(tmp_path)
        dataset = write_jsonl_dataset(
            tmp_path,
            [
                {"_meta": {"corpus": "外部冒烟语料"}},
                {"query_id": "q1", "query": "什么是检索增强生成？", "relevant_doc_ids": ["rag#000"]},
                {
                    "query_id": "q2",
                    "query": "BM25 为什么无需训练？",
                    "relevant_doc_ids": ["rag#001"],
                    "grades": {"rag#001": 2},
                },
                {"query_id": "q3", "query": "重排序为什么能提升排序质量？", "relevant_doc_ids": ["eval#000"]},
            ],
        )
        out = tmp_path / "reports"
        rc = main(
            [
                "--corpus-dir", str(corpus), "--dataset", str(dataset),
                "--out", str(out), "--fail-under", "recall_at_5=0.3",
            ]
        )
        assert rc == 0
        assert (out / "eval_report.md").exists()
        assert (out / "eval_report.html").exists()
        captured = capsys.readouterr()
        assert "已加载外部语料" in captured.out
        assert "Weighted NDCG@10" in captured.out
        assert "评测回归门禁通过" in captured.out
        assert "警告" not in captured.err
        report = (out / "eval_report.md").read_text(encoding="utf-8")
        assert f"语料来源：外部目录 {corpus}（2 个文件，3 个文本块）" in report
        assert "数字由外部语料实测得出" in report
        assert "内置 24 段玩具语料" not in report

    def test_corpus_dir_alone_synthesizes_cases(self, tmp_path, capsys):
        corpus = write_external_corpus(tmp_path)
        out = tmp_path / "reports"
        rc = main(["--corpus-dir", str(corpus), "--out", str(out), "--num-cases", "3"])
        assert rc == 0
        assert (out / "eval_report.md").exists()
        assert "语料文档数: 3" in capsys.readouterr().out
        report = (out / "eval_report.md").read_text(encoding="utf-8")
        assert f"语料来源：外部目录 {corpus}" in report
        assert "数字由外部语料实测得出" in report
        assert "内置 24 段玩具语料" not in report

    def test_dataset_unknown_doc_id_warns_against_external_corpus(self, tmp_path, capsys):
        corpus = write_external_corpus(tmp_path)
        dataset = write_jsonl_dataset(
            tmp_path,
            [{"query_id": "q1", "query": "什么是检索增强生成？", "relevant_doc_ids": ["rag#000", "d99"]}],
        )
        rc = main(["--corpus-dir", str(corpus), "--dataset", str(dataset), "--out", str(tmp_path)])
        captured = capsys.readouterr()
        assert rc == 0
        assert "外部语料中不存在的 doc_id" in captured.err
        assert "d99" in captured.err
        assert "bm25" in captured.out

    def test_corpus_dir_with_failing_gate_exits_1(self, tmp_path, capsys):
        corpus = write_external_corpus(tmp_path)
        dataset = write_jsonl_dataset(
            tmp_path,
            [{"query_id": "q1", "query": "什么是检索增强生成？", "relevant_doc_ids": ["rag#000", "d99"]}],
        )
        rc = main(
            [
                "--corpus-dir", str(corpus), "--dataset", str(dataset),
                "--out", str(tmp_path), "--fail-under", "recall_at_1=1.0",
            ]
        )
        assert rc == 1
        assert "门禁未达标" in capsys.readouterr().err

    def test_empty_corpus_dir_exits_2(self, tmp_path, capsys):
        empty = tmp_path / "empty"
        empty.mkdir()
        rc = main(["--corpus-dir", str(empty), "--out", str(tmp_path)])
        assert rc == 2
        assert "没有 .md/.txt 文件" in capsys.readouterr().err

    def test_missing_corpus_dir_exits_2(self, tmp_path, capsys):
        rc = main(["--corpus-dir", str(tmp_path / "nope"), "--out", str(tmp_path)])
        assert rc == 2
        assert "语料目录加载失败" in capsys.readouterr().err

    def test_corpus_dir_without_md_txt_exits_2(self, tmp_path, capsys):
        only_py = tmp_path / "pyonly"
        only_py.mkdir()
        (only_py / "x.py").write_text("print(1)", encoding="utf-8")
        rc = main(["--corpus-dir", str(only_py), "--out", str(tmp_path)])
        assert rc == 2
        assert "没有 .md/.txt 文件" in capsys.readouterr().err

    def test_corpus_dir_blank_files_exits_2(self, tmp_path, capsys):
        blank = tmp_path / "blank"
        blank.mkdir()
        (blank / "a.md").write_text("", encoding="utf-8")
        rc = main(["--corpus-dir", str(blank), "--out", str(tmp_path)])
        assert rc == 2
        assert "没有切出任何文本块" in capsys.readouterr().err

    def test_blank_corpus_dir_value_rejected(self, tmp_path):
        rc = main(["--corpus-dir", "", "--out", str(tmp_path)])
        assert rc == 2

    def test_paraphrase_ratio_conflicts_with_corpus_dir(self, tmp_path, capsys):
        corpus = write_external_corpus(tmp_path)
        rc = main(["--corpus-dir", str(corpus), "--out", str(tmp_path), "--paraphrase-ratio", "0.5"])
        assert rc == 2
        assert "互斥" in capsys.readouterr().err
