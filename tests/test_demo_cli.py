from llm_eval_kit.cli import main
from llm_eval_kit.demo import DEMO_DOCS, DEMO_SYNONYMS, ExtractiveAnswerer, build_demo_corpus, run_demo
from llm_eval_kit.retrieval import BM25Retriever


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
