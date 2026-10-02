import pytest

from llm_eval_kit import synth as synth_module
from llm_eval_kit.synth import extract_phrases, generate_eval_set


class TestExtractPhrases:
    def test_ascii_words_lowercased(self):
        phrases = extract_phrases("BM25 and RAG")
        assert phrases[:3] == ["bm25", "and", "rag"]

    def test_cjk_run_split_by_stop_chars(self):
        phrases = extract_phrases("基于词频的检索算法")
        assert "基于词频" in phrases
        assert "检索算法" in phrases

    def test_long_run_truncated_to_max_len(self):
        phrases = extract_phrases("这是一段没有任何虚词可以切分的超长中文串")
        assert all(len(p) <= 6 for p in phrases)
        assert "一段没有任何" in phrases

    def test_preserves_duplicates_and_order(self):
        phrases = extract_phrases("向量数据库。向量数据库")
        assert phrases.count("向量数据库") == 2
        assert phrases[0] == "向量数据库"

    def test_pure_digit_words_dropped(self):
        phrases = extract_phrases("叠加 10% 到 20% 的窗口重叠，分块策略很重要")
        assert "10" not in phrases
        assert "20" not in phrases
        assert "分块策略" in phrases

    def test_empty_text(self):
        assert extract_phrases("") == []


class TestGenerateEvalSet:
    def test_respects_num_cases(self, toy_docs):
        cases = generate_eval_set(toy_docs, num_cases=2, seed=1)
        assert len(cases) == 2

    def test_caps_at_available_docs(self, toy_docs):
        cases = generate_eval_set(toy_docs, num_cases=20, seed=1)
        assert len(cases) <= len(toy_docs)

    def test_deterministic_with_seed(self, toy_docs):
        a = generate_eval_set(toy_docs, num_cases=3, seed=7)
        b = generate_eval_set(toy_docs, num_cases=3, seed=7)
        assert [(c.qid, c.query, c.relevant_ids, c.answer) for c in a] == [
            (c.qid, c.query, c.relevant_ids, c.answer) for c in b
        ]

    def test_relevant_ids_exist_in_corpus(self, toy_docs):
        ids = {d.doc_id for d in toy_docs}
        for case in generate_eval_set(toy_docs, num_cases=3, seed=2):
            assert set(case.relevant_ids) <= ids

    def test_query_and_answer_nonempty(self, toy_docs):
        for case in generate_eval_set(toy_docs, num_cases=3, seed=3):
            assert case.query.strip()
            assert case.answer.strip()

    def test_qid_unique_and_ordered(self, toy_docs):
        cases = generate_eval_set(toy_docs, num_cases=3, seed=4)
        qids = [c.qid for c in cases]
        assert len(set(qids)) == len(qids)
        assert qids == sorted(qids)

    def test_queries_deduplicated(self, toy_docs):
        cases = generate_eval_set(toy_docs, num_cases=20, seed=5)
        queries = [c.query for c in cases]
        assert len(set(queries)) == len(queries)

    def test_paraphrase_flag_set(self, toy_docs):
        synonyms = {"bm25": "模糊匹配打分", "向量数据库": "卡片盒子", "微调": "继续训练"}
        cases = generate_eval_set(toy_docs, num_cases=6, seed=3, paraphrase_ratio=1.0, synonym_map=synonyms)
        assert cases, "toy 语料应至少产出一条用例"
        assert any(c.meta.get("paraphrased") for c in cases)

    def test_no_paraphrase_by_default(self, toy_docs):
        cases = generate_eval_set(toy_docs, num_cases=6, seed=3)
        assert not any(c.meta.get("paraphrased") for c in cases)

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError):
            generate_eval_set([], num_cases=3)

    def test_invalid_num_cases_raises(self, toy_docs):
        with pytest.raises(ValueError):
            generate_eval_set(toy_docs, num_cases=0)

    def test_reference_answer_prefers_keyword_sentences(self):
        text = "第一句讲别的事情。BM25 是打分算法。BM25 无需训练。结尾一句。"
        answer = synth_module._reference_answer(text, "BM25")
        assert answer.startswith("BM25 是打分算法。BM25 无需训练。")

    def test_reference_answer_falls_back_to_head(self):
        text = "第一句讲别的事情。第二句也是背景。"
        answer = synth_module._reference_answer(text, "不存在的词")
        assert answer == "第一句讲别的事情。第二句也是背景。"
