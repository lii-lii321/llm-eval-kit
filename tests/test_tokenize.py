from llm_eval_kit.tokenize import tokenize


class TestTokenize:
    def test_ascii_words_lowercased(self):
        assert tokenize("BM25 Retrieval") == ["bm25", "retrieval"]

    def test_cjk_character_bigrams(self):
        assert tokenize("检索评测") == ["检索", "索评", "评测"]

    def test_mixed_text(self):
        tokens = tokenize("用BM25做检索")
        assert "bm25" in tokens
        assert "检索" in tokens

    def test_single_cjk_char_kept(self):
        assert tokenize("卡") == ["卡"]

    def test_digits_and_versions(self):
        tokens = tokenize("GPT-4o v2")
        assert "gpt" in tokens
        assert "4o" in tokens
        assert "v2" in tokens

    def test_single_ascii_letter_dropped(self):
        assert tokenize("a b cd") == ["cd"]

    def test_punctuation_ignored_and_lowercased(self):
        tokens = tokenize("RAG，是检索方案！")
        assert "rag" in tokens
        assert "检索" in tokens

    def test_empty_string(self):
        assert tokenize("") == []
