import dataclasses

import pytest

from llm_eval_kit.corpus import Doc, load_corpus_from_dir, split_text


class TestDoc:
    def test_fields(self):
        doc = Doc(doc_id="d1", text="内容", source="a.txt")
        assert doc.doc_id == "d1"
        assert doc.text == "内容"
        assert doc.source == "a.txt"

    def test_source_optional(self):
        assert Doc(doc_id="d1", text="内容").source == ""

    def test_frozen(self):
        doc = Doc(doc_id="d1", text="内容")
        with pytest.raises(dataclasses.FrozenInstanceError):
            doc.doc_id = "d2"


class TestSplitText:
    def test_short_paragraph_single_chunk(self):
        assert split_text("一句话。另一句。") == ["一句话。另一句。"]

    def test_long_paragraph_split_by_sentence(self):
        text = "第一句话内容比较长一点。第二句话内容也不短。第三句话同样很长。第四句收尾。"
        chunks = split_text(text, max_chars=20)
        assert len(chunks) == 3
        assert "".join(chunks) == text
        assert all(len(c) <= 20 for c in chunks)

    def test_blank_line_separates_paragraphs(self):
        text = "第一段。\n\n第二段。"
        assert split_text(text) == ["第一段。", "第二段。"]

    def test_empty_text(self):
        assert split_text("") == []


class TestLoadCorpusFromDir:
    def test_loads_txt_and_md_chunks(self, tmp_path):
        (tmp_path / "a.txt").write_text("第一块。\n\n第二块。", encoding="utf-8")
        (tmp_path / "b.md").write_text("_md_ 内容唯一。", encoding="utf-8")
        docs = load_corpus_from_dir(tmp_path)
        ids = {d.doc_id for d in docs}
        assert "a#000" in ids and "a#001" in ids
        assert "b#000" in ids

    def test_ignores_other_suffixes(self, tmp_path):
        (tmp_path / "a.txt").write_text("内容。", encoding="utf-8")
        (tmp_path / "b.py").write_text("print(1)", encoding="utf-8")
        docs = load_corpus_from_dir(tmp_path)
        assert [d.doc_id for d in docs] == ["a#000"]

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            load_corpus_from_dir(tmp_path / "nope")
