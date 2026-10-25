"""校验 dataset_projects.jsonl：严格加载 + doc_id 一致性 + 分布统计。"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
from llm_eval_kit import load_corpus_from_dir, load_dataset

ds = load_dataset("examples/dataset_projects.jsonl")
print("queries:", len(ds.to_eval_cases()))

corpus = load_corpus_from_dir("examples/corpus_projects")
corpus_ids = {d.doc_id for d in corpus}
unknown = ds.unknown_doc_ids(corpus_ids)
print("unknown doc refs:", unknown if unknown else "NONE - all doc_ids verified")

s = ds.stats()
print(s.describe())
