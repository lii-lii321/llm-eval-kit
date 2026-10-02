"""llm-eval-kit：离线可跑的 RAG 检索与 LLM 应用评测工具库。"""

from .attribution import (
    REASON_CORPUS_MISSING,
    REASON_KEYWORD_MISMATCH,
    REASON_SEMANTIC_DRIFT,
    Badcase,
    attribute_badcases,
    summarize_attribution,
)
from .corpus import Doc, load_corpus_from_dir, split_text
from .judge import (
    DIMENSIONS,
    DIMENSIONS_ZH,
    JudgeProvider,
    JudgeScore,
    JudgeSummary,
    LLMJudgeError,
    LLMUnavailable,
    MockJudge,
    OpenAICompatibleJudge,
    judge_summary,
)
from .metrics import (
    ConfidenceInterval,
    LatencyStats,
    PipelineMetrics,
    average_precision,
    bootstrap_ci,
    latency_stats,
    mrr,
    ndcg_at_k,
    percentile,
    recall_at_k,
)
from .pipeline import run_evaluation, run_retrieval
from .report import ReportData, render_html, render_markdown, write_reports
from .retrieval import BaseRetriever, BM25Retriever, HybridRetriever, RetrievalResult, TFRetriever
from .synth import EvalCase, extract_phrases, generate_eval_set
from .tokenize import tokenize

__version__ = "0.1.0"

__all__ = [
    "Badcase",
    "BaseRetriever",
    "BM25Retriever",
    "ConfidenceInterval",
    "DIMENSIONS",
    "DIMENSIONS_ZH",
    "Doc",
    "EvalCase",
    "HybridRetriever",
    "JudgeProvider",
    "JudgeScore",
    "JudgeSummary",
    "LatencyStats",
    "LLMJudgeError",
    "LLMUnavailable",
    "MockJudge",
    "OpenAICompatibleJudge",
    "PipelineMetrics",
    "REASON_CORPUS_MISSING",
    "REASON_KEYWORD_MISMATCH",
    "REASON_SEMANTIC_DRIFT",
    "ReportData",
    "RetrievalResult",
    "TFRetriever",
    "__version__",
    "attribute_badcases",
    "average_precision",
    "bootstrap_ci",
    "extract_phrases",
    "generate_eval_set",
    "judge_summary",
    "latency_stats",
    "load_corpus_from_dir",
    "mrr",
    "ndcg_at_k",
    "percentile",
    "recall_at_k",
    "render_html",
    "render_markdown",
    "run_evaluation",
    "run_retrieval",
    "split_text",
    "summarize_attribution",
    "tokenize",
    "write_reports",
]
