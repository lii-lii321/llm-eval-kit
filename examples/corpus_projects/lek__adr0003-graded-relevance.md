# ADR-0003: 分级相关度只进 Weighted NDCG——Recall/MRR 维持二元口径

- 状态：已采纳
- 日期：2026-10-04
- 关联代码：`src/llm_eval_kit/metrics.py`、`src/llm_eval_kit/dataset.py`、`docs/datasets.md`

## 背景

JSONL 标注评测集支持 `grades`（0/1/2 分级相关度）后，需要决定分级信息
进入哪些指标。Recall@K、MRR 的教科书定义建立在二元相关度上；
分级扩展（如按增益加权的 Recall、分级 MRR）缺乏统一口径，自创定义
会破坏与外部基准的可比性。

## 决策

1. 分级信息只产出 **Weighted NDCG@10**（指数增益 2^g − 1，理想 DCG 按
   该查询全部相关文档排序），与二元 NDCG@10 并列、命名区分；
2. Recall@K / MRR / MAP@10 维持二元口径：`relevant_doc_ids` 中任一文档
   命中即算命中，不看 grade；
3. 无 `grades` 的评测集不产出 Weighted NDCG@10——报告显示 "—"、门禁报不可用，
   绝不以 0 值冒充"测过了"；demo（合成、无 grades）控制台因此不出现该列。

## 代价

- 用户若期望"分级信息全面提升所有指标"会失望——需要分级 Recall 时
  只能自定义实现；
- 报告列随评测集形态动态增减，比较两份报告时需注意列对齐。
