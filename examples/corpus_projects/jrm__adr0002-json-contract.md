# ADR-0002: --json 输出重塑为紧凑契约，不做全量 model_dump

- 状态：已采纳
- 日期：2026-10-04
- 关联代码：`cli.py`、`matcher/models.py`、`tests/test_streamlit_app.py`

## 背景

`--json` 最初直接 `model_dump()` 倾倒整个 MatchResult——字段全集、
层级随内部模型走，内部重构会静默破坏下游消费者；且批量条目携带
全量 breakdown，机器消费时大部分字段是噪声。

## 决策

1. `--json` 输出手写紧凑契约并写进 README：
   单份 `{result: {total_score, grade, grade_label, missing_required_skills,
   semantic_enabled, provider, degraded_note?, breakdown[], reasons[]}}`，
   breakdown 贡献 = score × effective_weight（合计即总分），
   reasons[].evidence[] 携带 `{source, snippet, start, end}`
   （测试断言 `原文[start:end] == snippet` 可回溯）；
2. 批量条目只留 `{file, total_score, grade, grade_label,
   missing_required_skills}` 或 `{file, error}`，排序与文本模式一致；
3. stdout 保证纯 JSON（提示/告警一律 stderr），`--json` 与 `--csv` 互斥退出码 2。

## 代价

- 一次性破坏了依赖旧全量形态的消费者（当时尚无外部消费者，窗口期唯一）；
- 新增核心字段时需要显式决定是否进契约——比"自动全量"多一步，
  但这正是契约的意义。
