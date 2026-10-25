# ADR-0001: 技能词表三入口同语义——CLI/API/Demo 共享一套校验与合并

- 状态：已采纳
- 日期：2026-10-04
- 关联代码：`matcher/taxonomy.py::merged_skill_taxonomy`、`cli.py::--vocab`、`matcher/main.py::custom_vocab`、`streamlit_app.py` 词表上传

## 背景

技能词表外置后出现三个叠加入口：CLI `--vocab`（文件路径）、API `custom_vocab`
（请求体内联 JSON）、Streamlit Demo（上传控件）。若各自实现解析与合并，
三个入口的匹配语义必然漂移，且不可信入口（API/上传）缺少资源上限会被
超大词表拖垮单次匹配。

## 决策

1. 校验、合并语义、资源上限收敛到 `taxonomy.py` 单点：
   `merged_skill_taxonomy(用户映射)`——按规范名合并（用户条目整体覆盖、
   新条目追加），上限 500 条目 / 单技能 50 别名 / 单别名 80 字符；
2. CLI 的文件路径入口先 `parse_vocab_document` 校验再走同一合并函数；
   API 与 Demo 把内联对象直接交给 `merged_skill_taxonomy`；
3. 校验失败显式报错（CLI 退出码 2 / API 422 / Demo 显式提示），
   绝不静默放行半份词表。

## 代价

- 三入口的能力上限一致意味着 CLI 也要受 500/50/80 约束——
  超大本地词表需走库接口直接构造 taxonomy；
- 新叠加入口（如未来 MCP 工具）必须复用同一函数，代码评审需守住这一点。
