# AI Data Research Lab · 立项书

> 2026-09-28 定盘。repo 名建议 `research-lab`。
> 核心原则：AI 只负责"提出计划、解释结果"，所有数字必须来自真实执行的代码。
>
> **2026-09-30 修订**：执行策略已从原稿的"LLM 生成代码 + 受限子进程沙箱"改为
> **方法白名单 + 确定性 scipy 分派**，决策记录见
> [docs/adr-0001-method-whitelist-over-codegen-sandbox.md](docs/adr-0001-method-whitelist-over-codegen-sandbox.md)。
> 下文涉及"受限子进程/Docker 沙箱"的表述为原稿保留，以 ADR 为准。

## 一句话

从"拿到一份 CSV"到"产出一份可复现的研究报告"：
数据画像 → 研究问题 → 实验计划 → 真实执行 → 统计/ML → 实验追踪 → 报告导出。

## 为什么是它

- 项目1 Math_Tutor_RAG 证明 AI / RAG / Agent；项目2 smart_tutor 证明后端 / SaaS / 系统设计
- 第三个补上数据科学工程能力（EDA、统计检验、特征工程、模型评估、实验追踪），三项目形成「AI + 后端 + 数据科学」画像，匹配数据科学专业身份
- 公开赛道"又一个 RAG tutor"已饱和，科研分析平台更容易做出深度和辨识度

## 明确不做（反范围）

- 不做用户系统、支付、多角色、多租户
- Phase 1 不做 Agent、不做论文 RAG、不做协作功能
- 不堆大而全 Dashboard——每个阶段只回答一个明确问题：
  「AI 能否自动完成一套可复现的数据科学实验？」

## Phase 1 · MVP（目标 4-6 周）

1. 上传 CSV/Excel → **Dataset Profiler**：行列数、类型分布、缺失率、重复率、异常值、高相关列预警、类别不平衡
2. LLM 生成**研究问题**（RQ 列表，附选择理由）
3. 每个问题生成**实验计划**：假设、方法、H0、H1、显著性水平 α
4. **受控 Python 执行**：生成分析代码 → 沙箱真实运行 → 得到 r / p 值等真实结果
5. LLM 只做**结果解释**（禁止编造数字）
6. 统计检验 + matplotlib 可视化 → 导出 Markdown / HTML 研究报告

## Phase 2 · ML + 实验追踪

- 回归 / 分类 / 聚类自动基线（Linear、Logistic、RF、XGBoost、LightGBM）
- Experiment Tracking：数据集版本、特征集、模型、超参、指标、代码、时间戳
- 实验历史对比视图

## Phase 3 · Research Agent

- Agent + 工具集（Data / Stats / ML / Literature）
- 一句话研究任务 → 自动走完全流程
- Paper RAG + 自动生成技术报告

## 技术栈（对齐 AGENTS.md 默认）

- FastAPI（async）+ Streamlit
- pandas / scipy / statsmodels / scikit-learn / XGBoost / LightGBM
- LLM 调用层复用 Math_Tutor_RAG 已有经验
- 沙箱：Phase 1 受限子进程（白名单库 + 资源限制），Phase 2 起换 Docker
- SQLite 起步；pytest + GitHub Actions CI

## 数据

MVP 用 Kaggle 公开数据集验证（如学生成绩 student performance），另备 2-3 个不同领域数据集做演示。

## 前置收尾（开工前完成）

1. MathMaster：PG / Tutor 真实评测 + 分享卡（v2.9.0 已 248 tests CI 绿，就差这两件）
2. D:\My_Project\Math_Tutor_RAG 旧版 7 文件决断：归档或重建，了断掉

## 成功标准（做到 90% 而不是 70%）

- 每个分析结果可溯源到真实执行的代码与输出
- 同一数据集重跑得到一致结论（可复现）
- 报告含完整实验记录：假设 → 方法 → 结果 → 解释
- pytest 覆盖核心链路，CI 绿
- README 一页讲清：问题、架构、demo GIF

## 路线节奏

| 周 | 交付 |
|---|---|
| 第 0 周 | 收尾 MathMaster 两件 + D:\ 旧版决断 |
| 第 1 周 | repo 骨架 + Dataset Profiler（不依赖 LLM 可独立运行） |
| 第 2 周 | LLM 研究问题生成 + 实验计划生成 |
| 第 3 周 | 受控执行器 + 统计检验落地 |
| 第 4 周 | 可视化 + 报告导出，MVP 闭环 |
| 第 5-6 周 | 打磨、演示数据集、README + demo GIF |
