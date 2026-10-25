# Changelog

## 1.10.0 - 2026-10-25

打包化与 CI 矩阵：

- 新增 pyproject.toml：`pip install -e .` 可用，依赖与 dev extras 声明齐备，动态版本读取 utils/version.py（保持单一来源）
- 新增控制台命令 `research-lab`（等价于 python cli.py）
- CI test job 升级为 Python 3.10/3.11/3.12/3.13 矩阵（lint+mypy 仍只在 3.10 执行）；新增 package job 验证打包安装路径与源码树外导入
- 222 测试全绿

## 1.9.0 - 2026-10-05

落实 UX 手册 S1.2 环境体检 + 报告库筛选：

- scripts/doctor.py：一条命令体检 Python 版本/依赖/LLM Key/数据目录/端口占用，退出码 0/1，含修复建议
- 报告库按数据集前缀筛选（报告名通常以数据集名开头）
- CLI reports --export 补测试与用例覆盖
- 222 测试全绿

## 1.8.0 - 2026-10-04

## 1.8.0 - 2026-10-04

落实 UX 手册 S2.3 大文件采样画像：

- profile_dataset 新增 sample 参数：超过 50 万行自动均匀采样（种子 42），overview.sampled/sample_note 标注
- 分析流程页采样时显示 ⚡ 声明（统计结论基于采样，ML 实验用全量）
- 测试 195 → 198 全绿

## 1.7.0 - 2026-10-04

落实 UX 手册 S7.2/S8.1 收尾：

- 实验追踪：同数据集同任务趋势图标题含数据集名与次数（更清晰定位历史范围）
- CLI reports --export：一条命令把报告打包为 zip 研究包（md+html+assets）
- i18n 中英切换：侧边栏语言选择器（中文/English），UI 骨架文案双语化
- 可达性：全局 CSS img max-width 100%
- ruff E402 豁免 cli.py（sys.path 需先于项目导入）
- 195 测试全绿

## 1.6.0 - 2026-10-04

落实 UX 手册 S6.2/S4.2 收尾：

- RFC 7807 安全实现：APIError 异常类 + install_problem_handler，仅对显式业务错误生效，不影响 FastAPI 默认 HTTPException；测试覆盖 problem+json 响应与正常端点不受影响
- PDF 导出：fpdf2 纯 Python 实现（weasyprint 需 GTK 原生库在 Windows 不可用，按手册预案降级），支持中文字体/标题层级/表格降级渲染
- 195 测试全绿

## 1.5.0 - 2026-10-04

落实 UX 手册 S0 多页面架构重构：

- app.py 从 700+ 行拆为纯导航入口（st.navigation + st.Page），五个页面文件在 pages/ 下独立维护
- 共享组件（时间线/下载卡/字段表/结果图）抽离到 ui/components.py，数据加载与画像缓存抽离到 ui/page_state.py
- AppTest 逐页面测试（7 用例：导航渲染/流程/ML/追踪/报告/Agent/引导卡），不再需要切 tab
- 193 测试全绿，ruff+mypy 零错误

## 1.4.0 - 2026-10-04

落实 UX 手册 S3.1/S4.1/S6.1 收尾：

- plotly 交互图：相关矩阵热图/散点+拟合/箱线图/卡方柱图/重要性条图五张图全部可交互（悬停查看数值、缩放关注区间），UI 全部替换为 st.plotly_chart；matplotlib 版保留给报告 PNG 导出
- 统计前提检查卡：每个实验自动生成结构化前提满足清单（样本量/组间比例/期望频数/残差正态性/异方差/方法适用性），写入 ExperimentResult 并渲染进报告与详情
- zip 研究包导出：报告库三列下载（md/html/zip），CLI reports --export 可用；zip 内含 md+html+assets+MANIFEST
- 人话错误映射测试补齐 10 组正反用例；修复 "观测不足" needle 匹配与 detail 回退逻辑
- 修 smoke 超时：wait_ready 超时从 40/60s 提升至 60/90s
- 测试 177 → 190 全绿

## 1.3.0 - 2026-10-04

落实 UX 手册 S6.3/S4.1/S3.1 + 前提检查卡：

- 统计前提检查卡：每个实验自动附带结构化前提满足清单（样本量/变量类型/组间比例/期望频数/残差正态性与异方差），写入 ExperimentResult.extra 并渲染进报告
- zip 研究包导出：reports_store.export_bundle 打包 md+html+assets 为 zip，CLI --export 与 API 均可调用
- plotly 交互图表加入 requirements（UI 逐步替换 matplotlib）
- 移除误加的 RFC 7807 全局 exception handler（FastAPI HTTPException 已有标准格式）
- 测试 179 → 177（移除 2 个因前提检查新增行为而需重写的重复断言，无净损失）

## 1.2.0 - 2026-10-03

落实 UX 手册反馈批次（S2.2/S5.1/S5.2/S7.1）：

- ML 训练分步进度：st.status 实时显示每个模型训练完成情况（run_ml_experiment 新增 progress_cb）
- Agent 分步进度：每步完成即实时渲染时间线（run_research_agent 新增 on_step 回调）
- 数据接入扩展：侧边栏新增粘贴表格（st.data_editor）与三领域数据集画廊（学生成绩/门店销售/医疗随访）；URL 拉取按手册设计默认关闭
- 刷新恢复：示例数据模式写入 st.query_params，刷新自动重载
- 测试 177 → 179 全绿

## 1.1.0 - 2026-10-03

落实 UX 手册快赢批次（S1.1/S1.3/S2.1/S6.1/S6.4）：

- 零步骤启动：示例数据运行时自愈生成（新克隆不跑脚本即可用），CI 不再需要前置数据生成，UI 界面测试在全新检出不再跳过
- 画像缓存：按数据指纹哈希缓存 profile 结果，同数据 rerun 零重算
- 人话错误卡：七类高频错误（格式/编码/超限/变量不匹配/样本不足/LLM 异常/记录不存在）渲染为「发生了什么/为什么/你可以试」卡片，UI 与 API 同源
- 首次引导卡（可关闭）与 CLI --version / analyze 下一步指引
- 测试 176 → 179 全绿（界面测试不再有 skip）

## 1.0.0 - 2026-10-02

**里程碑版本：落实手册 A–H 全部条目完成。**

- 论文复现专栏 reproductions/：第一篇复现 Student (1908) 配对 t 检验（Cushny-Peebles 数据，t = 4.0621 vs 论文 4.06，✅ 一致），含运行脚本/对照表/偏差说明/复现笔记
- 门面：新增英文版 README.en.md（主打三道闸辨识度）、README 顶部 mermaid 架构图与中英切换
- 测试 176 → 176 全绿（C1 为运行验证非单测）
- demo GIF 需交互式终端录制工具（VHS/asciinema），留待后续

## 0.10.0 - 2026-10-02

落实手册 v0.10.0 批次（E1/F1/F2/G1）：

- golden 评测集：18 个变量类型×方法组合案例，规则计划引擎逐一命中（进 CI 主 job 作为回归防线）；新增 scripts/eval_llm.py 输出 LLM 计划与规则基线一致率（需 Key 手动运行）
- 报告配图：画像章节嵌入分布直方图网格与缺失模式矩阵（HTML base64 自包含，MD 随报告落盘 assets），新增 services/report_images.py 收集器
- 八类质量告警逐条附处置建议（WarningItem.suggestion + 映射表），报告与界面同步
- 因果提示：显著结论自动列出潜在混杂与随机化/匹配建议；计划层新增数值≥2 且含类别列 → 线性回归的规则分支
- 测试 157 → 176 全绿

## 0.9.0 - 2026-10-02

落实手册 v0.9.0 批次（B1/B2/E2）：

- ML 三件套：基线最佳模型自动小网格调优（GridSearchCV，网格按模型硬编码；线性模型与 <50 训练样本自动跳过）、joblib 持久化到 data/models/（按追踪 uid 命名）、permutation importance 特征重要性 top10 进报告与界面
- 报告 ML 章节新增调优注记与模型卡片（数据指纹/任务/规模/局限）
- LLM 客户端指数退避重试：429/5xx 与传输异常最多 3 次（secrets 抖动），golden 测试覆盖 503-恢复与耗尽两路径
- 测试 152 → 157 全绿

## 0.8.0 - 2026-10-02

落实手册 v0.8.0 批次（D1/D2/D3/A1）：

- 工程门禁：CI 新增 ruff、mypy（29 文件零错误）、pytest-cov 覆盖率门禁 85%（实测 94%）
- 依赖锁定：requirements.in + pip-compile 生成锁定版 requirements.txt
- 统计闭环（statsmodels）：相关系数 Fisher z 95% CI；Welch/t 均值差 95% CI；线性回归 OLS 诊断（调整 R²、F、斜率 CI、Shapiro、Breusch-Pagan）；实验计划自动附功效分析（所需样本量），报告展示 95% CI 与所需样本量
- 测试 145 → 152 全绿；README 用例数漂移修正

## 0.7.1 - 2026-10-01

- **执行器退化样本防护**：kruskal 在每组仅 1 条观测时 H/p 仍有效但 ε² 分母为 0——效应量
  优雅置空而非 ZeroDivisionError；全同值样本 scipy 抛 ValueError——统一转为干净的
  `failed` 结果（附原因），API 不再 500；+2 退化场景测试（147 全绿）

## 0.7.0 - 2026-09-30

产品化一轮：从「目的 / 场景 / 方法」三个角度补齐使用体验。

- **场景 · 报告库**：报告自动沉淀到 `data/reports/`（跨会话可回看），新增 `GET /api/reports`、`GET /api/reports/{name}` 与 Streamlit ⑤ 报告库标签页；分析流程、自动研究、CLI 产出的报告统一汇集
- **场景 · 一键体验**：侧边栏「🚀 一键体验完整流程」——上传前即可用示例数据跑通全流程，内嵌展示执行时间线与报告
- **方法 · 命令行**：新增 `cli.py`——`profile`（30 秒看懂一份数据）/ `analyze`（一句话任务、可选 `--skip-ml` `--no-literature` `--no-llm`）/ `experiments` / `reports`，可接进终端与脚本管道
- **方法 · Python API**：`examples/api_demo.py` 演示在 Notebook 中组合服务层
- **文档**：README 新增「为谁设计 / 三个使用入口」与跨平台说明
- 测试 135 → 145 全绿

## 0.6.0 - 2026-09-30

- 监督学习基线升级：分层 k 折交叉验证（k=5/3 按样本量自适应，分类按 y 分层），每模型报告 CV 均值±标准差，最佳模型按 CV 均值选取而非单次 holdout；测试数据补噪声避免零噪声并列
- 统计实验入 tracking 库（kind=stats：方法/变量/α/原始 p/decision/计划来源），Agent 全流程自动入库；每个成功实验附可独立运行的 scipy 复现脚本（`services/repro.py`），嵌入报告"附录：复现脚本"并可单独下载
- task_description 真正驱动研究问题：LLM 模式作为硬约束写入 prompt 首行；规则模式按任务关键词对问题重排序
- 计划层加固：方法白名单与执行层单一来源（`RUNNER_METHODS`）、方法-变量类型组合校验（LLM 给 chi2 配两个数值变量会被规则降级并留痕）、偏度感知兜底（|skew|≥2 → 秩方法）
- 安全与部署加固：上传先查声明大小再读内存（413）、CORS 默认仅本地 Streamlit（`RESEARCH_LAB_CORS` 可覆盖）、compose 仅回环绑定、SQLite 连接超时
- CI/工具链：新增 smoke job（CI 真实起 uvicorn + Streamlit 做九项探活）；`scripts/smoke_check.py` 跨平台化（当前解释器 + 轮询就绪探测替代固定 sleep）；新增 `scripts/wait_for.py`（仅限回环地址的就绪等待）
- 文档：README 重构为"问题→机制→架构→上手"一页式；新增 ADR-0001 记录"白名单分派取代代码生成沙箱"的方法学决策；.env.example 与 PROJECT_BRIEF 过时口径修正
- 测试 124 → 135 全绿 0 跳过

## 0.5.1 - 2026-09-30

修复独立评审核实的 5 个问题：

- 测试真实有效性：CI 构建时先生成示例数据（UI 测试不再在全新检出时整体跳过）、`pytest -rs` 让 skip 透明、ML 流程测试隔离追踪库（不再污染本地实验记录）、删除两条恒真断言（`test_ml_lab.py` 的运算符优先级错误与 `test_agent.py` 的恒真状态断言）
- 统计严谨性：H0 判定改用未舍入的原始 p 值（真实 p=0.04996 不再被舍入成 0.05 而翻转结论，`p_value_raw` 入库）；2×2 稀疏列联表自动切换 Fisher 精确检验（`test_used` 标注实际方法）；Mann-Whitney/Kruskal 补齐 rank-biserial r 与 ε² 效应量；报告结论表新增 Benjamini–Hochberg 校正后 p 列，局限声明注明全部结论为探索性分析，稀疏卡方在结论表标注（稀疏）
- LLM 客户端补齐单测：非 200、畸形响应、断网、无 Key 四分支 + auto 模式 LLMError 规则回退（此前该层零覆盖）
- research 三个路由改 `run_in_threadpool`：LLM 调用最长 60 秒不再阻塞事件循环（对齐 agent/ml 路由的既有模式）

## 0.5.0 - 2026-09-29

- Paper RAG：`services/literature.py` —— 从目标候选与研究问题自动构造检索词，arXiv API 关键词检索 + 词面重排（零嵌入、零 Key），返回 top-N 相关文献
- Agent 新增"文献检索"步骤（时间线可见、失败优雅降级不阻断报告），报告新增"相关工作（文献引用）"章节（MD+HTML，摘要截断、作者折叠、来源声明"非系统性综述"）
- `POST /api/agent/run` 新增 `with_literature` 参数；Streamlit ④ 自动研究页新增开关与引用文献列表
- 测试 102 → 114（fetch 可注入，全部离线可跑；真实 arXiv 由冒烟脚本覆盖）
- 版本口径：文献检索需联网，无网络时自动跳过该步骤并在时间线标注原因

## 0.4.0 - 2026-09-29

- Research Agent：一句话研究任务自动走完 画像 → 研究问题 → 统计实验 → ML 基线 → 研究报告 全流程，带执行时间线（`POST /api/agent/run`，Streamlit ④ 自动研究页）
- 规则化研究问题生成：未配置 LLM 时按目标候选 × 变量类型确定性提出 RQ，零 Key 可跑通全链路；RQ/实验计划/结果解读均带来源标注（llm / rule）
- 报告生成器新增 ML 基线章节（模型对比表、最佳模型、排除特征），章节号动态编排
- 实验追踪深化：`GET /api/experiments/{uid}` 详情、界面行选中查看 JSON、同数据集同任务最佳指标趋势图
- Docker 部署：Dockerfile + docker-compose（api + ui 双服务，data 卷持久化追踪库）
- 测试 97 → 102（含 Streamlit AppTest 界面级冒烟：ML 流程与 Agent 流程真实点击验证）

## 0.3.0 - 2026-09-29

- ML 实验室：回归 / 分类 / 聚类多模型基线（Linear / Logistic / RandomForest / XGBoost / LightGBM），特征守门（排除标识符与高缺失列），任务自动推断，种子 42 可复现
- Experiment Tracking：SQLite 存储数据指纹 / 特征集 / 超参 / 指标 / 耗时，`GET /api/experiments`
- Streamlit 三标签页结构（分析流程 / ML 实验室 / 实验追踪）
- 测试 81 → 97

## 0.2.0 - 2026-09-29

- Phase 1 MVP 闭环：受控统计执行器（10 种 scipy 检验 + 效应量 + H0 决策判定），LLM 结果解读只允许使用真实数字
- 研究报告导出（Markdown + HTML，含复现说明与局限章节），实验历史管理
- 相关矩阵热力图，图表逻辑抽离 `utils/charts.py`
- 测试 41 → 81

## 0.1.0 - 2026-09-28

- 项目启动：Dataset Profiler（类型推断 / 缺失重复 / 离群 / 高相关 / 类别不平衡 / 目标候选）
- FastAPI + Streamlit 双入口，pytest 测试与冒烟脚本














