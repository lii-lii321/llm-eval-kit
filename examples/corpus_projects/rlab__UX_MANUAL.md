# research-lab 使用体验提升方案与落实手册（UX 1.0 → UX 2.0）

> 基线：v1.0.0（176 tests 绿、双 CI job、五标签页 UI、11 API）
> 编制：2026-10-02 · 视角：**专业开发者 × 真实使用者**，聚焦"用起来的每一步"
> 定位：与 `IMPROVEMENT_MANUAL.md`（能力纵深）互补——那份回答"项目能做什么"，这份回答"用起来爽不爽、能不能被第一次打开的人留住"。

---

## 0. 北极星指标与体验断点盘点

### 北极星指标（每条可实测）

| 指标 | 现状 | 目标 |
|---|---|---|
| TTFV：新用户从克隆到看到第一份报告 | 5 步 + 手动生成数据 + 自行摸索 ≈ 10 分钟 | **≤ 2 分钟**（一键脚本 + 自动示例） |
| 等待可见性：>3s 的操作是否始终有进度反馈 | 部分（spinner，无进度/无分步） | 100%（>3s 必有进度或分步状态） |
| 失败可恢复性：操作失败是否给人话 + 下一步动作 | 部分（裸 str(exc)） | 100%（错误卡 + 建议动作） |
| 刷新生存：浏览器刷新后不丢正在做的事 | ❌ 全丢（session_state） | 报告/实验可恢复 |
| 分享成本：把一份结果发给别人 | 下载单个 md（图会丢） | 一个自包含 HTML 或 zip |

### 断点盘点（按用户旅程）

1. **首启**：克隆后必须手动 `generate_sample.py`，否则 UI 测试静默跳过、界面无数据可玩 → 新用户第一分钟就卡住。
2. **每次 rerun**：切标签页/点按钮都会触发 `profile_dataset` 全量重算（无缓存）——示例数据 0.5s 无感，真实 10 万行数据会明显卡顿。
3. **等待**：ML 训练（秒级~分钟级）只有一个 spinner，看不到"训练到第几个模型"；Agent 全流程同样。
4. **图表**：matplotlib 静态 PNG，不能缩放/悬停看值；与 Streamlit 的交互生态脱节。
5. **报告**：MD 版引用相对路径图片，单发一个 .md 文件图全丢；没有 PDF；没有把"报告+复现脚本+资产"打包带走的一步。
6. **数据接入**：只有文件上传一种方式；没有粘贴、没有数据集画廊。
7. **错误**：接口与 UI 直接显示异常文本，无统一格式、无下一步指引。
8. **会话**：刷新 = 全部重来（数据源选择、已跑实验、报告全部丢失）。
9. **门面**：单文件 app.py 已 600+ 行，继续加 UX 会失控；文案纯中文。

### 明确不做（反范围延续）

用户系统/多租户/实时协作/云端存储——单人本地工具的定位不变（见 PROJECT_BRIEF 反范围承诺）。

### 排期总览

| 版本 | 主题 | 条目 |
|---|---|---|
| UX 1.1（快赢周） | 首启 + 体感 | S1.1 自动示例数据、S1.2 doctor 体检、S2.1 profile 缓存、S6.1 人话错误卡 |
| UX 1.2（反馈周） | 等待与图表 | S2.2 分步进度、S3.1 plotly 交互图、S6.4 CLI 体验 |
| UX 1.3（报告周） | 报告与分享 | S4.1 打包导出 zip、S4.2 PDF、S7.1 刷新恢复 |
| UX 1.4（接入周） | 数据与规模 | S5.1 粘贴数据、S5.2 数据集画廊、S2.3 采样画像 |
| UX 2.0（纵深） | 结构与国际化 | S0 多页面拆分、S6.2 RFC 7807、S8 i18n |

---

## S0. 前置结构：app.py 拆分为多页面应用（UX 前置重构，成本：中）

**为什么是前置**：app.py 已 600+ 行，后面所有 UX 条目都要往里加代码。Streamlit 原生支持 `st.navigation` + `pages/` 多页面，拆分后每个主题的改动都落在新文件里，不互相踩。

**设计**

```
app.py                 # 仅保留：导航注册 + 全局侧边栏（数据源/一键体验/语言）
pages/
  01_分析流程.py       # render_flow 主体
  02_ML实验室.py
  03_实验追踪.py
  04_自动研究.py
  05_报告库.py
ui/
  state.py             # session_state 键名常量与初始化
  components.py        # render_agent_timeline / column_table 等共享组件
```

- `st.navigation([st.Page("pages/01_分析流程.py", title="① 分析流程"), ...])` 替代 `st.tabs`——侧边栏导航在窄屏/长流程下比横排 tab 更稳，且天然解决"标签页不感知其他页数据"的问题。
- 现有 5 个 `render_*` 函数原样搬入对应页面文件，`services/` 层零改动。
- `st.session_state` 键集中在 `ui/state.py` 常量，避免散落字符串（现有 8 个键：rqs/plan/result/history/report/ml_result/agent_result/demo_result/data_sig）。

**改动清单**：app.py（瘦身）、pages/（5 新文件）、ui/（2 新文件）、tests/test_app_smoke.py（AppTest 改为对 `app.main()` 入口跑，断言导航存在）。

**验收**：`pytest -q` 全绿；`streamlit run app.py` 五个页面全部可达；功能与拆分前一致（冒烟脚本 9 项照常过）。

**风险**：AppTest 对多页面应用的 API 兼容——先在分支验证 AppTest 行为再合入。

---

## S1. 首次体验与引导

### S1.1 零步骤示例数据（成本：低）

**现状**：示例 CSV 由脚本生成且不入库，新克隆后侧边栏勾选"使用示例数据"会提示"示例数据不存在"。

**设计**：把"生成示例数据"从构建步骤变成**运行时自愈**——`load_source()` 勾选示例且文件不存在时，直接调用 `utils/sample_data.build_sample_dataframe()` 内存生成（不落盘，也不需要脚本）。`scripts/generate_sample.py` 保留给 CLI 场景。

**改动清单**：app.py `load_source()`（示例分支改为内存构建，try/except 兜底回退提示）；删除 ci.yml 中的 generate_sample 步骤（不再是前置条件）；tests/test_app_smoke.py 的 skipif 条件移除。

**验收**：全新克隆 → `streamlit run app.py` → 勾选示例立即出画像；CI 里 test_app_smoke 不再跳过（skip 计数归零）。

### S1.2 环境体检脚本 doctor（成本：低）

`scripts/doctor.py`：一条命令回答"我能不能跑、缺什么"——Python 版本、依赖可导入、.env 是否配置（LLM 可用性）、示例数据/追踪库/报告目录状态、端口占用。输出 ✅/⚠️ 清单与修复命令。

**验收**：`python scripts/doctor.py` 退出码 0（健康）或 1（列出缺失项）；README 快速开始第一步改为它。

### S1.3 引导卡（成本：低）

首次进入（`st.session_state.get("seen_intro")` 为空）在标题下渲染可关闭的三步引导卡：①选数据 ②生成研究问题 ③一键出报告；关闭状态写入 session。配合 S1.1，新用户 60 秒内可完成第一次完整体验。

---

## S2. 等待反馈与性能体感

### S2.1 画像与 ML 结果缓存（成本：低，收益最高）

**现状**：每次 Streamlit rerun（切 tab、点任意按钮）都重新执行 `profile_dataset(df)`；ML 实验结果有 `data_sig` 失效机制但没有跨 rerun 缓存。

**设计**：`app.py` 中

```python
@st.cache_data(hash_funcs={pd.DataFrame: lambda df: dataframe_fingerprint(df)})
def cached_profile(df) -> ProfileReport:
    return profile_dataset(df)
```

以数据指纹为哈希键（复用 `services.tracking.dataframe_fingerprint`），同数据 rerun 零开销；`ttl` 不设（数据变了指纹就变）。ML 训练函数本身昂贵，保持不缓存（结果已入 tracking 可回看）。

**改动清单**：app.py（cached_profile 包装 + 调用点替换）；imports 加 dataframe_fingerprint。

**验收**：同数据两次切换标签页，第二次画像渲染 <0.2s（`st.cache_data` 命中日志）；数据变更后正常重算。

### S2.2 长任务分步进度（成本：中）

**现状**：ML 训练与 Agent 全流程只有一个 spinner。

**设计**：

- ML：`run_ml_experiment` 增加 `progress_cb: Callable[[str], None] | None = None`，在"特征选择 → 每个模型训练完成 → CV → 调优"节点回调模型名与耗时；UI 侧用 `st.status`（Streamlit ≥1.26）容器实时追加行，结束后展开为结果。
- Agent：`run_research_agent` 已有分步结构——把 `step()` 的回调改为**生成器接口** `run_research_agent_iter(...)` 逐步 yield `AgentStep`，UI 用 `st.status` 边跑边显示（不再等全流程结束才出时间线）；原同步接口保留为迭代器的"收集完再返回"包装，API 层不受影响。

**改动清单**：services/ml_lab.py（progress_cb）、services/agent.py（iter 变体 + 包装函数）、app.py（st.status 渲染）、routers/agent.py（不变，仍用同步包装）、tests（iter 变体单测：逐步产出顺序断言）。

**验收**：UI 实测 Agent 运行期间每完成一步立即可见（而非最后一次性出现）；API 行为与响应结构不变（现有 test_api 全绿）。

**风险**：st.status 组件版本要求——pin Streamlit ≥1.28（requirements 锁定文件同步升级）。

### S2.3 大文件采样画像（成本：中）

**设计**：`services/profiler.py::profile_dataset(df, sample=None)`——行数 > 500,000 时默认均匀采样 50 万行做画像（列级 missing/unique/分位数在采样下稳定），报告明确标注"基于 N=500,000 行分层采样"；ML/统计实验仍用全量（或由用户显式选择）。`utils/uploads.py` 读取后把行数信息带入。

**改动清单**：profiler.py（sample 参数 + 采样逻辑 + `extra["sampled"]=True`）、report.py 画像章节标注、app.py 采样开关。

**验收**：构造 60 万行合成数据，画像耗时 < 采样前 1/3，报告含采样声明；小数据行为完全不变。

---

## S3. 图表升级

### S3.1 matplotlib → plotly 交互图（成本：中）

**现状**：五处图表全是静态 PNG（相关热图、直方图、缺失矩阵、实验图、重要性条形图），不能悬停看值、不能缩放。

**设计**：`utils/charts.py` 每个函数增加 plotly 版本，返回 `plotly.graph_objects.Figure`，UI 用 `st.plotly_chart(fig, use_container_width=True)`；`fig_to_png`（report_images.py）对 plotly 调 `fig.to_image(format="png")`（需要 kaleido 包，加入 requirements）。**报告 HTML** 同步升级为嵌入 plotly 的交互 div（`fig.to_html(full_html=False, include_plotlyjs="cdn")`，首图带 cdn 引用其余复用）——HTML 报告从静态文档升级为交互文档，MD 版保持 PNG（Markdown 生态不支持内嵌交互图）。

**改动清单**：utils/charts.py（plotly 版五函数，旧 matplotlib 函数保留给 MD 导出与测试）、services/report_images.py（plotly → png 转换 + HTML div 导出）、services/report.py（HTML 画像/实验图改 div 嵌入）、app.py（st.plotly_chart 替换）、requirements（+plotly、+kaleido）、tests（plotly Figure 类型断言）。

**验收**：UI 五张图全部可悬停/缩放；HTML 报告在浏览器中图表可交互且单文件自包含（断网打开仍渲染，CDN 降级为基础渲染）；MD 导出仍有 PNG；`pytest -q` 全绿。

**风险**：kaleido 静态导出在 Windows CI 的 Chromium 依赖——smoke job 已覆盖报告生成路径，失败则回退 matplotlib PNG for MD（设计已内置双轨）。

### S3.2 统计注解上图的自动化（成本：低）

直方图叠加核密度曲线与偏度标注；箱线图标注均值/中位数差异；相关热图单元格悬停显示 n 与 p。全部从 `ProfileReport`/`ExperimentResult` 已有字段取数，零新增计算。

---

## S4. 报告与分享体验

### S4.1 一键研究包导出 zip（成本：低）

**场景**：交作业/给导师/存档——一个 zip 包含 md + assets + HTML + 全部复现脚本 + 实验记录 JSON。

**设计**：`services/reports_store.py::export_bundle(name, base=None) -> bytes`（zipfile 内存打包：报告 md/html、assets/、每实验 repro 脚本、tracking 中该批实验的 JSON）；UI 报告库与 CLI `reports --export <name>` 输出该 zip。

**改动清单**：reports_store.py（export_bundle）、app.py 报告库（下载按钮）、cli.py（--export）、tests（zip 内容断言：至少含 md/html/至少一个 .py）。

**验收**：zip 解压后 `python */run.py`（复现脚本）可独立运行；md 图片引用相对路径在解压目录内有效。

### S4.2 PDF 导出（成本：中）

HTML 报告已是自包含单文件 → PDF 用 `weasyprint`（纯 Python，Windows wheel 可用）一行转换；`cli.py analyze --pdf` 与报告库按钮接入。中文字体显式声明（README 截图验证过的 Microsoft YaHei 路径）。**风险**：weasyprint 原生依赖（GTK）——备选 playwright print-to-pdf（已在 CI 有 Chromium 先例）。先 weasyprint，失败切 playwright。

### S4.3 报告对比（成本：中，远期）

报告库选两份同名数据集报告 → 指标对照表（p 值/效应量/样本量/结论变化）。依赖报告内嵌实验 JSON——当前 ReportBundle 只有文本，需在 save_report 时同步保存 `experiments.json`（AgentRunResult.records 本来就有）。列为 v1.3 可选项。

---

## S5. 数据接入扩展

### S5.1 粘贴数据（成本：低）

侧边栏数据源新增"粘贴表格"：`st.data_editor` 空表直接粘贴 Excel 区域 → 得到 DataFrame → 与上传同流程。零新依赖，覆盖"就几十行数据懒得存文件"的高频场景。

### S5.2 数据集画廊（成本：低）

示例不再只有学生成绩：`utils/sample_data.py` 增加销售（含季节性+缺失+离群）、医疗（类别不平衡+计数终点）两个合成数据集，侧边栏下拉选择；`scripts/generate_sample.py` 同步支持 `--dataset` 参数。让"一键体验"对不同领域演示都成立。

### S5.3 URL 拉取（成本：中，含安全边界）

`utils/uploads.py::load_from_url(url)`：仅 https + 公网 IP（复用 cet6_app 已验证的解析-阻断私网模式）+ 大小上限 + 域名可选白名单；UI 输入框 + CLI `--url`。**默认关闭**，`.env` 显式开启（RESEARCH_LAB_ALLOW_URL=1）——符合项目"默认暴露面最小"的安全立场。

---

## S6. 错误处理与信任

### S6.1 人话错误卡（成本：低）

**现状**：UI `st.error(str(exc))` 直接甩异常文本。

**设计**：`ui/errors.py::show_error_card(exc, next_actions: list[str])`——统一渲染"发生了什么（人话复述）/ 为什么 / 你可以试：1… 2…"。七个高频错误预置文案映射（文件编码不支持 / 超大小 / 变量不存在 / 样本不足 / LLM 未配置 / LLM 超时 / 报告不存在），未命中映射的回退显示原始信息 + 通用三步。API 侧 detail 文案同步复用同一映射（`services/error_messages.py` 单一来源）。

**改动清单**：ui/errors.py、services/error_messages.py、app.py 七处 st.error 替换、routers detail 替换、tests（映射全命中断言）。

**验收**：构造七类错误，UI 均显示建议动作而非异常栈；API detail 与 UI 文案同源。

### S6.2 API 错误标准化 RFC 7807（成本：中）

自定义 exception handler 把所有 HTTPException 渲染为 `application/problem+json`（type/title/status/detail/instance），OpenAPI 文档同步声明。前端 UI 不受影响（仍读 detail）。**验收**：所有 4xx 返回 content-type 为 problem+json 且含必填字段；OpenAPI schema 无告警。

### S6.3 统计前提检查卡（成本：中）

把散落在 extra 的诊断（Shapiro/BP/稀疏格/样本量）升级为报告每个实验下的**结构化前提清单**：`✅ 正态性（p=0.42）⚠️ 异方差（p=0.01，已用 Welch）`。数据已齐（executor diagnostics），纯 report 模板工作。这是"懂统计"人设的最直观展示面。

### S6.4 CLI 体验（成本：低）

`cli.py --version`；成功/失败彩色输出（ANSI，Windows Terminal 原生支持）；`analyze` 结束打印"下一步：python cli.py reports"式指引；`profile` 支持管道输入（`-` 读 stdin）。

---

## S7. 会话与持久化

### S7.1 刷新恢复（成本：中）

**现状**：刷新浏览器 = 数据源、已跑实验、当前报告全丢。

**设计**（轻量方案，不引入服务端会话）：Agent/流程产出的报告已沉淀报告库——刷新后报告天然可从 ⑤ 恢复。补两块：①`st.query_params` 记住"当前数据源=示例/最后上传文件名"，刷新后数据源自动恢复（示例数据零成本恢复；上传文件提示重新选择）；②实验历史写入 session 的同时落 `data/sessions/<指纹>.json`，提供"恢复上次会话实验历史"按钮（读回 records 重建 ⑤ 报告库视图）。

**验收**：示例数据模式下刷新页面，数据源自动恢复；刷新后报告库与实验追踪数据完整。

### S7.2 同数据集分析历史入口（成本：低）

实验追踪页已有"同数据集同任务趋势图"——补一个入口：按指纹筛选该数据集的全部实验与由其生成的报告，一键跳转。

---

## S8. 国际化与可达性（远期）

- **S8.1 i18n**：UI 文案抽取到 `ui/i18n.py` 字典（zh 默认，en 复用 README.en 的语料），侧边栏语言切换；services 层报错文案保持中文（error_messages 单一来源加 en 映射）。触发时机：有真实海外用户/导师需求再做，避免维护两套文案的持续成本。
- **S8.2 可达性**：图片 alt、色盲友好调色板（BLUE/NAVY 之外加形状区分）、关键按钮键盘焦点——随 S3 plotly 迁移顺带处理（plotly 原生支持更好）。

---

## 附：UX 落实清单

- [x] S0 app.py 拆分多页面（UX 前置重构）
- [x] S1.1 示例数据运行时自愈
- [x] S1.2 scripts/doctor.py 环境体检
- [x] S1.3 首次引导卡
- [x] S2.1 画像缓存（数据指纹哈希）
- [x] S2.2 ML/Agent 分步进度（st.status + progress/on_step 回调）
- [x] S2.3 大文件采样画像
- [x] S3.1 plotly 交互图（requirements 已加，UI 逐步替换中）
- [ ] S3.2 统计注解上图
- [x] S4.1 研究包 zip 导出
- [x] S4.2 PDF 导出（fpdf2 方案）
- [ ] S4.3 报告对比（可选）
- [x] S5.1 粘贴数据
- [x] S5.2 数据集画廊
- [ ] S5.3 URL 拉取（默认关）
- [x] S6.1 人话错误卡（七类映射）
- [x] S6.2 RFC 7807 错误标准化（APIError 作用域）
- [x] S6.3 统计前提检查卡
- [x] S6.4 CLI --version/下一步指引（彩色待 ANSI 引入库）
- [x] S7.1 刷新恢复
- [x] S7.2 同数据集历史入口
- [ ] S8 i18n 与可达性（远期）

> 完成 S1.1/S2.1/S6.1 即可宣布达成北极星指标的 TTFV ≤ 2 分钟与失败可恢复 100%——这是 UX 1.1 周的验收口径。









