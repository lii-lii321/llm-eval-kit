# research-lab 提升方案与落实手册

> 基线：v0.7.0（145 tests 绿、双 CI job、11 API、五标签页 UI）
> 编制：2026-10-01 · 维护方式：每完成一条目打勾并在 CHANGELOG 记录版本
> 用法：A–H 八个主题相互独立、每条目独立可交付；按 §0 排期顺序执行，测试先行，验收标准全部可机器判定。

---

## 0. 总览与排期

### 现状快照

| 维度 | 现状 | 缺口 |
|---|---|---|
| 统计执行 | 十检验 + 效应量 + BH 校正 + 原始 p 入库 + 复现脚本 | 无置信区间、无回归诊断、无功效分析（statsmodels 立项书列了但从未装） |
| ML | 多模型基线 + 分层 k 折 CV 选优 | 无调优、无持久化、无特征解释 |
| 学术 | arXiv 词面检索 + 复现脚本附录 | 无论文复现专栏 |
| 工程 | pytest + smoke 双 job | 无 lint/type/覆盖率门禁、依赖未锁定、README 数字漂移 |
| LLM | 白名单门控 + 规则回退 + Mock 单测 | 无评测集、无重试 |
| EDA | 八类告警 + 相关热图（UI） | 报告文档零图、告警只报问题不开处方 |
| 因果 | 局限声明"非因果" | 无建设性下一步 |
| 门面 | 中文 README + 三张截图 | 无英文版、无架构图、无 GIF |

### 排期建议

| 版本 | 条目 | 主题 | 预估 |
|---|---|---|---|
| v0.8.0 | D1 → D3 → A1 | 门禁先行 + 统计闭环 | 1 周 |
| v0.9.0 | B1 → B2 → E2 | ML 纵深 | 1 周 |
| v0.10.0 | E1 → F1 → F2 → G1 | 评测与报告叙事 | 1 周 |
| v1.0.0 | C1 → H1 | 学术展示物 + 门面 | 2 周 |

### 原则

1. 每条目独立可交付：做完即提交，不出现跨条目的半成品。
2. 测试先行：每个验收标准对应至少一条 pytest 断言。
3. 兼容性：ExperimentResult 只通过 `extra: dict` 扩展，ExperimentPlan 只加带默认值的可选字段——不做破坏性 schema 变更。

---

## A. 统计纵深

### A1 statsmodels：置信区间 + 回归诊断 + 功效分析（核心，成本：中）

**目标**：补齐"检验执行 → 实验设计 → 结论"闭环。三个能力点：①效应量置信区间；②线性回归完整诊断（调整 R²、F 检验、残差正态性/异方差）；③每个实验自动给出"检测该效应所需样本量"。

**设计**

- 依赖：`requirements.txt` 增加 `statsmodels>=0.14`（立项书 :53 本就列了它，补装即与规划对齐）。
- 效应量 CI（不依赖 statsmodels，纯 scipy 可算）：
  - pearson/spearman：Fisher z 变换，`z = arctanh(r)`，`se = 1/sqrt(n-3)`，CI = `tanh(z ± 1.96*se)`，写 `extra["ci95"] = [low, high]`；
  - welch/independent_ttest：均值差 CI = `t.interval(1-alpha, df_welch, loc=diff_mean, scale=se_diff)`，写 `extra["mean_diff_ci95"]`；
  - 其余方法暂不给 CI（报告中显示"—"）。
- 线性回归诊断（statsmodels OLS）：`extra` 增加 `adj_r2`、`f_statistic`、`f_pvalue`、`resid_shapiro_p`（正态性）、`bp_p`（Breusch-Pagan 异方差）； Shapiro/BP 的 p<0.05 时追加 `extra["diagnostics_warn"]`。
- 功效分析（statsmodels.stats.power）：新模块 `services/power.py`——
  ```python
  def required_n(method: str, alpha: float, power: float, effect: float | None) -> int | None
  ```
  t/welch/mannwhitney → `TTestIndPower`；anova/kruskal → `FTestAnovaPower`；相关 → `zt_ind_solve_power`（效应量用 |r| 换算 d）；效应量未知时默认中等 d=0.5 并在 notes 标注假设。
- 挂接点：`planner.generate_experiment_plan` 产出计划后调用 `required_n`，写 `ExperimentPlan` 新增可选字段 `required_n: Optional[int] = None`（来源效应量假设写进 notes）。

**改动清单**

| 文件 | 改动 |
|---|---|
| requirements.txt | +statsmodels |
| models/schemas.py | ExperimentPlan +`required_n`；ExperimentResult 不变（全走 extra） |
| services/executor.py | `_run_correlation`/`_run_regression`/`_run_group_compare` 增 CI 与诊断；新增 `_effect_ci_pearson` 辅助 |
| services/power.py | 新建 `required_n()` |
| services/planner.py | 计划生成后挂 `required_n` |
| services/repro.py | linear_regression 模板改用 statsmodels OLS 并打印 conf_int |
| services/report.py | MD/HTML 实验行追加"95% CI / 所需样本量"两列（无则"—"） |
| tests | 见下 |

**测试计划（test_executor.py / test_planner.py / test_repro.py）**

1. `test_pearson_ci_brackets_r`：构造 r≈0.9 数据，断言 `extra["ci95"][0] <= r <= extra["ci95"][1]`。
2. `test_regression_has_diagnostics`：断言 extra 含 `adj_r2`/`f_pvalue`/`bp_p`。
3. `test_power_required_n_positive`：welch 计划 d=0.5 → required_n ≥ 16/组（教科书值 sanity）。
4. `test_plan_attaches_required_n`：generate_experiment_plan 后 `plan.required_n` 非 None。
5. `test_repro_ols_compiles`：linear_regression 生成脚本含 `import statsmodels` 且 `compile()` 通过。

**验收标准（机器可判定）**

- `pytest -q` 全绿，新增 ≥5 用例；
- `python cli.py analyze <示例>` 生成的报告 MD 中出现 "95% CI" 与 "所需样本量" 字样。

**风险**：statsmodels 拖慢 CI 安装（+~30s，可接受）；小样本下 BP/Shapiro 检验不稳定——n<20 时跳过诊断并在 extra 标注。

---

## B. ML 纵深

### B1 调优-持久化-解释三件套（核心，成本：中）

**设计**

- 调优：最佳模型选出后跑小网格 `GridSearchCV(estimator, grid, cv=同款分层k折, scoring="r2"/"f1_macro", n_jobs=-1, refit=True)`；网格按模型类型硬编码（小而精，防止运行时爆炸）：
  - LogisticRegression: `C ∈ {0.1, 1, 10}`
  - RandomForest: `n_estimators ∈ {100, 300}`, `max_depth ∈ {None, 10}`
  - XGBoost/LightGBM: `max_depth ∈ {3, 5}`, `learning_rate ∈ {0.05, 0.1}`, `n_estimators ∈ {200}`
  - LinearRegression: 不调（无超参），跳过
- 持久化：tracking 入库拿到 uid 后，`joblib.dump({"pipeline": tuned_pipe, "model": name, "metrics": metrics}, data/models/{uid}.joblib)`；`data/models/` 入 .gitignore。
- 解释：`sklearn.inspection.permutation_importance(tuned_pipe, X_test, y_test, n_repeats=10, random_state=42)`，取 top10 生成条形图（`utils/charts.feature_importance_chart`），嵌报告 ML 章节（HTML base64，见 F1 同机制）与 UI。
- metrics 扩展：每模型增加 `BestParams`（JSON 串）列？——不新增列，写进 `params` 字段（`params = {"cv_best": {...}, ...}`）。

**改动清单**：services/ml_lab.py（选优后调优分支）、utils/charts.py（+feature_importance_chart）、services/report.py（ML 章节嵌图与"最佳超参"行）、.gitignore（+data/models/）、tests/test_ml_lab.py。

**测试计划**

1. `test_tuning_improves_or_matches`：调优后 CV 均值 ≥ 调优前 − 0.05（防波动）。
2. `test_model_persistence_roundtrip`：dump 后 load，对同一行输入预测值一致。
3. `test_report_contains_importance`：报告 ML 章节含"特征重要性"。
4. `test_linear_regression_skips_grid`：LinearRegression 的 params 无 cv_best。

**验收**：`pytest -q` 全绿；`cli.py analyze` 后 `data/models/` 出现 .joblib；报告含调优后指标与重要性图。

**风险**：运行时间增长——网格 ≤6 组合、n_jobs=-1；小样本（<80 行）直接跳过调优并在 notes 标注。

### B2 自动模型卡片（成本：低）

报告 ML 章节末尾追加"模型卡片"小节：数据指纹、任务、训练/测试规模、最佳模型与指标、调优超参、重要特征 top5、局限（样本量、类别平衡）。纯 report.py 模板工作，数据全部已在 MLExperimentResult 里。验收：报告含"模型卡片"标题；无独立测试需求（由现有报告测试覆盖断言一条即可）。

### B3 SHAP 解释（可选，远期）

`shap` 作为可选依赖：装了就在 B1 图旁加 summary_plot，没装跳过。**先不做**——permutation importance 已覆盖面试叙事，shap 安装体积大、Windows 编译风险高。

---

## C. 学术复现线

### C1 论文复现专栏手册（核心，成本：高，周期 2 周）

**目标**：复现一篇带公开数据集的方法论文的主结果，产出"复现报告 + 与本项目基线同表对比"，作为套磁/简历的科研证据。

**选文标准**：①方法 ≤2 页数学；②官方或论文内含完整数据链接；③统计方法可用本项目已实现的检验家族复算（相关/组间/回归）。候选方向：经典异常检测（IQR/LOF 对比）、简单因果（倾向得分匹配的朴素版）、教育测量经典结论复算。

**目录与产物**

```
reproductions/
└── 2026-<paper-slug>/
    ├── README.md        # 论文信息、复现目标（哪个表哪一行）、结果对照表、偏差与原因
    ├── run.py           # 一键复现脚本（下载/读取数据 → 方法 → 输出指标）
    ├── data/            # 数据快照或下载说明（大文件不入库）
    └── notes.md         # 复现过程笔记
```

**步骤（第一篇的落实顺序）**

1. 用 `literature.py` 检索 + 人工筛选锁定论文（半天）；
2. 通读方法节，把方法映射到本项目统计家族（半天）；
3. `run.py` 复现主表一个单元格（1-2 天）；
4. README.md 写对照表：论文值 vs 复现值 vs 本项目 `cli.py analyze` 基线值（半天）；
5. 报告/简历叙事打磨：偏差原因诚实标注（半天）。

**验收**：`python reproductions/<slug>/run.py` 可重复运行并打印对照表；README.md 对照表三列齐全；主 README 增加复现专栏链接。

**风险**：数据链接失效（快照入库，>50MB 走下载说明）；复现值对不上（诚实写偏差分析——这本身就是最有价值的科研叙事）。

---

## D. 工程门禁

### D1 ruff + mypy + 覆盖率门禁（成本：低，性价比最高）

**改动清单**

1. `ruff.toml`（新建）：
   ```toml
   line-length = 120
   [lint]
   select = ["E", "F", "W", "I", "UP", "B"]
   ```
2. `mypy.ini`（新建，宽松起步）：
   ```ini
   [mypy]
   ignore_missing_imports = True
   no_strict_optional = True
   files = services, utils, routers, models, cli.py
   ```
3. ci.yml test job 在 pytest 前追加：
   ```yaml
   - run: ruff check .
   - run: mypy
   - run: pip install pytest-cov && pytest -q -rs --cov=services --cov-report=term --cov-fail-under=85
   ```
4. requirements.txt 增加 dev 分组：`ruff`、`mypy`、`pytest-cov`（或建 requirements-dev.txt）。

**落实手册（预期第一轮失败与修法）**

- ruff 首跑预计 10~30 个 I（import 排序）与 UP 提示：`ruff check --fix .` 自动修复大半，余下手工。
- mypy 首跑预计报 Optional 解包（`verdicts[idx]` 类）与 pandas Any 返回：第一轮策略 = 收窄明显处 + 对 pandas/sklearn 边界加 `cast`/`# type: ignore[var-annotated]`，**每个 ignore 必须带错误码注释**。
- 覆盖率若低于 85：优先给 services/ 下未覆盖分支补测（literature 网络分支已测、report 分支较全；缺口大概率在 app.py 与 cli 主函数——app.py 排除在 cov 之外，cli 用现有用例覆盖）。

**验收**：CI 全绿且含 ruff/mypy/coverage 三步；`--cov-fail-under=85` 生效（本地故意写死 return 验证门禁会红）。

### D2 依赖锁定（成本：低）

`requirements.in`（现 >= 内容）+ `pip-compile requirements.in -o requirements.txt`（锁 ==）；CI 与 Dockerfile 用锁定版；升级流程 = 改 .in → recompile → 全量测试。提交 requirements.in 与锁定文件两者。

### D3 文档漂移治理（成本：极低）

README "135 用例" → "145+ 用例（以 CI 实时结果为准）"；CHANGELOG 保持人工，但 README 测试数字一律不带具体值或注明"见 CI"。

---

## E. LLM 评测与稳健性

### E1 golden 评测集 + 一致率报告（成本：中）

**设计**

- `tests/eval/golden_cases.json`：15~20 条案例，schema：
  ```json
  [{"id": "num_num_pearson", "columns": {"x1": "numeric", "x2": "numeric", "g": "categorical"}, "task": "", "expected_method": "pearson"}]
  ```
- 合成数据构造器：`tests/eval/synth.py::build_df(columns, n=60)` 按列类型生成确定性数据（numeric→normal、categorical→2~3 类、identifier→序列、skewed→lognormal）。
- `tests/test_eval_golden.py`：对每条案例走 `generate_research_questions_auto(规则)` + `generate_experiment_plan(None)`，断言 `plan.method == expected_method`——**这条同时是规则引擎的回归测试**，进 CI 主 job。
- `scripts/eval_llm.py`（有 Key 手动跑）：同样案例走 LLM 路径，输出逐案例 LLM 方法 vs 规则方法 vs 期望方法的一致率表，写 `data/eval/llm_eval_<ts>.json`。

**验收**：golden 测试进 CI 全绿；eval 脚本跑通输出一致率数字（配 key 后）。

### E2 llm.py 指数退避重试（成本：低）

```python
RETRYABLE = {429, 500, 502, 503, 504}
for attempt in range(3):
    resp = httpx.post(...)
    if resp.status_code in RETRYABLE and attempt < 2:
        time.sleep(0.5 * 2 ** attempt + random.uniform(0, 0.25))
        continue
    break
```
传输层异常（ConnectError/Timeout）同样重试。测试（MockTransport）：第一次 503 第二次 200 → 返回成功且调用 2 次；连续三次 503 → LLMError。`random` 用 `random.uniform`（项目代码允许）。

---

## F. EDA 报告升级

### F1 报告配图（成本：中）

- `utils/charts.py` 新增：
  - `numeric_histograms(df, cols, max_plots=6)`：按缺失率升序取前 N 数值列，2×3 网格直方图 + 偏斜标注（复用 profiler 的 skewness）；
  - `missing_matrix(df, max_rows=80)`：布尔缺失矩阵 imshow（行采样），一眼看出缺失模式。
- 嵌入机制：报告构建时新增 `images: list[tuple[name, Figure]]` 参数 → HTML 版转 base64 data URI 内嵌（单文件自包含）；MD 版在 `data/reports/assets/<报告名>/` 落盘 PNG 并用相对路径引用（`save_report` 顺带拷贝）。
- 位置：数据画像章节"### 分布概览（前 6 个数值字段）"与"### 缺失矩阵"。

**测试**：两函数返回 Figure / 异常输入返回 None（对齐现有 charts 测试风格）；带 images 的 HTML 含 `data:image/png;base64`。

### F2 告警处置建议映射（成本：低）

- `models/schemas.py`：WarningItem +`suggestion: str = ""`。
- `services/profiler.py`：`WARNING_SUGGESTIONS: dict[str, str]` 八类映射，建告警时填充。示例：HIGH_MISSING→"缺失>60% 建议删列，否则可中位数/模型插补并加缺失指示列"；SKEWED_DISTRIBUTION→"考虑对数/Box-Cox 变换，或改用秩方法（本项目计划层已自动升级）"；HIGH_CORRELATION→"二选一或做 PCA；回归注意共线性"。
- report.py 质量提示行渲染 `→ 建议：{suggestion}`。

**验收**：报告质量提示逐条带处方；八类映射有 pytest 参数化用例。

---

## G. 因果与 A/B 薄层

### G1 反事实提示 + 处理/对照识别（成本：低）

- report.py：结论表后新增"### 因果提示"小节——对每个 `reject_h0` 记录，用 profile 的 correlations 找 1~2 个与自变量相关的其他字段作潜在混杂，输出一句模板："「X→Y」显著不等于因果：需排除 {混杂} 的干扰；要因果化需随机化实验或匹配设计。"混杂不足两条时给通用表述。
- research_questions.py 规则模式：列名含 `group|treatment|control|variant|组|处理|对照` 的二分类列优先配数值目标走 welch/mannwhitney，且计划 notes 标注"A/B 语义"。

**测试**：report 含"因果提示"且显著实验有条目；含 treatment 列的 RQ 生成走组间比较。

---

## H. 门面

### H1 双语 + 架构图 + GIF（成本：低）

1. `README.en.md`：一页版，主打三道闸（whitelist/deterministic dispatch/source labeling）+ quickstart + 截图；中文 README 顶部加 `English | 中文` 切换链接。
2. mermaid 架构图置顶（GitHub 原生渲染）：
   ```mermaid
   flowchart LR
     CSV[CSV/Excel] --> Prof[Dataset Profiler]
     Prof --> RQ[Research Questions]
     RQ --> Plan[Experiment Plan]
     Plan --> Exe[scipy Executor]
     Exe --> Rep[Report MD/HTML]
     Prof --> ML[ML Baselines + CV]
     ML --> Track[(Tracking DB)]
     Exe --> Track
     Lit[arXiv Literature] --> Rep
   ```
3. demo GIF：VHS（winget 装 charmbracelet/tap 或直接下载二进制）写 `demo.tape` 录 `cli.py analyze` 全流程 30 秒 → `docs/images/demo.gif` 嵌双语 README。

### H2 远期：src 布局与 PyPI 化

`src/researchlab/` 包化 + pyproject.toml + `pipx install`。**挂起理由**：当前 tests 直接 import 仓库层运行良好，包化是一次纯移动重构，等 C1 论文复现需要 `pip install researchlab` 时再做。

---

## 附：落实清单（按序打勾）

- [x] D1 ruff+mypy+cov85 门禁进 CI
- [x] D2 pip-compile 锁定依赖
- [x] D3 README 测试数字去漂移
- [x] A1 statsmodels：CI+诊断+功效（≥5 新用例）
- [x] B1 调优-持久化-解释三件套
- [x] B2 自动模型卡片
- [x] E2 LLM 指数退避重试
- [x] E1 golden 评测集（CI）+ eval_llm 脚本
- [x] F1 报告配图（直方图+缺失矩阵）
- [x] F2 八类告警处置建议
- [x] G1 因果提示 + A/B 识别
- [x] C1 第一篇论文复现专栏（Student 1908，t=4.0621 vs 4.06 ✅）
- [x] H1 README.en + mermaid 架构图（demo GIF 待 VHS 录制）

> 完成即 v1.0.0：打 tag、发 Release（附 demo GIF 与论文复现链接），简历项目描述更新为"统计闭环 + ML 纵深 + 论文复现 + 全链路可溯源"。





