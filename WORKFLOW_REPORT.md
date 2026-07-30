# HLE Solver Agent — 工作流程与版本迭代报告

> 本报告记录从项目启动到当前版本的完整工作流：目标设定、架构选择、迭代过程、遇到的关键问题及解决方案。

---

## 1. 项目目标

构建一个能自动求解 **Humanity's Last Exam (HLE)** 20 道综合题的 Agent，并给出可审计的评测结果。

核心约束：
- 只能使用 `deepseek-v4-pro`（服务端 `https://api.ai-native-x.site/`）。
- 必须输出 `Explanation / Answer / Confidence` 三段式结果。
- 最终交付：项目文档、测试报告、工作流程报告。
- 代码需推送到 GitHub 仓库 `JeskoJay/HLE_agent`。

---

## 2. 架构设计（FF-STACK）

参考 Fieldframe Labs FF-STACK 多智能体架构，当前（v0.9.1）流水线：

```
Input → Sentinel → Planner 预分析 → 计划驱动 RAG → Solver Pool (3 branches × 自洽采样) → 分歧 Debate → Arbiter → Reflection → 后校验(多选逐判/数值复算) → 置信度压缩 → Output
```

- **Sentinel**：读取并预处理题目。
- **Planner（前置）**：拿到题目先做预分析，一次调用产出「分步解题计划 + 检索关键词」；计划注入各分支 prompt。
- **计划驱动 RAG**：用「题干 + 计划关键词」构造检索 query，BM25 打分，卡片式知识库一卡一 chunk，**top-1**（v0.9.1 由 v0.8 的 top-4 逐步收敛而来）。
- **Solver Pool**：3 个认知分支并行工作，每分支自洽采样 k 次多数投票：
  - `systematic`：系统性、分步推导。
  - `intuitive`：直觉式、快速判断。
  - `code_first`：ReAct 式「思考→调工具→观察」循环，优先写代码验证。
- **分歧 Debate（v0.9 新增）**：当分支间答案分歧时触发，让分歧分支深挖对方证据、交叉质证，输出更稳健的共识（见 `verifier.py`）。
- **Arbiter**：综合 3 个分支的答案与置信度，输出候选最终答案。
- **Reflection**：Critic 审查三分支的事实/逻辑/数值错误，Refiner 产出修正版最终答案（非选择题启用）。
- **后校验（v0.9 新增）**：
  - `multiselect.py`：对"选择所有正确项"的多选题，逐选项独立判定 True/False，拼出组合答案（命中 Q18/Q19/Q20，不误伤单选 Q5/Q10）。
  - `verifier.py`：数值 / `X:Y` 类答案用独立代码复算交叉验证；空答案触发兜底不变量（回退 Reflection→Arbiter→最高置信分支并封顶 60）。
- **置信度压缩（v0.9 新增）**：`min(conf, 40 + 0.5×conf)`，抑制 95%+ 过度自信，全部有效预测落入 75–89% 区间。

---

## 3. 版本迭代过程

### v0.1 — 单题跑通
- 实现基础 API 调用与单题求解。
- 验证 `deepseek-v4-pro` 可用，确认其为强制 thinking 模式（`reasoning_content` + `content` 同时返回）。

### v0.2 — 多智能体流水线
- 引入 Sentinel、Forge、Solver、Arbiter 模块。
- 实现三认知分支并行求解。
- 首次跑完前 10 题。

### v0.3 — 流式日志与工具调用
- 增加 SSE 流式输出，支持实时查看推理过程。
- 为每道题生成独立的 `reasoning.log`，避免并行时日志串台。
- 修复 DSML marker 泄漏问题（模型输出中的 `</｜｜DSML｜｜tool_calls>` 等标记被误写入最终答案）。

### v0.4 — 并发控制与输出规范化
- **问题发现**：20 题全并行 × 每题 3 分支 = 60+ 路真实并发，容易触发代理限流，重试耗尽后分支返回空答案，降低质量。
- **解决方案**：在 `api_client.py` 增加全局信号量 `_req_sem`，把真实并发请求封顶为 `MAX_CONCURRENT_REQUESTS = 10`。
- 输出文件夹从裸 `qid` 改为 `NN_<qid>`（如 `01_66b91693...`），并写入 `meta.json` 记录题号、ID、题型、题干。

### v0.5 — 知识库接入与检索策略优化
- 接入用户提供的 `HLE_20Q_Agent_Knowledge_Base_EN.md`（含 K01–K17 题型卡片 + 20 题实例缓存）。
- 检索策略从 top4 → top2 → top4。
- **问题发现**：原始按 800 字符滑窗分块会把一个 K 卡片切断，导致 TF-IDF 检索时 top2 命中错误卡片（例如 Q9 的布尔可分离性被错判为 K08 指数竞速）。
- **解决方案**：把分块改为按 Markdown 章节整体成块，并剥去 YAML 前言，提升 top-K 命中正确卡片的准确率。
- 将 `knowledge_base/README.md` 移出检索范围，避免说明文档污染检索语料。

### v0.6 — 评测方法对齐官方 HLE
- **问题发现**：原有 `evaluate.py` 只让 judge 输出 `{"correct": true/false}`，未提取 `extracted_final_answer` / `reasoning` / `confidence`，校准误差也采用简单 10% 分桶而非官方 sorted-bucket L2。
- **解决方案**：重写 `evaluate.py`：
  - 采用官方 `JUDGE_PROMPT`。
  - 要求 deepseek-v4-pro 输出 JSON：`extracted_final_answer`、`reasoning`、`correct`（yes/no）、`confidence`。
  - 校准误差改用官方 `calib_err`（排序后分桶，beta=100，L2）。
  - 新增 `judge_results.json` 保存每题详细判定结果，便于审计。

### v0.7 — 官方 gold 重建与 judge 截断修复
- 用官方 `hle_dataset.json`（2500 题完整集）重建 `gold_answers.json`（旧 gold 答案错误，已删除）。
- 修复 judge 截断 bug（`response[:4000]` → 全文，上限 120k），准确率从失真的 5% 修正为可信的 **40.0%**。

### v0.8 — P0/P1 架构升级（当前版本）
- **Planner 前置 + 计划驱动 RAG**（用户提出）：拿到题目先预分析生成解题计划与检索关键词，再用关键词驱动检索，之后进入求解流程。
- **Reflection 反思闭环**（P0）：新增 `reflect.py`，Arbiter 后 Critic+Refine 修正最终答案。
- **Self-Consistency 自洽采样**（P0）：每分支采样 k 次，答案与置信度多数投票。
- **编排级重试**（P0）：单题流水线失败自动指数退避重试。
- **ReAct 强化**（P1）：code_first 分支显式「思考→调工具→观察」循环。
- **测评增强**（P1）：准精确匹配（集合/顺序/单位/大小写归一化）、按题型/领域分层统计、可靠性图。
- **知识库与检索重构**：换用中英双语卡片式先验知识库（每题一张卡，共 40 张）；chunk 策略改为**一张卡片=一个 chunk**（修复了窗口切分导致只检索出半句话的问题）；TF-IDF 升级为 **BM25**。
- **工程**：`run.py` 新增 `--only N` 单题冒烟测试参数。
- **结果**：Accuracy 40.0% → **45.0%**（+Q3/+Q12，−Q5），Calibration Error 67.53% → **66.18%**。

### v0.9 — 多选逐判 + 数值复算 + 分歧 Debate（中途调整检索策略，未完整评测）
- **多选逐选项判定**（`multiselect.py`，P0）：对"选择所有正确项"题型，逐选项独立判定 True/False 并拼出组合答案，避免整体作答漏选（如 Q20 漏 C）。
- **数值代码复算 + 分歧 Debate + 置信度压缩**（`verifier.py`，P0）：数值 / `X:Y` 类答案用独立代码复算交叉验证；分支分歧时触发 Debate 深挖证据；置信度压缩 `min(conf, 40+0.5×conf)`。
- **检索 top4 → top2**：缩小噪声但保留关键卡片。
- 因中途将检索策略进一步调成 top1 被叫停，未跑完完整 20 题评测，结果并入 v0.9.1。

### v0.9.1 — P0 修复闭环（当前版本，完整 20 题评测）
- **修复 Debate 输出污染**：DSML 标记（`</｜｜DSML｜｜tool_calls>` 等）曾泄漏进 Arbiter 证据，新增 `sanitize` 二次防护（最终文本必须含 `BEST:`）+ `api_client.py` 收尾轮强约束，杜绝泄漏。
- **空答案兜底不变量**：当最终答案为空时回退 Reflection→Arbiter→最高置信分支，并封顶 conf=60，避免空答占位。
- **修复 Q1 崩溃根因（重大运行事故）**：`tools.python_execute` 每次执行后用 `os.remove` 删除临时脚本，单轮删除达 50 次触发**宿主安全删除守卫**无声终止整个 `run.py` 进程；移除该 `os.remove`，临时脚本留在 `.workbuddy/agent_tmp/` 定期人工清理。详见 §4 与 §6。
- **检索 top2 → top1**：最终收敛为单卡检索，配合一卡一 chunk 已足够命中正确题型卡。
- **结果**：Accuracy 45.0%（持平 v0.8，9/20 正确，正确题 Q2/Q6/Q8/Q9/Q10/Q11/Q13/Q16/Q20），Calibration Error 66.18% → **60.42%**；新答对 Q11/Q13/Q20，回退 Q1（崩溃非能力）/Q3/Q12。

---

## 4. 关键问题与解决方案

| 问题 | 影响 | 解决方案 | 涉及文件 |
|------|------|----------|----------|
| DSML marker 泄漏 | 最终答案里混入 `</｜｜DSML｜｜tool_calls>` 等控制标记 | 用更强正则 `_DSML_TAG_RE = re.compile(r"<[^>]*｜[^>]*>")` 清洗 | `solver.py` |
| 60+ 并发打爆代理 | 大量 429/5xx，分支答案为空 | 全局信号量限制真实并发为 10 | `api_client.py`, `config.py` |
| 输出文件夹无题号 | 20 个裸 qid 无法快速定位 | 改为 `NN_<qid>` 并写 `meta.json` | `pipeline.py` |
| 检索分块切断 K 卡片 | top2 命中错误题型卡片 | 按 Markdown 章节整体成块 | `rag_retriever.py` |
| README 污染检索 | 说明文档被当知识检索 | 移出 `knowledge_base/` 到根目录 | `knowledge_base/` |
| 评分方法与官方不一致 | 指标不可比、confidence 未验证 | 重写 `evaluate.py` 对齐官方 judge prompt 与 calib_err | `evaluate.py` |
| gold_answers.json 答案错误 | 旧 gold 与官方答案对不上（如 Q1/Q2），且 7 条无法溯源、疑似循环自评 | 用官方 `hle_dataset.json` 完整集重建 gold（20 题 qid 精确匹配） | `gold_answers.json` |
| **judge 截断 response 致误判** | `response[:4000]` 把 80k 字符末尾答案切掉，judge 抽不到答案、系统性判错 | 改为不截断（上限 120k），与官方一致；并给 judge 循环加并发 | `evaluate.py` |
| 卡片式知识库被窗口切碎 | 800 字符滑窗把一张题型卡切成碎片，检索只命中半句话 | chunk 策略改为一张卡片=一个 chunk，保持语义单元完整 | `rag_retriever.py` |
| RAG 检索与题目意图脱节 | 仅用题干做 query，专业术语覆盖不足 | Planner 前置产出检索关键词，query=题干+关键词 | `pipeline.py` |
| 单次采样偶发计算错误 | 如 Q12 位数偶尔算错 | Self-Consistency 采样 k 次多数投票 | `solver.py` |
| 长推理题无最终答案 | v0.7 的 Q3 在给出答案前耗尽输出 | Planner 明确步骤 + Reflection 兜底强制产出答案 | `pipeline.py`, `reflect.py` |
| **Debate 输出污染最终答案** | DSML 标记（`</｜｜DSML｜｜tool_calls>` 等）泄漏进 Arbiter 证据，干扰判定 | `sanitize` 二次防护（最终文本必须含 `BEST:`）+ 收尾轮强约束 | `verifier.py`, `api_client.py` |
| **过度自信 / 校准差** | 多数预测落在 95%+，Calibration Error 66.18% | 置信度压缩 `min(conf, 40+0.5×conf)`，全部有效预测落入 75–89% | `verifier.py` |
| **Q1 进程被宿主守卫终止（运行事故）** | `tools.python_execute` 每次执行后 `os.remove` 临时脚本，单轮删除达 50 次触发宿主"安全删除守卫"，**无声终止整个 run.py 进程**，Q1 以空答案计 0 分 | 移除 `os.remove`；临时脚本留在 `.workbuddy/agent_tmp/` 定期人工清理（工程红线：禁止在 tools 内批量删文件） | `tools.py` |

---

## 5. 当前状态

- **代码**：已就绪，待推送 GitHub `main` 分支（本轮一并提交新模块 `multiselect.py` / `verifier.py` 与架构图 `HLE_agent_architecture.svg` / `.html`）。
- **配置**：
  - 模型：`deepseek-v4-pro`
  - 流程：Sentinel → Planner 前置 → 计划驱动 RAG（BM25，top-1，一卡一 chunk）→ 3 分支×自洽采样 → 分歧 Debate → Arbiter → Reflection → 后校验（多选逐判 / 数值复算）→ 置信度压缩
  - 真实并发上限：10
- **评测基准**：官方 `hle_dataset.json`（20 题 qid 全部精确匹配，gold 以官方 `answer` 为准）
- **最新运行结果**（2026-07-30，v0.9.1）：
  - **Accuracy（LLM judge）：45.0%**（9/20 正确，正确题：Q2/Q6/Q8/Q9/Q10/Q11/Q13/Q16/Q20）
  - **Accuracy（准精确匹配）：35.0%**
  - **Calibration Error：60.42%**（较 v0.8 的 66.18% 显著改善）
  - 对比 v0.8（45.0% / 66.18%）：逐题结构变化大——新答对 Q11/Q13/Q20，回退 Q1（运行事故非能力）/Q3/Q12；Q1 若正常完成预计 10/20=50%。
- **已产出**：
  - `outputs/EVAL_SUMMARY.md`
  - `outputs/judge_results.json`
  - `README.md`
  - `TEST_REPORT.md`
  - `WORKFLOW_REPORT.md`
  - `HLE_agent_architecture.svg` / `HLE_agent_architecture_diagram.html`（架构图）

---

## 6. 后续可优化方向

1. **重跑 Q1 验证崩溃修复**：Q1 修复后尚未重跑，预计可达正确，整体 Accuracy 由 45% 升至 50%（10/20）。
2. **多选组合题深度优化**：`multiselect.py` 已落地逐选项判定并修复 Q20，但 Q18/Q19 仍有 3–4 处分歧；需增强选项隐式证据召回（必要时对每个选项单独跑验证代码）。
3. **更智能的检索**：当前是 BM25（top-1），可升级为 embedding-based dense retrieval（跨语言召回更稳）。
4. **数值题交叉验证增强**：Q4/Q7/Q14/Q17 仍错（建模/系数错）；可强制多分支用不同方法独立计算，不一致时降置信度并触发二次检索。
5. **稳定性**：Q3（butterfly vs humanity）、Q12（1.007:0 不稳定）属轮间波动，可加答案稳定性投票阈值。
6. **成本监控**：自洽采样 + Reflection + Debate 后 token 消耗明显上升，需增加用量统计。
