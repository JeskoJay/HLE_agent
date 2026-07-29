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

参考 Fieldframe Labs FF-STACK 多智能体架构，设计如下流水线：

```
Input → Sentinel → Forge/RAG → Solver Pool (3 branches) → Arbiter → Output
```

- **Sentinel**：读取并预处理题目。
- **Forge/RAG**：从知识库中检索相关题型卡片，为 solver 提供上下文。
- **Solver Pool**：3 个认知分支并行工作：
  - `systematic`：系统性、分步推导。
  - `intuitive`：直觉式、快速判断。
  - `code_first`：优先写代码/调用工具验证。
- **Arbiter**：综合 3 个分支的答案与置信度，输出最终答案。

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
| gold_answers.json 来源存疑 | 7 条答案无法溯源，可能循环自评 | 已识别并标记；建议用新知识库第 5 节答案重建 gold（待用户确认） | `gold_answers.json` |

---

## 5. 当前状态

- **代码**：已推送 GitHub `main` 分支。
- **配置**：
  - 模型：`deepseek-v4-pro`
  - RAG top-k：4
  - 真实并发上限：10
  - 分支：systematic / intuitive / code_first
- **运行中**：`wDUa0x` 正在以 top4 + 无 README 污染的配置干净重跑全部 20 题。
- **待产出**：`outputs/EVAL_SUMMARY.md`、`outputs/judge_results.json`、`TEST_REPORT.md`。

---

## 6. 后续可优化方向

1. **gold 答案重建**：用新知识库第 5 节给出的 20 题推荐答案替换旧 `gold_answers.json`，消除循环打分隐患。
2. **更智能的检索**：当前是 TF-IDF，可升级为 embedding-based dense retrieval。
3. **分支结果融合策略**：Arbiter 目前基于置信度简单综合，可引入证据强度加权。
4. **calibration 优化**：模型普遍过度自信，可在 Arbiter prompt 中加入 calibration 提示。
5. **成本监控**：增加 token 用量与 API 费用统计。
