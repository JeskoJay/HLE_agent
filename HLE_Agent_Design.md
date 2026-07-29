# HLE 解题 Agent 设计文档

> 版本：v1.0（待评审）  
> 日期：2026-07-29  
> 目标：针对 `HLE_text_only_20questions_student.jsonl` 中的 20 道高难度题目，构建一个基于 `deepseek-v4-pro` 的解题 Agent，输出符合指定格式的 `response`，并附带可追溯的推理日志。

---

## 一、需求确认（请重点核对）

| 编号 | 需求项    | 当前假设                                                                                                   | 需确认 |
| -- | ------ | ------------------------------------------------------------------------------------------------------ | --- |
| R1 | 可用模型   | 仅使用 `deepseek-v4-pro`，通过 `https://api.ai-native-x.site/` 调用；API Key 由调用方配置，不硬编码                        | 确认  |
| R2 | 输出字段   | 在原始 JSONL 每条记录上新增 `response` 字段，内容为：`Explanation: ...\nAnswer: ...\nConfidence: ...%`                  | 确认  |
| R3 | 答案格式   | 严格遵循截图 SYSTEM_PROMPT：`Explanation` + `Answer` + `Confidence(0%-100%)`                                  | 确认  |
| R4 | 题目类型   | `exactMatch`（开放式/字符串答案）与 `multipleChoice`（选择题）混合                                                       | 确认  |
| R5 | 是否需要自评 | 是否要在提交前加入 Judge 判定逻辑（因真实 `answer` 不提供，可用于本地验证）                                                         | 待确认 |
| R6 | 工具支持   | Agent 是否需要调用外部工具（Python 执行、网络搜索、代码分析）？建议开启，20 题中大量密码学/SageMath/代码语义题必须计算                               | 待确认 |
| R7 | 参考架构   | 参考 Fieldframe Labs 的 Sentinel→Forge→Solver(s)→Arbiter 流水线，但全部使用 `deepseek-v4-pro`（通过不同 prompt/温度模拟多分支） | 确认  |
| R8 | 运行环境   | Python 3.10+，依赖 `openai`/`requests`/`python-sandbox` 等                                                 | 待确认 |
| R9 | 最终交付   | 1）解题思路文档；2）可执行推理代码；3）带 `response` 的 JSONL 结果与评估报告                                                      | 确认  |

**请在审阅时回复：以上假设是否有调整？特别是 R5、R6、R8。**

---

## 二、参考架构分析（Fieldframe Labs HLE 方案）

Fieldframe Labs 的 FF-STACK 系统核心思路：

```
Sentinel（轻量分类）
    ↓
RAG（知识库检索，待补充语料）
    ↓
Solver(s)（多分支并行推理，带工具调用）
    ↓
Arbiter（分歧仲裁与最终答案）
    ↓
最终答案 + 溯源轨迹
```

关键启发：

1. **多分支投票**：不同模型/不同 prompt 的 Solver 并行运行，一致则直接提交，不一致则仲裁。
2. **工具增强**：对数学、密码学、代码类题目，必须能执行 Python / 调用 SageMath / 网络检索。
3. **置信度校准**：Confidence 不是装饰，而是评估指标（Calibration Error）的一部分。
4. **轻量治理**：分类器和预检索用低成本模型，核心推理用强模型。

---

## 三、本方案总体架构

由于仅接入 `deepseek-v4-pro`，我们用**同模型多分支 + 角色隔离**来复现 FF-STACK 的投票与仲裁效果。

```
┌─────────────────────────────────────────────────────────────┐
│                      HLE Solver Agent                        │
├─────────────────────────────────────────────────────────────┤
│ 1. Loader     读取 JSONL，逐题送入流水线                        │
│ 2. Sentinel   识别题目子域（Cryptography / CS / ML / Robotics…）│
│ 3. RAG        从知识库检索领域上下文（待用户补充语料）          │
│ 4. Solver Pool                                                  │
│    ├─ Branch A：Systematic Solver（逐步推理，偏好代码验证）      │
│    ├─ Branch B：Intuitive Solver（快速模式识别，偏好概念分析）     │
│    └─ Branch C（可选）：Code-First Solver（对代码题强制生成并执行） │
│ 5. Arbiter    汇总多分支答案与置信度，输出最终 response        │
│ 6. Exporter   写回 JSONL + 生成推理报告                         │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 设计原则

- **单模型多角色**：通过不同 system prompt 让同一模型扮演不同解题角色，降低单一路径偏差。
- **工具即推理**：对可计算、可执行的问题，必须让模型生成可执行代码并检查结果。
- **置信度有依据**：Confidence 由 Arbiter 根据分支一致性、工具验证结果、答案类型综合计算，而非让模型随意输出。
- **失败安全**：任何分支异常时，仍保证输出符合格式，不会中断整条流水线。

---

## 四、模块详细设计

### 4.1 Loader（数据加载）

- 输入：`HLE_text_only_20questions_student.jsonl`
- 输出：按 `id` 索引的题目对象，包含 `id`, `question`, `answer_type`, `raw_subject`, `category`
- 行为：
  - 校验 JSONL 完整性
  - 对超长题目（如 Shamir 共享题）保留原样，不做截断
  - 支持断点续跑（记录已处理的 `id`）

### 4.2 Sentinel（域分类器）

- 模型：`deepseek-v4-pro`（低 temperature，少量 token）
- 输入：题目文本
- 输出：标准化子域标签，例如 `cryptography`, `machine_learning`, `programming_semantics`, `robotics`, `signal_processing`, `computer_architecture`, `reinforcement_learning`, `domain_design`
- Prompt 要点：只输出一个标签，不解释

### 4.3 RAG 检索（领域知识预检索）

> 实现说明：原设计中的 Forge（规则注入）已按你的要求改为 **RAG 检索**，实现见 `hle_agent/rag_retriever.py`。

根据 Sentinel 标签与题目内容，从 `knowledge_base/` 语料库检索相关片段，作为参考材料注入 Solver：

- **SimpleRAGRetriever**（默认）：TF-IDF 词频检索，零依赖，运行时自动扫描知识库目录、分块并建立索引。
- **VectorRAGRetriever**（预留）：语义向量检索，提供 embedding 函数（本地 sentence-transformers 或 embedding API）即可启用余弦相似度。
- 知识库为空时退化为纯模型推理，不影响管线。

> 领域资料（`.txt/.md/.json`）由你后续放入 `knowledge_base/`，放入即生效，详见该目录 README。

### 4.4 Solver Pool（多分支求解）

每个分支使用 `deepseek-v4-pro`，但配置不同的 system prompt 与参数。

#### Branch A：Systematic Solver

- **角色**：严谨的逐步推理者
- **Prompt 核心**：
  - 要求先重述问题、识别已知条件、列出解题计划
  - 对代码/数学问题必须生成 Python/SageMath 代码并执行
  - 最终输出 `Explanation / Answer / Confidence`
- **temperature**：0.2（稳定、确定）

#### Branch B：Intuitive Solver

- **角色**：模式识别专家
- **Prompt 核心**：
  - 鼓励快速识别题目套路、常见陷阱、等价转换
  - 对选择题优先分析选项结构
  - 最终输出 `Explanation / Answer / Confidence`
- **temperature**：0.6（一定创造性）

#### Branch C：Code-First Solver（可选）

- **角色**：代码优先的执行者
- **触发条件**：Sentinel 标签为 `programming_semantics` / `cryptography` / `computer_architecture` / `machine_learning`
- **Prompt 核心**：
  - 强制把问题翻译成可执行代码
  - 对 C 语义题用 Python 模拟等价行为
  - 对 SageMath 题在本地安装 sage 或尽量用 sympy/numpy 近似

### 4.5 ToolKit（工具集）

| 工具名              | 功能                                | 适用题目                   |
| ---------------- | --------------------------------- | ---------------------- |
| `python_execute` | 在沙箱中执行 Python 代码，返回 stdout/stderr | 密码学、矩阵秩、Python 语义、数值计算 |
| `code_analyze`   | 静态分析 C/Python 代码片段，指出潜在语义陷阱       | C 语义题、SageMath 代码纠错    |
| `calculate`      | 调用 Python 表达式快速计算                 | 简单数值题                  |
| `web_search`（可选） | 搜索背景知识                            | 领域概念题                  |

**安全约束**：

- Python 沙箱使用受限子进程，禁止网络、文件写（除临时目录外）、系统命令
- 执行超时：30 秒
- 内存限制：512 MB

### 4.6 Arbiter（仲裁器）

- 输入：多个 Solver 分支的 `(explanation, answer, confidence)`
- 输出：最终 `response`
- 逻辑：
  1. **答案标准化**：去掉多余空格、统一大小写（对字符串答案保留原始含义）
  2. **一致性检测**：
     - 若多分支答案完全一致 → 提升 Confidence
     - 若不一致 → 进入仲裁推理
  3. **仲裁 Prompt**：向 `deepseek-v4-pro` 提供所有分支答案及解释，要求其选择最可靠答案并给出理由
  4. **Confidence 计算**：
     - 基础值：模型自报 confidence
     - 调整项：分支一致加分、工具验证通过加分、答案类型确定性加分
     - 范围：0% - 100%

### 4.7 Exporter（结果导出）

- 输出文件：
  - `HLE_text_only_20questions_responses.jsonl`：原始数据 + `response` 字段
  - `HLE_solver_report.md`：每题推理摘要、分支结果、仲裁理由、耗时
- 格式示例：
  ```json
  {"id":"66b91693d86bff9a12fc1f99","question":"...","answer":"","answer_type":"exactMatch","rationale":"","raw_subject":"Cybersecurity","category":"Computer Science/AI","response":"Explanation: ...\nAnswer: ...\nConfidence: 85%"}
  ```

---

## 五、模型接入方案

### 5.1 API 配置

- 模型：`deepseek-v4-pro`
- 基地址：`https://api.ai-native-x.site/`
- Key：`sk-EZwCtlLpzKIPNAn79YXVgEkjzFPcKKjQSmBYeSGSjxpWRVcm`
- 协议：OpenAI 兼容 API
- 调用方式：统一封装 `call_deepseek(messages, tools=None, temperature=0.3, max_tokens=4096)`

### 5.2 调用策略

- Sentinel：max_tokens=64，temperature=0.0
- Forge：不直接调用模型，仅规则注入
- Solver：max_tokens=4096，temperature 按分支设定
- Arbiter：max_tokens=2048，temperature=0.2
- 失败重试：指数退避，最多 3 次；网络超时单独捕获

### 5.3 成本控制（估算）

- 20 题 × 3 分支 × 约 4K tokens ≈ 240K tokens
- 按 deepseek-v4-pro 价格估算，总成本可控在数十元以内
- 如预算敏感，可关闭 Branch C 或降低 max_tokens

---

## 六、评估与评分

### 6.1 本地验证（利用已有分析报告）

由于提供的 20 题分析报告包含参考答案，开发阶段可用于：

- 快速验证 Agent 效果
- 调整 Prompt 和工具
- 计算 Accuracy 与 Calibration Error

### 6.2 官方指标

| 指标                | 说明      | 计算方式                            |
| ----------------- | ------- | ------------------------------- |
| Accuracy          | 最终答案正确率 | 正确题数 / 总题数                      |
| Calibration Error | 置信度校准误差 | 将 confidence 分桶，比较每桶平均置信度与实际正确率 |

### 6.3 评分脚本

参考 HLE 官方 `run_judge_results.py`，使用 `JUDGE_PROMPT` 让模型判定 `extracted_final_answer` 是否与 `correct_answer` 一致。

---

## 七、开发计划

| 阶段 | 任务                                  | 预计时间  |
| -- | ----------------------------------- | ----- |
| P1 | 搭建项目结构、API 封装、Loader/Exporter       | 0.5 天 |
| P2 | 实现 Sentinel + Forge                 | 0.5 天 |
| P3 | 实现 Solver Pool + ToolKit（Python 沙箱） | 1 天   |
| P4 | 实现 Arbiter + Confidence 校准          | 0.5 天 |
| P5 | 端到端跑通 20 题，生成分支结果与报告                | 1 天   |
| P6 | 根据本地验证结果迭代 Prompt 与工具               | 1 天   |
| P7 | 输出最终提交物：思路文档、代码、结果报告                | 0.5 天 |

---

## 八、待确认问题清单

1. **是否允许 Agent 调用 Python 代码执行工具？** 强烈建议开启，否则密码学/SageMath/代码题几乎无法正确解答。
2. **是否需要本地 Judge/评分逻辑？** 可用于开发期自评，但正式提交物是否包含？
3. **是否需要在代码中接入网络搜索工具？** 20 题基本为闭卷推理题，当前设计不强制依赖。
4. \*\* deepseek-v4-pro 是否支持 function calling？\*\* 若支持，工具调用采用原生 function call；若不支持，采用 "text2tool" 模式（模型输出 JSON 工具调用指令）。
5. **Confidence 是否完全由模型自报，还是由 Arbiter 统一计算？** 当前设计为 Arbiter 综合计算。

---

## 九、附录：针对 20 道题目的初步策略映射

| 题号    | 类型                        | 关键策略                   |
| ----- | ------------------------- | ---------------------- |
| 1     | exactMatch / 替换密码         | 频率分析 + Python 破译       |
| 2     | exactMatch / Shamir       | SageMath/自定义 GF(29) 插值 |
| 3     | exactMatch / 点阵           | Python 点阵识别            |
| 4     | exactMatch / SageMath 纠错  | 代码对比 + 执行验证            |
| 5-6   | multipleChoice / 网络安全     | 概念推理                   |
| 7     | multipleChoice / 机器人      | 运动学数值计算                |
| 8-10  | multipleChoice / ML       | 矩阵秩/水印/神经网络推理          |
| 11    | exactMatch / 信号处理         | DFT 运算量公式              |
| 12-13 | exactMatch / 异构架构         | 自定义类型语义分析              |
| 14    | exactMatch / C 语义         | 代码模拟执行                 |
| 15-16 | multipleChoice / RL+POMDP | 概念推理                   |
| 17    | exactMatch / 引力时间膨胀       | 分数类型计算                 |
| 18-19 | exactMatch / Python 语义    | Python 沙箱执行            |
| 20    | exactMatch / 领域驱动设计       | 概念推理                   |

---

**请审阅以上设计。如无重大调整，我将按此文档进入开发阶段。**
