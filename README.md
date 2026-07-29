# HLE 解题 Agent（deepseek-v4-pro）

针对 `HLE_text_only_20questions_student.jsonl` 20 道超高难度题目的自动解题 Agent。
参考 Fieldframe Labs HLE 方案的 Sentinel→Forge→Solver→Arbiter 流水线，全部使用
**deepseek-v4-pro** 单一模型（通过 `api.ai-native-x.site` 接入），领域知识检索改为 **RAG**。

## 架构

```
题目 JSONL
  → Sentinel     领域分类（deepseek-v4-pro，低成本）
  → RAG          从 knowledge_base/ 检索相关语料（TF-IDF，零依赖）
  → Solver Pool  3 个分支并行推理（同模型，不同角色 prompt + 温度）
       ├─ systematic  逐步严谨推理，优先代码验证
       ├─ intuitive   模式识别，快速定位陷阱
       └─ code_first  代码优先，强制 python_execute 实算
  → Arbiter      汇总多分支答案与置信度，校准后输出最终 response
  → 落盘         HLE_text_only_20questions_responses.jsonl + HLE_solver_report.md
```

每个 Solver 分支可调用工具：`python_execute`（受限沙箱，含 sympy/numpy）、`calculate`、`code_analyze`。

## 环境

- Python 3.13（managed），已建 venv：`.../python/envs/default`，含 sympy / numpy
- 运行命令统一用该 venv 的 python：
  `.../python/envs/default/Scripts/python.exe run.py`

## 运行

```bash
# 跑全部 20 题（支持断点续跑：已处理的 id 自动跳过）
python run.py

# 仅跑前 N 题（调试）
python run.py --limit 3

# 对照参考答案评估（Accuracy + Calibration Error）
python evaluate.py
```

## 配置（config.py）

- `API_KEY / BASE_URL / MODEL`：模型接入（默认已填你的 key，可用环境变量
  `HLE_API_KEY / HLE_BASE_URL / HLE_MODEL` 覆盖）
- `ENABLE_SENTINEL / ENABLE_RAG / ENABLE_TOOLS / ENABLE_ARBITER`：流水线开关
- `SOLVER_BRANCHES`：启用的分支
- `SOLVER_PARAMS.max_tokens`：推理模型需足够大（reasoning 消耗 token），默认 12000

> 重要：deepseek-v4-pro 是推理模型，`reasoning_content` 会占用大量 token，
> `max_tokens` 过小会导致最终答案被截断（content 为空）。已内置自动翻倍重试。

## 知识库（RAG）

把领域资料（`.txt/.md/.json`）放入 `knowledge_base/`，运行时会自动扫描、分块、
建立 TF-IDF 索引，作为参考材料注入 Solver。资料为空时退化为纯模型推理。
详见 `knowledge_base/README.md`。如需语义级向量检索，可在 `rag_retriever.py`
提供 embedding 函数并切换到 `VectorRAGRetriever`。

## 输出

- `HLE_text_only_20questions_responses.jsonl`：原始字段 + `response`
  （格式：`Explanation: ...\nAnswer: ...\nConfidence: ...%`）
- `HLE_solver_report.md`：每题域分类、RAG 命中、分支结果、仲裁理由
- `_meta` 字段（不导出到提交物，仅供调试）：domain / rag_hits / branches / confidence / elapsed

## 已知限制

- 单模型多分支的投票增益有限（FF-STACK 原方案用跨厂商模型配对）。
- 速度偏慢：单题约 10–15 分钟（推理模型 + 多分支 + 工具循环）。
- 极难题（如两步替换密码）依赖模型实际解密能力，可能答错但格式始终合规。
