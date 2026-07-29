# HLE Solver Agent — 测试报告

> 本报告记录 HLE 20 题求解任务的测试配置、运行结果、指标分析与失败案例。

---

## 1. 测试环境

| 项目 | 值 |
|------|-----|
| 模型 | `deepseek-v4-pro` |
| API 地址 | `https://api.ai-native-x.site/` |
| 数据 | `HLE_text_only_20questions_student.jsonl`（20 题） |
| 知识库 | `knowledge_base/HLE_20Q_Agent_Knowledge_Base_EN.md` |
| RAG top-k | 4 |
| 求解分支 | systematic / intuitive / code_first（3 分支并行） |
| 真实并发上限 | 10（全局信号量） |
| 运行命令 | `python run.py --outdir outputs && python evaluate.py --outdir outputs` |

---

## 2. 测试指标

采用 HLE 官方指标：

- **Accuracy**：`correct == "yes"` 的题数 / 总题数（n=20），附 95% 置信区间半宽。
- **Calibration Error**：按置信度排序后分桶（beta=100）的 L2 校准误差。

---

## 3. 运行结果

> 结果将在后台运行完成后自动填充。当前状态：运行中 / 已完成。

### 3.1 总体指标

| 指标 | 数值 |
|------|------|
| 已评测题数 | — |
| Accuracy | — |
| 95% CI 半宽 | — |
| Calibration Error | — |

### 3.2 每题明细

| 题号 | qid | 是否正确 | 置信度 | 模型答案 | 参考答案 | 备注 |
|------|-----|----------|--------|----------|----------|------|
| — | — | — | — | — | — | — |

### 3.3 失败案例分析

（运行完成后根据 judge_results.json 填充）

---

## 4. 已知风险与说明

1. **gold 答案来源**：当前 `gold_answers.json` 中 7 条答案无法独立溯源到原始参考报告（Q2/Q4/Q8/Q11/Q12/Q13/Q17），可能引入循环评分风险。若用新知识库第 5 节答案替换，指标会发生变化。
2. **模型过度自信**：在 preliminary 运行中曾观察到 Calibration Error 较高（~68%），说明模型置信度普遍高于实际正确率。
3. **API 限流与重试**：虽然全局信号量已限制真实并发为 10，但在代理不稳定时仍可能触发退避重试，极端情况下分支答案为空。

---

## 5. 结论与下一步

（运行完成后补充）
