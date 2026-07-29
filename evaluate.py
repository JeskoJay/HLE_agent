"""
本地评估：对照 gold_answers.json，使用 deepseek-v4-pro 作为 judge 进行 HLE 官方式评判。

参考官方实现：
  https://github.com/centerforaisafety/hle/blob/main/hle_eval/run_judge_results.py

指标：
  - Accuracy：最终答案正确率（按 n=20 计算，含 95% 置信区间半宽）
  - Calibration Error：置信度校准误差（官方 sorted-bucket L2 方法，beta=100）

输出：
  - <outdir>/EVAL_SUMMARY.md   人类可读摘要
  - <outdir>/judge_results.json 每题详细 judge 结果（可复用、可审计）

用法：
    python evaluate.py                       # 评 config.OUTPUT_FILE
    python evaluate.py --outdir outputs      # 评 outputs/responses.jsonl
    python evaluate.py --resp path.jsonl     # 指定任意 responses 文件
"""
import os
import sys
import json
import math
import re
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hle_agent"))
import config
import api_client
import loader

GOLD = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_answers.json")))

# 与官方脚本对齐的 judge prompt（强制 JSON 输出）
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer exactly 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer exactly 'no' otherwise.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.

Respond ONLY with a single JSON object in this exact format (no markdown, no extra text):
{{"extracted_final_answer": "...", "reasoning": "...", "correct": "yes|no", "confidence": 85}}"""


def _extract_confidence(response: str) -> int:
    """从 response 字符串中提取 0-100 的置信度；未找到则返回 100（与官方一致）。"""
    if not response:
        return 100
    # 匹配 Confidence: 85% / Confidence: 85 / confidence=85 等常见写法
    m = re.search(r"[Cc]onfidence\s*[:=]?\s*(\d+)(?:\s*%)?", response)
    if m:
        v = int(m.group(1))
        return max(0, min(100, v))
    return 100


def _extract_answer(response: str) -> str:
    """从 agent response 中抽取最终答案；兼容旧格式 Answer: ..."""
    if not response:
        return ""
    m = re.search(r"Answer\s*:\s*(.+?)(?=\n\s*(?:Confidence\s*:|$))", response, re.S | re.I)
    if m:
        return m.group(1).strip()
    return response.strip()


def _parse_json_object(text: str):
    """从模型输出中抓取第一个 JSON 对象。"""
    if not text:
        return None
    # 去掉可能的 markdown 代码块
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    # 找第一个 { ... }
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge(question: str, correct_answer: str, response: str, model_answer_hint: str = ""):
    """
    调用 deepseek-v4-pro 做 judge，返回 dict。
    若结构化输出失败，则 fallback 到简单二元判断。
    """
    prompt = JUDGE_PROMPT.format(
        question=question[:2500],
        response=response[:4000],
        correct_answer=correct_answer,
    )
    msgs = [{"role": "user", "content": prompt}]

    for attempt in range(config.MAX_RETRIES):
        try:
            r = api_client.chat(msgs, temperature=0.0, max_tokens=1024)
            text = r.get("content") or r.get("reasoning_content") or ""
            obj = _parse_json_object(text)
            if obj and isinstance(obj, dict) and "correct" in obj:
                correct_flag = str(obj.get("correct", "")).lower()
                is_correct = correct_flag == "yes"
                extracted = str(obj.get("extracted_final_answer", model_answer_hint or _extract_answer(response)))
                reasoning = str(obj.get("reasoning", ""))
                conf = obj.get("confidence")
                try:
                    confidence = max(0, min(100, int(conf)))
                except (TypeError, ValueError):
                    confidence = _extract_confidence(response)
                return {
                    "correct_answer": correct_answer,
                    "model_answer": extracted,
                    "reasoning": reasoning,
                    "correct": "yes" if is_correct else "no",
                    "confidence": confidence,
                }
            # 退化 fallback：看文本里有没有 yes/no
            lowered = text.lower()
            is_correct = '"correct": true' in lowered or '"correct": "yes"' in lowered or (
                "yes" in lowered and "no" not in lowered.replace("not", "", 1)
            )
            return {
                "correct_answer": correct_answer,
                "model_answer": model_answer_hint or _extract_answer(response),
                "reasoning": text[:500],
                "correct": "yes" if is_correct else "no",
                "confidence": _extract_confidence(response),
            }
        except Exception as e:
            if attempt == config.MAX_RETRIES - 1:
                # 最终失败：保守判错
                return {
                    "correct_answer": correct_answer,
                    "model_answer": model_answer_hint or _extract_answer(response),
                    "reasoning": f"judge error after {config.MAX_RETRIES} retries: {e}",
                    "correct": "no",
                    "confidence": _extract_confidence(response),
                }
            continue


def calib_err(confidence, correct, p="2", beta=100):
    """
    官方 Calibration Error 实现（来源：HLE run_judge_results.py）。
    confidence / correct 均为 numpy 数组；confidence 为 0-1 浮点，correct 为 bool/0-1。
    """
    import numpy as np
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]

    # beta 是目标桶大小；若样本少于 beta，则只有一桶
    n = len(confidence)
    if n == 0:
        return 0.0
    num_bins = max(1, n // beta)
    bins = [[i * beta, (i + 1) * beta] for i in range(num_bins)]
    bins[-1] = [bins[-1][0], n]

    cerr = 0.0
    for i in range(len(bins) - 1):
        lo, hi = bins[i]
        if hi > n:
            hi = n
        if lo >= hi:
            continue
        bin_conf = confidence[lo:hi]
        bin_corr = correct[lo:hi]
        num_examples = len(bin_conf)
        if num_examples == 0:
            continue
        diff = abs(np.nanmean(bin_conf) - np.nanmean(bin_corr))
        if p == "2":
            cerr += num_examples / n * (diff ** 2)
        elif p == "1":
            cerr += num_examples / n * diff
        elif p in ("infty", "infinity", "max"):
            cerr = max(cerr, diff)
        else:
            raise ValueError("p must be '1', '2', or 'infty'")

    if p == "2":
        cerr = math.sqrt(cerr)
    return cerr


def dump_metrics(predictions, n_total):
    """输出 Accuracy / Calibration Error，与官方对齐。"""
    import numpy as np
    correct_list = []
    conf_list = []
    for qid, rec in predictions.items():
        jr = rec.get("judge_response")
        if not jr:
            continue
        correct_list.append("yes" in str(jr.get("correct", "")).lower())
        conf_list.append(jr.get("confidence", 100))

    correct = np.array(correct_list, dtype=bool)
    confidence = np.array(conf_list, dtype=float) / 100.0

    available = len(correct)
    accuracy = round(100 * sum(correct) / n_total, 2) if n_total else 0.0
    half_width = round(1.96 * math.sqrt(accuracy * (100 - accuracy) / n_total), 2) if n_total else 0.0
    cal_err = 100 * round(calib_err(confidence, correct, p="2", beta=100), 4)

    return {
        "available": available,
        "n_total": n_total,
        "accuracy": accuracy,
        "accuracy_half_width": half_width,
        "calibration_error": cal_err,
        "correct_count": int(sum(correct)),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate HLE responses with official-style judge")
    ap.add_argument("--outdir", default=None,
                    help="评估某个输出文件夹（读 <outdir>/responses.jsonl，写 <outdir>/EVAL_SUMMARY.md 和 judge_results.json）")
    ap.add_argument("--resp", default=None, help="直接指定 responses.jsonl 路径（优先）")
    args = ap.parse_args()

    if args.resp:
        resp_path = args.resp
    elif args.outdir:
        resp_path = os.path.join(args.outdir, "responses.jsonl")
    else:
        resp_path = config.OUTPUT_FILE

    if not os.path.exists(resp_path):
        print(f"No responses file found at {resp_path}. Run the agent first.")
        return

    records = loader.load_questions(resp_path)
    judged = {}

    for rec in records:
        qid = rec.get("id")
        gold = GOLD.get(qid)
        if not gold or not qid:
            continue
        question_text = rec.get("question", "")
        response_text = rec.get("response", "")
        model_answer = _extract_answer(response_text)

        jr = judge(question_text, gold, response_text, model_answer_hint=model_answer)
        judged[qid] = {
            "id": qid,
            "question": question_text,
            "response": response_text,
            "judge_response": jr,
        }

    # 指标计算
    n_total = len(records)
    metrics = dump_metrics(judged, n_total)

    # 组装报告
    out = []
    out.append("# HLE Evaluation Summary")
    out.append("")
    out.append(f"- Graded items: {metrics['available']} / {metrics['n_total']}")
    out.append(f"- Accuracy: **{metrics['accuracy']}%** +/- {metrics['accuracy_half_width']}% (95% CI, n={metrics['n_total']})")
    out.append(f"- Calibration Error (official sorted-bucket L2, beta=100): **{metrics['calibration_error']}%**")
    out.append("")

    # 分桶可视化（10% 桶，便于阅读；与官方算法无关）
    buckets = {i: [] for i in range(0, 101, 10)}
    for rec in judged.values():
        jr = rec["judge_response"]
        conf = jr.get("confidence", 100)
        b = int(conf // 10) * 10
        buckets[b].append("yes" in str(jr.get("correct", "")).lower())
    out.append("## Confidence bucket (10% bins)")
    out.append("")
    out.append("| bucket | accuracy | n |")
    out.append("|--------|----------|---|")
    for b in sorted(buckets):
        vals = buckets[b]
        if not vals:
            out.append(f"| {b:3d}-{b+9:3d}% |    -     | 0 |")
        else:
            a = sum(vals) / len(vals) * 100
            out.append(f"| {b:3d}-{b+9:3d}% | {a:6.1f}%  | {len(vals)} |")
    out.append("")

    out.append("## Per-item detail")
    out.append("")
    out.append("| qid | correct | conf | model_answer | gold_answer | reasoning |")
    out.append("|-----|---------|------|--------------|-------------|-----------|")
    for rec in judged.values():
        qid = rec["id"][:8]
        jr = rec["judge_response"]
        corr = "yes" if "yes" in str(jr.get("correct", "")).lower() else "no"
        conf = jr.get("confidence", 100)
        ma = str(jr.get("model_answer", "")).replace("|", "\\|")[:60]
        ga = str(jr.get("correct_answer", "")).replace("|", "\\|")[:60]
        reason = str(jr.get("reasoning", "")).replace("\n", " ").replace("|", "\\|")[:120]
        out.append(f"| {qid} | {corr} | {conf}% | {ma} | {ga} | {reason} |")

    report = "\n".join(out) + "\n"

    # 输出
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        summary_path = os.path.join(args.outdir, "EVAL_SUMMARY.md")
        judge_path = os.path.join(args.outdir, "judge_results.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(report)
        with open(judge_path, "w", encoding="utf-8") as f:
            json.dump(judged, f, ensure_ascii=False, indent=2)
        print(f"[summary written to {summary_path}]")
        print(f"[judge results written to {judge_path}]")

    print(report)


if __name__ == "__main__":
    main()
