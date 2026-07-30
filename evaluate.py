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

# judge 输入上限：HLE 官方脚本不截断 response。我们的 response 可能很长
# （Agent 把全部推理塞在 Answer: 之后，最终答案在末尾），必须让 judge 看到
# 完整内容才能抽到最终答案。设一个安全上限避免超出模型上下文即可。
MAX_QUESTION_CHARS = 12000
MAX_RESPONSE_CHARS = 120000

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


def _quasi_exact_match(model_ans: str, gold: str) -> bool:
    """准精确匹配（GAIA 思想：P1-§8）：归一化后比较，缓解格式误判。
    处理：小写、去空白/标点、去常见单位、选择题集合排序、数值容差近似。
    作为 LLM judge 之外的辅助确定性指标。"""
    if not model_ans or not gold:
        return False

    def norm(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[.,;:]", "", s)
        s = re.sub(r"\b(km|cm|mm|kg|g|mg|hz|degrees?|radians?|bytes?|bits?|seconds?|minutes?|hours?)\b", "", s)
        s = re.sub(r"\s+", "", s)
        if s and all(c.isalpha() for c in s):
            s = "".join(sorted(s))
        return s

    if norm(model_ans) == norm(gold):
        return True
    # 数值近似（容差 max(1e-6, 1e-3*|gold|)）
    try:
        fm = float(re.sub(r"[^0-9.\-eE]", "", model_ans))
        fg = float(re.sub(r"[^0-9.\-eE]", "", gold))
        if abs(fm - fg) <= max(1e-6, 1e-3 * abs(fg)):
            return True
    except (ValueError, TypeError):
        pass
    return False


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
        question=question[:MAX_QUESTION_CHARS],
        response=response[:MAX_RESPONSE_CHARS],
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

    # beta 是目标桶大小；当样本数不足 beta 时只有一桶。
    # 官方脚本面向大规模数据集，len(bins)-1 循环在 n<beta 时会空转，
    # 因此这里显式处理单桶情况，确保小样本也能得到有效指标。
    n = len(confidence)
    if n == 0:
        return 0.0
    num_bins = max(1, n // beta)
    bins = [[i * beta, (i + 1) * beta] for i in range(num_bins)]
    bins[-1] = [bins[-1][0], n]

    cerr = 0.0
    if num_bins == 1:
        # 单桶：全量计算平均置信度与平均正确率的差异
        diff = abs(np.nanmean(confidence) - np.nanmean(correct))
        if p == "2":
            cerr = diff
        elif p == "1":
            cerr = diff
        elif p in ("infty", "infinity", "max"):
            cerr = diff
        else:
            raise ValueError("p must be '1', '2', or 'infty'")
    else:
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

    qem_list = []
    for qid, rec in predictions.items():
        if rec.get("judge_response"):
            qem_list.append(bool(rec.get("quasi_exact_match")))
    qem_acc = round(100 * sum(qem_list) / n_total, 2) if n_total else 0.0

    return {
        "available": available,
        "n_total": n_total,
        "accuracy": accuracy,
        "accuracy_half_width": half_width,
        "calibration_error": cal_err,
        "correct_count": int(sum(correct)),
        "quasi_exact_accuracy": qem_acc,
    }


def layered_metrics(judged):
    """P1-§8 分层指标：按 answer_type 报准确率 + 95% CI + 准精确匹配率。"""
    from collections import defaultdict
    groups = defaultdict(list)
    for qid, rec in judged.items():
        jr = rec.get("judge_response")
        if not jr:
            continue
        at = rec.get("answer_type") or "unknown"
        correct = "yes" in str(jr.get("correct", "")).lower()
        qem = bool(rec.get("quasi_exact_match"))
        groups[at].append((correct, qem))
    rows = []
    for at in sorted(groups):
        items = groups[at]
        n = len(items)
        acc = 100 * sum(c for c, _ in items) / n if n else 0.0
        qem_acc = 100 * sum(q for _, q in items) / n if n else 0.0
        half = round(1.96 * math.sqrt(acc * (100 - acc) / n), 1) if n else 0.0
        rows.append((at, n, round(acc, 1), half, round(qem_acc, 1)))
    return rows


def reliability_diagram(judged):
    """P1-§8 可靠性图数据：置信度 10% 分桶 -> 桶内平均置信度 vs 实际正确率。"""
    buckets = {b: [0, 0, 0.0] for b in range(0, 101, 10)}  # [n, correct_n, sum_conf]
    for qid, rec in judged.items():
        jr = rec.get("judge_response")
        if not jr:
            continue
        conf = jr.get("confidence", 100)
        b = int(conf // 10) * 10
        buckets[b][0] += 1
        if "yes" in str(jr.get("correct", "")).lower():
            buckets[b][1] += 1
        buckets[b][2] += conf
    rows = []
    for b in sorted(buckets):
        n, c, sc = buckets[b]
        if n == 0:
            rows.append((b, 0, 0.0, 0.0))
        else:
            rows.append((b, n, round(sc / n, 1), round(100 * c / n, 1)))
    return rows


def _save_reliability_assets(outdir, judged):
    """P1-§8：保存可靠性图数据 CSV，并尝试画 PNG（matplotlib 可用时）。"""
    import csv
    rows = reliability_diagram(judged)
    csv_path = os.path.join(outdir, "reliability_diagram.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["conf_bucket_lo", "n", "avg_conf", "accuracy"])
        for b, n, avg_conf, acc in rows:
            w.writerow([b, n, avg_conf, acc])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pts = [(b, ac, a) for b, n, ac, a in rows if n]
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 100], [0, 100], "--", color="gray", label="perfect calibration")
        ax.scatter(xs, ys, color="tab:blue", zorder=3, label="model")
        ax.set_xlabel("Average predicted confidence (%)")
        ax.set_ylabel("Actual accuracy (%)")
        ax.set_title("Reliability Diagram")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "reliability_diagram.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  [reliability plot] skipped (matplotlib unavailable): {e}")


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

    import concurrent.futures as _futures

    def _judge_one(rec):
        qid = rec.get("id")
        gold = GOLD.get(qid)
        if not gold or not qid:
            return None
        question_text = rec.get("question", "")
        response_text = rec.get("response", "")
        model_answer = _extract_answer(response_text)
        jr = judge(question_text, gold, response_text, model_answer_hint=model_answer)
        return qid, {
            "id": qid,
            "answer_type": rec.get("answer_type", ""),
            "question": question_text,
            "response": response_text,
            "judge_response": jr,
            "quasi_exact_match": _quasi_exact_match(model_answer, gold),
        }

    with _futures.ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_REQUESTS) as ex:
        futures = [ex.submit(_judge_one, rec) for rec in records]
        for fut in _futures.as_completed(futures):
            res = fut.result()
            if res:
                qid, payload = res
                judged[qid] = payload

    # 指标计算
    n_total = len(records)
    metrics = dump_metrics(judged, n_total)

    # 组装报告
    out = []
    out.append("# HLE Evaluation Summary")
    out.append("")
    out.append(f"- Graded items: {metrics['available']} / {metrics['n_total']}")
    out.append(f"- Accuracy (LLM judge): **{metrics['accuracy']}%** +/- {metrics['accuracy_half_width']}% (95% CI, n={metrics['n_total']})")
    out.append(f"- Accuracy (quasi-exact match, deterministic): **{metrics['quasi_exact_accuracy']}%**")
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

    # 分层指标（P1-§8）
    out.append("## Layered metrics (by answer_type)")
    out.append("")
    out.append("| answer_type | n | accuracy | 95% CI half | quasi-exact |")
    out.append("|-------------|---|----------|-------------|------------|")
    for at, n, acc, half, qem in layered_metrics(judged):
        out.append(f"| {at} | {n} | {acc}% | ±{half}% | {qem}% |")
    out.append("")

    # 可靠性图（P1-§8）
    out.append("## Reliability diagram (predicted confidence vs actual accuracy)")
    out.append("")
    out.append("| conf bucket | n | avg_conf | accuracy |")
    out.append("|-------------|---|----------|----------|")
    for b, n, avg_conf, acc in reliability_diagram(judged):
        out.append(f"| {b:3d}-{b+9:3d}% | {n} | {avg_conf}% | {acc}% |")
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
        _save_reliability_assets(args.outdir, judged)
        print(f"[summary written to {summary_path}]")
        print(f"[judge results written to {judge_path}]")

    print(report)


if __name__ == "__main__":
    main()
