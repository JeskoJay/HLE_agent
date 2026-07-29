"""
本地评估：对照 gold_answers.json，用 deepseek-v4-pro 做答案判定（符合 HLE 官方方法）。
计算：
  - Accuracy：最终答案正确率
  - Calibration Error：置信度校准误差（分桶）
用法：
    python evaluate.py                       # 评 config.OUTPUT_FILE
    python evaluate.py --outdir outputs      # 评 outputs/responses.jsonl，写 outputs/EVAL_SUMMARY.md
    python evaluate.py --resp path.jsonl     # 指定任意 responses 文件
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hle_agent"))
import config
import api_client
import loader

GOLD = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_answers.json")))

JUDGE_PROMPT = """You are a strict grader for Humanity's Last Exam (HLE).
Given the QUESTION, the CORRECT ANSWER, and the MODEL'S ANSWER, decide whether the
model's answer is CORRECT (semantically equivalent / contains the required answer).
Rules:
- For multiple-choice, the model is correct if its selected letter(s) match exactly (order-insensitive).
- For exactMatch, the model is correct if it conveys the same value/result, allowing minor formatting differences.
- Be strict but fair. If unsure, mark incorrect.
Respond with ONLY: {"correct": true} or {"correct": false}."""


def judge(question, correct, model_answer):
    msgs = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content":
            f"QUESTION:\n{question[:2000]}\n\nCORRECT ANSWER:\n{correct}\n\nMODEL'S ANSWER:\n{model_answer}"},
    ]
    for _ in range(config.MAX_RETRIES):
        try:
            r = api_client.chat(msgs, temperature=0.0, max_tokens=200)
            text = r["content"] or r["reasoning_content"]
            return "true" in text.lower() and "false" not in text.lower().replace("true", "", 1)
        except Exception:
            continue
    return False


def extract_answer(response: str) -> str:
    import re
    m = re.search(r"Answer\s*:\s*(.+?)(?=\n\s*Confidence\s*:|\Z)", response, re.S | re.I)
    return m.group(1).strip() if m else response


def main():
    ap = argparse.ArgumentParser(description="Evaluate HLE responses")
    ap.add_argument("--outdir", default=None,
                    help="评估某个输出文件夹（读 <outdir>/responses.jsonl，写 <outdir>/EVAL_SUMMARY.md）")
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
    results = []
    for rec in records:
        qid = rec["id"]
        gold = GOLD.get(qid)
        if not gold:
            continue
        ans = extract_answer(rec.get("response", ""))
        correct = judge(rec.get("question", ""), gold, ans)
        conf = (rec.get("_meta") or {}).get("final_confidence") or 0
        results.append((qid, correct, conf, ans, gold))

    out = []
    n = len(results)
    if n == 0:
        out.append("No graded items.")
        _emit(out, args.outdir)
        return
    acc = sum(1 for _, c, _, _, _ in results if c) / n
    out.append(f"Graded {n} items. Accuracy = {acc*100:.1f}%")

    buckets = {}
    for _, c, conf, _, _ in results:
        b = int(conf // 10) * 10
        buckets.setdefault(b, []).append(c)
    out.append("\nConfidence bucket | accuracy | n")
    total_err = 0.0
    total_n = 0
    for b in sorted(buckets):
        vals = buckets[b]
        a = sum(1 for x in vals if x) / len(vals)
        out.append(f"  {b:3d}-{b+9:3d}%      |  {a*100:5.1f}%   | {len(vals)}")
        total_err += abs(b / 100.0 - a) * len(vals)
        total_n += len(vals)
    if total_n:
        out.append(f"\nCalibration Error (weighted) = {total_err/total_n*100:.1f}%")

    out.append("\n--- per item ---")
    for qid, c, conf, ans, gold in results:
        out.append(f"[{'OK' if c else 'XX'}] {qid[:8]} conf={conf}% model='{ans[:40]}' gold='{gold[:40]}'")

    _emit(out, args.outdir)
    print("\n".join(out))


def _emit(lines, outdir):
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "EVAL_SUMMARY.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[summary written to {os.path.join(outdir, 'EVAL_SUMMARY.md')}]")


if __name__ == "__main__":
    main()
