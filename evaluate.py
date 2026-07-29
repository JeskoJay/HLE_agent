"""
本地评估：对照 gold_answers.json，用 deepseek-v4-pro 做答案判定（符合 HLE 官方方法）。
计算：
  - Accuracy：最终答案正确率
  - Calibration Error：置信度校准误差（分桶）
用法：python evaluate.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hle_agent"))
import config
import api_client
import loader

GOLD = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_answers.json")))
RESP_PATH = config.OUTPUT_FILE

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
    # 从最终 response 的 Answer: 行提取
    import re
    m = re.search(r"Answer\s*:\s*(.+?)(?=\n\s*Confidence\s*:|\Z)", response, re.S | re.I)
    return m.group(1).strip() if m else response


def main():
    if not os.path.exists(RESP_PATH):
        print("No responses file found. Run the agent first.")
        return
    records = loader.load_questions(RESP_PATH)
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

    n = len(results)
    if n == 0:
        print("No graded items.")
        return
    acc = sum(1 for _, c, _, _, _ in results if c) / n
    print(f"Graded {n} items. Accuracy = {acc*100:.1f}%")

    # Calibration error：10% 分桶
    buckets = {}
    for _, c, conf, _, _ in results:
        b = int(conf // 10) * 10
        buckets.setdefault(b, []).append(c)
    print("\nConfidence bucket | accuracy | n")
    total_err = 0.0
    total_n = 0
    for b in sorted(buckets):
        vals = buckets[b]
        a = sum(1 for x in vals if x) / len(vals)
        print(f"  {b:3d}-{b+9:3d}%      |  {a*100:5.1f}%   | {len(vals)}")
        total_err += abs(b/100.0 - a) * len(vals)
        total_n += len(vals)
    if total_n:
        print(f"\nCalibration Error (weighted) = {total_err/total_n*100:.1f}%")

    # 明细
    print("\n--- per item ---")
    for qid, c, conf, ans, gold in results:
        print(f"[{'OK' if c else 'XX'}] {qid[:8]} conf={conf}% model='{ans[:40]}' gold='{gold[:40]}'")


if __name__ == "__main__":
    main()
