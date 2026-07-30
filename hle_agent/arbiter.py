"""
Arbiter：汇总多分支结果，仲裁分歧，输出最终 response（含置信度校准）。
"""
import re

import config
import api_client
import solver
import stream_logger

ARBITER_SYSTEM = """You are the Arbiter for an HLE (Humanity's Last Exam) solving system.
Several independent solver branches produced candidate answers for the same problem.
Your job:
1. Compare their reasoning and answers.
2. Identify which is most reliable (correctness, consistency, verification by tools).
3. Produce the FINAL answer in the exact format:

Explanation: <concise synthesis of the best reasoning>
Answer: <final answer, concise>
Confidence: <integer 0-100>

Calibration guidance for Confidence:
- If all branches agree and at least one used tool verification: 90-98.
- If all branches agree without verification: 75-90.
- If branches disagree but one is clearly best-supported: 55-80.
- If branches disagree and none is clearly supported: 35-55.
- If the problem is genuinely ambiguous/unknown: 20-40.
Be honest; do not inflate confidence."""


def _normalize(a: str) -> str:
    a = (a or "").strip().lower()
    a = re.sub(r"\s+", "", a)
    # 纯字母选项排序（选择题）
    if a and all(c.isalpha() for c in a):
        a = "".join(sorted(a))
    return a


def is_consistent(branches: list) -> bool:
    """三分支归一化后答案是否完全一致（供分歧深挖判断使用）。"""
    norm_answers = [_normalize(b.get("answer", "")) for b in branches]
    return len(set(norm_answers)) == 1 and all(norm_answers)


def arbitrate(question: str, domain: str, branches: list, rag_context: str = "",
              extra_evidence: str = "") -> dict:
    """
    branches: solver.parse_response 结果列表，每个含 name/answer/explanation/confidence/reasoning/tool_log
    extra_evidence: 分歧深挖（debate）产出的针对性验证结论，作为额外证据注入仲裁
    返回 {"response": str, "final_answer": str, "final_confidence": int, "reason": str}
    """
    # 一致性快速判断
    norm_answers = [_normalize(b.get("answer", "")) for b in branches]
    consistent = len(set(norm_answers)) == 1 and all(norm_answers)

    if not config.ENABLE_ARBITER or len(branches) == 1:
        # 无仲裁：直接取第一个/置信度最高分支
        best = max(branches, key=lambda b: b.get("confidence") or 0)
        return _build_direct(best, consistent, branches)

    # 构造仲裁输入
    branch_text = []
    for i, b in enumerate(branches, 1):
        branch_text.append(
            f"Branch {i} ({b['name']}, self-confidence {b.get('confidence')}%):\n"
            f"Answer: {b.get('answer')}\n"
            f"Reasoning: {b.get('explanation') or b.get('reasoning','')[:1500]}\n"
            f"Tool verification: {'yes' if b.get('tool_log') else 'no'}"
        )
    user = f"Domain: {domain}\n\n"
    if rag_context:
        user += f"Reference material:\n{rag_context}\n\n"
    user += "Problem:\n" + question + "\n\nCandidate answers:\n" + "\n\n".join(branch_text)
    user += "\n\nAll branches agree: " + ("YES" if consistent else "NO")
    if extra_evidence:
        user += ("\n\nTargeted disagreement-resolution findings "
                 "(independent focused verification, weigh heavily):\n" + extra_evidence)
    user += "\n\nProduce the FINAL answer now."

    messages = [
        {"role": "system", "content": ARBITER_SYSTEM},
        {"role": "user", "content": user},
    ]
    stream_logger.section("ARBITER", "汇总多分支并仲裁最终答案…")
    result = api_client.chat(messages, temperature=config.ARBITER_PARAMS["temperature"],
                             max_tokens=config.ARBITER_PARAMS["max_tokens"],
                             stream=True,
                             on_token=lambda kind, t: stream_logger.token(t))
    parsed = solver.parse_response(result["content"], result.get("reasoning_content", ""))
    reason = result["reasoning_content"] or result["content"]
    resp = f"Explanation: {parsed['explanation']}\nAnswer: {parsed['answer']}\nConfidence: {parsed['confidence'] if parsed['confidence'] is not None else 70}%"
    return {
        "response": resp,
        "final_answer": parsed["answer"],
        "final_confidence": parsed["confidence"] if parsed["confidence"] is not None else 70,
        "reason": reason[:1500],
        "arbiter_content": result["content"],
    }


def _build_direct(best: dict, consistent: bool, branches: list) -> dict:
    base = best.get("confidence") or 60
    if consistent:
        base = min(100, base + 10)
    resp = f"Explanation: {best.get('explanation')}\nAnswer: {best.get('answer')}\nConfidence: {base}%"
    return {
        "response": resp,
        "final_answer": best.get("answer"),
        "final_confidence": base,
        "reason": "Direct (no arbiter): took highest-confidence branch." if len(branches) > 1
                  else "Single branch output.",
        "arbiter_content": best.get("raw", ""),
    }
