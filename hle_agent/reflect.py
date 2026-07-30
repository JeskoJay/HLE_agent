"""
P0-§2 Reflection 反思闭环：在 Arbiter 之后加 Critic + Refine 双步。

Critic 独立评审三个 Solver 分支答案与 Arbiter 的仲裁答案，指出事实错误 /
逻辑漏洞 / 数值偏差 / 完整性问题（对应我们观察到的"多选漏判元素""数值末位偏差"）。
Refiner 结合 Critic 反馈产出 final_answer_v2，并给出更诚实的置信度校准
（模型自报往往过度自信，Critic 指出真实错误时应下调）。

安全阀：REFLECT_MAX_ROUNDS（默认 1，即 Critic+Refine 一次）。
"""
import config
import api_client
import solver
import stream_logger


CRITIC_SYSTEM = """You are a rigorous Critic reviewing candidate answers to a problem from Humanity's Last Exam (HLE).
Several solver branches and an Arbiter produced candidate answers. Your job is to find ERRORS, not to solve from scratch.

For each candidate and for the Arbiter's proposed final answer, identify:
- Factual errors (wrong claims, misidentified entities, incorrect definitions)
- Logical flaws (invalid inference, missing cases, wrong assumption, fallacious step)
- Numerical/computation mistakes (off-by-one, unit errors, sign errors, truncation, wrong formula)
- Completeness issues (multiple-choice: missing options in a set; exactMatch: missing/extra terms; wrong ordering)

Be specific and concise. Cite which candidate the issue applies to. If an answer is solid, say so explicitly.
Output a structured review, ending with "OVERALL: <list the most reliable candidate(s) or 'none'>"."""

REFINE_SYSTEM = """You are the Refiner. You are given the original problem, candidate answers from multiple solver branches,
a Critic's review of their errors, and the Arbiter's proposed final answer.

Produce the BEST final answer by correcting the identified errors. Output exactly in this format:

Explanation: <concise synthesis, noting what was corrected and why>
Answer: <final answer, concise>
Confidence: <integer 0-100>

Calibration guidance for Confidence:
- If candidates strongly agree and verification supports it: 90-98.
- If agreement after correction but some residual uncertainty: 70-90.
- If candidates disagree and the best is only weakly supported: 45-70.
- If genuinely ambiguous/unknown: 20-40.
Be honest; do NOT inflate confidence. Model self-reported confidence is often overconfident, so adjust DOWN when the Critic found real errors."""


def _branch_summary(branches):
    lines = []
    for i, b in enumerate(branches, 1):
        ans = b.get("answer")
        exp = (b.get("explanation") or b.get("reasoning") or "")[:1200]
        lines.append(f"Branch {i} ({b['name']}, self-confidence {b.get('confidence')}%):\n"
                     f"Answer: {ans}\nReasoning: {exp}")
    return "\n\n".join(lines)


def _critique(question, domain, branches, arbiter_out, rag_context):
    user = f"Domain: {domain}\n\nProblem:\n{question}\n\n"
    if rag_context:
        user += f"Reference material:\n{rag_context}\n\n"
    user += "Candidate answers:\n" + _branch_summary(branches)
    user += f"\n\nArbiter proposed final answer: {arbiter_out.get('final_answer')}\n\n" \
            f"Critique the answers above. Be specific about any errors."
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": user},
    ]
    stream_logger.section("REFLECT", "Critic 评审三分支与仲裁答案…")
    result = api_client.chat(messages, temperature=0.2, max_tokens=4000,
                             stream=True, on_token=lambda k, t: stream_logger.token(t))
    return result.get("content") or ""


def _refine(question, domain, branches, arbiter_out, critique, rag_context):
    user = f"Domain: {domain}\n\nProblem:\n{question}\n\n"
    if rag_context:
        user += f"Reference material:\n{rag_context}\n\n"
    user += "Candidate answers:\n" + _branch_summary(branches)
    user += f"\n\nArbiter proposed final answer: {arbiter_out.get('final_answer')}\n\n"
    user += f"Critic's review:\n{critique}\n\nProduce the corrected BEST final answer now."
    messages = [
        {"role": "system", "content": REFINE_SYSTEM},
        {"role": "user", "content": user},
    ]
    stream_logger.section("REFLECT", "Refiner 结合批判产出 final_answer_v2…")
    result = api_client.chat(messages, temperature=0.2,
                             max_tokens=config.ARBITER_PARAMS["max_tokens"],
                             stream=True, on_token=lambda k, t: stream_logger.token(t))
    parsed = solver.parse_response(result.get("content", ""),
                                   result.get("reasoning_content", ""))
    conf = parsed["confidence"] if parsed["confidence"] is not None else 70
    resp = (f"Explanation: {parsed['explanation']}\n"
            f"Answer: {parsed['answer']}\n"
            f"Confidence: {conf}%")
    return {
        "response": resp,
        "final_answer": parsed["answer"],
        "final_confidence": conf,
        "reason": (result.get("reasoning_content") or result.get("content") or "")[:1500],
        "critique": critique,
        "refine_content": result.get("content") or "",
    }


def improve(question, domain, branches, arbiter_out, rag_context=""):
    """
    反思闭环主入口：Critic -> Refine。
    返回与 arbiter.arbitrate 相同结构的 dict（可无缝替换 response）。
    """
    if not config.ENABLE_REFLECTION:
        return arbiter_out
    critique = _critique(question, domain, branches, arbiter_out, rag_context)
    refined = _refine(question, domain, branches, arbiter_out, critique, rag_context)
    return refined
