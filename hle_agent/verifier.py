"""
高收益②：数值/特殊格式答案的代码复算交叉验证。
中收益④：分支分歧时的针对性辩论深挖（结论作为额外证据喂给 Arbiter）。

背景（v0.8 失分分析）：
- Q11 最终答案只差 1（62/68 vs 63/69），Q13/Q17 比例项与位数算错——这类
  X:Y 特殊格式数值题，草稿答案往往"接近但不精确"，用一次强制代码复算即可纠偏。
- 三分支答案互不一致的题，v0.8 中错误率显著更高，Arbiter 直接裁决容易被
  表面自信的错误分支带偏；分歧时先针对分歧点做一轮聚焦验证再裁决。
"""
import re

import config
import api_client
import tools
import stream_logger

# 数值/特殊格式答案：纯数值、比值 X:Y、分数、区间等（不含长公式）
_NUMERIC_ANS_RE = re.compile(r"^[\s\$]*-?[\d\.,eE\+\-]+(\s*[:/]\s*-?[\d\.,eE\+\-]+)*[\s\$]*$")

VERIFY_SYSTEM = """You are an independent numeric verifier for a hard exam problem.
A draft final answer has been produced. Your job: RECOMPUTE the answer INDEPENDENTLY using
python_execute (mandatory - write and run actual code; do not just reason in your head).
Pay extreme attention to: off-by-one errors, rounding/truncation rules, unit conversions,
digit/word counts, and the EXACT output format the problem demands (e.g. "X:Y").

After running code, compare your computed result with the draft answer. Output EXACTLY:

VERDICT: AGREE or DISAGREE
COMPUTED: <your independently computed answer, in the problem's required format>
CONFIDENCE: <integer 0-100 in YOUR computed answer>
REASON: <2-4 concise lines: what you computed and why any difference arises>"""


def is_numeric_answer(answer: str, answer_type: str = "") -> bool:
    a = (answer or "").strip()
    if not a or len(a) > 48:
        return False
    return bool(_NUMERIC_ANS_RE.match(a))


def verify_numeric(question, domain, rag_context, draft_answer, plan=""):
    """独立代码复算。返回 None（维持原答案）或
    {"answer": str, "confidence": int, "note": str}（建议替换）。"""
    if not getattr(config, "ENABLE_NUMERIC_VERIFY", True):
        return None
    stream_logger.section("VERIFY", f"数值复算验证（draft={draft_answer!r}）")
    user = f"Domain: {domain}\n\n"
    if plan:
        user += f"Solving plan (for reference):\n{plan}\n\n"
    if rag_context:
        user += f"Reference material:\n{rag_context}\n\n"
    user += (f"Problem:\n{question}\n\n"
             f"Draft final answer to check: {draft_answer}\n\n"
             f"Recompute independently with python_execute now.")
    messages = [{"role": "system", "content": VERIFY_SYSTEM},
                {"role": "user", "content": user}]
    try:
        r = api_client.chat_with_tools(
            messages, tools=tools.TOOL_SPECS, tool_executor=tools.exec_tool,
            tool_meta=tools.TOOL_META, temperature=0.1, max_tokens=10000,
            max_rounds=6)
    except Exception as e:
        stream_logger.block("VERIFY", "复算失败（维持原答案）", str(e))
        return None
    content = (api_client.sanitize_model_text(r.get("content") or "")
               + "\n" + (r.get("reasoning_content") or ""))
    mv = re.search(r"VERDICT\s*:\s*(AGREE|DISAGREE)", content, re.I)
    ma = re.search(r"COMPUTED\s*:\s*(.+)", content)
    mc = re.search(r"CONFIDENCE\s*:\s*(\d{1,3})", content, re.I)
    if not mv or not ma:
        return None
    computed = ma.group(1).strip().strip("`").strip()
    conf = min(100, int(mc.group(1))) if mc else 60
    mr = re.search(r"REASON\s*:\s*(.+)", content, re.S | re.I)
    note = re.sub(r"\s+", " ", mr.group(1)).strip()[:400] if mr else ""
    stream_logger.block("VERIFY", f"复算结论 {mv.group(1).upper()} conf={conf}",
                        f"computed={computed!r}\n{note}")
    if mv.group(1).upper() == "DISAGREE" and computed and conf >= 70:
        return {"answer": computed, "confidence": conf, "note": note}
    return None


DEBATE_SYSTEM = """You are a focused disagreement resolver for a hard exam problem.
Multiple solver branches produced DIFFERENT answers. Do NOT solve the whole problem again.
Instead:
1. Pinpoint exactly WHERE the branches diverge (which step / assumption / computation).
2. For each divergence point, determine which side is right, using python_execute for any
   checkable computation.
3. Conclude which candidate answer is best supported (or propose a corrected one).

Output format:
DIVERGENCE: <the key point(s) of disagreement, 1-3 lines>
FINDINGS: <what your targeted checks showed, 2-5 lines>
BEST: <the answer best supported by evidence>"""


def debate(question, domain, branches, rag_context=""):
    """分支分歧深挖：针对分歧点做聚焦验证。返回证据文本（空串=未执行/失败）。"""
    if not getattr(config, "ENABLE_DEBATE", True):
        return ""
    lines = []
    for b in branches:
        lines.append(f"- {b['name']} (conf {b.get('confidence')}%): {b.get('answer')!r}\n"
                     f"  key reasoning: {(b.get('explanation') or '')[:600]}")
    stream_logger.section("DEBATE", "分支分歧，针对分歧点深挖验证…")
    user = f"Domain: {domain}\n\n"
    if rag_context:
        user += f"Reference material:\n{rag_context}\n\n"
    user += (f"Problem:\n{question}\n\n"
             f"Conflicting candidate answers:\n" + "\n".join(lines) +
             "\n\nResolve the disagreement now (use python_execute for checkable claims).")
    messages = [{"role": "system", "content": DEBATE_SYSTEM},
                {"role": "user", "content": user}]
    try:
        r = api_client.chat_with_tools(
            messages, tools=tools.TOOL_SPECS, tool_executor=tools.exec_tool,
            tool_meta=tools.TOOL_META, temperature=0.2, max_tokens=8000,
            max_rounds=5)
        out = api_client.sanitize_model_text(r.get("content") or "")
        # 结构校验：净化后必须仍是有效结论（含 BEST 段），否则视为脏证据整体丢弃，
        # 防止半截文本/原始工具标记污染 Arbiter（v0.9 Q1/Q12 回退根因的第二道防线）。
        if not out or "BEST" not in out.upper():
            stream_logger.block("DEBATE", "结论无效（已丢弃，不注入 Arbiter）", out[:300])
            return ""
        stream_logger.block("DEBATE", "分歧深挖结论", out[:1500])
        return out[:2500]
    except Exception as e:
        stream_logger.block("DEBATE", "深挖失败（跳过）", str(e))
        return ""


def compress_confidence(conf) -> int:
    """中收益③：置信度压缩校准（只压高不抬低）。
    历史数据显示 90%+ 自报置信度实际正确率远低于 90%，用 min(conf, 40+0.5*conf)
    把 100→90、90→85、80→80，80 以下不变，降低 Calibration Error。"""
    if conf is None:
        return 70
    if not getattr(config, "ENABLE_CONF_COMPRESS", True):
        return int(conf)
    c = int(conf)
    return max(0, min(c, round(40 + 0.5 * c)))
