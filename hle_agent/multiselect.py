"""
高收益①：多选组合题（select-all-that-apply）逐选项判定子流程。

背景：v0.8 中 Q18–Q20 三道多选组合题全错，但错法都是"漏选/多选个别项"
（Q19 只漏 I/L/N，Q20 只漏 C），说明模型单项判断能力足够，弱点在"一口气
给出整个组合"时遗漏。本模块把组合判定拆成 N 次独立的单选项 True/False
判定（可用工具计算），最后拼装为最终组合答案。

调用时机：Reflection 之后（拿到草稿答案 draft_answer 作兜底）。
"""
import re
from concurrent.futures import ThreadPoolExecutor

import config
import api_client
import tools
import stream_logger

# 选项行：如 "A. xxx" / "B) xxx"
_OPT_RE = re.compile(r"^\s*([A-Z])[\.\)]\s+(.+)$")
# 多选组合措辞。注意：HLE 中 answer_type=multipleChoice 的题是单选（Q5/Q10），
# 真正的多选组合题是 answer_type=exactMatch + 下列措辞（Q18–Q20）。
_PHRASE_RE = re.compile(
    r"select all|all that apply|choose all|list all"
    r"|which statements are true|all the true statements"
    r"|select from the options all|answer with the letter choices"
    r"|all the (in)?appropriate", re.I)

JUDGE_SYSTEM = """You are a rigorous option verifier for a hard exam problem (Humanity's Last Exam).
You will be given the FULL problem and ONE specific option. Decide INDEPENDENTLY whether this
single option belongs in the correct answer set.

Rules:
- Judge ONLY this option; ignore how many total options might be true.
- Use tools (python_execute / calculate) whenever computation, simulation or enumeration can decide it.
- Think adversarially: look for the subtle trap that would make a careless solver flip this judgment.

Output EXACTLY:
VERDICT: TRUE or FALSE
CONFIDENCE: <integer 0-100>
REASON: <2-4 concise lines>"""


def detect(question: str, answer_type: str = ""):
    """检测多选组合题。是则返回选项列表 [(letter, text), ...]，否则 None。
    只对 exactMatch + 多选措辞触发；answer_type=multipleChoice 是单选题，不触发。"""
    if answer_type == "multipleChoice":
        return None
    if not _PHRASE_RE.search(question):
        return None
    opts = []
    for line in question.splitlines():
        m = _OPT_RE.match(line)
        if m:
            opts.append((m.group(1), m.group(2).strip()))
    # 去重保序（有些题目正文里会再次引用选项字母开头的句子，取首个出现）
    seen, uniq = set(), []
    for l, t in opts:
        if l not in seen:
            seen.add(l)
            uniq.append((l, t))
    # 选项字母应大致连续（A,B,C...），过滤误匹配正文行
    if len(uniq) >= 3:
        letters = [l for l, _ in uniq]
        expected = [chr(ord("A") + i) for i in range(len(letters))]
        if letters == expected:
            return uniq
    return None


def _judge_one(question, domain, rag_context, letter, text):
    """单个选项的独立 True/False 判定（可调工具）。失败返回 include=None。"""
    user = f"Domain: {domain}\n\n"
    if rag_context:
        user += f"Reference material (use if helpful):\n{rag_context}\n\n"
    user += (f"Full problem:\n{question}\n\n"
             f"Option under review: ({letter}) {text}\n\n"
             f"Should option {letter} be included in the final answer set? "
             f"Verify rigorously, then give VERDICT/CONFIDENCE/REASON.")
    messages = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user}]
    try:
        if config.ENABLE_TOOLS:
            r = api_client.chat_with_tools(
                messages, tools=tools.TOOL_SPECS, tool_executor=tools.exec_tool,
                tool_meta=tools.TOOL_META, temperature=0.1, max_tokens=8000,
                max_rounds=4)
        else:
            r = api_client.chat(messages, temperature=0.1, max_tokens=8000)
        content = (r.get("content") or "") + "\n" + (r.get("reasoning_content") or "")
        mv = re.search(r"VERDICT\s*:\s*(TRUE|FALSE)", content, re.I)
        include = (mv.group(1).upper() == "TRUE") if mv else None
        mc = re.search(r"CONFIDENCE\s*:\s*(\d{1,3})", content, re.I)
        conf = min(100, int(mc.group(1))) if mc else 60
        mr = re.search(r"REASON\s*:\s*(.+)", content, re.S | re.I)
        reason = re.sub(r"\s+", " ", mr.group(1)).strip()[:300] if mr else ""
        return {"letter": letter, "include": include, "confidence": conf, "reason": reason}
    except Exception as e:
        return {"letter": letter, "include": None, "confidence": 0, "reason": f"error: {e}"}


def refine(question, domain, rag_context, opts, draft_answer):
    """逐选项并行判定并拼装组合答案。
    返回 {"answer": "BCD...", "confidence": int, "detail": str}。"""
    stream_logger.section("MULTISELECT", f"多选组合题：逐选项判定（共 {len(opts)} 个选项）")
    results = []
    with ThreadPoolExecutor(max_workers=min(len(opts), 8)) as ex:
        futs = [ex.submit(_judge_one, question, domain, rag_context, l, t)
                for l, t in opts]
        for f in futs:
            results.append(f.result())
    results.sort(key=lambda r: r["letter"])

    draft_set = set(re.findall(r"[A-Z]", (draft_answer or "").upper()))
    letters, confs, lines = [], [], []
    for r in results:
        inc = r["include"]
        fallback = ""
        if inc is None:  # 判定调用失败 → 沿用草稿答案的成员关系兜底
            inc = r["letter"] in draft_set
            fallback = " (fallback->draft)"
        if inc:
            letters.append(r["letter"])
        confs.append(r["confidence"] or 60)
        lines.append(f"({r['letter']}) {'IN ' if inc else 'OUT'} conf={r['confidence']}"
                     f"{fallback} | {r['reason'][:140]}")
    detail = "\n".join(lines)
    stream_logger.block("MULTISELECT", "逐选项判定结果", detail)
    # 输出格式跟随题目要求：要求逗号分隔（如 "X,Y,Z"）则用逗号，否则紧凑拼接
    if re.search(r"comma separated|comma-separated", question, re.I):
        answer = ",".join(letters)
    else:
        answer = "".join(letters)
    # 组合置信度 = 各选项判定置信度均值打 9 折（组合出错概率高于单项）
    conf = round(sum(confs) / max(len(confs), 1) * 0.9) if confs else 60
    return {"answer": answer, "confidence": min(conf, 92), "detail": detail}
