"""
主流水线：Sentinel -> Planner(预分析+解题计划) -> RAG(计划驱动检索) -> Solver Pool -> Arbiter -> Reflection -> Exporter。
支持断点续跑、题间并行（--concurrency）、每题独立输出文件夹（--outdir）。
"""
import os
import re
import time
import json
import threading

import config
import loader
import exporter
import sentinel
import solver
import arbiter
import reflect
import verifier
import multiselect
import api_client
import rag_retriever
import stream_logger


def _format_rag(chunks) -> str:
    if not chunks:
        return ""
    parts = []
    for i, (text, score, source) in enumerate(chunks, 1):
        parts.append(f"[Ref {i}] (score={score}, source={source})\n{text}")
    return "\n\n".join(parts)


_KW_RE = re.compile(r"RETRIEVAL_KEYWORDS\s*:\s*(.+?)(?:\n|$)", re.I)


def _make_plan(question: str, domain: str) -> dict:
    """Planner（前置预分析）：拿到题目先分析题型/考点/陷阱，生成小的解题思路计划，
    并输出一行检索关键词（用于驱动后续 RAG 知识检索）。
    返回 {"plan": str, "keywords": str}；失败返回空值（不阻塞主流程）。"""
    if not config.ENABLE_PLANNER:
        return {"plan": "", "keywords": ""}
    system = (
        "You are a pre-analysis planner for a rigorous problem-solving agent "
        "(Humanity's Last Exam). Given a problem, do the following WITHOUT solving it:\n"
        "1. ANALYSIS: identify the problem type, the core concepts/theorems involved, "
        "and likely traps or subtle points (2-4 short lines).\n"
        "2. PLAN: a concise numbered step-by-step solving plan (3-6 steps), stating what "
        "to compute/verify at each step and whether code execution would help.\n"
        "3. RETRIEVAL_KEYWORDS: one single line of 5-12 English keywords/terms "
        "(comma-separated) that a knowledge-base search should use to find relevant "
        "reference material for this problem.\n\n"
        "Output format (EXACT):\n"
        "ANALYSIS:\n<lines>\n\nPLAN:\n<numbered steps>\n\n"
        "RETRIEVAL_KEYWORDS: <kw1, kw2, ...>"
    )
    user = f"Problem domain: {domain}\n\nProblem:\n{question}"
    try:
        r = api_client.chat([{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                            temperature=0.2, max_tokens=1000)
        text = (r.get("content") or "").strip()
        m = _KW_RE.search(text)
        keywords = m.group(1).strip() if m else ""
        # plan 部分 = 去掉关键词行的全文（ANALYSIS+PLAN 一起注入 Solver，信息更全）
        plan = _KW_RE.sub("", text).strip()
        return {"plan": plan, "keywords": keywords}
    except Exception as e:
        print(f"  [planner] failed: {e}")
        return {"plan": "", "keywords": ""}


class Pipeline:
    def __init__(self):
        self.retriever = rag_retriever.get_retriever()

    def process_one(self, q: dict, outdir=None, qindex=None) -> dict:
        t0 = time.time()
        qid = q["id"]
        question = q["question"]
        answer_type = q.get("answer_type", "")

        # 每题独立的流式日志（并行时不串台）；文件夹名带题号 NN_<qid> 便于定位
        if outdir:
            folder = f"{qindex:02d}_{qid}" if qindex else qid
            qdir = os.path.join(outdir, folder)
            os.makedirs(qdir, exist_ok=True)
            with open(os.path.join(qdir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "index": qindex,
                    "id": qid,
                    "answer_type": answer_type,
                    "question": question,
                }, f, ensure_ascii=False, indent=2)
            logger = stream_logger.StreamLogger(os.path.join(qdir, "reasoning.log"))
            logger.init()
            stream_logger.set_thread_logger(logger)

        qtag = f"Q{qindex} {qid}" if qindex else qid
        stream_logger.section("QUESTION", f"{qtag}  (type={answer_type})")
        stream_logger.block("QUESTION", "题目", question)
        domain = sentinel.classify(question) if config.ENABLE_SENTINEL else "other"
        stream_logger.block("SENTINEL", "域分类结果", domain)

        # 2. Planner 前置（预分析）：先分析题目生成解题思路计划 + 检索关键词
        plan_info = _make_plan(question, domain)
        plan = plan_info.get("plan", "")
        plan_keywords = plan_info.get("keywords", "")
        if plan:
            stream_logger.block("PLANNER", "预分析 + 解题计划", plan)
        if plan_keywords:
            stream_logger.block("PLANNER", "检索关键词", plan_keywords)

        # 3. RAG（计划驱动检索）：用「题目 + 计划关键词」构造 query，检索更贴合解题思路
        rag_chunks = []
        if self.retriever:
            rag_query = f"{question}\n{plan_keywords}" if plan_keywords else question
            rag_chunks = self.retriever.retrieve(rag_query, k=config.RAG_TOP_K)
        rag_context = _format_rag(rag_chunks)
        if rag_chunks:
            hits = "\n".join(f"  [{i+1}] score={s:.3f} src={src} len={len(t)}\n  {t}"
                             for i, (t, s, src) in enumerate(rag_chunks))
            stream_logger.block("RAG", f"检索到 {len(rag_chunks)} 条参考（计划驱动）", hits)

        # 4. Solver Pool（并行执行各分支；每分支可选自洽采样；计划注入各分支 prompt）
        branches = self._run_branches(question, domain, rag_context, plan, answer_type)

        # 4.5 分歧深挖（v0.9-④）：三分支答案不一致时，针对分歧点聚焦验证再裁决
        debate_notes = ""
        if config.ENABLE_DEBATE and len(branches) > 1 and not arbiter.is_consistent(branches):
            debate_notes = verifier.debate(question, domain, branches, rag_context)

        # 5. Arbiter（分歧深挖结论作为额外证据注入）
        arb = arbiter.arbitrate(question, domain, branches, rag_context,
                                extra_evidence=debate_notes)

        # 5.5 Reflection (P0-§2): Critic + Refine 反思闭环
        reflected = reflect.improve(question, domain, branches, arb, rag_context)

        # 6. 答案级后验证（v0.9-①②）
        final_answer = reflected.get("final_answer") or ""
        final_conf = reflected.get("final_confidence")
        override_note = ""
        opts = multiselect.detect(question, answer_type) \
            if config.ENABLE_MULTISELECT_JUDGE else None
        if opts:
            # ① 多选组合题：逐选项独立判定后拼装
            ms = multiselect.refine(question, domain, rag_context, opts, final_answer)
            if ms["answer"]:
                old_set = set(re.findall(r"[A-Z]", final_answer.upper()))
                new_set = set(re.findall(r"[A-Z]", ms["answer"]))
                if old_set != new_set:
                    override_note = (f"[multiselect] per-option judging changed answer "
                                     f"{final_answer!r} -> {ms['answer']!r}")
                final_answer, final_conf = ms["answer"], ms["confidence"]
        elif verifier.is_numeric_answer(final_answer, answer_type):
            # ② 数值/特殊格式答案：独立代码复算，DISAGREE 且高置信才替换
            v = verifier.verify_numeric(question, domain, rag_context, final_answer, plan)
            if v:
                override_note = (f"[verify] independent recompute changed answer "
                                 f"{final_answer!r} -> {v['answer']!r}: {v['note'][:200]}")
                final_answer, final_conf = v["answer"], v["confidence"]

        # ②.5 空答案兜底不变量（P0）：最终答案绝不允许为空。
        # 依次回退：Reflection -> Arbiter -> 置信度最高的分支；仍为空则置信度归 5。
        if not str(final_answer).strip():
            fallback = (arb.get("final_answer") or "").strip()
            if not fallback:
                best = max(branches, key=lambda b: b.get("confidence") or 0) \
                    if branches else None
                fallback = (best.get("answer") or "").strip() if best else ""
            if fallback:
                override_note = (override_note + " | " if override_note else "") + \
                    f"[fallback] empty final answer -> arbiter/branch answer {fallback!r}"
                final_answer = fallback
                final_conf = min(int(final_conf or 50), 60)
                stream_logger.block("FINAL", "空答案兜底触发", override_note)
            else:
                final_conf = 5
                stream_logger.block("FINAL", "空答案且无可回退来源", "confidence -> 5")

        # ③ 置信度压缩校准（只压高不抬低）
        final_conf = verifier.compress_confidence(final_conf)

        # 重组最终 response（保留 Reflection 的 Explanation，替换 Answer/Confidence）
        exp_m = re.search(r"Explanation\s*:\s*(.+?)(?=\n\s*Answer\s*:)",
                          reflected.get("response", ""), re.S | re.I)
        explanation = exp_m.group(1).strip() if exp_m else reflected.get("response", "")
        if override_note:
            explanation = f"{explanation}\n[Post-verification] {override_note}"
            stream_logger.block("VERIFY", "答案后验证覆写", override_note)
        final_response = (f"Explanation: {explanation}\n"
                          f"Answer: {final_answer}\nConfidence: {final_conf}%")

        # 7. 组装记录
        rec = dict(q)
        rec["response"] = final_response
        reflected["final_confidence"] = final_conf
        stream_logger.block("FINAL", f"{qid} 最终答案 (conf={final_conf}%)",
                            final_response)
        rec["_meta"] = {
            "domain": domain,
            "rag_hits": len(rag_chunks),
            "plan": plan,
            "plan_keywords": plan_keywords,
            "branches": [
                {"name": b["name"], "answer": b.get("answer"),
                 "confidence": b.get("confidence"), "tool_log": b.get("tool_log", "")}
                for b in branches
            ],
            "arbiter_reason": arb.get("reason", ""),
            "critic": reflected.get("critique", ""),
            "debate": debate_notes[:800],
            "post_verify": override_note,
            "final_confidence": final_conf,
            "elapsed": round(time.time() - t0, 1),
        }
        # 每题独立产出文件
        if outdir:
            folder = f"{qindex:02d}_{qid}" if qindex else qid
            qdir = os.path.join(outdir, folder)
            with open(os.path.join(qdir, "response.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
        return rec

    def _run_branches(self, question, domain, rag_context, plan, answer_type):
        """并行执行所有 Solver 分支（线程池），单分支失败不影响整体。"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        branches = []
        with ThreadPoolExecutor(max_workers=len(config.SOLVER_BRANCHES)) as ex:
            futs = {ex.submit(solver.solve_branch, b, question, domain, rag_context,
                              plan, answer_type, config.SELF_CONSISTENCY_K): b
                    for b in config.SOLVER_BRANCHES}
            for fut in as_completed(futs):
                try:
                    branches.append(fut.result())
                except Exception as e:
                    name = futs[fut]
                    branches.append({"name": name, "answer": "", "explanation": f"error:{e}",
                                      "confidence": None, "reasoning": "", "tool_log": ""})
        order = {b: i for i, b in enumerate(config.SOLVER_BRANCHES)}
        branches.sort(key=lambda x: order.get(x["name"], 99))
        return branches

    def _safe_one(self, q, idx, total, out_file, rep_file, outdir, rep_lock):
        """单题执行 + 整题级容错重试（P0-§10），写入报告条目。返回记录。"""
        last_err = None
        rec = None
        for attempt in range(config.MAX_PIPELINE_RETRIES + 1):
            try:
                rec = self.process_one(q, outdir=outdir, qindex=idx)
                break
            except Exception as e:
                last_err = e
                if attempt < config.MAX_PIPELINE_RETRIES:
                    backoff = config.PIPELINE_RETRY_BACKOFF * (2 ** attempt)
                    print(f"  RETRY [{q['id']}] attempt {attempt+1}/"
                          f"{config.MAX_PIPELINE_RETRIES}: {e} (wait {backoff}s)", flush=True)
                    time.sleep(backoff)
                else:
                    print(f"  ERROR [{q['id']}] after {config.MAX_PIPELINE_RETRIES} "
                          f"retries: {last_err}", flush=True)
        if rec is None:
            rec = dict(q)
            rec["response"] = "Explanation: pipeline error\nAnswer: \nConfidence: 0"
            rec["_meta"] = {"error": str(last_err)}
        meta = rec.get("_meta", {})
        entry = {
            "index": idx, "id": q["id"], "domain": meta.get("domain", "other"),
            "answer_type": q.get("answer_type", ""), "rag_hits": meta.get("rag_hits", 0),
            "elapsed": meta.get("elapsed", 0),
            "branches": meta.get("branches", []),
            "arbiter_reason": meta.get("arbiter_reason", ""),
            "response": rec.get("response", ""),
        }
        with rep_lock:
            exporter.append_report(rep_file, entry)
        print(f"  [{idx}/{total}] done in {meta.get('elapsed', 0)}s, "
              f"conf={meta.get('final_confidence')}%", flush=True)
        return rec

    def run(self, limit=None, concurrency=1, outdir=None, only=None):
        questions = loader.load_questions()
        n_all = len(questions)
        if only:
            if not (1 <= only <= n_all):
                raise ValueError(f"--only 超出范围: {only} (共 {n_all} 题)")
            pairs = [(only, questions[only - 1])]  # 保留原题号
        else:
            if limit:
                questions = questions[:limit]
            pairs = list(enumerate(questions, 1))
        total = len(pairs)
        if concurrency is None:
            concurrency = total  # 默认全部题并行

        # 输出路径：传入 outdir 则全部落在文件夹内
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            out_file = os.path.join(outdir, "responses.jsonl")
            rep_file = os.path.join(outdir, "report.md")
        else:
            if config.ENABLE_SENTINEL or True:
                stream_logger.init()
            out_file = config.OUTPUT_FILE
            rep_file = config.REPORT_FILE

        # 断点续跑：已在该 out_file 下完成的题跳过
        done_ids = set()
        completed_map = {}
        if os.path.exists(out_file):
            with open(out_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        done_ids.add(obj["id"])
                        completed_map[obj["id"]] = obj
                    except json.JSONDecodeError:
                        continue

        exporter.init_report(rep_file)
        rep_lock = threading.Lock()
        f_lock = threading.Lock()
        records = [None] * total
        pos_of = {idx: i for i, (idx, _) in enumerate(pairs)}  # 原题号 -> records 下标

        def handle(idx, q):
            if q["id"] in done_ids and q["id"] in completed_map:
                records[pos_of[idx]] = completed_map[q["id"]]
                print(f"[{idx}] skip (done) {q['id']}", flush=True)
                return
            print(f"[{idx}] solving {q['id']} ...", flush=True)
            rec = self._safe_one(q, idx, total, out_file, rep_file, outdir, rep_lock)
            records[pos_of[idx]] = rec
            with f_lock:
                exporter.write_all_responses([r for r in records if r is not None], out_file)

        if concurrency and concurrency > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = {ex.submit(handle, idx, q): idx for idx, q in pairs}
                for fut in as_completed(futs):
                    fut.result()  # 异常已在 handle 内吞掉
        else:
            for idx, q in pairs:
                handle(idx, q)

        final = [r for r in records if r is not None]
        exporter.write_all_responses(final, out_file)
        return final
