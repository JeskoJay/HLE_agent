"""
主流水线：Sentinel -> RAG -> Solver Pool -> Arbiter -> Exporter。
支持断点续跑。
"""
import os
import time
import json

import config
import loader
import exporter
import sentinel
import solver
import arbiter
import rag_retriever
import stream_logger


def _format_rag(chunks) -> str:
    if not chunks:
        return ""
    parts = []
    for i, (text, score, source) in enumerate(chunks, 1):
        parts.append(f"[Ref {i}] (score={score}, source={source})\n{text}")
    return "\n\n".join(parts)


class Pipeline:
    def __init__(self):
        self.retriever = rag_retriever.get_retriever()

    def process_one(self, q: dict) -> dict:
        t0 = time.time()
        qid = q["id"]
        question = q["question"]
        answer_type = q.get("answer_type", "")

        # 1. Sentinel
        stream_logger.section("QUESTION", f"{qid}  (type={answer_type})")
        stream_logger.block("QUESTION", "题目", question)
        domain = sentinel.classify(question) if config.ENABLE_SENTINEL else "other"
        stream_logger.block("SENTINEL", "域分类结果", domain)

        # 2. RAG
        rag_chunks = []
        if self.retriever:
            rag_chunks = self.retriever.retrieve(question, k=4)
        rag_context = _format_rag(rag_chunks)
        if rag_chunks:
            hits = "\n".join(f"  [{i+1}] score={s:.3f} src={src}\n  {t[:200]}"
                             for i, (t, s, src) in enumerate(rag_chunks))
            stream_logger.block("RAG", f"检索到 {len(rag_chunks)} 条参考", hits)

        # 3. Solver Pool（并行执行各分支以加速）
        branches = self._run_branches(question, domain, rag_context)

        # 4. Arbiter
        arb = arbiter.arbitrate(question, domain, branches, rag_context)

        # 5. 组装记录
        rec = dict(q)
        rec["response"] = arb["response"]
        stream_logger.block("FINAL", f"{qid} 最终答案 (conf={arb.get('final_confidence')}%)",
                            arb["response"])
        rec["_meta"] = {
            "domain": domain,
            "rag_hits": len(rag_chunks),
            "branches": [
                {"name": b["name"], "answer": b.get("answer"),
                 "confidence": b.get("confidence"), "tool_log": b.get("tool_log", "")}
                for b in branches
            ],
            "arbiter_reason": arb.get("reason", ""),
            "final_confidence": arb.get("final_confidence"),
            "elapsed": round(time.time() - t0, 1),
        }
        return rec

    def _run_branches(self, question, domain, rag_context):
        """并行执行所有 Solver 分支（线程池），单分支失败不影响整体。"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        branches = []
        with ThreadPoolExecutor(max_workers=len(config.SOLVER_BRANCHES)) as ex:
            futs = {ex.submit(solver.solve_branch, b, question, domain, rag_context): b
                    for b in config.SOLVER_BRANCHES}
            for fut in as_completed(futs):
                try:
                    branches.append(fut.result())
                except Exception as e:
                    name = futs[fut]
                    branches.append({"name": name, "answer": "", "explanation": f"error:{e}",
                                      "confidence": None, "reasoning": "", "tool_log": ""})
        # 按配置顺序重排，便于报告阅读
        order = {b: i for i, b in enumerate(config.SOLVER_BRANCHES)}
        branches.sort(key=lambda x: order.get(x["name"], 99))
        return branches

    def run(self, limit=None):
        questions = loader.load_questions()
        if limit:
            questions = questions[:limit]
        done = loader.load_done_ids()
        # 载入已完成记录（断点续跑）
        completed_map = {}
        if os.path.exists(config.OUTPUT_FILE):
            with open(config.OUTPUT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        completed_map[obj["id"]] = obj
                    except json.JSONDecodeError:
                        continue

        exporter.init_report(config.REPORT_FILE)
        stream_logger.init()
        records = []
        total = len(questions)
        for idx, q in enumerate(questions, 1):
            if q["id"] in done and q["id"] in completed_map:
                records.append(completed_map[q["id"]])
                print(f"[{idx}/{total}] skip (done) {q['id']}")
                continue
            print(f"[{idx}/{total}] solving {q['id']} ...", flush=True)
            try:
                rec = self.process_one(q)
            except Exception as e:
                rec = dict(q)
                rec["response"] = f"Explanation: pipeline error\nAnswer: \nConfidence: 0"
                rec["_meta"] = {"error": str(e)}
                print(f"  ERROR: {e}")
            records.append(rec)
            # 报告条目
            meta = rec.get("_meta", {})
            exporter.append_report(config.REPORT_FILE, {
                "index": idx, "id": q["id"], "domain": meta.get("domain", "other"),
                "answer_type": q.get("answer_type", ""), "rag_hits": meta.get("rag_hits", 0),
                "elapsed": meta.get("elapsed", 0),
                "branches": meta.get("branches", []),
                "arbiter_reason": meta.get("arbiter_reason", ""),
                "response": rec.get("response", ""),
            })
            print(f"  done in {meta.get('elapsed', 0)}s, conf={meta.get('final_confidence')}%")
            # 实时写回，防止中断丢失
            exporter.write_all_responses(records, config.OUTPUT_FILE)

        exporter.write_all_responses(records, config.OUTPUT_FILE)
        return records
