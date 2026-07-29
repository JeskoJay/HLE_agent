"""
主流水线：Sentinel -> RAG -> Solver Pool -> Arbiter -> Exporter。
支持断点续跑、题间并行（--concurrency）、每题独立输出文件夹（--outdir）。
"""
import os
import time
import json
import threading

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

        # 2. RAG
        rag_chunks = []
        if self.retriever:
            rag_chunks = self.retriever.retrieve(question, k=config.RAG_TOP_K)
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
        # 每题独立产出文件
        if outdir:
            folder = f"{qindex:02d}_{qid}" if qindex else qid
            qdir = os.path.join(outdir, folder)
            with open(os.path.join(qdir, "response.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
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
        order = {b: i for i, b in enumerate(config.SOLVER_BRANCHES)}
        branches.sort(key=lambda x: order.get(x["name"], 99))
        return branches

    def _safe_one(self, q, idx, total, out_file, rep_file, outdir, rep_lock):
        """单题执行 + 容错，写入报告条目。返回记录。"""
        try:
            rec = self.process_one(q, outdir=outdir, qindex=idx)
        except Exception as e:
            rec = dict(q)
            rec["response"] = "Explanation: pipeline error\nAnswer: \nConfidence: 0"
            rec["_meta"] = {"error": str(e)}
            print(f"  ERROR [{q['id']}]: {e}")
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
              f"conf={meta.get('final_confidence')}%")
        return rec

    def run(self, limit=None, concurrency=1, outdir=None):
        questions = loader.load_questions()
        if limit:
            questions = questions[:limit]
        total = len(questions)
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

        def handle(idx, q):
            if q["id"] in done_ids and q["id"] in completed_map:
                records[idx - 1] = completed_map[q["id"]]
                print(f"[{idx}/{total}] skip (done) {q['id']}", flush=True)
                return
            print(f"[{idx}/{total}] solving {q['id']} ...", flush=True)
            rec = self._safe_one(q, idx, total, out_file, rep_file, outdir, rep_lock)
            records[idx - 1] = rec
            with f_lock:
                exporter.write_all_responses([r for r in records if r is not None], out_file)

        if concurrency and concurrency > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = {ex.submit(handle, idx, q): idx for idx, q in enumerate(questions, 1)}
                for fut in as_completed(futs):
                    fut.result()  # 异常已在 handle 内吞掉
        else:
            for idx, q in enumerate(questions, 1):
                handle(idx, q)

        final = [r for r in records if r is not None]
        exporter.write_all_responses(final, out_file)
        return final
