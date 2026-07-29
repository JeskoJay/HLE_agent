"""验证并行 Solver 分支：对第18题（较快）跑完整 pipeline，计时并查限流。"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hle_agent"))
import pipeline, loader

qs = loader.load_questions()
q = qs[17]  # 第18题
print("Running process_one on Q18 (parallel branches)...")
t0 = time.time()
rec = pipeline.Pipeline().process_one(q)
dt = time.time() - t0
print(f"elapsed: {dt:.1f}s")
print("branches:", [(b["name"], b.get("answer", "")[:30], b.get("confidence")) for b in rec["_meta"]["branches"]])
print("final conf:", rec["_meta"]["final_confidence"])
print("RESPONSE:", rec["response"][:400])
