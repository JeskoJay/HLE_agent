"""
Monitor the HLE full run and auto-evaluate when it finishes.

- Polls until HLE_text_only_20questions_responses.jsonl has 20 valid JSON lines
  (all questions done) OR the run.py process has exited with partial progress.
- Then runs evaluate.py (Accuracy + Calibration Error) and captures the result
  into EVAL_SUMMARY.md.
- Logs progress to monitor.log.

This is a passive watcher: it does NOT launch or kill the run.py process, so it
can never cause a double-run race. If the run is already alive (expected), it
simply waits.
"""
import os
import sys
import time
import json
import subprocess

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
# Use whichever interpreter is running this monitor (portable across machines).
PY = sys.executable
OUTPUT_FILE = os.path.join(WORKSPACE, "HLE_text_only_20questions_responses.jsonl")
EVAL_SUMMARY = os.path.join(WORKSPACE, "EVAL_SUMMARY.md")
LOG = os.path.join(WORKSPACE, "monitor.log")

TOTAL = 20
POLL_SEC = 90
MAX_WAIT_SEC = 6 * 3600  # 6 hours safety cap


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def count_valid_lines(path):
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                json.loads(ln)
                n += 1
            except json.JSONDecodeError:
                continue
    return n


def run_py_alive():
    """Return True if a python process running run.py is alive."""
    try:
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "CommandLine", "/value"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        for ln in out.splitlines():
            ln = ln.strip()
            if ln.startswith("CommandLine=") and "run.py" in ln:
                return True
    except Exception as e:
        log(f"wmic check failed: {e}")
    return False


def run_evaluate():
    log("Running evaluate.py ...")
    try:
        res = subprocess.run(
            [PY, "evaluate.py"], cwd=WORKSPACE,
            capture_output=True, text=True, timeout=1800,
        )
        out = res.stdout + "\n" + res.stderr
    except Exception as e:
        out = f"evaluate.py failed to run: {e}"
    with open(EVAL_SUMMARY, "w", encoding="utf-8") as f:
        f.write("# HLE Agent — 评估结果\n\n")
        f.write("```\n")
        f.write(out)
        f.write("```\n")
    log("evaluate.py finished; summary written to EVAL_SUMMARY.md")
    log("--- evaluate output ---\n" + out)


def main():
    log("Monitor started. Watching for full-run completion.")
    start = time.time()
    while True:
        elapsed = time.time() - start
        lines = count_valid_lines(OUTPUT_FILE)
        alive = run_py_alive()
        log(f"elapsed={int(elapsed)}s valid_lines={lines}/{TOTAL} run_alive={alive}")

        if lines >= TOTAL:
            log("All 20 questions completed.")
            run_evaluate()
            break
        if (not alive) and lines > 0:
            log(f"Run process exited with partial progress ({lines} lines). Evaluating what we have.")
            run_evaluate()
            break
        if (not alive) and lines == 0:
            log("Run process not alive and no output produced. Nothing to evaluate; exiting.")
            break
        if elapsed >= MAX_WAIT_SEC:
            log("Max wait reached. Evaluating current output (may be partial).")
            run_evaluate()
            break
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
