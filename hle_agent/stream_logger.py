"""
流式推理日志：把每个阶段（Sentinel / RAG / Solver 各分支 / Arbiter）的模型思考链
逐 token 实时写入 reasoning_stream.log，可用 `tail -f reasoning_stream.log` 实时查看。

设计：
- init()       : 开新会话，写分隔头
- section()    : 阶段大标题（如“[SOLVER] systematic 分支开始推理”）
- token()      : 写入流式到达的单个 token（实时追加，立即 flush）
- block()      : 写入一段完整文本（如最终 Answer）
- sep()        : 分隔线
文件以行缓冲 + 每次 flush 打开，确保 tail -f 能看到实时更新。
"""
import os
import threading
import datetime

_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(_WORKSPACE, "reasoning_stream.log")
_lock = threading.Lock()
_fh = None


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def init():
    global _fh
    with _lock:
        if _fh is None:
            # 行缓冲 + 每次写入后 flush，保证 tail -f 实时可见
            _fh = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _fh.write(f"\n{'=' * 72}\n 推理会话开始  {ts}\n{'=' * 72}\n")
        _fh.flush()


def _write(text):
    with _lock:
        if _fh is None:
            init()
        _fh.write(text)
        _fh.flush()


def sep():
    _write("\n" + "-" * 64 + "\n")


def section(phase, title):
    _write(f"\n[{_ts()}] ▶ {phase} — {title}\n")


def token(text):
    if text:
        _write(text)


def block(phase, title, text):
    sep()
    _write(f"[{_ts()}] ■ {phase} — {title}\n{text}\n")
