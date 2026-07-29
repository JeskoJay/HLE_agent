"""
流式推理日志：把每个阶段（Sentinel / RAG / Solver 各分支 / Arbiter）的模型思考链
逐 token 实时写入日志文件，可用 `tail -f` 实时查看。

支持两种用法：
1. 串行（默认）：直接调用模块级 section/block/token/sep，写入全局 reasoning_stream.log。
2. 并行（每题一个文件）：用 StreamLogger(path) 建一个独立记录器，并通过
   set_thread_logger(logger) 把它绑定到当前线程。sentinel/solver/arbiter 内部统一调用
   模块级函数，会自动路由到“当前线程绑定的 logger”（没有绑定则回退到全局默认）。
   这样 20 题并行时，每题的思考链各写各的文件夹，互不串台。
"""
import os
import threading
import datetime

_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(_WORKSPACE, "reasoning_stream.log")

# 全局默认 logger 的锁（模块级函数使用）
_lock = threading.Lock()
_tlocal = threading.local()


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


class StreamLogger:
    """一个题目一个日志文件的流式记录器（线程安全，独立文件句柄）。"""

    def __init__(self, log_path):
        self.log_path = log_path
        self._lock = threading.Lock()
        self._fh = None
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _ensure(self):
        if self._fh is None:
            # 行缓冲 + 每次写入后 flush，保证 tail -f 实时可见
            self._fh = open(self.log_path, "a", encoding="utf-8", buffering=1)

    def init(self):
        with self._lock:
            self._ensure()
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._fh.write(f"\n{'=' * 72}\n 推理会话开始  {ts}\n{'=' * 72}\n")
            self._fh.flush()

    def _write(self, text):
        with self._lock:
            self._ensure()
            self._fh.write(text)
            self._fh.flush()

    def sep(self):
        self._write("\n" + "-" * 64 + "\n")

    def section(self, phase, title):
        self._write(f"\n[{_ts()}] ▶ {phase} — {title}\n")

    def token(self, text):
        if text:
            self._write(text)

    def block(self, phase, title, text):
        self.sep()
        self._write(f"[{_ts()}] ■ {phase} — {title}\n{text}\n")


# 模块级默认 logger（串行模式 / 兜底）
_default = StreamLogger(LOG_PATH)


def get_logger():
    return getattr(_tlocal, "logger", None) or _default


def set_thread_logger(logger):
    """把 logger 绑定到当前线程；并行模式下每个题目线程各自绑定自己的 logger。"""
    _tlocal.logger = logger


def init():
    _default.init()


def section(phase, title):
    get_logger().section(phase, title)


def token(text):
    get_logger().token(text)


def block(phase, title, text):
    get_logger().block(phase, title, text)


def sep():
    get_logger().sep()
