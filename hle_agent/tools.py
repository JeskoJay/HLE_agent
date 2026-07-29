"""
Solver 工具集。核心工具：python_execute（受限沙箱执行）。
工具以 OpenAI function-calling 格式暴露给模型。
"""
import os
import sys
import json
import tempfile
import subprocess
import textwrap

import config

PYTHON_BIN = "C:/Users/SXF-Admin/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
TOOL_TIMEOUT = 30          # 单段代码执行超时（秒）
TOOL_MEM_MB = 512

# 工具沙箱临时目录
TMP_DIR = os.path.join(config.WORKSPACE, ".workbuddy", "agent_tmp")
os.makedirs(TMP_DIR, exist_ok=True)


def python_execute(code: str) -> str:
    """
    在受限子进程中执行 Python 代码，返回 stdout/stderr。
    限制：超时 30s，捕获输出；不提供网络（按需由代码自行处理）。
    对 SageMath 题，模型可用 sympy/numpy 做等价数值/符号计算。
    """
    if not isinstance(code, str):
        return "Error: code must be a string."
    code = textwrap.dedent(code)
    # 禁用联网（设置无代理 + 卸载 socket 的简便做法：设置环境变量，运行时拦截）
    script = (
        "import os, sys\n"
        "os.environ['http_proxy']=''; os.environ['https_proxy']=''\n"
        "import socket\n"
        "def _blocked(*a, **k):\n"
        "    raise RuntimeError('network access is disabled in the sandbox')\n"
        "socket.socket = _blocked\n"
        "import runpy\n"
        "src = sys.argv[1]\n"
        "with open(src) as f:\n"
        "    code = f.read()\n"
        "ns = {}\n"
        "exec(compile(code, src, 'exec'), ns)\n"
    )
    path = os.path.join(TMP_DIR, f"exec_{abs(hash(code)) % 10**8}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    runner = os.path.join(TMP_DIR, "_runner.py")
    with open(runner, "w", encoding="utf-8") as f:
        f.write(script)
    try:
        proc = subprocess.run(
            [PYTHON_BIN, runner, path],
            capture_output=True, text=True, timeout=TOOL_TIMEOUT,
            cwd=TMP_DIR,
        )
        out = proc.stdout or ""
        err = proc.stderr or ""
        combined = out
        if err:
            combined += "\n[stderr]\n" + err
        if proc.returncode != 0 and not err:
            combined += f"\n[process exited with code {proc.returncode}]"
        return combined.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[timeout] execution exceeded {TOOL_TIMEOUT}s and was killed."
    except Exception as e:
        return f"[error] {e}"
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def calculate(expression: str) -> str:
    """安全计算简单数学表达式（允许 math 与基础运算）。"""
    try:
        import math
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "min": min, "max": max})
        # 禁止危险名称
        for bad in ("__", "import", "open", "eval", "exec", "os", "sys"):
            if bad in expression:
                return "Error: disallowed token in expression."
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def code_analyze(code: str, language: str = "python") -> str:
    """
    轻量静态分析：指出潜在语义陷阱（运算符、作用域、版本差异等）。
    复杂语义题仍以模型推理为主，此工具提供提示性检查。
    """
    tips = []
    if language == "python":
        if "/ " in code and "//" not in code:
            tips.append("注意 Python 2/3 的 '/' 除法差异：3.x 中 '/' 为真除，'//' 为整除。")
        if "and" in code or "or" in code:
            tips.append("Python 中 'and'/'or' 返回操作数之一而非 bool，注意真值语义。")
        if "set(" in code or "{}" in code:
            tips.append("空集合用 set()，{} 是空字典；集合与字典的真值、运算需留意。")
    if language == "c":
        if "^" in code:
            tips.append("C 中 '^' 是按位异或，不是幂运算；幂需用 pow 或乘法。")
        if "scanf" in code:
            tips.append("检查 scanf 格式串与参数类型是否匹配，以及取地址符使用。")
    if not tips:
        tips.append("未发现明显静态陷阱。")
    return "\n".join(tips)


# ---- function-calling 规格 ----
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "python_execute",
            "description": "Execute Python code in a sandboxed subprocess (timeout 30s). Use for computation, "
                           "cryptanalysis, simulation, or verifying program behavior. For SageMath problems, "
                           "use sympy/numpy equivalents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Safely evaluate a mathematical expression (math module available).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '2**10 * log(8, 2)'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_analyze",
            "description": "Static analysis hints for a code snippet (python/c): flags common semantic pitfalls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "language": {"type": "string", "enum": ["python", "c"]}
                },
                "required": ["code"]
            }
        }
    },
]

TOOL_EXECUTORS = {
    "python_execute": python_execute,
    "calculate": calculate,
    "code_analyze": code_analyze,
}

TOOL_META = {"blocklist": []}   # 可禁用某些工具


def exec_tool(name: str, args: dict) -> str:
    fn = TOOL_EXECUTORS.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    return fn(**args)
