"""
全局配置：模型接入、运行参数。
模型仅允许使用用户提供的 deepseek-v4-pro（api.ai-native-x.site）。
"""
import os


def _load_env_file():
    """从仓库根目录 .env 读取本地配置（.env 不入库，避免泄露密钥）。"""
    env = {}
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env


_ENV = _load_env_file()

# ---- 模型接入（仅此一个模型）----
# 密钥从环境变量 HLE_API_KEY 或仓库根 .env 读取，不要硬编码入库。
API_KEY = os.getenv("HLE_API_KEY", _ENV.get("HLE_API_KEY", ""))
if not API_KEY:
    raise RuntimeError(
        "未配置 HLE_API_KEY。请设置环境变量，或在仓库根目录创建 .env 文件写入 HLE_API_KEY=sk-..."
    )
BASE_URL = os.getenv("HLE_BASE_URL", _ENV.get("HLE_BASE_URL", "https://api.ai-native-x.site"))
MODEL = os.getenv("HLE_MODEL", _ENV.get("HLE_MODEL", "deepseek-v4-pro"))

# ---- 调用参数 ----
REQUEST_TIMEOUT = 180          # 单次请求超时（秒），推理模型较慢
MAX_RETRIES = 4                # 失败重试次数（含 429/5xx 限流退避）
RETRY_BACKOFF = 8              # 重试退避基数（秒），高并发限流时退避更足

# 全局并发 API 请求上限。题间(20)×分支(3)嵌套并行理论峰值 ~60+ 路同时打 API，
# 远超代理通常的 RPM/TPM 限制。用信号量把"同时在飞的真实请求"封顶为本值，
# 线程再多也只是排队重叠 IO，不会把代理打爆导致限流/重试耗尽。可按代理实际
# 额度上调（如 15~20），不确定时保持 10 最稳。
MAX_CONCURRENT_REQUESTS = 10

# 各角色默认参数
# 注意：deepseek-v4-pro 是推理模型，reasoning_content 会消耗大量 token，
# 必须给足 max_tokens 才能让最终 answer（content）有空间输出。
SENTINEL_PARAMS = {"temperature": 0.0, "max_tokens": 1024}
SOLVER_PARAMS = {"temperature": 0.3, "max_tokens": 12000}
ARBITER_PARAMS = {"temperature": 0.2, "max_tokens": 8000}

# ---- 流水线开关 ----
ENABLE_SENTINEL = True         # 域分类
ENABLE_RAG = True              # RAG 知识检索（知识库待用户提供资料后接入）
ENABLE_TOOLS = True            # 工具调用（Python 沙箱等）
ENABLE_ARBITER = True          # 多分支仲裁
SOLVER_BRANCHES = ["systematic", "intuitive", "code_first"]  # 启用的 Solver 分支

# ---- 目录 ----
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(WORKSPACE, "HLE_text_only_20questions_student.jsonl")
OUTPUT_FILE = os.path.join(WORKSPACE, "HLE_text_only_20questions_responses.jsonl")
REPORT_FILE = os.path.join(WORKSPACE, "HLE_solver_report.md")
KB_DIR = os.path.join(WORKSPACE, "knowledge_base")
