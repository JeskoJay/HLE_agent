"""
HLE 解题 Agent 入口。
用法:
    python run.py            # 跑全部 20 题
    python run.py --limit 3  # 仅跑前 3 题（调试）
"""
import sys
import os
import argparse

# 确保能找到同目录下的 hle_agent 包
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "hle_agent"))

import pipeline


def main():
    parser = argparse.ArgumentParser(description="HLE solving Agent (deepseek-v4-pro)")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 题（调试用）")
    args = parser.parse_args()

    print("=== HLE Solver Agent ===")
    print("模型: deepseek-v4-pro (api.ai-native-x.site)")
    print("分支:", ", ".join(__import__("config").SOLVER_BRANCHES))
    print("RAG:", "on" if __import__("config").ENABLE_RAG else "off",
          "| Tools:", "on" if __import__("config").ENABLE_TOOLS else "off",
          "| Arbiter:", "on" if __import__("config").ENABLE_ARBITER else "off")
    print("=======================")

    pipe = pipeline.Pipeline()
    records = pipe.run(limit=args.limit)
    print(f"\n完成。共处理 {len(records)} 题。")
    print("结果:", __import__("config").OUTPUT_FILE)
    print("报告:", __import__("config").REPORT_FILE)


if __name__ == "__main__":
    main()
