"""
HLE 解题 Agent 入口。
用法:
    python run.py                          # 全部 20 题同时跑，输出到 outputs/
    python run.py --limit 3                # 仅前 3 题（调试）
    python run.py --concurrency 5          # 5 题并行
    python run.py --outdir my_run         # 指定输出文件夹
    python run.py --concurrency 8 --outdir run_a --limit 10
"""
import sys
import os
import argparse

# 确保能找到同目录下的 hle_agent 包
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "hle_agent"))

import config
import pipeline


def main():
    parser = argparse.ArgumentParser(description="HLE solving Agent (deepseek-v4-pro)")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 题（调试用）")
    parser.add_argument("--only", type=int, default=None, help="只跑第 N 题（保留原题号，调试用）")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="题间并行数（默认=全部题数，即 20 题同时跑）")
    parser.add_argument("--outdir", type=str, default="outputs",
                        help="输出文件夹：每题一个子文件夹，外加汇总的 responses.jsonl / report.md")
    args = parser.parse_args()

    print("=== HLE Solver Agent ===")
    print("模型: deepseek-v4-pro (api.ai-native-x.site)")
    print("分支:", ", ".join(config.SOLVER_BRANCHES))
    print("RAG:", "on" if config.ENABLE_RAG else "off",
          "| Tools:", "on" if config.ENABLE_TOOLS else "off",
          "| Arbiter:", "on" if config.ENABLE_ARBITER else "off")
    print(f"并发题数: {args.concurrency or '全部(20)'}"
          f" | 输出文件夹: {args.outdir}")
    print("=======================")

    pipe = pipeline.Pipeline()
    records = pipe.run(limit=args.limit, concurrency=args.concurrency,
                       outdir=args.outdir, only=args.only)
    print(f"\n完成。共处理 {len(records)} 题。")
    out_file = os.path.join(args.outdir, "responses.jsonl") if args.outdir else config.OUTPUT_FILE
    rep_file = os.path.join(args.outdir, "report.md") if args.outdir else config.REPORT_FILE
    print("结果:", out_file)
    print("报告:", rep_file)


if __name__ == "__main__":
    main()
