"""结果导出：写回带 response 的 JSONL 与推理报告。"""
import json
import os


def write_response(record: dict, output_path: str):
    """以追加方式写入一行（带 response 的完整题目记录）。"""
    # 简单实现：每次整体重写，保证断点续跑一致性
    raise NotImplementedError("use write_all_responses instead")


def write_all_responses(records: list, output_path: str):
    """整体写出所有记录（保留原始字段 + response）。"""
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_report(report_path: str, entry: dict):
    """将单题推理摘要追加到报告（markdown）。"""
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(format_entry_md(entry))


def format_entry_md(entry: dict) -> str:
    lines = []
    lines.append(f"## 题目 {entry['index']} — `{entry['id']}`\n")
    lines.append(f"- **子域(Sentinel)**: {entry.get('domain', 'N/A')}")
    lines.append(f"- **题型**: {entry.get('answer_type', 'N/A')}")
    lines.append(f"- **RAG 检索命中**: {entry.get('rag_hits', 'N/A')} 段")
    lines.append(f"- **耗时**: {entry.get('elapsed', 0):.1f}s")
    lines.append("")
    lines.append("### 分支结果")
    for b in entry.get("branches", []):
        lines.append(f"- **{b['name']}** (conf {b.get('confidence', '?')}%)")
        lines.append(f"  - Answer: `{b.get('answer', '')}`")
        if b.get("tool_log"):
            lines.append(f"  - 工具: {b['tool_log']}")
    lines.append("")
    lines.append("### 仲裁理由")
    lines.append(entry.get("arbiter_reason", "N/A"))
    lines.append("")
    lines.append(f"### 最终 Response\n```\n{entry.get('response', '')}\n```")
    lines.append("\n---\n")
    return "\n".join(lines)


def init_report(report_path: str):
    header = "# HLE Solver Agent — 推理报告\n\n"
    header += "本文件记录每道题的域分类、RAG 检索、分支推理与仲裁过程。\n\n---\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(header)
