"""题目数据加载。"""
import json
import os

import config


def load_questions(path: str = None) -> list:
    """读取 JSONL，返回题目 dict 列表。"""
    path = path or config.DATA_FILE
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def load_done_ids(output_path: str = None) -> set:
    """读取已处理（含 response）的 id 集合，支持断点续跑。"""
    output_path = output_path or config.OUTPUT_FILE
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("response"):
                    done.add(obj["id"])
            except json.JSONDecodeError:
                continue
    return done
