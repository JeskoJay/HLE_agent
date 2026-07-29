"""Sentinel：轻量域分类器（使用 deepseek-v4-pro，低成本）。"""
import json

import config
import api_client
import stream_logger

DOMAIN_TAXONOMY = [
    "cryptography", "sage_math", "programming_semantics", "machine_learning",
    "robotics", "signal_processing", "computer_architecture", "reinforcement_learning",
    "pomdp_planning", "domain_design", "information_systems", "data_science", "other",
]

SYSTEM = (
    "You are a question domain classifier for the HLE (Humanity's Last Exam) dataset. "
    "Given a question, output a JSON object {\"domain\": <one of the allowed labels>}. "
    "Allowed labels: " + ", ".join(DOMAIN_TAXONOMY) + ". "
    "Choose the single most specific label. Output ONLY the JSON, no other text."
)


def classify(question: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question[:3000]},
    ]
    try:
        stream_logger.section("SENTINEL", "域分类中…")
        r = api_client.chat(msgs, temperature=config.SENTINEL_PARAMS["temperature"],
                            max_tokens=config.SENTINEL_PARAMS["max_tokens"],
                            stream=True,
                            on_token=lambda kind, t: stream_logger.token(t))
        text = r["content"].strip() or r["reasoning_content"].strip()
        # 提取 JSON
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            obj = json.loads(text[start:end])
            dom = obj.get("domain", "other")
            if dom in DOMAIN_TAXONOMY:
                return dom
    except Exception as e:
        pass
    return "other"
