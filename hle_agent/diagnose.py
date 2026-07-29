"""诊断：观察 deepseek-v4-pro 在解题时的完整交互（工具调用格式、DSML标记、content）。"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hle_agent"))
import config
import api_client
import tools
import loader

# 取第 18 题（Python 语义，较短）
qs = loader.load_questions()
q = qs[17]
print("Q:", q["question"][:300])

SYSTEM = (
    "You are a world-class expert solving HLE problems. "
    "Respond EXACTLY in format:\nExplanation: ...\nAnswer: ...\nConfidence: <0-100>\n"
    "Use python_execute to verify program behavior when helpful."
)
messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "Problem:\n" + q["question"]},
]

for rnd in range(4):
    res = api_client.chat(messages, tools=tools.TOOL_SPECS, tool_choice="auto",
                          temperature=0.3, max_tokens=12000)
    print(f"\n--- round {rnd} ---")
    print("finish_reason:", res["finish_reason"])
    print("content[:400]:", repr(res["content"][:400]))
    print("reasoning[:200]:", repr(res["reasoning_content"][:200]))
    print("tool_calls:", json.dumps(res["tool_calls"], ensure_ascii=False)[:400] if res["tool_calls"] else None)

    assistant_msg = {"role": "assistant", "content": res["content"]}
    if res["reasoning_content"]:
        assistant_msg["reasoning_content"] = res["reasoning_content"]
    if res["tool_calls"]:
        assistant_msg["tool_calls"] = res["tool_calls"]
    messages.append(assistant_msg)

    if res["finish_reason"] != "tool_calls" or not res["tool_calls"]:
        print("DONE (no tool call)")
        break
    for tc in res["tool_calls"]:
        fn = tc["function"]
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {}
        out = tools.exec_tool(fn["name"], args)
        print(f"  TOOL {fn['name']}({args}) -> {out[:300]}")
        messages.append({"role": "tool", "tool_call_id": tc["id"], "name": fn["name"], "content": out[:300]})
