import json
import urllib.request

API_KEY = "sk-EZwCtlLpzKIPNAn79YXVgEkjzFPcKKjQSmBYeSGSjxpWRVcm"
BASE_URL = "https://api.ai-native-x.site"
MODEL = "deepseek-v4-pro"

def call(payload, timeout=120):
    url = BASE_URL.rstrip("/") + "/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {API_KEY}")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

# Full conversation: ask a math question, expect reasoning + answer in content
print("=== Test: full content flow ===")
r = call({
    "model": MODEL,
    "messages": [{"role": "user", "content": "Solve: what is 2+3? Answer with just the number."}],
    "max_tokens": 2000,
    "temperature": 0.3,
})
msg = r["choices"][0]["message"]
print("content:", repr(msg.get("content")))
print("reasoning_content:", repr(msg.get("reasoning_content", "")[:200]))
print("finish_reason:", r["choices"][0].get("finish_reason"))

# Tool call: ask to add with explicit tool, give enough tokens
print("\n=== Test: complete tool call ===")
r2 = call({
    "model": MODEL,
    "messages": [{"role": "user", "content": "Use the add tool to compute 2 + 3. Then tell me the result."}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"]
            }
        }
    }],
    "max_tokens": 2000,
    "temperature": 0.3,
})
msg2 = r2["choices"][0]["message"]
print("content:", repr(msg2.get("content")))
print("tool_calls:", json.dumps(msg2.get("tool_calls"), ensure_ascii=False))
print("finish_reason:", r2["choices"][0].get("finish_reason"))
