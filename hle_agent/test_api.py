import json
import urllib.request

API_KEY = "sk-EZwCtlLpzKIPNAn79YXVgEkjzFPcKKjQSmBYeSGSjxpWRVcm"
BASE_URL = "https://api.ai-native-x.site"
MODEL = "deepseek-v4-pro"

def call(endpoint, payload, timeout=60):
    url = BASE_URL.rstrip("/") + endpoint
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {API_KEY}")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

# Test 1: basic completion (try a few endpoint shapes)
endpoints_to_try = ["/v1/chat/completions", "/chat/completions"]
result = None
last_err = None
for ep in endpoints_to_try:
    try:
        result = call(ep, {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply with the single word: PONG"}],
            "max_tokens": 16,
            "temperature": 0.0,
        })
        print(f"[OK] endpoint {ep} worked")
        break
    except Exception as e:
        last_err = e
        print(f"[FAIL] endpoint {ep}: {e}")

if result is None:
    print("ALL ENDPOINTS FAILED:", last_err)
else:
    print("RESPONSE:", json.dumps(result, ensure_ascii=False, indent=2))
    # Test 2: tools/function calling support
    try:
        tool_res = call("/v1/chat/completions", {
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is 2+3? Use the add tool."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        },
                        "required": ["a", "b"]
                    }
                }
            }],
            "max_tokens": 64,
            "temperature": 0.0,
        })
        msg = tool_res.get("choices", [{}])[0].get("message", {})
        has_tool = "tool_calls" in msg
        print("TOOL_CALLS_SUPPORTED:", has_tool)
        print("TOOL_MSG:", json.dumps(msg, ensure_ascii=False, indent=2))
    except Exception as e:
        print("TOOLS TEST ERROR:", e)
