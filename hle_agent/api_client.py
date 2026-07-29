"""
deepseek-v4-pro 客户端封装。
- 兼容 OpenAI chat/completions 协议
- 推理模型：返回 reasoning_content（思考链）+ content（最终答案）
- 完整支持 function calling（tools / tool_calls）
"""
import json
import time
import urllib.request
import urllib.error

import config


class APIError(Exception):
    pass


def _open_with_retry(req, timeout=None):
    """带限流/重试地打开请求连接。

    仅对 429 / 5xx / 网络错误做指数退避重试；4xx（如 400 参数错误）直接抛出，
    避免对无效请求无意义重试。高并发（多题并行）打 API 时，代理返回 429 不会直接崩。
    """
    last_err = None
    for attempt in range(config.MAX_RETRIES):
        try:
            return urllib.request.urlopen(req, timeout=timeout or config.REQUEST_TIMEOUT)
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504) and attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_BACKOFF * (2 ** attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_BACKOFF * (2 ** attempt))
                continue
            raise APIError(f"API request failed after {config.MAX_RETRIES} attempts: {last_err}")
    raise APIError(f"API request failed after {config.MAX_RETRIES} attempts: {last_err}")


def raw_chat(payload: dict, timeout: int = None) -> dict:
    """底层请求，返回原始 JSON。带指数退避重试。"""
    url = config.BASE_URL.rstrip("/") + "/v1/chat/completions"
    payload = dict(payload)
    payload["model"] = config.MODEL
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {config.API_KEY}")
    req.add_header("Accept", "application/json")
    with _open_with_retry(req, timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat(messages, tools=None, tool_choice="auto",
         temperature=0.3, max_tokens=4000, timeout=None,
         stream=False, on_token=None) -> dict:
    """
    高层封装：发送一次对话，返回结构化结果。
    {
      "content": str,
      "reasoning_content": str,
      "tool_calls": list | None,
      "finish_reason": str,
      "raw": dict
    }
    若推理模型把窗口耗光（finish_reason=length 且 content 为空），自动翻倍
    max_tokens 重试，确保最终答案有空间输出。

    stream=True 时走 SSE 流式：token 逐块返回，并通过 on_token(kind, delta)
    回调实时传出（kind in {"reasoning","content"}），便于上层做流式日志。
    返回值结构与非流式完全一致（content/tool_calls 已聚合）。
    """
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    attempt_tokens = max_tokens
    last_result = None
    for _ in range(3):
        payload["max_tokens"] = attempt_tokens
        if stream:
            result = raw_stream_chat(payload, on_token=on_token, timeout=timeout)
        else:
            resp = raw_chat(payload, timeout=timeout)
            choice = resp["choices"][0]
            msg = choice["message"]
            result = {
                "content": msg.get("content") or "",
                "reasoning_content": msg.get("reasoning_content") or "",
                "tool_calls": msg.get("tool_calls"),
                "finish_reason": choice.get("finish_reason"),
                "raw": resp,
            }
        last_result = result
        # 推理模型把 reasoning 写满导致 content 为空 -> 翻倍重试
        if result["finish_reason"] == "length" and not (result["content"] or "").strip():
            attempt_tokens = min(attempt_tokens * 2, 32000)
            continue
        return result
    return last_result


def raw_stream_chat(payload: dict, on_token=None, timeout: int = None) -> dict:
    """
    SSE 流式请求：解析 data: 块，聚合 reasoning_content / content / tool_calls，
    并通过 on_token(kind, delta) 实时回调（kind: "reasoning"|"content"）。
    返回与非流式一致的结构化 dict。
    """
    url = config.BASE_URL.rstrip("/") + "/v1/chat/completions"
    payload = dict(payload)
    payload["model"] = config.MODEL
    payload["stream"] = True
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {config.API_KEY}")
    req.add_header("Accept", "text/event-stream")

    reasoning_acc = ""
    content_acc = ""
    tool_calls_acc = {}      # index -> {id, type, function:{name, arguments}}
    finish_reason = None
    raw = {}

    with _open_with_retry(req, timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_s = line[5:].strip()
            if data_s == "[DONE]":
                break
            try:
                o = json.loads(data_s)
            except Exception:
                continue
            raw = o
            choices = o.get("choices") or []
            if not choices:
                continue
            ch = choices[0]
            if ch.get("finish_reason"):
                finish_reason = ch["finish_reason"]
            delta = ch.get("delta", {})
            # 思考链
            rc = delta.get("reasoning_content") or ""
            if rc:
                reasoning_acc += rc
                if on_token:
                    on_token("reasoning", rc)
            # 最终答案
            ct = delta.get("content") or ""
            if ct:
                content_acc += ct
                if on_token:
                    on_token("content", ct)
            # 工具调用（流式分片，需要按 index 聚合）
            for tc in (delta.get("tool_calls") or []):
                idx = tc.get("index", 0)
                slot = tool_calls_acc.setdefault(
                    idx, {"id": "", "type": "function",
                          "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                if tc.get("type"):
                    slot["type"] = tc["type"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]

    tool_calls = [v for _, v in sorted(tool_calls_acc.items())] or None
    return {
        "content": content_acc,
        "reasoning_content": reasoning_acc,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "raw": raw,
    }


def chat_with_tools(messages, tools, tool_executor, tool_meta,
                    temperature=0.3, max_tokens=4000, max_rounds=6, timeout=None,
                    stream=False, on_token=None) -> dict:
    """
    Agent loop：当模型返回 tool_calls 时执行工具，把结果回填，继续对话，
    直到模型给出最终答案（finish_reason != 'tool_calls'）或达到轮数上限。
    tool_executor(name, args) -> str  返回工具执行结果字符串
    返回最后一次的结构化结果。
    stream/on_token 透传给 chat，实现思考链实时流出。
    """
    for _ in range(max_rounds):
        result = chat(messages, tools=tools, tool_choice="auto",
                      temperature=temperature, max_tokens=max_tokens, timeout=timeout,
                      stream=stream, on_token=on_token)
        # 将助手消息加入历史
        assistant_msg = {
            "role": "assistant",
            "content": result["content"],
        }
        if result["reasoning_content"]:
            assistant_msg["reasoning_content"] = result["reasoning_content"]
        if result["tool_calls"]:
            assistant_msg["tool_calls"] = result["tool_calls"]
        messages.append(assistant_msg)

        if result["finish_reason"] != "tool_calls" or not result["tool_calls"]:
            return result

        # 执行每个工具调用
        for tc in result["tool_calls"]:
            fn = tc["function"]
            name = fn["name"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if name in tool_meta.get("blocklist", []):
                output = f"Tool '{name}' is disabled."
            else:
                try:
                    output = tool_executor(name, args)
                except Exception as e:
                    output = f"Tool execution error: {e}"
            # 截断过长输出
            if len(output) > 12000:
                output = output[:12000] + "\n...[truncated]"
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": name,
                "content": output,
            })
    # 达到轮数上限，强制让模型总结
    result = chat(messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout,
                  stream=stream, on_token=on_token)
    return result
