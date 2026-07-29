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


def raw_chat(payload: dict, timeout: int = None) -> dict:
    """底层请求，返回原始 JSON。带指数退避重试。"""
    url = config.BASE_URL.rstrip("/") + "/v1/chat/completions"
    payload = dict(payload)
    payload["model"] = config.MODEL
    data = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(config.MAX_RETRIES):
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {config.API_KEY}")
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=timeout or config.REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            if attempt < config.MAX_RETRIES - 1:
                wait = config.RETRY_BACKOFF * (2 ** attempt)
                time.sleep(wait)
    raise APIError(f"API request failed after {config.MAX_RETRIES} attempts: {last_err}")


def chat(messages, tools=None, tool_choice="auto",
         temperature=0.3, max_tokens=4000, timeout=None) -> dict:
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
        if choice.get("finish_reason") == "length" and not (result["content"] or "").strip():
            attempt_tokens = min(attempt_tokens * 2, 32000)
            continue
        return result
    return last_result


def chat_with_tools(messages, tools, tool_executor, tool_meta,
                    temperature=0.3, max_tokens=4000, max_rounds=6, timeout=None) -> dict:
    """
    Agent loop：当模型返回 tool_calls 时执行工具，把结果回填，继续对话，
    直到模型给出最终答案（finish_reason != 'tool_calls'）或达到轮数上限。
    tool_executor(name, args) -> str  返回工具执行结果字符串
    返回最后一次的结构化结果。
    """
    for _ in range(max_rounds):
        result = chat(messages, tools=tools, tool_choice="auto",
                      temperature=temperature, max_tokens=max_tokens, timeout=timeout)
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
    result = chat(messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    return result
