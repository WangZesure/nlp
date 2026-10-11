#!/usr/bin/env python3
"""Unified API client for the reliability-tax experiments.

All supported providers expose an OpenAI-compatible /chat/completions
endpoint, so one client covers them. Domestic providers only (cost).

Env vars (set the ones you use):
    DEEPSEEK_API_KEY, ARK_API_KEY, DASHSCOPE_API_KEY,
    SILICONFLOW_API_KEY, MOONSHOT_API_KEY, ZHIPU_API_KEY
"""
import json, os, time, urllib.error, urllib.request

# provider -> (base_url, api_key_env, note)
PROVIDERS = {
    "deepseek":    ("https://api.deepseek.com/v1",                       "DEEPSEEK_API_KEY"),
    "volcengine":  ("https://ark.cn-beijing.volces.com/api/v3",          "ARK_API_KEY"),
    "dashscope":   ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "siliconflow": ("https://api.siliconflow.cn/v1",                     "SILICONFLOW_API_KEY"),
    "moonshot":    ("https://api.moonshot.cn/v1",                        "MOONSHOT_API_KEY"),
    "zhipu":       ("https://open.bigmodel.cn/api/paas/v4",              "ZHIPU_API_KEY"),
}


class ApiError(RuntimeError):
    pass


class Client:
    """One (provider, model) endpoint with retry + timeout."""

    def __init__(self, provider, model, base_url=None, temperature=0.7,
                 timeout=180, max_retries=3):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}")
        base, env = PROVIDERS[provider]
        self.provider, self.model = provider, model
        self.base = (base_url or base).rstrip("/")
        self.key = os.environ.get(env)
        if not self.key:
            raise ApiError(f"{env} is not set (provider={provider})")
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, system, user, max_tokens, extra_body=None):
        """Return {"text", "usage", "finish_reason"}. Retries on transient errors."""
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if extra_body:
            body.update(extra_body)          # e.g. {"enable_thinking": False}
        data = json.dumps(body).encode()
        last = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    self.base + "/chat/completions", data=data,
                    headers={"Content-Type": "application/json",
                             "Authorization": "Bearer " + self.key})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    obj = json.loads(r.read().decode())
                ch = obj["choices"][0]
                msg = ch["message"]
                return {"text": msg.get("content") or "",
                        "reasoning": msg.get("reasoning_content") or "",
                        "usage": obj.get("usage", {}),
                        "finish_reason": ch.get("finish_reason")}
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:200]
                last = f"HTTP {e.code}: {detail}"
                if e.code in (400, 401, 403, 404):     # not retryable
                    raise ApiError(last)
            except Exception as e:                     # timeout, conn reset...
                last = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (attempt + 1))
        raise ApiError(f"gave up after {self.max_retries} tries: {last}")
