#!/usr/bin/env python3
"""One DIRECT HTTP call to a provider -- a connectivity / key smoke test.

This script talks to the provider's REST endpoint with urllib. It does NOT go
through any agent, conversation, or model-in-the-loop: it prints the raw HTTP
response so you can see the call is real.

Usage:
    DEEPSEEK_API_KEY=sk-... python -I smoke_test.py
    python -I smoke_test.py --provider deepseek --models deepseek-chat,deepseek-reasoner
    python -I smoke_test.py --prompt "What is 2+2?" --max-tokens 256
"""
import argparse, json, os, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from providers import PROVIDERS  # noqa: E402

PROMPT = ("Solve step by step, then end with exactly 'Answer: <number>' and "
          "'Confidence: <0-1>'. Question: A shop had 48 apples and sold 29. "
          "How many are left?")


def call(provider, model, prompt, max_tokens, timeout):
    base, env = PROVIDERS[provider]
    key = os.environ.get(env)
    if not key:
        raise SystemExit(f"{env} is not set")
    url = base.rstrip("/") + "/chat/completions"
    body = {"model": model, "temperature": 0.0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "url": url, "model": model,
                "error": e.read().decode("utf-8", "replace")[:400],
                "secs": time.time() - t0}
    obj = json.loads(raw)
    msg = obj["choices"][0]["message"]
    return {"ok": True, "status": status, "url": url, "model": model,
            "content": msg.get("content"),
            "reasoning": msg.get("reasoning_content"),
            "usage": obj.get("usage", {}),
            "secs": time.time() - t0, "raw_keys": sorted(msg.keys())}


def main():
    try:                                  # Windows consoles default to GBK
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--models", default="deepseek-chat,deepseek-reasoner")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args()

    print(">>> DIRECT HTTP CALL TEST (no agent, no conversation) <<<\n")
    for model in a.models.split(","):
        r = call(a.provider, model, a.prompt, a.max_tokens, a.timeout)
        print(f"--- {a.provider} / {model} ---")
        print(f"POST {r['url']}")
        print(f"HTTP {r['status']}   ({r['secs']:.1f}s)")
        if not r["ok"]:
            print("ERROR:", r["error"])
            print()
            continue
        print(f"message keys : {r['raw_keys']}")
        print(f"usage        : {json.dumps(r['usage'])}")
        if r.get("reasoning"):
            print(f"reasoning    : {len(r['reasoning'])} chars  "
                  f"| first 160: {r['reasoning'][:160]!r}")
        else:
            print("reasoning    : (none returned for this model)")
        print(f"content      : {r['content']!r}")
        print()


if __name__ == "__main__":
    main()
