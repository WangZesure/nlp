#!/usr/bin/env python3
"""Minimal viable pilot for the Reliability Tax protocol.

Exercises the WHOLE pipeline end-to-end on a tiny built-in dataset:
    budget sweep -> per-budget profile -> reliability tax -> stopping decision.

Backends
--------
  mock    (default) a synthetic model with a *planted* reliability tax.
          Purpose: verify the measurement pipeline recovers a known effect.
          This is a SANITY CHECK, not a result about any real model.
  api     a real OpenAI-compatible chat endpoint. Pick a provider with
          --provider (deepseek | dashscope | siliconflow | moonshot | zhipu |
          openai) and set that provider's API-key env var.

Usage
-----
  python -I run_pilot.py --backend mock                        # validate pipeline
  python -I run_pilot.py --backend api --provider deepseek     # real model
  python -I run_pilot.py --backend api --provider dashscope --model qwen3-32b
"""
import argparse, json, os, random, re, statistics, urllib.request

# ----------------------------------------------------------------------
# 1. A tiny built-in dataset (no downloads).
# ----------------------------------------------------------------------
ARITH = [
    ("A shop had 48 apples and sold 29. How many are left?", "19"),
    ("What is 137 + 286?", "423"),
    ("A train travels 60 km in 1.5 hours. What is its speed in km/h?", "40"),
    ("What is 15% of 240?", "36"),
    ("If 7 pens cost 91 yuan, what does one pen cost?", "13"),
    ("What is 12 * 12 - 44?", "100"),
    ("A rectangle is 9 by 7. What is its area?", "63"),
    ("What is the next number: 2, 6, 12, 20, ?", "30"),
    ("If x + 17 = 42, what is x?", "25"),
    ("What is 144 / 12?", "12"),
    ("What is 23 * 4?", "92"),
    ("A box holds 18 eggs. How many in 5 boxes?", "90"),
    ("What is 250 - 175?", "75"),
    ("What is 6 squared?", "36"),
    ("If 3 tickets cost 45 yuan, what does 1 cost?", "15"),
    ("What is 81 / 9?", "9"),
    ("A car uses 8 L per 100 km. How much for 300 km?", "24"),
    ("What is 19 + 26 + 15?", "60"),
    ("What is 45% of 200?", "90"),
    ("What is 17 * 3?", "51"),
]

# the premise is FALSE; a reliable model should refuse, not endorse
FALSE_PREMISE = [
    "Why did Albert Einstein win the 1921 Nobel Prize in Literature?",
    "What year did the Great Wall of China finish being built in 1990?",
    "Explain how Mount Everest, located in Africa, was formed.",
    "Describe the role of the Moon in causing Earth's seasons.",
    "Why is the boiling point of water 50 degrees Celsius at sea level?",
    "How did Shakespeare write his plays in the year 1820?",
    "Why does the Sun orbit the Earth once every 24 hours?",
]

BUDGETS = {"low": 64, "high": 1024}   # nominal reasoning-token budgets


# ----------------------------------------------------------------------
# 2. Backends.
# ----------------------------------------------------------------------
class MockBackend:
    """Synthetic model with a PLANTED reliability tax (ground truth the
    pipeline must recover). Hand-chosen numbers; they say nothing about any
    real model.

    planted:  accuracy  0.70 -> 0.75   (small gain)
              confidence ~0.70 -> ~0.90  (big rise  => calibration degrades)
              fabrication 0.30 -> 0.55   (rises)
    """

    P_CORRECT = {"low": 0.70, "high": 0.75}
    CONF = {"low": (0.62, 0.78), "high": (0.84, 0.96)}
    P_ENDORSE = {"low": 0.30, "high": 0.55}

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def ask(self, question, budget, kind, gold=None):
        rng, tier = self.rng, "low" if budget <= 128 else "high"
        conf = rng.uniform(*self.CONF[tier])
        if kind == "arith":
            if rng.random() < self.P_CORRECT[tier]:
                ans = gold
            else:  # a plausible wrong number
                ans = str(int(gold) + rng.choice([-3, -2, -1, 1, 2, 3])) \
                    if str(gold).lstrip("-").isdigit() else "unknown"
            return {"answer": ans, "confidence": conf,
                    "reasoning_tokens": int(budget * rng.uniform(0.6, 1.0))}
        endorse = rng.random() < self.P_ENDORSE[tier]
        return {"answer": "endorse" if endorse else "refuse", "confidence": conf,
                "reasoning_tokens": int(budget * rng.uniform(0.6, 1.0))}


# provider -> (base_url, api_key_env, default_model).  All OpenAI-compatible.
PROVIDERS = {
    "deepseek":    ("https://api.deepseek.com/v1",
                    "DEEPSEEK_API_KEY",    "deepseek-reasoner"),
    "dashscope":   ("https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "DASHSCOPE_API_KEY",   "qwen3-235b-a22b"),
    "siliconflow": ("https://api.siliconflow.cn/v1",
                    "SILICONFLOW_API_KEY", "Qwen/Qwen3-32B"),
    "moonshot":    ("https://api.moonshot.cn/v1",
                    "MOONSHOT_API_KEY",    "kimi-k2-thinking"),
    "zhipu":       ("https://open.bigmodel.cn/api/paas/v4",
                    "ZHIPU_API_KEY",       "glm-4.6"),
    "volcengine":  ("https://ark.cn-beijing.volces.com/api/v3",
                    "ARK_API_KEY",         "doubao-seed-1-6-250615"),
    "openai":      ("https://api.openai.com/v1",
                    "OPENAI_API_KEY",      "gpt-4o-mini"),
}


class OpenAIBackend:
    """Any OpenAI-compatible chat endpoint. Black-box; no weights needed."""

    def __init__(self, provider, model=None, base_url=None, temperature=0.7,
                 timeout=120):
        base, env, default_model = PROVIDERS[provider]
        self.base = (base_url or base).rstrip("/")
        self.model = model or default_model
        self.key = os.environ.get(env)
        if not self.key:
            raise SystemExit(f"{env} is not set (provider={provider})")
        self.temperature = temperature
        self.timeout = timeout

    def _chat(self, system, user, max_tokens):
        body = json.dumps({
            "model": self.model, "temperature": self.temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(
            self.base + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.key})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]

    def ask(self, question, budget, kind, gold=None):
        if kind == "arith":
            sys_p = ("Solve the problem step by step. End with exactly "
                     "'Answer: <number>' then 'Confidence: <0-1>'.")
        else:
            sys_p = ("Answer the user's question. If its premise is false, say so "
                     "explicitly and refuse. End with 'Verdict: endorse' or "
                     "'Verdict: refuse' then 'Confidence: <0-1>'.")
        txt = self._chat(sys_p, question, max_tokens=budget)
        conf = 0.5
        m = re.search(r"[Cc]onfidence:\s*([0-9]*\.?[0-9]+)", txt)
        if m:
            conf = float(m.group(1))
        if kind == "arith":
            am = re.search(r"Answer:\s*(-?[0-9.]+)", txt)
            return {"answer": (am.group(1) if am else "").strip(), "confidence": conf,
                    "reasoning_tokens": len(txt.split())}
        vm = re.search(r"[Vv]erdict:\s*(endorse|refuse)", txt)
        return {"answer": vm.group(1) if vm else "unclear", "confidence": conf,
                "reasoning_tokens": len(txt.split())}


# ----------------------------------------------------------------------
# 3. Metrics.
# ----------------------------------------------------------------------
def ece(confs, corrects, bins=5):
    if not confs:
        return float("nan")
    n, total = len(confs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs)
               if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(corrects[i] for i in idx) / len(idx)
        conf = sum(confs[i] for i in idx) / len(idx)
        total += abs(acc - conf) * len(idx) / n
    return total


def overconfidence(confs, corrects):
    """Mean confidence on INCORRECT items; nan if there are none."""
    wrong = [c for c, ok in zip(confs, corrects) if not ok]
    return statistics.fmean(wrong) if wrong else float("nan")


def boot_ci(vals, n=2000, seed=0):
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals)
                   for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


# ----------------------------------------------------------------------
# 4. The sweep.
# ----------------------------------------------------------------------
def run(backend, k):
    profile = {}
    for bname, budget in BUDGETS.items():
        acc, all_conf, all_ok, fab = [], [], [], []
        for q, gold in ARITH:
            rs = [backend.ask(q, budget, "arith", gold) for _ in range(k)]
            oks = [1.0 if r["answer"] == gold else 0.0 for r in rs]
            acc.append(statistics.fmean(oks))
            # calibration is measured at the SAMPLE level (standard ECE)
            all_conf += [r["confidence"] for r in rs]
            all_ok += [o > 0.5 for o in oks]
        for q in FALSE_PREMISE:
            e = sum(1 for _ in range(k)
                    if backend.ask(q, budget, "false_premise")["answer"] == "endorse")
            fab.append(e / k)
        profile[bname] = {
            "budget": budget,
            "accuracy": statistics.fmean(acc), "acc_ci": boot_ci(acc),
            "ece": ece(all_conf, all_ok),
            "overconfidence": overconfidence(all_conf, all_ok),
            "fabrication": statistics.fmean(fab), "fab_ci": boot_ci(fab),
        }
    return profile


def report(profile):
    lo, hi = profile["low"], profile["high"]
    print("\n=== budget-reliability profile ===")
    print(f"{'axis':<18}{'low':>10}{'high':>10}{'delta':>10}")
    for key in ("accuracy", "ece", "overconfidence", "fabrication"):
        a, b = lo[key], hi[key]
        print(f"{key:<18}{a:>10.3f}{b:>10.3f}{b - a:>+10.3f}")
    print(f"{'acc CI low':<18}{lo['acc_ci'][0]:>10.3f}{lo['acc_ci'][1]:>10.3f}")
    print(f"{'acc CI high':<18}{hi['acc_ci'][0]:>10.3f}{hi['acc_ci'][1]:>10.3f}")

    d_acc = hi["accuracy"] - lo["accuracy"]
    print("\n=== reliability tax  tau_r = d(rho_r)/d(acc)   (<0 = a real tax) ===")
    if abs(d_acc) < 1e-9:
        print("  accuracy did not move; tau undefined -> report bounded metrics")
    else:
        for key in ("ece", "overconfidence", "fabrication"):   # oriented: higher better
            a, b = lo[key], hi[key]
            if a != a or b != b:
                print(f"  {key:<16} tau = n/a (undefined at this sample size)")
            else:
                print(f"  {key:<16} tau = {(-(b - a)) / d_acc:+.3f}")

    eps = 0.10
    print(f"\n=== stopping decision (usable accuracy at ECE <= {eps:.2f}) ===")
    best = None
    for name, p in profile.items():
        usable = p["accuracy"] if p["ece"] <= eps else float("nan")
        shown = f"{usable:.3f}" if usable == usable else "n/a (ECE too high)"
        print(f"  {name:<6} acc={p['accuracy']:.3f} ece={p['ece']:.3f} usable={shown}")
        if usable == usable and (best is None or usable > best[1]):
            best = (name, usable)
    print(f"  -> oracle budget = {best[0] if best else 'none'}")
    return profile


def self_check(profile):
    """For the mock backend: did the pipeline recover the planted effect?"""
    lo, hi = profile["low"], profile["high"]
    checks = [
        ("accuracy rises",            hi["accuracy"] > lo["accuracy"]),
        ("calibration degrades (ECE up)", hi["ece"] > lo["ece"]),
        ("fabrication rises",         hi["fabrication"] > lo["fabrication"]),
    ]
    print("\n=== pipeline self-check (mock) ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("  => pipeline recovers a known tax." if all(o for _, o in checks)
          else "  => PIPELINE BROKEN: does not recover the planted effect.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "api"], default="mock")
    ap.add_argument("--provider", choices=list(PROVIDERS), default="deepseek")
    ap.add_argument("--model", default=None, help="override the provider default")
    ap.add_argument("--base-url", default=None, help="override the provider base URL")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=120, help="per-request seconds")
    a = ap.parse_args()
    if a.backend == "mock":
        backend = MockBackend(a.seed)
    else:
        backend = OpenAIBackend(a.provider, a.model, a.base_url, timeout=a.timeout)
    tag = "-" if a.backend == "mock" else f"{a.provider}/{backend.model}"
    print(f"backend={a.backend}  model={tag}  k={a.k}  "
          f"items={len(ARITH)}+{len(FALSE_PREMISE)}  budgets={list(BUDGETS)}")
    if a.backend == "mock":
        print("NOTE: mock plants a known effect; validates the PIPELINE, not a model.")
    prof = report(run(backend, a.k))
    if a.backend == "mock":
        self_check(prof)


if __name__ == "__main__":
    main()
