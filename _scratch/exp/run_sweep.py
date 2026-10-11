#!/usr/bin/env python3
"""Orchestrate the budget sweep: models x tasks x budgets x templates x k.

Writes one JSONL per (model, task, budget, template) into runs/.
Resumable: re-running skips (item, sample) pairs already recorded.
Cost is logged from the API `usage` field.

Usage:
    python -I run_sweep.py --mock                 # validate offline, no API
    python -I run_sweep.py                        # real run (needs API keys)
    python -I run_sweep.py --models deepseek-chat # subset
    python -I run_sweep.py --tasks math --budgets b1,b2
"""
import argparse, concurrent.futures as cf, json, os, random, re, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PROMPTS = {
    "math": ("Solve the problem step by step. End your reply with exactly "
             "'Answer: <number>' and then 'Confidence: <0-1>'."),
    "factqa": ("Answer the question. End your reply with exactly "
               "'Answer: <short answer>' and then 'Confidence: <0-1>'."),
    "multihop": ("Answer the question. End your reply with exactly "
                 "'Answer: <short answer>' and then 'Confidence: <0-1>'."),
    "falsepremise": ("Answer the user's question. If its premise is false, say so "
                     "explicitly and refuse to answer as posed. End your reply with "
                     "exactly 'Verdict: endorse' or 'Verdict: refuse' and then "
                     "'Confidence: <0-1>'."),
}
TEMPLATE_SUFFIX = {"plain": "", "strict": " Be precise and follow the format exactly."}

# providers that accept an explicit thinking toggle; others select thinking by
# choosing a different model.
THINKING_TOGGLE_PROVIDERS = {"dashscope", "zhipu", "volcengine", "siliconflow"}


def thinking_body(model_cfg, budget):
    """Provider-specific thinking control.

    DeepSeek: {"thinking": {"type": enabled|disabled}} plus optional
              {"reasoning_effort": low|high|max}  (its official budget knob).
    Others : {"enable_thinking": bool}.
    """
    p = model_cfg["provider"]
    if p == "deepseek":
        body = {"thinking": {"type": budget.get("thinking", "enabled")}}
        if budget.get("reasoning_effort"):
            body["reasoning_effort"] = budget["reasoning_effort"]
        return body
    if p in THINKING_TOGGLE_PROVIDERS:
        return {"enable_thinking": budget.get("thinking", "enabled") == "enabled"}
    return None


# ---------------------------------------------------------------- mock backend
class MockClient:
    """Planted reliability tax, for offline pipeline validation only.

    Gold-aware, so that the metrics can be checked end-to-end: accuracy rises
    modestly with budget while confidence (and thus ECE) and false-premise
    endorsement rise faster.
    """

    P_OK = {"low": 0.70, "high": 0.75}

    def __init__(self, model):
        self.model = model
        self.rng = random.Random(abs(hash(model)) % 100000)

    def respond(self, item, budget_name):
        rng = self.rng
        tier = "low" if budget_name in ("off", "low") else "high"
        conf = rng.uniform(*(0.62, 0.78) if tier == "low" else (0.84, 0.96))
        task = item["task"]
        if task == "falsepremise":
            endorse = rng.random() < (0.30 if tier == "low" else 0.55)
            body = f"Verdict: {'endorse' if endorse else 'refuse'}"
        else:
            ok = rng.random() < self.P_OK[tier]
            gold = str(item["gold"])
            if ok:
                ans = gold
            elif task == "math":
                m = re.search(r"-?\d[\d,]*", gold.replace(",", ""))
                ans = str(int(m.group(0)) + 1) if m else "0"
            else:
                ans = "unknown"
            body = f"Answer: {ans}"
        return {"text": f"...\n{body}\nConfidence: {conf:.2f}",
                "reasoning": "",
                "usage": {"completion_tokens": rng.randint(20, 2048)},
                "finish_reason": "stop"}


# ---------------------------------------------------------------- orchestration
def load_items(data_dir, task_file, n):
    path = os.path.join(data_dir, task_file)
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    return items[:n]


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
                keys.add((r["item_id"], r["sample_idx"]))
            except Exception:
                pass
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--models", default=None, help="comma-separated model names")
    ap.add_argument("--tasks", default=None, help="comma-separated task names")
    ap.add_argument("--budgets", default=None, help="comma-separated budget names")
    ap.add_argument("--n", type=int, default=None, help="override items per task")
    ap.add_argument("--k", type=int, default=None)
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding="utf-8"))
    data_dir = os.path.join(HERE, cfg["paths"]["data"])
    runs_dir = os.path.join(HERE, cfg["paths"]["runs"])
    os.makedirs(runs_dir, exist_ok=True)

    models = [m for m in cfg["models"]
              if not a.models or m["name"] in a.models.split(",")]
    tasks = [t for t in cfg["tasks"]
             if not a.tasks or t["name"] in a.tasks.split(",")]
    budgets = [b for b in cfg["budgets"]
               if not a.budgets or b["name"] in a.budgets.split(",")]
    k = a.k or cfg["sampling"]["k"]
    temp = cfg["sampling"]["temperature"]
    conc = cfg["run"]["concurrency"]
    lock, stats = threading.Lock(), {"calls": 0, "errors": 0, "tokens": 0}

    def make_client(m):
        if a.mock:
            return MockClient(m["name"])
        from providers import Client
        return Client(m["provider"], m.get("model_id", m["name"]),
                      temperature=temp, timeout=cfg["run"]["timeout"],
                      max_retries=cfg["run"]["max_retries"])

    def work(client, model_cfg, item, budget, template, sample_idx, out):
        try:
            if a.mock:
                res = client.respond(item, budget["name"])
            else:
                system = PROMPTS[item["task"]] + TEMPLATE_SUFFIX[template]
                res = client.chat(system, item["question"],
                                  max_tokens=budget["max_tokens"],
                                  extra_body=thinking_body(model_cfg, budget))
            usage = res.get("usage", {})
            rec = {"item_id": item["id"], "task": item["task"],
                   "sample_idx": sample_idx, "budget": budget["name"],
                   "max_tokens": budget["max_tokens"], "template": template,
                   "question": item["question"], "gold": item["gold"],
                   "aliases": item.get("aliases", []),
                   "response": res.get("text", ""),
                   "reasoning_tokens": usage.get("completion_tokens_details", {})
                                       .get("reasoning_tokens"),
                   "finish_reason": res.get("finish_reason"),
                   "usage": usage, "error": None}
            with lock:
                stats["calls"] += 1
                stats["tokens"] += usage.get("completion_tokens", 0)
        except Exception as e:
            rec = {"item_id": item["id"], "task": item["task"],
                   "sample_idx": sample_idx, "budget": budget["name"],
                   "max_tokens": budget["max_tokens"], "template": template,
                   "question": item["question"], "gold": item["gold"],
                   "aliases": item.get("aliases", []),
                   "response": "", "reasoning_tokens": None,
                   "finish_reason": None, "usage": {},
                   "error": f"{type(e).__name__}: {e}"}
            with lock:
                stats["errors"] += 1
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()

    for m in models:
        client = make_client(m)
        for t in tasks:
            items = load_items(data_dir, t["file"], a.n or t["n"])
            for b in budgets:
                for tmpl in cfg["templates"]:
                    path = os.path.join(
                        runs_dir, f"{m['name']}__{t['name']}__{b['name']}__{tmpl}.jsonl")
                    have = done_keys(path)
                    todo = [(it, s) for it in items for s in range(k)
                            if (it["id"], s) not in have]
                    if not todo:
                        print(f"skip  {os.path.basename(path)} (complete)")
                        continue
                    print(f"run   {os.path.basename(path)}  {len(todo)} calls")
                    with open(path, "a", encoding="utf-8") as out, \
                            cf.ThreadPoolExecutor(max_workers=conc) as ex:
                        list(ex.map(lambda x: work(client, m, x[0], b, tmpl, x[1], out),
                                    todo))
    print(f"\ncalls={stats['calls']} errors={stats['errors']} "
          f"completion_tokens={stats['tokens']}")


if __name__ == "__main__":
    main()
