#!/usr/bin/env python3
"""Download and normalise the evaluation sets into one JSONL schema.

Unified item schema (one JSON object per line):
    {"id": str, "task": str, "question": str, "gold": str,
     "aliases": [str], "meta": {...}}

Tasks: math (GSM8K), factqa (Natural Questions), multihop (HotpotQA),
       falsepremise (SciFact claims known to be contradicted).

Usage:
    python -I fetch_data.py            # uses config.json defaults
    python -I fetch_data.py --n 50     # small sample for a smoke test
"""
import argparse, json, os, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Mozilla/5.0 (research)"}


def get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def hf_rows(dataset, config, split, total):
    """Page through the HF datasets-server /rows endpoint."""
    out, off = [], 0
    while len(out) < total:
        take = min(100, total - len(out))
        q = urllib.parse.urlencode({"dataset": dataset, "config": config,
                                    "split": split, "offset": off, "length": take})
        obj = json.loads(get("https://datasets-server.huggingface.co/rows?" + q))
        rows = obj.get("rows", [])
        if not rows:
            break
        out += [r["row"] for r in rows]
        off += len(rows)
        time.sleep(0.2)
    return out[:total]


def gsm8k(n):
    raw = get("https://raw.githubusercontent.com/openai/grade-school-math/"
              "master/grade_school_math/data/test.jsonl")
    items = []
    for i, line in enumerate(raw.splitlines()):
        if i >= n:
            break
        d = json.loads(line)
        gold = d["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"id": f"gsm8k-{i:04d}", "task": "math",
                      "question": d["question"], "gold": gold, "aliases": [],
                      "meta": {}})
    return items


def nq(n):
    rows = hf_rows("google-research-datasets/nq_open", "nq_open", "train", n)
    out = []
    for i, r in enumerate(rows):
        ans = r.get("answer") or []
        out.append({"id": f"nq-{i:04d}", "task": "factqa",
                    "question": r["question"].strip(),
                    "gold": ans[0] if ans else "", "aliases": ans[1:], "meta": {}})
    return out


def hotpotqa(n):
    rows = hf_rows("hotpotqa/hotpot_qa", "distractor", "validation", n)
    out = []
    for i, r in enumerate(rows):
        out.append({"id": f"hotpot-{i:04d}", "task": "multihop",
                    "question": r["question"].strip(), "gold": r["answer"],
                    "aliases": [], "meta": {"type": r.get("type", "")}})
    return out


SCIFACT_TGZ = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def _scifact_claim_files():
    """Local SciFact claim files, downloading the AI2 release if absent."""
    raw = os.path.join(HERE, "data", "_raw")
    files = [os.path.join(raw, "data", f)
             for f in ("claims_train.jsonl", "claims_dev.jsonl")]
    if all(os.path.exists(f) for f in files):
        return files
    import tarfile
    os.makedirs(raw, exist_ok=True)
    tgz = os.path.join(raw, "scifact.tar.gz")
    req = urllib.request.Request(SCIFACT_TGZ, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(tgz, "wb") as f:
        f.write(r.read())
    with tarfile.open(tgz) as t:
        t.extractall(raw, filter="data")     # no path traversal
    return files


def falsepremise(n):
    """SciFact claims whose evidence CONTRADICTS them -> false premises.

    Each false claim becomes a question that *presupposes* it, so the
    reliable behaviour is to refute. gold = expected behaviour.
    """
    items = []
    for path in _scifact_claim_files():
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            if len(items) >= n:
                break
            d = json.loads(line)
            labels = [e.get("label") for ev in d.get("evidence", {}).values()
                      for e in ev]
            if "CONTRADICT" not in labels:
                continue
            claim = d["claim"].strip().rstrip(".")
            items.append({
                "id": f"scifact-{d['id']}", "task": "falsepremise",
                "question": f"Why is it the case that {claim}?",
                "gold": "refute", "aliases": ["refute"],
                "meta": {"claim": claim, "source": "SciFact"}})
    return items


FETCHERS = {"gsm8k": gsm8k, "nq": nq, "hotpotqa": hotpotqa,
            "falsepremise": falsepremise}
TARGET = {"gsm8k": "gsm8k.jsonl", "nq": "nq.jsonl",
          "hotpotqa": "hotpotqa.jsonl", "falsepremise": "falsepremise.jsonl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="items per task")
    ap.add_argument("--outdir", default=os.path.join(HERE, "data"))
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
    default_n = {t["name"]: t["n"] for t in cfg["tasks"]}
    n = a.n or max(default_n.values())
    os.makedirs(a.outdir, exist_ok=True)
    for key, fn in FETCHERS.items():
        try:
            items = fn(n)
            path = os.path.join(a.outdir, TARGET[key])
            with open(path, "w", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            print(f"OK   {key:12s} {len(items):5d} items -> {TARGET[key]}")
        except Exception as e:
            print(f"FAIL {key:12s} {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
