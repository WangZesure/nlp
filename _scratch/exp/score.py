#!/usr/bin/env python3
"""Score sweep outputs into the four reliability axes.

Reads runs/*.jsonl, parses each response, and writes per-cell metrics to
results/scores.json plus a flat CSV. No third-party libraries.

Axes:
    accuracy        numeric / alias match (math, factqa, multihop)
    ece             expected calibration error (sample level)
    overconfidence  mean verbalized confidence on incorrect answers
    fabrication     share of false-premise items ENDORSED (refute = correct)
"""
import argparse, glob, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))

ANS_RE = re.compile(r"Answer:\s*(.+)")
CONF_RE = re.compile(r"[Cc]onfidence:\s*([0-9]*\.?[0-9]+)")
VERD_RE = re.compile(r"[Vv]erdict:\s*(endorse|refuse)")


def parse(resp):
    m = ANS_RE.search(resp)
    ans = m.group(1).strip().split("\n")[0] if m else None
    c = re.search(r"[Cc]onfidence:\s*([0-9]*\.?[0-9]+)", resp)
    conf = float(c.group(1)) if c else None
    v = VERD_RE.search(resp)
    return ans, conf, (v.group(1) if v else None)


def norm(s):
    s = (s or "").lower()
    s = s.replace(" ", " ").replace("–", "-").replace("—", "-")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def num(s):
    if s is None:
        return None
    m = re.search(r"-?\d[\d,]*\.?\d*", str(s).replace("$", ""))
    return float(m.group(0).replace(",", "")) if m else None


def correct(task, gold, aliases, resp):
    ans, _, _ = parse(resp)
    if task == "math":
        g, p = num(gold), num(ans)
        return (g is not None and p is not None and abs(g - p) < 1e-6)
    # factqa / multihop: alias match
    if not ans:
        return False
    pa = norm(ans)
    for cand in [gold] + list(aliases or []):
        nc = norm(cand)
        if nc and (nc == pa or nc in pa):
            return True
    return False


def ece(confs, oks, bins=5):
    if not confs:
        return None
    n, tot = len(confs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs)
               if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(oks[i] for i in idx) / len(idx)
        cf = sum(confs[i] for i in idx) / len(idx)
        tot += abs(acc - cf) * len(idx) / n
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cells = {}
    for path in sorted(glob.glob(os.path.join(a.runs, "*.jsonl"))):
        base = os.path.basename(path)[:-6]
        model, task, budget, tmpl = base.split("__")
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        recs = [r for r in recs if not r.get("error")]
        if not recs:
            continue
        rows = []
        for r in recs:
            _, conf, verdict = parse(r["response"])
            if task == "falsepremise":
                ok = 1.0 if verdict == "refuse" else 0.0
                fab = 1.0 if verdict == "endorse" else 0.0
                hedge = 1.0 if verdict is None else 0.0
            else:
                ok = 1.0 if correct(task, r["gold"], r.get("aliases", []),
                                    r["response"]) else 0.0
                fab = hedge = None
            rows.append({"ok": ok, "conf": conf, "fab": fab, "hedge": hedge,
                         "tok": r.get("usage", {}).get("completion_tokens", 0),
                         "rtok": r.get("reasoning_tokens"),
                         "trunc": 1.0 if r.get("finish_reason") == "length" else 0.0})
        cells[(model, task, budget, tmpl)] = rows

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.fmean(xs) if xs else None

    out = {}
    for (model, task, budget, tmpl), rows in sorted(cells.items()):
        paired = [(r["conf"], r["ok"]) for r in rows if r["conf"] is not None]
        confs = [c for c, _ in paired]
        oks = [o for _, o in paired]
        out[f"{model}|{task}|{budget}|{tmpl}"] = {
            "model": model, "task": task, "budget": budget, "template": tmpl,
            "n": len(rows),
            "accuracy": mean([r["ok"] for r in rows]),
            "mean_conf": mean(confs),
            "ece": ece(confs, oks) if confs else None,
            "overconfidence": mean([c for c, o in paired if o < 0.5]),
            "fabrication": mean([r["fab"] for r in rows]),
            "hedge_rate": mean([r["hedge"] for r in rows]),
            "truncation": mean([r["trunc"] for r in rows]),
            "mean_tokens": mean([r["tok"] for r in rows]),
            "mean_reasoning_tokens": mean([r["rtok"] for r in rows]),
        }
    json.dump(out, open(os.path.join(a.out, "scores.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    cols = ("model", "task", "budget", "template", "n", "accuracy", "ece",
            "overconfidence", "fabrication", "hedge_rate", "truncation",
            "mean_tokens", "mean_reasoning_tokens")
    with open(os.path.join(a.out, "scores.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for v in out.values():
            f.write(",".join(str(v[k]) for k in cols) + "\n")
    print(f"scored {len(out)} cells -> {a.out}/scores.json")


if __name__ == "__main__":
    main()
