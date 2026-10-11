#!/usr/bin/env python3
"""Turn per-cell scores into the objects the paper reports:
profile, reliability tax, and the oracle stopping budget.

Usage:  python -I analyze.py
"""
import argparse, collections, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGET_ORDER = ["off", "low", "high", "max"]
LO, HI = "b1", None      # HI set at runtime to the last budget present
# orient each axis so that higher is better
AXES = {"accuracy": +1, "ece": -1, "overconfidence": -1, "fabrication": -1}


def usable(acc, ece, eps):
    """Accuracy at a fixed reliability floor: served only if ECE <= eps."""
    return acc if (ece is not None and ece <= eps) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--eps", type=float, default=0.10, help="calibration floor")
    a = ap.parse_args()
    sc = json.load(open(os.path.join(a.results, "scores.json"), encoding="utf-8"))

    # average over templates -> one profile per (model, task, budget)
    cells = collections.defaultdict(list)
    for v in sc.values():
        cells[(v["model"], v["task"], v["budget"])].append(v)

    def agg(model, task, budget, key):
        xs = [c[key] for c in cells.get((model, task, budget), []) if c[key] is not None]
        return statistics.fmean(xs) if xs else None

    models = sorted({k[0] for k in cells})
    tasks = sorted({k[1] for k in cells})
    budgets = [b for b in BUDGET_ORDER if any(k[2] == b for k in cells)]
    lo_b, hi_b = budgets[0], budgets[-1]
    report = {}
    for model in models:
        for task in tasks:
            prof = {b: {k: agg(model, task, b, k) for k in AXES} for b in budgets}
            lo, hi = prof[lo_b], prof[hi_b]
            d_acc = (hi["accuracy"] or 0) - (lo["accuracy"] or 0)
            tax = {}
            if abs(d_acc) > 1e-9:
                for ax, orient in AXES.items():
                    if ax == "accuracy" or lo[ax] is None or hi[ax] is None:
                        continue
                    tax[ax] = orient * (hi[ax] - lo[ax]) / d_acc
            # oracle: budget maximising usable accuracy
            best = max(
                ((b, usable(p["accuracy"], p["ece"], a.eps)) for b, p in prof.items()),
                key=lambda t: (t[1] is not None, t[1] or -1))
            report[f"{model}|{task}"] = {
                "profile": prof, "tax_b1_to_b4": tax,
                "oracle_budget": best[0], "usable_accuracy": best[1],
            }

    json.dump(report, open(os.path.join(a.results, "profile.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)

    fmt = lambda x: f"{x:.3f}" if x is not None else "  -  "
    print(f"{'model':<18}{'task':<13}{'acc lo->hi':<16}{'ece hi':>8}"
          f"{'fab hi':>8}{'tax_ece':>9}{'oracle':>8}{'usable':>8}")
    for k, r in sorted(report.items()):
        model, task = k.split("|")
        bl = list(r["profile"])
        lo, hi = r["profile"][bl[0]], r["profile"][bl[-1]]
        acc = f"{fmt(lo['accuracy'])}->{fmt(hi['accuracy'])}"
        print(f"{model:<18}{task:<13}{acc:<16}{fmt(hi['ece']):>8}"
              f"{fmt(hi['fabrication']):>8}{fmt(r['tax_b1_to_b4'].get('ece')):>9}"
              f"{r['oracle_budget']:>8}{fmt(r['usable_accuracy']):>8}")
    print(f"\nwrote {a.results}/profile.json   (calibration floor eps={a.eps})")


if __name__ == "__main__":
    main()
