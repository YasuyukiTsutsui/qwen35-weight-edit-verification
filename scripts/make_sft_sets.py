"""Issue #32, stage 1: choose the training samples from the 8-sample run on the unsealed training problems T1.  Host only; uses saved flags, no model.

    python -I scripts/make_sft_sets.py [--samples results/t1_samples/t1] [--out results/t1_sft]   -> M.json, P.json, G.json, stats.json

Rules (written into Issue #32 before any training):
  eligible sample : finish == 'stop', an extracted answer, at most 8,192 response tokens.
  vote            : over ALL 8 samples of a problem, by the normalised answer string (as in analyze_unsealed.py step D).
  M (majority)    : problems whose vote has a UNIQUE plurality with at least 4 of 8 votes; the eligible samples that agree with it, the first 2 by k.   (no gold label used)
  P (placebo)     : the same problems as M, the same number of samples per problem, drawn at random (seed 0) from the problem's eligible samples whatever they answer.
  G (gold)        : problems with at least one eligible correct sample; the first 2 correct ones by k; problems drawn in random order (seed 0) until the number of samples reaches that of M.
"""
import argparse
import gzip
import json
import os
import random
from collections import Counter

MAX_TOK, MIN_VOTES = 8192, 4


def norm(x):
    return " ".join(str(x).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="results/t1_samples/t1")
    ap.add_argument("--out", default="results/t1_sft")
    a = ap.parse_args()
    byp = {}
    for l in gzip.open(os.path.join(a.samples, "samples.jsonl.gz"), "rt", encoding="utf-8"):
        r = json.loads(l)
        byp.setdefault(r["id"], []).append(r)
    for rs in byp.values():
        rs.sort(key=lambda r: r["k"])
    elig = lambda r: r["finish"] == "stop" and r["pred"] is not None and r["tokens"] <= MAX_TOK  # noqa: E731
    M, P, G, info = [], [], [], {}
    rnd = random.Random(0)
    for pid in sorted(byp):
        rs = byp[pid]
        c = Counter(norm(r["pred"]) for r in rs if r["pred"] is not None)
        if not c:
            continue
        top = max(c.values())
        win = [k for k in c if c[k] == top]
        if top >= MIN_VOTES and len(win) == 1:
            agree = [r for r in rs if elig(r) and norm(r["pred"]) == win[0]][:2]
            if agree:
                M += [(pid, r["k"]) for r in agree]
                info[pid] = {"n": len(agree), "pool": [r for r in rs if elig(r)]}
    for pid, v in info.items():
        pool = v["pool"]
        pick = rnd.sample(pool, min(v["n"], len(pool)))
        P += [(pid, r["k"]) for r in pick]
    gold = []
    for pid in sorted(byp):
        cor = [r for r in byp[pid] if elig(r) and r["correct"]][:2]
        if cor:
            gold.append((pid, cor))
    rnd.shuffle(gold)
    for pid, cor in gold:
        if len(G) >= len(M):
            break
        G += [(pid, r["k"]) for r in cor][:max(len(M) - len(G), 0)]
    flat = {(r["id"], r["k"]): r for rs in byp.values() for r in rs}
    st = {}
    for name, S in (("M", M), ("P", P), ("G", G)):
        rs = [flat[p] for p in S]
        st[name] = {"samples": len(S), "problems": len({p[0] for p in S}), "gsm8k_samples": sum(p[0].startswith("gsm8k") for p in S),
                    "label_correct": sum(bool(r["correct"]) for r in rs) / max(len(rs), 1), "mean_tokens": sum(r["tokens"] for r in rs) / max(len(rs), 1),
                    "total_tokens": sum(r["tokens"] for r in rs)}
        print(name, st[name])
    os.makedirs(a.out, exist_ok=True)
    for name, S in (("M", M), ("P", P), ("G", G)):
        json.dump({"rule": name, "pairs": [list(p) for p in S]}, open(os.path.join(a.out, f"{name}.json"), "w"))
    allr = [r for rs in byp.values() for r in rs]
    st["all"] = {"problems": len(byp), "samples": len(allr), "eligible": sum(elig(r) for r in allr), "cut_share": 1 - sum(r["finish"] == "stop" for r in allr) / len(allr),
                 "pass@1": sum(bool(r["correct"]) for r in allr) / len(allr)}
    print("all", st["all"])
    json.dump(st, open(os.path.join(a.out, "stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
