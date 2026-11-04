"""Issue #25 (N3): how cheaply can the best edit location / strength be found?  (CPU only, saved samples, no GPU)

The expensive step of every search in this repo is one full evaluation: 128 problems x 2 samples at a 16,384-token cap
(14-17 GPU minutes). Cost is dominated by long (looping / non-terminating) samples, so cost is counted in GENERATED TOKENS.
This script replays cheaper search protocols on the edits that were already fully evaluated and asks how often each one picks
the edit that the full protocol ranks first (and how many accuracy points it loses otherwise):

  S0  full evaluation of every edit                             (reference, cost 100%)
  S1  evaluate every edit on a random subset of m problems      (uniform subsampling)
  S2  successive halving: 16 -> 32 -> 64 -> 128 problems, the better half survives each round
  S3  label-free prefilter: well-formed-completion share on m problems, top-4 are then evaluated fully (#21)
  S4  cheap token cap: accuracy at a 4,096 / 8,192 cap on all problems, top-4 are then evaluated fully

Limits (read before quoting): the "truth" is the full-protocol accuracy of the SAME 128 problems, so this measures how well a cheaper
protocol reproduces the full ranking, not generalisation (that is the job of the validation / fresh sets). Accuracy at a lower cap is
approximated as "correct and finished within the cap". The pool is the edits of this study (layer sets, strengths, column counts), not a
random sample of all possible locations; the top of the pool is within about +-1.5 pt of sampling noise, so some regret is unavoidable.

    python -I scripts/analyze_search_efficiency.py        # in-sample pool v1 (frozen in results/search_efficiency_pool_v1.json) -> results/analysis_search_efficiency.json
    python -I scripts/analyze_search_efficiency.py --oos  # out-of-sample: only edits NOT in pool v1 (strategies fixed in advance, see Issue #25)
"""
import glob
import gzip
import json
import math
import os
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import grade as G  # noqa: E402

BOX = chr(92) + "boxed"
R = 400            # Monte-Carlo repetitions
TOPK = 4           # survivors that get a full evaluation in S3 / S4


def load(tag):
    f = os.path.join(ROOT, "results", tag, "practice_select128", "samples.jsonl.gz")
    by = {}
    for line in gzip.open(f, "rt", encoding="utf-8"):
        r = json.loads(line)
        by.setdefault(r["id"], []).append(r)
    return by


def build_pool(exclude=frozenset(), exclude_prefix=()):
    tags, per = [], {}
    for sp in sorted(glob.glob(os.path.join(ROOT, "results", "*", "practice_select128", "summary.json"))):
        s = json.load(open(sp))
        tag = sp.split(os.sep)[-3]
        if s.get("max_tokens") != 16384 or "base" in tag or tag in exclude or tag.startswith(exclude_prefix):
            continue
        by = load(tag)
        if any(len(v) != 2 for v in by.values()):
            continue
        tags.append(tag)
        per[tag] = by
    if not tags:
        raise SystemExit("no edits in the pool yet (for --oos: no fully evaluated edit outside pool v1)")
    ids = sorted(set.intersection(*[set(per[t]) for t in tags]))
    E, P = len(tags), len(ids)
    corr = np.zeros((E, P, 2), bool)
    tok = np.zeros((E, P, 2), np.int64)
    wf = np.zeros((E, P, 2), bool)
    for e, t in enumerate(tags):
        for p, i in enumerate(ids):
            for k, r in enumerate(per[t][i][:2]):
                corr[e, p, k] = bool(r["correct"])
                tok[e, p, k] = int(r["tokens"])
                wf[e, p, k] = (r["finish"] == "stop" and G.think_closed(r["text"]) and BOX in G.split_think(r["text"])[1])
    return tags, ids, corr, tok, wf


def main():
    oos = "--oos" in sys.argv
    frozen_path = os.path.join(ROOT, "results", "search_efficiency_pool_v1.json")
    if oos:
        tags, ids, corr, tok, wf = build_pool(set(json.load(open(frozen_path))["pool"]), ("null_",))  # null-control edits were not in the pre-registered list
        print("OUT-OF-SAMPLE pool (edits not in pool v1)")
    else:
        tags, ids, corr, tok, wf = build_pool()
        if not os.path.exists(frozen_path):
            json.dump({"pool": tags, "note": "frozen 2026-11-04; the out-of-sample check uses only edits not listed here"}, open(frozen_path, "w"), indent=1)
    E, P = len(tags), len(ids)
    rng = np.random.default_rng(0)
    full_acc = corr.mean((1, 2))
    best = full_acc.max()
    top3 = set(np.argsort(-full_acc)[:3].tolist())
    c_full = float(tok.sum())
    print(f"pool: {E} edits x {P} problems x 2 samples; best full accuracy {100 * best:.1f}% ({tags[int(full_acc.argmax())]}); "
          f"mean tokens/sample {tok.mean():.0f}\n")

    def pick(score):
        return int(np.argmax(score + rng.random(len(score)) * 1e-9))

    def summarize(name, picks, costs):
        picks = np.array(picks)
        reg = 100 * (best - full_acc[picks])
        r = {"cost_pct": float(100 * np.mean(costs)), "regret_mean_pt": float(reg.mean()), "p_best_within_1pt": float(np.mean(reg <= 1.0)),
             "p_top3": float(np.mean([int(p) in top3 for p in picks]))}
        print(f"{name:44s} cost {r['cost_pct']:6.1f}%   regret {r['regret_mean_pt']:5.2f} pt   within 1pt of best {100 * r['p_best_within_1pt']:5.1f}%   top-3 {100 * r['p_top3']:5.1f}%")
        return r

    out = {"pool": tags, "full_acc": {t: float(a) for t, a in zip(tags, full_acc)}, "strategies": {}}
    print(f"{'strategy':44s} {'cost':>12s}   {'regret':>14s}   {'within 1pt':>20s}   {'top-3':>10s}")
    out["strategies"]["S0 full"] = summarize("S0 full evaluation of every edit", [int(full_acc.argmax())], [1.0])

    for m in (8, 16, 32, 64):
        picks, costs = [], []
        for _ in range(R):
            idx = rng.choice(P, m, replace=False)
            picks.append(pick(corr[:, idx].mean((1, 2))))
            costs.append(tok[:, idx].sum() / c_full)
        out["strategies"][f"S1 subsample m={m}"] = summarize(f"S1 uniform subsample, {m} problems", picks, costs)

    for m0 in (8, 16):
        sizes, p_, c_ = [], [], []
        s = m0
        while s < P:
            sizes.append(s)
            s *= 2
        sizes.append(P)
        for _ in range(R):
            order = rng.permutation(P)
            alive = np.arange(E)
            cost, prev = 0.0, 0
            for j, sz in enumerate(sizes):
                new = order[prev:sz]
                cost += tok[np.ix_(alive, new)].sum()
                prev = sz
                acc = corr[np.ix_(alive, order[:sz])].mean((1, 2))
                if j < len(sizes) - 1:
                    alive = alive[np.argsort(-(acc + rng.random(len(alive)) * 1e-9))[:max(1, math.ceil(len(alive) / 2))]]
            p_.append(int(alive[pick(acc)]))
            c_.append(cost / c_full)
        out["strategies"][f"S2 halving from {m0}"] = summarize(f"S2 successive halving {m0}->{P} problems", p_, c_)

    for m in (16, 32, 64):
        picks, costs = [], []
        for _ in range(R):
            idx = rng.choice(P, m, replace=False)
            keep = np.argsort(-(wf[:, idx].mean((1, 2)) + rng.random(E) * 1e-9))[:TOPK]
            cost = tok[:, idx].sum() + tok[keep].sum() - tok[np.ix_(keep, idx)].sum()
            picks.append(int(keep[pick(full_acc[keep])]))
            costs.append(cost / c_full)
        out["strategies"][f"S3 label-free prefilter m={m}"] = summarize(f"S3 label-free prefilter on {m} problems, top-{TOPK} full", picks, costs)

    caps = {}
    for cap in (2048, 4096, 8192):
        acc_cap = (corr & (tok <= cap)).mean((1, 2))
        keep = np.argsort(-acc_cap)[:TOPK]
        cost = np.minimum(tok, cap).sum() + tok[keep].sum()
        rho = spearmanr(acc_cap, full_acc)[0]
        caps[cap] = float(rho)
        picks = [int(keep[int(np.argmax(full_acc[keep]))])]
        out["strategies"][f"S4 cap {cap}"] = summarize(f"S4 cap {cap} on all, top-{TOPK} full  (rank rho {rho:+.2f})", picks, [cost / c_full])
    out["rank_corr_acc_at_cap_vs_full"] = caps

    pr = {}
    for name, arr in (("wellformed", wf), ("closed_free_accuracy_proxy", corr)):
        pr[name] = float(spearmanr(arr.mean((1, 2)), full_acc)[0])
    out["rank_corr_full_sample"] = pr
    print(f"\nrank correlation with the full accuracy over the pool: well-formed share {pr['wellformed']:+.2f}; "
          f"accuracy at cap 2048/4096/8192: " + " / ".join(f"{caps[c]:+.2f}" for c in (2048, 4096, 8192)))
    json.dump(out, open(os.path.join(ROOT, "results", "analysis_search_efficiency_oos.json" if oos else "analysis_search_efficiency.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
