"""Issue #13: can the edit strength / the edit itself be chosen WITHOUT answer labels? (CPU only, saved samples)

For every 16,384-token evaluation on the 128 select problems with 2 samples per problem (screens, search points, variation and extra
studies) we compute label-free proxies and compare them with the accuracy (which needs labels):
  loop share     share of samples cut by the cap with a repeating tail          (needs no labels; the proxy the first plan used)
  closed share   share of samples whose thinking was closed                       (needs no labels)
  agreement      share of problems where the two samples give the same non-empty answer (needs no labels; self-consistency)
  agreement|closed  the same among problems where both samples closed their thinking
  well-formed    share of samples that finished (stop) with a boxed answer after the thinking (needs no labels; format only)
Outputs: Spearman correlation of each proxy with accuracy across the edits, and what accuracy the edit picked by each proxy has
(regret = best accuracy among the edits - accuracy of the pick). Also: how many problems solved by the unedited model are lost.

    python -I scripts/analyze_label_free.py        # writes results/analysis_label_free.json
"""
import glob
import gzip
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import grade as G  # noqa: E402


def load(tag):
    f = os.path.join(ROOT, "results", tag, "practice_select128", "samples.jsonl.gz")
    by = {}
    for line in gzip.open(f, "rt", encoding="utf-8"):
        r = json.loads(line)
        by.setdefault(r["id"], []).append(r)
    return by


def main():
    rows = []
    for sp in sorted(glob.glob(os.path.join(ROOT, "results", "*", "practice_select128", "summary.json"))):
        s = json.load(open(sp))
        if s.get("max_tokens") != 16384:
            continue
        tag = sp.split(os.sep)[-3]
        rows.append((tag, s))
    base = load("rb16k_base")
    base_solved = {i for i, v in base.items() if any(r["correct"] for r in v)}

    out = {"edits": {}, "label_free_selection": {}, "correlations": {}}
    K2 = []
    print(f"{'edit':34s} {'K':>2s} {'acc':>6s} {'loop':>6s} {'closed':>7s} {'wellf':>6s} {'agree':>6s} {'agree|cl':>8s} {'lost base-solved':>16s}")
    for tag, s in rows:
        by = load(tag)
        n = sum(len(v) for v in by.values())
        K = round(n / len(by))
        acc = sum(r["correct"] for v in by.values() for r in v) / n
        loop = sum(1 for v in by.values() for r in v if r["finish"] == "length" and r["verbatim_loop_end"]) / n
        closed_flags = {i: [G.think_closed(r["text"]) for r in v] for i, v in by.items()}
        closed = sum(sum(f) for f in closed_flags.values()) / n
        BOX = chr(92) + 'boxed'
        wellformed = sum(1 for v in by.values() for r in v if r['finish'] == 'stop' and G.think_closed(r['text']) and BOX in G.split_think(r['text'])[1]) / n
        lost = sum(1 for i in base_solved if i in by and not any(r["correct"] for r in by[i]))
        agree = agree_cl = None
        if K >= 2:
            pairs = [(v[0], v[1]) for v in by.values() if len(v) >= 2]
            agree = float(np.mean([1.0 if (a["pred"] is not None and a["pred"] == b["pred"]) else 0.0 for a, b in pairs]))
            both = [(a, b) for (a, b), i in zip(pairs, [i for i, v in by.items() if len(v) >= 2])
                    if closed_flags[i][0] and closed_flags[i][1]]
            agree_cl = float(np.mean([1.0 if (a["pred"] is not None and a["pred"] == b["pred"]) else 0.0 for a, b in both])) if both else None
            K2.append(tag)
        out["edits"][tag] = {"K": K, "acc": acc, "loop": loop, "closed": closed, "wellformed": wellformed, "agree": agree, "agree_closed": agree_cl, "lost_base_solved": lost}
        print(f"{tag:34s} {K:2d} {acc:6.3f} {loop:6.3f} {closed:7.3f} {wellformed:6.3f} {('%.3f' % agree) if agree is not None else '   -  ':>6s} "
              f"{('%.3f' % agree_cl) if agree_cl is not None else '   -   ':>8s} {lost:>10d}/{len(base_solved)}")

    E = out["edits"]
    cand = [t for t in K2 if "base" not in t]
    acc = np.array([E[t]["acc"] for t in cand])
    print(f"\nacross {len(cand)} edits evaluated with 2 samples/problem (Spearman correlation of the proxy with accuracy)")
    for name, sign in [("loop", -1), ("closed", +1), ("wellformed", +1), ("agree", +1), ("agree_closed", +1)]:
        vals = np.array([E[t][name] if E[t][name] is not None else np.nan for t in cand], dtype=float)
        m = ~np.isnan(vals)
        rho, p = spearmanr(vals[m], acc[m]) if m.sum() >= 4 else (float("nan"), float("nan"))
        pick = cand[int(np.nanargmax(sign * vals))]
        out["correlations"][name] = {"spearman": float(rho), "p": float(p), "n": int(m.sum())}
        out["label_free_selection"][name] = {"pick": pick, "acc_of_pick": E[pick]["acc"], "regret": float(acc.max() - E[pick]["acc"])}
        print(f"  {name:13s} rho {rho:+.2f} (p {p:.2f}, n={int(m.sum())})  pick -> {pick} acc {E[pick]['acc']:.3f}  regret {100 * (acc.max() - E[pick]['acc']):.1f} pt")
    best = cand[int(acc.argmax())]
    print(f"  accuracy (needs labels) picks -> {best} acc {E[best]['acc']:.3f}")
    out["label_free_selection"]["accuracy"] = {"pick": best, "acc_of_pick": E[best]["acc"], "regret": 0.0}
    json.dump(out, open(os.path.join(ROOT, "results", "analysis_label_free.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
