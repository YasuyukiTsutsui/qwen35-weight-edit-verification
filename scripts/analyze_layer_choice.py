"""Why do layers 15+19 work better than layers 10 / 14 / 10+14?  CPU-only analysis of saved 16,384-token samples.

Inputs : results/<run>/practice_select128/samples.jsonl.gz  (same 128 problems for every run; K=1 for rb16k_*, K=2 for scr_*)
Outputs: results/analysis_layer_choice.json  (+ a printed summary)

Contents
  1. outcome decomposition  accuracy = P(thinking closed) x P(correct | closed), cut-off type (verbatim loop / other repetition / still thinking)
  2. paired problem-level bootstrap of accuracy differences
  3. where the loops start and their period (find_loop from find_direction_mp.py, copied here to avoid importing torch)
"""
import gzip
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import grade as G  # noqa: E402

RUNS = {
    "base": "rb16k_base", "published p50c": "rb16k_p50c", "old-dir L15+19 a6": "rb16k_own_k64_a6", "old-dir L15+19 a12": "rb16k_own_k64_a12",
    "mp-dir L10": "scr_t1_L10_r12", "mp-dir L14": "scr_t1_L14_r12", "mp-dir L10+14": "scr_t1_L10_14_r12", "mp-dir L15+19": "scr_t1_mp_L15_19_r12",
}


def find_loop(a, min_p=6, max_p=700, tail=3000):
    n = len(a)
    if n < 2000:
        return None
    T = a[-tail:]
    p = None
    for q in range(min_p, min(max_p, len(T) // 4) + 1):
        if (T[q:] == T[:-q]).mean() >= 0.9:
            p = q
            break
    if p is None:
        return None
    eq = a[p:] == a[:-p]
    good = 0
    for b in range(1, len(eq) // p + 1):
        blk = eq[len(eq) - b * p: len(eq) - (b - 1) * p]
        if blk.mean() >= 0.9:
            good = b
        else:
            break
    if good < 6:
        return None
    i0 = n - good * p - p
    if i0 < 0:
        return None
    while i0 > 0 and eq[i0 - 1]:
        i0 -= 1
    return i0 + p, p


def load(run):
    f = os.path.join(ROOT, "results", run, "practice_select128", "samples.jsonl.gz")
    return [json.loads(l) for l in gzip.open(f, "rt", encoding="utf-8")]


def main():
    rng = np.random.default_rng(0)
    data = {k: load(v) for k, v in RUNS.items()}
    out = {"decomposition": {}, "bootstrap": {}, "loops": {}}

    print(f"{'edit':20s} {'n':>4s} {'acc':>6s} {'closed':>7s} {'acc|closed':>10s} {'verbatim':>9s} {'other-rep':>9s} {'still-thinking':>14s}")
    for k, R in data.items():
        n = len(R)
        closed = [G.think_closed(r["text"]) for r in R]
        cut = [r for r in R if r["finish"] == "length"]
        vb = sum(r["verbatim_loop_end"] for r in cut)
        rp = sum(1 for r in cut if r["repetitive_end"] and not r["verbatim_loop_end"])
        d = {"n": n, "acc": float(np.mean([r["correct"] for r in R])), "closed": sum(closed) / n,
             "acc_given_closed": float(np.mean([r["correct"] for r, c in zip(R, closed) if c])) if any(closed) else None,
             "cut_verbatim_loop": vb / n, "cut_other_repetition": rp / n, "cut_still_thinking": (len(cut) - vb - rp) / n}
        out["decomposition"][k] = d
        print(f"{k:20s} {n:4d} {d['acc']:6.3f} {d['closed']:7.3f} {d['acc_given_closed']:10.3f} {d['cut_verbatim_loop']:9.3f} "
              f"{d['cut_other_repetition']:9.3f} {d['cut_still_thinking']:14.3f}")

    ids = sorted({r["id"] for r in data["base"]})
    P = {k: np.array([np.mean([r["correct"] for r in R if r["id"] == i]) for i in ids]) for k, R in data.items()}
    print("\npaired problem-level bootstrap (128 problems), accuracy difference [95% CI]")
    for a, b in [("mp-dir L15+19", "base"), ("mp-dir L15+19", "mp-dir L10+14"), ("mp-dir L15+19", "mp-dir L10"), ("mp-dir L15+19", "mp-dir L14"),
                 ("mp-dir L10+14", "mp-dir L10"), ("mp-dir L10+14", "mp-dir L14"), ("mp-dir L15+19", "old-dir L15+19 a6"),
                 ("mp-dir L15+19", "published p50c"), ("published p50c", "old-dir L15+19 a6")]:
        d = P[a] - P[b]
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        out["bootstrap"][f"{a} - {b}"] = [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        print(f"  {a:18s} - {b:18s}: {100 * d.mean():+5.1f} pt [{100 * np.percentile(bs, 2.5):+5.1f}, {100 * np.percentile(bs, 97.5):+5.1f}]")

    from tokenizers import Tokenizer
    _tk = Tokenizer.from_file(os.path.join(ROOT, "models", "Qwen3.5-0.8B", "tokenizer.json"))

    def tok(text, add_special_tokens=False):
        return {"input_ids": _tk.encode(text, add_special_tokens=False).ids}

    print("\nloops: onset (tokens before the periodic region starts) and period, verbatim-loop samples only")
    print(f"{'edit':20s} {'n_loops':>7s} {'found':>6s} {'onset median':>12s} {'onset p10-p90':>16s} {'period median':>13s} {'period p10-p90':>15s} {'<=100':>6s}")
    for k, R in data.items():
        found = []
        nl = 0
        for r in R:
            if r["finish"] == "length" and r["verbatim_loop_end"]:
                nl += 1
                fl = find_loop(np.asarray(tok(r["text"], add_special_tokens=False)["input_ids"]))
                if fl:
                    found.append(fl)
        if not found:
            continue
        s = np.array([f[0] for f in found])
        p = np.array([f[1] for f in found])
        d = {"n_loops": nl, "found": len(found), "onset_median": float(np.median(s)), "onset_p10_p90": [float(np.percentile(s, 10)), float(np.percentile(s, 90))],
             "period_median": float(np.median(p)), "period_p10_p90": [float(np.percentile(p, 10)), float(np.percentile(p, 90))],
             "share_period_le_100": float((p <= 100).mean())}
        out["loops"][k] = d
        print(f"{k:20s} {nl:7d} {len(found):6d} {d['onset_median']:12.0f} {d['onset_p10_p90'][0]:7.0f}-{d['onset_p10_p90'][1]:<7.0f} "
              f"{d['period_median']:13.0f} {d['period_p10_p90'][0]:7.0f}-{d['period_p10_p90'][1]:<7.0f} {d['share_period_le_100']:6.2f}")

    json.dump(out, open(os.path.join(ROOT, "results", "analysis_layer_choice.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
