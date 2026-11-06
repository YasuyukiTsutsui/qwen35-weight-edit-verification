"""Merge the per-chunk sampling outputs of Issue #32 stage 1 into results/t1_samples/t1/ (gz members concatenate losslessly).
    python -I scripts/merge_sample_parts.py results/t1_samples_p0/t1_p0 results/t1_samples_p1/t1_p1 ... --out results/t1_samples/t1
"""
import argparse, gzip, json, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument("parts", nargs="+")
ap.add_argument("--out", default="results/t1_samples/t1")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
S = [json.load(open(os.path.join(p, "summary.json"))) for p in a.parts]
for name in ("samples.jsonl.gz", "tokens.jsonl.gz"):
    with open(os.path.join(a.out, name), "wb") as w:
        for p in a.parts:
            with open(os.path.join(p, name), "rb") as r:
                shutil.copyfileobj(r, w)
tot = sum(s["total"] for s in S)
m = dict(S[0])
m.update({"n_problems": sum(s["n_problems"] for s in S), "total": tot, "ok": sum(s["ok"] for s in S),
          "acc": sum(s["ok"] for s in S) / tot,
          "cut_share": sum(s["cut_share"] * s["total"] for s in S) / tot,
          "cut_with_loop_tail": sum(s["cut_with_loop_tail"] for s in S),
          "think_closed_share": sum(s["think_closed_share"] * s["total"] for s in S) / tot,
          "mean_tokens": sum(s["mean_tokens"] * s["total"] for s in S) / tot,
          "gen_seconds": sum(s["gen_seconds"] for s in S), "set": "data/t1.jsonl (merged from %d parts)" % len(S)})
json.dump(m, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
n = sum(1 for _ in gzip.open(os.path.join(a.out, "samples.jsonl.gz"), "rt", encoding="utf-8"))
nt = sum(1 for _ in gzip.open(os.path.join(a.out, "tokens.jsonl.gz"), "rt", encoding="utf-8"))
print("merged", len(S), "parts:", n, "samples,", nt, "token records; acc %.4f" % m["acc"])
assert n == tot == nt, (n, tot, nt)
