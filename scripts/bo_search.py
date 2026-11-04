"""Method B: black-box search of the edit strength on the real objective (accuracy at a 16,384-token cap).

Space: alpha in [2, 16] (log scale; applied to layers 15 and 19) x k in {64, 128, 256} (columns edited per layer).
Direction u and the column ranking come from --dir (find_direction output, fixed).
Algorithm: Sobol initial design, then noisy expected improvement on a GP (observation noise p(1-p)/n_seq fixed from the data),
then the top-3 configurations by posterior mean are re-evaluated on a disjoint validation set; the one with the best
validation accuracy is reported as the method-B candidate. Resumable: state in results/bo/state.json.

    python scripts/bo_search.py --build-informative            # after the 16k re-baseline runs exist
    python scripts/bo_search.py --run --n-init 8 --n-bo 9
    python scripts/bo_search.py --synthetic                     # offline test of the GP / EI code, no GPU
"""
import argparse
import glob
import gzip
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from scipy.stats import norm, qmc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASH = r"C:\Program Files\Git\bin\bash.exe"
STATE = os.path.join(ROOT, "results", "bo", "state.json")
A_LO, A_HI = 2.0, 16.0
KS = (64, 128, 256)
AGRID = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10, 12, 14, 16]


def to_x(alpha, k):
    return np.array([(math.log(alpha) - math.log(A_LO)) / (math.log(A_HI) - math.log(A_LO)), (math.log2(k) - 6) / 2.0])


def kern(A, B, ls):
    d = (A[:, None, :] - B[None, :, :]) / ls
    return np.exp(-0.5 * (d ** 2).sum(-1))


def fit_gp(X, y, nv):
    best = None
    m = float(np.mean(y))
    for l1 in (0.25, 0.4, 0.7):
        for l2 in (0.3, 0.6, 1.2):
            for sf in (0.5, 1.0, 2.0):
                sf2 = sf * max(float(np.var(y)), 1e-4)
                ls = np.array([l1, l2])
                K = sf2 * kern(X, X, ls) + np.diag(nv)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    continue
                a = np.linalg.solve(L.T, np.linalg.solve(L, y - m))
                ll = -0.5 * (y - m) @ a - np.log(np.diag(L)).sum()
                if best is None or ll > best[0]:
                    best = (ll, ls, sf2, m)
    return best


def predict(X, y, nv, Xs, hp):
    _, ls, sf2, m = hp
    K = sf2 * kern(X, X, ls) + np.diag(nv)
    L = np.linalg.cholesky(K)
    Ks = sf2 * kern(X, Xs, ls)
    a = np.linalg.solve(L.T, np.linalg.solve(L, y - m))
    mu = m + Ks.T @ a
    v = np.linalg.solve(L, Ks)
    return mu, np.sqrt(np.maximum(sf2 - (v ** 2).sum(0), 1e-12))


def posterior_table(state):
    X = np.array([to_x(s["alpha"], s["k"]) for s in state])
    y = np.array([s["ok"] / s["n_seq"] for s in state])
    nv = np.array([max(p * (1 - p), 0.02) / s["n_seq"] for p, s in zip(y, state)])
    hp = fit_gp(X, y, nv)
    return X, y, nv, hp


def next_by_ei(state):
    X, y, nv, hp = posterior_table(state)
    mu_obs, _ = predict(X, y, nv, X, hp)
    best = float(mu_obs.max())
    cands = [(a, k) for a in AGRID for k in KS]
    Xc = np.array([to_x(a, k) for a, k in cands])
    mu, sd = predict(X, y, nv, Xc, hp)
    z = (mu - best) / sd
    ei = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
    for j, c in enumerate(cands):          # at most 2 evaluations of the same configuration
        if sum((s["alpha"], s["k"]) == c for s in state) >= 2:
            ei[j] = -1.0
    i = int(np.argmax(ei))
    return cands[i], {"ei": float(ei[i]), "mu": float(mu[i]), "sd": float(sd[i]), "best_mu": best, "ls": hp[1].tolist(), "sf2": hp[2]}


def init_design(n):
    u = qmc.Sobol(d=2, scramble=True, seed=0).random(n)
    pts = []
    for u1, u2 in u:
        a = math.exp(math.log(A_LO) + u1 * (math.log(A_HI) - math.log(A_LO)))
        pts.append((round(a * 2) / 2, KS[min(int(u2 * 3), 2)]))
    return pts


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else []


def save_state(st):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    json.dump(st, open(STATE, "w"), indent=1)


def sh(cmd, log):
    """Run a command in the GPU container; retry a few times if the preflight check (free RAM / GPU) refuses to start."""
    rc = 1
    for _ in range(5):
        with open(os.path.join(ROOT, log), "a") as f:
            rc = subprocess.run([BASH, "scripts/drun.sh", *cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            return 0
        if "preflight failed" in open(os.path.join(ROOT, log), errors="ignore").read()[-600:]:
            time.sleep(90)
            continue
        return rc
    return rc


def evaluate(alpha, k, rows_file, samples, seed, direction, tag_prefix):
    name = f"bo_k{k}_a{alpha:g}"
    edit = f"models/edits/{name}"
    log = f"results/logs/{tag_prefix}_{name}.log"
    if not os.path.exists(os.path.join(ROOT, edit, "edit_log.json")):
        rc = sh(["python3", "scripts/make_edit.py", "--base", "models/Qwen3.5-0.8B", "--dir", direction, "--layers", "15,19",
                 "--k", str(k), "--alpha", f"{alpha:g}", "--out", edit], log)
        if rc:
            raise RuntimeError(f"make_edit failed for {name} (see {log})")
    tag = f"{tag_prefix}_{name}_s{seed}"
    sp = os.path.join(ROOT, "results", tag, os.path.splitext(os.path.basename(rows_file))[0], "summary.json")
    for attempt in range(3):  # a run can die for host reasons (sleep/wake, Docker hiccup); a re-evaluated point must really be re-run
        if os.path.exists(sp) and attempt == 0:
            os.replace(sp, sp + ".prev")  # so that a stale summary of an earlier evaluation of the same tag is never mistaken for this run
        rc = sh(["python3", "scripts/eval_math.py", "--model", edit, "--tag", tag, "--rows-file", rows_file, "--samples", str(samples),
                 "--max-tokens", "16384", "--max-num-seqs", "64", "--gpu-mem", "0.7", "--seed", str(seed)], log)
        if not rc and os.path.exists(sp):
            break
        time.sleep(120)
    if rc or not os.path.exists(sp):
        raise RuntimeError(f"eval failed for {name} (see {log})")
    s = json.load(open(sp))
    return {"alpha": alpha, "k": k, "tag": tag, "ok": s["ok"], "n_seq": s["total"], "acc": s["acc"], "loops": s["cut_with_loop_tail"],
            "closed": round(s["think_closed_share"] * s["total"]), "gen_seconds": s["gen_seconds"]}


def build_informative(out):
    solved = set()
    for pat in ("results/sweep_*/practice_select128/samples.jsonl.gz", "results/rb16k_*/practice_select128/samples.jsonl.gz"):
        for f in glob.glob(os.path.join(ROOT, pat)):
            for line in gzip.open(f, "rt", encoding="utf-8"):
                r = json.loads(line)
                if r["correct"]:
                    solved.add(r["id"])
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "practice_select128.jsonl"), encoding="utf-8")]
    inf = [r for r in rows if r["id"] in solved]
    with open(os.path.join(ROOT, out), "w", encoding="utf-8") as f:
        for r in inf:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"informative problems: {len(inf)} of {len(rows)} (solved by at least one run) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-informative", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-init", type=int, default=8)
    ap.add_argument("--n-bo", type=int, default=9)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--rows", default="data/practice_select128.jsonl")
    ap.add_argument("--valid-rows", default="data/practice_select_rest128.jsonl")
    ap.add_argument("--dir", default="results/qwen_dir")
    ap.add_argument("--tag-prefix", default="bo")
    a = ap.parse_args()

    if a.build_informative:
        build_informative(a.rows)
        return
    if a.synthetic:                      # offline check: a bump around alpha=9, k=128 plus Bernoulli noise
        rng = np.random.default_rng(0)

        def f(al, k):
            return 0.12 + 0.2 * math.exp(-((math.log(al) - math.log(9)) ** 2) / 0.3 - ((math.log2(k) - 7) ** 2) / 1.5)
        state = []
        for al, k in init_design(a.n_init):
            p = f(al, k)
            ok = rng.binomial(162, p)
            state.append({"alpha": al, "k": k, "ok": int(ok), "n_seq": 162})
        for _ in range(a.n_bo):
            (al, k), info = next_by_ei(state)
            ok = rng.binomial(162, f(al, k))
            state.append({"alpha": al, "k": k, "ok": int(ok), "n_seq": 162})
            print(f"EI pick alpha={al:g} k={k} ei={info['ei']:.4f} mu={info['mu']:.3f} -> acc {ok / 162:.3f} (true {f(al, k):.3f})")
        X, y, nv, hp = posterior_table(state)
        mu, _ = predict(X, y, nv, X, hp)
        top = np.argsort(-mu)[:3]
        print("top-3 by posterior mean:", [(state[i]["alpha"], state[i]["k"], round(float(mu[i]), 3)) for i in top], "true optimum ~ (9, 128)")
        return

    assert a.run
    state = load_state()
    total = a.n_init + a.n_bo
    design = init_design(a.n_init)
    while len(state) < total:
        if len(state) < a.n_init:
            (al, k), info = design[len(state)], {"phase": "init"}
        else:
            (al, k), info = next_by_ei(state)
            info["phase"] = "ei"
        print(f"[{len(state) + 1}/{total}] alpha={al:g} k={k} {info}", flush=True)
        r = evaluate(al, k, a.rows, a.samples, 0, a.dir, a.tag_prefix)
        r["info"] = info
        state.append(r)
        save_state(state)
        print(f"   -> acc {r['acc']:.3f} ({r['ok']}/{r['n_seq']}), loops {r['loops']}, closed {r['closed']}", flush=True)
    X, y, nv, hp = posterior_table(state)
    mu, sd = predict(X, y, nv, X, hp)
    seen, top = set(), []
    for i in np.argsort(-mu):
        key = (state[i]["alpha"], state[i]["k"])
        if key not in seen:
            seen.add(key)
            top.append((key, float(mu[i])))
        if len(top) == 3:
            break
    final = {"top3_by_posterior_mean": top, "validation": []}
    for (al, k), m in top:
        r = evaluate(al, k, a.valid_rows, 2, 1, a.dir, a.tag_prefix + "_val")
        final["validation"].append({"alpha": al, "k": k, "posterior_mean": m, "val_acc": r["acc"], "val_ok": r["ok"], "val_n": r["n_seq"]})
        print(f"validation alpha={al:g} k={k}: {r['acc']:.3f} ({r['ok']}/{r['n_seq']})", flush=True)
    best = max(final["validation"], key=lambda v: (v["val_acc"], v["posterior_mean"]))
    final["chosen"] = best
    json.dump(final, open(os.path.join(ROOT, "results", "bo", "final.json"), "w"), indent=1)
    print("CHOSEN", json.dumps(best))


if __name__ == "__main__":
    main()
