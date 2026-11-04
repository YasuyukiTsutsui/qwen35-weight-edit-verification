"""Unit tests for scripts/final_analysis.py (synthetic data only; CPU; writes only under results/analysis_test/).

    python -I results/analysis_test/test_final_analysis.py [--reps 200] [--B 1000]
"""
import argparse
import contextlib
import gzip
import io
import json
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import final_analysis as fa  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(("PASS" if ok else "FAIL"), name, detail, flush=True)


# ---------------------------------------------------------------- 1. Holm
def holm_reference(p):
    """Independent textbook version: adj_(i) = max_{j<=i} min(1, (m-j+1) p_(j))."""
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    for rank, i in enumerate(order):
        adj[i] = max(min(1.0, (m - j) * p[order[j]]) for j in range(rank + 1))
    return adj


def test_holm():
    got = fa.holm([0.01, 0.04, 0.03, 0.005])
    # hand calculation: sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04 -> running max 0.02, 0.03, 0.06, 0.06
    check("holm hand example 1", np.allclose(got, [0.03, 0.06, 0.06, 0.02]), str(got))
    got = fa.holm([0.001, 0.2, 0.04])  # 3*0.001, 2*0.04, 1*0.2
    check("holm hand example 2", np.allclose(got, [0.003, 0.2, 0.08]), str(got))
    got = fa.holm([0.5, 0.6])
    check("holm cap at 1", np.allclose(got, [1.0, 1.0]), str(got))
    got = fa.holm([0.02])
    check("holm m=1 unchanged", np.allclose(got, [0.02]))
    got = fa.holm([0.01, 0.01, 0.01])
    check("holm ties", np.allclose(got, [0.03, 0.03, 0.03]), str(got))
    rng = np.random.default_rng(1)
    ok = True
    for _ in range(200):
        p = rng.random(int(rng.integers(1, 9))) ** 3
        ok &= np.allclose(fa.holm(p), holm_reference(list(p)))
    check("holm == independent reference on 200 random vectors", ok)


# ---------------------------------------------------------------- 2. bootstrap
def sim_problems(rng, N, delta_mode, spp):
    """Per-problem success probabilities (base, cand) and number of samples per problem."""
    q = rng.beta(1.2, 4.0, N)  # skewed, like a hard benchmark
    pb = q
    pc = q if delta_mode == "null" else np.clip(q * 1.6 + 0.08 * (1 - q), 0, 1)
    n = np.full(N, spp) if spp != "mixed" else rng.integers(1, 4, N)
    return pb, pc, n


def true_diff(delta_mode, spp):
    rng = np.random.default_rng(999)
    pb, pc, n = sim_problems(rng, 2_000_000, delta_mode, spp)
    w = n.astype(float)
    return float(((pc - pb) * w).sum() / w.sum())


def coverage_scenario(name, delta_mode, spp, reps, B, N=256):
    truth = true_diff(delta_mode, spp)
    rng = np.random.default_rng(12345)
    cover = rej = 0
    for r in range(reps):
        pb, pc, n = sim_problems(rng, N, delta_mode, spp)
        cb = rng.binomial(n, pb).astype(float)
        cc = rng.binomial(n, pc).astype(float)
        nn = n.astype(float)[None, :]
        boot = fa.cluster_bootstrap({"b": cb[None, :], "c": cc[None, :]}, {"b": nn, "c": nn}, B, seed=1000 + r)
        d_hat = cc.sum() / nn.sum() - cb.sum() / nn.sum()
        db = boot["c"][:, 0] - boot["b"][:, 0]
        lo, hi = fa.pct_ci(db)
        cover += lo <= truth <= hi
        p, _ = fa.boot_p_values(d_hat, db)
        rej += p < 0.05
    return truth, cover / reps, rej / reps


def test_bootstrap(reps, B):
    for name, mode, spp in [("1 sample/problem, effect", "eff", 1), ("mixed 1-3 samples/problem, effect", "eff", "mixed"),
                            ("2 samples/problem, null", "null", 2)]:
        t0 = time.time()
        truth, cov, rej = coverage_scenario(name, mode, spp, reps, B)
        if mode == "null":
            check(f"type-I error of p<0.05 ({name})", 0.01 <= rej <= 0.09, f"rate={rej:.3f} (reps={reps}, B={B}, {time.time() - t0:.0f}s)")
            check(f"95% CI coverage of 0 under null ({name})", 0.91 <= cov <= 0.99, f"coverage={cov:.3f}")
        else:
            check(f"95% CI coverage ({name})", 0.91 <= cov <= 0.99, f"true diff={truth:+.4f} coverage={cov:.3f} (reps={reps}, B={B}, {time.time() - t0:.0f}s)")
            check(f"power sanity: p<0.05 in most reps ({name})", rej > 0.8, f"rejection rate={rej:.3f}")
    # chunk invariance and seed determinism
    rng = np.random.default_rng(5)
    num = {"a": rng.binomial(2, 0.3, (3, 50)).astype(float)}
    den = {"a": np.full((3, 50), 2.0)}
    x1 = fa.cluster_bootstrap(num, den, 600, seed=7, chunk=600)["a"]
    x2 = fa.cluster_bootstrap(num, den, 600, seed=7, chunk=100)["a"]
    x3 = fa.cluster_bootstrap(num, den, 600, seed=8, chunk=600)["a"]
    check("bootstrap independent of chunk size", np.array_equal(x1, x2))
    check("bootstrap: same seed -> same, other seed -> different", np.array_equal(x1, x2) and not np.array_equal(x1, x3))
    # cluster structure: a problem is resampled as a whole (replicate mean equals weighted mean of problem totals)
    est = x1[:, 0]
    check("bootstrap mean ~ point estimate", abs(np.nanmean(est) - num["a"][0].sum() / den["a"][0].sum()) < 0.01)
    # p-value edge cases
    p, prc = fa.boot_p_values(0.0, np.zeros(99))
    check("p-value = 1 when the difference is exactly 0", p == 1.0 and prc == 1.0, f"{p}, {prc}")
    p, _ = fa.boot_p_values(0.2, np.full(9999, 0.2))
    check("p-value floor = 2/(B+1) when 0 is never reached", abs(p - 2 / 10000) < 1e-12, f"{p}")


# ---------------------------------------------------------------- 3. end-to-end on synthetic samples
def sample(i, k, correct, finish, tokens, closed, answered=True, loop_text=None, cat=None):
    if loop_text is not None:
        text = loop_text
    elif closed:
        text = "reasoning words </think> so the answer is " + ("\\boxed{1}" if answered else "unknown")
    else:
        text = "reasoning words that never end"
    vb = cat == "verbatim"
    rp = cat in ("verbatim", "other_rep")
    return {"id": f"p{i}", "k": k, "answer": "1", "pred": "1" if (closed and answered) else None, "correct": correct,
            "finish": finish, "tokens": tokens, "verbatim_loop_end": vb, "repetitive_end": rp, "text": text}


def write_run(root, tag, samples, max_tokens=16384):
    d = os.path.join(root, tag, "synth")
    os.makedirs(d, exist_ok=True)
    with gzip.open(os.path.join(d, "samples.jsonl.gz"), "wt", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    json.dump({"tag": tag, "max_tokens": max_tokens, "preset": "qwen35_think", "thinking": 1, "gen_seconds": 1},
              open(os.path.join(d, "summary.json"), "w"))


def run_main(root, base, cands, out, extra=()):
    argv = ["--base", base, "--cands", *cands, "--set", "synth", "--out", out, "--results-root", root, "--boot", "400",
            "--min-free-gb", "0", *extra]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = fa.main(argv)
    return res, buf.getvalue()


def test_end_to_end():
    root = os.path.join(HERE, "synthetic_root")
    outdir = os.path.join(HERE, "synthetic_out")
    # base: 10 problems, 1 sample each; correct on p0, p1, p2 (p2 is correct but the thinking never closed and it was cut)
    base = [sample(0, 0, True, "stop", 3000, True), sample(1, 0, True, "stop", 6000, True),
            sample(2, 0, True, "length", 16384, False, cat="none")]
    base += [sample(i, 0, False, "length", 16384, False, cat="verbatim") for i in range(3, 8)]
    base += [sample(8, 0, False, "stop", 5000, True), sample(9, 0, False, "stop", 9000, True, answered=False)]
    # cand1: 2 samples on p0-p4, 1 sample on p5-p9 (unequal counts)
    c1 = [sample(0, 0, True, "stop", 3500, True), sample(0, 1, True, "stop", 12000, True),
          sample(1, 0, True, "stop", 4000, True), sample(1, 1, False, "length", 16384, True, answered=True, cat="other_rep"),
          sample(2, 0, True, "stop", 7000, True), sample(2, 1, True, "stop", 15000, True),
          sample(3, 0, False, "stop", 2000, True), sample(3, 1, True, "stop", 2500, True),
          sample(4, 0, False, "length", 16384, False, cat="none"), sample(4, 1, False, "length", 16384, False, cat="verbatim")]
    c1 += [sample(i, 0, True, "stop", 4000 + 1000 * i, True) for i in range(5, 9)] + [sample(9, 0, False, "stop", 1000, True)]
    # cand2: identical to base but every sample duplicated (k=0,1) -> must give the same estimate and bootstrap as the base
    c2 = [dict(s) for s in base] + [dict(s, k=1) for s in base]
    # cand3: identical to the base (1 sample/problem)
    c3 = [dict(s) for s in base]
    for tag, smp in [("base", base), ("c1", c1), ("c2dup", c2), ("c3same", c3)]:
        write_run(root, tag, smp)
    res, md = run_main(root, "base", ["c1", "c2dup", "c3same"], outdir, ["--no-loops"])
    M = {t: res["runs"][t]["metrics"] for t in res["runs"]}
    C = {t: res["runs"][t]["counts"] for t in res["runs"]}

    def hand(smp, B=None):
        n = len(smp)
        return {"acc": sum(s["correct"] for s in smp) / n,
                "b": sum(s["correct"] and s["finish"] == "stop" and s["tokens"] <= B for s in smp) / n if B else None,
                "closed": sum("</think>" in s["text"] for s in smp) / n,
                "agc": sum(s["correct"] and "</think>" in s["text"] for s in smp) / max(1, sum("</think>" in s["text"] for s in smp))}

    h = hand(base)
    check("base accuracy (hand = 3/10)", abs(M["base"]["acc"]["est"] - 0.3) < 1e-12 and h["acc"] == 0.3)
    check("base budget 4096 (only p0: 1/10)", abs(M["base"]["budget_4096"]["est"] - hand(base, 4096)["b"]) < 1e-12 and hand(base, 4096)["b"] == 0.1)
    check("base budget 8192 (p0, p1: 2/10)", abs(M["base"]["budget_8192"]["est"] - 0.2) < 1e-12)
    check("base budget 16384 (2/10: the cut-but-correct sample is excluded)", abs(M["base"]["budget_16384"]["est"] - 0.2) < 1e-12)
    check("base counts: correct but cut / not closed", C["base"]["n_correct_but_cut"] == 1 and C["base"]["n_correct_not_closed"] == 1)
    check("base closed share = 4/10, acc|closed = 2/4", abs(M["base"]["closed"]["est"] - 0.4) < 1e-12 and abs(M["base"]["acc_given_closed"]["est"] - 0.5) < 1e-12)
    check("base wrong-after-closing = 2 (1 answered wrong, 1 no boxed)", C["base"]["n_closed_wrong"] == 2 and C["base"]["closed_wrong_answered"] == 1
          and C["base"]["closed_wrong_no_boxed"] == 1 and C["base"]["closed_wrong_cut_in_answer"] == 0)
    check("base cut breakdown: 6 cut = 5 verbatim + 0 other + 1 none", (C["base"]["n_cut"], C["base"]["cut_verbatim_loop"],
          C["base"]["cut_other_repetition"], C["base"]["cut_no_repetition"]) == (6, 5, 0, 1), str(C["base"]))
    # cand1: pooled ratio with unequal samples per problem: correct = 2+1+2+1+0+4+0 -> count
    n_c1 = len(c1)
    ok_c1 = sum(s["correct"] for s in c1)
    check("cand1 pooled accuracy = correct/samples (15 samples)", n_c1 == 15 and abs(M["c1"]["acc"]["est"] - ok_c1 / n_c1) < 1e-12, f"{ok_c1}/{n_c1}")
    check("cand1 budget 16384 hand", abs(M["c1"]["budget_16384"]["est"] - hand(c1, 16384)["b"]) < 1e-12)
    check("cand1 budget 4096 hand", abs(M["c1"]["budget_4096"]["est"] - hand(c1, 4096)["b"]) < 1e-12)
    check("cand1 closed-wrong with cut in the answer (p1 k1) counted", C["c1"]["closed_wrong_cut_in_answer"] == 1)
    check("duplicated base == base (estimates)", all(abs(M["c2dup"][k]["est"] - M["base"][k]["est"]) < 1e-12 for k in M["base"] if M["base"][k]))
    check("duplicated base == base (bootstrap CI identical)", all(M["c2dup"][k]["ci95"] == M["base"][k]["ci95"] for k in M["base"] if M["base"][k]))
    f1 = {r["a"]: r for r in res["family1_primary"]}
    check("identical run: diff 0, CI [0,0], p=1, holm=1", f1["c3same"]["diff"] == 0 and f1["c3same"]["ci95"] == [0.0, 0.0] and f1["c3same"]["p_boot"] == 1.0)
    check("duplicated run: diff 0 and p=1", f1["c2dup"]["diff"] == 0 and f1["c2dup"]["p_boot"] == 1.0)
    d_hand = ok_c1 / n_c1 - 0.3
    check("cand1 - base difference matches hand value", abs(f1["c1"]["diff"] - d_hand) < 1e-12, f"{f1['c1']['diff']:+.4f}")
    # per-problem comparison: cand1 per-problem acc vs base
    pc1 = {}
    for s in c1:
        pc1.setdefault(s["id"], []).append(s["correct"])
    pc1 = {k: np.mean(v) for k, v in pc1.items()}
    pb = {s["id"]: float(s["correct"]) for s in base}
    better = sum(pc1[i] > pb[i] for i in pb)
    worse = sum(pc1[i] < pb[i] for i in pb)
    check("cand1 problems better / worse than base", (f1["c1"]["problems_a_better"], f1["c1"]["problems_b_better"]) == (better, worse), f"{better}/{worse}")
    ps = [r["p_boot"] for r in res["family1_primary"]]
    check("family-1 Holm equals reference Holm of the family's p-values", np.allclose([r["p_holm"] for r in res["family1_primary"]], holm_reference(ps)))
    check("CI contains the point estimate", all(r["ci95"][0] <= r["diff"] <= r["ci95"][1] for r in res["family1_primary"]))
    # secondary family is corrected separately
    res2, _ = run_main(root, "base", ["c1", "c2dup", "c3same"], outdir + "2", ["--no-loops", "--secondary", "c1:c2dup", "c1:c3same", "c2dup:c3same"])
    ps2 = [r["p_boot"] for r in res2["family2_secondary"]]
    check("family-2 Holm uses only family-2 p-values", np.allclose([r["p_holm"] for r in res2["family2_secondary"]], holm_reference(ps2)) and res2["meta"]["family2_m"] == 3)
    check("family-1 results unchanged by adding family 2", [r["p_holm"] for r in res2["family1_primary"]] == [r["p_holm"] for r in res["family1_primary"]])
    res3, _ = run_main(root, "base", ["c1", "c2dup", "c3same"], outdir + "3", ["--no-loops", "--all-pairs"])
    check("--all-pairs gives C(3,2)=3 comparisons", res3["meta"]["family2_m"] == 3)
    # determinism: same arguments -> identical numbers
    res4, _ = run_main(root, "base", ["c1", "c2dup", "c3same"], outdir + "4", ["--no-loops"])
    check("deterministic with the fixed seed", res4["family1_primary"] == res["family1_primary"])
    # problem set mismatch
    write_run(root, "c_missing", [s for s in base if s["id"] != "p9"])
    try:
        run_main(root, "base", ["c_missing"], outdir + "5", ["--no-loops"])
        check("different problem sets -> error", False)
    except SystemExit as e:
        check("different problem sets -> error", "problem sets differ" in str(e), str(e)[:60])
    res6, _ = run_main(root, "base", ["c_missing"], outdir + "6", ["--no-loops", "--allow-intersection"])
    check("--allow-intersection analyses the common 9 problems and warns", res6["meta"]["n_problems"] == 9 and res6["meta"]["warnings"])
    # budget above the run's own cap is not reported
    write_run(root, "capped", base, max_tokens=8192)
    res7, _ = run_main(root, "base", ["capped"], outdir + "7", ["--no-loops"])
    check("budget above the run's max_tokens is null", res7["runs"]["capped"]["metrics"]["budget_16384"] is None and res7["runs"]["capped"]["metrics"]["budget_8192"] is not None
          and any("max_tokens" in w for w in res7["meta"]["warnings"]))
    check("markdown rendered with all sections", all(f"## {i}." in md for i in range(1, 8)))


def test_loops():
    """End-to-end find_loop: a text that repeats a sentence 300 times must be detected with the right period bin."""
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(ROOT, "models", "Qwen3.5-0.8B", "tokenizer.json"))
    sent = "The quick brown fox jumps over the lazy dog near the old river bank today. "
    period = len(tk.encode(sent * 3, add_special_tokens=False).ids) - len(tk.encode(sent * 2, add_special_tokens=False).ids)
    lt = "Here is my thinking about the problem step by step. " * 5 + sent * 300
    root = os.path.join(HERE, "synthetic_root")
    base = [sample(0, 0, False, "length", 16384, False, loop_text=lt, cat="verbatim"), sample(1, 0, True, "stop", 3000, True),
            sample(2, 0, False, "length", 16384, False, cat="none")]
    write_run(root, "loop_base", base)
    write_run(root, "loop_c", [sample(0, 0, True, "stop", 3000, True), sample(1, 0, True, "stop", 3000, True), sample(2, 0, True, "stop", 3000, True)])
    res, _ = run_main(root, "loop_base", ["loop_c"], os.path.join(HERE, "synthetic_out_loops"))
    lp = res["runs"]["loop_base"]["loops"]
    lab = fa.period_bin(period)
    check("find_loop detects the repeated sentence in the right period bin", lp["by_period_bin"][lab] == 1 and lp["n_detected"] == 1 and lp["by_period_bin"]["undetected"] == 1,
          f"expected period {period} -> bin {lab}; got {lp['by_period_bin']}; median period {lp.get('period_median')}")
    check("detected period equals the tokenised sentence length", lp.get("period_median") == period, f"{lp.get('period_median')} vs {period}")
    check("candidate without cut samples has an empty loop table", res["runs"]["loop_c"]["loops"]["n_cut"] == 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    t0 = time.time()
    test_holm()
    test_end_to_end()
    test_loops()
    test_bootstrap(a.reps, a.B)
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} passed in {time.time() - t0:.0f}s; peak working set {fa.peak_mem_mb()} MB")
    sys.exit(1 if bad else 0)
