"""「思考を閉じたのに誤答」の標本の分析（保存済みの標本のみを使う。GPU・モデル・Docker は使わない）。

使い方（1 つずつ、低優先度で実行する）:
  python -I scripts/analyze_closed_wrong.py extract   # 編集ごとに標本から特徴量を取り出す（標本のテキストは保持しない）
  python -I scripts/analyze_closed_wrong.py report    # 特徴量から表を作る（results/analysis_cbw/tables.json, t_*.md）
  python -I scripts/analyze_closed_wrong.py assemble  # results/analysis_cbw/report_template.md の {{T_xxx}} を表で置換し、
                                                      # reports/analysis_closed_wrong.md を書く
出力は results/analysis_cbw/ と reports/analysis_closed_wrong.md のみ。入力は読み取り専用。

資源の制約: 空きメモリが 6 GB を下回る間は待つ。プロセスは最低優先度で動かし、最大使用量（ピーク）を記録する。
"""
import ctypes
import gc
import gzip
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True   # scripts/__pycache__ を変更しない
sys.path.insert(0, str(ROOT / "scripts"))
from grade import loop_signals, split_think, think_closed  # noqa: E402

OUT = ROOT / "results" / "analysis_cbw"
SET = "practice_select128"
MAXTOK = 16384
MIN_FREE_GB = 6.0

# 順序は編集の強さ（思考を閉じた割合）の目安ではなく、固定の並び。表では閉じた割合の昇順に並べ替える。
FIXED_TAGS = [
    ("rb16k_base", "基底"),
    ("rb16k_p50c", "公開編集"),
    ("rb16k_own_k64_a6", "旧方向 α6"),
    ("rb16k_own_k64_a12", "旧方向 α12"),
    ("scr_t1_L10_r12", "対比 L10"),
    ("scr_t1_L14_r12", "対比 L14"),
    ("scr_t1_L10_14_r12", "対比 L10+14"),
    ("scr_t1_mp_L15_19_r12", "対比 L15+19"),
]


# ---------------------------------------------------------------- 資源の管理（Windows）
class _MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def free_gb():
    try:
        m = _MEMSTAT()
        m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 2 ** 30
    except Exception:  # noqa: BLE001
        return 99.0


def peak_mb():
    try:
        p = _PMC()
        p.cb = ctypes.sizeof(p)
        k32, ps = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        ps.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), ctypes.c_ulong]
        ps.GetProcessMemoryInfo.restype = ctypes.c_int
        ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(p), p.cb)
        return p.PeakWorkingSetSize / 2 ** 20
    except Exception:  # noqa: BLE001
        return -1.0


def lower_priority():
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)  # IDLE
    except Exception:  # noqa: BLE001
        pass


def wait_for_memory():
    while free_gb() < MIN_FREE_GB:
        print(f"[wait] free memory {free_gb():.1f} GB < {MIN_FREE_GB} GB; sleeping 30 s", flush=True)
        time.sleep(30)


# ---------------------------------------------------------------- 特徴量
MARK = {
    "wait": re.compile(r"\bWait\b"),
    "verify": re.compile(r"(?i)\b(?:check\w*|verif\w*|double-?check\w*|sanity)"),
    "reread": re.compile(r"(?i)\b(?:re-?read\w*|re-?evaluat\w*|re-?calculat\w*|re-?examin\w*|re-?check\w*|"
                         r"let me re\w*|let's re\w*)"),
    "hmm": re.compile(r"\b(?:Actually|Hmm|Alternatively|However)\b"),
    "selfcorr": re.compile(r"(?i)self-?correct"),
}
FINAL_RE = re.compile(r"(?i)final (?:answer|output|response|polish|review|check|decision|verification|"
                      r"calculation|result|step)|ready to (?:write|generate|output|respond)|construct final|"
                      r"looks good|matches the thought")
HDR1 = re.compile(r"^ {0,3}(\d{1,2})\.\s+\*\*([^*\n]{1,80})\*\*", re.M)      # "1.  **Analyze the Request:**"
HDR2 = re.compile(r"^ {0,3}\*\*(\d{1,2})\.\s+([^*\n]{1,80})\*\*", re.M)      # "**1. Analyze the inequalities:**"
FINAL_TITLE = re.compile(r"(?i)final|output|draft|response|polish|verif|review|check|construct|conclusion|answer|format")
BOX = re.compile(r"\\(?:boxed|fbox)\s*\{")


def last_boxed_span(text):
    """(content, end_index_after_closing_brace) of the last \\boxed{...}; None when absent/unbalanced."""
    ms = list(BOX.finditer(text))
    if not ms:
        return None
    i = ms[-1].end()
    depth, j = 1, i
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return (text[i:j - 1].strip(), j) if depth == 0 else None


def norm_ans(s):
    s = s.strip()
    s = re.sub(r"\\text\s*\{[^{}]*\}", "", s)
    for a in ("\\left", "\\right", "\\,", "\\!", "\\;", "\\ ", "$", "\\$", "\\%", "%", "^\\circ", "^{\\circ}", "\\displaystyle"):
        s = s.replace(a, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = re.sub(r"\s+", "", s).rstrip(".")
    return s


def loose_equal(pred, gold):
    a, b = norm_ans(pred), norm_ans(gold)
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-9 * max(1.0, abs(float(b)))
    except ValueError:
        return False


def brace_glitch(seg):
    """行ごとの { と } の数が合わない行の割合（編集後の「打ち間違い」の粗い目安）。"""
    seg = seg.replace("\\{", "").replace("\\}", "")
    lines = [ln for ln in seg.split("\n") if ln.strip()]
    if not lines:
        return 0.0, 0
    bad = sum(ln.count("{") != ln.count("}") for ln in lines)
    return bad / len(lines), len(lines)


def count_marks(seg):
    return {k: len(r.findall(seg)) for k, r in MARK.items()}


def step_info(th):
    hs = [(m.start(), int(m.group(1)), m.group(2)) for m in HDR1.finditer(th)]
    hs += [(m.start(), int(m.group(1)), m.group(2)) for m in HDR2.finditer(th)]
    hs.sort()
    if not hs:
        return {"n_hdr": 0, "max_step": 0, "last_final": None}
    return {"n_hdr": len(hs), "max_step": max(h[1] for h in hs), "last_final": bool(FINAL_TITLE.search(hs[-1][2]))}


def trail_kind(trail):
    if not trail.strip() and len(trail) <= 3:
        return "none"
    if not trail.strip():
        return "ws_long" if len(trail) > 20 else "ws_short"
    if len(trail.strip()) >= 20:
        return "text"
    return "short"


def max_ws_run(s):
    m = re.findall(r"\s+", s)
    return max((len(x) for x in m), default=0)


def features(row, prob, tok):
    text = row["text"]
    closed = think_closed(text)
    if closed:
        th, an = split_think(text)
        think_tok = len(tok.encode(th, add_special_tokens=False).ids)
        # 総数 = 思考 + </think> 1 + 答え + EOS 1（stop のとき）。8 標本で残差 2 を確認済み
        ans_tok = max(0, row["tokens"] - think_tok - (2 if row["finish"] == "stop" else 1))
    else:
        th, an = text, ""
        think_tok, ans_tok = row["tokens"], 0                # 未閉鎖は全体が思考（採点は全体から \boxed を探す）
    f = {
        "id": row["id"], "k": row["k"], "finish": row["finish"], "correct": bool(row["correct"]),
        "closed": closed, "tokens": row["tokens"], "think_tok": think_tok, "ans_tok": ans_tok,
        "pred": row["pred"], "gold": row["answer"],
        "src": "gsm8k" if row["id"].startswith("gsm8k") else "math",
        "end_verbatim": bool(row["verbatim_loop_end"]), "end_repetitive": bool(row["repetitive_end"]),
    }
    for k, v in count_marks(th).items():
        f["th_" + k] = v
    f["th_boxed"] = len(BOX.findall(th)) if closed else None
    f["th_chars"] = len(th)
    gl, nl = brace_glitch(th)
    f["th_glitch"], f["th_lines"] = gl, nl
    if closed:
        ls = loop_signals(th)
        f["th_tail_loop"] = bool(ls["verbatim_loop_end"] or ls["repetitive_end"])
        f["th_concluded"] = bool(FINAL_RE.search(th[-1000:]))
        f.update({"th_" + k: v for k, v in step_info(th).items()})
        for k, v in count_marks(an).items():
            f["an_" + k] = v
        f["an_chars"] = len(an)
        f["an_boxed"] = len(BOX.findall(an))
        gl, nl = brace_glitch(an)
        f["an_glitch"], f["an_lines"] = gl, nl
        lb = last_boxed_span(an)
        if lb is None:
            f["trail_kind"] = None
            f["trail_len"] = None
        else:
            tr = an[lb[1]:]
            f["trail_kind"] = trail_kind(tr)
            f["trail_len"] = len(tr)
            f["trail_ws_run"] = max_ws_run(tr)
            f["trail_nonascii"] = (sum(ord(c) > 127 for c in tr) / len(tr)) if tr else 0.0
        # 答えの分類（closed 内）
        if row["pred"] is None:
            f["ans_type"] = "nobox_cut" if row["finish"] == "length" else "nobox_stop"
        elif row["correct"]:
            f["ans_type"] = "correct"
        elif loose_equal(row["pred"], row["answer"]):
            f["ans_type"] = "format_miss"
        else:
            f["ans_type"] = "wrong_value"
    return f


def tag_list():
    tags = list(FIXED_TAGS)
    st = ROOT / "results" / "bo" / "state.json"
    try:
        for p in json.loads(st.read_text(encoding="utf-8")):
            t = p.get("tag")
            if t and (ROOT / "results" / t / SET / "summary.json").exists() and \
                    (ROOT / "results" / t / SET / "samples.jsonl.gz").exists():
                tags.append((t, f"探索 k{p['k']} α{p['alpha']:g}"))
    except Exception as e:  # noqa: BLE001
        print("state.json not readable:", e)
    return [(t, l) for t, l in tags if (ROOT / "results" / t / SET / "samples.jsonl.gz").exists()]


def cmd_extract(force=False):
    from tokenizers import Tokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    probs = {}
    for ln in (ROOT / "data" / f"{SET}.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            d = json.loads(ln)
            probs[d["id"]] = d
    tags = tag_list()
    (OUT / "tags.json").write_text(json.dumps(tags, ensure_ascii=False, indent=1), encoding="utf-8")
    tok = Tokenizer.from_file(str(ROOT / "models" / "Qwen3.5-0.8B" / "tokenizer.json"))
    for tag, label in tags:
        fo = OUT / f"features_{tag}.json"
        if fo.exists() and not force:
            print("skip", tag)
            continue
        wait_for_memory()
        t0 = time.time()
        rows = []
        with gzip.open(ROOT / "results" / tag / SET / "samples.jsonl.gz", "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                rows.append(features(r, probs.get(r["id"]), tok))
                del r
        fo.write_text(json.dumps(rows), encoding="utf-8")
        del rows
        gc.collect()
        print(f"{tag}: done in {time.time() - t0:.0f}s, peak working set {peak_mb():.0f} MB, free {free_gb():.1f} GB",
              flush=True)


# ---------------------------------------------------------------- 集計
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def pct(k, n, ci=True):
    if n == 0:
        return "–"
    s = f"{100 * k / n:.1f}%"
    if ci:
        lo, hi = wilson(k, n)
        s += f" [{100 * lo:.0f}–{100 * hi:.0f}]"
    return s


def pctn(k, n):
    return f"{pct(k, n)} ({k}/{n})" if n else "–"


def fmt_q(v, nd=0):
    import numpy as np
    if len(v) == 0:
        return "–"
    q = np.percentile(v, [25, 50, 75])
    return f"{q[1]:.{nd}f} ({q[0]:.{nd}f}–{q[2]:.{nd}f})"


def mdtable(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def auc(pos, neg):
    import numpy as np
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    p, n = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (len(pos) * len(neg)))


def cluster_boot(rows, stat, B=1000, seed=0):
    """問題単位のブートストラップ。rows: 辞書のリスト（'id' を持つ）。stat(rows) -> float。95% 区間を返す。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["id"]].append(r)
    ids = list(by)
    vals = []
    for _ in range(B):
        samp = rng.integers(0, len(ids), len(ids))
        rs = [x for i in samp for x in by[ids[i]]]
        v = stat(rs)
        if v == v:
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def fe_logit_coefs(rows, extras=(), lam=0.5, iters=25):
    """closed 標本のロジスティック回帰: correct ~ log2(思考長) + extras + 編集の固定効果 + 問題の固定効果（リッジ λ）。
    先頭 1+len(extras) 個の係数（log2 思考長, extras...）を返す。"""
    import numpy as np
    ne = 1 + len(extras)
    if len(rows) < 30:
        return [float("nan")] * ne
    pid = {p: i for i, p in enumerate(sorted({r["id"] for r in rows}))}
    tg = {t: i for i, t in enumerate(sorted({r["tag"] for r in rows}))}
    n, P, T = len(rows), len(pid), len(tg)
    X = np.zeros((n, ne + (T - 1) + P))
    y = np.array([1.0 if r["correct"] else 0.0 for r in rows])
    ls = np.array([math.log2(max(r["think_tok"], 1)) for r in rows])
    X[:, 0] = ls - ls.mean()
    for j, ex in enumerate(extras):
        v = np.array([float(r[ex]) for r in rows])
        X[:, 1 + j] = v - v.mean()
    for i, r in enumerate(rows):
        t = tg[r["tag"]]
        if t > 0:
            X[i, ne + t - 1] = 1
        X[i, ne + T - 1 + pid[r["id"]]] = 1
    pen = np.full(X.shape[1], lam)
    pen[:ne] = 0.0
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        pr = 1 / (1 + np.exp(-z))
        g = X.T @ (y - pr) - pen * w
        H = (X * (pr * (1 - pr))[:, None]).T @ X + np.diag(pen) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-6:
            break
    return [float(v) for v in w[:ne]]


def fe_logit_coef(rows, tags_order=None, lam=0.5, iters=25):
    return fe_logit_coefs(rows, (), lam, iters)[0]


def load_features():
    tags = json.loads((OUT / "tags.json").read_text(encoding="utf-8"))
    F = {}
    for tag, label in tags:
        fo = OUT / f"features_{tag}.json"
        if fo.exists():
            F[tag] = json.loads(fo.read_text(encoding="utf-8"))
            for r in F[tag]:
                r["tag"] = tag
    return [(t, l) for t, l in tags if t in F], F


def cmd_report():
    import numpy as np
    from scipy.stats import spearmanr
    tags, F = load_features()
    label = dict(tags)
    # 閉じた割合の昇順に並べる（編集の強さの目安）
    def cshare(t):
        return sum(r["closed"] for r in F[t]) / len(F[t])
    order = sorted([t for t, _ in tags], key=cshare)
    group = {t: ("弱" if cshare(t) < 0.5 else "中" if cshare(t) < 0.75 else "強") for t in order}
    T = {}   # 表の名前 -> markdown
    J = {"order": order, "group": group, "label": label}

    allrows = [r for t in order for r in F[t]]
    base = "rb16k_base"
    # --- 問題ごとの基準
    by_prob_tag = defaultdict(lambda: defaultdict(list))
    for r in allrows:
        by_prob_tag[r["id"]][r["tag"]].append(r)
    pids = sorted(by_prob_tag)

    def other_pass(pid, tag):
        rs = [r for t, v in by_prob_tag[pid].items() if t != tag for r in v]
        return (sum(r["correct"] for r in rs) / len(rs)) if rs else float("nan"), len(rs)

    base_correct = {pid: any(r["correct"] for r in by_prob_tag[pid].get(base, [])) for pid in pids}
    base_closed_correct = {pid: [r for r in by_prob_tag[pid].get(base, []) if r["closed"] and r["correct"]] for pid in pids}
    # 参照思考長: その問題で、他の編集の closed かつ正解の標本の思考長の中央値
    def ref_len(pid, tag):
        v = [r["think_tok"] for t, rr in by_prob_tag[pid].items() if t != tag for r in rr if r["closed"] and r["correct"]]
        return (float(np.median(v)), len(v)) if v else (None, 0)

    # ---- 問題の分類（128 問）
    ever = {pid: sum(r["correct"] for v in by_prob_tag[pid].values() for r in v) for pid in pids}
    nall = {pid: sum(len(v) for v in by_prob_tag[pid].values()) for pid in pids}
    never = [pid for pid in pids if ever[pid] == 0]
    rate = {pid: ever[pid] / nall[pid] for pid in pids}
    nb_any = sum(any(r["correct"] for r in by_prob_tag[pid].get(base, [])) for pid in pids)
    rows = [["全標本（編集 10 本＋基底）で一度も正解がない問題", len(never)],
            ["全標本での正解率 <10% の問題（上の問題を含む）", sum(rate[p_] < 0.1 for p_ in pids)],
            ["全標本での正解率 10–50% の問題", sum(0.1 <= rate[p_] < 0.5 for p_ in pids)],
            ["全標本での正解率 ≥50% の問題", sum(rate[p_] >= 0.5 for p_ in pids)],
            ["基底（1 回/問）が正解した問題（未閉鎖での正解を含む）", nb_any],
            ["基底が closed で正解した問題", sum(bool(base_closed_correct[pid]) for pid in pids)]]
    T["problem_pool"] = mdtable(["区分", "問題数（全 128 問中）"], rows)
    J["problem_pool"] = {r[0]: r[1] for r in rows}

    # ---- T_overview
    rows = []
    for t in order:
        R = F[t]
        n = len(R)
        nc = sum(r["closed"] for r in R)
        cc = sum(r["closed"] and r["correct"] for r in R)
        cw = nc - cc
        us = sum((not r["closed"]) and r["finish"] == "stop" for r in R)
        ul = sum((not r["closed"]) and r["finish"] == "length" for r in R)
        uc = sum((not r["closed"]) and r["correct"] for r in R)
        rows.append([label[t], group[t], n, pct(sum(r["correct"] for r in R), n), pct(nc, n, False),
                     pct(cc, nc), f"{cw} ({100 * cw / n:.0f}%)", us, ul, uc])
        J.setdefault("overview", {})[t] = dict(n=n, closed=nc, closed_correct=cc, closed_wrong=cw,
                                               unclosed_stop=us, unclosed_len=ul, unclosed_correct=uc)
    T["overview"] = mdtable(["編集", "強さ群", "系列数", "正解率", "閉じた割合", "閉じた中の正解率 [95%区間]",
                             "閉じて誤答 (全体比)", "未閉鎖 stop", "未閉鎖 length", "未閉鎖で正解"], rows)

    # ---- 1a 思考長
    rows = []
    for t in order:
        R = [r for r in F[t] if r["closed"]]
        c = [r["think_tok"] for r in R if r["correct"]]
        w = [r["think_tok"] for r in R if not r["correct"]]
        refs = [(r["think_tok"], ref_len(r["id"], t)[0]) for r in R if not r["correct"] and ref_len(r["id"], t)[0]]
        ratio = [a / b for a, b in refs]
        refc = [(r["think_tok"], ref_len(r["id"], t)[0]) for r in R if r["correct"] and ref_len(r["id"], t)[0]]
        ratioc = [a / b for a, b in refc]
        nshort = sum(x < 1000 for x in w)
        rows.append([label[t], f"{len(c)}: {fmt_q(c)}", f"{len(w)}: {fmt_q(w)}",
                     pct(nshort, len(w), False) + f" ({nshort}/{len(w)})",
                     f"{len(ratio)}: {np.median(ratio):.2f}" if ratio else "–",
                     (pct(sum(x < 0.5 for x in ratio), len(ratio), False) + f" ({sum(x < 0.5 for x in ratio)})") if ratio else "–",
                     (pct(sum(x > 2 for x in ratio), len(ratio), False) + f" ({sum(x > 2 for x in ratio)})") if ratio else "–",
                     f"{len(ratioc)}: {np.median(ratioc):.2f}" if ratioc else "–"])
        J.setdefault("think_len", {})[t] = dict(correct=c, wrong=w, ratio_wrong=ratio, ratio_correct=ratioc)
    T["think_len"] = mdtable(["編集", "正解: 件数: 思考トークン 中央値 (四分位)", "誤答: 件数: 同左", "誤答のうち <1,000 tok",
                              "誤答 / 参照長 の中央値 (件数)", "誤答で比 <0.5", "誤答で比 >2", "正解 / 参照長 の中央値 (件数)"], rows)

    # ---- 1a' 基底が closed で正解した問題
    bsol = [pid for pid in pids if base_closed_correct[pid]]
    bsol_len = {pid: base_closed_correct[pid][0]["think_tok"] for pid in bsol}
    rows = []
    for t in order:
        if t == base:
            continue
        R = [r for pid in bsol for r in by_prob_tag[pid].get(t, [])]
        n = len(R)
        ok = sum(r["correct"] for r in R)
        cl = [r for r in R if r["closed"]]
        wr_cl = [r for r in cl if not r["correct"]]
        rt = [r["think_tok"] / bsol_len[r["id"]] for r in cl]
        rtw = [r["think_tok"] / bsol_len[r["id"]] for r in wr_cl]
        rows.append([label[t], n, pct(ok, n), len(cl), len(wr_cl),
                     f"{np.median(rt):.2f}" if rt else "–", f"{np.median(rtw):.2f}" if rtw else "–"])
        J.setdefault("base_solved", {})[t] = dict(n=n, ok=ok, closed=len(cl), closed_wrong=len(wr_cl), ratio_closed=rt, ratio_wrong=rtw)
    T["base_solved"] = mdtable(["編集", f"基底が closed で解いた {len(bsol)} 問の系列数", "正解率", "うち closed", "closed で誤答",
                                "編集の思考長 / 基底の思考長（closed 全体の中央値）", "同（closed 誤答のみ）"], rows)

    # ---- 1b 答えの形式と答えの後の文字
    rows = []
    for t in order:
        R = [r for r in F[t] if r["closed"] and not r["correct"]]
        n = len(R)
        c = Counter(r["ans_type"] for r in R)
        resume = sum((r["an_wait"] + r["an_hmm"]) >= 2 and r["ans_type"].startswith("nobox") for r in R)
        resume_any = sum(r["an_wait"] >= 1 for r in R)
        loop_end = sum(r["end_verbatim"] or r["end_repetitive"] for r in R)
        rows.append([label[t], n, f"{c['nobox_cut']} ({100 * c['nobox_cut'] / n:.0f}%)" if n else "–",
                     f"{c['nobox_stop']} ({100 * c['nobox_stop'] / n:.0f}%)" if n else "–",
                     c["format_miss"], f"{c['wrong_value']} ({100 * c['wrong_value'] / n:.0f}%)" if n else "–",
                     f"{resume_any} ({100 * resume_any / n:.0f}%)" if n else "–", loop_end])
        J.setdefault("ans_type", {})[t] = dict(n=n, **c, resume_any=resume_any, answer_end_loop=loop_end)
    T["ans_type"] = mdtable(["編集", "closed 誤答", "boxed なし・長さで切れ", "boxed なし・自然終了", "boxed あり・形式のみ不一致(ゆるい照合で一致)",
                             "boxed あり・値が違う", "答え部分に Wait を含む", "答えの末尾が繰り返し"], rows)

    rows = []
    for t in order:
        for nm, sel in (("正解", lambda r: r["correct"]), ("誤答", lambda r: not r["correct"])):
            R = [r for r in F[t] if r["closed"] and sel(r) and r.get("trail_kind")]
            n = len(R)
            if n == 0:
                continue
            c = Counter(r["trail_kind"] for r in R)
            rows.append([label[t], nm, n, f"{c['none']} ({100 * c['none'] / n:.0f}%)",
                         f"{c['short']} ({100 * c['short'] / n:.0f}%)", f"{c['text']} ({100 * c['text'] / n:.0f}%)",
                         f"{c['ws_short'] + c['ws_long']} ({100 * (c['ws_short'] + c['ws_long']) / n:.0f}%)",
                         max((r["trail_ws_run"] for r in R), default=0)])
    T["trail"] = mdtable(["編集", "closed かつ", "boxed あり件数", "後ろなし(空白≤3)", "後ろ短い文字(<20)", "後ろ長い文(≥20字)", "後ろが空白のみ(>3)",
                          "空白連続の最大長"], rows)

    # ---- 1c 検証語・結論ステップ
    rows = []
    for t in order:
        for nm, sel in (("正解", lambda r: r["correct"]), ("誤答", lambda r: not r["correct"])):
            R = [r for r in F[t] if r["closed"] and sel(r)]
            if not R:
                continue
            tt = sum(r["think_tok"] for r in R) / 1000
            wait = sum(r["th_wait"] for r in R) / tt
            ver = sum(r["th_verify"] for r in R) / tt
            rer = sum(r["th_reread"] for r in R) / tt
            hmm = sum(r["th_hmm"] for r in R) / tt
            zero = sum((r["th_wait"] + r["th_verify"] + r["th_reread"] + r["th_selfcorr"]) == 0 for r in R)
            conc = sum(r["th_concluded"] for r in R)
            hdr = [r for r in R if r["th_n_hdr"] > 0]
            fin = sum(bool(r["th_last_final"]) for r in hdr)
            loop = sum(r["th_tail_loop"] for r in R)
            aw = sum(r["an_wait"] >= 1 for r in R)
            rows.append([label[t], nm, len(R), f"{wait:.1f}", f"{ver:.1f}", f"{rer:.1f}", f"{hmm:.1f}",
                         f"{zero} ({100 * zero / len(R):.0f}%)", f"{conc} ({100 * conc / len(R):.0f}%)",
                         f"{fin}/{len(hdr)}", f"{loop} ({100 * loop / len(R):.0f}%)",
                         f"{aw} ({100 * aw / len(R):.0f}%)"])
    T["markers"] = mdtable(["編集", "closed かつ", "件数", "Wait /1k tok", "check・verify /1k tok", "re-read 系 /1k tok", "Actually・Hmm・Alt・However /1k tok",
                            "検証語ゼロ", "末尾に結論表現", "番号つき見出しの最後が最終段階/見出しありの件数", "思考の末尾が繰り返し", "答え部分に Wait を含む"], rows)

    rows = []
    for g in ("弱", "中", "強"):
        for nm, sel in (("正解", True), ("誤答", False)):
            R = [r for t in order if group[t] == g for r in F[t] if r["closed"] and r["correct"] == sel]
            if not R:
                continue
            tt = sum(r["think_tok"] for r in R) / 1000
            hdr = [r for r in R if r["th_n_hdr"] > 0]
            ln = sum(r["th_lines"] for r in R)
            zero = sum((r["th_wait"] + r["th_verify"] + r["th_reread"] + r["th_selfcorr"]) == 0 for r in R)
            rows.append([g, nm, len(R), f"{sum(r['th_wait'] for r in R) / tt:.1f}", f"{sum(r['th_verify'] for r in R) / tt:.1f}",
                         zero, pctn(sum(r["th_concluded"] for r in R), len(R)),
                         pctn(sum(bool(r["th_last_final"]) for r in hdr), len(hdr)),
                         pctn(sum(r["th_tail_loop"] for r in R), len(R)),
                         pctn(sum(r["an_wait"] >= 1 for r in R), len(R)),
                         f"{100 * sum(r['th_glitch'] * r['th_lines'] for r in R) / ln:.2f}%" if ln else "–"])
    T["markers_group"] = mdtable(["強さ群", "closed かつ", "件数", "Wait /1k tok", "check・verify /1k tok", "検証語が 1 つもない件数",
                                  "思考末尾 1,000 字に結論表現", "番号つき見出しの最後が最終段階", "思考末尾が繰り返し", "答え部分に Wait を含む",
                                  "思考中の括弧不整合行の割合"], rows)

    # 検証語の密度を思考長の帯で比べる（長さ交絡の確認）
    bins = [(0, 1000), (1000, 3000), (3000, 8000), (8000, 17000)]
    rows = []
    for lo, hi in bins:
        row = [f"{lo:,}–{hi:,}"]
        for g in ("弱", "中", "強"):
            R = [r for t in order if group[t] == g for r in F[t] if r["closed"] and lo <= r["think_tok"] < hi]
            if not R:
                row.append("–")
                continue
            tt = sum(r["think_tok"] for r in R) / 1000
            w_ = sum(r["th_wait"] for r in R) / tt
            wrong = sum(not r["correct"] for r in R)
            row.append(f"n={len(R)}, Wait {w_:.1f}/1k, 誤答 {100 * wrong / len(R):.0f}%")
        rows.append(row)
    T["marker_bins"] = mdtable(["思考長(tok)", "弱群", "中群", "強群"], rows)

    # ---- 1d 難度か劣化か
    rows = []
    for t in order:
        R = [r for r in F[t] if r["closed"] and not r["correct"]]
        n = len(R)
        cls = Counter()
        bs = 0
        for r in R:
            op, no = other_pass(r["id"], t)
            c = "H(他で0%)" if op == 0 else "E(他で≥50%)" if op >= 0.5 else "M(他で0–50%)"
            cls[c] += 1
            if base_correct[r["id"]] and t != base:
                bs += 1
        rows.append([label[t], n] + [f"{cls[c]} ({100 * cls[c] / n:.0f}%)" if n else "–" for c in ("H(他で0%)", "M(他で0–50%)", "E(他で≥50%)")]
                    + [f"{bs}" if t != base else "–"])
        J.setdefault("difficulty", {})[t] = dict(n=n, **cls, base_solved_problem=bs)
    T["difficulty"] = mdtable(["編集", "closed 誤答", "難問: 他の全編集・基底で一度も正解なし", "中間: 他での正解率 0–50%", "易問: 他での正解率 ≥50%",
                               "基底が正解した問題での誤答"], rows)

    # 問題ごとの正解の遷移（基底 → 編集）
    rows = []
    for t in order:
        if t == base:
            continue
        a = Counter()
        for pid in pids:
            b = by_prob_tag[pid].get(base, [])
            e = by_prob_tag[pid].get(t, [])
            if not b or not e:
                continue
            bc = b[0]["correct"]
            for r in e:
                key = ("基底○" if bc else "基底×") + "→" + ("編集○" if r["correct"] else
                                                          ("編集× closed" if r["closed"] else "編集× 未閉鎖"))
                a[key] += 1
        n = sum(a.values())
        rows.append([label[t], n, a["基底○→編集○"], a["基底○→編集× closed"], a["基底○→編集× 未閉鎖"],
                     a["基底×→編集○"], a["基底×→編集× closed"], a["基底×→編集× 未閉鎖"]])
        J.setdefault("transition", {})[t] = dict(a)
    T["transition"] = mdtable(["編集", "系列数", "基底○→○", "基底○→×(closed)", "基底○→×(未閉鎖)", "基底×→○", "基底×→×(closed)", "基底×→×(未閉鎖)"], rows)

    # ---- 2 思考長と正解
    rows = []
    z_by = {}
    allclosed = [r for r in allrows if r["closed"]]
    lm = defaultdict(list)
    for r in allclosed:
        lm[r["id"]].append(math.log(max(r["think_tok"], 1)))
    for r in allclosed:
        v = lm[r["id"]]
        r["z"] = math.log(max(r["think_tok"], 1)) - sum(v) / len(v) if len(v) >= 2 else None
    for t in order:
        R = [r for r in F[t] if r["closed"]]
        c = [r["think_tok"] for r in R if r["correct"]]
        w = [r["think_tok"] for r in R if not r["correct"]]
        a = auc(c, w)
        lo, hi = cluster_boot(R, lambda rs: auc([x["think_tok"] for x in rs if x["correct"]], [x["think_tok"] for x in rs if not x["correct"]]), B=500)
        zc = [r["z"] for r in R if r["correct"] and r["z"] is not None]
        zw = [r["z"] for r in R if not r["correct"] and r["z"] is not None]
        # 同じ問題・同じ編集の 2 標本のうち、一方が closed 正解・他方が closed 誤答の組
        pairs = []
        for pid in pids:
            rr = [r for r in by_prob_tag[pid].get(t, []) if r["closed"]]
            cs = [r for r in rr if r["correct"]]
            ws = [r for r in rr if not r["correct"]]
            for x in cs:
                for y in ws:
                    pairs.append((x["think_tok"], y["think_tok"]))
        npair = len(pairs)
        pl = sum(x > y for x, y in pairs)
        rows.append([label[t], f"{len(c)}/{len(w)}", f"{a:.2f} [{lo:.2f}–{hi:.2f}]" if a == a else "–",
                     f"{np.mean(zc):+.2f} / {np.mean(zw):+.2f}" if zc and zw else "–",
                     f"{pl}/{npair}" if npair else "–"])
        J.setdefault("length_vs_correct", {})[t] = dict(auc=a, auc_ci=[lo, hi], n_correct=len(c), n_wrong=len(w), pairs=npair, pairs_correct_longer=pl)
    T["length_auc"] = mdtable(["編集", "closed の 正解/誤答 件数", "AUC: 正解が誤答より長い確率 [95%区間, 問題単位ブートストラップ]",
                               "問題内で中心化した log 思考長の平均 (正解 / 誤答)", "同一問題・同一編集の組で、正解の方が長い組/全組"], rows)

    rows = []
    for gname, gl in (("全編集", list(order)), ("弱群", [t for t in order if group[t] == "弱"]),
                      ("中群", [t for t in order if group[t] == "中"]), ("強群", [t for t in order if group[t] == "強"])):
        R = [r for r in allclosed if r["tag"] in gl]
        if len(R) < 30 or len(gl) < 1:
            continue
        coef = fe_logit_coef(R, gl)
        lo, hi = cluster_boot(R, lambda rs: fe_logit_coef(rs, gl), B=150)
        rows.append([gname, len(R), len({r["id"] for r in R}), f"{coef:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]"])
        J.setdefault("fe_logit", {})[gname] = dict(n=len(R), coef=coef, ci=[lo, hi])
    T["fe_logit"] = mdtable(["対象 (closed のみ)", "標本数", "問題数", "log2 思考長 の係数 (対数オッズ/倍)", "95%区間 (問題単位ブートストラップ 150 回)"], rows)

    rows = []
    for gname, gl in (("弱群", [t for t in order if group[t] == "弱"]), ("中群", [t for t in order if group[t] == "中"]),
                      ("強群", [t for t in order if group[t] == "強"])):
        R = [dict(r, loop=int(r["th_tail_loop"])) for r in allclosed if r["tag"] in gl]
        if len(R) < 30:
            continue
        c = fe_logit_coefs(R, ("loop",))
        bs = []
        import numpy as _np
        by = defaultdict(list)
        for r in R:
            by[r["id"]].append(r)
        ids = list(by)
        rng = _np.random.default_rng(1)
        for _ in range(150):
            smp = rng.integers(0, len(ids), len(ids))
            bs.append(fe_logit_coefs([x for i in smp for x in by[ids[i]]], ("loop",)))
        bs = _np.array(bs)
        ci = [(_np.percentile(bs[:, j], 2.5), _np.percentile(bs[:, j], 97.5)) for j in range(2)]
        rows.append([gname, len(R), f"{c[0]:+.2f} [{ci[0][0]:+.2f}, {ci[0][1]:+.2f}]", f"{c[1]:+.2f} [{ci[1][0]:+.2f}, {ci[1][1]:+.2f}]"])
        J.setdefault("fe_logit_adj", {})[gname] = dict(len=c[0], loop=c[1], ci_len=ci[0], ci_loop=ci[1])
    T["fe_logit_adj"] = mdtable(["対象 (closed のみ)", "標本数", "log2 思考長 の係数 [95%区間]（思考末尾が繰り返しで調整）", "思考末尾が繰り返し の係数 [95%区間]"], rows)

    # acc@T と closed@T
    rows = []
    for t in order:
        R = F[t]
        n = len(R)
        row = [label[t]]
        for T_ in (2048, 4096, 8192, 16384):
            ok = sum(r["correct"] and r["tokens"] <= T_ for r in R)
            cl = sum(r["closed"] and r["think_tok"] <= T_ for r in R)
            row.append(f"{100 * ok / n:.1f}% / {100 * cl / n:.0f}%")
        rows.append(row)
    T["acc_at"] = mdtable(["編集", "≤2,048 tok", "≤4,096 tok", "≤8,192 tok", "≤16,384 tok"], rows)

    # 帯ごとの正解率（closed 内。同じ問題・同じ強さ群の closed 標本の平均 log 長で中心化した値の三分位）
    glm = defaultdict(list)
    for r in allclosed:
        glm[(r["id"], group[r["tag"]])].append(math.log(max(r["think_tok"], 1)))
    for r in allclosed:
        v = glm[(r["id"], group[r["tag"]])]
        r["zg"] = math.log(max(r["think_tok"], 1)) - sum(v) / len(v) if len(v) >= 2 else None
    rows = []
    for g in ("弱", "中", "強"):
        R = [r for r in allclosed if group[r["tag"]] == g and r["zg"] is not None]
        if len(R) < 12:
            continue
        zs = np.array([r["zg"] for r in R])
        q1, q2 = np.percentile(zs, [33.3, 66.7])
        row = [g, len(R), len({r["id"] for r in R})]
        for sel in (lambda z: z < q1, lambda z: q1 <= z < q2, lambda z: z >= q2):
            sub = [r for r in R if sel(r["zg"])]
            ok = sum(r["correct"] for r in sub)
            row.append(f"{100 * ok / len(sub):.0f}% ({ok}/{len(sub)})" if sub else "–")
        rows.append(row)
    T["z_terciles"] = mdtable(["強さ群", "closed 標本数(同じ問題・同じ群に closed が 2 件以上)", "問題数", "同じ問題の中で短い 1/3: 正解率", "中央 1/3", "長い 1/3"], rows)

    # 編集ごとの問題内固定効果ロジスティック（2 回/問の編集のみ）と、参照長との比に対するロジスティック
    rows = []
    for t in order:
        if len(F[t]) < 256:
            continue
        R = [r for r in allclosed if r["tag"] == t]
        coef = fe_logit_coef(R, [t])
        lo, hi = cluster_boot(R, lambda rs: fe_logit_coef(rs, [t]), B=150)
        rows.append([label[t], group[t], len(R), f"{coef:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]"])
        J.setdefault("fe_logit_tag", {})[t] = dict(n=len(R), coef=coef, ci=[lo, hi])
    T["fe_logit_tag"] = mdtable(["編集 (256 系列)", "群", "closed 標本数", "log2 思考長 の係数 (問題内固定効果)", "95%区間 (150 回)"], rows)

    def logit_slope(xs, ys, iters=30):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if len(xs) < 20 or ys.min() == ys.max():
            return float("nan")
        X = np.c_[np.ones(len(xs)), xs - xs.mean()]
        w = np.zeros(2)
        for _ in range(iters):
            p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
            H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
            st = np.linalg.solve(H, X.T @ (ys - p))
            w += st
            if np.abs(st).max() < 1e-8:
                break
        return float(w[1])
    rows = []
    for g in ("弱", "中", "強"):
        R = []
        for t in order:
            if group[t] != g:
                continue
            for r in allclosed:
                if r["tag"] == t:
                    ref, nref = ref_len(r["id"], t)
                    if ref:
                        R.append(dict(r, lr=math.log2(max(r["think_tok"], 1) / ref)))
        if len(R) < 20:
            continue
        f_ = lambda rs: logit_slope([x["lr"] for x in rs], [1.0 if x["correct"] else 0.0 for x in rs])  # noqa: E731
        c = f_(R)
        lo, hi = cluster_boot(R, f_, B=300)
        rows.append([g, len(R), len({r["id"] for r in R}), f"{c:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]",
                     pctn(sum(r["correct"] for r in R if r["lr"] < -1), sum(r["lr"] < -1 for r in R)),
                     pctn(sum(r["correct"] for r in R if -1 <= r["lr"] <= 1), sum(-1 <= r["lr"] <= 1 for r in R)),
                     pctn(sum(r["correct"] for r in R if r["lr"] > 1), sum(r["lr"] > 1 for r in R))])
        J.setdefault("ref_logit", {})[g] = dict(n=len(R), coef=c, ci=[lo, hi])
    T["ref_logit"] = mdtable(["強さ群", "参照長のある closed 標本", "問題数", "log2(思考長/参照長) の係数", "95%区間 (300 回)",
                              "比 <0.5 の正解率", "0.5–2 の正解率", "比 >2 の正解率"], rows)

    # 思考長の帯の分布（群 × 正誤）
    rows = []
    edges = [(0, 2000), (2000, 4000), (4000, 8000), (8000, 12000), (12000, 17000)]
    for g in ("弱", "中", "強"):
        for nm, sel in (("正解", True), ("誤答", False)):
            R = [r for r in allclosed if group[r["tag"]] == g and r["correct"] == sel]
            if not R:
                continue
            rows.append([g, nm, len(R)] + [f"{sum(lo_ <= r['think_tok'] < hi_ for r in R)} ({100 * sum(lo_ <= r['think_tok'] < hi_ for r in R) / len(R):.0f}%)" for lo_, hi_ in edges])
    T["length_bins"] = mdtable(["強さ群", "closed かつ", "件数"] + [f"{lo_:,}–{hi_:,}" for lo_, hi_ in edges], rows)

    # 思考長の帯ごとの正解率（closed のみ。上限で打ち切られた未閉鎖は含まれない点に注意）
    rows = []
    for g in ("弱", "中", "強"):
        row = [g]
        for lo_, hi_ in edges:
            R = [r for r in allclosed if group[r["tag"]] == g and lo_ <= r["think_tok"] < hi_]
            row.append(pctn(sum(r["correct"] for r in R), len(R)) if R else "–")
        n_un = sum(1 for t in order if group[t] == g for r in F[t] if not r["closed"])
        n_all = sum(len(F[t]) for t in order if group[t] == g)
        row.append(f"{n_un}/{n_all} ({100 * n_un / n_all:.0f}%)")
        rows.append(row)
    T["length_acc"] = mdtable(["強さ群"] + [f"{lo_:,}–{hi_:,}" for lo_, hi_ in edges] + ["参考: 未閉鎖(上限で打ち切り)の件数/全系列"], rows)

    # 答えの長さと予算（closed 誤答のうち boxed なし・長さ切れ）
    rows = []
    for g in ("弱", "中", "強"):
        C = [r for r in allclosed if group[r["tag"]] == g and r["correct"]]
        W = [r for r in allclosed if group[r["tag"]] == g and not r["correct"]]
        cut = [r for r in W if r["ans_type"] == "nobox_cut"]
        stop_nb = [r for r in W if r["ans_type"] == "nobox_stop"]
        val = [r for r in W if r["ans_type"] == "wrong_value"]
        rows.append([g, f"{len(C)}: {fmt_q([r['ans_tok'] for r in C])}",
                     f"{len(cut)}: {fmt_q([r['think_tok'] for r in cut])}",
                     pct(sum(r['think_tok'] > 15000 for r in cut), len(cut), False),
                     f"{len(stop_nb)}: {fmt_q([r['ans_tok'] for r in stop_nb])}",
                     f"{len(val)}: {fmt_q([r['ans_tok'] for r in val])}"])
    T["budget"] = mdtable(["強さ群", "正解: 答え部分トークン 中央値 (四分位)", "boxed なし・長さ切れ: 件数: 思考トークン",
                           "同: 思考が 15,000 tok 超", "boxed なし・自然終了: 件数: 答えトークン", "値が違う: 件数: 答えトークン"], rows)

    # 出典別
    rows = []
    for g in ("弱", "中", "強"):
        for src in ("gsm8k", "math"):
            R = [r for t in order if group[t] == g for r in F[t] if r["src"] == src]
            C = [r for r in R if r["closed"]]
            W = [r for r in C if not r["correct"]]
            rows.append([g, src, len(R), pct(sum(r["correct"] for r in R), len(R), False), pct(len(C), len(R), False),
                         pct(len(C) - len(W), len(C), False), len(W)])
    T["src"] = mdtable(["強さ群", "出典", "系列数", "正解率", "閉じた割合", "閉じた中の正解率", "closed 誤答数"], rows)

    # ラベルなしで測れる指標
    rows = []
    b = F[base]
    acc_b = sum(r["correct"] for r in b) / len(b)
    cl_b = sum(r["closed"] for r in b) / len(b)
    for t in order:
        R = F[t]
        n = len(R)
        acc = sum(r["correct"] for r in R) / n
        cl = sum(r["closed"] for r in R) / n
        stop_nb = sum(r["finish"] == "stop" and r["pred"] is None for r in R)
        nobox = sum(r["pred"] is None for r in R)
        ln = sum(r["th_lines"] for r in R)
        bad = sum(r["th_glitch"] * r["th_lines"] for r in R)
        dcl = cl - cl_b
        yld = (acc - acc_b) / dcl if dcl > 0.05 else None
        CL = [r for r in R if r["closed"]]
        lt = sum(r["th_tail_loop"] for r in CL)
        rows.append([label[t], f"{100 * acc:.1f}%", f"{100 * cl:.0f}%", pct(stop_nb, n, False) + f" ({stop_nb})",
                     f"{100 * bad / ln:.2f}%" if ln else "–", f"{lt}/{len(CL)} ({100 * lt / len(CL):.0f}%)",
                     f"{yld:.2f}" if yld is not None else "–"])
        J.setdefault("indicators", {})[t] = dict(acc=acc, closed=cl, stop_nobox=stop_nb / n, nobox=nobox / n, glitch=bad / ln if ln else None, yield_=yld)
    T["indicators"] = mdtable(["編集", "正解率", "閉じた割合", "自然終了(stop)なのに boxed なし", "思考中の括弧不整合行の割合(全標本)",
                               "閉じた標本のうち思考末尾が繰り返し", "変換効率 = (正解率−基底の)/(閉じた割合−基底の)"], rows)

    # 答え部分に Wait があるかどうか別の closed 正解率（群ごとの合計）
    rows = []
    for g in ("弱", "中", "強"):
        CL = [r for t in order if group[t] == g for r in F[t] if r["closed"]]
        a = [r for r in CL if r["an_wait"] >= 1]
        b_ = [r for r in CL if r["an_wait"] == 0]
        rows.append([g, len(CL), pctn(sum(r["correct"] for r in a), len(a)), pctn(sum(r["correct"] for r in b_), len(b_))])
    T["answer_wait"] = mdtable(["強さ群", "closed 数", "答え部分に Wait あり: 正解率 (正解/件数)", "答え部分に Wait なし: 正解率"], rows)

    # 「値が違う」誤答が、別の編集でも同じ値になっているか
    dd = defaultdict(set)
    for r in allclosed:
        if not r["correct"] and r["ans_type"] == "wrong_value":
            dd[(r["id"], r["pred"])].add(r["tag"])
    nw = sum(1 for r in allclosed if not r["correct"] and r["ans_type"] == "wrong_value")
    nsh = sum(1 for r in allclosed if not r["correct"] and r["ans_type"] == "wrong_value" and len(dd[(r["id"], r["pred"])]) >= 2)
    nprob = len({r["id"] for r in allclosed if not r["correct"] and r["ans_type"] == "wrong_value"})
    T["wrong_repeat"] = mdtable(["指標", "値"], [
        ["「boxed あり・値が違う」の closed 誤答（全編集）", nw],
        ["それが出た問題数", nprob],
        ["同じ問題で同じ誤答値が 2 本以上の編集に現れる標本", f"{nsh} ({100 * nsh / nw:.0f}%)"]])
    J["wrong_repeat"] = dict(n=nw, problems=nprob, shared=nsh)

    # 答え部分の長さ（</think> の後）
    rows = []
    for g in ("弱", "中", "強"):
        CL = [r for t in order if group[t] == g for r in F[t] if r["closed"]]
        c = [r for r in CL if r["correct"]]
        w = [r for r in CL if not r["correct"]]
        rows.append([g, pctn(sum(r["ans_tok"] <= 1000 for r in c), len(c)), pctn(sum(r["ans_tok"] > 1000 for r in w), len(w)),
                     pctn(sum(r["ans_tok"] <= 1000 for r in CL if r["correct"]), sum(r["ans_tok"] <= 1000 for r in CL)),
                     f"{np.median([r['ans_tok'] for r in w]):.0f}" if w else "–"])
    T["answer_len"] = mdtable(["強さ群", "closed 正解のうち答えが 1,000 tok 以下", "closed 誤答のうち答えが 1,000 tok 超",
                               "答えが 1,000 tok 以下の closed の正解率", "closed 誤答の答えトークン中央値"], rows)

    # 思考末尾が繰り返しだったかどうか別の closed 正解率
    rows = []
    for t in order:
        CL = [r for r in F[t] if r["closed"]]
        a = [r for r in CL if r["th_tail_loop"]]
        b_ = [r for r in CL if not r["th_tail_loop"]]
        rows.append([label[t], pctn(sum(r["correct"] for r in a), len(a)), pctn(sum(r["correct"] for r in b_), len(b_))])
    for g in ("弱", "中", "強"):
        CL = [r for t in order if group[t] == g for r in F[t] if r["closed"]]
        a = [r for r in CL if r["th_tail_loop"]]
        b_ = [r for r in CL if not r["th_tail_loop"]]
        rows.append([f"**{g}群 合計**", pctn(sum(r["correct"] for r in a), len(a)), pctn(sum(r["correct"] for r in b_), len(b_))])
    T["loop_tail_acc"] = mdtable(["編集", "closed かつ 思考末尾が繰り返し: 正解率 (正解/件数)", "closed かつ 思考末尾が繰り返しでない: 正解率"], rows)

    # 強い群の closed 誤答を難度クラス別に
    rows = []
    for g in ("中", "強"):
        for cname, sel in (("難問(他で0%)", lambda op: op == 0), ("中間", lambda op: 0 < op < 0.5), ("易問(他で≥50%)", lambda op: op >= 0.5)):
            W = [r for t in order if group[t] == g for r in F[t] if r["closed"] and not r["correct"] and sel(other_pass(r["id"], t)[0])]
            if not W:
                continue
            tt = sum(r["th_lines"] for r in W)
            gl = sum(r["th_glitch"] * r["th_lines"] for r in W) / tt if tt else 0
            ga = sum(r["an_glitch"] * r["an_lines"] for r in W) / max(1, sum(r["an_lines"] for r in W))
            rows.append([g, cname, len(W), f"{np.median([r['think_tok'] for r in W]):.0f}",
                         pctn(sum(r["ans_type"] == "nobox_stop" for r in W), len(W)),
                         pctn(sum(r["ans_type"] == "nobox_cut" for r in W), len(W)),
                         pctn(sum(r["ans_type"] == "wrong_value" for r in W), len(W)),
                         f"{100 * gl:.1f}% / {100 * ga:.1f}%"])
    T["difficulty_type"] = mdtable(["強さ群", "問題の難度クラス(他の編集での正解率)", "closed 誤答数", "思考トークン中央値", "boxed なし・自然終了", "boxed なし・長さ切れ",
                                    "値が違う", "括弧不整合行の割合 (思考 / 答え)"], rows)

    # ---- 3 強い編集と弱い編集で誤答の型
    def end_type(r):
        ref, nref = ref_len(r["id"], r["tag"])
        if r["th_tail_loop"]:
            return "繰り返し末尾で閉じた"
        if r["think_tok"] < 1000 or (ref and r["think_tok"] < 0.5 * ref):
            return "短く閉じた(<1,000 tok か参照の半分未満)"
        return "それ以外"
    rows = []
    taxo = {}
    ets = ["繰り返し末尾で閉じた", "短く閉じた(<1,000 tok か参照の半分未満)", "それ以外"]
    ats = ["nobox_cut", "nobox_stop", "format_miss", "wrong_value"]
    for g in ("弱", "中", "強"):
        R = [r for r in allclosed if group[r["tag"]] == g and not r["correct"]]
        n = len(R)
        if n == 0:
            continue
        for e in ets:
            sub = [r for r in R if end_type(r) == e]
            c = Counter(r["ans_type"] for r in sub)
            rows.append([g, e, len(sub), f"{100 * len(sub) / n:.0f}%"] + [c[a] for a in ats])
            taxo[f"{g}|{e}"] = dict(n=len(sub), **c)
    J["taxonomy"] = taxo
    T["taxonomy"] = mdtable(["強さ群", "思考の閉じ方", "件数", "群内の割合", "boxed なし(切れ)", "boxed なし(自然終了)", "形式のみ不一致", "値が違う"], rows)

    # 群ごとの要約
    rows = []
    for g in ("弱", "中", "強"):
        gl = [t for t in order if group[t] == g]
        R = [r for t in gl for r in F[t]]
        C = [r for r in R if r["closed"]]
        CW = [r for r in C if not r["correct"]]
        CC = [r for r in C if r["correct"]]
        if not C:
            continue
        rows.append([g, "、".join(label[t] for t in gl), len(R), len(C), pct(len(CC), len(C), False) + f" ({len(CC)}/{len(C)})",
                     f"{np.median([r['think_tok'] for r in CC]):.0f}" if CC else "–",
                     f"{np.median([r['think_tok'] for r in CW]):.0f}" if CW else "–",
                     pct(sum(r["th_tail_loop"] for r in CW), len(CW), False),
                     pct(sum(r["ans_type"].startswith("nobox") for r in CW), len(CW), False),
                     f"{np.mean([r['th_glitch'] for r in CW]):.3f}" if CW else "–",
                     f"{np.mean([r['th_glitch'] for r in CC]):.3f}" if CC else "–"])
    T["group_summary"] = mdtable(["強さ群", "含む編集", "系列数", "closed 数", "closed の正解率", "closed 正解の思考長の中央値",
                                  "closed 誤答の思考長の中央値", "誤答のうち思考末尾が繰り返し", "誤答のうち boxed なし", "思考中の括弧不整合行の割合(誤答)", "同(正解)"], rows)

    # 括弧不整合（劣化の目安）
    rows = []
    for t in order:
        row = [label[t]]
        for nm, sel in (("正解", lambda r: r["correct"]), ("誤答", lambda r: not r["correct"])):
            R = [r for r in F[t] if r["closed"] and sel(r)]
            ln = sum(r["th_lines"] for r in R)
            bad = sum(r["th_glitch"] * r["th_lines"] for r in R)
            la = sum(r["an_lines"] for r in R)
            ba = sum(r["an_glitch"] * r["an_lines"] for r in R)
            row += [f"{100 * bad / ln:.2f}%" if ln else "–", f"{100 * ba / la:.2f}%" if la else "–"]
        rows.append(row)
    T["glitch"] = mdtable(["編集", "思考 正解", "答え 正解", "思考 誤答", "答え 誤答"], rows)

    # 編集間の順位相関
    xs = [cshare(t) for t in order]
    corr_rows = []
    def share_of(t, f):
        R = [r for r in F[t] if r["closed"] and not r["correct"]]
        return (sum(f(r) for r in R) / len(R)) if R else None
    metrics = {
        "closed の正解率": lambda t: (lambda C: sum(r["correct"] for r in C) / len(C))([r for r in F[t] if r["closed"]]),
        "closed 誤答のうち 思考末尾が繰り返し": lambda t: share_of(t, lambda r: r["th_tail_loop"]),
        "closed 誤答のうち boxed なし": lambda t: share_of(t, lambda r: r["ans_type"].startswith("nobox")),
        "closed 誤答のうち 値が違う": lambda t: share_of(t, lambda r: r["ans_type"] == "wrong_value"),
        "closed 誤答のうち 検証語ゼロ": lambda t: share_of(t, lambda r: (r["th_wait"] + r["th_verify"] + r["th_reread"] + r["th_selfcorr"]) == 0),
        "closed 全体の Wait /1k tok": lambda t: (lambda C: sum(r["th_wait"] for r in C) / (sum(r["think_tok"] for r in C) / 1000))([r for r in F[t] if r["closed"]]),
        "closed の思考長の中央値": lambda t: float(np.median([r["think_tok"] for r in F[t] if r["closed"]])),
    }
    for nm, fn in metrics.items():
        pts = [(cshare(t), fn(t)) for t in order if fn(t) is not None and sum(r["closed"] for r in F[t]) >= 10]
        if len(pts) >= 5:
            rho, p = spearmanr([a for a, b in pts], [b for a, b in pts])
            corr_rows.append([nm, len(pts), f"{rho:+.2f}", f"{p:.3f}"])
    T["spearman"] = mdtable(["指標(編集ごと)", "編集数", "閉じた割合との Spearman ρ", "p 値(検定は目安のみ)"], corr_rows)

    # 3 のうち基底の closed 正解との並べ替え: 編集ごとの closed 誤答の「閉じた時点の思考長」分位
    # 出力
    (OUT / "tables.json").write_text(json.dumps(J, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for k, v in T.items():
        (OUT / f"t_{k}.md").write_text(v + "\n", encoding="utf-8")
    print("tables:", ", ".join(T))
    print("tags used:", ", ".join(order))
    print(f"peak working set {peak_mb():.0f} MB")


def cmd_assemble():
    tpl = (OUT / "report_template.md").read_text(encoding="utf-8")
    tags, _ = load_features()

    def sub(m):
        fp = OUT / f"t_{m.group(1)}.md"
        return fp.read_text(encoding="utf-8").strip() if fp.exists() else f"(表 {m.group(1)} が未生成)"
    out = re.sub(r"\{\{T_(\w+)\}\}", sub, tpl)
    (ROOT / "reports" / "analysis_closed_wrong.md").write_text(out, encoding="utf-8")
    print("written reports/analysis_closed_wrong.md,", len(out), "chars; tags:", ", ".join(t for t, _ in tags))


if __name__ == "__main__":
    lower_priority()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "extract":
        cmd_extract(force="--force" in sys.argv)
    elif cmd == "report":
        wait_for_memory()
        cmd_report()
    elif cmd == "assemble":
        cmd_assemble()
    else:
        sys.exit("usage: extract | report | assemble")
