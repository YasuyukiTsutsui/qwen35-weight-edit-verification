#!/usr/bin/env python3
"""Build the paper-style HTML report. Every number is read from results/ (summary.json, per_question.jsonl, monitor.csv).

    python scripts/build_report.py <fragment_out.html> [<standalone_out.html>]

fragment_out: page content for the Artifact tool (it adds the document skeleton).
standalone_out: the same page wrapped in a full HTML document, for opening from disk.
"""
import csv
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def summ(rel):
    return json.load(open(os.path.join(ROOT, "results", rel, "summary.json"), encoding="utf-8"))


def jl(rel):
    return [json.loads(l) for l in open(os.path.join(ROOT, "results", rel), encoding="utf-8")]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def fisher(a, n1, b, n2):
    kk, nn = a + b, n1 + n2
    pr = lambda x: math.comb(n1, x) * math.comb(n2, kk - x) / math.comb(nn, kk)
    po = pr(a)
    return sum(pr(x) for x in range(max(0, kk - n2), min(n1, kk) + 1) if pr(x) <= po + 1e-12)


def P(x, d=1):
    return f"{x * 100:.{d}f}%"


# ---------------------------------------------------------------- data
SW = [("基底モデル", "sweep_base", "base"),
      ("自前 k=64, α=1", "sweep_own_k64_a1", "own"), ("自前 k=64, α=2", "sweep_own_k64_a2", "own"),
      ("自前 k=64, α=3", "sweep_own_k64_a3", "own"), ("自前 k=64, α=4", "sweep_own_k64_a4", "own"),
      ("自前 k=64, α=6", "sweep_own_k64_a6", "own"), ("自前 k=64, α=12", "sweep_own_k64_a12", "own"),
      ("自前 k=128, α=6", "sweep_own_k128_a6", "own"), ("自前 k=128, α=12", "sweep_own_k128_a12", "own"),
      ("自前 k=256, α=6", "sweep_own_k256_a6", "own"), ("公開編集（k=64, α=6）", "sweep_p50c", "pub")]


def mkrow(label, rel, ent):
    s = summ(rel)
    return {"label": label, "ent": ent, "ok": s["ok"], "n": s["total"], "loop": s["cut_with_loop_tail"],
            "closed": s["think_closed_share"], "cut": s["cut_share"]}


SWEEP = [mkrow(l, f"{t}/practice_select128", e) for l, t, e in SW]
GSM = [mkrow("基底モデル", "quick_qwen_base/gsm8k_test", "base"), mkrow("公開編集", "quick_qwen_p50c/gsm8k_test", "pub"),
       mkrow("自前 k=64, α=12", "final_own_k64_a12/gsm8k_test", "own"), mkrow("自前 k=64, α=6", "quick_qwen_own/gsm8k_test", "own"),
       mkrow("ランダム対照", "quick_qwen_rand/gsm8k_test", "rand")]
MATH = [mkrow("基底モデル", "final_base/math500", "base"), mkrow("公開編集", "final_p50c/math500", "pub"),
        mkrow("自前 k=64, α=12", "final_own_k64_a12/math500", "own")]
GEM = summ("gemma_base/practice_find")
QB = summ("qwen_base/practice_find")
JM = summ("jev_4bit/decisionbench_medium_v1")
JH = summ("jev_4bit/decisionbench_hard_v1")
PQ = {"medium": jl("jev_4bit/decisionbench_medium_v1/per_question.jsonl"),
      "hard": jl("jev_4bit/decisionbench_hard_v1/per_question.jsonl")}


def logodds(x):
    return math.log(x["probs"][0] + 1e-12) - math.log(x["probs"][1] + 1e-12)


def noul_acc(sub, shift):
    nou = [x for x in PQ[sub] if x["type"] == "noul"]
    return sum((0 if logodds(x) + shift > 0 else 1) == x["gold"] for x in nou) / len(nou), len(nou)


SHIFTS = [0, 0.5, 0.75, 1.0, 1.5, 2.0]
SHIFT_ACC = {s: [noul_acc(s_, sh)[0] for sh in SHIFTS] for s, s_ in (("medium", "medium"), ("hard", "hard"))}
CM = {}
for sub in PQ:
    c = {(g, p): 0 for g in (0, 1) for p in (0, 1)}
    for x in PQ[sub]:
        if x["type"] == "noul":
            c[(x["gold"], x["pred"])] += 1
    CM[sub] = c

# ---------------------------------------------------------------- generic html bits


def table(headers, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in headers)
    tb = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table></div>'


def twin(headers, rows):
    return '<details class="tv"><summary>表で見る</summary>' + table(headers, rows) + "</details>"


ENT = {"base": "var(--c-base)", "own": "var(--c-own)", "pub": "var(--c-pub)", "rand": "var(--c-rand)"}


def rows_chart(rows, amax, step, title_acc="正答率（%）と95%信頼区間"):
    base = rows[0]
    n = base["n"]
    sc = 100 / amax
    h = ['<div class="rc">',
         f'<div class="rr hd"><div>設定<br><span style="font-weight:400">p は基底との差</span></div><div>{title_acc}</div><div>繰り返しで終わった応答（件）</div></div>']
    for r in rows:
        k, nn = r["ok"], r["n"]
        acc = k / nn
        lo, hi = wilson(k, nn)
        p = None if r is base else fisher(base["ok"], base["n"], k, nn)
        ptxt = "" if p is None else f"<small>p = {p:.3f}</small>"
        tipa = f"{acc * 100:.1f}%||{r['label']}：{k}/{nn} 問正解、95%信頼区間 {lo * 100:.1f}〜{hi * 100:.1f}%" + \
            ("" if p is None else f"、基底との差 p = {p:.3f}")
        tipl = f"{r['loop']} 件||{r['label']}：{nn} 応答のうち、繰り返しで終わったもの。思考を閉じた割合 {r['closed'] * 100:.1f}%"
        col = ENT[r["ent"]]
        h.append(
            f'<div class="rr"><div class="rl">{esc(r["label"])}{ptxt}</div>'
            f'<div class="cell"><div class="trk" style="--step:{step * sc:.3f}%" data-tip="{esc(tipa)}" tabindex="0">'
            f'<div class="bar" style="width:{acc * 100 * sc:.2f}%;background:{col}"></div>'
            f'<div class="ci" style="left:{lo * 100 * sc:.2f}%;width:{(hi - lo) * 100 * sc:.2f}%"></div>'
            f'<span class="val" style="left:{max(hi, acc) * 100 * sc:.2f}%">{acc * 100:.1f}%</span></div></div>'
            f'<div class="cell"><div class="trk" style="--step:50%" data-tip="{esc(tipl)}" tabindex="0">'
            f'<div class="bar" style="width:{r["loop"] / nn * 100:.2f}%;background:{col}"></div>'
            f'<span class="val" style="left:{r["loop"] / nn * 100:.2f}%">{r["loop"]}</span></div></div></div>')
    ax1 = "".join(f'<span style="left:{t * sc:.2f}%">{t:g}</span>' for t in [i * step for i in range(int(amax / step) + 1)])
    ax2 = "".join(f'<span style="left:{f * 100:g}%">{round(n * f)}</span>' for f in (0, 0.5, 1))
    h.append(f'<div class="rr ax"><div></div><div class="cell"><div class="trk0">{ax1}</div></div>'
             f'<div class="cell"><div class="trk0">{ax2}</div></div></div></div>')
    return "".join(h)


def rows_table(rows):
    b = rows[0]
    out = []
    for r in rows:
        lo, hi = wilson(r["ok"], r["n"])
        p = "—" if r is b else f"{fisher(b['ok'], b['n'], r['ok'], r['n']):.3f}"
        out.append([esc(r["label"]), f"{r['ok']}/{r['n']}", P(r["ok"] / r["n"]), f"{lo * 100:.1f}〜{hi * 100:.1f}%",
                    str(r["loop"]), P(r["closed"]), p])
    return twin(["設定", "正答", "正答率", "95%信頼区間", "繰り返しで終わった応答", "思考を閉じた割合", "p 値"], out)


# ---------------------------------------------------------------- svg
def topround(x, y, w, h, r=3):
    r = min(r, w / 2, h)
    return f"M{x:.1f},{y + h:.1f}V{y + r:.1f}Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f}H{x + w - r:.1f}Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}V{y + h:.1f}Z"


def line_svg(W, H, series, xr, xticks, yr, yticks, aria, refs=(), vlines=(), bands=(), xfmt=None, yfmt=None,
             ylab="", xlab="", ml=42, mr=88, mt=26, mb=36):
    xfmt = xfmt or (lambda v: f"{v:g}")
    yfmt = yfmt or (lambda v: f"{v:g}")
    x0, x1 = xr
    y0, y1 = yr
    pw, ph = W - ml - mr, H - mt - mb
    X = lambda v: ml + (v - x0) / (x1 - x0) * pw
    Y = lambda v: mt + (y1 - v) / (y1 - y0) * ph
    o = [f'<svg viewBox="0 0 {W} {H}" class="ch" role="img" aria-label="{esc(aria)}">']
    for b0, b1, lab in bands:
        o.append(f'<rect x="{X(b0):.1f}" y="{mt}" width="{X(b1) - X(b0):.1f}" height="{ph}" class="band"/>')
        o.append(f'<text x="{(X(b0) + X(b1)) / 2:.1f}" y="{mt + 11}" class="tl" text-anchor="middle">{esc(lab)}</text>')
    for t in yticks:
        o.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>')
        o.append(f'<text x="{ml - 6}" y="{Y(t) + 4:.1f}" class="tl" text-anchor="end">{esc(yfmt(t))}</text>')
    o.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(y0):.1f}" y2="{Y(y0):.1f}" class="axis"/>')
    for t in xticks:
        o.append(f'<text x="{X(t):.1f}" y="{mt + ph + 16}" class="tl" text-anchor="middle">{esc(xfmt(t))}</text>')
    if xlab:
        o.append(f'<text x="{ml + pw / 2:.1f}" y="{H - 4}" class="tl" text-anchor="middle">{esc(xlab)}</text>')
    if ylab:
        o.append(f'<text x="{ml}" y="{mt - 10}" class="tl">{esc(ylab)}</text>')
    for v, lab, anch in vlines:
        o.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{mt}" y2="{mt + ph}" class="vl"/>')
        o.append(f'<text x="{X(v) + (4 if anch == "start" else -4):.1f}" y="{mt + 11}" class="tl" text-anchor="{anch}">{esc(lab)}</text>')
    for yv, col, lab in refs:
        o.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" stroke="{col}" stroke-width="1.5"/>')
        o.append(f'<line x1="{ml + pw + 6}" x2="{ml + pw + 18}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" stroke="{col}" stroke-width="2"/>')
        o.append(f'<text x="{ml + pw + 22}" y="{Y(yv) + 4:.1f}" class="tl2">{esc(lab)}</text>')
    for s in series:
        pts = s["pts"]
        o.append(f'<polyline points="{" ".join(f"{X(p["x"]):.1f},{Y(p["y"]):.1f}" for p in pts)}" fill="none" stroke="{s["color"]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        if s.get("hits") == "col":
            for i, p in enumerate(pts):
                xl = X(p["x"]) - (X(pts[1]["x"]) - X(pts[0]["x"])) / 2
                o.append(f'<rect x="{xl:.1f}" y="{mt}" width="{X(pts[1]["x"]) - X(pts[0]["x"]):.1f}" height="{ph}" fill="transparent" data-tip="{esc(p["tip"])}"/>')
            continue
        for p in pts:
            o.append(f'<circle cx="{X(p["x"]):.1f}" cy="{Y(p["y"]):.1f}" r="4.5" fill="{s["color"]}" stroke="var(--surface)" stroke-width="2"/>')
            o.append(f'<circle cx="{X(p["x"]):.1f}" cy="{Y(p["y"]):.1f}" r="14" fill="transparent" tabindex="0" data-tip="{esc(p["tip"])}"/>')
            if p.get("note"):
                o.append(f'<text x="{X(p["x"]) + p.get("nx", 8):.1f}" y="{Y(p["y"]) + p.get("ny", -8):.1f}" class="tl2" text-anchor="{p.get("anch", "start")}">{esc(p["note"])}</text>')
        if s.get("end"):
            e = pts[-1]
            o.append(f'<line x1="{X(e["x"]) + 9:.1f}" x2="{X(e["x"]) + 21:.1f}" y1="{Y(e["y"]):.1f}" y2="{Y(e["y"]):.1f}" stroke="{s["color"]}" stroke-width="2"/>')
            o.append(f'<text x="{X(e["x"]) + 25:.1f}" y="{Y(e["y"]) + 4:.1f}" class="tl2">{esc(s["end"])}</text>')
    o.append("</svg>")
    return "".join(o)


def hist_svg(W, H, edges, yes, no, color, vlines, aria, ymax, yticks, ml=36, mr=10, mt=26, mb=36):
    pw, ph = W - ml - mr, H - mt - mb
    x0, x1 = edges[0], edges[-1]
    X = lambda v: ml + (v - x0) / (x1 - x0) * pw
    Y = lambda v: mt + (ymax - v) / ymax * ph
    o = [f'<svg viewBox="0 0 {W} {H}" class="ch" role="img" aria-label="{esc(aria)}">']
    for t in yticks:
        o.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>')
        o.append(f'<text x="{ml - 6}" y="{Y(t) + 4:.1f}" class="tl" text-anchor="end">{t}</text>')
    o.append(f'<text x="{ml}" y="{mt - 10}" class="tl">質問数</text>')
    for i in range(len(edges) - 1):
        xl, bw = X(edges[i]), X(edges[i + 1]) - X(edges[i])
        gw = bw * 0.86
        off = (bw - gw) / 2
        hw = gw / 2 - 1
        if yes[i]:
            o.append(f'<path d="{topround(xl + off, Y(yes[i]), hw, Y(0) - Y(yes[i]))}" fill="{color}"/>')
        if no[i]:
            o.append(f'<path d="{topround(xl + off + hw + 2, Y(no[i]), hw, Y(0) - Y(no[i]))}" fill="{color}" fill-opacity="0.14" stroke="{color}" stroke-width="1.5"/>')
        rng = f"{edges[i]:g}〜{edges[i + 1]:g}"
        o.append(f'<rect x="{xl:.1f}" y="{mt}" width="{bw:.1f}" height="{ph}" fill="transparent" data-tip="{esc(f"正解 Yes {yes[i]} 件、正解 No {no[i]} 件||Yes − No のログオッズ {rng}")}" />')
    o.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="axis"/>')
    for e in edges[::2]:
        o.append(f'<text x="{X(e):.1f}" y="{mt + ph + 16}" class="tl" text-anchor="middle">{e:g}</text>')
    o.append(f'<text x="{ml + pw / 2:.1f}" y="{H - 4}" class="tl" text-anchor="middle">Yes − No のログオッズ</text>')
    for v, lab, anch in vlines:
        o.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{mt}" y2="{mt + ph}" class="vl"/>')
        o.append(f'<text x="{X(v) + (4 if anch == "start" else -4):.1f}" y="{mt + 11}" class="tl" text-anchor="{anch}">{esc(lab)}</text>')
    o.append("</svg>")
    return "".join(o)


# ---------------------------------------------------------------- figures
def fig1():
    # (a) schematic of the edited matrix, (b) remaining coefficient on u as a function of alpha
    xs = (34, 52, 78, 101, 118, 140, 166, 181, 207, 229, 250, 268, 283, 301)
    sch = ('<svg viewBox="0 0 360 270" class="ch" role="img" aria-label="MLP の出力行列 W の一部の列から、方向 u の成分を引く編集の模式図">'
           '<text x="20" y="22" class="tl">W（down_proj）　1024 行 × 3584 列</text>'
           '<rect x="20" y="32" width="300" height="110" class="mat"/>'
           + "".join(f'<rect x="{x}" y="32" width="2.5" height="110" fill="var(--c-own)"/>' for x in xs)
           + '<rect x="20" y="156" width="10" height="10" fill="var(--c-own)"/><text x="38" y="165" class="tl2">編集する 64 列（S）。1 列は MLP ニューロン 1 個分</text>'
           '<rect x="20" y="174" width="10" height="10" class="mat"/><text x="38" y="183" class="tl2">それ以外の列（変更しない）</text>'
           '<text x="20" y="218" class="eq">w′ = w − α · u (uᵀw)</text>'
           '<text x="20" y="240" class="tl">S の各列 w に適用する。u は繰り返しに入る方向</text>'
           '<text x="20" y="258" class="tl">（単位ベクトル）、α は編集の強さ。</text>'
           '</svg>')
    pts = [{"x": a, "y": 1 - a, "tip": f"{1 - a:g}||α = {a} のとき、u 方向の成分は元の {1 - a:g} 倍"} for a in (0, 1, 2, 6, 12)]
    coef = line_svg(420, 270, [{"color": "var(--ink2)", "pts": pts, "end": "α=12"}], (0, 12.4), (0, 2, 4, 6, 8, 10, 12), (-12, 2), (-12, -8, -4, 0),
                    "α と、編集後に残る u 方向の成分の倍率", refs=[(0, "var(--muted)", "0（α=1）"), (-1, "var(--muted)", "−1（α=2）")],
                    yfmt=lambda v: f"{v:g}", xlab="α", ylab="u 方向の成分の倍率（1 − α）", ml=42, mr=92)
    return (f'<figure id="fig1"><div class="two"><div>{sch}</div><div>{coef}</div></div>'
            '<figcaption><b>図1　編集の仕組み。</b>左: 編集は MLP の出力行列の一部の列だけを書き換える。右: 書き換え後に残る u 方向の成分は元の（1 − α）倍になる。'
            'α=1 は成分を取り除くだけ（倍率 0）で、α=2 で符号が反転する（倍率 −1）。α が大きいほど逆向きに強く押す。</figcaption></figure>')


def fig2():
    return (f'<figure id="fig2">{rows_chart(SWEEP, 50, 10)}'
            f'<figcaption><b>図2　設定選択用 128 問での比較。</b>各設定の正答率（左）と、繰り返しで終わった応答の件数（右）。上限 8,192 トークン、1 問 1 回。'
            f'色は基底モデル（灰）、自前編集（橙）、公開編集（青）を表す。ひげは Wilson の 95% 信頼区間。p 値は Fisher の正確確率検定（両側）で、基底との比較。'
            f'α=12 の行は、この表の最高値として選んだ設定なので、p 値は実際より小さく出ている。</figcaption>{rows_table(SWEEP)}</figure>')


def fig3():
    own = [r for r in SWEEP if r["label"].startswith("自前 k=64")]
    alphas = [int(r["label"].split("α=")[1]) for r in own]
    base, pub = SWEEP[0], SWEEP[-1]

    def panel(key, yr, yt, fmt, lab, unit):
        pts = [{"x": a, "y": (r[key] if key == "loop" else r[key] * 100),
                "tip": f"{fmt(r[key] if key == 'loop' else r[key] * 100)}||自前 k=64, α={a}"} for a, r in zip(alphas, own)]
        bv = base[key] if key == "loop" else base[key] * 100
        pv = pub[key] if key == "loop" else pub[key] * 100
        return line_svg(420, 250, [{"color": "var(--c-own)", "pts": pts}], (0, 12.8), (1, 2, 3, 4, 6, 12), yr, yt, lab,
                        refs=[(bv, "var(--c-base)", f"基底 {fmt(bv)}"), (pv, "var(--c-pub)", f"公開 {fmt(pv)}")],
                        xlab="α（k=64）", ylab=unit, ml=40, mr=84)
    a = panel("loop", (0, 100), (0, 25, 50, 75, 100), lambda v: f"{v:.0f}", "α と、繰り返しで終わった応答の件数", "繰り返しで終わった応答（件 / 128）")
    b = panel("closed", (0, 70), (0, 20, 40, 60), lambda v: f"{v:.1f}%", "α と、思考を閉じた割合", "思考を閉じた割合（%）")
    tw = twin(["α", "繰り返しで終わった応答", "思考を閉じた割合"], [[str(x), str(r["loop"]), P(r["closed"])] for x, r in zip(alphas, own)])
    return (f'<figure id="fig3"><div class="two"><div><div class="pl">(a)</div>{a}</div><div><div class="pl">(b)</div>{b}</div></div>'
            f'<figcaption><b>図3　α を変えたときの応答の変化（自前編集、k=64）。</b>橙の線が自前編集、水平線が基底モデル（灰）と公開編集（青）。'
            f'横軸は α の値に比例している。α=1〜6 で繰り返しで終わった応答が 68 件から 20 件へ減り、α=12 では 26 件に戻る一方、思考を閉じた割合は 56% まで上がる。</figcaption>{tw}</figure>')


def fig4():
    return (f'<figure id="fig4"><h4 class="fh">GSM8K（テスト）64 問</h4>{rows_chart(GSM, 60, 10)}{rows_table(GSM)}'
            f'<h4 class="fh">MATH-500　64 問</h4>{rows_chart(MATH, 30, 10)}{rows_table(MATH)}'
            f'<figcaption><b>図4　選択に使っていない問題での比較。</b>上限 8,192 トークン、1 問 1 回。GSM8K の 64 問は最初の簡易測定と同じ問題で、α の選択には使っていない。'
            f'MATH-500 は基底モデルの応答の 95% が上限で打ち切られ、この条件では比較の力が弱い。見方は図2と同じ。</figcaption></figure>')


def fig5():
    errs = [("GSM8K の誤答", 12), ("MATH：上限で打ち切り", 14), ("MATH：\\boxed{} なしで完了", 6), ("MATH：誤答", 7)]
    total = GEM["total"]
    bars = "".join(
        f'<div class="rr eb"><div class="rl">{esc(l)}</div><div class="cell"><div class="trk" style="--step:25%" data-tip="{esc(f"{c} 問||{l}（全 {total} 問中）")}" tabindex="0">'
        f'<div class="bar" style="width:{c / 16 * 100:.1f}%;background:var(--c-base)"></div><span class="val" style="left:{c / 16 * 100:.1f}%">{c}</span></div></div></div>'
        for l, c in errs)
    ax = "".join(f'<span style="left:{t / 16 * 100:g}%">{t}</span>' for t in (0, 4, 8, 12, 16))
    assert 12 + 14 + 6 + 7 == total - GEM["ok"]
    tw = twin(["区分", "問数"], [["正解", str(GEM["ok"])]] + [[esc(l), str(c)] for l, c in errs])
    return (f'<figure id="fig5"><div class="hero"><div class="big">{GEM["acc"] * 100:.1f}%</div>'
            f'<div class="herol">{GEM["ok"]} / {total} 問正解<br>打ち切り {GEM["cut_share"] * 100:.1f}%　・　繰り返しで終わった応答 {GEM["cut_with_loop_tail"]} 件　・　思考を閉じた割合 {GEM["think_closed_share"] * 100:.1f}%</div></div>'
            f'<div class="rc"><div class="rr hd eb"><div>誤り {total - GEM["ok"]} 問の内訳</div><div>問数</div></div>{bars}'
            f'<div class="rr ax eb"><div></div><div class="cell"><div class="trk0">{ax}</div></div></div></div>'
            f'<figcaption><b>図5　Gemma 4 E2B（編集前）の結果。</b>方向探索用 256 問、思考あり、上限 6,144 トークン。誤りは 39 問あり、'
            f'繰り返しで終わった応答は 0 件だった。打ち切られた 14 問はすべて MATH で、長く続いた推論が上限に達したものである。</figcaption>{tw}</figure>')


def fig6():
    rowsd = [("全問（micro）", JM["micro_acc"], JH["micro_acc"], 0.8601),
             ("状況ごとの平均（state-macro）", JM["state_macro_acc"], JH["state_macro_acc"], 0.8757),
             ("noul（Yes/No）", JM["by_type"]["noul"], JH["by_type"]["noul"], None),
             ("choice（選択）", JM["by_type"]["choice"], JH["by_type"]["choice"], None),
             ("score（段階）", JM["by_type"]["score"], JH["by_type"]["score"], None)]
    h = ['<div class="rc"><div class="leg"><span><i class="sw" style="background:var(--c-med)"></i>medium</span>'
         '<span><i class="sw" style="background:var(--c-hard)"></i>hard</span><span><i class="tk"></i>medium の公開値</span></div>']
    for lab, m, hd, pubv in rowsd:
        tick = f'<i class="pubtick" style="left:{pubv * 100:.2f}%" data-tip="{esc(f"{pubv * 100:.2f}%||公開値（medium）")}"></i>' if pubv else ""
        h.append(f'<div class="rr jb"><div class="rl">{esc(lab).replace("（state-macro）", "<small>state-macro</small>").replace("（micro）", "<small>micro</small>")}</div><div class="cell"><div class="trk2" style="--step:25%">'
                 f'<div class="b2" tabindex="0" data-tip="{esc(f"{m * 100:.1f}%||{lab}、medium")}"><div class="bar" style="width:{m * 100:.2f}%;background:var(--c-med)"></div><span class="val in" style="left:{m * 100:.2f}%">{m * 100:.1f}%</span></div>'
                 f'<div class="b2" tabindex="0" data-tip="{esc(f"{hd * 100:.1f}%||{lab}、hard")}"><div class="bar" style="width:{hd * 100:.2f}%;background:var(--c-hard)"></div><span class="val in" style="left:{hd * 100:.2f}%">{hd * 100:.1f}%</span></div>'
                 f'{tick}</div></div></div>')
    ax = "".join(f'<span style="left:{t}%">{t}%</span>' for t in (0, 25, 50, 75, 100))
    h.append(f'<div class="rr ax jb"><div></div><div class="cell"><div class="trk0">{ax}</div></div></div></div>')
    tw = twin(["指標", "medium", "hard", "medium の公開値"],
              [[esc(l), P(m), P(hd), (f"{pv * 100:.2f}%" if pv else "—")] for l, m, hd, pv in rowsd] +
              [["ECE", f"{JM['ece']:.3f}", f"{JH['ece']:.3f}", "0.0400"]])
    return (f'<figure id="fig6">{"".join(h)}'
            f'<figcaption><b>図6　Jev-Omni（4bit）の正答率。</b>各 293 問。縦の短い線は medium の公開値。'
            f'hard は medium より全指標で低く、ECE（確信度と正答率のずれ）は {JM["ece"]:.3f} から {JH["ece"]:.3f} に増えた。</figcaption>{tw}</figure>')


def fig7():
    def ser(sub, color, name):
        return {"color": color, "end": name, "pts": [{"x": s, "y": a * 100, "tip": f"{a * 100:.1f}%||{name}、補正量 +{s:g}"} for s, a in zip(SHIFTS, SHIFT_ACC[sub])]}
    svg = line_svg(480, 280, [ser("medium", "var(--c-med)", "medium"), ser("hard", "var(--c-hard)", "hard")], (0, 2), SHIFTS, (55, 100), (60, 70, 80, 90, 100),
                   "noul の正答率と補正量", vlines=[(0.75, "事前に固定した +0.75", "start")], xlab="補正量（Yes − No のログオッズに足す値）",
                   ylab="noul の正答率（%、縦軸は 55% から）", xfmt=lambda v: f"+{v:g}", yfmt=lambda v: f"{v:g}", ml=44, mr=84)
    tw = twin(["補正量"] + [f"+{s:g}" for s in SHIFTS],
              [["medium"] + [P(a) for a in SHIFT_ACC["medium"]], ["hard"] + [P(a) for a in SHIFT_ACC["hard"]]])
    return (f'<figure id="fig7">{svg}<figcaption><b>図7　noul の正答率と補正量。</b>補正量を足すと medium（紫）は 83.6% から最大 93.0% に上がるが、'
            f'hard（桃）は 66.4% から 67.2% にしか上がらず、補正量を変えても単調には動かない。各 128 問。</figcaption>{tw}</figure>')


def fig8():
    edges = list(range(-14, 13, 2))
    out = []
    for sub, col, nm in (("medium", "var(--c-med)", "medium"), ("hard", "var(--c-hard)", "hard")):
        yes, no = [0] * (len(edges) - 1), [0] * (len(edges) - 1)
        for x in PQ[sub]:
            if x["type"] != "noul":
                continue
            v = min(max(logodds(x), edges[0] + 1e-9), edges[-1] - 1e-9)
            i = int((v - edges[0]) // 2)
            (yes if x["gold"] == 0 else no)[i] += 1
        svg = hist_svg(420, 250, edges, yes, no, col, [(0, "しきい値 0", "start"), (-0.75, "補正後 −0.75", "end")],
                       f"{nm} の noul：Yes − No のログオッズの分布", 20, (0, 5, 10, 15, 20))
        out.append((nm, svg, yes, no))
    tw = twin(["範囲"] + [f"{edges[i]:g}〜{edges[i + 1]:g}" for i in range(len(edges) - 1)],
              [[f"{nm}　正解 Yes"] + [str(v) for v in y] for nm, _, y, _ in out] + [[f"{nm}　正解 No"] + [str(v) for v in n] for nm, _, _, n in out])
    c = CM
    return (f'<figure id="fig8"><div class="leg"><span><i class="sw" style="background:var(--ink2)"></i>塗り：正解が Yes の質問</span>'
            f'<span><i class="sw ring"></i>枠のみ：正解が No の質問</span></div>'
            f'<div class="two"><div><div class="pl">medium</div>{out[0][1]}</div><div><div class="pl">hard</div>{out[1][1]}</div></div>'
            f'<figcaption><b>図8　noul の Yes − No のログオッズの分布。</b>各 128 問。両端の棒は範囲外の値を含みうる。medium では正解が Yes の質問（塗り）と No の質問（枠）が左右に分かれているが、'
            f'hard では重なりが大きく、しきい値をずらしても分けきれない。'
            f'誤りの内訳は medium が Yes→No {c["medium"][(0, 1)]} 件・No→Yes {c["medium"][(1, 0)]} 件、hard が Yes→No {c["hard"][(0, 1)]} 件・No→Yes {c["hard"][(1, 0)]} 件。</figcaption>{tw}</figure>')


def fig9():
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "results", "logs", "monitor.csv")))]
    T = lambda r: time.mktime(time.strptime(r["time"], "%Y-%m-%d %H:%M:%S"))
    day = [r for r in rows if r["time"].startswith("2026-11-04") and "06:20:00" <= r["time"][11:] <= "06:45:00"]
    t0 = T(day[0]) - (T(day[0]) % 300) if False else time.mktime(time.strptime("2026-11-04 06:20:00", "%Y-%m-%d %H:%M:%S"))
    mins = lambda r: (T(r) - t0) / 60
    busy = [r for r in rows if r["time"].startswith("2026-11-04") and float(r["gpu_mem_mib"]) > 5000]
    segs = [[busy[0]]]
    for r in busy[1:]:
        (segs[-1].append(r) if T(r) - T(segs[-1][-1]) <= 20 else segs.append([r]))
    names = ["試験", "medium", "hard"]
    bands = [(mins(s[0]), mins(s[-1]), n) for s, n in zip(segs[-3:], names)]
    xt = list(range(0, 26, 5))
    xf = lambda v: f"06:{20 + int(v):02d}"

    def panel(key, yr, yt, name, refs, unit, fmt):
        pts = [{"x": mins(r), "y": float(r[key]), "tip": f"{fmt(float(r[key]))}||{r['time'][11:16]}"} for r in day]
        return line_svg(420, 250, [{"color": "var(--ink2)", "pts": pts, "hits": "col"}], (0, 25), xt, yr, yt, name, refs=refs, bands=bands, xfmt=xf,
                        xlab="時刻（2026-11-04）", ylab=unit, ml=40, mr=84)
    a = panel("ram_free_gb", (0, 24), (0, 8, 16, 24), "実行中のホストの空きメモリ",
              [(1.5, "var(--crit)", "停止 1.5")], "ホストの空きメモリ（GB）", lambda v: f"{v:.1f} GB")
    b = panel("gpu_temp_c", (30, 90), (30, 50, 70, 90), "GPU 温度", [(85, "var(--crit)", "停止 85")], "GPU 温度（℃）", lambda v: f"{v:.0f} ℃")
    peaks = []
    for s, n in zip(segs[-3:], names):
        peaks.append([n, f"{max(float(x['gpu_mem_mib']) for x in s) / 1024:.1f} GB", f"{max(float(x['gpu_temp_c']) for x in s):.0f} ℃",
                      f"{max(float(x['gpu_power_w']) for x in s):.0f} W"])
    for pk, (s, n) in zip(peaks, zip(segs[-3:], names)):
        lo, hi = s[0], s[-1]
        w = [x for x in rows if T(lo) - 60 <= T(x) <= T(hi) + 30]
        m = min(w, key=lambda x: float(x["ram_free_gb"]))
        pk.append(f"{float(m['ram_free_gb']):.1f} GB（{m['time'][11:16]}）")
    tw = twin(["実行", "GPU メモリのピーク", "温度のピーク", "電力のピーク", "空きメモリの最小"], peaks)
    return (f'<figure id="fig9"><div class="two"><div><div class="pl">(a)</div>{a}</div><div><div class="pl">(b)</div>{b}</div></div>'
            f'<figcaption><b>図9　Jev-Omni（4bit）実行中の資源。</b>5 秒ごとの記録。薄い帯が 3 回の実行（3 状況の試験、medium、hard）。'
            f'赤線は見張りの停止基準（空きメモリ 1.5 GB 未満、GPU 温度 85 ℃超）で、実行中の値は基準に達しなかった。実行前の点検（空きメモリ 8 GB 以上）は起動前の条件なので、実行中に空きメモリが 8 GB を下回るのは想定内である。</figcaption>{tw}</figure>')


# ---------------------------------------------------------------- key results table
def key_table():
    g0, gp, go = GSM[0], GSM[1], GSM[2]
    m0, mp, mo = MATH[0], MATH[1], MATH[2]
    s0, sp = SWEEP[0], SWEEP[-1]
    so = SWEEP[6]
    f = lambda a, b: f"{fisher(a['ok'], a['n'], b['ok'], b['n']):.3f}"
    r = [
        ["Qwen3.5-0.8B　設定選択用 128 問", "公開編集", P(s0["ok"] / s0["n"]), P(sp["ok"] / sp["n"]), f"p = {f(s0, sp)}"],
        ["Qwen3.5-0.8B　GSM8K 64 問", "公開編集", P(g0["ok"] / g0["n"]), P(gp["ok"] / gp["n"]), f"p = {f(g0, gp)}"],
        ["Qwen3.5-0.8B　GSM8K 64 問", "自前編集（k=64, α=12）", P(g0["ok"] / g0["n"]), P(go["ok"] / go["n"]), f"p = {f(g0, go)}"],
        ["Qwen3.5-0.8B　MATH-500 64 問", "公開編集", P(m0["ok"] / m0["n"]), P(mp["ok"] / mp["n"]), f"p = {f(m0, mp)}"],
        ["Qwen3.5-0.8B　MATH-500 64 問", "自前編集（k=64, α=12）", P(m0["ok"] / m0["n"]), P(mo["ok"] / mo["n"]), f"p = {f(m0, mo)}"],
        ["Gemma 4 E2B　方向探索用 256 問", "編集せず", P(GEM["acc"]), "—", "繰り返しで終わった応答 0 件"],
        ["Jev-Omni　medium の noul 128 問", "補正量 +0.75（調整に使用）", P(SHIFT_ACC["medium"][0]), P(SHIFT_ACC["medium"][2]), "同じデータで調整"],
        ["Jev-Omni　hard の noul 128 問", "補正量 +0.75（未使用データ）", P(SHIFT_ACC["hard"][0]), P(SHIFT_ACC["hard"][2]), "正解が 1 問増えただけ"],
    ]
    return table(["対象", "操作", "編集前", "編集後", "差の確かさ"], r, "kt")


# ---------------------------------------------------------------- page
CSS = r"""
:root{--plane:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
--accent:#1c5cab;--band:#efeee9;--c-base:#898781;--c-pub:#2a78d6;--c-own:#eb6834;--c-rand:#1baf7a;--c-med:#6250d6;--c-hard:#e87ba4;--crit:#d03b3b;
--serif:"Noto Serif JP","Hiragino Mincho ProN","Yu Mincho","MS PMincho",serif;
--sans:"Noto Sans JP","Hiragino Sans","Yu Gothic UI","Meiryo",system-ui,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){@@DARK@@color-scheme:dark}}
:root[data-theme="dark"]{@@DARK@@color-scheme:dark}
body{background:var(--plane);color:var(--ink)}
.paper{max-width:46rem;margin:0 auto;padding-block:32px 72px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));
font:400 16px/1.95 var(--sans);color:var(--ink);overflow-wrap:anywhere}
.paper *{box-sizing:border-box}
.paper h1,.paper h2,.paper h3,.paper h4{font-family:var(--serif);font-weight:700;line-height:1.5;text-wrap:balance;margin:0}
.kicker{font:500 12px/1 var(--sans);letter-spacing:.12em;color:var(--ink2);margin:0 0 14px}
.paper h1{font-size:clamp(1.55rem,5.4vw,2.15rem);margin-bottom:8px}
.sub{font:500 1.02rem/1.6 var(--sans);color:var(--ink2);margin:0 0 14px}
.by{font-size:.86rem;color:var(--ink2);margin:0;padding-bottom:18px;border-bottom:1px solid var(--axis);line-height:1.8}
.paper h2{font-size:1.32rem;margin:2.6rem 0 .7rem;padding-top:1.2rem;border-top:1px solid var(--grid)}
.paper h2 .sn{color:var(--accent);margin-right:.5em;font-family:var(--sans);font-weight:500}
.paper h3{font-size:1.06rem;margin:1.8rem 0 .5rem}
.paper p{margin:.7rem 0}
.paper .abs{margin-top:1.6rem}
.paper .abs h2{border:0;padding-top:0;margin-top:0;font-size:1rem;letter-spacing:.2em}
.paper .abs p{font-size:.97rem}
.kw{font-size:.86rem;color:var(--ink2)}
.paper a{color:var(--accent);text-underline-offset:3px}
.paper code{font:500 .86em ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--band);padding:.1em .35em;border-radius:3px}
.paper ul,.paper ol{padding-left:1.4em;margin:.6rem 0}.paper li{margin:.25rem 0}
.tw{overflow-x:auto;margin:.6rem 0}
.paper table{border-collapse:collapse;width:100%;font-size:.86rem;line-height:1.55}
.paper th,.paper td{padding:.45rem .6rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--grid);overflow-wrap:normal}
.paper thead th{font-weight:700;border-bottom:1.5px solid var(--axis);color:var(--ink2);white-space:nowrap}
.paper td{font-variant-numeric:tabular-nums}
.paper table.tm td:first-child{white-space:nowrap}
.paper table.kt td:nth-child(n+3):nth-child(-n+4){white-space:nowrap;text-align:right}
.paper table.kt th:nth-child(n+3):nth-child(-n+4){text-align:right}
.tcap{font-size:.86rem;color:var(--ink2);margin:1.4rem 0 .2rem;line-height:1.7}.tcap b{color:var(--ink)}
figure{margin:1.6rem 0;padding:16px;background:var(--surface);border:1px solid var(--ring);border-radius:6px}
figcaption{font-size:.84rem;line-height:1.75;color:var(--ink2);margin-top:12px}figcaption b{color:var(--ink)}
.fh{font:700 .92rem/1.4 var(--sans)!important;margin:6px 0 4px!important;color:var(--ink)}
.pl{font:700 .8rem/1 var(--sans);color:var(--ink2);margin-bottom:4px}
.two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
@media (max-width:640px){.two{grid-template-columns:minmax(0,1fr)}}
.ch{display:block;width:100%;height:auto;max-width:100%;font-family:var(--sans)}
.ch .tl{font-size:11.5px;fill:var(--muted)}.ch .tl2{font-size:12px;fill:var(--ink2)}.ch .eq{font:500 16px var(--sans);fill:var(--ink)}
.ch .grid{stroke:var(--grid);stroke-width:1}.ch .axis{stroke:var(--axis);stroke-width:1}.ch .vl{stroke:var(--ink2);stroke-width:1;fill:none}
.ch .band{fill:var(--band)}.ch .mat{fill:var(--band);stroke:var(--axis);stroke-width:1}
.ch circle[tabindex],.ch rect[data-tip]{outline:none}.ch circle[tabindex]:focus-visible{stroke:var(--accent);stroke-width:2}
/* rows chart */
.rc{--lw:9.2rem;font-size:.86rem}
.rr{display:grid;grid-template-columns:var(--lw) minmax(0,1.15fr) minmax(0,1fr);gap:0 12px;align-items:center;min-height:34px;padding:3px 0;border-bottom:1px solid var(--grid)}
.rr.eb,.rr.jb{grid-template-columns:var(--lw) minmax(0,1fr)}
.rr.hd{font-size:.76rem;color:var(--ink2);font-weight:700;align-items:end;line-height:1.4;border-bottom:1.5px solid var(--axis);padding-bottom:6px}
.rr.ax{border:0;min-height:24px}
.rl{line-height:1.35}.rl small{display:block;color:var(--ink2);font-size:.72rem;line-height:1.3;margin-top:1px}
.cell{min-width:0;padding-right:2.9em}
.trk{position:relative;height:22px;background-image:linear-gradient(to right,var(--grid) 1px,transparent 1px);background-size:var(--step) 100%;outline:none}
.trk:focus-visible,.b2:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.trk .bar{position:absolute;left:0;top:4px;height:14px;border-radius:0 4px 4px 0;min-width:2px}
.trk .ci{position:absolute;top:50%;height:10px;transform:translateY(-50%);border-left:1.5px solid var(--ink);border-right:1.5px solid var(--ink);opacity:.75}
.trk .ci::before{content:"";position:absolute;left:0;right:0;top:50%;height:1.5px;background:var(--ink)}
.val{position:absolute;top:50%;transform:translateY(-50%);margin-left:7px;font-size:.78rem;white-space:nowrap;color:var(--ink);font-variant-numeric:tabular-nums;background:var(--surface);padding:0 2px;border-radius:2px}
.trk0{position:relative;height:16px;font-size:.7rem;color:var(--muted)}.trk0 span{position:absolute;top:0;transform:translateX(-50%);white-space:nowrap}
.trk0 span:first-child{transform:none}
.trk2{position:relative;display:flex;flex-direction:column;gap:3px;padding:3px 0;background-image:linear-gradient(to right,var(--grid) 1px,transparent 1px);background-size:var(--step) 100%}
.b2{position:relative;height:20px;outline:none}.b2 .bar{top:2px;height:16px}
.rr.jb{min-height:56px}
.val.in{margin:0 7px 0 0;transform:translate(-100%,-50%);background:none;color:#fff;padding:0;font-weight:700}
.pubtick{position:absolute;top:0;bottom:0;width:0;border-left:2px solid var(--ink);opacity:.8}
.leg{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.8rem;color:var(--ink2);margin:0 0 8px}
.leg .sw{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:-1px}.leg .sw.ring{background:transparent;border:1.5px solid var(--ink2)}
.leg .tk{display:inline-block;width:0;height:13px;border-left:2px solid var(--ink);margin-right:7px;vertical-align:-2px}
.hero{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 18px;margin-bottom:10px}
.hero .big{font:700 clamp(2.6rem,12vw,3.6rem)/1 var(--sans);letter-spacing:-.01em}
.hero .herol{font-size:.86rem;color:var(--ink2);line-height:1.7}
details.tv{margin-top:10px}details.tv summary{cursor:pointer;font-size:.8rem;color:var(--accent);width:max-content;padding:2px 0}
details.tv summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.refs{font-size:.84rem;color:var(--ink2);padding-left:1.4em}.refs li{margin:.3rem 0;line-height:1.7}
.note{background:var(--band);border-radius:6px;padding:10px 14px;font-size:.9rem;margin:1rem 0}
#tip{position:fixed;z-index:50;pointer-events:none;max-width:min(320px,calc(100vw - 24px));background:var(--ink);color:var(--plane);padding:7px 10px;border-radius:5px;font-size:.78rem;line-height:1.5;box-shadow:0 2px 10px rgba(0,0,0,.25)}
#tip b{display:block;font-size:.95rem}#tip span{opacity:.85}
@media (max-width:480px){.rc{--lw:6.4rem}.rr{gap:0 8px}.cell{padding-right:2.6em}.paper{font-size:15.5px}}
@media (prefers-reduced-motion:no-preference){.paper details.tv summary{transition:color .15s}}
"""

DARK = ("--plane:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);"
        "--accent:#6da7ec;--band:#232321;--c-base:#898781;--c-pub:#3987e5;--c-own:#d95926;--c-rand:#199e70;--c-med:#9085e9;--c-hard:#d55181;--crit:#e66767;")

JS = r"""
(function(){
var tip=document.getElementById('tip');
function fill(t){while(tip.firstChild)tip.removeChild(tip.firstChild);var p=t.split('||');
 var b=document.createElement('b');b.textContent=p[0];tip.appendChild(b);
 if(p[1]){var s=document.createElement('span');s.textContent=p[1];tip.appendChild(s);}tip.hidden=false;}
function place(x,y){var w=tip.offsetWidth,h=tip.offsetHeight;var l=x+14,t=y+14;
 if(l+w>window.innerWidth-8)l=Math.max(8,x-w-14);if(t+h>window.innerHeight-8)t=Math.max(8,y-h-14);
 tip.style.left=l+'px';tip.style.top=t+'px';}
document.addEventListener('pointermove',function(e){var el=e.target.closest&&e.target.closest('[data-tip]');
 if(!el){tip.hidden=true;return;}fill(el.getAttribute('data-tip'));place(e.clientX,e.clientY);});
document.addEventListener('pointerleave',function(){tip.hidden=true;});
document.addEventListener('focusin',function(e){var el=e.target.closest&&e.target.closest('[data-tip]');
 if(!el)return;fill(el.getAttribute('data-tip'));var r=el.getBoundingClientRect();place(r.left+r.width/2,r.top+r.height/2);});
document.addEventListener('focusout',function(){tip.hidden=true;});
})();
"""


def build():
    css = CSS.replace("@@DARK@@", DARK)
    f1, f2, f3, f4, f5, f6, f7, f8, f9 = fig1(), fig2(), fig3(), fig4(), fig5(), fig6(), fig7(), fig8(), fig9()
    base_pub = fisher(SWEEP[0]["ok"], SWEEP[0]["n"], SWEEP[-1]["ok"], SWEEP[-1]["n"])
    g_pub = fisher(GSM[0]["ok"], GSM[0]["n"], GSM[1]["ok"], GSM[1]["n"])
    g_own = fisher(GSM[0]["ok"], GSM[0]["n"], GSM[2]["ok"], GSM[2]["n"])
    terms = table(["区分", "用語", "意味"], cls="tm", rows=[
        ["元資料", "p50c", "元リポジトリで、公開編集を作った実行の呼称。本稿では「公開編集」と書く。"],
        ["元資料", "α、u、ランク1", "編集の強さ、引く方向（単位ベクトル）、編集で変わる量が 1 次元に収まること。"],
        ["元資料", "DecisionBench、medium、hard", "Jev-Omni の評価データと、その難易度の名前。各 80 状況・293 問。"],
        ["元資料", "noul、choice、score", "DecisionBench の質問型の名前。Yes/No、選択肢から選ぶもの、段階を答えるもの。"],
        ["元資料", "micro、state-macro、ECE", "公開値の指標名。全問の正答率、状況ごとの正答率の平均、確信度と正答率のずれ。"],
        ["本稿", "基底モデル", "編集する前のモデル。"],
        ["本稿", "公開編集", "元リポジトリが公開している編集済みの重み。"],
        ["本稿", "自前編集", "Claude が 2.4 節の手順で作った編集。"],
        ["本稿", "k", "各層で編集する列の数。"],
        ["本稿", "繰り返しの文脈／進んでいる文脈", "2.4 節で方向を求めるために使った 2 種類の文脈。元資料の英語では repetitive / productive。"],
        ["本稿", "繰り返しで終わった応答", "上限で打ち切られた応答のうち、末尾が同じ内容の繰り返しになっているもの。判定は <code>scripts/grade.py</code> の <code>loop_signals</code>（同じ文の完全な繰り返しと、4-gram の繰り返しの多さ）。"],
        ["本稿", "思考を閉じた割合", "思考を閉じるタグに到達した応答の割合。Qwen は <code>&lt;/think&gt;</code>、Gemma 4 は <code>&lt;channel|&gt;</code>。"],
        ["本稿", "方向探索用／設定選択用の問題", "方向を求めるための 256 問／α と k の候補を比べるための 128 問。"],
        ["本稿", "ランダム対照", "ランダムな方向で同じ式の編集をしたモデル。"],
        ["本稿", "組み立て方 v1／v2", "Jev-Omni の質問型から選択肢を作る規則（<code>--fmt</code>）。公開物に記載がないため Claude が推測した。"],
        ["本稿", "補正量／Yes − No のログオッズ", "noul で「Yes の確率の対数 − No の確率の対数」を求め、これが正なら Yes と答える。補正量はこの値に足す定数。"],
    ])
    t3 = table(["項目", "公開編集", "自前編集"], [
        ["編集の式", "w − α·u(uᵀw)、列の部分集合に適用", "同じ式"],
        ["方向 u の求め方", "元リポジトリの記録による（層16・層20の出力、4組の対応させた文脈）", "層15・層19の MLP 出力の差（2.4 節）"],
        ["層", "15 と 19", "15 と 19（元ブログから借用）"],
        ["列数と強さ", "各層 64 列、α = 6.00", "各層 64 列、α = 12（k = 128, 256 も試験）"],
        ["行列の相対変化（層15 / 層19）", "10.6% / 8.1%", "23.5% / 28.0%（α = 12）"],
        ["公開編集の u との類似度", "—", "コサイン 0.365（層15）、0.464（層19）"],
        ["上位 64 列の重なり", "—", "30/64（層15）、24/64（層19）"],
        ["検証の規模", "選択に使っていない 1,152 問、ノルムを合わせた直交ランダム対照（対照の上昇は +1.56 ポイント、標準誤差 3.52）、α=1・2・4 の用量試験", "64〜128 問、ノルムを合わせていないランダム対照（変化 約 2%、方向を使った編集は 12〜14%）"],
    ])
    t4 = table(["指標", "medium（本検証）", "hard（本検証）", "medium の公開値"], [
        ["全問の正答率（micro）", P(JM["micro_acc"]), P(JH["micro_acc"]), "86.01%"],
        ["状況ごとの平均（state-macro）", P(JM["state_macro_acc"]), P(JH["state_macro_acc"]), "87.57%"],
        ["ECE", f"{JM['ece']:.3f}", f"{JH['ece']:.3f}", "0.0400"],
        ["noul / choice / score", f"{P(JM['by_type']['noul'])} / {P(JM['by_type']['choice'])} / {P(JM['by_type']['score'])}",
         f"{P(JH['by_type']['noul'])} / {P(JH['by_type']['choice'])} / {P(JH['by_type']['score'])}", "—"],
        ["所要時間", f"{JM['seconds']} 秒", f"{JH['seconds']} 秒", "—"],
    ])
    t5 = table(["対策", "内容", "導入"], [
        ["WSL2 のメモリ上限", "<code>%USERPROFILE%\\.wslconfig</code> に <code>memory=14GB</code>、<code>swap=0</code>。戻すには、このファイルを削除して <code>wsl --shutdown</code>。", "依頼者の承認を得て実施"],
        ["コンテナのメモリ上限", "12 GB、スワップなし（<code>scripts/drun.sh</code>）", "Claude が追加"],
        ["実行前の点検", "ホストの空きメモリが 8 GB 未満、または GPU 使用量が 2,500 MiB 超なら起動しない", "Claude が追加"],
        ["資源の記録", "5 秒ごとに CSV へ書き、毎回 fsync（<code>scripts/monitor.py</code>）", "Claude が追加"],
        ["見張り", "空きメモリが 1.5 GB 未満、または GPU 温度が 85 ℃超ならコンテナを止める（<code>scripts/guarded_jev.sh</code>）", "Claude が追加"],
        ["GPU メモリの制限", "評価プロセスの PyTorch が確保する量を GPU 全体の 80% までにする", "Claude が追加"],
        ["4bit 読み込みと再開", "8bit から 4bit に変更。状況ごとに途中経過を保存し、中断しても再開できる", "依頼者の承認を得て実施"],
    ])

    body = f"""
<main class="paper">
<header>
<p class="kicker">検証報告　·　2026-11-04</p>
<h1>既存モデルの重み編集で、特定の用途を改善できるか</h1>
<p class="sub">Qwen3.5-0.8B・Gemma 4 E2B・Jev-Omni での検証</p>
<p class="by">実施（実装・実行・集計・執筆）: Claude　／　依頼と方針の決定: 依頼者<br>
本稿の数値は、すべて <code>results/</code> の結果ファイルから <code>scripts/build_report.py</code> が読み込んで作った。</p>
</header>

<section class="abs">
<h2>要旨</h2>
<p>元ブログの実験8は、モデルの重みの一部を直接書き換えて、特定の癖を直す方法を示している。本検証では、この方法が別のモデルや別の用途にも通用するかを調べた。対象は Qwen3.5-0.8B、Gemma 4 E2B、Jev-Omni の 3 つである。</p>
<p>Qwen3.5-0.8B では、元リポジトリが公開している編集（公開編集）を、sha256 が一致する形で復元した。公開編集は、思考モードで応答が繰り返しに陥る問題を減らし、設定選択用の 128 問で正答率を {P(SWEEP[0]['ok'] / SWEEP[0]['n'])} から {P(SWEEP[-1]['ok'] / SWEEP[-1]['n'])} に上げた（p = {base_pub:.3f}）。Claude が自動で作った編集（自前編集）も繰り返しを減らしたが、選択に使っていない GSM8K 64 問での正答率の上昇（{P(GSM[0]['ok'] / GSM[0]['n'])} → {P(GSM[2]['ok'] / GSM[2]['n'])}）は、有意と言える水準に届かなかった（p = {g_own:.3f}）。Gemma 4 E2B は繰り返しで終わる応答が 0 件で、編集の対象となる障害がなかった。Jev-Omni では、Yes/No 質問で No に偏る傾向を補正すると medium で正答率が 9.4 ポイント上がったが、未使用の hard では 0.8 ポイントにとどまり、重み編集には進まなかった。</p>
<p class="kw">キーワード: 重み編集、MLP、応答の繰り返し、Qwen3.5、Gemma 4、Jev-Omni、DecisionBench</p>
</section>

<p class="tcap"><b>表1　主要な結果の一覧。</b>差の確かさは Fisher の正確確率検定（両側）の p 値。標本が小さいため、p が 0.05 を超えるものは差があるとは言えない。</p>
{key_table()}

<h2><span class="sn">1</span>はじめに</h2>
<h3>1.1　背景と問い</h3>
<p>元ブログ [1] は、実験8で Qwen3.5-0.8B の MLP の出力行列を直接書き換えた。元リポジトリ [2] の記録によれば、書き換えは層15と層19の各 64 列から、応答が繰り返しに入る方向の成分を引く操作である。本検証では、次の 3 つの問いを立てた。</p>
<ol>
<li><b>問1</b>　公開編集は、Claude の環境と評価条件でも効果を示すか。</li>
<li><b>問2</b>　Claude が自動で作った編集（自前編集）でも、同様の効果が得られるか。</li>
<li><b>問3</b>　Gemma 4 E2B と Jev-Omni にも、編集で直せる障害があるか。</li>
</ol>
<h3>1.2　作業の分担</h3>
<p>依頼者が目的、対象モデル、方針を決めた。Claude が実装、実行、集計、本稿の執筆を行った。以下、特に断らない限り、実装・実行・集計の主語は Claude である。</p>
<h3>1.3　用語</h3>
<p>「元ブログ」は依頼者が示したブログ [1]、「元リポジトリ」はそのブログに対応するリポジトリ [2] を指す。元の資料にない呼称は、本稿で付けたものとして表2に区別して示す。</p>
<p class="tcap"><b>表2　用語。</b>「元資料」は元ブログ・元リポジトリ・各モデルの公開物にある名前、「本稿」は本検証で付けた呼称である。</p>
{terms}

<h2><span class="sn">2</span>方法</h2>
<h3>2.1　重みの編集</h3>
<p>編集の対象は MLP の <code>down_proj</code> の重み行列 W（Qwen3.5-0.8B では 1024 × 3584）である。W の 1 列は MLP のニューロン 1 個分の出力側の重みに当たる。選んだ列の集合 S の各列 w を、方向 u の成分を引いた形に置き換える（図1）。</p>
<p style="text-align:center;font-family:var(--serif)">W′[:, S] = W[:, S] − α · u (uᵀ W[:, S])</p>
{f1}
<h3>2.2　実行環境</h3>
<p>計算機は RTX 5070 Ti（16 GB）を載せた Windows 11 である。Qwen と Gemma の生成は、Docker 上の vLLM 0.32.0（transformers 5.19.0、torch 2.13+cu130）で行った。Jev-Omni の評価は、同じコンテナ内で transformers と bitsandbytes（4bit、NF4）を使った。計算機側の torch は CPU 版のため、GPU を使う処理はすべてコンテナ内で実行した。</p>
<h3>2.3　Qwen の評価条件</h3>
<ul>
<li>思考モードで生成した。温度 0.6、top_p 0.95、top_k 20、presence penalty 1.5。</li>
<li>質問文の後ろに「Please reason step by step, and put your final answer within \\boxed{{}}.」を付けた。</li>
<li>最後の <code>\\boxed{{}}</code> の中身を math-verify 0.9.0 で照合して採点した。</li>
<li>生成の上限は 8,192 トークン（方向探索用の生成だけ 16,384）、1 問につき 1 回。元ブログの評価条件と同一ではない。</li>
<li>問題は、元リポジトリの練習用ファイル（訓練分割のみ）から、乱数の種 1234 で並べ替えて作った。方向探索用は GSM8K 128 問と MATH 128 問、設定選択用は次の 256 問のうち GSM8K 64 問と MATH 64 問である。GSM8K（テスト）と MATH-500 の各 64 問は、固定の乱数の種で抜き出し、全モデルで同じ問題を使った。</li>
<li>差の確かさは Fisher の正確確率検定（両側）、信頼区間は Wilson の 95% 区間で示した。</li>
</ul>
<h3>2.4　自前編集の作り方</h3>
<ol>
<li>方向探索用の 256 問で、基底モデルに 16,384 トークンまで生成させた。</li>
<li>打ち切られた応答の末尾 1,024 トークンのうち、繰り返しに入っている部分を「繰り返しの文脈」とした。同じ応答の 256〜1,280 トークン目と、正しく完了した応答の思考の中ほどを「進んでいる文脈」とした。</li>
<li>層15と層19で、2 つの文脈の MLP 出力の平均の差を u とした。</li>
<li>各ニューロン j に、(u · w<sub>j</sub>) × (繰り返しの文脈での平均活性 − 進んでいる文脈での平均活性) という点数を付け、上位 k 列を編集した。</li>
<li>層15と層19という選び方は元ブログから借りた。Claude は層を探索していない。</li>
</ol>
<h3>2.5　Jev-Omni の評価</h3>
<p>Jev-Omni は、Gemma4Unified の 12B のバックボーンの最後の隠れ状態（3,840 次元）を正規化し、256 出力の線形ヘッドに通して、先頭の n 個の選択肢で softmax を取る。Claude は、公開されているローダーの推論手順を再実装した。質問型から選択肢を作る規則は公開物に記載がないため、Claude が推測した規則（組み立て方 v1）を使った。バックボーンは 4bit（NF4）に量子化し、画像と音声は使わず、文字だけで評価した。そのため、公開値とは直接比較できない。</p>

<h2><span class="sn">3</span>結果</h2>
<h3>3.1　公開編集の復元（問1）</h3>
<p>Claude は、基底モデルと公開編集の差分（<code>edit.npz</code>）から編集後の重みを組み立てた。基底と編集後の各テンソルの sha256 が、元リポジトリの記録と一致した。構造は、ランク1、α = 6.00、層15と層19の各 64 列で、変更された値は 130,188 個（全体の 0.0163%）である。差分から復元した u の層15と層19のコサイン類似度は 0.849 だった。</p>
<p>基底モデルを方向探索用の 256 問（上限 16,384 トークン）で測ると、正答率は {P(QB['acc'])}（{QB['ok']}/{QB['total']}）、打ち切りは {P(QB['cut_share'])}、うち繰り返しで終わった応答は {QB['cut_with_loop_tail']} 件、思考を閉じた割合は {P(QB['think_closed_share'])} だった。</p>
<h3>3.2　α と k を変えた自前編集（問2）</h3>
<p>設定選択用の 128 問で、α と k を変えた自前編集を比べた（図2、図3）。</p>
<ul>
<li>α = 1（成分を取り除くだけ）では、正答率は基底とほぼ同じだった（p = {fisher(SWEEP[0]['ok'], SWEEP[0]['n'], SWEEP[1]['ok'], SWEEP[1]['n']):.3f}）。</li>
<li>α = 1〜6 では、α が大きいほど繰り返しで終わった応答が減った（68 → 20 件）。α = 12 では 26 件とやや戻る一方、思考を閉じた割合は 56% まで上がった。</li>
<li>正答率の差は 128 問の標準誤差（3〜4 ポイント）に近く、自前編集の設定どうしの優劣は決められない。Claude は、正答率が最も高かった k=64, α=12 を最終候補にした。</li>
</ul>
{f2}
{f3}
<h3>3.3　選択に使っていない問題での比較（問1・問2）</h3>
<p>図4に、GSM8K（テスト）と MATH-500 の各 64 問の結果を示す。</p>
<ul>
<li>GSM8K では、基底モデルとの差の p 値は、公開編集が {g_pub:.3f}、自前 α=12 が {g_own:.3f} だった。5% 水準で有意なものはない。</li>
<li>繰り返しで終わった応答は、基底モデルの 42 件から、公開編集で 4 件、自前 α=12 で 20 件に減った。</li>
<li>ランダム対照は正答率も繰り返しの件数も基底と変わらなかったが、ノルムを合わせていないため、方向に意味があることの確認としては弱い。</li>
<li>MATH-500 では、基底モデルの応答の 95% が上限で打ち切られ、3 つのモデルの差は判断できなかった。</li>
</ul>
{f4}
<p class="tcap"><b>表3　公開編集と自前編集の違い。</b>自前編集は、式は同じだが、方向の求め方、強さ、検証の規模が異なる。</p>
{t3}
<h3>3.4　Gemma 4 E2B（問3）</h3>
<p>方向探索用の 256 問で、思考あり、上限 6,144 トークン、温度 1.0、top_p 0.95、top_k 64 で測った（図5）。上限を 6,144 トークンにしたのは、起動時の KV キャッシュ不足を避けるためである。誤りの内訳は、GSM8K が誤答 12 問、MATH が上限での打ち切り 14 問、<code>\\boxed{{}}</code> なしで完了 6 問、誤答 7 問だった。繰り返しで終わった応答がなく、取り除ける障害が見つからなかったため、Gemma 4 E2B の編集は行っていない。</p>
{f5}
<h3>3.5　Jev-Omni（問3）</h3>
<p>組み立て方 v1、4bit で、medium と hard を各 293 問測った（図6、表4）。medium の全問正答率は {P(JM['micro_acc'])} で、公開値の 86.01% より約 3 ポイント低い。hard は {P(JH['micro_acc'])} だった。</p>
{f6}
<p class="tcap"><b>表4　Jev-Omni の結果と公開値。</b>hard の公開値は、本稿の作成時には確認していない。</p>
{t4}
<p>medium の誤答 50 件のうち、noul が 21 件で、そのうち 17 件は「正解が Yes なのに No と答えた」ものだった。noul の予測は No が 75 件、Yes が 53 件で、正解（No が 62 件、Yes が 66 件）より No に偏っていた。score の誤答 15 件のうち 13 件は、正解から 1 段階ずれたものだった。誤答の確信度は平均 0.60 で、正答の 0.886 より低い。</p>
<p>この偏りを直す案として、noul の Yes − No のログオッズに一定の補正量を足す方法を調べた。medium で補正量を変えると正答率が上がったが、同じデータで調整して測ると過大評価になる。そこで、補正量を medium で効いた範囲（+0.5〜+1.0）の中ほどの +0.75 に、hard を見る前に固定し、hard を未使用の確認用データにした（図7）。</p>
{f7}
<p>hard の noul（128 問）では、+0.75 の補正で正解が 1 問増えただけだった。誤りは Yes→No が {CM['hard'][(0, 1)]} 件、No→Yes が {CM['hard'][(1, 0)]} 件で、medium（Yes→No が {CM['medium'][(0, 1)]} 件、No→Yes が {CM['medium'][(1, 0)]} 件）と違って両方向にある。hard では、正解が Yes の質問と No の質問のログオッズの重なりが大きく、補正量を足すだけでは分けきれない（図8）。事前に決めた基準は「hard でも上がれば重み編集に進む」だったため、ヘッドの重み編集は行っていない。</p>
{f8}

<h2><span class="sn">4</span>考察</h2>
<ol>
<li><b>繰り返しを減らす効果は再現した。</b>公開編集では繰り返しで終わった応答が 84 件から 4 件（設定選択用 128 問）、42 件から 4 件（GSM8K 64 問）に減った。自前編集でも、α を上げるとこの件数が減った。この効果は、正答率の差よりも安定して見える。</li>
<li><b>正答率の上昇は、公開編集で確認でき、自前編集では傾向にとどまった。</b>公開編集は設定選択用の 128 問で有意（p = {base_pub:.3f}）、GSM8K 64 問で有意に近い（p = {g_pub:.3f}）。自前編集は GSM8K 64 問で p = {g_own:.3f} であり、傾向はあるが有意とは言えない。</li>
<li><b>取り除くだけでは足りなかった。</b>α = 1 では繰り返しで終わった応答が 84 件から 68 件にしか減らず、正答率も変わらなかった。強く逆向きに押す α の領域で効果が出た。ただし、これは観察であり、仕組みの説明ではない。</li>
<li><b>自前編集は、同じ α でも公開編集より弱かった。</b>α = 6 で、繰り返しで終わった応答は公開編集で 4 件、自前編集で 20 件だった。α = 12 にすると行列の相対変化は公開編集の 2 倍から 3 倍以上になる。方向の求め方が元リポジトリの手順と違うことが一因と Claude は推測しているが、検証していない。</li>
<li><b>Gemma 4 E2B には、編集で直せる障害がなかった。</b>基底モデルの正答率が 84.8% と高く、繰り返しで終わった応答が 0 件である。</li>
<li><b>同じデータで調整して測ると、効果を過大に見積もる。</b>Jev-Omni の補正は、調整に使った medium で +9.4 ポイント、未使用の hard で +0.8 ポイントだった。調整と測定に同じデータを使うと、効果を実際より大きく見積もる。</li>
</ol>

<h2><span class="sn">5</span>限界</h2>
<ul>
<li>標本が小さい。64 問では、正答率の標準誤差が約 5 ポイントある。</li>
<li>生成は 1 問につき 1 回で、温度が 0 より大きいため、同じ設定でも結果は揺れる。</li>
<li>Qwen の評価は上限が 8,192 トークンで、MATH-500 では応答の大半が打ち切られ、判断できない。</li>
<li>自前編集の方向の求め方、層の選び方、ランダム対照は、元リポジトリの手順と同じ水準で検証していない。</li>
<li>Jev-Omni の結果には、4bit 化と、推測した選択肢の組み立て規則が含まれる。公開値とは直接比較できない。</li>
<li>Jev-Omni の補正の検証は noul だけで、choice と score への影響は測っていない。</li>
</ul>

<h2><span class="sn">6</span>結論と今後</h2>
<p>問1と問2: 公開編集は Claude の環境でも繰り返しを減らして正答率を上げた。自前編集は繰り返しを減らしたが、正答率の上昇は有意に届かなかった。問3: Gemma 4 E2B には編集の対象がなく、Jev-Omni の Yes 側への補正は未使用データで通用しなかった。</p>
<p>未実施の作業は次のとおりである。</p>
<ul>
<li>Qwen: 元リポジトリの手順に近づけた方向の作り直し、ノルムを合わせたランダム対照、32,768 トークンなど長い上限での MATH-500 の測定、α の再検討。</li>
<li>Jev-Omni: 組み立て方 v2 の測定、8bit での全件測定。</li>
<li>PC: Windows メモリ診断（再起動が必要なため実行していない）。</li>
</ul>

<h2><span class="sn">付録A</span>実行環境の問題と対策</h2>
<p>2026-11-04 に PC が 2 回、正常終了せずに再起動した。05:38 にバグチェック 0x133（DPC_WATCHDOG_VIOLATION）、06:10 にバグチェック 0x20001（HYPERVISOR_ERROR）である。どちらも、Jev-Omni を 8bit で実行している最中か、読み込みの直後だった。依頼者によると、再起動のとき、BIOS 画面のような表示にメモリの番号らしきものが出ていた。原因は特定できていない。Qwen の vLLM 実行は、数時間続けても問題が起きなかった。ハードウェア、ドライバー、Hyper-V のどれが原因かは調べていない。</p>
<p class="tcap"><b>表5　追加した対策。</b>承認を得て実施したものと、プロジェクト内のスクリプトとして Claude が追加したものがある。</p>
{t5}
<p>対策後の 4bit 実行（medium と hard）は、異常終了も見張りの作動もなく完了した（<code>watchdog.log</code> は作られていない）。ホストの空きメモリが最小になった時刻が、モデルの読み込み中か推論中かは確認していない。Jev-Omni の 8bit での全件実行は完走していない。</p>
{f9}

<h2><span class="sn">付録B</span>参考資料</h2>
<ol class="refs">
<li>元ブログ: <a href="https://rikiyat.github.io/blog/weight-space-programming/" target="_blank" rel="noopener">https://rikiyat.github.io/blog/weight-space-programming/</a></li>
<li>元リポジトリ（ブログに対応）: <code>lm_understand-logs</code>。ローカルの <code>external/lm_understand-logs</code> に複製し、読み取りのみ行った。リポジトリの URL は記録していない。</li>
<li>Qwen/Qwen3.5-0.8B（リビジョン 2fc06364715b967f1860aea9cf38778875588b17）。</li>
<li>Gemma 4 E2B（<code>gemma-4-E2B-it</code>、ローカルの <code>models/</code> に保存）。</li>
<li>Jev-Omni（モデル名 <code>akhilaaa3/jev-omni</code>）と、同梱の DecisionBench データ。</li>
<li>vLLM 0.32.0、transformers 5.19.0、torch 2.13+cu130、bitsandbytes、math-verify 0.9.0。</li>
</ol>
</main>
<div id="tip" hidden role="status"></div>
"""
    head = ('<title>重み編集の検証</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="preconnect" href="https://fonts.googleapis.com">\n<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&family=Noto+Serif+JP:wght@700&display=swap">\n')
    return head + f"<style>{css}</style>\n" + body + f"<script>{JS}</script>\n"


def main():
    frag = build()
    open(sys.argv[1], "w", encoding="utf-8").write(frag)
    if len(sys.argv) > 2:
        head, rest = frag.split("</style>", 1)
        doc = ('<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
               + head.replace("<style>", "<style>html{color-scheme:light}body{margin:0;font-size:14px}") + "</style></head><body>" + rest + "</body></html>")
        open(sys.argv[2], "w", encoding="utf-8").write(doc)
    print("ok", len(frag), "bytes")


if __name__ == "__main__":
    main()
