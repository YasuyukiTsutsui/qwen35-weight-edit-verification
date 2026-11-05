#!/usr/bin/env python3
"""Build the HTML version of reports/report.md.

    python -I scripts/build_report.py <fragment_out.html> [<standalone_out.html>]

reports/report.md is the single source of the text. The two figures of section 3.11 are drawn from
results/final_analysis/final_fresh/analysis.json (the pre-registered analysis), at the places marked
<!--fig:final_acc--> and <!--fig:final_mech--> in the Markdown. The look comes from scripts/report_assets/.

fragment_out: page content for the Artifact tool (it adds the document skeleton).
standalone_out: the same page wrapped in a full HTML document, for opening from disk.
"""
import html
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "scripts", "report_assets")
TITLE = "重み編集の検証"
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">\n<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&family=Noto+Serif+JP:wght@700&display=swap">')


# ------------------------------------------------------------------ inline Markdown
def inline(s):
    codes = []

    def keep(m):
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"
    s = re.sub(r"`([^`]+)`", keep, s)
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', s)
    s = re.sub(r'(?<!["=>])(https?://[^\s<)、。）]+)', r'<a href="\1" target="_blank" rel="noopener">\1</a>', s)
    return re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{html.escape(codes[int(m.group(1))], quote=False)}</code>", s)


def render_table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], cells[2:]
    th = "".join(f"<th>{inline(c)}</th>" for c in head)
    tb = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table></div>'


def render_list(items):
    """items: list of (indent, ordered, text). One or two levels."""
    out, stack = [], []   # stack of (indent, tag)
    for ind, ordered, text in items:
        tag = "ol" if ordered else "ul"
        while stack and ind < stack[-1][0]:
            out.append(f"</li></{stack.pop()[1]}>")
        if stack and ind == stack[-1][0]:
            out.append("</li>")
        elif not stack or ind > stack[-1][0]:
            out.append(f"<{tag}>")
            stack.append((ind, tag))
        out.append(f"<li>{inline(text)}")
    while stack:
        out.append(f"</li></{stack.pop()[1]}>")
    return "".join(out)


def md_to_html(md, figs):
    lines = md.split("\n")
    out, i = [], 0
    in_toc = []
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append(f'<pre class="cb">{html.escape(chr(10).join(lines[i + 1:j]), quote=False)}</pre>')
            i = j + 1
        elif m := re.match(r"<!--fig:(\w+)-->", ln):
            out.append(figs[m.group(1)])
            i += 1
        elif ln.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append(render_table(lines[i:j]))
            i = j
        elif re.match(r"\s*(- |\d+\. )", ln):
            items, j = [], i
            while j < len(lines) and re.match(r"\s*(- |\d+\. )", lines[j]):
                m = re.match(r"(\s*)(- |\d+\. )(.*)", lines[j])
                items.append((len(m.group(1)), m.group(2) != "- ", m.group(3)))
                j += 1
            out.append(render_list(items))
            i = j
        elif ln.startswith("# "):
            i += 1
        elif m := re.match(r"(#{2,3}) (\d+(?:\.\d+)?)?[.\s]*(.*)", ln):
            lvl, num, text = len(m.group(1)), m.group(2), m.group(3)
            sn = f'<span class="sn">{num}</span>' if num else ""
            hid = "s" + (num or re.sub(r"\W+", "", text)).replace(".", "-")
            if lvl == 2:
                in_toc.append((hid, (num + " " if num else "") + text))
            out.append(f'<h{lvl} id="{hid}">{sn}{inline(text)}</h{lvl}>')
            i += 1
        elif ln.strip() == "":
            i += 1
        else:
            j = i
            while j < len(lines) and lines[j].strip() and not re.match(r"(#|\||```|\s*- |\s*\d+\. |<!--)", lines[j]):
                j += 1
            out.append(f"<p>{'<br>'.join(inline(x) for x in lines[i:j])}</p>")
            i = j
    return "\n".join(out), in_toc


# ------------------------------------------------------------------ figures from the pre-registered analysis
COL = {"base": "var(--c-base)", "pub": "var(--c-pub)", "own": "var(--c-own)", "null": "var(--c-rand)"}
ROWS = [("ff16k_B", "方法B（α6, k256）", "own"), ("ff16k_p50c", "公開編集", "pub"), ("ff16k_LF", "ラベルなし選択（α5.5, k256）", "own"),
        ("ff16k_A", "方法A（AdamW）", "own"), ("ff16k_own_a6", "自前 α6（旧方向, k64）", "own"), ("ff16k_mp_L15_19", "対比方向（層15+19）", "own"),
        ("ff16k_null", "ヌル対照の最良（ランダム方向）", "null"), ("ff16k_base", "基底モデル", "base")]
LEGEND = ('<div class="leg"><span><i class="sw" style="background:var(--c-base)"></i>基底</span><span><i class="sw" style="background:var(--c-pub)"></i>公開編集</span>'
          '<span><i class="sw" style="background:var(--c-own)"></i>本検証で作った編集</span><span><i class="sw" style="background:var(--c-rand)"></i>ヌル対照</span></div>')


def pct(x):
    return f"{100 * x:.1f}%"


def bar_cell(est, lo, hi, color, amax, step, tip):
    s = 100 / amax
    return (f'<div class="cell"><div class="trk" style="--step:{100 * step / amax:.3f}%" data-tip="{html.escape(tip)}" tabindex="0">'
            f'<div class="bar" style="width:{est * 100 * s:.2f}%;background:{color}"></div>'
            f'<div class="ci" style="left:{lo * 100 * s:.2f}%;width:{(hi - lo) * 100 * s:.2f}%"></div>'
            f'<span class="val" style="left:{hi * 100 * s + 0.5:.2f}%">{pct(est)}</span></div></div>')


def count_cell(n, total, color, tip):
    return (f'<div class="cell"><div class="trk" style="--step:50%" data-tip="{html.escape(tip)}" tabindex="0">'
            f'<div class="bar" style="width:{100 * n / total:.2f}%;background:{color}"></div>'
            f'<span class="val" style="left:{100 * n / total:.2f}%">{n}</span></div></div>')


def axis(vals, amax, unit=""):
    sp = "".join(f'<span style="left:{100 * v / amax:.2f}%">{v}</span>' for v in vals)
    return f'<div class="cell"><div class="trk0">{sp}</div></div>'


def make_figs():
    a = json.load(open(os.path.join(ROOT, "results", "final_analysis", "final_fresh", "analysis.json"), encoding="utf-8"))
    runs = a["runs"]
    n = 256
    # figure A: accuracy with 95% interval
    fa = ['<figure id="fig-final"><div class="rc">', LEGEND,
          '<div class="rr hd eb"><div>候補</div><div>正答率（%）と95%区間（新しい256問、上限16,384）</div></div>']
    for tag, label, kind in ROWS:
        m = runs[tag]["metrics"]["acc"]
        fa.append(f'<div class="rr eb"><div class="rl">{label}</div>' + bar_cell(m["est"], m["ci95"][0], m["ci95"][1], COL[kind], 70, 10,
                  f"{pct(m['est'])}||{label}：95%区間 {pct(m['ci95'][0])}〜{pct(m['ci95'][1])}") + "</div>")
    fa.append('<div class="rr ax eb"><div></div>' + axis(range(0, 71, 10), 70) + "</div></div>")
    fa.append("<figcaption><b>図　最終確認の正答率。</b>新しい256問、1問1回、上限16,384トークン。ひげは問題単位のクラスタ・ブートストラップによる95%区間（補正なし）。"
              "基底との差は、ヌル対照以外の6本で補正後も有意（Holm補正後 p = 0.001）、ヌル対照は区別できない（p = 0.664）。"
              "上位4本（方法B・公開編集・ラベルなし選択・方法A）の区間は大きく重なり、順位は主張しない。</figcaption></figure>")
    # figure B: budget accuracy at 8,192 and verbatim loops
    fb = ['<figure id="fig-mech"><div class="rc">', LEGEND,
          '<div class="rr hd"><div>候補</div><div>8,192 トークンまでに正解した割合（%）</div><div>末尾が周期のある繰り返しで終わった件数（/256）</div></div>']
    for tag, label, kind in ROWS:
        m = runs[tag]["metrics"]["budget_8192"]
        c = runs[tag]["counts"]["cut_verbatim_loop"]
        fb.append(f'<div class="rr"><div class="rl">{label}</div>'
                  + bar_cell(m["est"], m["ci95"][0], m["ci95"][1], COL[kind], 50, 10, f"{pct(m['est'])}||{label}：8,192トークン以内に完走して正解")
                  + count_cell(c, 256, COL[kind], f"{c} 件||{label}：上限で打ち切られ、末尾が完全一致の繰り返しになっていた応答") + "</div>")
    fb.append('<div class="rr ax"><div></div>' + axis(range(0, 51, 10), 50) + axis([0, 128, 256], 256) + "</div></div>")
    fb.append("<figcaption><b>図　早く終わるか（左）、周期のある繰り返しに入るか（右）。</b>左は「正解 かつ 完走 かつ 8,192 トークン以内」の割合。"
              "公開編集は早く正解に届き、方法B・ラベルなし選択は16,384で追いつく。この差は検定していない。"
              "右は、基底とヌル対照で200件を超える周期のある繰り返しが、編集後は3〜79件に減ることを示す。"
              "残る打ち切りの多くは繰り返しのない「考え続け」で、編集で直ったのは繰り返しである。</figcaption></figure>")
    return {"final_acc": "".join(fa), "final_mech": "".join(fb)}


# ------------------------------------------------------------------ page
def build():
    md = open(os.path.join(ROOT, "reports", "report.md"), encoding="utf-8").read()
    lines = md.split("\n")
    h1 = lines[0][2:].strip()
    j = 1
    meta = []
    while j < len(lines) and not lines[j].startswith("## "):
        if lines[j].strip():
            meta.append(lines[j])
        j += 1
    body, toc = md_to_html("\n".join(lines[j:]), make_figs())
    nav = '<nav class="toc" aria-label="目次"><b>目次</b>' + " ".join(f'<a href="#{h}">{html.escape(t)}</a>' for h, t in toc) + "</nav>"
    head = (f'<header><p class="kicker">検証報告　·　更新 2026-11-05</p><h1>{inline(h1)}</h1>'
            f'<p class="by">{"<br>".join(inline(m) for m in meta)}</p></header>')
    css = open(os.path.join(ASSETS, "style.css"), encoding="utf-8").read()
    extra = ("\n.toc{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:.84rem;line-height:1.7;margin:14px 0 0;padding:10px 0;border-bottom:1px solid var(--grid)}"
             ".toc b{color:var(--ink2)}.toc a{color:var(--accent);text-decoration:none;white-space:nowrap}.toc a:hover{text-decoration:underline}"
             "\n.paper pre.cb{background:var(--band);padding:10px 14px;border-radius:6px;overflow-x:auto;font:500 .86rem/1.7 ui-monospace,SFMono-Regular,Consolas,monospace;margin:.8rem 0}"
             "\n.paper h2[id],.paper h3[id]{scroll-margin-top:12px}.paper h3 .sn{color:var(--accent);margin-right:.5em;font-family:var(--sans);font-weight:500}")
    js = open(os.path.join(ASSETS, "tip.js"), encoding="utf-8").read()
    frag = (f'<title>{TITLE}</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n{FONTS}\n<style>\n{css}{extra}\n</style>\n'
            f'<main class="paper">\n{head}\n{nav}\n{body}\n</main>\n<div id="tip" role="tooltip" hidden></div>\n<script>\n{js}</script>\n')
    return frag


def main():
    frag = build()
    open(sys.argv[1], "w", encoding="utf-8").write(frag)
    if len(sys.argv) > 2:
        page = ('<!doctype html>\n<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
                '<style>:root{color-scheme:light}body{margin:0}</style></head><body>\n' + frag + "</body></html>\n")
        open(sys.argv[2], "w", encoding="utf-8").write(page)
    print(f"wrote {sys.argv[1]} ({len(frag)} bytes)")


if __name__ == "__main__":
    main()
