"""Math grading: the last \\boxed{...} of the answer (after the last </think>), checked with math-verify.
Plus the loop signals used to classify failures (verbatim repetition / repetitive tail)."""
import re
from collections import Counter

_BOX = re.compile(r"\\(?:boxed|fbox)\s*\{")


THINK_END = ("</think>", "<channel|>")   # Qwen3.5 / Gemma 4


def think_closed(text):
    return any(m in text for m in THINK_END)


def split_think(text):
    """(thinking, answer); thinking is '' when no end marker is present."""
    for m in THINK_END:
        if m in text:
            i = text.rfind(m)
            return text[:i], text[i + len(m):]
    return "", text


def answer_text(text):
    return split_think(text)[1]


def last_boxed(text):
    ms = list(_BOX.finditer(text))
    if not ms:
        return None
    i = ms[-1].end()
    depth, j = 1, i
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i:j - 1].strip() if depth == 0 else None


def extract(text):
    return last_boxed(answer_text(text))


_cache = {}


def is_correct(pred, gold):
    if pred is None or gold is None:
        return False
    from math_verify import parse, verify
    try:
        g = _cache.get(gold) or _cache.setdefault(gold, parse("\\boxed{" + gold + "}"))
        p = parse("\\boxed{" + pred + "}")
        return bool(g and p and verify(g, p))
    except Exception:  # noqa: BLE001
        return False


def loop_signals(text, tail_words=200):
    """verbatim: the text ends with a block of 1..200 words repeated >=3 times (>=40 words);
    repetitive: in the last 200 words, >=50% of 4-gram positions belong to a 4-gram seen >=3 times."""
    w = text.split()[:-1]
    n = len(w)
    verbatim = False
    for p in range(1, 201):
        reps = max(3, -(-40 // p))
        if reps * p > n:
            break
        blk = w[n - p:]
        if all(w[n - (j + 1) * p:n - j * p] == blk for j in range(1, reps)):
            verbatim = True
            break
    t = w[-tail_words:]
    grams = [tuple(t[i:i + 4]) for i in range(max(0, len(t) - 3))]
    c = Counter(grams)
    rep = bool(grams) and sum(c[g] >= 3 for g in grams) >= 0.5 * len(grams)
    return {"verbatim_loop_end": verbatim, "repetitive_end": rep}
