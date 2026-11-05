"""Seal the problem blocks for the next pre-registered confirmation (Issue #29). CPU only, deterministic, no model, no outcome used.

    python -I scripts/make_final2_blocks.py            -> data/final2/manifest.json  (committed)  and  data/final2/C{1..5}.jsonl  (local)

Procedure (fixed BEFORE the script was run; the commit that contains this file and SALT precedes the commit of the manifest):
  1. 'Used' problems = every ID in data/practice_*.jsonl, data/final_fresh.jsonl, and every ID that appears in any saved evaluation
     (results/**/samples.jsonl.gz).  Their texts are looked up in all pools.
  2. Pools: GSM8K test (data/gsm8k_test.jsonl), MATH-500 (data/math500.jsonl)  = test-origin;
            GSM8K train and MATH train from the blog's practice files (external/lm_understand-logs/exp8_math/practice/)  = train-origin.
  3. Within each pool, problems are ordered by sha256(SALT + '|' + ID) ascending (hex string). Problems are visited in that order and
     ACCEPTED unless they are used, or their normalised text equals, or has word-4-gram Jaccard similarity >= 0.5 with, a used problem
     or an already accepted problem (of any pool). Rejections are listed in the manifest with the reason and the score.
  4. Blocks are consecutive intervals of the accepted lists.  C1 = test-origin: n GSM8K-test + n MATH-500, n = min(308, number accepted
     in either pool).  C2..C5 = train-origin: 308 GSM8K-train + 308 MATH-train each (consecutive intervals).
  5. Nothing is evaluated here and the sealed problem texts are not printed.  Gold answers must parse with math-verify; that check is done
     in the evaluation container before any evaluation (reports/preregistration_final2_frame.md), and exclusions are recorded there.
"""
import glob
import gzip
import hashlib
import json
import os
import re
import sys

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SALT = "wsp-final2-salt-2026-11-05-a41c9e"
JACCARD_MAX = 0.5
BLOCK_PER_STRATUM = 308
TRAIN_BLOCKS = 4
PRACTICE = "external/lm_understand-logs/exp8_math/practice/"
POOLS = {  # name -> (file, stratum, origin)
    "gsm8k_test": ("data/gsm8k_test.jsonl", "GSM8K", "test"),
    "math500": ("data/math500.jsonl", "MATH", "test"),
    "gsm8k_train": (PRACTICE + "gsm8k_train.jsonl", "GSM8K", "train"),
    "math_train": (PRACTICE + "math_train.jsonl", "MATH", "train"),
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def shingles(text):
    w = words(text)
    if len(w) < 4:
        return frozenset([" ".join(w)])
    return frozenset(" ".join(w[i:i + 4]) for i in range(len(w) - 3))


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


class Index:
    """Inverted index word-4-gram -> items, to find the near duplicates of a problem fast."""

    def __init__(self):
        self.items, self.inv = [], {}

    def add(self, key, text):
        sh = shingles(text)
        n = len(self.items)
        self.items.append((key, sh, " ".join(words(text))))
        for s in sh:
            self.inv.setdefault(s, set()).add(n)

    def worst(self, text):
        sh, norm = shingles(text), " ".join(words(text))
        cand = set()
        for s in sh:
            cand |= self.inv.get(s, set())
        best = (0.0, None, False)
        for n in cand:
            key, sh2, norm2 = self.items[n]
            j = jaccard(sh, sh2)
            if j > best[0]:
                best = (j, key, False)
            if norm == norm2:
                return (1.0, key, True)
        return best


def main():
    pools = {k: read_jsonl(v[0]) for k, v in POOLS.items()}
    text_of = {}
    for rows in pools.values():
        for r in rows:
            text_of.setdefault(r["id"], r["problem"])
    for f in ["data/gsm8k_train.jsonl"]:
        for r in read_jsonl(f):
            text_of.setdefault(r["id"], r["problem"])
    used = set()
    for f in glob.glob("data/practice_*.jsonl") + ["data/final_fresh.jsonl"]:
        for r in read_jsonl(f):
            used.add(r["id"])
            text_of.setdefault(r["id"], r["problem"])
    n_files = 0
    for f in glob.glob("results/**/samples.jsonl.gz", recursive=True):
        n_files += 1
        for l in gzip.open(f, "rt", encoding="utf-8"):
            used.add(json.loads(l)["id"])
    missing = sorted(i for i in used if i not in text_of)
    idx = Index()
    for i in sorted(used):
        if i in text_of:
            idx.add(("used", i), text_of[i])
    print(f"used IDs {len(used)} (from {n_files} sample files), without text {len(missing)}", flush=True)

    accepted = {k: [] for k in POOLS}
    rejected = []
    order = {k: sorted(pools[k], key=lambda r: hashlib.sha256((SALT + "|" + r["id"]).encode()).hexdigest()) for k in POOLS}
    need = {"gsm8k_test": BLOCK_PER_STRATUM, "math500": BLOCK_PER_STRATUM,
            "gsm8k_train": BLOCK_PER_STRATUM * TRAIN_BLOCKS, "math_train": BLOCK_PER_STRATUM * TRAIN_BLOCKS}
    for name in ["math500", "gsm8k_test", "gsm8k_train", "math_train"]:   # scarce pools first (the order does not depend on any outcome)
        for r in order[name]:
            if len(accepted[name]) >= need[name]:
                break
            if r["id"] in used:
                rejected.append({"pool": name, "id": r["id"], "reason": "used"})
                continue
            j, key, exact = idx.worst(r["problem"])
            if exact or j >= JACCARD_MAX:
                rejected.append({"pool": name, "id": r["id"], "reason": "exact_text" if exact else "near_duplicate", "jaccard": round(j, 3), "with": list(key)})
                continue
            accepted[name].append(r)
            idx.add(("accepted", r["id"]), r["problem"])
    n1 = min(BLOCK_PER_STRATUM, len(accepted["gsm8k_test"]), len(accepted["math500"]))
    blocks = {"C1": [("gsm8k_test", r) for r in accepted["gsm8k_test"][:n1]] + [("math500", r) for r in accepted["math500"][:n1]]}
    for b in range(TRAIN_BLOCKS):
        lo, hi = b * BLOCK_PER_STRATUM, (b + 1) * BLOCK_PER_STRATUM
        blocks[f"C{b + 2}"] = [("gsm8k_train", r) for r in accepted["gsm8k_train"][lo:hi]] + [("math_train", r) for r in accepted["math_train"][lo:hi]]
    os.makedirs("data/final2", exist_ok=True)
    man = {"script": "scripts/make_final2_blocks.py", "script_sha256": sha256_file(__file__), "salt": SALT, "jaccard_max": JACCARD_MAX,
           "order": "sha256(SALT|ID) ascending within a pool; scarce pools first", "used_id_count": len(used), "used_without_text": missing,
           "sample_files_scanned": n_files, "sources": {k: {"file": v[0], "sha256": sha256_file(v[0]), "rows": len(pools[k]), "accepted": len(accepted[k])}
                                                       for k, v in POOLS.items()},
           "rejected": rejected, "blocks": {}}
    for bname, items in blocks.items():
        path = f"data/final2/{bname}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for name, r in items:
                f.write(json.dumps({"id": r["id"], "problem": r["problem"], "answer": str(r["answer"]), "stratum": POOLS[name][1], "origin": POOLS[name][2]},
                                   ensure_ascii=False) + "\n")
        man["blocks"][bname] = {"file": path, "sha256": sha256_file(path), "n": len(items), "origin": POOLS[items[0][0]][2],
                                "n_by_stratum": {s: sum(1 for n, _ in items if POOLS[n][1] == s) for s in ("GSM8K", "MATH")},
                                "ids": [r["id"] for _, r in items]}
    json.dump(man, open("data/final2/manifest.json", "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
    print("rejected:", len(rejected), {x: sum(1 for r in rejected if r["reason"] == x) for x in ("used", "exact_text", "near_duplicate")})
    for b, v in man["blocks"].items():
        print(b, v["n"], v["n_by_stratum"], v["origin"], v["sha256"][:16])


if __name__ == "__main__":
    main()
