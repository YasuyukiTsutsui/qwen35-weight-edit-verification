"""Unsealed TRAINING-problem set T1 (Issue #32, stage 1): 384 GSM8K-train + 384 MATH-train problems that were never evaluated (E1 included, because its samples are saved), are not in any sealed block (C1..C5)
and are not near-duplicates of any of those.  Deterministic, no model, no outcome used.  Run in the container (math-verify checks the gold answers):

    bash scripts/drun.sh python3 scripts/make_t1_set.py      -> data/t1.jsonl, data/t1_manifest.json

Procedure (fixed BEFORE the script was run; this file and SALT are committed first):
  1. 'Used' = every ID in data/practice_*.jsonl, data/final_fresh.jsonl and in any saved evaluation (results/**/samples.jsonl.gz);
     'sealed' = every ID in data/final2/manifest.json.  Their texts are looked up in all pools.
  2. Pools: GSM8K train and MATH train (the blog's practice files).  Order: sha256(SALT|ID) ascending.
  3. A problem is ACCEPTED unless it is used or sealed, its normalised text equals / has word-4-gram Jaccard >= 0.5 with a used, sealed or already
     accepted problem, or its gold answer does not parse with math-verify.  Rejections are counted in the manifest.
  4. 384 accepted per stratum.  Problem texts are not printed.
"""
import glob
import gzip
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_final2_blocks import POOLS, Index, read_jsonl, sha256_file  # noqa: E402

SALT = "wsp-t1-salt-2026-11-06-c41e8a"
JACCARD_MAX = 0.5
PER_STRATUM = 384


def gold_parses(ans):
    try:
        from math_verify import parse
        return bool(parse("\\boxed{" + str(ans) + "}"))
    except Exception:  # noqa: BLE001
        return False


def main():
    pools = {k: read_jsonl(POOLS[k][0]) for k in ("gsm8k_train", "math_train")}
    text_of = {}
    for k, v in POOLS.items():
        for r in read_jsonl(v[0]):
            text_of.setdefault(r["id"], r["problem"])
    for r in read_jsonl("data/gsm8k_train.jsonl"):
        text_of.setdefault(r["id"], r["problem"])
    used = set()
    for f in glob.glob("data/practice_*.jsonl") + ["data/final_fresh.jsonl"]:
        for r in read_jsonl(f):
            used.add(r["id"])
            text_of.setdefault(r["id"], r["problem"])
    for f in glob.glob("results/**/samples*.jsonl.gz", recursive=True):
        for l in gzip.open(f, "rt", encoding="utf-8"):
            used.add(json.loads(l)["id"])
    sealed = set()
    for b in json.load(open("data/final2/manifest.json"))["blocks"].values():
        sealed |= set(b["ids"])
    idx = Index()
    for i in sorted(used | sealed):
        if i in text_of:
            idx.add(("used" if i in used else "sealed", i), text_of[i])
    print(f"used {len(used)}, sealed {len(sealed)}, without text {sum(1 for i in used | sealed if i not in text_of)}", flush=True)
    accepted, rejected = {}, {"used": 0, "sealed": 0, "exact_text": 0, "near_duplicate": 0, "gold_unparsable": 0}
    for name in ("gsm8k_train", "math_train"):
        accepted[name] = []
        for r in sorted(pools[name], key=lambda r: hashlib.sha256((SALT + "|" + r["id"]).encode()).hexdigest()):
            if len(accepted[name]) >= PER_STRATUM:
                break
            if r["id"] in used:
                rejected["used"] += 1
                continue
            if r["id"] in sealed:
                rejected["sealed"] += 1
                continue
            j, key, exact = idx.worst(r["problem"])
            if exact or j >= JACCARD_MAX:
                rejected["exact_text" if exact else "near_duplicate"] += 1
                continue
            if not gold_parses(r["answer"]):
                rejected["gold_unparsable"] += 1
                continue
            accepted[name].append(r)
            idx.add(("accepted", r["id"]), r["problem"])
    with open("data/t1.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for name in ("gsm8k_train", "math_train"):
            for r in accepted[name]:
                f.write(json.dumps({"id": r["id"], "problem": r["problem"], "answer": str(r["answer"]), "stratum": POOLS[name][1], "origin": "train"},
                                   ensure_ascii=False) + "\n")
    man = {"script": "scripts/make_t1_set.py", "script_sha256": sha256_file(__file__), "salt": SALT, "jaccard_max": JACCARD_MAX, "per_stratum": PER_STRATUM,
           "rejected_while_scanning": rejected, "file": "data/t1.jsonl", "file_sha256": sha256_file("data/t1.jsonl"),
           "n": {k: len(v) for k, v in accepted.items()}, "ids": {k: [r["id"] for r in v] for k, v in accepted.items()}}
    json.dump(man, open("data/t1_manifest.json", "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
    print("accepted", man["n"], "rejected", rejected, "sha256", man["file_sha256"][:16])


if __name__ == "__main__":
    main()
