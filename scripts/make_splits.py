"""Practice splits (from the blog's practice files, train sets only): find / select, balanced GSM8K + MATH."""
import json, random
P = "external/lm_understand-logs/exp8_math/practice/"
def rows(f):
    return [{"id": r["id"], "problem": r["problem"], "answer": str(r["answer"])} for r in map(json.loads, open(P + f, encoding="utf-8"))]
g, m = rows("gsm8k_train.jsonl"), rows("math_train.jsonl")
rnd = random.Random(1234); rnd.shuffle(g); rnd.shuffle(m)
splits = {"find": g[:128] + m[:128], "select": g[128:256] + m[128:256]}
for k, v in splits.items():
    with open(f"data/practice_{k}.jsonl", "w", encoding="utf-8") as f:
        for r in v: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(k, len(v))
