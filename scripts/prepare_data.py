"""Download evaluation / practice data into data/*.jsonl (id, problem, answer).

    .venv/Scripts/python scripts/prepare_data.py
"""
import json
import os

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(ROOT, exist_ok=True)


def dump(name, rows):
    with open(os.path.join(ROOT, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows))


def parquet_rows(repo, pattern):
    files = [f for f in list_repo_files(repo, repo_type="dataset") if pattern in f and f.endswith(".parquet")]
    rows = []
    for f in sorted(files):
        rows += pq.read_table(hf_hub_download(repo, f, repo_type="dataset")).to_pylist()
    return rows


# MATH-500 (test, held out from everything we tune on)
p = hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset")
dump("math500", [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"],
                  "level": r.get("level"), "subject": r.get("subject")}
                 for r in map(json.loads, open(p, encoding="utf-8"))])

# GSM8K test (held out) and train (for direction finding)
for split in ("test", "train"):
    rows = parquet_rows("openai/gsm8k", f"main/{split}-")
    dump(f"gsm8k_{split}", [{"id": f"gsm8k_{split}_{i}", "problem": r["question"],
                             "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
                            for i, r in enumerate(rows)])
