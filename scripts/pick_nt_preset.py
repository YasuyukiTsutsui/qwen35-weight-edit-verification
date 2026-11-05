"""Issue #31 rule (fixed before the sweep): among the model-card non-thinking sampler and the four swept ones, take the one with the highest
accuracy on the 128 select problems (2 samples, seed 2); a tie goes to the model card.   python -I scripts/pick_nt_preset.py -> results/nt_best_preset.txt
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDS = [("qwen35_nothink", "bl_nothink"), ("qwen35_nt_T07_p0", "bl_nts_T07_p0"), ("qwen35_nt_T07_p15", "bl_nts_T07_p15"),
         ("qwen35_nt_T10_p0", "bl_nts_T10_p0"), ("qwen35_nt_T10_p15", "bl_nts_T10_p15")]
best = None
for preset, tag in CANDS:
    s = json.load(open(os.path.join(ROOT, "results", tag, "practice_select128", "summary.json")))
    print(f"{preset:22s} {100 * s['acc']:.1f}%  ({s['ok']}/{s['total']})")
    if best is None or s["ok"] > best[1]:     # strict '>' keeps the earlier one (the card is first) on ties
        best = (preset, s["ok"])
open(os.path.join(ROOT, "results", "nt_best_preset.txt"), "w").write(best[0] + "\n")
print("best:", best[0])
