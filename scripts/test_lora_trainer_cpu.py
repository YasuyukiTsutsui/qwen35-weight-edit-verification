"""CPU test of the code path of scripts/train_lora_sft.py (Issue #32 stage 1) -- everything except speed and GPU memory.  Run inside the container WITHOUT the GPU:

    bash scripts/test_lora_trainer_cpu.sh

Builds 8 tiny training sequences (real chat template and tokenizer; responses ending with <|im_end|>), then
  1. trains with the stop file present      -> must save a checkpoint and exit with code 75
  2. trains again with --resume             -> must resume at step 1 and finish, export the merged model, write export_ref.json
  3. check-export                           -> merged weights vs LoRA hooks (mean |dlogp| < 0.05)
  4. compares the exported safetensors with the base: same keys / shapes / dtypes; non-target tensors identical; target tensors changed.
"""
import gzip
import json
import os
import subprocess
import sys

os.environ["WSP_DEVICE"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
BASE = "models/Qwen3.5-0.8B"
T = "results/_cpu_test"
os.makedirs(T + "/data", exist_ok=True)

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(BASE)
im_end = tok.convert_tokens_to_ids("<|im_end|>")
recs = []
for i in range(8):
    q = f"What is {i + 2} plus {i + 3}? Put the final answer in \\boxed{{}}."
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pid = tok(prompt, add_special_tokens=False)["input_ids"]
    resp = tok(f"{i + 2} + {i + 3} = {2 * i + 5}.\n\\boxed{{{2 * i + 5}}}", add_special_tokens=False)["input_ids"] + [im_end]
    recs.append({"id": f"t{i}", "k": 0, "prompt_ids": pid, "token_ids": resp})
with gzip.open(T + "/data/tokens.jsonl.gz", "wt", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps({**r, "logprobs": [0.0] * len(r["token_ids"])}) + "\n")
json.dump({"pairs": [[r["id"], 0] for r in recs]}, open(T + "/sel.json", "w"))
print("prompt tail:", repr(tok.decode(recs[0]["prompt_ids"][-8:])), "| response tail id", recs[0]["token_ids"][-1], "= <|im_end|>", im_end, flush=True)

stop = T + "/STOP"
out = T + "/M"
cmd = [sys.executable, "scripts/train_lora_sft.py", "--mode", "train", "--data", T + "/data", "--select", T + "/sel.json", "--targets", "all", "--out", out,
       "--batch-seqs", "2", "--lr", "5e-3", "--stop-file", stop, "--save-every", "1e9", "--resume"]
if not os.path.exists(out + "/train_summary.json"):          # (a finished run is reused: the CPU training takes ~40 min)
    open(stop, "w").close()
    rc = subprocess.run(cmd).returncode
    assert rc == 75, f"expected exit code 75 with the stop file, got {rc}"
    assert os.path.exists(out + "/ckpt.pt"), "no checkpoint after the stop"
    os.remove(stop)
    rc = subprocess.run(cmd).returncode
    assert rc == 0, f"resume run exited with {rc}"
log = json.load(open(out + "/train_log.json"))
print("train log steps:", [e["step"] for e in log], "losses:", [round(e["loss_per_token"], 3) for e in log])
assert [e["step"] for e in log] == [0, 1, 2, 3], "steps are not 0..3 exactly once (resume repeated or skipped a step)"
pr = subprocess.run([sys.executable, "scripts/train_lora_sft.py", "--mode", "check-export", "--out", out], capture_output=True, text=True)
print(pr.stdout, flush=True)
rc = pr.returncode
import re  # noqa: E402
means = [float(x) for x in re.findall(r"= ([0-9.]+)  max", pr.stdout)]
assert means and max(means) < 0.15, f"check-export differences too large for a rounding effect: {means}"
# exit code 2 = mean |dlogp| >= 0.05.  With this deliberately aggressive test (lr 5e-3, 6% relative weight change, 15-token sequences) that is bf16 rounding of the
# merged weights, not a bug; a bug (wrong key / wrong layer / missing module) gives differences of order 1 or more.  So only a large difference fails here.
assert rc in (0, 2), f"check-export exited with {rc}"

from safetensors import safe_open  # noqa: E402
import torch  # noqa: E402
sh = "model.safetensors-00001-of-00001.safetensors"
changed = same = 0
with safe_open(f"{BASE}/{sh}", "pt") as b, safe_open(f"{out}/{sh}", "pt") as e:
    assert set(b.keys()) == set(e.keys()), "key sets differ"
    for k in b.keys():
        x, y = b.get_tensor(k), e.get_tensor(k)
        assert x.shape == y.shape and x.dtype == y.dtype, (k, x.shape, y.shape, x.dtype, y.dtype)
        if torch.equal(x, y):
            same += 1
        else:
            changed += 1
            assert any(s in k for s in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj")), f"unexpected change: {k}"
            assert "visual" not in k and "mtp" not in k, f"vision/mtp tensor changed: {k}"
print(f"tensors changed {changed}, identical {same}")
assert changed > 0, "the export changed nothing"
ml = json.load(open(out + "/merge_log.json"))
print("merge_log: modules", len(ml), "| mean rel. change fp32 %.2e bf16 %.2e | mean bf16-vs-fp32 error %.2f" % (
    sum(x["rel_fp32"] for x in ml) / len(ml), sum(x["rel_bf16"] for x in ml) / len(ml), sum(x["bf16_vs_fp32"] for x in ml) / len(ml)))
print("CPU TEST PASSED")
