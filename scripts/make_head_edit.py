"""Scale the output of single attention heads:  o_proj.weight[:, h*256:(h+1)*256] *= s   (s = 0 removes the head, s = 2 doubles it).

    python scripts/make_head_edit.py --base models/Qwen3.5-0.8B --edits 15:1:0,15:3:2 --out models/edits/wo_example
`--edits` is a comma-separated list of layer:head:scale.  Heads are the 8 query heads of the full-attention layers (head_dim 256).
"""
import argparse
import json
import os
import re
import shutil

from safetensors.torch import load_file, save_file

HD = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--edits", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    edits = [(int(l), int(h), float(s)) for l, h, s in (e.split(":") for e in a.edits.split(","))]
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.base):
        p = os.path.join(a.base, f)
        if os.path.isfile(p) and not f.endswith(".safetensors"):
            shutil.copyfile(p, os.path.join(a.out, f))
    log = {"base": a.base, "edits": [{"layer": l, "head": h, "scale": s} for l, h, s in edits], "tensors": {}}
    for sh in sorted(f for f in os.listdir(a.base) if f.endswith(".safetensors")):
        sd = load_file(os.path.join(a.base, sh))
        for L in sorted({l for l, _, _ in edits}):
            ks = [k for k in sd if re.search(rf"layers\.{L}\.self_attn\.o_proj\.weight$", k) and not re.search(r"vision|audio|visual|mtp", k)]
            if not ks:
                continue
            assert len(ks) == 1, ks
            W = sd[ks[0]].double()
            for l, h, s in edits:
                if l == L:
                    W[:, h * HD:(h + 1) * HD] *= s
            new = W.to(sd[ks[0]].dtype)
            log["tensors"][ks[0]] = {"rel_change": float((new.double() - sd[ks[0]].double()).norm() / sd[ks[0]].double().norm())}
            sd[ks[0]] = new.contiguous()
        save_file(sd, os.path.join(a.out, sh), metadata={"format": "pt"})
    assert len(log["tensors"]) == len({l for l, _, _ in edits}), log["tensors"]
    json.dump(log, open(os.path.join(a.out, "edit_log.json"), "w"), indent=1)
    print(json.dumps(log["tensors"]))


if __name__ == "__main__":
    main()
