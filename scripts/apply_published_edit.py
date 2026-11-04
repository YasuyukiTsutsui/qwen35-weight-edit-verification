"""Rebuild an edited checkpoint from a base safetensors file and a published edit (edit.npz + manifest.json).

Own re-implementation of the logic of lm_understand-logs/exp8_math/code/apply_edit.py (sparse entries and low-rank
factors on top of the base bfloat16 bits), checked against the manifest's sha256 of every base and edited tensor.

    python scripts/apply_published_edit.py <base.safetensors> <edit dir> <out.safetensors>
"""
import hashlib
import json
import os
import sys

import numpy as np
from safetensors.numpy import load_file, save_file


def bf16_bits(x32):
    u = np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def read_bf16_bits(path):
    """safetensors.numpy cannot hold bfloat16; read raw headers and return {name: uint16 bits} for BF16 tensors."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        off = 8 + n
        out = {}
        for name, info in head.items():
            if name == "__metadata__":
                continue
            a, b = info["data_offsets"]
            f.seek(off + a)
            out[name] = (info["dtype"], np.frombuffer(f.read(b - a), dtype=np.uint8).copy(), info["shape"])
    return out


def main(base_path, edit_dir, out_path):
    man = json.load(open(os.path.join(edit_dir, "manifest.json")))
    z = np.load(os.path.join(edit_dir, "edit.npz"), allow_pickle=False)
    raw = read_bf16_bits(base_path)
    for i, t in enumerate(man["tensors"]):
        dt, buf, shape = raw[t["name"]]
        assert dt == "BF16" and list(shape) == t["shape"], (t["name"], dt, shape)
        bits = buf.view(np.uint16).reshape(shape)
        assert hashlib.sha256(bits.tobytes()).hexdigest() == t["base_sha256"], f"{t['name']}: base differs"
        if t["form"] == "lowrank":
            w = bf16_to_f32(bits).astype(np.float64)
            a, b = z[f"t{i}_a"], z[f"t{i}_b"]
            for k in range(a.shape[1]):
                w = w + np.outer(a[:, k], b[k])
            e = bf16_bits(w.astype(np.float32)).reshape(-1)
        else:
            e = bits.copy().reshape(-1)
        e[z[f"t{i}_idx"]] = z[f"t{i}_val"]
        e = e.reshape(shape)
        assert hashlib.sha256(e.tobytes()).hexdigest() == t["edited_sha256"], f"{t['name']}: rebuilt hash mismatch"
        raw[t["name"]] = ("BF16", e.view(np.uint8).reshape(-1), shape)
        print(f"{t['name']}: {t['form']}, {t['changed_entries']:,} entries changed, hash OK")
    write_raw(out_path, raw)
    print("->", out_path)


def write_raw(path, raw):
    import struct
    head, off, blobs = {}, 0, []
    for name in sorted(raw):
        dt, buf, shape = raw[name]
        b = buf.tobytes()
        head[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    head["__metadata__"] = {"format": "pt"}
    h = json.dumps(head, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


if __name__ == "__main__":
    main(*sys.argv[1:4])
