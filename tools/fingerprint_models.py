"""Print a structural fingerprint of every checkpoint in a folder.

Run this over one model of each architecture to collect the evidence needed to
name them apart. The inspector currently labels most modern DiTs as the generic
"DiT / Diffusion Model" because nothing in the header has been mapped to a
specific architecture yet -- and guessing from a filename is what this is meant
to replace.

Reads safetensors headers only: no tensors are decoded, no model is loaded, and
torch is not required. A 6 GB checkpoint costs a few kilobytes of reading.

    python tools/fingerprint_models.py "D:/models/Stable-diffusion"
    python tools/fingerprint_models.py <folder> --json > fingerprints.json

Deliberately outside `scripts/`: Forge loads every `scripts/*.py` in an
extension as an extension script, and this is a standalone utility.
"""

from __future__ import annotations

import argparse
import json
import os

import struct
import sys
from collections import Counter

#: Namespaces stripped before naming a stack, so the same architecture reads
#: the same whichever prefix its publisher used.
_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "net.")


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError("file too short to be safetensors")
        length = struct.unpack("<Q", raw_len)[0]
        return json.loads(f.read(length)), 8 + length


def fingerprint(path: str) -> dict:
    header, _ = read_header(path)
    metadata = header.get("__metadata__") or {}
    keys = [k for k in header if k != "__metadata__"]

    # Which numbered stacks exist, and how deep each one is. This is the single
    # most discriminating fact: one stack, two parallel series, or three
    # sections tells the architecture family apart before anything else.
    # Any path segment that is a bare number ends a stack, and everything
    # before it names that stack. Finding them this way rather than by a list
    # of known names is what lets an unfamiliar architecture describe itself --
    # which is the whole point of running this.
    stacks: dict[str, int] = {}
    for key in keys:
        name = key
        for prefix in _PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        segments = name.split(".")
        for i, segment in enumerate(segments):
            if segment.isdigit() and i:
                stack = ".".join(segments[:i])
                stacks[stack] = max(stacks.get(stack, -1), int(segment))
                break
    stacks = {name: count + 1 for name, count in sorted(stacks.items())}

    # Top-level namespaces, which say where components were stored.
    roots = Counter(k.split(".", 1)[0] for k in keys)

    # A few shapes that differ between architectures of the same family.
    def shape_of(*fragments):
        for fragment in fragments:
            for key in keys:
                if fragment in key:
                    entry = header.get(key) or {}
                    return {"key": key, "shape": entry.get("shape"), "dtype": entry.get("dtype")}
        return None

    dtypes = Counter(
        (header[k] or {}).get("dtype", "?") for k in keys if isinstance(header.get(k), dict)
    )

    return {
        "file": os.path.basename(path),
        "size_mb": round(os.path.getsize(path) / (1024 * 1024)),
        "tensors": len(keys),
        "stacks": stacks,
        "roots": dict(roots.most_common(12)),
        "dtypes": dict(dtypes),
        "has_comfy_quant": any(k.endswith(".comfy_quant") for k in keys),
        "metadata_keys": sorted(metadata)[:20],
        "samples": {
            "first_block_qkv": shape_of("blocks.0.self_attn.q_proj.weight", "blocks.0.attn.qkv.weight"),
            "mlp": shape_of("blocks.0.mlp.layer1.weight", "blocks.0.mlp.fc1.weight", "blocks.0.ff.net.0.proj.weight"),
            "patch_embed": shape_of("x_embedder", "patch_embed", "img_in", "input_blocks.0.0.weight"),
            "final_layer": shape_of("final_layer", "out.2.weight", "proj_out"),
        },
        # Cheap, distinctive, and free: the sorted first key of each root often
        # identifies a family on its own.
        "first_keys": sorted(keys)[:6],
    }


def describe(fp: dict) -> str:
    stacks = ", ".join(f"{n}={c}" for n, c in fp["stacks"].items()) or "(none)"
    lines = [
        f"  {fp['file']}",
        f"    {fp['size_mb']:,} MB, {fp['tensors']} tensors, dtypes {fp['dtypes']}",
        f"    stacks:  {stacks}",
        f"    roots:   {', '.join(fp['roots'])}",
    ]
    for name, sample in fp["samples"].items():
        if sample:
            lines.append(f"    {name:<16} {sample['key']}  {sample['shape']}")
    if fp["metadata_keys"]:
        lines.append(f"    metadata: {', '.join(fp['metadata_keys'])}")
    if fp["has_comfy_quant"]:
        lines.append("    quantized (comfy_quant)")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="folder holding .safetensors checkpoints")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--recursive", action="store_true", help="descend into subfolders")
    args = parser.parse_args()

    paths = []
    if args.recursive:
        for root, _, files in os.walk(args.folder):
            paths += [os.path.join(root, f) for f in files if f.endswith(".safetensors")]
    else:
        paths = [
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if f.endswith(".safetensors")
        ]

    results = []
    for path in sorted(paths):
        try:
            results.append(fingerprint(path))
        except Exception as exc:  # a broken file should not stop the sweep
            results.append({"file": os.path.basename(path), "error": str(exc)})

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        return 0

    print(f"{len(results)} checkpoint(s) in {args.folder}\n")
    for fp in results:
        print(describe(fp) if "error" not in fp else f"  {fp['file']}\n    ERROR: {fp['error']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
