"""Name every checkpoint in a folder, from its safetensors header alone.

A command line over `architecture_guess`, which is where the reasoning lives.
This file is a driver and holds no detection logic of its own -- an earlier
version carried its own copy and the copy went stale, reporting every Anima
and the FP8 Krea2 as unrecognised long after the module had stopped doing so.

    # inside Forge, the package is importable already
    python tools/guess_architecture.py "D:/Base Models"
    # outside, point at a copy of modules_forge/packages/
    python tools/guess_architecture.py <folder> --guess-dir <path>
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture_guess import guess_architecture, install_torch_stub  # noqa: E402


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(length))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument(
        "--guess-dir", help="folder containing huggingface_guess/ (outside Forge)"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    install_torch_stub()
    if args.guess_dir:
        sys.path.insert(0, args.guess_dir)

    results = []
    for filename in sorted(os.listdir(args.folder)):
        if not filename.endswith(".safetensors"):
            continue
        path = os.path.join(args.folder, filename)
        try:
            guess = guess_architecture(read_header(path), fallback="Diffusion Model")
        except Exception as exc:  # unreadable file, not an unrecognised model
            results.append({"file": filename, "error": f"{type(exc).__name__}: {exc}"})
            continue
        results.append(
            {
                "file": filename,
                "label": guess.label,
                "architecture": guess.detector,
                "repo": guess.repo,
                "config": guess.config,
                "error": guess.error,
            }
        )

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        return 0

    for r in results:
        print(r["file"])
        if r.get("architecture"):
            print(f"    {r['architecture']}  ({r['repo']})")
            if r["config"]:
                print(f"    {r['config']}")
        else:
            print(f"    unrecognised — {r['error']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
