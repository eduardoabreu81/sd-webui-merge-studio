"""Diagnose and repair .safetensors checkpoints whose `comfy_quant` blobs
(used by backend/operations_mixed_precision.py in Forge Neo) are missing the
required "format" field, which causes:

    ValueError: Unknown quantization format for layer <name>

Works by streaming (reads/writes in chunks) without loading the weight
tensors into RAM/GPU -- only the small comfy_quant blobs (a few dozen bytes
each) are decoded and rewritten.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from dataclasses import dataclass

logger = logging.getLogger("checkpoint_doctor")

CHUNK = 16 * 1024 * 1024  # 16MB


@dataclass
class BrokenLayer:
    key: str
    inferred_format: str | None
    raw_conf: dict


def _read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def _infer_format(prefix: str, quant_conf: dict, header: dict) -> str | None:
    """Infers the "format" that should have been in the comfy_quant blob,
    from the sibling weight's dtype and the auxiliary keys present in the
    blob itself."""
    weight_info = header.get(f"{prefix}.weight")
    weight_dtype = weight_info["dtype"] if weight_info else None

    if weight_dtype in ("F8_E4M3", "F8_E4M3FN"):
        return "float8_e4m3fn"
    if weight_dtype == "F8_E5M2":
        return "float8_e5m2"

    if weight_dtype in ("I8", "U8"):
        if f"{prefix}.weight_s_rel" in header:
            return "asym_w4a8_int8"
        if "linear_dtype" in quant_conf:
            return "convrot_w4a4"
        scale_info = header.get(f"{prefix}.weight_scale")
        if scale_info and scale_info["dtype"] in ("F8_E8M0FNU",):
            return "mxfp8"
        if f"{prefix}.weight_scale_2" in header:
            return "nvfp4"
        return "int8_tensorwise"

    return None


def diagnose(path: str) -> list[BrokenLayer]:
    """Scans only the file's header (fast, no GPU needed) and returns the
    layers whose comfy_quant blob is missing the "format" field."""
    header, data_start = _read_header(path)
    broken: list[BrokenLayer] = []

    with open(path, "rb") as f:
        for key, info in header.items():
            if key == "__metadata__" or not key.endswith(".comfy_quant"):
                continue
            off0, off1 = info["data_offsets"]
            f.seek(data_start + off0)
            raw = f.read(off1 - off0)
            try:
                conf = json.loads(raw)
            except Exception:
                broken.append(BrokenLayer(key, None, {"_error": "invalid JSON"}))
                continue
            if "format" not in conf:
                prefix = key[: -len(".comfy_quant")]
                inferred = _infer_format(prefix, conf, header)
                broken.append(BrokenLayer(key, inferred, conf))

    return broken


def repair(src: str, dst: str, progress_cb=None) -> dict:
    """Rewrites `src` into `dst` (streaming, without loading weights into
    RAM), injecting "format" into comfy_quant blobs that were missing it.
    `dst` must be a different path than `src` -- for in-place repair, the
    caller writes to a temp file and os.replace's it after validating."""
    header, data_start = _read_header(src)
    broken = diagnose(src)

    if not broken:
        return {"fixed": 0, "output": None, "formats": []}

    unresolved = [b.key for b in broken if b.inferred_format is None]
    if unresolved:
        raise ValueError(f"Could not infer the quantization format for {len(unresolved)} layer(s) " f"(e.g. {unresolved[0]}). Aborting rather than writing incorrect metadata.")

    fixes = {b.key: b.inferred_format for b in broken}

    tensor_items = [(k, v) for k, v in header.items() if k != "__metadata__"]
    tensor_items.sort(key=lambda kv: kv[1]["data_offsets"][0])

    new_header = {}
    if "__metadata__" in header:
        new_header["__metadata__"] = header["__metadata__"]

    patched_bytes: dict[str, bytes] = {}
    cursor = 0
    with open(src, "rb") as fin:
        for key, info in tensor_items:
            off0, off1 = info["data_offsets"]
            if key in fixes:
                fin.seek(data_start + off0)
                conf = json.loads(fin.read(off1 - off0))
                conf = {"format": fixes[key], **conf}
                new_bytes = json.dumps(conf).encode("utf-8")
                patched_bytes[key] = new_bytes
                length = len(new_bytes)
            else:
                length = off1 - off0

            new_header[key] = {
                "dtype": info["dtype"],
                "shape": [length] if key in fixes else info["shape"],
                "data_offsets": [cursor, cursor + length],
            }
            cursor += length

    header_bytes = json.dumps(new_header).encode("utf-8")
    total_bytes = cursor
    written = 0

    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(struct.pack("<Q", len(header_bytes)))
        fout.write(header_bytes)
        for key, info in tensor_items:
            if key in patched_bytes:
                fout.write(patched_bytes[key])
                written += len(patched_bytes[key])
            else:
                off0, off1 = info["data_offsets"]
                remaining = off1 - off0
                fin.seek(data_start + off0)
                while remaining > 0:
                    block = fin.read(min(CHUNK, remaining))
                    if not block:
                        raise IOError(f"Unexpected EOF while reading {key}")
                    fout.write(block)
                    remaining -= len(block)
                    written += len(block)
            if progress_cb:
                progress_cb(written, total_bytes)

    return {"fixed": len(fixes), "output": dst, "formats": sorted(set(fixes.values()))}


def repair_in_place(path: str, progress_cb=None) -> dict:
    """Repairs `path` by writing to a temp file in the same directory,
    validating it, and only then replacing the original (atomic
    os.replace). The original file is never touched until the new version
    is fully written and validated."""
    tmp = path + ".doctor_tmp"
    result = repair(path, tmp, progress_cb=progress_cb)
    if result["output"] is None:
        return result

    remaining = diagnose(tmp)
    if remaining:
        os.remove(tmp)
        raise RuntimeError("Post-repair validation failed: some layers are still missing 'format'. Original file was not touched.")

    os.replace(tmp, path)
    result["output"] = path
    return result
