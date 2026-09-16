"""Restore output tensor dtypes from a safetensors source header."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any

from checkpoint_inspector import read_safetensors_header


def match_source_dtypes(
    state_dict: MutableMapping[str, Any],
    source_path: str,
    keys: Iterable[str],
    dtype_map: Mapping[str, Any],
) -> int:
    """Cast selected floating tensors to the dtype recorded in ``source_path``.

    Only exact tensor-key matches and explicitly supported floating dtype codes
    are changed. Quantized/integer payloads and keys absent from the source are
    left untouched.
    """
    header, _ = read_safetensors_header(source_path)
    changed = 0
    for key in keys:
        tensor = state_dict.get(key)
        source_info = header.get(key)
        if tensor is None or not isinstance(source_info, dict):
            continue
        target_dtype = dtype_map.get(source_info.get("dtype"))
        current_dtype = getattr(tensor, "dtype", None)
        if target_dtype is None or current_dtype is None:
            continue
        if not getattr(current_dtype, "is_floating_point", False):
            continue
        if current_dtype == target_dtype:
            continue
        state_dict[key] = tensor.to(target_dtype)
        changed += 1
    return changed
