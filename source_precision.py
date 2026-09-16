"""Restore output tensor dtypes from a safetensors source header."""

from __future__ import annotations

import os
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


def try_match_source_dtypes(
    state_dict: MutableMapping[str, Any],
    source_path: str,
    keys: Iterable[str],
    dtype_map: Mapping[str, Any],
) -> tuple[int, str]:
    """``match_source_dtypes``, but a source with no readable safetensors header
    is reported instead of raised.

    A ``.ckpt``/``.pt`` primary model has no header to read precision back from,
    and neither does a file that moved mid-merge. That is a reason to leave the
    tensors at the precision they already carry -- never a reason to discard a
    merge that has already been fully computed.

    Returns ``(tensors_changed, warning)``; ``warning`` is empty on success.
    """
    try:
        return match_source_dtypes(state_dict, source_path, keys, dtype_map), ""
    except Exception as e:
        return 0, (
            f"Could not read the source precision of {os.path.basename(source_path)} ({e}). "
            f"Keeping the precision the tensors already have."
        )
