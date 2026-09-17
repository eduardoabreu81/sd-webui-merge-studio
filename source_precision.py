"""Restore output tensor dtypes from a safetensors source header."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any

from checkpoint_inspector import read_safetensors_header


# The diffusion model is not stored under the same prefix by every release.
# backend/detection.py::unet_prefix_from_state_dict accepts either of these, and
# huggingface_guess strips whichever it finds on load; on save,
# model_list.py::process_unet_state_dict_for_saving always writes back
# "model.diffusion_model.". So a checkpoint that shipped as "net." is read under
# one name and written under another, and matching the full key string finds
# nothing. Measured on the official Anima releases: anima-preview, -preview2,
# -preview3-base and -base-v1.0 ship as "net.", while -aesthetic-v1.0/v1.0b/v1.1
# and -turbo-v1.0/v1.1 ship as "model.diffusion_model.".
DIFFUSION_KEY_PREFIXES = ("model.diffusion_model.", "net.")


def _strip_diffusion_prefix(key: str) -> str:
    for prefix in DIFFUSION_KEY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _index_by_stripped_key(header: Mapping[str, Any]) -> dict[str, Any]:
    """Maps each source tensor to its key without the diffusion-model prefix.

    A name that two different source keys both reduce to is dropped rather than
    guessed at: without a prefix there is nothing left to tell them apart, and
    casting the wrong tensor is worse than casting none.
    """
    index: dict[str, Any] = {}
    ambiguous: set[str] = set()
    for key, info in header.items():
        if key == "__metadata__" or not isinstance(info, dict):
            continue
        stripped = _strip_diffusion_prefix(key)
        if stripped == key:
            continue
        if stripped in index:
            ambiguous.add(stripped)
        index[stripped] = info
    for name in ambiguous:
        del index[name]
    return index


def match_source_dtypes(
    state_dict: MutableMapping[str, Any],
    source_path: str,
    keys: Iterable[str],
    dtype_map: Mapping[str, Any],
) -> int:
    """Cast selected floating tensors to the dtype recorded in ``source_path``.

    Keys are matched exactly first, then by name with the diffusion-model prefix
    removed from both sides, so a source that shipped as ``net.`` still matches
    the ``model.diffusion_model.`` keys the save path produces. Only explicitly
    supported floating dtype codes are changed; quantized/integer payloads and
    keys absent from the source are left untouched.
    """
    header, _ = read_safetensors_header(source_path)
    by_stripped_key = _index_by_stripped_key(header)
    changed = 0
    for key in keys:
        tensor = state_dict.get(key)
        source_info = header.get(key)
        if not isinstance(source_info, dict):
            source_info = by_stripped_key.get(_strip_diffusion_prefix(key))
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
