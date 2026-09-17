"""Restore output tensor dtypes from a safetensors source header."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any

from checkpoint_inspector import read_safetensors_header


# A component is not stored under the same prefix by every release, and the read
# side is more permissive than the write side.
#
# Diffusion model: backend/detection.py::unet_prefix_from_state_dict accepts
# either prefix and strips whichever it finds, while model_list.py
# ::process_unet_state_dict_for_saving always writes "model.diffusion_model.".
# Measured on the official Anima releases: anima-preview, -preview2,
# -preview3-base and -base-v1.0 ship as "net.", -aesthetic-v1.0/v1.0b/v1.1 and
# -turbo-v1.0/v1.1 as "model.diffusion_model.", and every community expansion
# (Anima-2.9B, its int8_convrot build, Anima-3.8B and -3.8B-v1.1) as "net.".
#
# Autoencoder: same shape, in two model configs. Mugen accepts both prefixes
# (model_list.py:249) but saves through vae_key_prefix[0], which is "vae.";
# Chroma renames "first_stage_model." to "vae." on load (model_list.py:394) and
# also saves "vae.".
#
# Grouped on purpose. Stripping every prefix into one namespace would let a
# diffusion tensor and an autoencoder tensor with the same trailing name look
# like the same tensor, and a dtype would cross between components.
KEY_PREFIX_GROUPS = (
    ("model.diffusion_model.", "net."),
    ("vae.", "first_stage_model."),
)


def _grouped_key(key: str) -> tuple[int, str] | None:
    """``(component, name without its prefix)``, or None for an unprefixed key.

    A key that carries no prefix this module knows is matched by its full name
    only: there is nothing to say which component it belongs to.
    """
    for component, prefixes in enumerate(KEY_PREFIX_GROUPS):
        for prefix in prefixes:
            if key.startswith(prefix):
                return component, key[len(prefix) :]
    return None


def _index_by_grouped_key(header: Mapping[str, Any]) -> dict[tuple[int, str], Any]:
    """Maps each source tensor to its component and prefix-less name.

    A name that two source keys in the same component both reduce to is dropped
    rather than guessed at: without the prefix there is nothing left to tell
    them apart, and casting the wrong tensor is worse than casting none.
    """
    index: dict[tuple[int, str], Any] = {}
    ambiguous: set[tuple[int, str]] = set()
    for key, info in header.items():
        if key == "__metadata__" or not isinstance(info, dict):
            continue
        grouped = _grouped_key(key)
        if grouped is None:
            continue
        if grouped in index:
            ambiguous.add(grouped)
        index[grouped] = info
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

    Keys are matched exactly first, then by component and name with the prefix
    removed from both sides, so a source that shipped its diffusion model as
    ``net.`` still matches the ``model.diffusion_model.`` keys the save path
    produces. Only explicitly supported floating dtype codes are changed;
    quantized/integer payloads and keys absent from the source are left
    untouched.
    """
    header, _ = read_safetensors_header(source_path)
    by_grouped_key = _index_by_grouped_key(header)
    changed = 0
    for key in keys:
        tensor = state_dict.get(key)
        source_info = header.get(key)
        if not isinstance(source_info, dict):
            grouped = _grouped_key(key)
            source_info = by_grouped_key.get(grouped) if grouped else None
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
