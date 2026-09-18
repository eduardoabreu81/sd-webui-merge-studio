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


# Tensors that carry quantization scaling rather than weights. Casting one of
# these to fp16 silently wrecks the component it belongs to, and no test
# downstream would catch it, so they are excluded from every cast -- including
# the explicit formats that are already refused for scaled components.
_SCALE_SUFFIXES = ("weight_scale", "weight_scale_2", "scale_weight", "_scale")

#: The output formats a component may be converted to. Anything stored with
#: scales is `same` only: converting it would mean dequantizing, which this
#: path does not do.
PLAIN_COMPONENT_FORMATS = ("same", "fp16", "bf16", "fp32")
_FORMAT_DTYPE_CODES = {"fp16": "F16", "bf16": "BF16", "fp32": "F32"}


def _is_scale_tensor(key: str) -> bool:
    return any(key.endswith(suffix) for suffix in _SCALE_SUFFIXES)


def _castable(tensor, dtype_map, target_code=None):
    """Whether this tensor may be converted at all."""
    current = getattr(tensor, "dtype", None)
    if current is None or not getattr(current, "is_floating_point", False):
        return None
    target = dtype_map.get(target_code) if target_code else None
    if target_code and target is None:
        return None
    return target if target_code else current


def apply_component_precision(
    state_dict: MutableMapping[str, Any],
    component,
    slot,
    dtype_map: Mapping[str, Any],
) -> int:
    """Set one component's precision, scoped to its own slot and its own file.

    Scoping is what makes this correct rather than merely convenient. Two
    encoders in one state dict have keys that reduce to the same suffix, so a
    global index would cast one to the other's dtype. It also keeps the Anima
    LLM Adapter out of reach: Forge's `process_anima` moves it into the text
    encoder's bucket at load, but it came from Model A and belongs to the
    diffusion model, and it does not sit under the encoder's namespace.

    `same` reads the dtype recorded in the component's own header -- never
    Model A's by proxy, and never the dtype Forge happened to materialise.
    A component stored with scales is copied untouched, since converting it
    would mean dequantizing.

    Returns the number of tensors changed.
    """
    prefixes = tuple(getattr(slot, "internal_prefixes", ()) or ())
    if not prefixes:
        return 0

    keys = [
        key
        for key in list(state_dict)
        if key.startswith(prefixes) and not _is_scale_tensor(key)
    ]
    if not keys:
        return 0

    fmt = getattr(component, "output_format", "same")
    if fmt != "same":
        target_code = _FORMAT_DTYPE_CODES.get(fmt)
        if target_code is None:
            return 0
        changed = 0
        for key in keys:
            target = _castable(state_dict[key], dtype_map, target_code)
            if target is None or state_dict[key].dtype == target:
                continue
            state_dict[key] = state_dict[key].to(target)
            changed += 1
        return changed

    source_path = getattr(component, "path", None)
    if not source_path:
        return 0
    try:
        header, _ = read_safetensors_header(source_path)
    except Exception:
        # A source that cannot be read is a reason to keep the dtypes the
        # tensors already carry, never a reason to guess at them.
        return 0

    # Only this component's own keys, matched by name with its namespace
    # removed from the state-dict side. The source file is standalone, so its
    # keys carry no such prefix; an embedded component's source is the
    # checkpoint itself, whose keys carry the architecture's outer prefix too.
    by_suffix: dict[str, Any] = {}
    ambiguous: set[str] = set()
    for key, info in header.items():
        if key == "__metadata__" or not isinstance(info, dict):
            continue
        if _is_scale_tensor(key):
            continue
        suffix = key
        for prefix in prefixes:
            marker = prefix.rstrip(".")
            if marker and marker in key:
                suffix = key.split(marker, 1)[1].lstrip(".")
                break
        if suffix in by_suffix:
            ambiguous.add(suffix)
        by_suffix[suffix] = info
    for name in ambiguous:
        del by_suffix[name]

    changed = 0
    for key in keys:
        suffix = key
        for prefix in prefixes:
            if key.startswith(prefix):
                suffix = key[len(prefix):]
                break
        info = header.get(key)
        if not isinstance(info, dict):
            info = by_suffix.get(suffix)
        if not isinstance(info, dict):
            continue
        target = _castable(state_dict[key], dtype_map, info.get("dtype"))
        if target is None or state_dict[key].dtype == target:
            continue
        state_dict[key] = state_dict[key].to(target)
        changed += 1
    return changed


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
