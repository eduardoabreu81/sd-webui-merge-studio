"""Reading a dtype off a collection of tensors, and casting to match it.

Kept free of torch and of the Forge backend so the rules can be exercised
directly: everything here works off ``.dtype``, ``.numel()`` and ``.to()``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _element_count(tensor: Any) -> int:
    numel = getattr(tensor, "numel", None)
    if callable(numel):
        try:
            return int(numel())
        except Exception:
            pass
    shape = getattr(tensor, "shape", None)
    if shape:
        total = 1
        for dim in shape:
            total *= int(dim)
        return total
    return 1


def dominant_float_dtype(state_dict: Mapping[str, Any]) -> Any | None:
    """The floating dtype that holds most of a state dict's weight.

    Weighted by element count, which is how backend/utils.py::weight_dtype --
    Forge's own answer to this question -- defines it. Counting tensors instead
    lets a quantized checkpoint's scales outvote its actual weights: in
    Anima-2.9B-preview-v1_int8_convrot the scales are 640 F32 tensors totalling
    1.8M elements, against 288 BF16 tensors totalling 153.3M. By tensor count
    that reads as F32; by element count, as BF16 -- which is the precision the
    model's unquantized weights are actually in.

    Integer/quantized payloads are excluded, so what remains is the plain
    floating precision the component is carried in.
    """
    totals: dict[Any, int] = {}
    for value in state_dict.values():
        dtype = getattr(value, "dtype", None)
        if dtype is None or not getattr(dtype, "is_floating_point", False):
            continue
        totals[dtype] = totals.get(dtype, 0) + _element_count(value)
    return max(totals, key=totals.get) if totals else None


def match_dtype(tensor: Any, dtype: Any) -> Any:
    """Casts a floating tensor to dtype, leaving anything else untouched so a
    quantized or integer payload is never silently reinterpreted."""
    if dtype is None or not hasattr(tensor, "dtype"):
        return tensor
    if not getattr(tensor.dtype, "is_floating_point", False) or tensor.dtype == dtype:
        return tensor
    try:
        return tensor.to(dtype)
    except Exception:
        return tensor
