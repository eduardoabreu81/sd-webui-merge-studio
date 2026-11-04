"""Merge the parameters that are not a module's `weight` or `bias`.

The merge walks modules by their `.weight`, which leaves out anything a model
stores under another name. Krea 2 has 159 of them: `.scale` on its RMSNorm
layers and `.lin` on its modulation. Measured on a Weighted Sum at 0.5, every
weight and bias sat at the midpoint of the two models and every one of these
was still Model A's.

Runs after the weights, so a mode that draws random numbers keeps the sequence
it has always used for them, and the same seed still gives the same weights.
"""

from __future__ import annotations

from typing import Callable

from torch import nn

from .merge_modes import blend_tensors

_HANDLED_BY_THE_MODULE_WALK = ("weight", "bias")


def merge_loose_parameters(
    named_a: dict[str, nn.Module],
    named_b: dict[str, nn.Module],
    named_c: dict[str, nn.Module],
    interp_method: str,
    beta: float,
    ratio_for: Callable[[str], float],
    translate_name: Callable[[str], str | None] | None = None,
    skip_module: Callable[[nn.Module], bool] | None = None,
    target_device=None,
    rand=None,
) -> tuple[int, list[str]]:
    """Blends each loose parameter of A with its counterpart, in place.

    `ratio_for` gives the multiplier for a module path, so per-block rules
    apply here as they do to the weights. A module `translate_name` maps to
    None has no counterpart, such as a block a newer generation inserted, and
    keeps A's values. `skip_module` is for quantized modules, whose extra
    parameters describe the encoding rather than the model.

    Returns (merged_count, skipped): a counterpart that is missing or has
    another shape is reported as skipped, and A's value stays.
    """
    merged = 0
    skipped: list[str] = []
    for name, module in named_a.items():
        if skip_module is not None and skip_module(module):
            continue
        for param_name, param in list(module.named_parameters(recurse=False)):
            if param_name in _HANDLED_BY_THE_MODULE_WALK or not param.is_floating_point():
                continue
            label = f"{name}.{param_name}" if name else param_name

            lookup = translate_name(name) if translate_name is not None else name
            if lookup is None:
                continue
            other_b = getattr(named_b.get(lookup), param_name, None)
            if other_b is None or other_b.shape != param.shape:
                skipped.append(label)
                continue
            other_c = getattr(named_c.get(lookup), param_name, None)

            dev = target_device if target_device is not None else param.device
            result = blend_tensors(
                interp_method,
                ratio_for(name),
                beta,
                param.data.float().to(device=dev),
                other_b.data.float().to(device=dev),
                other_c.data.float().to(device=dev) if other_c is not None and other_c.shape == param.shape else None,
                xp=_torch(),
                rand=rand,
            )
            if result is None:
                skipped.append(label)
                continue
            setattr(module, param_name, nn.Parameter(result.to(dtype=param.dtype, device=dev), requires_grad=False))
            merged += 1
    return merged, skipped


def _torch():
    import torch

    return torch
