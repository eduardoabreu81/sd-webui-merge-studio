"""Bake LoRAs in float32 instead of Forge's working precision.

Forge applies a LoRA the way it does at generation time: the patch is computed
in `memory_management.lora_compute_dtype(device)` -- float16 on any GPU that
handles it, float32 only on the CPU -- and the result is rounded back to the
weight's own dtype (backend/patcher/base.py::patch_weight_to_device). On a
BF16 checkpoint that rounding discards every change smaller than half a BF16
step, and those are exactly what distilled few-step LoRAs carry: lodestones'
MOSS-Image 8-step DMD README says its delta "is smaller than bf16's rounding
step". Quantizing afterwards cannot bring it back, and `set_module_weight`
used to round to BF16 once more before handing the weight to the quantizer.

Measured through `set_module_weight` on Forge Neo's backend, BF16-grid
2048x3072 weights plus a random delta, mean of 4 seeds. "kept" is the cosine
between the change that survives and the delta asked for; the reference is
quantized from the same input dtype.

    BF16 output alone (no quantization; plain torch rounding, 1024x3072), kept:
      delta 2^-6: 0.995    2^-8: 0.924    2^-10: 0.625    2^-12: 0.326

    quantized, kept via BF16 -> via FP32, and RMS error to base + delta:
      int8 convrot  2^-8: 0.364 -> 0.393   2^-10: 0.097 -> 0.182   rms 1.90e-4 -> 1.73e-4
      int8 tensor   2^-8: 0.354 -> 0.338   2^-10: 0.137 -> 0.169   rms 2.48e-4 -> 2.38e-4
      fp8 e4m3fn    2^-8: 0.290 -> 0.250   2^-10: 0.171 -> 0.125   rms 5.37e-4 -> 5.31e-4

FP32 input lowers the overall error in every format (ConvRot most, since its
Hadamard rotation then runs in float32 too); for FP8 the error is lower while
slightly less of a tiny delta survives.

The FP32 mode upcasts the diffusion model, forces the patch math to float32 on
any device and quantizes from the float32 result. It roughly doubles the DiT's
memory. An already-quantized base still works: its quantized layers stay as
they are, and Forge dequantizes each one, patches it in float32 and
requantizes it once (MixedPrecisionOps.Linear.set_weight). What its own
quantization already lost does not come back.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch
from torch import nn

MERGE_PRECISION_CHOICES = [
    ("Native (Forge's working precision)", "native"),
    ("FP32 (keeps small deltas such as turbo/DMD LoRAs -- about 2x RAM)", "fp32"),
]
MERGE_PRECISION_LABEL_TO_KEY = dict(MERGE_PRECISION_CHOICES)


def upcast_to_float32(root_module: nn.Module, skip: Callable[[torch.Tensor], bool] | None = None) -> int:
    """Casts every floating-point parameter under root_module to float32 in
    place and returns how many changed.

    Parameters only: the LoRA patches target weights, and buffers (RoPE
    tables and the like) are never saved from a patched weight.

    skip: parameters to leave alone. Pass one for quantized weights -- a
    QuantizedTensor reports its original dtype, so `.float()` would silently
    dequantize it, and Forge already patches those in the compute dtype.
    """
    changed = 0
    for module in root_module.modules():
        for param in module.parameters(recurse=False):
            if skip is not None and skip(param):
                continue
            if param.dtype.is_floating_point and param.dtype != torch.float32:
                param.data = param.data.float()
                changed += 1
    return changed


def fp32_notes(base_quantized: bool, output_format: str, base_label: str, plain_formats) -> list[str]:
    """What an FP32 merge cannot do with this base and output, said up front.

    Notes rather than refusals: a quantized checkpoint is often all a user
    has, and FP32 still improves it -- the patch math runs in float32 instead
    of float16 and each layer is requantized once.
    """
    if base_quantized:
        notes = [
            f"Note: {base_label} is already quantized. Its quantized layers are patched in float32 and requantized "
            "once, but what its own quantization lost does not come back -- a bf16 build keeps more."
        ]
        if output_format != "same" and output_format not in set(plain_formats):
            notes.append(
                f"Note: converting {base_label}'s quantized layers to {output_format} quantizes them a second time. "
                "'Same as source' keeps the single requantization."
            )
        return notes
    if output_format == "same":
        return [
            f"Note: 'Same as source' saves the diffusion model back in {base_label}'s dtype, which rounds away what "
            "FP32 kept. Pick FP32 or a quantized format to keep it."
        ]
    return []


@contextmanager
def forced_lora_compute_dtype(memory_management, dtype: torch.dtype):
    """Makes Forge compute LoRA patches in `dtype` on every device until the
    block exits. patch_weight_to_device looks the function up on the module
    each time, so swapping the attribute is enough."""
    original = memory_management.lora_compute_dtype
    memory_management.lora_compute_dtype = lambda device: dtype
    try:
        yield
    finally:
        memory_management.lora_compute_dtype = original
