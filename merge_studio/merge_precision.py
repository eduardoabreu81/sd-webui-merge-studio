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

An F32 file -- an earlier FP32 merge, say -- is a different case: left to
itself Forge loads it as BF16 and the extra precision is gone before any of
this runs. float32_source_storage keeps it float32 through the load, for the
FP32 mode here and for the Quantize tab, which then quantizes from it.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Callable

import torch
from torch import nn

from .source_precision import diffusion_storage_code

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


def fp32_notes(base_quantized: bool, output_format: str, base_label: str, plain_formats, base_float32: bool = False) -> list[str]:
    """What an FP32 merge cannot do with this base and output, said up front.

    Notes rather than refusals: a quantized checkpoint is often all a user
    has, and FP32 still improves it -- the patch math runs in float32 instead
    of float16 and each layer is requantized once.

    base_float32: the base was loaded as float32 (an F32 file, see
    float32_source_storage), so 'Same as source' already saves F32.
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
    if output_format == "same" and not base_float32:
        return [
            f"Note: 'Same as source' saves the diffusion model back in {base_label}'s dtype, which rounds away what "
            "FP32 kept. Pick FP32 or a quantized format to keep it."
        ]
    return []


@contextmanager
def forced_unet_storage_dtype(dynamic_args, dtype: torch.dtype):
    """Makes Forge store the next diffusion model it loads in `dtype` until
    the block exits.

    backend/loader.py reads dynamic_args.forge_unet_storage_dtype (the
    "Diffusion in Low Bits" choice) on every load and falls back to
    memory_management.unet_dtype() when it is None. That fallback picks BF16
    on any GPU that handles it, F32 file or not, so an FP32 merge saved to
    disk came back BF16 the moment it was loaded again. The previous value is
    put back so generation keeps the user's own choice.
    """
    original = getattr(dynamic_args, "forge_unet_storage_dtype", None)
    dynamic_args.forge_unet_storage_dtype = dtype
    try:
        yield
    finally:
        dynamic_args.forge_unet_storage_dtype = original


def float32_source_storage(dynamic_args, checkpoint_path: str):
    """forced_unet_storage_dtype(float32) when the file stores its diffusion
    model in F32, a no-op otherwise. Loading a BF16 file as float32 adds
    nothing but memory; upcast_to_float32 does that later, where it is asked
    for."""
    if diffusion_storage_code(checkpoint_path) == "F32":
        return forced_unet_storage_dtype(dynamic_args, torch.float32)
    return nullcontext()


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
