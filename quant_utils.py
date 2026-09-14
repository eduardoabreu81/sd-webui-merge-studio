"""Shared helpers for dealing with MixedPrecisionOps-quantized modules
(backend/operations_mixed_precision.py) from outside the normal
patch_model()/state_dict() round trip -- used by lora_bake.py,
checkpoint_merge.py and checkpoint_quantize.py.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile

import torch
import torch.nn as nn
from safetensors.torch import save as safetensors_save
from safetensors.torch import save_file

import backend.args
from backend.operations_mixed_precision import _quantized_weight_state_dict
from backend.quant_ops import QUANT_ALGOS, QuantizedTensor

PLAIN_FORMATS = {"fp16": torch.float16, "bf16": torch.bfloat16}


class _Colors:
    """Same ANSI codes/layout as other Forge Neo extensions (e.g. CivitAI
    Browser Neo's Colors class) so [DEBUG] lines look consistent."""

    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


def debug_print(msg: str) -> None:
    print(f"{_Colors.MAGENTA}[DEBUG] {_Colors.BLUE}[Checkpoint Doctor]{_Colors.RESET} - {msg}", flush=True)


# "int8_tensorwise" is the on-disk format name (backend/quant_ops.py::QUANT_ALGOS).
# It can be written two ways: a single scalar scale for the whole tensor
# (comfy-kitchen's literal default), or per-channel scale + a Hadamard
# rotation ("convrot") -- the scheme most community Anima int8 builds
# actually use (confirmed by inspecting a real broken checkpoint: its
# comfy_quant blobs all carried convrot=true, convrot_groupsize=256).
# convrot needs per_channel=True (comfy_kitchen/tensor/int8.py enforces this).
QUANT_FORMAT_VARIANTS: dict[str, tuple[str, dict]] = {
    "int8_tensorwise": ("int8_tensorwise", {}),
    "int8_tensorwise_convrot": ("int8_tensorwise", {"per_channel": True, "convrot": True, "convrot_groupsize": 256}),
}

# (label, output_format key) pairs shared by every UI tab that offers an
# output-format choice.
OUTPUT_FORMAT_CHOICES = [
    ("Same as source checkpoint", "same"),
    ("FP16 (no quantization)", "fp16"),
    ("BF16 (no quantization)", "bf16"),
    ("FP8 (e4m3fn)", "float8_e4m3fn"),
    ("FP8 (e5m2)", "float8_e5m2"),
    ("INT8 (tensor-wise, single scale)", "int8_tensorwise"),
    ("INT8 (convrot: per-channel + rotation, recommended)", "int8_tensorwise_convrot"),
    ("NVFP4", "nvfp4"),
    ("INT4 (convrot W4A4)", "convrot_w4a4"),
]

# Reference heuristic from comfy-kitchen's own sample
# (samples/mxfp8_model_patcher.py::patch_model_with_mxfp8) for picking which
# layers to quantize when the base checkpoint isn't already quantized.
DEFAULT_MIN_FEATURES = 64
DEFAULT_SKIP_SUBSTRINGS = ("norm", "embed", "modulation")


class QuantConversionError(RuntimeError):
    pass


def is_quantized(module: nn.Module) -> bool:
    return isinstance(getattr(module, "weight", None), QuantizedTensor)


def weight_as_float(module: nn.Module) -> torch.Tensor | None:
    """Returns the module's current weight as a plain float32 tensor,
    dequantizing first if needed. Returns None if the module has no weight."""
    w = getattr(module, "weight", None)
    if w is None:
        return None
    if isinstance(w, QuantizedTensor):
        return w.dequantize().float()
    return w.data.float()


def _resolve_quant_format(target_format: str) -> tuple[str, dict]:
    """Returns (on-disk quant_format name, extra .quantize() kwargs) for an
    output-format key, which may be a plain QUANT_ALGOS key or a variant
    like 'int8_tensorwise_convrot'."""
    if target_format in QUANT_FORMAT_VARIANTS:
        return QUANT_FORMAT_VARIANTS[target_format]
    if target_format in QUANT_ALGOS:
        return target_format, {}
    raise QuantConversionError(f"Unknown target format: {target_format}")


def set_module_weight(module: nn.Module, float_weight: torch.Tensor, target_format: str | None = None) -> None:
    """Writes float_weight back into module.

    - If target_format is given (a key of PLAIN_FORMATS, QUANT_ALGOS, or
      QUANT_FORMAT_VARIANTS), the module is converted to that exact format.
    - If target_format is None: modules that were already quantized are
      re-quantized to their *own* existing format (round-trip, e.g. for a
      merge that keeps the primary model's architecture); modules that
      weren't quantized just get the float tensor cast back to their
      current plain dtype.
    """
    current = module.weight

    if target_format is None:
        if isinstance(current, QuantizedTensor):
            layout_name = module.layout_type
            extra_kwargs = {}
            if getattr(module, "quant_format", None) in QUANT_FORMAT_VARIANTS:
                _, extra_kwargs = QUANT_FORMAT_VARIANTS[module.quant_format]
            elif hasattr(current, "_layout") and hasattr(current._layout, "params"):
                p = current._layout.params
                if getattr(p, "convrot", False):
                    extra_kwargs["convrot"] = True
                    extra_kwargs["per_channel"] = True
                    if hasattr(p, "convrot_groupsize"):
                        extra_kwargs["convrot_groupsize"] = p.convrot_groupsize
            qt = QuantizedTensor.from_float(float_weight.to(torch.bfloat16), layout_name, scale="recalculate", **extra_kwargs)
            module.weight = nn.Parameter(qt, requires_grad=False)
        else:
            module.weight = nn.Parameter(float_weight.to(current.dtype), requires_grad=False)
        return

    if target_format in PLAIN_FORMATS:
        module.weight = nn.Parameter(float_weight.to(PLAIN_FORMATS[target_format]), requires_grad=False)
        for attr in ("quant_format", "layout_type"):
            if hasattr(module, attr):
                try:
                    delattr(module, attr)
                except AttributeError:
                    setattr(module, attr, None)
        return

    actual_format, extra_kwargs = _resolve_quant_format(target_format)
    layout_name = QUANT_ALGOS[actual_format]["comfy_tensor_layout"]
    try:
        qt = QuantizedTensor.from_float(float_weight.to(torch.bfloat16), layout_name, scale="recalculate", **extra_kwargs)
    except Exception as e:
        raise QuantConversionError(f"Failed to quantize to {target_format}: {e}") from e
    module.weight = nn.Parameter(qt, requires_grad=False)
    module.quant_format = actual_format
    module.layout_type = layout_name


def iter_quantizable_linears(root_module: nn.Module, min_features: int = DEFAULT_MIN_FEATURES, skip_substrings: tuple[str, ...] = DEFAULT_SKIP_SUBSTRINGS):
    """Best-effort discovery of nn.Linear-shaped weights when the base
    checkpoint has no existing quantization to mirror. Mirrors the reference
    heuristic from comfy-kitchen's own sample (min feature size + name-based
    skip list) -- not architecture-specific, so it's not guaranteed to match
    what an "official" quantized build of a given model would pick."""
    for name, module in root_module.named_modules():
        w = getattr(module, "weight", None)
        if w is None:
            continue
        raw = w._qdata if isinstance(w, QuantizedTensor) else w
        if not hasattr(raw, "dim") or raw.dim() != 2:
            continue
        if any(s in name for s in skip_substrings):
            continue
        out_f, in_f = raw.shape[0], raw.shape[1]
        if out_f < min_features or in_f < min_features:
            continue
        yield name, module


def convert_module_tree_precision(root_module: nn.Module, output_format: str, progress_cb=None) -> tuple[int, dict[str, torch.Tensor]]:
    """Rewrites, in place, every eligible weight under root_module to
    output_format (a key of PLAIN_FORMATS, QUANT_ALGOS or
    QUANT_FORMAT_VARIANTS).

    Returns (converted_count, state_dict_overrides). state_dict_overrides
    contains the correctly-serialized replacement entries (weight/scale/
    comfy_quant, keyed the same way root_module.state_dict() keys its
    entries) for every converted layer -- callers MUST merge these into
    the state dict obtained from root_module.state_dict() (e.g. via
    backend.utils.get_state_dict_after_quant) AFTER conversion, overwriting
    whatever that produced for the same keys.

    This is necessary because only MixedPrecisionOps.Linear's own
    state_dict() override knows how to split a QuantizedTensor into
    qdata/scale/comfy_quant. A layer that wasn't already
    MixedPrecisionOps-quantized is a plain nn.Linear (backend/operations.py
    ForgeOperations.Linear): reassigning its .weight to a QuantizedTensor
    and then calling the module's own (generic, PyTorch-default)
    state_dict() does NOT correctly serialize it -- generic tensor ops in
    the save pipeline (.clone()/.contiguous()/etc, none of which
    QuantizedTensor specially implements) silently unwrap/dequantize it
    back to a plain float tensor instead of erroring, so the bug is
    invisible until you inspect the actual output file's dtypes."""
    already_quantized = [(n, m) for n, m in root_module.named_modules() if is_quantized(m)]
    targets = already_quantized if already_quantized else list(iter_quantizable_linears(root_module))

    total = len(targets)
    overrides: dict[str, torch.Tensor] = {}
    debug_print(f"Starting precision conversion to '{output_format}' ({total} layers)...")
    for i, (name, module) in enumerate(targets):
        if progress_cb:
            progress_cb(i + 1, total, name)
        debug_print(f"Converting layer ({i + 1}/{total}): {name}")
        float_w = weight_as_float(module)
        set_module_weight(module, float_w, target_format=output_format)
        _quantized_weight_state_dict(module, overrides, f"{name}.")

    debug_print(f"Successfully converted all {total} layers to '{output_format}'.")
    return total, overrides


def to_cpu_contiguous_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """safetensors.torch.save_file requires real, owned CPU memory -- on
    Windows, serializing a tensor that is still a memory-mapped view into
    the *source* checkpoint file (every weight we didn't touch: most of
    CLIP, VAE, any untouched UNet layer) fails with a low-level "invalid
    user buffer" I/O error (os error 1784) instead of a clear Python
    exception. `.contiguous()`/`.cpu()` are no-ops when the tensor already
    satisfies them -- which an mmap'd tensor does -- so they don't help.
    `.clone()` is the one op that always allocates fresh storage."""
    return {k: v.detach().clone().contiguous().cpu() for k, v in sd.items()}


def _save_safetensors_streaming(sd: dict[str, torch.Tensor], filepath: str, metadata: dict[str, str] | None = None) -> None:
    """Zero-memory-duplication streaming safetensors writer that bypasses
    Windows WriteFile buffer issues (os error 1784) and never holds a full
    multi-gigabyte bytes copy in RAM."""
    import json
    import struct

    header = {}
    offset = 0

    dtype_map = {
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.bool: "BOOL",
    }
    for f8_name, code in [("float8_e4m3fn", "F8_E4M3"), ("float8_e5m2", "F8_E5M2")]:
        if hasattr(torch, f8_name):
            dtype_map[getattr(torch, f8_name)] = code

    for k, v in sd.items():
        dt = dtype_map.get(v.dtype, str(v.dtype).split(".")[-1].upper())
        nbytes = v.nelement() * v.element_size()
        header[k] = {
            "dtype": dt,
            "shape": list(v.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    if metadata:
        header["__metadata__"] = metadata

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_json)
    padding = (8 - ((8 + header_len) % 8)) % 8
    header_json += b" " * padding
    header_len = len(header_json)

    debug_print(f"Streaming {len(sd)} tensors to '{filepath}'...")
    with open(filepath, "wb", buffering=16 * 1024 * 1024) as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        total = len(sd)
        for i, (k, v) in enumerate(sd.items()):
            raw = v.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            f.write(raw)
            if (i + 1) % 50 == 0 or i + 1 == total:
                debug_print(f"Saved {i + 1}/{total} tensors to disk...")


def fix_anima_state_dict_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Anima stores its LLMAdapter inside the DiT under 'model.diffusion_model.llm_adapter.*'.
    When Forge loads Anima, loader.py::process_anima pops 'llm_adapter.*' from transformer
    and moves it into text_encoder. When saving via process_clip_state_dict_for_saving,
    Forge prepends 'text_encoders.qwen3_06b.' to these keys.
    This helper restores any 'llm_adapter' keys back to 'model.diffusion_model.llm_adapter.*'
    so that huggingface_guess detection.py finds them when reloading the saved checkpoint.
    """
    if not any("llm_adapter" in k for k in sd):
        return sd
    debug_print("Detected Anima checkpoint with LLM adapter -- ensuring 'model.diffusion_model.llm_adapter.*' prefix...")
    remapped = {}
    fixed_count = 0
    for k, v in sd.items():
        if "llm_adapter" in k and not k.startswith("model.diffusion_model.llm_adapter"):
            suffix = k[k.index("llm_adapter") :]
            remapped[f"model.diffusion_model.{suffix}"] = v
            fixed_count += 1
        else:
            remapped[k] = v
    debug_print(f"Fixed {fixed_count} Anima LLM adapter key(s) to 'model.diffusion_model.llm_adapter.*'")
    return remapped


def save_checkpoint_file(sd: dict[str, torch.Tensor], output_path: str, metadata: dict[str, str] | None = None) -> None:
    sd = fix_anima_state_dict_keys(sd)
    tmp_path = os.path.join(tempfile.gettempdir(), f".checkpoint_doctor_{os.getpid()}_{os.path.basename(output_path)}")
    debug_print(f"Saving checkpoint ({len(sd)} tensors)...")
    try:
        try:
            debug_print(f"Attempting fast save_file() to temp: {tmp_path}")
            save_file(sd, tmp_path, metadata=metadata)
            debug_print("Fast save_file() completed successfully.")
        except Exception as e:
            if "1784" not in str(e) and "invalid" not in str(e).lower():
                raise
            debug_print(f"save_file() failed ({e}) -- using memory-safe streaming writer...")
            _log_tensor_diagnostics(sd)
            _save_safetensors_streaming(sd, tmp_path, metadata=metadata)
            debug_print("Streaming write completed successfully.")

        debug_print(f"Moving temp file to target destination: {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.move(tmp_path, output_path)
        debug_print(f"Checkpoint successfully finalized at {output_path}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _log_tensor_diagnostics(sd: dict[str, torch.Tensor], logger=None) -> None:
    try:
        anomalies = [(k, v.device, v.dtype, tuple(v.shape), v.is_contiguous()) for k, v in sd.items() if v.device.type != "cpu" or not v.is_contiguous()]
        if anomalies:
            debug_print(f"Non-CPU or non-contiguous tensors found right before save ({len(anomalies)}): {anomalies[:10]}")
        else:
            debug_print(f"All {len(sd)} tensors are CPU + contiguous; dtypes: {sorted({str(v.dtype) for v in sd.values()})}")
    except Exception:
        pass


def detect_incompatible_engine(engine) -> str | None:
    """Returns a description of why a loaded ForgeDiffusionEngine isn't
    compatible with the dequantize/merge/requantize helpers in this module
    (baking, merging, quantizing), or None if it is compatible.

    Nunchaku/SVDQuant and GGUF/nf4/fp4 checkpoints don't store weights as
    plain tensors or MixedPrecisionOps QuantizedTensor objects, so
    weight_as_float()/set_module_weight() can't handle them."""
    if getattr(backend.args.dynamic_args, "nunchaku", False):
        return "Nunchaku/SVDQuant (these models store weights a completely different way, without MixedPrecisionOps)"

    diffusion_model = engine.forge_objects.unet.model.diffusion_model
    storage_dtype = getattr(diffusion_model, "storage_dtype", None)
    if storage_dtype in ("gguf", "nf4", "fp4"):
        return f"'{storage_dtype}' storage (does not use MixedPrecisionOps, not supported here)"

    return None
