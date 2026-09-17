"""Bake one or more LoRAs permanently into a checkpoint, reusing the Forge
Neo internal pipeline (the same one used to apply LoRA at generation time)
instead of reimplementing the merge math.

Also supports choosing an output format (fp16/bf16, or a quantized format
supported by backend/quant_ops.py) for the diffusion model.
"""

from __future__ import annotations

import gc
import json
import os

import torch

from backend import memory_management, utils
from backend.loader import forge_loader
from modules import shared

from aux_inspector import embedded_activation_text
from checkpoint_inspector import load_custom_vae_state_dict, read_safetensors_header
from quant_utils import LLM_ADAPTER_MODULE_NAMES, PLAIN_FORMATS, SAFETENSORS_FLOAT_DTYPES, convert_module_tree_precision, debug_print, detect_incompatible_engine, fix_anima_state_dict_keys, save_checkpoint_file, to_cpu_contiguous_state_dict
from precision_stats import dominant_float_dtype, match_dtype
from source_precision import try_match_source_dtypes


class BakeError(RuntimeError):
    pass


def _import_lora_networks():
    # extensions-builtin/sd_forge_lora puts its own root on the Forge
    # sys.path, so its modules are importable as top-level "networks"/
    # "network" (sd_forge_lora itself does "import network" internally).
    import networks  # type: ignore

    return networks


def available_loras() -> dict[str, str]:
    """name -> absolute path, to populate the UI dropdown."""
    networks = _import_lora_networks()
    if not networks.available_networks:
        networks.list_available_networks()
    return {name: net.filename for name, net in networks.available_networks.items()}


def _lora_activation_text(lora_path: str) -> tuple[str, str]:
    """Return the trigger declaration embedded in the LoRA file itself."""
    try:
        header, _ = read_safetensors_header(lora_path)
        return embedded_activation_text(header.get("__metadata__", {}) or {})
    except Exception:
        return "", ""


# An llm_adapter block whose weights are this small relative to the LoRA's
# main blocks carries no real change. LoRAs extracted by SVD from a pair of
# checkpoints emit a factor pair for every module they scan, including ones
# whose delta was zero, so "has llm_adapter keys" on its own says nothing.
# Measured spread on real Anima turbo LoRAs: an SVD extraction of a zero
# delta sits ~700x below its own main blocks, while a LoRA that genuinely
# trained the adapter sits at roughly 1x. 1% separates the two with room
# to spare.
_LLM_ADAPTER_SIGNIFICANCE = 0.01

# Covers the LyCORIS family as well as plain LoRA, since Forge applies them
# all through the same weight-adapter path and any of them can carry
# llm_adapter weights. Dots kept on the diff-style markers: a bare "diff" is
# a substring of "model.diffusion_model.", so it would match every key in the
# LoRA and pull scalars like alpha into the magnitude comparison.
_LORA_FACTOR_MARKERS = (
    "lora_down", "lora_up", "lora_A", "lora_B",
    "hada_w", "lokr_w", "oft_blocks", "oft_diag", "dora_scale",
    ".diff.", ".diff_b",
)


def _lora_touches_llm_adapter(lora_sd: dict) -> bool:
    """Whether a LoRA meaningfully changes the LLM (Qwen3) adapter.

    Anima's own training guidance says never to train that adapter alongside
    a LoRA, and baking one that does permanently alters the checkpoint's text
    understanding -- so we flag it rather than applying it silently.

    Presence of llm_adapter keys is not enough to conclude that, though: an
    SVD-extracted LoRA writes a factor pair for every module it scanned, so
    modules whose delta was zero still show up, carrying only numerical
    noise. Comparing magnitude against the LoRA's own main blocks tells the
    two apart. Matches both key conventions handled by
    extensions-builtin/sd_forge_lora/networks.py::process_anima
    ("diffusion_model.llm_adapter" and "lora_unet_llm_adapter")."""
    llm_sq = llm_n = main_sq = main_n = 0.0
    for k, v in lora_sd.items():
        if not any(m in k for m in _LORA_FACTOR_MARKERS):
            continue
        try:
            t = v.float()
            sq = float(t.pow(2).sum())
            n = t.numel()
        except Exception:
            continue
        if not n:
            continue
        if "llm_adapter" in k:
            llm_sq += sq
            llm_n += n
        else:
            main_sq += sq
            main_n += n

    if not llm_n:
        return False
    if not main_n:
        # Nothing to compare against: an adapter-only LoRA is worth flagging.
        return True

    llm_rms = (llm_sq / llm_n) ** 0.5
    main_rms = (main_sq / main_n) ** 0.5
    if main_rms == 0.0:
        return llm_rms > 0.0
    return (llm_rms / main_rms) > _LLM_ADAPTER_SIGNIFICANCE


def _sanitize_metadata(metadata: dict) -> dict[str, str]:
    out = {}
    for k, v in metadata.items():
        if v is None:
            continue
        out[k] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
    return out


# How much headroom to require beyond the models' own resident size before
# picking GPU: patch_model(lowvram_model_memory=0) needs the whole UNet +
# CLIP to fit with none of the smart partial-loading a normal generation
# gets, and LoRA merging/requantizing needs scratch memory on top of that
# (dequantizing to fp32/bf16, computing deltas, etc) -- 2.5x is a
# conservative margin, not a measured figure.
GPU_SAFETY_FACTOR = 2.5


def _pick_device(unet, clip, choice: str = "auto") -> tuple[torch.device, str]:
    """Returns (device, reason). "gpu"/"cpu" force that device outright;
    "auto" picks GPU only if there's comfortably enough free VRAM to fully
    materialize UNet (+ CLIP, if not None -- pass None when CLIP won't be
    materialized/saved, e.g. save_mode="unet_only") at once, falling back
    to CPU otherwise (slower, but 32GB+ of system RAM is typical and has no
    such hard ceiling)."""
    gpu_device = memory_management.get_torch_device()

    if choice == "gpu":
        return gpu_device, "forced to GPU"
    if choice == "cpu":
        return torch.device("cpu"), "forced to CPU"

    if gpu_device.type != "cuda":
        return torch.device("cpu"), "no CUDA device available"

    try:
        required = unet.model_size() + (clip.patcher.model_size() if clip is not None else 0)
        free = memory_management.get_free_memory(gpu_device)
        if free >= required * GPU_SAFETY_FACTOR:
            return gpu_device, f"{free / 1e9:.1f} GB free VRAM, estimated need ~{required * GPU_SAFETY_FACTOR / 1e9:.1f} GB"
    except Exception as e:
        debug_print(f"Could not estimate VRAM requirement, falling back to CPU: {e}")
        return torch.device("cpu"), "VRAM estimate failed"

    return torch.device("cpu"), f"only {free / 1e9:.1f} GB free VRAM, estimated need ~{required * GPU_SAFETY_FACTOR / 1e9:.1f} GB"


def bake_lora_into_checkpoint(
    checkpoint_path: str,
    loras: list[tuple[str, float]],
    output_path: str,
    output_format: str = "same",
    clip_output_format: str = "same",
    vae_output_format: str = "same",
    save_mode: str = "unet_only",
    device_choice: str = "auto",
    bake_vae: str | None = "original",
    progress_cb=None,
) -> dict:
    """loras: list of (lora_path, strength), applied in sequence.
    save_mode: "unet_only" (default) writes just the diffusion model with
    the LoRA(s) baked in -- matches the native "Save UNet Only" button, and
    is what you want if you already load this checkpoint's VAE/text encoder
    as separate files (shared.opts.forge_additional_modules); "full" also
    bundles CLIP and VAE into the same file, self-contained but much
    bigger (a checkpoint's CLIP/VAE are typically NOT quantized even when
    its diffusion model is, so they can dominate output size).
    output_format/clip_output_format/vae_output_format: "same" to keep that
    component's original format, or a key from quant_utils.PLAIN_FORMATS /
    backend.quant_ops.QUANT_ALGOS. clip_output_format/vae_output_format are
    ignored when save_mode is "unet_only" (nothing to convert). VAE is the
    most quality-sensitive to quantize -- Anima's own VAE mirrors Wan2.1's
    (video-capable), which is considerably larger than a typical image VAE.
    device_choice: "auto" (pick GPU only if it comfortably fits), "gpu"
    (force, may OOM on smaller cards), or "cpu" (force, always fits but
    slower)."""

    if progress_cb:
        progress_cb("Loading base checkpoint...")

    # Reuse whatever additional modules (VAE / text encoder files) the user
    # currently has configured for normal loading -- same as how a regular
    # generation loads this checkpoint. Without this, checkpoints that ship
    # without an embedded VAE fail with "You do not have VAE state dict!".
    engine = forge_loader(checkpoint_path, additional_state_dicts=shared.opts.forge_additional_modules)

    try:
        incompat = detect_incompatible_engine(engine)
        if incompat:
            raise BakeError(f"Checkpoint incompatible with LoRA baking: {incompat}.")

        networks = _import_lora_networks()

        unet = engine.forge_objects.unet
        clip = engine.forge_objects.clip
        applied = []

        save_full = save_mode == "full"
        device, device_reason = _pick_device(unet, clip if save_full else None, device_choice)
        if progress_cb:
            progress_cb(f"Materializing on {device.type.upper()} ({device_reason})")

        for lora_path, strength in loras:
            if progress_cb:
                progress_cb(f"Applying LoRA: {os.path.basename(lora_path)} (strength={strength})...")
            lora_sd = networks.load_lora_state_dict(lora_path)

            touches_llm_adapter = _lora_touches_llm_adapter(lora_sd)
            if touches_llm_adapter and progress_cb:
                progress_cb(f"WARNING: {os.path.basename(lora_path)} contains LLM adapter weights (Anima guidance says never to train these alongside a LoRA)")

            unet, clip = networks.load_lora_for_models(unet, clip, lora_sd, strength, strength, filename=lora_path)
            activation_text, activation_text_source = _lora_activation_text(lora_path)
            applied.append(
                {
                    "name": os.path.basename(lora_path),
                    "strength": strength,
                    "activation_text": activation_text,
                    "activation_text_source": activation_text_source,
                    "llm_adapter_warning": touches_llm_adapter,
                }
            )

        if progress_cb:
            progress_cb("Materializing patched weights (this can take a while)...")

        unet.patch_model(device_to=device, lowvram_model_memory=0, force_patch_weights=True)
        if save_full:
            clip.patcher.patch_model(device_to=device, lowvram_model_memory=0, force_patch_weights=True)

        unet_overrides = {}
        clip_overrides = {}
        vae_overrides = {}

        if output_format != "same":
            if progress_cb:
                progress_cb(f"Converting diffusion model to {output_format}...")
            _, unet_overrides = convert_module_tree_precision(
                unet.model.diffusion_model,
                output_format,
                progress_cb=(lambda i, total, name: progress_cb(f"Quantizing ({i}/{total}): {name}")) if progress_cb else None,
            )

        if save_full and clip_output_format != "same":
            if progress_cb:
                progress_cb(f"Converting text encoder to {clip_output_format}...")
            _, clip_overrides = convert_module_tree_precision(
                clip.cond_stage_model,
                clip_output_format,
                progress_cb=(lambda i, total, name: progress_cb(f"Quantizing text encoder ({i}/{total}): {name}")) if progress_cb else None,
                skip_names=LLM_ADAPTER_MODULE_NAMES,
            )

        is_custom_vae = bool(bake_vae and bake_vae not in ("original", "none", ""))
        strip_vae = bake_vae == "none"
        has_vae_model = getattr(engine.forge_objects, "vae", None) is not None

        if save_full and not is_custom_vae and not strip_vae and vae_output_format != "same" and has_vae_model:
            if progress_cb:
                progress_cb(f"Converting VAE to {vae_output_format}...")
            _, vae_overrides = convert_module_tree_precision(
                engine.forge_objects.vae.first_stage_model,
                vae_output_format,
                progress_cb=(lambda i, total, name: progress_cb(f"Quantizing VAE ({i}/{total}): {name}")) if progress_cb else None,
            )

        if progress_cb:
            progress_cb("Assembling final state dict...")

        # Merge each component's *_overrides on top of the plain state dict
        # BEFORE prefixing -- see convert_module_tree_precision's docstring:
        # a converted layer's own state_dict() entry is not trustworthy on
        # its own (it can silently be a dequantized plain tensor).
        unet_sd = utils.get_state_dict_after_quant(unet.model.diffusion_model)
        unet_sd.update(unet_overrides)

        processed_unet = fix_anima_state_dict_keys(
            engine.model_config.process_unet_state_dict_for_saving(unet_sd)
        )
        sd = dict(processed_unet)
        unet_output_keys = set(processed_unet)
        clip_output_keys: set[str] = set()
        vae_output_keys: set[str] = set()
        if save_full:
            clip_sd = utils.get_state_dict_after_quant(clip.cond_stage_model)
            clip_sd.update(clip_overrides)
            processed_clip = fix_anima_state_dict_keys(
                engine.model_config.process_clip_state_dict_for_saving(clip_sd)
            )
            llm_adapter_keys = {k for k in processed_clip if "llm_adapter" in k}
            unet_output_keys.update(llm_adapter_keys)
            clip_output_keys.update(set(processed_clip) - llm_adapter_keys)
            sd.update(processed_clip)
            # The adapter rides along in the text encoder but belongs to the
            # DiT, so it takes the DiT's precision, not the encoder's.
            dit_dtype = dominant_float_dtype(processed_unet)
            for k in llm_adapter_keys:
                sd[k] = match_dtype(sd[k], dit_dtype)
            if not is_custom_vae and not strip_vae and has_vae_model:
                vae_sd = utils.get_state_dict_after_quant(engine.forge_objects.vae.first_stage_model)
                vae_sd.update(vae_overrides)
                processed_vae = engine.model_config.process_vae_state_dict_for_saving(vae_sd)
                vae_output_keys.update(processed_vae)
                sd.update(processed_vae)
        else:
            # For Anima, llm_adapter was moved into clip.cond_stage_model by loader.py,
            # but on disk it is part of Anima's DiT (model.diffusion_model.llm_adapter.*).
            # We must include it even in unet_only mode!
            clip_sd = utils.get_state_dict_after_quant(clip.cond_stage_model)
            dit_dtype = dominant_float_dtype(sd)
            for k, v in clip_sd.items():
                if "llm_adapter" in k:
                    suffix = k[k.index("llm_adapter") :]
                    output_key = f"model.diffusion_model.{suffix}"
                    sd[output_key] = match_dtype(v, dit_dtype)
                    unet_output_keys.add(output_key)

        if is_custom_vae:
            if progress_cb:
                progress_cb(f"Baking custom VAE: {os.path.basename(bake_vae)}...")
            custom_vae_sd = load_custom_vae_state_dict(bake_vae)
            if vae_output_format in PLAIN_FORMATS:
                target_dt = PLAIN_FORMATS[vae_output_format]
                custom_vae_sd = {k: (v.to(target_dt) if hasattr(v, "to") else v) for k, v in custom_vae_sd.items()}
            if hasattr(engine, "model_config") and hasattr(engine.model_config, "process_vae_state_dict_for_saving"):
                try:
                    processed_vae = engine.model_config.process_vae_state_dict_for_saving(custom_vae_sd)
                except Exception:
                    prefix = "first_stage_model." if "SDXL" in type(engine.model_config).__name__ or "SD1" in type(engine.model_config).__name__ else "vae."
                    processed_vae = {f"{prefix}{k}": v for k, v in custom_vae_sd.items()}
            else:
                prefix = "first_stage_model."
                processed_vae = {f"{prefix}{k}": v for k, v in custom_vae_sd.items()}
            sd.update(processed_vae)

        same_source_keys: set[str] = set()
        if output_format == "same":
            same_source_keys.update(unet_output_keys)
        if save_full and clip_output_format == "same":
            same_source_keys.update(clip_output_keys)
        if save_full and not is_custom_vae and vae_output_format == "same":
            same_source_keys.update(vae_output_keys)
        if same_source_keys:
            restored, precision_warning = try_match_source_dtypes(
                sd, checkpoint_path, same_source_keys, SAFETENSORS_FLOAT_DTYPES
            )
            if progress_cb and restored:
                progress_cb(f"Restored source precision for {restored} tensor(s).")
            if progress_cb and precision_warning:
                progress_cb(precision_warning)

        sd = to_cpu_contiguous_state_dict(sd)

        # `sd` now holds independent CPU copies of everything we need --
        # drop the live (GPU-resident) model before the save step, which
        # itself needs headroom to hold the whole checkpoint again while
        # writing. Skipping this risks a silent OOM on modest-RAM machines.
        engine = None
        unet = None
        clip = None
        memory_management.soft_empty_cache()
        gc.collect()

        recipe_dict = {
            "type": "CheckpointDoctor-LoRABake",
            "base_model": os.path.basename(checkpoint_path),
            "loras": applied,
            "save_mode": save_mode,
            "output_format": output_format,
            "clip_output_format": clip_output_format if save_full else "n/a (unet_only)",
            "vae_output_format": vae_output_format if save_full else "n/a (unet_only)",
        }
        if is_custom_vae:
            recipe_dict["baked_vae"] = os.path.basename(bake_vae)
        elif strip_vae:
            recipe_dict["baked_vae"] = "none (stripped)"

        metadata = _sanitize_metadata({"sd_checkpoint_doctor_recipe": recipe_dict, "sd_merge_recipe": recipe_dict})

        if progress_cb:
            progress_cb("Saving file...")
        save_checkpoint_file(sd, output_path, metadata=metadata)

        return {
            "output": output_path,
            "loras": applied,
            "output_format": output_format,
            "baked_vae": os.path.basename(bake_vae) if is_custom_vae else ("none" if strip_vae else None),
        }
    finally:
        del engine
        memory_management.soft_empty_cache()
        gc.collect()
