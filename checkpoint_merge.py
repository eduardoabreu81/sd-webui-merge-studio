"""Quantization-aware checkpoint merging.

The native Checkpoint Merger (modules/extras.py::run_modelmerger) merges
raw tensors with torch.lerp/torch.add. That works fine for plain fp16/bf16
checkpoints, but breaks for quantized models (e.g. Anima int8 builds):
comfy_quant metadata blobs are raw JSON bytes stored as uint8 tensors, and
the actual quantized weight tensors are integer-coded values that cannot be
linearly interpolated directly -- they first need to be dequantized.

This module loads each source checkpoint as a real model (via forge_loader,
same as normal generation), merges every matching layer in float space, and
writes the result back through the module's own weight-setting logic --
which re-quantizes automatically for layers that were quantized.
"""

from __future__ import annotations

import gc
import json
import os
import re

import torch.nn as nn

from backend import memory_management, utils
from backend.loader import forge_loader
from modules import sd_models, shared

from lora_bake import (
    _import_lora_networks,
    _lora_activation_text,
    _lora_touches_llm_adapter,
    _pick_device,
)
from checkpoint_inspector import load_custom_vae_state_dict
import anima_remap
from quant_utils import PLAIN_FORMATS, SAFETENSORS_FLOAT_DTYPES, convert_module_tree_precision, detect_incompatible_engine, fix_anima_state_dict_keys, save_checkpoint_file, set_module_weight, to_cpu_contiguous_state_dict, weight_as_float
from precision_stats import dominant_float_dtype, match_dtype
from source_precision import try_match_source_dtypes

INTERP_NO_INTERPOLATION = "no_interpolation"
INTERP_WEIGHTED_SUM = "weighted_sum"
INTERP_ADD_DIFFERENCE = "add_difference"


class MergeError(RuntimeError):
    pass


def _sanitize_metadata(metadata: dict) -> dict[str, str]:
    out = {}
    for k, v in metadata.items():
        if v is None:
            continue
        out[k] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
    return out


def _checkpoint_info(name: str) -> "sd_models.CheckpointInfo":
    info = sd_models.checkpoint_aliases.get(name)
    if info is None:
        raise MergeError(f"Checkpoint not found: {name}")
    return info


def _load_engine(checkpoint_path: str):
    return forge_loader(checkpoint_path, additional_state_dicts=shared.opts.forge_additional_modules)


def _build_metadata(
    primary_info,
    secondary_info,
    tertiary_info,
    interp_method: str,
    multiplier: float,
    discard_regex: str,
    output_format: str,
    save_metadata: bool,
    config_source: tuple[str, ...],
    add_merge_recipe: bool,
    save_mode: str = "unet_only",
    loras: list[dict] | None = None,
    bake_vae: str | None = None,
    anima_remap: dict | None = None,
) -> dict:
    """Mirrors modules/extras.py::run_modelmerger's metadata handling, so
    merges produced here carry the same sd_merge_recipe/sd_merge_models
    provenance convention as the native Checkpoint Merger."""
    metadata: dict = {}
    if not save_metadata:
        return metadata

    if "A" in config_source and primary_info is not None:
        metadata.update(primary_info.metadata)
    if "B" in config_source and secondary_info is not None:
        metadata.update(secondary_info.metadata)
    if "C" in config_source and tertiary_info is not None:
        metadata.update(tertiary_info.metadata)

    if add_merge_recipe:
        merge_recipe = {
            "type": "MergeStudio-AnimaMerge",
            "interp_method": interp_method,
            "discard_weights": discard_regex,
            "config_source": list(config_source),
            "output_format": output_format,
            "save_mode": save_mode,
        }
        # No Interpolation takes a single model, so the multiplier slider's
        # value never entered the result. Recording it anyway leaves a number
        # in the recipe that reads like a blend ratio and means nothing.
        if interp_method != INTERP_NO_INTERPOLATION:
            merge_recipe["multiplier"] = multiplier
        if bake_vae and bake_vae not in ("original", "none", ""):
            merge_recipe["baked_vae"] = os.path.basename(bake_vae)
        elif bake_vae == "none":
            merge_recipe["baked_vae"] = "none (stripped)"
        if anima_remap:
            merge_recipe["anima_remap"] = {
                "from_blocks": anima_remap["from_blocks"],
                "to_blocks": anima_remap["to_blocks"],
                "merged_blocks": anima_remap["frozen_blocks"],
                "inserted_blocks": anima_remap["inserted_blocks"],
                "extend_ratio": anima_remap.get("extend_ratio", 0.0),
            }
        if loras:
            # A trigger explicitly declared in the LoRA's safetensors header
            # travels inside the checkpoint recipe. External sidecars are not
            # consulted, so the recipe remains attributable to its input files.
            merge_recipe["baked_loras"] = [
                {
                    "name": a["name"],
                    "strength": a["strength"],
                    "activation_text": a.get("activation_text", ""),
                    "activation_text_raw": a.get("activation_text_raw", ""),
                    "activation_text_source": a.get("activation_text_source", ""),
                    "llm_adapter_warning": a["llm_adapter_warning"],
                }
                for a in loras
            ]
        sd_merge_models: dict = {}

        def add_model_metadata(key: str, checkpoint_info):
            checkpoint_info.calculate_shorthash()
            merge_recipe[key] = checkpoint_info.sha256
            sd_merge_models[checkpoint_info.sha256] = {"name": checkpoint_info.name, "legacy_hash": checkpoint_info.hash}
            if (r := checkpoint_info.metadata.get("sd_merge_recipe")) is not None:
                sd_merge_models["sd_merge_recipe"] = r
            if (m := checkpoint_info.metadata.get("sd_merge_models")) is not None:
                sd_merge_models["sd_merge_models"] = m

        if primary_info:
            add_model_metadata("primary_model_hash", primary_info)
        if secondary_info:
            add_model_metadata("secondary_model_hash", secondary_info)
        if tertiary_info:
            add_model_metadata("tertiary_model_hash", tertiary_info)

        metadata["sd_merge_recipe"] = json.dumps(merge_recipe)
        metadata["sd_merge_models"] = json.dumps(sd_merge_models)

    return metadata


def _guard_anima_block_order(primary_insp, other_insp, fam_primary, fam_other, label):
    """Rejects an Anima pairing where B/C is a NEWER (larger) generation than A,
    before any engine is loaded -- the merge output is always A's architecture,
    and there is no defined way to collapse a larger model's blocks onto a
    smaller one. Reads block counts from the .safetensors header, so this
    costs microseconds and runs before the expensive load."""
    if fam_primary != "anima" or fam_other != "anima":
        return
    blocks_a = primary_insp.get("block_count")
    blocks_other = other_insp.get("block_count")
    if blocks_a is None or blocks_other is None or blocks_other <= blocks_a:
        return
    raise MergeError(
        f"Wrong model order: {label} has {blocks_other} blocks but Primary Model (A) has only {blocks_a}. "
        f"Anima's newer generations add blocks, and the merge output always takes Model A's architecture, "
        f"so the larger model must be A. Swap them (put the {blocks_other}-block model in A and the "
        f"{blocks_a}-block model in {label})."
    )


def _build_anima_translator(engine_a, engine_b, engine_c, diffusion_a, diffusion_b, diffusion_c):
    """Anima's generations (28 / 40 / 52 blocks) were each built by inserting
    new blocks between the previous generation's, so a name-for-name merge
    lines up unrelated layers. When A and B (and C) are Anima checkpoints of
    different generations, returns a dict carrying a name translator that
    rewrites A's block indices into B/C's, plus a human-readable note.

    Returns None when no remapping is needed or the models aren't Anima.
    Raises MergeError if the merge is impossible in the requested direction.
    """
    if diffusion_b is None:
        return None
    if not all(anima_remap.is_anima_engine(e) for e in (engine_a, engine_b) if e is not None):
        return None
    if engine_c is not None and not anima_remap.is_anima_engine(engine_c):
        return None

    blocks_a = anima_remap.block_count(diffusion_a)
    blocks_b = anima_remap.block_count(diffusion_b)
    if blocks_a is None or blocks_b is None or blocks_a == blocks_b:
        # Same generation (or not a block-list model): plain name matching.
        if diffusion_c is not None and anima_remap.block_count(diffusion_c) not in (None, blocks_a):
            raise MergeError(
                "Add Difference across Anima generations requires Model B and Model C to have the "
                f"same block count (B has {blocks_b}, C has {anima_remap.block_count(diffusion_c)})."
            )
        return None

    if diffusion_c is not None:
        blocks_c = anima_remap.block_count(diffusion_c)
        if blocks_c is not None and blocks_c != blocks_b:
            raise MergeError(
                "Add Difference across Anima generations requires Model B and Model C to have the "
                f"same block count (B has {blocks_b}, C has {blocks_c})."
            )

    try:
        mapping = anima_remap.target_to_source(blocks_b, blocks_a)
    except anima_remap.AnimaRemapError as e:
        raise MergeError(str(e)) from e

    frozen, inserted = anima_remap.split_frozen_inserted(mapping)
    return {
        "translate": anima_remap.make_name_translator(frozen),
        "extend": anima_remap.make_extend_translator(inserted),
        "from_blocks": blocks_b,
        "to_blocks": blocks_a,
        "frozen_blocks": len(frozen),
        "inserted_blocks": len(inserted),
        "message": (
            f"Anima cross-generation merge: remapping {blocks_b}-block -> {blocks_a}-block "
            f"({len(frozen)} shared blocks merged, {len(inserted)} inserted blocks kept from Model A)"
        ),
    }


def _merge_module_tree(
    module_a: nn.Module | None,
    module_b: nn.Module | None,
    module_c: nn.Module | None,
    interp_method: str,
    multiplier: float,
    target_device: torch.device | None = None,
    progress_cb=None,
    translate_name=None,
    extend_name=None,
    extend_ratio: float = 0.0,
) -> tuple[int, list[str]]:
    """translate_name: optional f(name_in_a) -> name_in_b_and_c, or None when
    the A-side module has no counterpart on the other side (used for
    cross-generation Anima merges, where B's block indices are shifted).
    Defaults to matching module names one-for-one.

    extend_name / extend_ratio: for A-side modules that translate_name maps to
    None (blocks the newer generation inserted, which have no counterpart),
    extend_name gives the B-side module the inserted block was originally
    copied from at initialization. With extend_ratio > 0 that module is
    blended in at that weight instead of the block being left untouched."""
    if module_a is None:
        return 0, []
    named_a = dict(module_a.named_modules())
    named_b = dict(module_b.named_modules()) if module_b is not None else {}
    named_c = dict(module_c.named_modules()) if module_c is not None else {}

    weighted_names = [n for n, m in named_a.items() if getattr(m, "weight", None) is not None]
    total = len(weighted_names)
    merged_count = 0
    skipped: list[str] = []

    for i, name in enumerate(weighted_names):
        m_a = named_a[name]
        w_a = weight_as_float(m_a)
        if w_a is None:
            continue

        if interp_method == INTERP_NO_INTERPOLATION:
            merged_count += 1
            if progress_cb:
                progress_cb(i + 1, total, name)
            continue

        lookup = translate_name(name) if translate_name is not None else name
        # An inserted block has no counterpart; extend_ratio optionally blends
        # in the block it was copied from instead of leaving it at A's weights.
        effective_multiplier = multiplier
        if lookup is None and extend_name is not None and extend_ratio > 0.0:
            lookup = extend_name(name)
            effective_multiplier = extend_ratio
        m_b = named_b.get(lookup) if lookup is not None else None
        if m_b is None:
            skipped.append(name)
            continue
        w_b = weight_as_float(m_b)
        if w_b is None or w_b.shape != w_a.shape:
            skipped.append(name)
            continue

        dev = target_device if target_device is not None else w_a.device
        if w_a.device != dev:
            w_a = w_a.to(device=dev)
        if w_b.device != dev:
            w_b = w_b.to(device=dev)

        if interp_method == INTERP_WEIGHTED_SUM:
            merged = w_a.lerp(w_b, effective_multiplier)
        elif interp_method == INTERP_ADD_DIFFERENCE:
            m_c = named_c.get(lookup)
            if m_c is None:
                skipped.append(name)
                continue
            w_c = weight_as_float(m_c)
            if w_c is None or w_c.shape != w_a.shape:
                skipped.append(name)
                continue
            if w_c.device != dev:
                w_c = w_c.to(device=dev)
            merged = w_a + effective_multiplier * (w_b - w_c)
            del w_c
        else:
            raise MergeError(f"Unknown interpolation method: {interp_method}")

        set_module_weight(m_a, merged, target_format=None)
        merged_count += 1
        del w_a, w_b, merged

        # bias is always plain float, no quantization to worry about
        if getattr(m_a, "bias", None) is not None and interp_method != INTERP_NO_INTERPOLATION:
            b_a = m_a.bias.data.float()
            b_b = getattr(m_b, "bias", None)
            if b_b is not None and b_b.shape == b_a.shape:
                b_b_f = b_b.data.float()
                if b_a.device != dev:
                    b_a = b_a.to(device=dev)
                if b_b_f.device != dev:
                    b_b_f = b_b_f.to(device=dev)

                if interp_method == INTERP_WEIGHTED_SUM:
                    new_bias = b_a.lerp(b_b_f, effective_multiplier)
                else:
                    m_c = named_c.get(lookup)
                    b_c = getattr(m_c, "bias", None) if m_c is not None else None
                    if b_c is not None and b_c.shape == b_a.shape:
                        b_c_f = b_c.data.float()
                        if b_c_f.device != dev:
                            b_c_f = b_c_f.to(device=dev)
                        new_bias = b_a + effective_multiplier * (b_b_f - b_c_f)
                        del b_c_f
                    else:
                        new_bias = None
                if new_bias is not None:
                    m_a.bias = nn.Parameter(new_bias.to(dtype=m_a.bias.dtype, device=dev), requires_grad=False)
                del b_a, b_b_f

        if progress_cb:
            progress_cb(i + 1, total, name)

    return merged_count, skipped


def merge_checkpoints(
    primary_name: str,
    secondary_name: str | None,
    tertiary_name: str | None,
    interp_method: str,
    multiplier: float,
    output_path: str,
    output_format: str = "same",
    clip_output_format: str = "same",
    vae_output_format: str = "same",
    discard_regex: str = "",
    save_metadata: bool = True,
    config_source: tuple[str, ...] = ("A", "B", "C"),
    add_merge_recipe: bool = True,
    save_mode: str = "unet_only",
    loras: list[tuple[str, float]] | None = None,
    device_choice: str = "auto",
    bake_vae: str | None = "original",
    anima_extend_ratio: float = 0.0,
    progress_cb=None,
) -> dict:
    if interp_method != INTERP_NO_INTERPOLATION and not secondary_name:
        raise MergeError("This interpolation method requires a Secondary Model (B).")
    if interp_method == INTERP_ADD_DIFFERENCE and not tertiary_name:
        raise MergeError("Add Difference requires a Tertiary Model (C).")

    primary_info = _checkpoint_info(primary_name)
    secondary_info = _checkpoint_info(secondary_name) if secondary_name else None
    tertiary_info = _checkpoint_info(tertiary_name) if tertiary_name else None

    # Fast architecture compatibility check before heavy engine loading
    try:
        from checkpoint_inspector import inspect_checkpoint, get_model_family
        p_insp = inspect_checkpoint(primary_info.filename)
        fam_a = get_model_family(p_insp.get("architecture", ""))

        if secondary_info:
            s_insp = inspect_checkpoint(secondary_info.filename)
            fam_b = get_model_family(s_insp.get("architecture", ""))
            _guard_anima_block_order(p_insp, s_insp, fam_a, fam_b, "Secondary Model (B)")
            if fam_a != "other" and fam_b != "other" and fam_a != fam_b:
                arch_a = p_insp.get("architecture", "Unknown")
                arch_b = s_insp.get("architecture", "Unknown")
                raise MergeError(f"Incompatible models: Model A is '{arch_a}' ({fam_a.upper()}) but Model B is '{arch_b}' ({fam_b.upper()}). Checkpoints from different architecture families cannot be merged.")

        if tertiary_info:
            t_insp = inspect_checkpoint(tertiary_info.filename)
            fam_c = get_model_family(t_insp.get("architecture", ""))
            _guard_anima_block_order(p_insp, t_insp, fam_a, fam_c, "Tertiary Model (C)")
            if fam_a != "other" and fam_c != "other" and fam_a != fam_c:
                arch_a = p_insp.get("architecture", "Unknown")
                arch_c = t_insp.get("architecture", "Unknown")
                raise MergeError(f"Incompatible models: Model A is '{arch_a}' ({fam_a.upper()}) but Model C is '{arch_c}' ({fam_c.upper()}). Checkpoints from different architecture families cannot be merged.")
    except MergeError:
        raise
    except Exception:
        pass

    if progress_cb:
        progress_cb("Loading Primary Model (A)...")
    engine_a = _load_engine(primary_info.filename)
    engine_b = None
    engine_c = None

    try:
        incompat = detect_incompatible_engine(engine_a)
        if incompat:
            raise MergeError(f"Primary Model (A) is incompatible with this merge: {incompat}.")

        if secondary_info:
            if progress_cb:
                progress_cb("Loading Secondary Model (B)...")
            engine_b = _load_engine(secondary_info.filename)
            incompat = detect_incompatible_engine(engine_b)
            if incompat:
                raise MergeError(f"Secondary Model (B) is incompatible with this merge: {incompat}.")
        if tertiary_info:
            if progress_cb:
                progress_cb("Loading Tertiary Model (C)...")
            engine_c = _load_engine(tertiary_info.filename)
            incompat = detect_incompatible_engine(engine_c)
            if incompat:
                raise MergeError(f"Tertiary Model (C) is incompatible with this merge: {incompat}.")

        diffusion_a = engine_a.forge_objects.unet.model.diffusion_model
        diffusion_b = engine_b.forge_objects.unet.model.diffusion_model if engine_b else None
        diffusion_c = engine_c.forge_objects.unet.model.diffusion_model if engine_c else None

        clip_a = engine_a.forge_objects.clip.cond_stage_model if getattr(engine_a.forge_objects, "clip", None) else None
        clip_b = engine_b.forge_objects.clip.cond_stage_model if engine_b and getattr(engine_b.forge_objects, "clip", None) else None
        clip_c = engine_c.forge_objects.clip.cond_stage_model if engine_c and getattr(engine_c.forge_objects, "clip", None) else None

        vae_a = engine_a.forge_objects.vae.first_stage_model if getattr(engine_a.forge_objects, "vae", None) else None
        vae_b = engine_b.forge_objects.vae.first_stage_model if engine_b and getattr(engine_b.forge_objects, "vae", None) else None
        vae_c = engine_c.forge_objects.vae.first_stage_model if engine_c and getattr(engine_c.forge_objects, "vae", None) else None

        # Determine target device
        target_device, device_reason = _pick_device(
            engine_a.forge_objects.unet,
            engine_a.forge_objects.clip if save_mode == "full" else None,
            device_choice,
        )
        if progress_cb:
            progress_cb(f"Device: {target_device.type.upper()} ({device_reason})")
        if target_device.type == "cpu":
            memory_management.soft_empty_cache()

        # Cross-generation Anima: block indices shifted when each generation
        # inserted new blocks, so B/C must be looked up by translated name.
        anima_remap_note = _build_anima_translator(
            engine_a, engine_b, engine_c,
            diffusion_a, diffusion_b, diffusion_c,
        )
        unet_translate = anima_remap_note.pop("translate", None) if anima_remap_note else None
        unet_extend = anima_remap_note.pop("extend", None) if anima_remap_note else None
        if anima_remap_note:
            anima_remap_note["extend_ratio"] = anima_extend_ratio
            if anima_extend_ratio > 0.0:
                anima_remap_note["message"] += f", inserted blocks blended at extend_ratio={anima_extend_ratio}"
            if progress_cb:
                progress_cb(anima_remap_note["message"])

        if progress_cb:
            progress_cb("Merging diffusion model...")
        merged_unet, skipped_unet = _merge_module_tree(
            diffusion_a, diffusion_b, diffusion_c, interp_method, multiplier,
            target_device=target_device,
            progress_cb=(lambda i, t, n: progress_cb(f"Merging UNet ({i}/{t}): {n}")) if progress_cb else None,
            translate_name=unet_translate,
            extend_name=unet_extend,
            extend_ratio=anima_extend_ratio,
        )

        if clip_a is not None:
            if progress_cb:
                progress_cb("Merging text encoder...")
            merged_clip, skipped_clip = _merge_module_tree(
                clip_a, clip_b, clip_c, interp_method, multiplier,
                target_device=target_device,
                progress_cb=(lambda i, t, n: progress_cb(f"Merging CLIP ({i}/{t}): {n}")) if progress_cb else None,
            )
        else:
            merged_clip, skipped_clip = 0, []

        is_custom_vae = bool(bake_vae and bake_vae not in ("original", "none", ""))
        strip_vae = bake_vae == "none"

        if save_mode == "full" and not is_custom_vae and not strip_vae and vae_a is not None:
            if progress_cb:
                progress_cb("Merging VAE...")
            merged_vae, skipped_vae = _merge_module_tree(
                vae_a, vae_b, vae_c, interp_method, multiplier,
                target_device=target_device,
                progress_cb=(lambda i, t, n: progress_cb(f"Merging VAE ({i}/{t}): {n}")) if progress_cb else None,
            )
        else:
            merged_vae, skipped_vae = 0, []

        # Drop secondary and tertiary engines immediately after merging to free RAM/VRAM before LoRA baking
        engine_b = engine_c = None
        diffusion_b = diffusion_c = None
        clip_b = clip_c = None
        vae_b = vae_c = None
        memory_management.soft_empty_cache()
        gc.collect()

        applied_loras = []
        if loras:
            networks = _import_lora_networks()
            unet = engine_a.forge_objects.unet
            clip = engine_a.forge_objects.clip
            save_full = save_mode == "full"
            if progress_cb:
                progress_cb(f"Materializing LoRA on {target_device.type.upper()} ({device_reason})")

            for lora_path, strength in loras:
                if progress_cb:
                    progress_cb(f"Applying LoRA: {os.path.basename(lora_path)} (strength={strength})...")
                lora_sd = networks.load_lora_state_dict(lora_path)

                touches_llm_adapter = _lora_touches_llm_adapter(lora_sd)
                if touches_llm_adapter and progress_cb:
                    progress_cb(f"WARNING: {os.path.basename(lora_path)} contains LLM adapter weights (Anima guidance says never to train these alongside a LoRA)")

                unet, clip = networks.load_lora_for_models(unet, clip, lora_sd, strength, strength, filename=lora_path)
                activation_text, activation_text_source = _lora_activation_text(lora_path)
                applied_loras.append(
                    {
                        "name": os.path.basename(lora_path),
                        "strength": strength,
                        "activation_text": activation_text,
                        "activation_text_raw": activation_text,
                        "activation_text_source": activation_text_source,
                        "llm_adapter_warning": touches_llm_adapter,
                    }
                )

            if progress_cb:
                progress_cb("Materializing patched weights (this can take a while)...")

            unet.patch_model(device_to=target_device, lowvram_model_memory=0, force_patch_weights=True)
            if (save_full or any(a["llm_adapter_warning"] for a in applied_loras)) and clip is not None and getattr(clip, "patcher", None) is not None:
                clip.patcher.patch_model(device_to=target_device, lowvram_model_memory=0, force_patch_weights=True)

        unet_overrides = {}
        clip_overrides = {}
        vae_overrides = {}

        if output_format != "same":
            if progress_cb:
                progress_cb(f"Converting diffusion model to {output_format}...")
            _, unet_overrides = convert_module_tree_precision(
                diffusion_a, output_format,
                progress_cb=(lambda i, t, n: progress_cb(f"Quantizing ({i}/{t}): {n}")) if progress_cb else None,
            )

        if save_mode == "full" and clip_output_format != "same" and clip_a is not None:
            if progress_cb:
                progress_cb(f"Converting text encoder to {clip_output_format}...")
            _, clip_overrides = convert_module_tree_precision(
                clip_a, clip_output_format,
                progress_cb=(lambda i, t, n: progress_cb(f"Quantizing text encoder ({i}/{t}): {n}")) if progress_cb else None,
            )

        if save_mode == "full" and not is_custom_vae and not strip_vae and vae_output_format != "same" and vae_a is not None:
            if progress_cb:
                progress_cb(f"Converting VAE to {vae_output_format}...")
            _, vae_overrides = convert_module_tree_precision(
                vae_a, vae_output_format,
                progress_cb=(lambda i, t, n: progress_cb(f"Quantizing VAE ({i}/{t}): {n}")) if progress_cb else None,
            )

        if progress_cb:
            progress_cb("Assembling final state dict...")

        # Merge overrides on top of the plain state dict BEFORE prefixing --
        # see convert_module_tree_precision's docstring: a converted layer's
        # own state_dict() entry is not trustworthy on its own (it can
        # silently be a dequantized plain tensor).
        unet_sd = utils.get_state_dict_after_quant(diffusion_a)
        unet_sd.update(unet_overrides)

        processed_unet = fix_anima_state_dict_keys(
            engine_a.model_config.process_unet_state_dict_for_saving(unet_sd)
        )
        sd = dict(processed_unet)
        unet_output_keys = set(processed_unet)
        clip_output_keys: set[str] = set()
        vae_output_keys: set[str] = set()
        if save_mode == "full":
            if clip_a is not None:
                clip_sd = utils.get_state_dict_after_quant(clip_a)
                clip_sd.update(clip_overrides)
                processed_clip = fix_anima_state_dict_keys(
                    engine_a.model_config.process_clip_state_dict_for_saving(clip_sd)
                )
                llm_adapter_keys = {k for k in processed_clip if "llm_adapter" in k}
                unet_output_keys.update(llm_adapter_keys)
                clip_output_keys.update(set(processed_clip) - llm_adapter_keys)
                sd.update(processed_clip)
            if not is_custom_vae and not strip_vae and vae_a is not None:
                vae_sd = utils.get_state_dict_after_quant(vae_a)
                vae_sd.update(vae_overrides)
                processed_vae = engine_a.model_config.process_vae_state_dict_for_saving(vae_sd)
                vae_output_keys.update(processed_vae)
                sd.update(processed_vae)
        else:
            # For Anima, llm_adapter was moved into clip.cond_stage_model by loader.py,
            # but on disk it is part of Anima's DiT (model.diffusion_model.llm_adapter.*).
            # We must include it even in unet_only mode!
            if clip_a is not None:
                clip_sd = utils.get_state_dict_after_quant(clip_a)
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
            if hasattr(engine_a, "model_config") and hasattr(engine_a.model_config, "process_vae_state_dict_for_saving"):
                try:
                    processed_vae = engine_a.model_config.process_vae_state_dict_for_saving(custom_vae_sd)
                except Exception:
                    prefix = "first_stage_model." if "SDXL" in type(engine_a.model_config).__name__ or "SD1" in type(engine_a.model_config).__name__ else "vae."
                    processed_vae = {f"{prefix}{k}": v for k, v in custom_vae_sd.items()}
            else:
                prefix = "first_stage_model."
                processed_vae = {f"{prefix}{k}": v for k, v in custom_vae_sd.items()}
            sd.update(processed_vae)

        if discard_regex:
            pattern = re.compile(discard_regex)
            sd = {k: v for k, v in sd.items() if not pattern.search(k)}

        same_source_keys: set[str] = set()
        if output_format == "same":
            same_source_keys.update(unet_output_keys)
        if save_mode == "full" and clip_output_format == "same":
            same_source_keys.update(clip_output_keys)
        if save_mode == "full" and not is_custom_vae and vae_output_format == "same":
            same_source_keys.update(vae_output_keys)
        if same_source_keys:
            restored, precision_warning = try_match_source_dtypes(
                sd, primary_info.filename, same_source_keys, SAFETENSORS_FLOAT_DTYPES
            )
            if progress_cb and restored:
                progress_cb(f"Restored source precision for {restored} tensor(s).")
            if progress_cb and precision_warning:
                progress_cb(precision_warning)

        sd = to_cpu_contiguous_state_dict(sd)

        # `sd` now holds independent CPU copies of everything we need --
        # drop the live (GPU-resident) models before the save step, which
        # itself needs headroom to hold the whole checkpoint again while
        # writing. Skipping this risks a silent OOM on modest-RAM machines.
        engine_a = engine_b = engine_c = None
        diffusion_a = diffusion_b = diffusion_c = None
        clip_a = clip_b = clip_c = None
        vae_a = vae_b = vae_c = None
        memory_management.soft_empty_cache()
        gc.collect()

        metadata = _sanitize_metadata(
            _build_metadata(
                primary_info,
                secondary_info,
                tertiary_info,
                interp_method,
                multiplier,
                discard_regex,
                output_format,
                save_metadata,
                tuple(config_source),
                add_merge_recipe,
                save_mode=save_mode,
                loras=applied_loras if applied_loras else None,
                bake_vae=bake_vae,
                anima_remap=anima_remap_note,
            )
        )

        if progress_cb:
            progress_cb("Saving file...")
        save_checkpoint_file(sd, output_path, metadata=metadata)

        return {
            "output": output_path,
            "merged": {"unet": merged_unet, "clip": merged_clip, "vae": (len(custom_vae_sd) if is_custom_vae else merged_vae)},
            "skipped": {"unet": skipped_unet, "clip": skipped_clip, "vae": skipped_vae},
            "anima_remap": anima_remap_note,
            "output_format": output_format,
            "save_mode": save_mode,
            "loras": applied_loras,
            "baked_vae": os.path.basename(bake_vae) if is_custom_vae else ("none" if strip_vae else None),
        }
    finally:
        del engine_a, engine_b, engine_c
        memory_management.soft_empty_cache()
        gc.collect()
