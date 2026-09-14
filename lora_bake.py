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
from modules import extra_networks, shared

from checkpoint_inspector import load_custom_vae_state_dict
from quant_utils import PLAIN_FORMATS, convert_module_tree_precision, debug_print, detect_incompatible_engine, save_checkpoint_file, to_cpu_contiguous_state_dict


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


def _lora_activation_text(lora_path: str) -> str:
    """Trigger word(s) for this LoRA, if the user set them in the Lora tab's
    metadata editor ("Activation Text" field). This lives in a sidecar
    <lora_name>.json file, not in the LoRA's weight tensors -- baking the
    LoRA does not remove the need to type this in the prompt, since the
    baked weights still expect it exactly like the live LoRA did."""
    try:
        metadata = extra_networks.get_user_metadata(lora_path)
        return (metadata.get("activation text") or "").strip()
    except Exception:
        return ""


def _is_anima_engine(engine) -> bool:
    return type(engine.model_config).__name__ == "Anima"


def _normalize_activation_text(activation_text: str, engine) -> str:
    """Anima's own training convention uses lowercase tags with spaces
    instead of underscores (e.g. "hatsune miku", not raw Danbooru's
    "hatsune_miku") -- normalize the reminder we embed so it actually
    matches what the model was trained to expect in the prompt."""
    if not activation_text or not _is_anima_engine(engine):
        return activation_text
    return activation_text.replace("_", " ").lower()


def _lora_touches_llm_adapter(lora_sd: dict) -> bool:
    """Anima's own training guidance says to never train the LLM (Qwen3)
    adapter alongside a LoRA. A LoRA that still carries llm_adapter weights
    (e.g. trained incorrectly, or with a script that doesn't follow that
    guidance) could degrade the baked checkpoint's text understanding, so we
    flag it instead of silently applying it. Matches both key conventions
    used by extensions-builtin/sd_forge_lora/networks.py::process_anima
    ("diffusion_model.llm_adapter" and "lora_unet_llm_adapter")."""
    return any("llm_adapter" in k for k in lora_sd)


def _write_checkpoint_notes(output_path: str, notes: str) -> None:
    """Writes a sidecar <output>.json with a Notes field, so the baked
    checkpoint's activation-text reminder shows up in the Checkpoints tab
    the same way it would for a LoRA."""
    if not notes:
        return
    metadata_path = os.path.splitext(output_path)[0] + ".json"
    try:
        existing = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing["notes"] = (existing.get("notes", "") + "\n" + notes).strip()
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4, ensure_ascii=False)
    except Exception as e:
        debug_print(f"Could not write sidecar metadata for {output_path}: {e}")


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
            activation_text = _lora_activation_text(lora_path)
            applied.append(
                {
                    "name": os.path.basename(lora_path),
                    "strength": strength,
                    "activation_text": _normalize_activation_text(activation_text, engine),
                    "activation_text_raw": activation_text,
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

        sd = {}
        sd.update(engine.model_config.process_unet_state_dict_for_saving(unet_sd))
        if save_full:
            clip_sd = utils.get_state_dict_after_quant(clip.cond_stage_model)
            clip_sd.update(clip_overrides)
            sd.update(engine.model_config.process_clip_state_dict_for_saving(clip_sd))
            if not is_custom_vae and not strip_vae and has_vae_model:
                vae_sd = utils.get_state_dict_after_quant(engine.forge_objects.vae.first_stage_model)
                vae_sd.update(vae_overrides)
                sd.update(engine.model_config.process_vae_state_dict_for_saving(vae_sd))
        else:
            # For Anima, llm_adapter was moved into clip.cond_stage_model by loader.py,
            # but on disk it is part of Anima's DiT (model.diffusion_model.llm_adapter.*).
            # We must include it even in unet_only mode!
            clip_sd = utils.get_state_dict_after_quant(clip.cond_stage_model)
            for k, v in clip_sd.items():
                if "llm_adapter" in k:
                    suffix = k[k.index("llm_adapter") :]
                    sd[f"model.diffusion_model.{suffix}"] = v

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

        note_lines = []
        for a in applied:
            if a["activation_text"]:
                line = f"LoRA '{a['name']}' (strength {a['strength']}) — activation text: {a['activation_text']}"
                if a["activation_text"] != a["activation_text_raw"]:
                    line += f" (normalized from '{a['activation_text_raw']}' to match Anima's tag convention)"
                note_lines.append(line)
            if a["llm_adapter_warning"]:
                note_lines.append(f"WARNING: LoRA '{a['name']}' contains LLM adapter weights -- Anima's own training guidance says never to train these alongside a LoRA.")

        if note_lines:
            _write_checkpoint_notes(output_path, "Baked-in LoRA activation text(s), still required in the prompt:\n" + "\n".join(note_lines))

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
