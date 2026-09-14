"""Standalone precision/quantization conversion for a single checkpoint --
no LoRA, no merge. Loads the checkpoint, converts the diffusion model's
weights to the chosen output format, and saves a new file. CLIP and VAE are
kept untouched, matching how real Anima int8 builds work in practice (only
the diffusion model is quantized).
"""

from __future__ import annotations

import gc
import json
import os

from backend import memory_management, utils
from backend.loader import forge_loader
from modules import shared

from quant_utils import convert_module_tree_precision, detect_incompatible_engine, save_checkpoint_file, to_cpu_contiguous_state_dict


class QuantizeError(RuntimeError):
    pass


def _sanitize_metadata(metadata: dict) -> dict[str, str]:
    out = {}
    for k, v in metadata.items():
        if v is None:
            continue
        out[k] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
    return out


def quantize_checkpoint(checkpoint_path: str, output_path: str, output_format: str, save_mode: str = "unet_only", progress_cb=None) -> dict:
    """save_mode: "unet_only" (default) writes just the converted diffusion
    model -- matches the native "Save UNet Only" button, and is what you
    want if you already load this checkpoint's VAE/text encoder as separate
    files (shared.opts.forge_additional_modules); "full" also bundles the
    (untouched) CLIP and VAE into the same file, self-contained but much
    bigger, since they're not converted/shrunk here."""
    if output_format == "same":
        raise QuantizeError("Pick an output format other than 'same as source' -- there would be nothing to do.")

    if progress_cb:
        progress_cb("Loading checkpoint...")
    engine = forge_loader(checkpoint_path, additional_state_dicts=shared.opts.forge_additional_modules)

    try:
        incompat = detect_incompatible_engine(engine)
        if incompat:
            raise QuantizeError(f"Checkpoint incompatible with quantization: {incompat}.")

        diffusion_model = engine.forge_objects.unet.model.diffusion_model

        if progress_cb:
            progress_cb(f"Converting diffusion model to {output_format}...")
        converted, overrides = convert_module_tree_precision(
            diffusion_model,
            output_format,
            progress_cb=(lambda i, total, name: progress_cb(f"Quantizing ({i}/{total}): {name}")) if progress_cb else None,
        )

        if progress_cb:
            progress_cb("Assembling final state dict...")

        # Merge overrides on top of the plain state dict BEFORE prefixing --
        # see convert_module_tree_precision's docstring: a converted layer's
        # own state_dict() entry is not trustworthy on its own (it can
        # silently be a dequantized plain tensor).
        unet_sd = utils.get_state_dict_after_quant(diffusion_model)
        unet_sd.update(overrides)

        sd = {}
        sd.update(engine.model_config.process_unet_state_dict_for_saving(unet_sd))
        if save_mode == "full":
            sd.update(engine.model_config.process_clip_state_dict_for_saving(utils.get_state_dict_after_quant(engine.forge_objects.clip.cond_stage_model)))
            sd.update(engine.model_config.process_vae_state_dict_for_saving(utils.get_state_dict_after_quant(engine.forge_objects.vae.first_stage_model)))
        else:
            # For Anima, llm_adapter was moved into clip.cond_stage_model by loader.py,
            # but on disk it is part of Anima's DiT (model.diffusion_model.llm_adapter.*).
            # We must include it even in unet_only mode!
            clip_sd = utils.get_state_dict_after_quant(engine.forge_objects.clip.cond_stage_model)
            for k, v in clip_sd.items():
                if "llm_adapter" in k:
                    suffix = k[k.index("llm_adapter") :]
                    sd[f"model.diffusion_model.{suffix}"] = v
        sd = to_cpu_contiguous_state_dict(sd)

        # `sd` now holds independent CPU copies of everything we need --
        # drop the live (GPU-resident) model before the save step, which
        # itself needs headroom to hold the whole checkpoint again while
        # writing. Skipping this risks a silent OOM on modest-RAM machines.
        engine = None
        diffusion_model = None
        memory_management.soft_empty_cache()
        gc.collect()

        metadata = _sanitize_metadata(
            {
                "sd_checkpoint_doctor_recipe": {
                    "type": "CheckpointDoctor-Quantize",
                    "base_model": os.path.basename(checkpoint_path),
                    "save_mode": save_mode,
                    "output_format": output_format,
                }
            }
        )

        if progress_cb:
            progress_cb("Saving file...")
        save_checkpoint_file(sd, output_path, metadata=metadata)

        return {"output": output_path, "converted_layers": converted, "output_format": output_format}
    finally:
        del engine
        memory_management.soft_empty_cache()
        gc.collect()
