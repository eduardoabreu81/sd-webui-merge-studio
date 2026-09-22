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

import torch
import torch.nn as nn

from backend import memory_management, utils
from backend.loader import forge_loader
from modules import sd_models, shared

from .lora_bake import (
    _import_lora_networks,
    _lora_activation_text,
    _lora_touches_llm_adapter,
    _pick_device,
)
from .checkpoint_inspector import load_custom_vae_state_dict
from .component_bundle import (
    ComponentSelection,
    ComponentValidationError,
    OutputValidation,
    build_component_plan,
    component_provenance,
    validate_aio_output,
    plan_merge_composition,
    preflight_component_plan,
    validate_loaded_components,
)
from .forge_capabilities import vae_key_prefix_for_saving
from . import anima_remap
from . import elemental_weights
from .quant_utils import LLM_ADAPTER_MODULE_NAMES, PLAIN_FORMATS, SAFETENSORS_FLOAT_DTYPES, convert_module_tree_precision, detect_incompatible_engine, fix_anima_state_dict_keys, save_checkpoint_file, set_module_weight, to_cpu_contiguous_state_dict, weight_as_float
from .precision_stats import dominant_float_dtype, match_dtype
from .source_precision import apply_component_precision, try_match_source_dtypes
from .merge_modes import (
    INTERP_NO_INTERPOLATION,
    MergeError,
    blend_tensors,
    delta_write,
    make_seeded_rand,
    merge_mode,
)

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


def _load_engine(
    checkpoint_path: str,
    additional_state_dicts=None,
    *,
    use_global_modules: bool = True,
):
    """Load a checkpoint, with either an explicit component list or Forge's.

    The traditional path keeps consulting `shared.opts.forge_additional_modules`
    so its outputs do not change. The AIO path always passes its own list and
    sets `use_global_modules=False`: what goes into the file has to be what was
    chosen, not whatever happened to be configured at the time.
    """
    if additional_state_dicts is None:
        additional_state_dicts = (
            shared.opts.forge_additional_modules if use_global_modules else []
        )
    return forge_loader(checkpoint_path, additional_state_dicts=list(additional_state_dicts))


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
    components: list[dict] | None = None,
    beta: float = 0.0,
    block_weights: str = "",
    seed: int = 0,
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
        if merge_mode(interp_method).needs_b:
            merge_recipe["multiplier"] = multiplier
        # Only the modes that have one. A beta in the recipe of a mode that
        # ignores it reads like a second ratio was applied, and none was.
        if merge_mode(interp_method).needs_beta:
            merge_recipe["beta"] = beta
        # In the syntax it was written in, so the Inspector reads it back with
        # the same parser and another tool can too.
        if block_weights and str(block_weights).strip():
            merge_recipe["block_weights"] = str(block_weights).strip()
        # Without this a DARE recipe cannot reproduce its own merge.
        if merge_mode(interp_method).needs_seed:
            merge_recipe["seed"] = seed
        if components:
            # What went into this AIO, so another run can reconstruct the
            # same composition instead of inheriting whatever is configured.
            merge_recipe["components"] = components
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
                **(
                    {"extend_rule": anima_remap["extend_rule"]}
                    if anima_remap.get("extend_rule")
                    else {}
                ),
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


def _parse_component_selections(raw) -> list[ComponentSelection]:
    """Accept the interface's plain dicts, or already-built selections."""
    out: list[ComponentSelection] = []
    for entry in raw or ():
        if isinstance(entry, ComponentSelection):
            out.append(entry)
            continue
        try:
            out.append(
                ComponentSelection(
                    slot_id=entry["slot_id"],
                    source=entry.get("source", "file"),
                    path=entry.get("path"),
                    output_format=entry.get("output_format", "same"),
                )
            )
        except (KeyError, TypeError) as e:
            raise MergeError(f"Malformed component selection: {entry!r} ({e}).") from e
    return out


def _compose_modular_primary(primary_info, selections, save_mode, progress_cb):
    """Resolve, preflight and reconcile the AIO composition for Model A.

    Returns the final composition and the loaded engine. Everything that can
    fail here fails before the merge starts, because a component problem found
    after the tensors have been blended costs the whole run.
    """
    from .checkpoint_inspector import infer_architecture_id, inspect_checkpoint

    info = inspect_checkpoint(primary_info.filename)
    if "error" in info:
        raise MergeError(f"Could not inspect Primary Model (A): {info['error']}")

    provisional = build_component_plan(
        info.get("architecture_id") or "unknown",
        primary_info.filename,
        info,
        selections,
    )

    if progress_cb:
        progress_cb("Loading Primary Model (A) with selected components...")
    try:
        engine_a = preflight_component_plan(primary_info.filename, provisional)
    except ComponentValidationError as e:
        raise MergeError(str(e)) from e

    try:
        plan = validate_loaded_components(engine_a, provisional)
        composition = plan_merge_composition(save_mode, selections, plan)
    except ComponentValidationError as e:
        # The engine is already resident at this point -- several GB of it.
        # Dropping it before raising keeps a rejected composition from holding
        # onto memory the next attempt is about to need.
        del engine_a
        memory_management.soft_empty_cache()
        gc.collect()
        raise MergeError(str(e)) from e

    return composition, engine_a


def _apply_plan_precision(state_dict, plan, kind: str, progress_cb=None) -> int:
    """Set each selected component's precision, scoped to its own namespace.

    Runs on the internal state dict, before the architecture stamps its save
    prefix on it. Two encoders in one bucket have keys that reduce to the same
    suffix, so the slot namespace is what keeps them apart -- and what keeps
    the Anima LLM Adapter out of reach, since Forge parked it in the encoder's
    bucket but it belongs to the diffusion model.
    """
    if plan is None:
        return 0

    changed = 0
    for component in plan.components:
        want_vae = component.slot_id == "vae"
        if (kind == "vae") != want_vae:
            continue
        if not component.internal_prefixes:
            continue
        changed += apply_component_precision(
            state_dict, component, component, SAFETENSORS_FLOAT_DTYPES
        )
    if changed and progress_cb:
        progress_cb(f"Applied component precision to {changed} tensor(s).")
    return changed


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
    # The delta write rule reads Model A's own kept floor from inside the
    # insert, and `_merge_module_tree` merges in block order -- so the floor
    # has to come first or the read straddles a merged and an unmerged block.
    # True for every published Anima mapping; checked rather than assumed.
    kept_for_insert = anima_remap.kept_target_for_insert(frozen, inserted)
    if not anima_remap.kept_precedes_insert(kept_for_insert):
        raise MergeError(
            f"Anima {blocks_b}-block -> {blocks_a}-block mapping puts an inserted block before the "
            f"floor it was copied from. The delta write rule cannot read a floor that has not been "
            f"merged yet; use the blend rule for this pairing."
        )
    return {
        "translate": anima_remap.make_name_translator(frozen),
        "extend": anima_remap.make_extend_translator(inserted),
        "kept": anima_remap.make_kept_translator(frozen, inserted),
        "from_blocks": blocks_b,
        "to_blocks": blocks_a,
        "frozen_blocks": len(frozen),
        "inserted_blocks": len(inserted),
        "message": (
            f"Anima cross-generation merge: remapping {blocks_b}-block -> {blocks_a}-block "
            f"({len(frozen)} shared blocks merged, {len(inserted)} inserted blocks kept from Model A)"
        ),
    }


def _delta_write(w_a, w_b, w_kept, ratio: float, dev):
    """`merge_modes.delta_write` with this module's device handling around it.

    The arithmetic lives in merge_modes so a machine without torch can check
    it; moving the tensor is this side's job, as everywhere else here."""
    if w_kept is not None and getattr(w_kept, "device", dev) != dev:
        w_kept = w_kept.to(device=dev)
    return delta_write(w_a, w_b, w_kept, ratio)


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
    kept_name=None,
    extend_rule: str = anima_remap.EXTEND_RULE_BLEND,
    beta: float = 0.0,
    weight_spec=None,
    block_index=None,
    rand=None,
) -> tuple[int, list[str]]:
    """translate_name: optional f(name_in_a) -> name_in_b_and_c, or None when
    the A-side module has no counterpart on the other side (used for
    cross-generation Anima merges, where B's block indices are shifted).
    Defaults to matching module names one-for-one.

    extend_name / extend_ratio: for A-side modules that translate_name maps to
    None (blocks the newer generation inserted, which have no counterpart),
    extend_name gives the B-side module the inserted block was originally
    copied from at initialization. With extend_ratio > 0 that module is
    blended in at that weight instead of the block being left untouched.

    kept_name / extend_rule: which write rule those inserted blocks get.
    `blend` lerps A's insert towards the B-side module extend_name found, the
    behaviour this has always had. `delta` instead writes

        insert += extend_ratio * (B[extend_name] - A[kept_name])

    -- the donor's own displacement from the floor it shares an origin with,
    never an average of two checkpoints inside one block. `kept_name` gives
    the A-side floor; see `anima_remap.make_kept_translator`. The delta rule
    is defined against a single donor, so it ignores Model C even in the
    three-model modes: the stem still merges on the mode's own terms, and only
    the inserts take the delta.

    beta: the second ratio, for the modes that have one. Ignored by the rest.

    rand: a seeded `rand_like` for the modes that draw random numbers. One
    generator per merge, advanced in `named_modules()` order, so the same
    seed reproduces the same file.

    weight_spec / block_index: per-block weights. `weight_spec` is a parsed
    `elemental_weights.WeightSpec` whose base is already the multiplier, and
    `block_index` turns a module path into the layer number the rules were
    written against. Both or neither -- with either missing the merge is
    uniform, which is what every merge was before this.
    """
    if module_a is None:
        return 0, []
    mode = merge_mode(interp_method)
    per_block = (
        weight_spec if weight_spec is not None and block_index is not None else None
    )
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
        delta_kept = None
        if lookup is None and extend_name is not None and extend_ratio > 0.0:
            lookup = extend_name(name)
            effective_multiplier = extend_ratio
            if extend_rule == anima_remap.EXTEND_RULE_DELTA:
                kept = kept_name(name) if kept_name is not None else None
                delta_kept = named_a.get(kept) if kept is not None else None
                if delta_kept is None:
                    # No floor to measure against means no defined delta. The
                    # block keeps A's weights, the same as extend_ratio 0.
                    skipped.append(name)
                    continue
        elif per_block is not None:
            # `elif`, not `if`: an inserted block's extend_ratio wins over a
            # per-block rule. The rule was written for a layer that exists in
            # both models, and an inserted one exists in neither Model B nor
            # the numbering the rule was written against.
            #
            # A module outside the numbered stack -- an embedder, the final
            # norm -- has no layer a rule could name, so it takes the spec's
            # base. Not the slider: when the rule text carries its own base
            # that is the one in force, and leaving these on the slider would
            # merge the stack and everything around it on different ratios.
            block = block_index(name)
            effective_multiplier = (
                per_block.effective_base
                if block is None
                else elemental_weights.resolve_weight(per_block, block, name)
            )
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

        w_c = None
        if mode.needs_c and delta_kept is None:
            m_c = named_c.get(lookup)
            w_c = weight_as_float(m_c) if m_c is not None else None
            if w_c is None or w_c.shape != w_a.shape:
                skipped.append(name)
                continue
            if w_c.device != dev:
                w_c = w_c.to(device=dev)

        if delta_kept is not None:
            merged = _delta_write(w_a, w_b, weight_as_float(delta_kept), effective_multiplier, dev)
            if merged is None:
                skipped.append(name)
                del w_a, w_b
                continue
        else:
            merged = blend_tensors(
                interp_method, effective_multiplier, beta, w_a, w_b, w_c,
                xp=torch, rand=rand,
            )
        del w_c

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

                b_c_f = None
                if mode.needs_c and delta_kept is None:
                    m_c = named_c.get(lookup)
                    b_c = getattr(m_c, "bias", None) if m_c is not None else None
                    if b_c is not None and b_c.shape == b_a.shape:
                        b_c_f = b_c.data.float()
                        if b_c_f.device != dev:
                            b_c_f = b_c_f.to(device=dev)
                # A module whose weight merged but whose bias has no
                # counterpart in C leaves the bias alone rather than blending
                # it on different terms from the weight beside it.
                if delta_kept is not None:
                    kept_bias = getattr(delta_kept, "bias", None)
                    new_bias = _delta_write(
                        b_a, b_b_f,
                        kept_bias.data.float() if kept_bias is not None else None,
                        effective_multiplier, dev,
                    )
                else:
                    new_bias = blend_tensors(
                        interp_method, effective_multiplier, beta, b_a, b_b_f, b_c_f,
                        xp=torch, rand=rand,
                    )
                del b_c_f
                if new_bias is not None:
                    m_a.bias = nn.Parameter(new_bias.to(dtype=m_a.bias.dtype, device=dev), requires_grad=False)
                del b_a, b_b_f

        if progress_cb:
            progress_cb(i + 1, total, name)

    return merged_count, skipped


def _per_block_spec(
    block_weights: str, multiplier: float, engine_a, diffusion_a, interp_method: str
):
    """The parsed per-block rules for this merge, or None for a uniform one.

    Refuses rather than ignores. Per-block weights are enabled for Anima only
    -- `L00-L27` assumes one numbered stack, which SDXL's three sections and
    Flux's two parallel series are not -- and a recipe carrying rules that
    silently did nothing would produce a uniform merge under a name that says
    otherwise.
    """
    if not block_weights or not str(block_weights).strip():
        return None

    if not merge_mode(interp_method).needs_b:
        raise MergeError(
            f"{merge_mode(interp_method).label} copies Model A through without "
            "blending, so there is no multiplier for a per-block rule to "
            "replace. Clear the rules, or pick a mode that merges."
        )

    if not anima_remap.is_anima_engine(engine_a):
        raise MergeError(
            "Per-block weights are for Anima only. The layer syntax assumes one "
            "numbered stack of blocks, which this architecture does not have. "
            "Clear the per-block rules to merge it uniformly."
        )

    spec = elemental_weights.spec_with_base(
        elemental_weights.parse_weight_spec(block_weights), multiplier
    )
    problems = elemental_weights.validate_spec(
        spec, anima_remap.block_count(diffusion_a)
    )
    fatal = [p for p in problems if "still applies" not in p]
    if fatal:
        raise MergeError("Per-block weights: " + " ".join(fatal))
    if not spec.rules:
        return None
    return spec


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
    anima_extend_rule: str = anima_remap.EXTEND_RULE_BLEND,
    component_selections: list[dict] | None = None,
    progress_cb=None,
    beta: float = 0.0,
    block_weights: str = "",
    seed: int = 0,
) -> dict:
    # Which models a mode needs is a property of the mode, declared once in
    # MERGE_MODES rather than restated here every time one is added.
    mode = merge_mode(interp_method)
    if mode.needs_b and not secondary_name:
        raise MergeError(f"{mode.label} requires a Secondary Model (B).")
    if mode.needs_c and not tertiary_name:
        raise MergeError(f"{mode.label} requires a Tertiary Model (C).")

    # Decided before anything expensive happens. With no selections this is the
    # traditional path, unchanged down to the global module list it reads.
    selections = _parse_component_selections(component_selections)
    try:
        composition = plan_merge_composition(save_mode, selections)
    except ComponentValidationError as e:
        raise MergeError(str(e)) from e

    primary_info = _checkpoint_info(primary_name)
    secondary_info = _checkpoint_info(secondary_name) if secondary_name else None
    tertiary_info = _checkpoint_info(tertiary_name) if tertiary_name else None

    # Fast architecture compatibility check before heavy engine loading
    try:
        from .checkpoint_inspector import inspect_checkpoint, get_model_family
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

    # An AIO composition is resolved and preflighted before any merge work, so
    # a missing component costs a load rather than a whole merge. The engine it
    # loads is the one the merge then uses -- loading A twice would double the
    # most expensive step for nothing.
    if composition.modular:
        if progress_cb:
            progress_cb("Resolving components...")
        composition, engine_a = _compose_modular_primary(
            primary_info, selections, save_mode, progress_cb
        )
    else:
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
            engine_b = _load_engine(
                secondary_info.filename,
                use_global_modules=composition.secondary_uses_global_modules,
            )
            incompat = detect_incompatible_engine(engine_b)
            if incompat:
                raise MergeError(f"Secondary Model (B) is incompatible with this merge: {incompat}.")
        if tertiary_info:
            if progress_cb:
                progress_cb("Loading Tertiary Model (C)...")
            engine_c = _load_engine(
                tertiary_info.filename,
                use_global_modules=composition.secondary_uses_global_modules,
            )
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
        unet_kept = anima_remap_note.pop("kept", None) if anima_remap_note else None
        if anima_extend_rule not in anima_remap.EXTEND_RULES:
            raise MergeError(
                f"Unknown inserted-block write rule {anima_extend_rule!r} "
                f"(expected one of {', '.join(anima_remap.EXTEND_RULES)})."
            )
        if anima_remap_note:
            anima_remap_note["extend_ratio"] = anima_extend_ratio
            # Recorded only when it governed something: at ratio 0 no inserted
            # block is written at all, so naming a rule there would put a
            # decision in the recipe that never ran.
            anima_remap_note["extend_rule"] = (
                anima_extend_rule if anima_extend_ratio > 0.0 else ""
            )
            if anima_extend_ratio > 0.0:
                anima_remap_note["message"] += (
                    f", inserted blocks written with the {anima_extend_rule} rule "
                    f"at extend_ratio={anima_extend_ratio}"
                )
            if progress_cb:
                progress_cb(anima_remap_note["message"])

        # Per-block weights apply to the diffusion model and nowhere else:
        # the layer syntax names positions in one numbered stack, and neither
        # the text encoder nor the VAE has one.
        # One generator for the whole merge, so the same seed reproduces the
        # same file. Only the stochastic modes ask for it.
        merge_rand = make_seeded_rand(seed, torch, target_device) if mode.needs_seed else None

        unet_weight_spec = _per_block_spec(
            block_weights, multiplier, engine_a, diffusion_a, interp_method
        )
        if unet_weight_spec is not None and progress_cb:
            progress_cb(
                f"Per-block weights: {len(unet_weight_spec.rules)} rule(s) over a "
                f"base of {unet_weight_spec.effective_base:g}"
            )

        if progress_cb:
            progress_cb("Merging diffusion model...")
        merged_unet, skipped_unet = _merge_module_tree(
            diffusion_a, diffusion_b, diffusion_c, interp_method, multiplier, beta=beta,
            target_device=target_device,
            progress_cb=(lambda i, t, n: progress_cb(f"Merging UNet ({i}/{t}): {n}")) if progress_cb else None,
            translate_name=unet_translate,
            extend_name=unet_extend,
            extend_ratio=anima_extend_ratio,
            kept_name=unet_kept,
            extend_rule=anima_extend_rule,
            rand=merge_rand,
            weight_spec=unet_weight_spec,
            block_index=anima_remap.main_block_index if unet_weight_spec else None,
        )

        if clip_a is not None and composition.merge_text_encoder:
            if progress_cb:
                progress_cb("Merging text encoder...")
            merged_clip, skipped_clip = _merge_module_tree(
                clip_a, clip_b, clip_c, interp_method, multiplier, beta=beta,
                rand=merge_rand,
                target_device=target_device,
                progress_cb=(lambda i, t, n: progress_cb(f"Merging CLIP ({i}/{t}): {n}")) if progress_cb else None,
            )
        else:
            merged_clip, skipped_clip = 0, []

        # On the AIO path the VAE slot replaces Bake VAE; keeping both would
        # be two controls setting the same thing.
        is_custom_vae = composition.allow_bake_vae and bool(
            bake_vae and bake_vae not in ("original", "none", "")
        )
        strip_vae = composition.allow_bake_vae and bake_vae == "none"

        if composition.merge_vae and not is_custom_vae and not strip_vae and vae_a is not None:
            if progress_cb:
                progress_cb("Merging VAE...")
            merged_vae, skipped_vae = _merge_module_tree(
                vae_a, vae_b, vae_c, interp_method, multiplier, beta=beta,
                rand=merge_rand,
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
                skip_names=LLM_ADAPTER_MODULE_NAMES,
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
                # Per-component precision runs on the internal state dict,
                # before the architecture applies its save namespace: the slot
                # namespaces are the internal ones, and after serialisation
                # they are gone.
                _apply_plan_precision(clip_sd, composition.plan, "text_encoder", progress_cb)
                processed_clip = fix_anima_state_dict_keys(
                    engine_a.model_config.process_clip_state_dict_for_saving(clip_sd)
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
            if not is_custom_vae and not strip_vae and vae_a is not None:
                vae_sd = utils.get_state_dict_after_quant(vae_a)
                vae_sd.update(vae_overrides)
                _apply_plan_precision(vae_sd, composition.plan, "vae", progress_cb)
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
            config = getattr(engine_a, "model_config", None)
            processed_vae = None
            if hasattr(config, "process_vae_state_dict_for_saving"):
                try:
                    processed_vae = config.process_vae_state_dict_for_saving(custom_vae_sd)
                except Exception:
                    processed_vae = None
            if processed_vae is None:
                # The architecture declares its own VAE namespace; read it from
                # there rather than testing the config's class name.
                prefix = vae_key_prefix_for_saving(config)
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
                components=component_provenance(composition.plan, with_hashes=True),
                beta=beta,
                block_weights=block_weights,
                seed=seed,
            )
        )

        if progress_cb:
            progress_cb("Saving file...")
        save_checkpoint_file(sd, output_path, metadata=metadata)

        # An AIO is only an AIO if it reopens on its own. The file is never
        # deleted or rewritten when this fails -- it is kept, clearly marked
        # as unvalidated, with the reasons intact.
        if composition.modular:
            if progress_cb:
                progress_cb("Verifying the saved checkpoint reopens on its own...")
            validation = validate_aio_output(output_path, composition.plan)
            if progress_cb and not validation.validated:
                progress_cb("Saved, but NOT validated as a self-contained AIO.")
        else:
            validation = OutputValidation(validated=True)

        return {
            "output": output_path,
            "merged": {"unet": merged_unet, "clip": merged_clip, "vae": (len(custom_vae_sd) if is_custom_vae else merged_vae)},
            "skipped": {"unet": skipped_unet, "clip": skipped_clip, "vae": skipped_vae},
            "anima_remap": anima_remap_note,
            "output_format": output_format,
            "save_mode": save_mode,
            "loras": applied_loras,
            "baked_vae": os.path.basename(bake_vae) if is_custom_vae else ("none" if strip_vae else None),
            "modular_full": composition.modular,
            "validated": validation.validated,
            "validation": validation,
            "component_plan": composition.plan,
            "components_attached": component_provenance(composition.plan),
        }
    finally:
        del engine_a, engine_b, engine_c
        memory_management.soft_empty_cache()
        gc.collect()
