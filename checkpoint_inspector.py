"""Fast checkpoint component, architecture, and merge recipe inspector.

Inspects .safetensors headers in < 5ms without loading tensor weights into RAM/VRAM.
Detects:
  - Components: UNet / DiT, Text Encoder (CLIP/T5), VAE, LLM adapter
  - Architecture: SD 1.5, SDXL, Flux, Anima / Wan2.1, SD3, etc.
  - Precision / Quantization: FP16, BF16, FP8, INT8 (convrot / tensorwise)
  - Merge Recipe: provenance, parent models (names + SHA256 hashes),
    interpolation method, multiplier, baked LoRAs, recursive merge history.
"""

from __future__ import annotations

import json
import html as html_lib
import os
import struct
from typing import Any

from anima_remap import block_count_from_keys


def _esc(value: Any) -> str:
    """Escapes a value read out of a checkpoint for interpolation into the
    dashboard HTML.

    Everything these dashboards display -- recipe methods, parent model names,
    LoRA names, workflow node types, the raw metadata dump -- is attacker-shaped
    text that travels inside a downloaded .safetensors header. Without this it
    is markup, and the Gradio page renders whatever the file author wrote.
    """
    return html_lib.escape(str(value), quote=True)


def read_safetensors_header(path: str) -> tuple[dict[str, Any], int]:
    """Reads the JSON header from a .safetensors file without loading weights."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if not path.endswith(".safetensors"):
        raise ValueError("Only .safetensors files support fast header inspection.")

    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("Corrupted or empty file.")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        raw_header = f.read(header_len)
        header = json.loads(raw_header)
    return header, 8 + header_len


def _detect_architecture(
    keys: list[str],
    has_llm_adapter: bool,
    metadata: dict[str, Any] | None = None,
    filename: str = "",
) -> str:
    fn_lower = filename.lower()
    meta_str = json.dumps(metadata, ensure_ascii=False).lower() if metadata else ""

    if any(k.startswith(("double_blocks.", "model.diffusion_model.double_blocks.", "img_in.")) for k in keys):
        return "Flux (MMDiT)"
    if any(k.startswith(("model.diffusion_model.joint_blocks.", "joint_blocks.")) for k in keys):
        return "SD3 / SD3.5 (MMDiT)"
    if has_llm_adapter or any(k.startswith("net.") for k in keys) or any("qwen" in k for k in keys) or "anima" in fn_lower or "anima" in meta_str:
        return "Anima (DiT)"
    if any(k.startswith(("v_blocks.", "model.diffusion_model.v_blocks.", "head.weight")) for k in keys):
        return "Wan2.1 (DiT)"
    if any(k.startswith("model.diffusion_model.blocks.") for k in keys):
        return "DiT / Diffusion Model"

    is_sdxl = any(k.startswith("conditioner.embedders.") for k in keys)
    has_stage_3 = False
    if not is_sdxl and any(k.startswith(("model.diffusion_model.input_blocks.", "diffusion_model.input_blocks.", "input_blocks.")) for k in keys):
        has_stage_3 = any("input_blocks.9." in k or "input_blocks.10." in k or "input_blocks.11." in k for k in keys)
        if not has_stage_3:
            is_sdxl = True

    if is_sdxl:
        if "pony" in fn_lower or "pony" in meta_str or "ponydiffusion" in meta_str:
            return "Pony (SDXL)"
        if "illustrious" in fn_lower or "illustrious" in meta_str or "il_v" in fn_lower or "il-v" in fn_lower:
            return "Illustrious (SDXL)"
        return "SDXL (UNet)"

    if any(k.startswith("cond_stage_model.") for k in keys) or has_stage_3:
        return "SD 1.5 / SD 2.1 (UNet)"

    return "Diffusion Model"


def get_model_family(arch: str) -> str:
    """Returns the base architecture family for compatibility verification."""
    if not arch:
        return "other"
    arch_lower = arch.lower()
    if "anima" in arch_lower:
        return "anima"
    if "flux" in arch_lower:
        return "flux"
    if "wan" in arch_lower:
        return "wan"
    if "sd3" in arch_lower:
        return "sd3"
    if "sdxl" in arch_lower or "pony" in arch_lower or "illustrious" in arch_lower:
        return "sdxl"
    if "sd 1.5" in arch_lower or "sd 2.1" in arch_lower:
        return "sd1"
    return "other"



# backend/quant_ops.py::QUANT_ALGOS keys, in the vocabulary the output-format
# dropdown already uses. int8_tensorwise is listed twice because the rotation is
# what distinguishes the two builds a user can choose between.
_QUANT_FORMAT_LABELS = {
    ("int8_tensorwise", True): "INT8 convrot",
    ("int8_tensorwise", False): "INT8 tensor-wise",
    ("float8_e4m3fn", False): "FP8 (e4m3fn)",
    ("float8_e5m2", False): "FP8 (e5m2)",
    ("nvfp4", False): "NVFP4",
    ("mxfp8", False): "MXFP8",
    ("convrot_w4a4", True): "INT4 convrot W4A4",
    ("convrot_w4a4", False): "INT4 W4A4",
    ("asym_w4a8_int8", False): "W4A8 asym INT8",
}

# Enough layers to notice a checkpoint quantized in more than one format,
# without walking thousands of tiny reads over a network share.
_QUANT_SAMPLE_LAYERS = 24


def _read_quant_configs(
    filepath: str, header: dict[str, Any], data_offset: int
) -> set[tuple[str, bool]]:
    """(format, uses convrot) for a sample of the quantized layers.

    Each quantized layer carries a ``comfy_quant`` tensor that is not weights at
    all: it is the JSON config Forge wrote for that layer, stored as uint8
    (operations_mixed_precision.py:194). The header alone gives only its size,
    so the rotation that separates "INT8 (convrot: per-channel + rotation)" from
    plain tensor-wise INT8 -- two different entries in this extension's own
    output-format dropdown -- is invisible until the bytes are read.
    """
    found: set[tuple[str, bool]] = set()
    keys = [k for k in header if k.endswith(".comfy_quant")][:_QUANT_SAMPLE_LAYERS]
    if not keys:
        return found
    try:
        with open(filepath, "rb") as f:
            for key in keys:
                info = header.get(key)
                if not isinstance(info, dict):
                    continue
                start, end = info["data_offsets"]
                if end - start > 4096:
                    continue
                f.seek(data_offset + start)
                conf = json.loads(f.read(end - start).decode("utf-8"))
                fmt = conf.get("format")
                if isinstance(fmt, str) and fmt:
                    found.add((fmt, bool(conf.get("convrot"))))
    except Exception:
        # A config we cannot read is one we do not report; the dtype-based
        # fallback still names the family.
        return found
    return found


def _detect_precision(
    header: dict[str, Any], filepath: str = "", data_offset: int = 0
) -> str:
    # Check comfy_quant presence first
    has_comfy_quant = any(k.endswith(".comfy_quant") for k in header)
    if has_comfy_quant:
        if filepath:
            configs = _read_quant_configs(filepath, header, data_offset)
            labels = sorted(
                _QUANT_FORMAT_LABELS.get((fmt, convrot), fmt) for fmt, convrot in configs
            )
            if len(labels) == 1:
                return f"{labels[0]} (comfy_quant)"
            if labels:
                return f"Mixed: {', '.join(labels)} (comfy_quant)"

        format_name = None
        for k in header:
            if k.endswith(".comfy_quant"):
                prefix = k[:-12]
                w_info = header.get(f"{prefix}.weight")
                if w_info and w_info.get("dtype") in ("I8", "U8"):
                    format_name = "INT8"
                elif w_info and "F8" in w_info.get("dtype", ""):
                    format_name = "FP8"
        return f"{format_name or 'Quantized'} (comfy_quant)"

    # Inspect weight dtypes
    dtypes = set()
    for k, v in header.items():
        if k == "__metadata__":
            continue
        if isinstance(v, dict) and k.endswith(".weight") and "dtype" in v:
            dtypes.add(v["dtype"])

    if "F8_E4M3" in dtypes or "F8_E4M3FN" in dtypes:
        return "FP8 (e4m3fn)"
    if "F8_E5M2" in dtypes:
        return "FP8 (e5m2)"
    if "BF16" in dtypes:
        return "BF16"
    if "F16" in dtypes:
        return "FP16"
    if "I8" in dtypes or "U8" in dtypes:
        return "INT8"
    return "/".join(sorted(dtypes)) if dtypes else "Unknown"


def _model_name(value: Any) -> str | None:
    """A ComfyUI node input is either a literal or a link to another node,
    and a link is serialised as [node_id, output_index]. Only a literal names
    a model, so anything else is dropped -- otherwise a wired-up input reaches
    the dedup step as an unhashable list and takes the whole inspection down.
    """
    return value if isinstance(value, str) and value else None


def _parse_comfy_recipe(metadata: dict[str, Any]) -> dict[str, Any] | None:
    prompt_raw = metadata.get("prompt")
    wf_raw = metadata.get("workflow")
    if not prompt_raw and not wf_raw:
        return None

    prompt = {}
    if prompt_raw:
        try:
            prompt = json.loads(prompt_raw) if isinstance(prompt_raw, str) else prompt_raw
        except Exception:
            pass

    wf = {}
    if wf_raw:
        try:
            wf = json.loads(wf_raw) if isinstance(wf_raw, str) else wf_raw
        except Exception:
            pass

    hashes = {}
    if isinstance(wf, dict):
        extra = wf.get("extra", {})
        if isinstance(extra, dict):
            hashes = extra.get("anomalous_hashes", {})

    base_models = []
    loras = []
    merges = []
    encoders = []
    vaes = []

    if isinstance(prompt, dict):
        for nid, node in prompt.items():
            if not isinstance(node, dict):
                continue
            ctype = node.get("class_type", "")
            inp = node.get("inputs", {})

            if ctype in ("UNETLoader", "CheckpointLoaderSimple", "CheckpointLoader", "DiffusersLoader"):
                m_name = _model_name(inp.get("unet_name") or inp.get("ckpt_name") or inp.get("model_path"))
                if m_name:
                    h = ""
                    for k, v in hashes.items():
                        if m_name in k and isinstance(v, dict):
                            h = v.get("hash", "")
                            break
                    base_models.append({"name": m_name, "node": ctype, "hash": h})

            elif ctype in ("CLIPLoader", "DualCLIPLoader"):
                c_name = _model_name(inp.get("clip_name") or inp.get("clip_name1"))
                if c_name:
                    encoders.append(c_name)

            elif ctype == "VAELoader":
                v_name = _model_name(inp.get("vae_name"))
                if v_name:
                    vaes.append(v_name)

            elif ctype in ("LoraLoaderModelOnly", "LoraLoader", "LoraLoaderBlockWeight"):
                l_name = _model_name(inp.get("lora_name"))
                l_str = inp.get("strength_model", inp.get("strength", 1.0))
                if l_name:
                    h = ""
                    for k, v in hashes.items():
                        if l_name in k and isinstance(v, dict):
                            h = v.get("hash", "")
                            break
                    loras.append({"name": l_name, "strength": l_str, "hash": h})

            elif "ModelMerge" in ctype:
                ratio = inp.get("ratio", "N/A")
                merges.append({"type": ctype, "ratio": ratio})

        if not loras:
            for nid, node in prompt.items():
                if not isinstance(node, dict):
                    continue
                ctype = node.get("class_type", "")
                inp = node.get("inputs", {})
                if "Power Lora" in ctype or "PowerLora" in ctype or "CR LoRA Stack" in ctype:
                    for k, v in inp.items():
                        if isinstance(v, dict) and v.get("on", True) and v.get("lora"):
                            l_name = _model_name(v.get("lora"))
                            l_str = v.get("strength", 1.0)
                            h = ""
                            for hk, hv in hashes.items():
                                if l_name in hk and isinstance(hv, dict):
                                    h = hv.get("hash", "")
                                    break
                            loras.append({"name": l_name, "strength": l_str, "hash": h})

    unique_loras = []
    seen = set()
    for l in loras:
        key = (l["name"], l["strength"])
        if key not in seen:
            seen.add(key)
            unique_loras.append(l)

    unique_base = []
    seen_base = set()
    for b in base_models:
        if b["name"] not in seen_base:
            seen_base.add(b["name"])
            unique_base.append(b)

    if not unique_base and not unique_loras and not merges:
        return None

    return {
        "source": "ComfyUI Workflow",
        "base_models": unique_base,
        "loras": unique_loras,
        "merges": merges,
        "encoders": list(dict.fromkeys(encoders)),
        "vaes": list(dict.fromkeys(vaes)),
    }


def inspect_checkpoint(filepath: str) -> dict[str, Any]:
    """Inspects a checkpoint's .safetensors header and returns structured metadata."""
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}

    if not filepath.endswith(".safetensors"):
        return {
            "error": "Non-safetensors format (.ckpt/.pt). Header inspection is only supported on .safetensors files."
        }

    try:
        header, data_offset = read_safetensors_header(filepath)
    except Exception as e:
        return {"error": f"Failed to read header: {e}"}

    metadata = header.get("__metadata__", {})
    keys = [k for k in header.keys() if k != "__metadata__"]

    DIFFUSION_MODEL_PREFIXES = (
        "model.diffusion_model.",
        "diffusion_model.",
        "double_blocks.",
        "single_blocks.",
        "joint_blocks.",
        "transformer_blocks.",
        "transformer.",
        "net.",
        "blocks.",
        "v_blocks.",
        "layers.",
        "input_blocks.",
        "middle_block.",
        "output_blocks.",
        "img_in.",
    )
    TEXT_ENCODER_PREFIXES = (
        "cond_stage_model.",
        "conditioner.embedders.",
        "text_encoders.",
        "text_model.",
        "clip_l.",
        "clip_g.",
        "t5xxl.",
        "text_encoder.",
        "text_encoder_2.",
        "text_encoder_3.",
        "qwen2.",
        "qwen3.",
        "encoder.block.",
    )
    VAE_PREFIXES = (
        "first_stage_model.",
        "vae.",
        "vae_model.",
        "autoencoder.",
    )

    has_unet = any(k.startswith(DIFFUSION_MODEL_PREFIXES) for k in keys)
    has_clip = any(k.startswith(TEXT_ENCODER_PREFIXES) for k in keys)
    has_vae = any(k.startswith(VAE_PREFIXES) for k in keys)
    has_llm_adapter = any("llm_adapter" in k for k in keys)

    filename = os.path.basename(filepath)
    arch = _detect_architecture(keys, has_llm_adapter, metadata=metadata, filename=filename)
    precision = _detect_precision(header, filepath, data_offset)

    # Parse sd_merge_recipe if present
    recipe = None
    if "sd_merge_recipe" in metadata:
        raw_r = metadata["sd_merge_recipe"]
        try:
            recipe = json.loads(raw_r) if isinstance(raw_r, str) else raw_r
        except Exception:
            recipe = {"raw": str(raw_r)}

    # Parse sd_merge_models if present
    models = None
    if "sd_merge_models" in metadata:
        raw_m = metadata["sd_merge_models"]
        try:
            models = json.loads(raw_m) if isinstance(raw_m, str) else raw_m
        except Exception:
            models = {"raw": str(raw_m)}

    comfy_recipe = _parse_comfy_recipe(metadata)

    file_size = os.path.getsize(filepath)
    gb = file_size / (1024 ** 3)
    mb = file_size / (1024 ** 2)
    size_str = f"{gb:.2f} GB ({mb:,.0f} MB)" if gb >= 1.0 else f"{mb:.1f} MB"

    info_dict = {
        "filename": filename,
        "filepath": filepath,
        "file_size": file_size,
        "size_str": size_str,
        "architecture": arch,
        "block_count": block_count_from_keys(keys),
        "precision": precision,
        "total_tensors": len(keys),
        "components": {
            "unet": has_unet,
            "clip": has_clip,
            "vae": has_vae,
            "llm_adapter": has_llm_adapter,
        },
        "recipe": recipe,
        "comfy_recipe": comfy_recipe,
        "models": models,
        "raw_metadata": metadata,
    }
    info_dict["turbo"] = detect_turbo(info_dict)
    return info_dict


def detect_turbo(info: dict[str, Any]) -> dict[str, Any]:
    """Detects whether a checkpoint contains a Turbo LoRA, was merged from Anima Turbo 1.1,
    or is an Anima Turbo checkpoint itself.
    """
    filename = info.get("filename", "").lower()
    arch = info.get("architecture", "")
    metadata = info.get("raw_metadata") or {}
    recipe = info.get("recipe")
    comfy_recipe = info.get("comfy_recipe")
    models = info.get("models") or {}

    found_turbo = False
    turbo_kind = None
    details = []

    # 1. Check baked LoRAs in WebUI merge recipe
    if recipe and isinstance(recipe, dict):
        baked = recipe.get("baked_loras") or recipe.get("loras") or []
        for lora in baked:
            l_name = lora.get("name", "")
            if "turbo" in l_name.lower():
                found_turbo = True
                turbo_kind = "Baked Turbo LoRA"
                l_str = lora.get("strength", 1.0)
                details.append(f"Baked Turbo LoRA: <b>{_esc(l_name)}</b> (strength: {_esc(l_str)})")

        p_hash = recipe.get("primary_model_hash", "")
        s_hash = recipe.get("secondary_model_hash", "")
        t_hash = recipe.get("tertiary_model_hash", "")
        for role, h in [("Model A", p_hash), ("Model B", s_hash), ("Model C", t_hash)]:
            if h and isinstance(models, dict) and h in models:
                m_name = models[h].get("name", "")
                if "turbo" in m_name.lower():
                    found_turbo = True
                    if not turbo_kind:
                        turbo_kind = "Merged from Turbo Checkpoint"
                    details.append(f"Merge Parent ({role}): <b>{_esc(m_name)}</b>")

    # 2. Check ComfyUI workflow / prompt recipe
    if comfy_recipe and isinstance(comfy_recipe, dict):
        for lora in comfy_recipe.get("loras", []):
            l_name = lora.get("name", "")
            if "turbo" in l_name.lower():
                found_turbo = True
                if not turbo_kind:
                    turbo_kind = "Baked Turbo LoRA"
                l_str = lora.get("strength_model", lora.get("strength", 1.0))
                details.append(f"ComfyUI Baked LoRA: <b>{_esc(l_name)}</b> (strength: {_esc(l_str)})")

        for bm in comfy_recipe.get("base_models", []):
            b_name = bm.get("name", "")
            if "turbo" in b_name.lower():
                found_turbo = True
                if not turbo_kind:
                    turbo_kind = "Merged from Turbo Checkpoint"
                details.append(f"ComfyUI Base Model: <b>{_esc(b_name)}</b>")

    # 3. Check raw metadata strings
    if not found_turbo and metadata:
        meta_str = json.dumps(metadata, ensure_ascii=False).lower()
        if any(w in meta_str for w in ("anima-turbo-v1.1", "anima_turbo_v1.1", "animaturbo_v1.1", "animaturbov1.1", "anima turbo 1.1")):
            found_turbo = True
            turbo_kind = "Anima Turbo 1.1 in Lineage"
            details.append("Anima Turbo 1.1 identified in checkpoint metadata / workflow graph")
        elif "turbo" in meta_str and "anima" in arch.lower():
            if any(w in meta_str for w in ("anima-turbo", "anima_turbo", "turbo.safetensors")):
                found_turbo = True
                turbo_kind = "Turbo Acceleration Detected"
                details.append("Turbo model reference detected in workflow metadata")

    # 4. Check filename itself
    if not found_turbo and "turbo" in filename:
        if any(w in filename for w in ("1.1", "v11", "v1.1", "1-1")):
            found_turbo = True
            turbo_kind = "Anima Turbo 1.1 Checkpoint"
            details.append(f"Checkpoint filename identifies as Anima Turbo 1.1 ({_esc(info.get('filename'))})")
        else:
            found_turbo = True
            turbo_kind = "Turbo Checkpoint"
            details.append(f"Checkpoint filename identifies as Turbo ({_esc(info.get('filename'))})")

    return {
        "has_turbo": found_turbo,
        "kind": turbo_kind or "Turbo",
        "details": details,
    }


def _architecture_label(info: dict[str, Any]) -> str:
    """Architecture, with the Anima generation when there is one.

    Anima ships in three depths -- 28 blocks (circlestone-labs/Anima), 40
    (Anima-2.9B) and 52 (Anima-3.8B) -- and which one a file is decides whether
    it can be merged and in which slot. The block count was already read and
    used to guard merge order; it just never reached the badge, so all three
    generations displayed identically. The LoRA inspector already labels its
    own with "Anima <n>-block".
    """
    arch = str(info.get("architecture", "Unknown"))
    blocks = info.get("block_count")
    if get_model_family(arch) == "anima" and blocks:
        return f"{arch}, {blocks}-block"
    return arch


def format_badges_html(info: dict[str, Any], compatible_with_info: dict[str, Any] | None = None) -> str:
    """Returns a compact HTML line with component badges for the merge studio tab."""
    if not info or "error" in info:
        return ""

    comps = info.get("components", {})
    arch = info.get("architecture", "Unknown")
    is_dit = any(w in arch for w in ("DiT", "Flux", "SD3", "Wan", "Anima"))

    # Only show components that are PRESENT
    present_parts = []
    if comps.get("unet"):
        model_type = "DiT" if is_dit else "UNet"
        present_parts.append(f'<span style="color: #10b981; font-weight: bold;">{model_type}</span>')

    if comps.get("clip"):
        if "Anima" in arch:
            te_label = "Text Encoder (Qwen)"
        elif "Flux" in arch or "SD3" in arch:
            te_label = "Text Encoder (T5/CLIP)"
        else:
            te_label = "CLIP"
        present_parts.append(f'<span style="color: #10b981; font-weight: bold;">{te_label}</span>')

    if comps.get("vae"):
        present_parts.append('<span style="color: #10b981; font-weight: bold;">VAE</span>')

    if comps.get("llm_adapter"):
        present_parts.append('<span style="color: #38bdf8; font-weight: bold;">LLM Adapter</span>')

    comp_str = " | ".join(present_parts) if present_parts else '<span style="color: #9ca3af;">No standard components</span>'

    turbo_data = info.get("turbo") or detect_turbo(info)
    turbo_badge = ""
    if turbo_data.get("has_turbo"):
        turbo_badge = f' &nbsp;<span style="background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.4); padding: 1px 6px; border-radius: 4px; font-weight: bold; font-size: 11px;">{turbo_data.get("kind", "Turbo")}</span>'

    prec = info.get("precision", "Unknown")
    size = info.get("size_str", "")

    html = (
        f"<div style='margin-top: 4px; font-size: 12px; color: #9ca3af; line-height: 1.5;'>"
        f"[{comp_str}] &nbsp;•&nbsp; "
        f"<b>{_architecture_label(info)}</b>{turbo_badge} &nbsp;•&nbsp; <code>{prec}</code> &nbsp;•&nbsp; {size}"
        f"</div>"
    )

    # Incompatibility check against Model A
    if compatible_with_info and not compatible_with_info.get("error"):
        fam_self = get_model_family(arch)
        arch_other = compatible_with_info.get("architecture", "Unknown")
        fam_other = get_model_family(arch_other)
        blocks_self = info.get("block_count")
        blocks_other = compatible_with_info.get("block_count")
        if (
            fam_self == fam_other == "anima"
            and blocks_self is not None
            and blocks_other is not None
            and blocks_self > blocks_other
        ):
            html += (
                f"<div style='margin-top: 4px; padding: 4px 8px; border-radius: 4px; background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fcd34d; font-size: 11.5px; font-weight: 600;'>"
                f"Wrong order: this model has {blocks_self} blocks but Model A has only {blocks_other}. "
                f"The <b>newer / larger</b> Anima generation must be <b>Primary Model (A)</b>, with an equal or older one here. Swap them."
                f"</div>"
            )
        if fam_self != "other" and fam_other != "other" and fam_self != fam_other:
            html += (
                f"<div style='margin-top: 4px; padding: 4px 8px; border-radius: 4px; background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #fca5a5; font-size: 11.5px; font-weight: 600;'>"
                f"Incompatible Architecture: <b>{arch}</b> cannot be merged with Model A (<b>{arch_other}</b>)"
                f"</div>"
            )

    return html


def _describe_merge_math(raw_method: str, recipe: dict[str, Any]) -> tuple[str, str]:
    """(readable method label, HTML for the formula line).

    The multiplier means a different thing per method, and for one of them it
    means nothing at all -- No Interpolation takes a single model, so there is
    no Model B to be a percentage of. Describing every method as a blend, which
    is what a single shared format did, is wrong for two of the three.
    """
    key = raw_method.strip().lower().replace(" ", "_")

    def _line(body: str) -> str:
        return (
            f"<div style='margin-bottom: 12px; font-size: 13px; color: #d1d5db;'>{body}</div>"
        )

    try:
        m = float(recipe.get("multiplier"))
    except (TypeError, ValueError):
        m = None

    if "no_interpolation" in key:
        # Format/precision conversion or a LoRA bake: one model in, one out.
        return "No Interpolation", _line(
            "<b>Formula:</b> <code style='color: #f97316;'>A</code> "
            "<span style='color:#9ca3af;'>&mdash; single model, converted or baked into rather than blended. "
            "No multiplier applies.</span>"
        )

    if "add_difference" in key:
        m_txt = f"{m:.2f}" if m is not None else "M"
        return "Add Difference", _line(
            f"<b>Formula:</b> <code style='color: #f97316;'>A + {m_txt} &times; (B &minus; C)</code> "
            f"<span style='color:#9ca3af;'>&mdash; what B learned over C, transplanted onto A. "
            f"Not a blend ratio.</span>"
        )

    if "weighted_sum" in key:
        if m is None:
            return "Weighted Sum", _line("<b>Formula:</b> <code style='color: #f97316;'>A &times; (1 &minus; M) + B &times; M</code>")
        return "Weighted Sum", _line(
            f"<b>Multiplier (M):</b> <code style='color: #f97316;'>{m:.2f}</code> "
            f"<span style='color:#9ca3af;'>&mdash; {round((1 - m) * 100)}% Model A / {round(m * 100)}% Model B</span>"
        )

    if "lorabake" in key.replace("-", "").replace("_", ""):
        return "LoRA Bake", _line(
            "<b>Formula:</b> <code style='color: #f97316;'>A + baked LoRA(s)</code> "
            "<span style='color:#9ca3af;'>&mdash; per-LoRA strengths are listed below.</span>"
        )

    # Unknown method: show the multiplier if there is one, but don't claim to
    # know what it means.
    if m is not None:
        return raw_method, _line(f"<b>Multiplier:</b> <code style='color: #f97316;'>{m:.2f}</code>")
    return raw_method, ""


def format_recipe_dashboard_html(info: dict[str, Any]) -> str:
    """Generates the full styled dashboard card for the dedicated Recipe Inspector tab."""
    if not info:
        return "<div style='padding: 20px; color: #9ca3af;'>Select a checkpoint above to inspect its components and merge recipe.</div>"

    if "error" in info:
        return (
            f"<div style='padding: 16px; border-radius: 8px; background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; color: #fca5a5;'>"
            f"<b>Error inspecting checkpoint:</b> {_esc(info['error'])}</div>"
        )

    comps = info.get("components", {})
    arch = info.get("architecture", "Unknown")
    prec = info.get("precision", "Unknown")
    size = info.get("size_str", "")
    filename = info.get("filename", "")
    recipe = info.get("recipe")
    models = info.get("models") or {}
    raw_meta = info.get("raw_metadata") or {}

    # Component status pills (only present components are rendered)
    def _comp_pill(name: str, note: str = ""):
        desc = f" ({note})" if note else ""
        return (
            f"<div style='display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; border-radius: 6px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);'>"
            f"<span style='font-weight: 500;'>{name}</span>"
            f"<span style='color: #10b981; font-weight: bold;'>Present{desc}</span>"
            f"</div>"
        )

    is_dit = any(w in arch for w in ("DiT", "Flux", "SD3", "Wan", "Anima"))
    model_name = "Diffusion Transformer (DiT)" if is_dit else "Diffusion Model (UNet)"
    unet_note = "DiT denoising network" if is_dit else "UNet denoising network"
    if "Anima" in arch:
        te_pill_label = "Text Encoder (Qwen 3)"
        clip_note = "Embedded Qwen 3 text encoder"
    elif "Flux" in arch or "SD3" in arch:
        te_pill_label = "Text Encoder (T5/CLIP)"
        clip_note = "Embedded T5/CLIP text encoder"
    else:
        te_pill_label = "Text Encoder (CLIP)"
        clip_note = "Embedded CLIP text encoder"
    vae_note = "Embedded autoencoder"

    comp_pills = []
    if comps.get("unet"):
        comp_pills.append(_comp_pill(model_name, unet_note))
    if comps.get("clip"):
        comp_pills.append(_comp_pill(te_pill_label, clip_note))
    if comps.get("vae"):
        comp_pills.append(_comp_pill("VAE (Autoencoder)", vae_note))
    if comps.get("llm_adapter"):
        comp_pills.append(_comp_pill("Anima LLM Adapter", "Embedded DiT alignment weights"))

    if not comp_pills:
        comp_pills.append("<div style='padding: 8px 12px; border-radius: 6px; background: rgba(239, 68, 68, 0.1); color: #ef4444;'>No standard neural components detected in file header.</div>")

    comp_grid_html = "".join(comp_pills)

    # Turbo Acceleration Callout
    turbo_data = info.get("turbo") or detect_turbo(info)
    turbo_html = ""
    if turbo_data.get("has_turbo"):
        details_items = "".join(f"<div style='margin-top: 3px;'>• {d}</div>" for d in turbo_data.get("details", []))
        turbo_html = (
            f"<div style='margin-top: 14px; padding: 12px 16px; border-radius: 8px; background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.35);'>"
            f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
            f"<div style='font-size: 13px; font-weight: 700; color: #f59e0b; text-transform: uppercase;'>Turbo Acceleration Detected</div>"
            f"<span style='font-size: 11px; background: rgba(245, 158, 11, 0.2); color: #f59e0b; padding: 2px 8px; border-radius: 4px; font-weight: bold;'>{_esc(turbo_data.get('kind'))}</span>"
            f"</div>"
            f"<div style='font-size: 13px; color: #f3f4f6; margin-top: 6px; line-height: 1.5;'>{details_items}</div>"
            f"</div>"
        )

    # Recipe section HTML
    recipe_html = ""
    if recipe:
        raw_method = str(recipe.get("interp_method", recipe.get("type", "Custom Merge")))
        method, math_html = _describe_merge_math(raw_method, recipe)

        # Lookup model names from models dict
        p_hash = recipe.get("primary_model_hash", "")
        s_hash = recipe.get("secondary_model_hash", "")
        t_hash = recipe.get("tertiary_model_hash", "")

        def _model_card(label: str, hash_val: str):
            if not hash_val:
                return ""
            m_info = models.get(hash_val, {}) if isinstance(models, dict) else {}
            name = m_info.get("name", "Unknown model name")
            return (
                f"<div style='padding: 10px 14px; border-radius: 6px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); margin-bottom: 8px;'>"
                f"<div style='font-size: 11px; text-transform: uppercase; color: #f97316; font-weight: 700; margin-bottom: 2px;'>{label}</div>"
                f"<div style='font-size: 14px; font-weight: 600; color: #f3f4f6;'>{_esc(name)}</div>"
                f"<div style='font-size: 11px; color: #6b7280; font-family: monospace; margin-top: 2px;'>SHA256: {_esc(hash_val)}</div>"
                f"</div>"
            )

        models_cards = _model_card("Primary Model (A)", p_hash)
        models_cards += _model_card("Secondary Model (B)", s_hash)
        models_cards += _model_card("Tertiary Model (C)", t_hash)

        # Baked LoRAs
        baked_loras = recipe.get("baked_loras") or recipe.get("loras") or []
        loras_html = ""
        if baked_loras:
            lora_rows = ""
            for lora in baked_loras:
                l_name = html_lib.escape(str(lora.get("name", "LoRA")))
                l_str = lora.get("strength", 1.0)
                l_trig = lora.get("activation_text", "")
                if l_trig:
                    safe_trigger = _esc(l_trig)
                    # Where the trigger came from was already recorded per LoRA;
                    # until now nothing read it back, so every trigger looked
                    # equally sourced.
                    origin = str(lora.get("activation_text_source") or "").strip()
                    tooltip = (
                        f"Recorded in the embedded merge recipe, from the LoRA's {origin}"
                        if origin
                        else "Recorded in the embedded merge recipe; not inferred from tensors"
                    )
                    trig_badge = (
                        f"<span title='{_esc(tooltip)}' "
                        f"style='background: rgba(249,115,22,0.15); color: #f97316; padding: 2px 6px; "
                        f"border-radius: 4px; font-size: 11px;'>declared trigger: {safe_trigger}</span>"
                    )
                else:
                    trig_badge = ""
                lora_rows += (
                    f"<div style='display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 13px;'>"
                    f"<span><b>{l_name}</b> {trig_badge}</span>"
                    f"<span style='color: #10b981; font-family: monospace;'>strength: {_esc(l_str)}</span>"
                    f"</div>"
                )
            loras_html = (
                f"<div style='margin-top: 14px;'>"
                f"<div style='font-size: 12px; font-weight: 600; text-transform: uppercase; color: #9ca3af; margin-bottom: 6px;'>Baked LoRAs</div>"
                f"<div style='border-radius: 6px; background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.06); padding: 4px;'>{lora_rows}</div>"
                f"<div style='color: #6b7280; font-size: 11px; margin-top: 5px;'>Trigger declarations shown here were recorded in embedded merge recipe metadata.</div>"
                f"</div>"
            )

        recipe_html = (
            f"<div style='margin-top: 20px; padding: 16px; border-radius: 8px; background: rgba(249,115,22,0.05); border: 1px solid rgba(249,115,22,0.25);'>"
            f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>"
            f"<div style='font-size: 16px; font-weight: bold; color: #f97316; display: flex; align-items: center; gap: 8px;'>"
            f"<span>Merge Recipe Detected</span></div>"
            f"<span style='font-size: 12px; background: rgba(249,115,22,0.2); color: #f97316; padding: 3px 8px; border-radius: 4px;'>Method: {_esc(method)}</span>"
            f"</div>"
            f"{math_html}"
            f"<div style='margin-top: 10px;'>{models_cards}</div>"
            f"{loras_html}"
            f"</div>"
        )
    else:
        recipe_html = ""

    # ComfyUI recipe section HTML
    comfy_recipe = info.get("comfy_recipe")
    comfy_html = ""
    if comfy_recipe:
        c_models = comfy_recipe.get("base_models", [])
        c_loras = comfy_recipe.get("loras", [])
        c_merges = comfy_recipe.get("merges", [])
        c_encoders = comfy_recipe.get("encoders", [])
        c_vaes = comfy_recipe.get("vaes", [])

        # Cards for base models
        b_cards = ""
        for bm in c_models:
            b_name = bm.get("name", "Unknown")
            b_hash = bm.get("hash", "")
            b_node = bm.get("node", "Loader")
            hash_snippet = f"<div style='font-size: 11px; color: #6b7280; font-family: monospace; margin-top: 2px;'>SHA256: {_esc(b_hash)}</div>" if b_hash else ""
            b_cards += (
                f"<div style='padding: 10px 14px; border-radius: 6px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); margin-bottom: 8px;'>"
                f"<div style='font-size: 11px; text-transform: uppercase; color: #38bdf8; font-weight: 700; margin-bottom: 2px;'>Base Model ({_esc(b_node)})</div>"
                f"<div style='font-size: 14px; font-weight: 600; color: #f3f4f6;'>{_esc(b_name)}</div>"
                f"{hash_snippet}"
                f"</div>"
            )

        # Merge cards if any
        m_cards = ""
        for mg in c_merges:
            m_type = mg.get("type", "ModelMerge")
            m_ratio = mg.get("ratio", "N/A")
            m_cards += (
                f"<div style='margin-bottom: 8px; font-size: 13px; color: #d1d5db;'>"
                f"<b>ComfyUI Merge Node:</b> <code style='color: #38bdf8;'>{_esc(m_type)} (Ratio: {_esc(m_ratio)})</code>"
                f"</div>"
            )

        # LoRA rows
        c_loras_html = ""
        if c_loras:
            lora_rows = ""
            for l in c_loras:
                l_name = l.get("name", "")
                l_str = l.get("strength", 1.0)
                l_h = l.get("hash", "")
                h_badge = f"<span style='font-size: 10px; color: #6b7280; font-family: monospace;'> [{_esc(l_h[:12])}...]</span>" if l_h else ""
                lora_rows += (
                    f"<div style='display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 13px;'>"
                    f"<span><b>{_esc(l_name)}</b>{h_badge}</span>"
                    f"<span style='color: #10b981; font-family: monospace; font-weight: 600;'>strength: {_esc(l_str)}</span>"
                    f"</div>"
                )
            c_loras_html = (
                f"<div style='margin-top: 14px;'>"
                f"<div style='font-size: 12px; font-weight: 600; text-transform: uppercase; color: #9ca3af; margin-bottom: 6px;'>Baked LoRAs ({len(c_loras)})</div>"
                f"<div style='border-radius: 6px; background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.06); padding: 4px;'>{lora_rows}</div>"
                f"</div>"
            )

        # Extra info (encoders / vaes)
        extra_info = ""
        if c_encoders or c_vaes:
            extra_bits = []
            if c_encoders:
                extra_bits.append(f"<b>Text Encoder:</b> <code>{_esc(', '.join(str(x) for x in c_encoders))}</code>")
            if c_vaes:
                extra_bits.append(f"<b>VAE:</b> <code>{_esc(', '.join(str(x) for x in c_vaes))}</code>")
            extra_info = f"<div style='margin-top: 10px; font-size: 12px; color: #9ca3af;'>{' &nbsp;•&nbsp; '.join(extra_bits)}</div>"

        comfy_html = (
            f"<div style='margin-top: 20px; padding: 16px; border-radius: 8px; background: rgba(56, 189, 248, 0.05); border: 1px solid rgba(56, 189, 248, 0.25);'>"
            f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>"
            f"<div style='font-size: 16px; font-weight: bold; color: #38bdf8;'>ComfyUI Workflow Recipe Detected</div>"
            f"<span style='font-size: 12px; background: rgba(56, 189, 248, 0.2); color: #38bdf8; padding: 3px 8px; border-radius: 4px;'>Source: ComfyUI Node Graph</span>"
            f"</div>"
            f"{m_cards}"
            f"<div style='margin-top: 10px;'>{b_cards}</div>"
            f"{c_loras_html}"
            f"{extra_info}"
            f"</div>"
        )

    if not recipe and not comfy_recipe:
        empty_recipe_html = (
            f"<div style='margin-top: 20px; padding: 14px 16px; border-radius: 8px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); color: #9ca3af; font-size: 13px; line-height: 1.5;'>"
            f"<b>No merge recipe or workflow recorded in header.</b><br>"
            f"This checkpoint does not contain 'sd_merge_recipe' or ComfyUI workflow metadata. This is typical for checkpoints trained from scratch, exported without metadata, or pruned by external cleaning tools."
            f"</div>"
        )
    else:
        empty_recipe_html = ""

    all_recipes_html = recipe_html + comfy_html + empty_recipe_html

    # Extra metadata table / JSON accordion
    meta_json_str = json.dumps(raw_meta, indent=2, ensure_ascii=False) if raw_meta else "{}"

    html = (
        f"<div style='font-family: system-ui, -apple-system, sans-serif; line-height: 1.5;'>"
        # Top Card: Overview
        f"<div style='padding: 16px; border-radius: 8px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); margin-bottom: 16px;'>"
        f"<div style='display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;'>"
        f"<div>"
        f"<div style='font-size: 18px; font-weight: bold; color: #f3f4f6;'>{_esc(filename)}</div>"
        f"<div style='font-size: 12px; color: #6b7280; margin-top: 2px;'>{size} &nbsp;•&nbsp; {info.get('total_tensors', 0):,} tensors</div>"
        f"</div>"
        f"<div style='display: flex; gap: 8px; flex-wrap: wrap;'>"
        f"<span style='background: #1e3a8a; color: #93c5fd; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600;'>{_esc(_architecture_label(info))}</span>"
        f"<span style='background: #312e81; color: #c7d2fe; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600;'>{_esc(prec)}</span>"
        f"</div>"
        f"</div>"
        # Component Grid
        f"<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; margin-top: 14px;'>"
        f"{comp_grid_html}"
        f"</div>"
        f"{turbo_html}"
        f"</div>"
        # Recipe Section
        f"{all_recipes_html}"
        # Metadata Accordion
        f"<details style='margin-top: 16px; padding: 10px 14px; border-radius: 8px; background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.06);'>"
        f"<summary style='cursor: pointer; font-size: 13px; font-weight: 600; color: #9ca3af;'>View Raw Metadata (JSON - {len(raw_meta)} fields)</summary>"
        f"<pre style='margin-top: 10px; padding: 12px; border-radius: 6px; background: #0b0f19; color: #a7f3d0; font-size: 12px; overflow-x: auto; max-height: 400px; border: 1px solid #1f2937;'>{_esc(meta_json_str)}</pre>"
        f"</details>"
        f"</div>"
    )

    return html


def available_vaes() -> dict[str, str]:
    """Returns {name: file_path} of available VAE files."""
    vaes: dict[str, str] = {}
    try:
        from modules import sd_vae
        sd_vae.refresh_vae_list()
        if hasattr(sd_vae, "vae_dict"):
            for k, v in sd_vae.vae_dict.items():
                vaes[k] = v
    except Exception:
        pass

    try:
        from modules import paths, shared
        vae_dirs = []
        if hasattr(paths, "models_path") and paths.models_path:
            vae_dirs.append(os.path.join(paths.models_path, "VAE"))
        if hasattr(shared, "opts") and getattr(shared.opts, "vae_dir", None):
            vae_dirs.append(shared.opts.vae_dir)
        for d in vae_dirs:
            if d and os.path.isdir(d):
                for root, _, files in os.walk(d):
                    for file in files:
                        if file.endswith((".safetensors", ".pt", ".ckpt")):
                            rel = os.path.relpath(os.path.join(root, file), d)
                            if rel not in vaes:
                                vaes[rel] = os.path.join(root, file)
    except Exception:
        pass
    return vaes


def load_custom_vae_state_dict(vae_path_or_name: str) -> dict[str, Any]:
    """Loads a VAE state dict from a safetensors or pt/ckpt file.
    Strips common prefixes ('first_stage_model.', 'vae.') so it can be passed
    cleanly to model_config.process_vae_state_dict_for_saving()."""
    vaes = available_vaes()
    filepath = vaes.get(vae_path_or_name, vae_path_or_name)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"VAE file not found: {filepath}")

    import torch
    if filepath.endswith(".safetensors"):
        from safetensors.torch import load_file
        raw_sd = load_file(filepath, device="cpu")
    else:
        raw_sd = torch.load(filepath, map_location="cpu")
        if isinstance(raw_sd, dict) and "state_dict" in raw_sd:
            raw_sd = raw_sd["state_dict"]

    cleaned = {}
    for k, v in raw_sd.items():
        clean_k = k
        if clean_k.startswith("first_stage_model."):
            clean_k = clean_k[len("first_stage_model.") :]
        elif clean_k.startswith("vae."):
            clean_k = clean_k[len("vae.") :]
        cleaned[clean_k] = v
    return cleaned
