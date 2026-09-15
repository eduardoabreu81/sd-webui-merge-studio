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
import os
import struct
from typing import Any


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


def _detect_architecture(keys: list[str], has_llm_adapter: bool) -> str:
    if any(k.startswith(("double_blocks.", "model.diffusion_model.double_blocks.", "img_in.")) for k in keys):
        return "Flux (MMDiT)"
    if any(k.startswith(("model.diffusion_model.joint_blocks.", "joint_blocks.")) for k in keys):
        return "SD3 / SD3.5 (MMDiT)"
    if has_llm_adapter or any(k.startswith("net.") for k in keys) or any("qwen" in k for k in keys):
        return "Anima (DiT)"
    if any(k.startswith(("v_blocks.", "model.diffusion_model.v_blocks.", "head.weight")) for k in keys):
        return "Wan2.1 (DiT)"
    if any(k.startswith("model.diffusion_model.blocks.") for k in keys):
        return "DiT / Diffusion Model"
    if any(k.startswith("conditioner.embedders.") for k in keys):
        return "SDXL (UNet)"
    if any(k.startswith("cond_stage_model.") for k in keys):
        return "SD 1.5 / SD 2.1 (UNet)"
    if any(k.startswith(("model.diffusion_model.input_blocks.", "diffusion_model.input_blocks.", "input_blocks.")) for k in keys):
        # Heuristic for SDXL vs SD 1.5 if text encoder is missing
        # SDXL has 3 stages (320, 640, 1280) while SD 1.5 has 4 stages (320, 640, 1280, 1280)
        has_stage_3 = any("input_blocks.9." in k or "input_blocks.10." in k or "input_blocks.11." in k for k in keys)
        return "SD 1.5 / SD 2.1 (UNet)" if has_stage_3 else "SDXL (UNet)"
    return "Diffusion Model"


def _detect_precision(header: dict[str, Any]) -> str:
    # Check comfy_quant presence first
    has_comfy_quant = any(k.endswith(".comfy_quant") for k in header)
    if has_comfy_quant:
        has_convrot = False
        format_name = None
        for k in header:
            if k.endswith(".comfy_quant"):
                # Header might not hold the json blob content, but let's check weight dtype
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
                m_name = inp.get("unet_name") or inp.get("ckpt_name") or inp.get("model_path")
                if m_name:
                    h = ""
                    for k, v in hashes.items():
                        if m_name in k and isinstance(v, dict):
                            h = v.get("hash", "")
                            break
                    base_models.append({"name": m_name, "node": ctype, "hash": h})

            elif ctype in ("CLIPLoader", "DualCLIPLoader"):
                c_name = inp.get("clip_name") or inp.get("clip_name1")
                if c_name:
                    encoders.append(c_name)

            elif ctype == "VAELoader":
                v_name = inp.get("vae_name")
                if v_name:
                    vaes.append(v_name)

            elif ctype in ("LoraLoaderModelOnly", "LoraLoader", "LoraLoaderBlockWeight"):
                l_name = inp.get("lora_name")
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
                            l_name = v.get("lora")
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
        header, _ = read_safetensors_header(filepath)
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

    arch = _detect_architecture(keys, has_llm_adapter)
    precision = _detect_precision(header)

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

    return {
        "filename": os.path.basename(filepath),
        "filepath": filepath,
        "file_size": file_size,
        "size_str": size_str,
        "architecture": arch,
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


def format_badges_html(info: dict[str, Any]) -> str:
    """Returns a compact HTML line with component badges for the merge studio tab."""
    if not info or "error" in info:
        return ""

    comps = info.get("components", {})
    arch = info.get("architecture", "Unknown")
    is_dit = any(w in arch for w in ("DiT", "Flux", "SD3", "Wan", "Anima"))
    model_type = "DiT" if is_dit else "UNet"

    dit_badge = (
        f'<span style="color: #10b981; font-weight: bold;">{model_type}: Present</span>'
        if comps.get("unet")
        else f'<span style="color: #ef4444; font-weight: bold;">{model_type}: Missing</span>'
    )
    # Dynamic text encoder label
    if "Anima" in arch:
        te_label = "Text Encoder (Qwen)"
    elif "Flux" in arch or "SD3" in arch:
        te_label = "Text Encoder (T5/CLIP)"
    else:
        te_label = "CLIP"

    clip_badge = (
        f'<span style="color: #10b981; font-weight: bold;">{te_label}: Present</span>'
        if comps.get("clip")
        else f'<span style="color: #f59e0b; font-weight: bold;">{te_label}: Missing</span>'
    )
    vae_badge = (
        '<span style="color: #10b981; font-weight: bold;">VAE: Present</span>'
        if comps.get("vae")
        else '<span style="color: #f59e0b; font-weight: bold;">VAE: Missing</span>'
    )

    llm_badge = ""
    if comps.get("llm_adapter"):
        llm_badge = ' | <span style="color: #38bdf8; font-weight: bold;">LLM Adapter: Present</span>'

    prec = info.get("precision", "Unknown")
    size = info.get("size_str", "")

    return (
        f"<div style='margin-top: 4px; font-size: 12px; color: #9ca3af; line-height: 1.5;'>"
        f"[{dit_badge} | {clip_badge} | {vae_badge}{llm_badge}] &nbsp;•&nbsp; "
        f"<b>{arch}</b> &nbsp;•&nbsp; <code>{prec}</code> &nbsp;•&nbsp; {size}"
        f"</div>"
    )


def format_recipe_dashboard_html(info: dict[str, Any]) -> str:
    """Generates the full styled dashboard card for the dedicated Recipe Inspector tab."""
    if not info:
        return "<div style='padding: 20px; color: #9ca3af;'>Select a checkpoint above to inspect its components and merge recipe.</div>"

    if "error" in info:
        return (
            f"<div style='padding: 16px; border-radius: 8px; background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; color: #fca5a5;'>"
            f"<b>Error inspecting checkpoint:</b> {info['error']}</div>"
        )

    comps = info.get("components", {})
    arch = info.get("architecture", "Unknown")
    prec = info.get("precision", "Unknown")
    size = info.get("size_str", "")
    filename = info.get("filename", "")
    recipe = info.get("recipe")
    models = info.get("models") or {}
    raw_meta = info.get("raw_metadata") or {}

    # Component status pills
    def _comp_pill(name: str, present: bool, note: str = ""):
        color = "#10b981" if present else "#f59e0b"
        desc = f" ({note})" if note else ""
        status_text = "Present" if present else "Missing"
        return (
            f"<div style='display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; border-radius: 6px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);'>"
            f"<span style='font-weight: 500;'>{name}</span>"
            f"<span style='color: {color}; font-weight: bold;'>{status_text}{desc}</span>"
            f"</div>"
        )

    is_dit = any(w in arch for w in ("DiT", "Flux", "SD3", "Wan", "Anima"))
    model_name = "Diffusion Transformer (DiT)" if is_dit else "Diffusion Model (UNet)"
    unet_note = "DiT denoising network" if (comps.get("unet") and is_dit) else ("UNet denoising network" if comps.get("unet") else "No diffusion model detected")
    if "Anima" in arch:
        te_pill_label = "Text Encoder (Qwen 3)"
        clip_note = "Embedded Qwen 3 text encoder" if comps.get("clip") else "Requires external Qwen text encoder"
    elif "Flux" in arch or "SD3" in arch:
        te_pill_label = "Text Encoder (T5/CLIP)"
        clip_note = "Embedded T5/CLIP text encoder" if comps.get("clip") else "Requires external text encoder"
    else:
        te_pill_label = "Text Encoder (CLIP)"
        clip_note = "Embedded CLIP text encoder" if comps.get("clip") else "Requires external text encoder"
    vae_note = "Embedded autoencoder" if comps.get("vae") else "Requires external VAE"

    llm_pill = ""
    if comps.get("llm_adapter"):
        llm_pill = _comp_pill("Anima LLM Adapter", True, "Embedded DiT alignment weights")

    # Recipe section HTML
    recipe_html = ""
    if recipe:
        method = recipe.get("interp_method", recipe.get("type", "Custom Merge"))
        mult = recipe.get("multiplier", "N/A")
        try:
            mult_val = float(mult)
            mult_desc = f"{mult_val:.2f} ({int((1 - mult_val)*100)}% Model A / {int(mult_val*100)}% Model B)"
        except Exception:
            mult_desc = str(mult)

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
                f"<div style='font-size: 14px; font-weight: 600; color: #f3f4f6;'>{name}</div>"
                f"<div style='font-size: 11px; color: #6b7280; font-family: monospace; margin-top: 2px;'>SHA256: {hash_val}</div>"
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
                l_name = lora.get("name", "LoRA")
                l_str = lora.get("strength", 1.0)
                l_trig = lora.get("activation_text", "")
                trig_badge = f"<span style='background: rgba(249,115,22,0.15); color: #f97316; padding: 2px 6px; border-radius: 4px; font-size: 11px;'>trigger: {l_trig}</span>" if l_trig else ""
                lora_rows += (
                    f"<div style='display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 13px;'>"
                    f"<span><b>{l_name}</b> {trig_badge}</span>"
                    f"<span style='color: #10b981; font-family: monospace;'>strength: {l_str}</span>"
                    f"</div>"
                )
            loras_html = (
                f"<div style='margin-top: 14px;'>"
                f"<div style='font-size: 12px; font-weight: 600; text-transform: uppercase; color: #9ca3af; margin-bottom: 6px;'>Baked LoRAs</div>"
                f"<div style='border-radius: 6px; background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.06); padding: 4px;'>{lora_rows}</div>"
                f"</div>"
            )

        recipe_html = (
            f"<div style='margin-top: 20px; padding: 16px; border-radius: 8px; background: rgba(249,115,22,0.05); border: 1px solid rgba(249,115,22,0.25);'>"
            f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>"
            f"<div style='font-size: 16px; font-weight: bold; color: #f97316; display: flex; align-items: center; gap: 8px;'>"
            f"<span>Merge Recipe Detected</span></div>"
            f"<span style='font-size: 12px; background: rgba(249,115,22,0.2); color: #f97316; padding: 3px 8px; border-radius: 4px;'>Method: {method}</span>"
            f"</div>"
            f"<div style='margin-bottom: 12px; font-size: 13px; color: #d1d5db;'>"
            f"<b>Multiplier (M):</b> <code style='color: #f97316;'>{mult_desc}</code>"
            f"</div>"
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
            hash_snippet = f"<div style='font-size: 11px; color: #6b7280; font-family: monospace; margin-top: 2px;'>SHA256: {b_hash}</div>" if b_hash else ""
            b_cards += (
                f"<div style='padding: 10px 14px; border-radius: 6px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); margin-bottom: 8px;'>"
                f"<div style='font-size: 11px; text-transform: uppercase; color: #38bdf8; font-weight: 700; margin-bottom: 2px;'>Base Model ({b_node})</div>"
                f"<div style='font-size: 14px; font-weight: 600; color: #f3f4f6;'>{b_name}</div>"
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
                f"<b>ComfyUI Merge Node:</b> <code style='color: #38bdf8;'>{m_type} (Ratio: {m_ratio})</code>"
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
                h_badge = f"<span style='font-size: 10px; color: #6b7280; font-family: monospace;'> [{l_h[:12]}...]</span>" if l_h else ""
                lora_rows += (
                    f"<div style='display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 13px;'>"
                    f"<span><b>{l_name}</b>{h_badge}</span>"
                    f"<span style='color: #10b981; font-family: monospace; font-weight: 600;'>strength: {l_str}</span>"
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
                extra_bits.append(f"<b>Text Encoder:</b> <code>{', '.join(c_encoders)}</code>")
            if c_vaes:
                extra_bits.append(f"<b>VAE:</b> <code>{', '.join(c_vaes)}</code>")
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
        f"<div style='font-size: 18px; font-weight: bold; color: #f3f4f6;'>{filename}</div>"
        f"<div style='font-size: 12px; color: #6b7280; margin-top: 2px;'>{size} &nbsp;•&nbsp; {info.get('total_tensors', 0):,} tensors</div>"
        f"</div>"
        f"<div style='display: flex; gap: 8px; flex-wrap: wrap;'>"
        f"<span style='background: #1e3a8a; color: #93c5fd; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600;'>{arch}</span>"
        f"<span style='background: #312e81; color: #c7d2fe; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600;'>{prec}</span>"
        f"</div>"
        f"</div>"
        # Component Grid
        f"<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; margin-top: 14px;'>"
        f"{_comp_pill(model_name, comps.get('unet'), unet_note)}"
        f"{_comp_pill(te_pill_label, comps.get('clip'), clip_note)}"
        f"{_comp_pill('VAE (Autoencoder)', comps.get('vae'), vae_note)}"
        f"{llm_pill}"
        f"</div>"
        f"</div>"
        # Recipe Section
        f"{all_recipes_html}"
        # Metadata Accordion
        f"<details style='margin-top: 16px; padding: 10px 14px; border-radius: 8px; background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.06);'>"
        f"<summary style='cursor: pointer; font-size: 13px; font-weight: 600; color: #9ca3af;'>View Raw Metadata (JSON - {len(raw_meta)} fields)</summary>"
        f"<pre style='margin-top: 10px; padding: 12px; border-radius: 6px; background: #0b0f19; color: #a7f3d0; font-size: 12px; overflow-x: auto; max-height: 400px; border: 1px solid #1f2937;'>{meta_json_str}</pre>"
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
