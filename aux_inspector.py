"""Header inspection for the model types that aren't checkpoints: LoRAs, and
the auxiliary modules (text encoders, VAEs) Forge loads alongside a checkpoint.

Checkpoints are handled by checkpoint_inspector.py. This module answers the
questions that only make sense for the others:

  LoRA            which Anima generation it targets, whether it needs a trigger
                  word, rank, what it touches, whether it meaningfully changes
                  the LLM adapter, and whether it was extracted rather than
                  trained.
  Text encoder    which encoder it is, so a checkpoint built for one is not
                  paired with another.
  VAE             which family it belongs to.

Everything is read from the .safetensors header, except the LLM-adapter
magnitude check, which samples a bounded slice of tensor bytes (see
_llm_adapter_ratio).
"""

from __future__ import annotations

import json
import os
import re
import struct
from typing import Any

from anima_remap import ANIMA_BLOCK_SIZES
from checkpoint_inspector import _detect_precision, read_safetensors_header

# Key conventions a LoRA may use for the main block stack, matching
# extensions-builtin/sd_forge_lora/networks.py::process_anima.
_LORA_PREFIXES = (
    ("lora_unet_blocks_", "_"),
    ("diffusion_model.blocks.", "."),
)

# Tensors that carry an adapter's actual magnitude. `alpha` is a scalar and
# `.weight` alone would also catch bias/norm copies, so match the factors.
#
# Covers the LyCORIS family too, not just plain LoRA: Forge applies all of
# them through the same weight-adapter path, so a LoHa or LoKr bakes exactly
# like a LoRA and has to be recognised the same way.
#
# The diff-style markers keep their surrounding dots: a bare "diff" is a
# substring of "model.diffusion_model.", which would match every key of an
# ordinary checkpoint.
# Taken from the adapters Forge actually registers, in
# modules_forge/packages/comfy/weight_adapter/: lora, loha, lokr, oft, oftv2,
# boft and glora. Reading their key names from the source beats recalling
# LyCORIS conventions, and keeps this list aligned with what will really load.
_FACTOR_MARKERS = (
    "lora_down", "lora_up", "lora_A", "lora_B",    # LoRA
    "lora.down.weight", "lora.up.weight",          # LoRA, diffusers spelling
    "lora_linear_layer.",                          # LoRA, another diffusers spelling
    "hada_w", "hada_t",                            # LoHa
    "lokr_w", "lokr_t",                            # LoKr
    "oft_blocks", "oft_R.",                        # OFT / BOFT / OFTv2
    "dora_scale",                                  # DoRA, layered on the above
    ".diff.", ".diff_b",                           # plain difference patches
)

# GLoRA names its factors a1/a2/b1/b2, which are too generic to test one at a
# time, so it is recognised only when the pair appears together.
_GLORA_MARKERS = (".a1.weight", ".b1.weight")

# (marker, name) in priority order -- the first hit names the algorithm.
_ALGORITHMS = (
    ("hada_w", "LoHa (LyCORIS, Hadamard product)"),
    ("lokr_w", "LoKr (LyCORIS, Kronecker product)"),
    ("oft_R.", "OFTv2 (orthogonal fine-tuning)"),
    ("oft_blocks", "OFT / BOFT (orthogonal fine-tuning)"),
    ("lora_down", "LoRA"),
    ("lora_A", "LoRA"),
    ("lora.down.weight", "LoRA"),
    ("lora_linear_layer.", "LoRA"),
    (".diff.", "Plain difference patch"),
)

_TARGET_GROUPS = ("self_attn", "cross_attn", "mlp", "adaln")

# Same threshold and reasoning as lora_bake.py::_lora_touches_llm_adapter.
_LLM_ADAPTER_SIGNIFICANCE = 0.01

# Caps for the sampled magnitude check, so inspection stays fast on a network
# share: at most this many tensors, and this many bytes from each.
_SAMPLE_TENSORS = 12
_SAMPLE_BYTES = 32768


def _human_size(n: int) -> str:
    mb = n / (1024 * 1024)
    return f"{mb / 1024:.2f} GB ({mb:,.0f} MB)" if mb >= 1024 else f"{mb:,.1f} MB"


# --- LoRA -----------------------------------------------------------------


def _lora_block_info(keys: list[str]) -> dict[str, Any]:
    """Which Anima generation a LoRA targets, decided exactly the way Forge
    decides it at load time: find the highest main-block index it references,
    then round up to the next known generation size."""
    prefix, sep = next(
        ((p, s) for p, s in _LORA_PREFIXES if any(k.startswith(p) for k in keys)),
        (None, None),
    )
    if prefix is None:
        return {"prefix": None, "sep": None, "indices": set(), "max_block": None, "generation": None}

    indices = set()
    for k in keys:
        if not k.startswith(prefix):
            continue
        num, found, _ = k[len(prefix):].partition(sep)
        if found and num.isdigit():
            indices.add(int(num))

    raw = (max(indices) + 1) if indices else 0
    generation = next((size for size in ANIMA_BLOCK_SIZES if raw <= size), None) if raw else None
    return {
        "prefix": prefix,
        "sep": sep,
        "indices": indices,
        "max_block": (raw - 1) if raw else None,
        "generation": generation,
    }


# A LoRA with no Anima block keys is not broken -- it targets a different
# architecture. Naming which one is more use than calling it unrecognised,
# since a misfiled LoRA is otherwise hard to tell from a damaged one.
_FOREIGN_LAYOUTS = (
    (("input_blocks", "output_blocks", "middle_block"), "SD1.x / SDXL UNet"),
    (("double_blocks", "single_blocks"), "Flux"),
    (("joint_blocks",), "SD3"),
    (("transformer_blocks",), "DiT (transformer_blocks layout)"),
)


def _foreign_layout(keys: list[str]) -> str | None:
    joined = "\n".join(keys[:2000])
    for markers, name in _FOREIGN_LAYOUTS:
        if any(m in joined for m in markers):
            return name
    return None


def _is_glora(joined: str) -> bool:
    return all(m in joined for m in _GLORA_MARKERS)


def _lora_algorithm(keys: list[str]) -> str:
    """Which adapter algorithm the file uses. DoRA is a modifier rather than a
    format of its own, so it is appended to whatever it sits on top of."""
    joined = "\n".join(keys)
    name = next((n for marker, n in _ALGORITHMS if marker in joined), None)
    if name is None:
        name = "GLoRA" if _is_glora(joined) else "Unrecognised adapter format"
    if "dora_scale" in joined:
        name += " + DoRA"
    return name


# Where the rank lives, per algorithm. LoKr's effective rank is a property of
# its factorisation rather than one tensor dimension, so it is read from the
# trainer's own metadata instead of guessed from a shape.
_RANK_TENSORS = ("lora_down", "lora_A", "hada_w1_b", "lokr_w2_b")


def _lora_ranks(header: dict[str, Any]) -> dict[int, int]:
    """{rank: how many modules use it}. A single entry means a uniform rank;
    several means it was chosen per layer."""
    ranks: dict[int, int] = {}
    for k, v in header.items():
        if k == "__metadata__" or not isinstance(v, dict):
            continue
        if any(t in k for t in _RANK_TENSORS):
            shape = v.get("shape") or []
            if shape:
                ranks[shape[0]] = ranks.get(shape[0], 0) + 1
    return ranks


def _lora_targets(keys: list[str], prefix: str | None, sep: str | None) -> dict[str, int]:
    """How many distinct modules per block the LoRA touches, grouped by the
    part of the block they belong to."""
    if not prefix:
        return {}
    mods: set[str] = set()
    for k in keys:
        m = re.search(re.escape(prefix) + r"\d+" + re.escape(sep) + r"(.+?)\.(?:lora_|alpha)", k)
        if m:
            mods.add(m.group(1))
    out = {g: sum(1 for m in mods if m.startswith(g)) for g in _TARGET_GROUPS}
    out["total"] = len(mods)
    return out


def _decode(raw: bytes, dtype: str):
    """Returns a list of floats, or None for a dtype we don't need to read."""
    if dtype == "F32":
        return struct.unpack(f"<{len(raw) // 4}f", raw[: len(raw) // 4 * 4])
    if dtype == "F16":
        import numpy as np

        return np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.float16).astype(float).tolist()
    if dtype == "BF16":
        n = len(raw) // 2
        return struct.unpack(f"<{n}f", b"".join(b"\x00\x00" + raw[i * 2 : i * 2 + 2] for i in range(n)))
    return None


def _llm_adapter_ratio(path: str, header: dict[str, Any], data_offset: int) -> float | None:
    """RMS of the LoRA's llm_adapter factors relative to its main-block ones.

    Presence of llm_adapter keys says nothing on its own: a LoRA extracted by
    SVD emits a factor pair for every module it scanned, so modules whose delta
    was zero still appear, carrying only numerical noise. Magnitude separates
    those from a LoRA that genuinely trained the adapter.

    Samples a bounded slice rather than reading whole tensors, so this stays
    quick even over a network share. Returns None if nothing could be read.
    """

    def _pick(is_llm: bool) -> list[str]:
        out = [
            k
            for k, v in header.items()
            if k != "__metadata__"
            and isinstance(v, dict)
            and any(m in k for m in _FACTOR_MARKERS)
            and (("llm_adapter" in k) == is_llm)
        ]
        return sorted(out)[:_SAMPLE_TENSORS]

    def _rms(keys: list[str], f) -> float | None:
        total = count = 0
        for k in keys:
            info = header[k]
            start, end = info["data_offsets"]
            n = min(_SAMPLE_BYTES, end - start)
            f.seek(data_offset + start)
            vals = _decode(f.read(n), info.get("dtype", ""))
            if not vals:
                continue
            total += sum(x * x for x in vals)
            count += len(vals)
        return (total / count) ** 0.5 if count else None

    llm_keys, main_keys = _pick(True), _pick(False)
    if not llm_keys or not main_keys:
        return None
    try:
        with open(path, "rb") as f:
            llm_rms, main_rms = _rms(llm_keys, f), _rms(main_keys, f)
    except Exception:
        return None
    if llm_rms is None or not main_rms:
        return None
    return llm_rms / main_rms


def lora_activation_text(path: str) -> tuple[str, str]:
    """(text, where it came from). Forge's own sidecar <name>.json wins, since
    that is what the user typed in the Lora tab; a trainer's embedded
    ss_output_name is a weak fallback and is labelled as such."""
    try:
        from modules import extra_networks

        text = (extra_networks.get_user_metadata(path).get("activation text") or "").strip()
        if text:
            return text, "sidecar"
    except Exception:
        pass

    sidecar = os.path.splitext(path)[0] + ".json"
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = (data.get("activation text") or "").strip()
            if text:
                return text, "sidecar"
        except Exception:
            pass
    return "", ""


def _extraction_info(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """A LoRA produced by subtracting two checkpoints rather than by training
    records how it was made. Recognising that matters: it tells you the LoRA is
    a difference, and which direction it was taken in."""
    if not metadata:
        return None
    fmt = str(metadata.get("format", ""))
    mode = str(metadata.get("mode", ""))
    if "delta" not in fmt.lower() and mode.lower() not in ("svd",):
        return None
    return {
        "format": fmt or "(unspecified)",
        "mode": mode or "(unspecified)",
        "base_prefix": metadata.get("base_prefix", ""),
        "target_prefix": metadata.get("target_prefix", ""),
        "subtraction_dtype": metadata.get("subtraction_dtype", ""),
        "output_dtype": metadata.get("output_dtype", ""),
        "rank": metadata.get("rank", ""),
    }


def inspect_lora(filepath: str) -> dict[str, Any]:
    """Structured description of a LoRA, from its header plus a bounded sample
    of tensor bytes for the LLM-adapter magnitude check."""
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}
    if not filepath.endswith(".safetensors"):
        return {"error": "Only .safetensors LoRAs support header inspection."}

    try:
        header, data_offset = read_safetensors_header(filepath)
    except Exception as e:
        return {"error": f"Failed to read header: {e}"}

    metadata = header.get("__metadata__", {}) or {}
    keys = [k for k in header if k != "__metadata__"]

    blocks = _lora_block_info(keys)
    ranks = _lora_ranks(header)
    targets = _lora_targets(keys, blocks["prefix"], blocks["sep"])
    llm_keys = [k for k in keys if "llm_adapter" in k]
    ratio = _llm_adapter_ratio(filepath, header, data_offset) if llm_keys else None
    text, text_source = lora_activation_text(filepath)
    name = os.path.basename(filepath)

    size = os.path.getsize(filepath)
    return {
        "kind": "lora",
        "filename": name,
        "filepath": filepath,
        "file_size": size,
        "size_str": _human_size(size),
        "total_tensors": len(keys),
        "precision": _detect_precision(header),
        "algorithm": _lora_algorithm(keys),
        "generation": blocks["generation"],
        "foreign_layout": None if blocks["generation"] else _foreign_layout(keys),
        "max_block": blocks["max_block"],
        "blocks_touched": len(blocks["indices"]),
        "covers_all_blocks": (
            blocks["generation"] is not None and len(blocks["indices"]) == blocks["generation"]
        ),
        "key_convention": (
            f"{blocks['prefix']}<N>{blocks['sep']}" if blocks["prefix"] else "(no Anima block keys found)"
        ),
        "ranks": ranks,
        "uniform_rank": (next(iter(ranks)) if len(ranks) == 1 else None),
        "declared_dim": metadata.get("ss_network_dim", ""),
        "targets": targets,
        "llm_adapter_tensors": len(llm_keys),
        "llm_adapter_ratio": ratio,
        "llm_adapter_significant": (
            None if ratio is None else ratio > _LLM_ADAPTER_SIGNIFICANCE
        ),
        "activation_text": text,
        "activation_text_source": text_source,
        "is_turbo": "turbo" in name.lower(),
        "extraction": _extraction_info(metadata),
        "raw_metadata": metadata,
    }


# --- Text encoders and VAEs ------------------------------------------------

# Which encoder a module is, keyed on a tensor prefix Forge's own configs use
# (huggingface_guess/model_list.py::clip_target).
_ENCODER_SIGNATURES = (
    ("qwen3_06b", "Qwen3 0.6B — Anima's native text encoder"),
    ("qwen35_4b", "Qwen3.5 4B — Anima-3.8B expanded adapter"),
    ("qwen3_4b", "Qwen3 4B — Z-Image"),
    ("qwen3vl_4b", "Qwen3-VL 4B — Krea 2"),
    ("qwen25_7b", "Qwen2.5 7B — Qwen-Image"),
    ("ministral3_3b", "Ministral3 3B — Ernie-Image"),
    ("gemma2_2b", "Gemma2 2B — Lumina / PiD"),
    ("t5xxl", "T5-XXL"),
    ("clip_g", "CLIP-G"),
    ("clip_l", "CLIP-L"),
)


# A standalone encoder file carries no "qwen3_06b."-style prefix -- that only
# appears once it is bundled into a checkpoint. Standalone files are told apart
# by shape instead: (transformer layers, hidden size, has a vision tower).
# Measured from real files; the "used by" half comes from Forge's own
# clip_target declarations in huggingface_guess/model_list.py.
_ENCODER_SHAPES = {
    (28, 1024, False): "Qwen3 0.6B — Anima's native text encoder",
    (36, 2560, False): "Qwen3 4B — Z-Image, and Anima-3.8B's expanded adapter",
    (36, 2560, True): "Qwen3-VL 4B — Krea 2",
}


def _encoder_shape(keys: list[str], header: dict[str, Any]) -> tuple[int, int, bool] | None:
    layers = {int(m.group(1)) for k in keys if (m := re.search(r"\blayers\.(\d+)\.", k))}
    if not layers:
        return None
    embed = next(
        (header[k].get("shape") for k in keys if k.endswith("embed_tokens.weight") and isinstance(header.get(k), dict)),
        None,
    )
    hidden = embed[1] if embed and len(embed) > 1 else 0
    has_vision = any(k.startswith("visual.") or ".visual." in k for k in keys)
    return len(layers), hidden, has_vision


def _classify_module(keys: list[str], header: dict[str, Any]) -> tuple[str, str]:
    """(kind, description) for an auxiliary module file."""
    # A file that still carries a bundled prefix names itself outright.
    joined = "\n".join(keys[:4000])
    for token, desc in _ENCODER_SIGNATURES:
        if token in joined:
            return "text_encoder", desc

    if any(k.startswith(("encoder.", "decoder.", "first_stage_model.", "vae.")) for k in keys):
        if any("time_conv" in k or "conv_in.conv" in k for k in keys):
            return "vae", "Video-capable VAE (Wan / Anima / Qwen-Image family)"
        return "vae", "Autoencoder"

    shape = _encoder_shape(keys, header)
    if shape is not None:
        layers, hidden, vision = shape
        known = _ENCODER_SHAPES.get(shape)
        if known:
            return "text_encoder", known
        # Report what was measured rather than guessing a name.
        return "text_encoder", (
            f"Transformer text encoder — {layers} layers, hidden size {hidden}"
            + (", with a vision tower" if vision else "")
            + " (not one of the layouts recognised here)"
        )
    return "unknown", "Could not classify from tensor names"


def detect_file_kind(filepath: str) -> str:
    """"lora" | "checkpoint" | "module" | "unknown", decided from the header.

    A file states what it is, so nothing has to be told which type it is --
    and a LoRA filed under the checkpoints folder still reads as a LoRA.

    LoRA is tested first on purpose: its keys look like a diffusion model's
    (`diffusion_model.blocks.0....`) and are only distinguishable by the
    factor suffix, so testing for a checkpoint first would swallow it.
    """
    try:
        header, _ = read_safetensors_header(filepath)
    except Exception:
        return "unknown"

    keys = [k for k in header if k != "__metadata__"]
    if not keys:
        return "unknown"

    joined = "\n".join(keys)
    if any(m in joined for m in _FACTOR_MARKERS) or _is_glora(joined):
        return "lora"

    diffusion_markers = (
        "model.diffusion_model.",
        "diffusion_model.",
        "double_blocks.",
        "joint_blocks.",
        "input_blocks.",
        "net.blocks.",
        "blocks.",
    )
    if any(k.startswith(diffusion_markers) for k in keys):
        return "checkpoint"

    kind, _ = _classify_module(keys, header)
    return "module" if kind in ("text_encoder", "vae") else "unknown"


def inspect_module(filepath: str) -> dict[str, Any]:
    """Structured description of a text encoder or VAE file."""
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}
    if not filepath.endswith(".safetensors"):
        return {"error": "Only .safetensors modules support header inspection."}

    try:
        header, _ = read_safetensors_header(filepath)
    except Exception as e:
        return {"error": f"Failed to read header: {e}"}

    keys = [k for k in header if k != "__metadata__"]
    kind, desc = _classify_module(keys, header)
    size = os.path.getsize(filepath)
    return {
        "kind": kind,
        "description": desc,
        "filename": os.path.basename(filepath),
        "filepath": filepath,
        "file_size": size,
        "size_str": _human_size(size),
        "total_tensors": len(keys),
        "precision": _detect_precision(header),
        "raw_metadata": header.get("__metadata__", {}) or {},
    }


# --- Rendering -------------------------------------------------------------

_CARD = "background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 14px;"
_GREEN, _AMBER, _RED, _GREY, _BLUE = "#10b981", "#f59e0b", "#ef4444", "#9ca3af", "#38bdf8"


def _err_card(msg: str) -> str:
    return (
        f"<div style='padding: 16px; border-radius: 8px; background: rgba(239,68,68,0.1); "
        f"border: 1px solid {_RED}; color: #fca5a5;'><b>Error:</b> {msg}</div>"
    )


def _badge(text: str, color: str) -> str:
    return (
        f"<span style='background: {color}22; color: {color}; border: 1px solid {color}66; "
        f"padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;'>{text}</span>"
    )


def _row(label: str, value: str) -> str:
    return (
        f"<div style='display:flex; justify-content:space-between; gap:16px; padding:5px 0; "
        f"border-bottom:1px solid rgba(255,255,255,0.05); font-size:13px;'>"
        f"<span style='color:{_GREY};'>{label}</span><span>{value}</span></div>"
    )


def _header_block(info: dict[str, Any], badges: str) -> str:
    return (
        f"<div style='display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;'>"
        f"<div><div style='font-size:17px; font-weight:bold;'>{info['filename']}</div>"
        f"<div style='color:{_GREY}; font-size:12px; margin-top:2px;'>"
        f"{info['size_str']} &nbsp;&bull;&nbsp; {info['total_tensors']:,} tensors</div></div>"
        f"<div style='display:flex; gap:6px; flex-wrap:wrap;'>{badges}</div></div>"
    )


def _activation_card(info: dict[str, Any]) -> str:
    """The one thing you have to act on, so it leads the card."""
    text = info.get("activation_text", "")
    if text:
        return (
            f"<div style='{_CARD} border-color:{_AMBER}66; background:rgba(245,158,11,0.08);'>"
            f"<div style='color:{_AMBER}; font-weight:bold; font-size:12px; margin-bottom:6px;'>"
            f"TRIGGER WORD REQUIRED IN THE PROMPT</div>"
            f"<div style='font-family:monospace; font-size:14px; user-select:all;'>{text}</div>"
            f"<div style='color:{_GREY}; font-size:11px; margin-top:6px;'>From the sidecar metadata. "
            f"Baking this LoRA into a checkpoint does not remove the need to type it.</div></div>"
        )
    if info.get("is_turbo"):
        return (
            f"<div style='{_CARD}'><div style='color:{_GREEN}; font-weight:bold; font-size:12px;'>"
            f"NO TRIGGER WORD</div><div style='color:{_GREY}; font-size:12px; margin-top:4px;'>"
            f"Acceleration LoRAs change sampling behaviour rather than a concept keyed to a token, "
            f"so they apply to every prompt.</div></div>"
        )
    return (
        f"<div style='{_CARD}'><div style='color:{_GREY}; font-weight:bold; font-size:12px;'>"
        f"NO TRIGGER WORD RECORDED</div><div style='color:{_GREY}; font-size:12px; margin-top:4px;'>"
        f"Either this LoRA needs none, or none was set in the Lora tab's <i>Activation Text</i> field. "
        f"Merge Studio reads that field, so filling it in there makes the trigger travel into any "
        f"checkpoint you bake this LoRA into.</div></div>"
    )


def _llm_adapter_line(info: dict[str, Any]) -> str:
    ratio = info.get("llm_adapter_ratio")
    count = info.get("llm_adapter_tensors")
    if not count:
        return f"<span style='color:{_GREEN};'>absent</span>"
    if ratio is None:
        return f"{count} tensors <span style='color:{_GREY};'>(magnitude not readable)</span>"
    if info.get("llm_adapter_significant"):
        return (
            f"<span style='color:{_AMBER};'>modified</span> "
            f"<span style='color:{_GREY};'>({count} tensors, {ratio:.2f}x the main blocks)</span>"
        )
    return (
        f"<span style='color:{_GREEN};'>present but empty</span> "
        f"<span style='color:{_GREY};'>({count} tensors at {ratio:.1e} of the main blocks &mdash; "
        f"extraction noise, not a real change)</span>"
    )


def format_lora_dashboard_html(info: dict[str, Any]) -> str:
    if not info:
        return f"<div style='padding:20px; color:{_GREY};'>Select a LoRA to inspect.</div>"
    if "error" in info:
        return _err_card(info["error"])

    gen = info.get("generation")
    foreign = info.get("foreign_layout")
    if gen:
        badges = _badge(f"Anima {gen}-block", _BLUE)
    elif foreign:
        badges = _badge(f"Not Anima — {foreign}", _AMBER)
    else:
        badges = _badge("Unrecognised block layout", _GREY)
    badges += " " + _badge(info.get("precision", "?"), _GREY)
    if info.get("is_turbo"):
        badges += " " + _badge("Turbo", _AMBER)
    if info.get("extraction"):
        badges += " " + _badge("Extracted", _BLUE)

    ranks = info.get("ranks") or {}
    if info.get("uniform_rank"):
        rank_str = f"{info['uniform_rank']} (uniform)"
    elif ranks:
        rank_str = f"adaptive, {min(ranks)}&ndash;{max(ranks)} across {sum(ranks.values())} modules"
    elif info.get("declared_dim"):
        rank_str = f"{info['declared_dim']} <span style='color:{_GREY};'>(declared by the trainer)</span>"
    else:
        rank_str = "&mdash;"

    t = info.get("targets") or {}
    target_str = (
        f"{t.get('total', 0)} per block <span style='color:{_GREY};'>"
        f"(self-attn {t.get('self_attn', 0)}, cross-attn {t.get('cross_attn', 0)}, "
        f"mlp {t.get('mlp', 0)}, adaln {t.get('adaln', 0)})</span>"
    ) if t else "&mdash;"

    if gen:
        coverage = f"{info['blocks_touched']} of {gen}"
        if not info.get("covers_all_blocks"):
            coverage += f" <span style='color:{_AMBER};'>(partial)</span>"
    else:
        coverage = "&mdash;"

    if gen:
        arch_row = f"Anima, {gen}-block"
    elif foreign:
        arch_row = f"<span style='color:{_AMBER};'>{foreign}</span> &mdash; not an Anima LoRA"
    else:
        arch_row = "unrecognised"
    body = _row("Adapter format", info.get("algorithm", "&mdash;"))
    body += _row("Architecture", arch_row)
    body += _row("Blocks covered", coverage)
    body += _row("Rank", rank_str)
    body += _row("Modules touched", target_str)
    body += _row("LLM adapter", _llm_adapter_line(info))
    body += _row("Key convention", f"<code>{info.get('key_convention')}</code>")

    warn = ""
    if info.get("llm_adapter_significant"):
        warn = (
            f"<div style='{_CARD} border-color:{_AMBER}66; background:rgba(245,158,11,0.08); margin-top:10px;'>"
            f"<span style='color:{_AMBER}; font-weight:bold;'>Changes the LLM adapter.</span> "
            f"<span style='font-size:12px;'>Anima's own training guidance says never to train it alongside a "
            f"LoRA. Baking this alters the checkpoint's text understanding permanently.</span></div>"
        )

    ext = info.get("extraction")
    ext_html = ""
    if ext:
        ext_html = (
            f"<div style='{_CARD} margin-top:10px;'>"
            f"<div style='font-weight:bold; font-size:12px; margin-bottom:6px;'>"
            f"Produced by extraction, not training</div>"
            + _row("Format / mode", f"<code>{ext['format']}</code> / <code>{ext['mode']}</code>")
            + (
                _row("Taken from", f"<code>{ext['base_prefix']}</code> &rarr; <code>{ext['target_prefix']}</code>")
                if ext.get("base_prefix")
                else ""
            )
            + (
                _row("Subtraction precision", f"<code>{ext['subtraction_dtype']}</code>")
                if ext.get("subtraction_dtype")
                else ""
            )
            + "</div>"
        )

    remap = ""
    if gen and gen < max(ANIMA_BLOCK_SIZES):
        remap = (
            f"<div style='{_CARD} margin-top:10px; font-size:12px; color:{_GREY};'>"
            f"Built for the {gen}-block generation. Forge remaps it automatically onto larger Anima models, "
            f"including onto the blocks those generations inserted.</div>"
        )

    return (
        f"<div style='{_CARD} line-height:1.5;'>"
        + _header_block(info, badges)
        + _activation_card(info)
        + f"<div style='{_CARD} margin-top:10px;'>{body}</div>"
        + warn
        + ext_html
        + remap
        + "</div>"
    )


def format_module_dashboard_html(info: dict[str, Any]) -> str:
    if not info:
        return f"<div style='padding:20px; color:{_GREY};'>Select a text encoder or VAE to inspect.</div>"
    if "error" in info:
        return _err_card(info["error"])

    kind = info.get("kind", "unknown")
    label = {"text_encoder": "Text Encoder", "vae": "VAE"}.get(kind, "Unrecognised")
    colour = _GREEN if kind in ("text_encoder", "vae") else _GREY
    badges = _badge(label, colour) + " " + _badge(info.get("precision", "?"), _GREY)

    body = _row("Identified as", info.get("description", "&mdash;"))
    body += _row("Precision", f"<code>{info.get('precision')}</code>")

    note = ""
    if kind == "text_encoder":
        note = (
            f"<div style='{_CARD} margin-top:10px; font-size:12px; color:{_GREY};'>"
            f"A checkpoint conditions on the encoder it was built for. Pairing it with a different one still "
            f"runs, but the result reflects the wrong conditioning &mdash; and in Full Checkpoint mode the "
            f"encoder loaded here is the one baked into the output.</div>"
        )

    return (
        f"<div style='{_CARD} line-height:1.5;'>"
        + _header_block(info, badges)
        + f"<div style='{_CARD}'>{body}</div>"
        + note
        + "</div>"
    )
