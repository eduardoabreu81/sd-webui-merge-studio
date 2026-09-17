import datetime
import json
import os
import shutil
import sys
import traceback

import gradio as gr

from modules import call_queue, script_callbacks, sd_models, shared
from modules.ui_common import create_refresh_button
from modules.ui_components import FormRow, InputAccordion

_EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)

import aux_inspector  # noqa: E402
import checkpoint_merge  # noqa: E402
import checkpoint_quantize  # noqa: E402
import lora_bake  # noqa: E402
import quant_repair  # noqa: E402
from checkpoint_inspector import (
    available_vaes,
    format_badges_html,
    format_recipe_dashboard_html,
    inspect_checkpoint,
)
from quant_utils import OUTPUT_FORMAT_CHOICES, debug_print  # noqa: E402


NONE_LABEL = "(none)"
ORIGINAL_VAE_LABEL = "Original / Merge from Models"
NO_VAE_LABEL = "None (Strip VAE)"


def _vae_choices() -> list[str]:
    vaes = available_vaes()
    return [ORIGINAL_VAE_LABEL, NO_VAE_LABEL] + sorted(vaes.keys())


_checkpoint_info_cache: dict[tuple[str, float], dict] = {}


def _get_cached_checkpoint_info(name: str) -> dict | None:
    if not name:
        return None
    try:
        path = _checkpoint_path(name)
        mtime = os.path.getmtime(path) if os.path.exists(path) else 0
        cache_key = (path, mtime)
        if cache_key in _checkpoint_info_cache:
            return _checkpoint_info_cache[cache_key]
        info = inspect_checkpoint(path)
        _checkpoint_info_cache[cache_key] = info
        return info
    except Exception:
        return None


def primary_badge_handler(p_name: str, s_name: str, t_name: str):
    p_info = _get_cached_checkpoint_info(p_name)
    s_info = _get_cached_checkpoint_info(s_name)
    t_info = _get_cached_checkpoint_info(t_name)

    p_html = format_badges_html(p_info) if p_info else ""
    s_html = format_badges_html(s_info, compatible_with_info=p_info) if s_info else ""
    t_html = format_badges_html(t_info, compatible_with_info=p_info) if t_info else ""
    return p_html, s_html, t_html


def secondary_badge_handler(p_name: str, s_name: str):
    p_info = _get_cached_checkpoint_info(p_name)
    s_info = _get_cached_checkpoint_info(s_name)
    return format_badges_html(s_info, compatible_with_info=p_info) if s_info else ""


def tertiary_badge_handler(p_name: str, t_name: str):
    p_info = _get_cached_checkpoint_info(p_name)
    t_info = _get_cached_checkpoint_info(t_name)
    return format_badges_html(t_info, compatible_with_info=p_info) if t_info else ""


def _module_choices() -> list[str]:
    """Text encoders and VAEs, from the same list Forge offers as Additional
    Modules -- so what can be inspected here matches what can actually be
    loaded alongside a checkpoint."""
    try:
        from modules_forge import main_entry

        if not getattr(main_entry, "module_list", None):
            main_entry.refresh_models()
        return sorted(main_entry.module_list.keys())
    except Exception:
        try:
            return sorted(available_vaes().keys())
        except Exception:
            return []


# One list for every inspectable model. The prefix is only there so the
# entries stay grouped and typing "lora" narrows the list -- what a file
# actually is gets decided from its header, not from which pool it came out of.
_POOL_PREFIXES = (("Checkpoint", "ckpt"), ("LoRA", "lora"), ("Module", "module"))


def _inspect_choices() -> list[str]:
    def _safe(fn):
        try:
            return sorted(fn())
        except Exception:
            return []

    out = [f"[ckpt] {n}" for n in _safe(sd_models.checkpoint_tiles)]
    out += [f"[lora] {n}" for n in _safe(lambda: lora_bake.available_loras().keys())]
    out += [f"[module] {n}" for n in _safe(_module_choices)]
    return out


def _inspect_path(entry: str) -> str:
    """Resolves a prefixed dropdown entry back to a file path."""
    label, _, name = entry.partition("] ")
    pool = label.lstrip("[")

    if pool == "lora":
        path = lora_bake.available_loras().get(name)
    elif pool == "module":
        path = None
        try:
            from modules_forge import main_entry

            path = (main_entry.module_list or {}).get(name)
        except Exception:
            pass
        path = path or available_vaes().get(name)
    else:
        return _checkpoint_path(name or entry)

    if not path:
        raise ValueError(f"Not found: {name}")
    return path


def run_inspection_handler(entry: str) -> str:
    """Inspects whatever was selected, choosing the card from what the file
    says it is rather than from where it was listed -- so a LoRA sitting in
    the checkpoints folder still reads as a LoRA."""
    if not entry:
        return "<div style='padding: 20px; color: #9ca3af;'>Select a model above to inspect it.</div>"
    try:
        path = _inspect_path(entry)
        kind = aux_inspector.detect_file_kind(path)
        if kind == "lora":
            return aux_inspector.format_lora_dashboard_html(aux_inspector.inspect_lora(path))
        if kind == "module":
            return aux_inspector.format_module_dashboard_html(aux_inspector.inspect_module(path))
        return format_recipe_dashboard_html(inspect_checkpoint(path))
    except Exception as e:
        return _err_html(e)

FORMAT_LABEL_TO_KEY = dict(OUTPUT_FORMAT_CHOICES)

COMPATIBLE_MODELS_NOTE = (
    "**Which models work here:** anything Forge Neo can load as a normal checkpoint through the standard model loader "
    "(SD1, SDXL, Flux, Anima, etc.) in plain precision (fp16/bf16) or quantized via the MixedPrecisionOps system "
    "(fp8/int8/nvfp4/mxfp8/convrot). **Does NOT work** for Nunchaku/SVDQuant checkpoints or GGUF/nf4/fp4 storage — "
    "those store weights in a completely different way and are rejected with a clear error instead of producing a "
    "broken file."
)

INTERP_CHOICES = [
    ("No Interpolation", checkpoint_merge.INTERP_NO_INTERPOLATION),
    ("Weighted Sum", checkpoint_merge.INTERP_WEIGHTED_SUM),
    ("Add Difference", checkpoint_merge.INTERP_ADD_DIFFERENCE),
]
INTERP_LABEL_TO_KEY = dict(INTERP_CHOICES)


def _checkpoint_path(name: str) -> str:
    info = sd_models.checkpoint_aliases.get(name)
    if info is None:
        raise ValueError(f"Checkpoint not found: {name}")
    return info.filename


def _err_html(e: Exception) -> str:
    debug_print(f"error:\n{traceback.format_exc()}")
    return f"<div class='error'>Error: {e}</div>"


def _format_dropdown(label: str, default_same=True):
    choices = [lbl for lbl, _ in OUTPUT_FORMAT_CHOICES]
    value = choices[0] if default_same else choices[1]
    return gr.Dropdown(label=label, choices=choices, value=value)


# --- Quant Format Doctor -----------------------------------------------


def diagnose_handler(checkpoint_name: str):
    if not checkpoint_name:
        return "<div>Select a checkpoint.</div>"
    try:
        path = _checkpoint_path(checkpoint_name)
        broken = quant_repair.diagnose(path)
        if not broken:
            return "<div>No problems found in this checkpoint.</div>"

        formats = sorted({b.inferred_format for b in broken if b.inferred_format})
        examples = ", ".join(b.key for b in broken[:3])
        unresolved = sum(1 for b in broken if b.inferred_format is None)

        html = (
            f"<div><b>{len(broken)} layer(s)</b> have quantization metadata missing the 'format' field.</div>"
            f"<div>Inferred format: {', '.join(formats) if formats else 'could not infer'}</div>"
            f"<div>Examples: {examples}</div>"
        )
        if unresolved:
            html += f"<div style='color:orange'>{unresolved} layer(s) could not have their format inferred — repair will abort for safety.</div>"
        return html
    except Exception as e:
        return _err_html(e)


def repair_handler(id_task, checkpoint_name: str, mode: str, new_filename: str):
    if not checkpoint_name:
        gr.Warning("Select a checkpoint.")
        return gr.update(), "<div>Select a checkpoint.</div>"
    try:
        path = _checkpoint_path(checkpoint_name)

        last_printed_decile = [-1]

        def progress_cb(done, total):
            if total:
                pct = 100 * done / total
                shared.state.textinfo = f"Repairing {os.path.basename(path)}: {pct:.0f}%"
                shared.state.sampling_steps = 100
                shared.state.sampling_step = int(pct)
                decile = int(pct) // 10
                if decile != last_printed_decile[0]:
                    last_printed_decile[0] = decile
                    debug_print(f"[fix] {os.path.basename(path)}: {pct:.0f}%")

        if mode == "inplace":
            result = quant_repair.repair_in_place(path, progress_cb=progress_cb)
        else:
            if not new_filename:
                raise ValueError("Enter a filename for the repaired copy.")
            if not new_filename.endswith(".safetensors"):
                new_filename += ".safetensors"
            dst = os.path.join(os.path.dirname(path), new_filename)
            result = quant_repair.repair(path, dst, progress_cb=progress_cb)

        if result["output"] is None:
            gr.Info("Nothing to repair.", duration=5)
            return gr.update(), "<div>Nothing to repair.</div>"

        sd_models.list_models()
        size_str = _format_size(result["output"])
        html = (
            f"<div style='margin-top: 10px; line-height: 1.6; font-size: 14px;'>"
            f"<b>Repaired checkpoint:</b> <code>{result['output']}</code><br>"
            f"<b>File size:</b> <span style='color: #10b981; font-weight: bold;'>{size_str}</span><br>"
            f"<b>Repaired layers:</b> {result['fixed']} ({', '.join(result['formats'])})"
            f"</div>"
        )
        return gr.update(choices=sorted(sd_models.checkpoint_tiles())), html
    except Exception as e:
        gr.Warning("Failed to repair checkpoint.", duration=5)
        return gr.update(), _err_html(e)


# --- LoRA -> Checkpoint (bake) ------------------------------------------


def _lora_choices():
    try:
        loras = lora_bake.available_loras()
    except Exception:
        loras = {}
    return [NONE_LABEL] + sorted(loras.keys())


DEVICE_CHOICES = [
    ("Auto (recommended)", "auto"),
    ("GPU (force -- may run out of VRAM on smaller cards)", "gpu"),
    ("CPU (force -- always fits, slower)", "cpu"),
]
DEVICE_LABEL_TO_KEY = dict(DEVICE_CHOICES)

SAVE_MODE_CHOICES = [
    ("UNet Only (smaller, needs external VAE/text encoder)", "unet_only"),
    ("Full Checkpoint (UNet + CLIP + VAE, self-contained, bigger)", "full"),
]
SAVE_MODE_LABEL_TO_KEY = dict(SAVE_MODE_CHOICES)


def _format_size(path: str) -> str:
    if not os.path.exists(path):
        return "N/A"
    bytes_size = os.path.getsize(path)
    gb = bytes_size / (1024 ** 3)
    mb = bytes_size / (1024 ** 2)
    if gb >= 1.0:
        return f"{gb:.2f} GB ({mb:,.0f} MB)"
    return f"{mb:.1f} MB ({bytes_size:,} bytes)"


def _make_progress_cb(tag: str):
    shared.state.job_count = 1
    shared.state.sampling_steps = 100
    shared.state.sampling_step = 0

    def progress_cb(msg):
        debug_print(f"[{tag}] {msg}")
        shared.state.job_count = 1
        shared.state.textinfo = str(msg)
        if "(" in msg and "/" in msg and ")" in msg:
            try:
                part = msg.split("(", 1)[1].split(")", 1)[0]
                cur, tot = part.split("/")
                shared.state.sampling_steps = int(tot)
                shared.state.sampling_step = int(cur)
            except Exception:
                pass
        elif "loading" in msg.lower():
            shared.state.sampling_step = 5
        elif "applying lora" in msg.lower():
            shared.state.sampling_step = 25
        elif "materializing" in msg.lower():
            shared.state.sampling_step = 40
        elif "assembling" in msg.lower():
            shared.state.sampling_step = 88
        elif "saving" in msg.lower():
            shared.state.sampling_step = 95

    return progress_cb


def bake_handler(
    id_task,
    checkpoint_name: str,
    lora1: str,
    strength1: float,
    lora2: str,
    strength2: float,
    lora3: str,
    strength3: float,
    format_label: str,
    clip_format_label: str,
    vae_format_label: str,
    save_mode_label: str,
    device_label: str,
    output_filename: str,
):
    if not checkpoint_name:
        gr.Warning("Select a base checkpoint.")
        return gr.update(), "<div>Select a base checkpoint.</div>"
    try:
        checkpoint_path = _checkpoint_path(checkpoint_name)
        loras_by_name = lora_bake.available_loras()

        selected = []
        for name, strength in ((lora1, strength1), (lora2, strength2), (lora3, strength3)):
            if name and name != NONE_LABEL:
                if name not in loras_by_name:
                    raise ValueError(f"LoRA not found: {name}")
                selected.append((loras_by_name[name], strength))

        if not selected:
            raise ValueError("Select at least one LoRA.")

        if not output_filename:
            raise ValueError("Enter an output filename.")
        if not output_filename.endswith(".safetensors"):
            output_filename += ".safetensors"
        output_path = os.path.join(sd_models.model_path, output_filename)

        output_format = FORMAT_LABEL_TO_KEY.get(format_label, "same")
        clip_output_format = FORMAT_LABEL_TO_KEY.get(clip_format_label, "same")
        vae_output_format = FORMAT_LABEL_TO_KEY.get(vae_format_label, "same")
        save_mode = SAVE_MODE_LABEL_TO_KEY.get(save_mode_label, "unet_only")
        device_choice = DEVICE_LABEL_TO_KEY.get(device_label, "auto")

        progress_cb = _make_progress_cb("bake")

        result = lora_bake.bake_lora_into_checkpoint(
            checkpoint_path,
            selected,
            output_path,
            output_format=output_format,
            clip_output_format=clip_output_format,
            vae_output_format=vae_output_format,
            save_mode=save_mode,
            device_choice=device_choice,
            progress_cb=progress_cb,
        )

        sd_models.list_models()
        gr.Info("LoRA(s) applied successfully!", duration=5)
        size_str = _format_size(result["output"])
        loras_desc = ", ".join(f"{a['name']} ({a['strength']})" + (f" [trigger: {a['activation_text']}]" if a["activation_text"] else "") for a in result["loras"])
        html = (
            f"<div style='margin-top: 10px; line-height: 1.6; font-size: 14px;'>"
            f"<b>Checkpoint saved to:</b> <code>{result['output']}</code><br>"
            f"<b>File size:</b> <span style='color: #10b981; font-weight: bold;'>{size_str}</span><br>"
            f"<b>LoRAs:</b> {loras_desc}<br>"
            f"<b>Format:</b> <code>{result['output_format']}</code> (Mode: <code>{save_mode}</code>)"
            f"</div>"
        )
        llm_adapter_hits = [a["name"] for a in result["loras"] if a["llm_adapter_warning"]]
        if llm_adapter_hits:
            gr.Warning(f"LoRA(s) with LLM adapter weights baked in: {', '.join(llm_adapter_hits)}. Anima's own training guidance says never to train these alongside a LoRA.", duration=10)
            html += f"<div style='color:orange; margin-top: 6px;'>Warning: {', '.join(llm_adapter_hits)} contains LLM adapter weights — not recommended by Anima's own training guidance.</div>"
        return gr.update(choices=sorted(sd_models.checkpoint_tiles())), html
    except lora_bake.BakeError as e:
        gr.Warning(str(e), duration=8)
        return gr.update(), f"<div class='error'>{e}</div>"
    except Exception as e:
        gr.Warning("Failed to apply LoRA(s).", duration=5)
        return gr.update(), _err_html(e)


# --- Checkpoint Merge (Anima-aware) -------------------------------------


def preview_metadata_handler(primary_name: str, secondary_name: str, tertiary_name: str):
    # Note: we resolve checkpoints via checkpoint_aliases (accepts any of
    # name/title/hash/etc), not sd_models.checkpoints_list (keyed only by
    # title, i.e. "name [hash]") -- the dropdowns here (like the native
    # Checkpoint Merger's) are populated from checkpoint_tiles(), which
    # returns plain .name values, so a checkpoints_list lookup would miss.
    metadata = {}
    for name in (primary_name, secondary_name, tertiary_name):
        info = sd_models.checkpoint_aliases.get(name) if name else None
        if info is not None:
            metadata[name] = info.metadata
    return gr.update(value=json.dumps(metadata, indent=4, ensure_ascii=False), visible=True)


INTERP_DESCRIPTIONS = {
    checkpoint_merge.INTERP_NO_INTERPOLATION: "Require 1 Model ; Mainly for format conversion",
    checkpoint_merge.INTERP_WEIGHTED_SUM: "Require 2 Model ; Result is calculated as A * (1 - M) + B * M",
    checkpoint_merge.INTERP_ADD_DIFFERENCE: "Require 3 Model ; Result is calculated as A + (B - C) * M",
}


# --- Merge recipes (save / load the whole tab as JSON) -------------------

RECIPE_VERSION = 1
MAX_LORAS = 10

# Field order is the contract between save and load. Adding a field at the end
# stays backwards compatible: load() falls back to the component's current
# value for anything a older recipe doesn't carry.
RECIPE_FIELDS = (
    "primary", "secondary", "tertiary",
    "interp", "multiplier", "anima_extend_ratio",
    "save_mode", "device", "output_name", "discard",
    "format", "clip_format", "vae_format",
    "save_metadata", "config_source", "add_merge_recipe", "bake_vae",
)


_MODEL_SLOT_LABELS = {"primary": "Model A", "secondary": "Model B", "tertiary": "Model C"}


def _recipes_dir() -> str:
    """Recipes live inside the extension, so everything Merge Studio owns sits
    in one folder instead of leaving a merge_studio_recipes/ directory in the
    Forge root."""
    root = os.path.join(_EXT_ROOT, "recipes")
    os.makedirs(root, exist_ok=True)
    _adopt_legacy_recipes(root)
    return root


def _adopt_legacy_recipes(root: str) -> None:
    """Copies recipes written to the old Forge-root location into the extension.

    Copies rather than moves: an install that has not been updated yet still
    reads the old folder, and losing a saved recipe to a path change is not a
    reasonable trade for tidiness. Runs once -- a recipe already adopted is
    never overwritten, so edits made here survive.
    """
    base = getattr(getattr(shared, "cmd_opts", None), "data_dir", None)
    if not base:
        return
    legacy = os.path.join(base, "merge_studio_recipes")
    if not os.path.isdir(legacy) or os.path.abspath(legacy) == os.path.abspath(root):
        return
    try:
        for name in os.listdir(legacy):
            if not name.endswith(".json"):
                continue
            target = os.path.join(root, name)
            if os.path.exists(target):
                continue
            shutil.copy2(os.path.join(legacy, name), target)
    except Exception as e:
        debug_print(f"Could not adopt recipes from {legacy}: {e}")


def _recipe_path(name: str) -> str:
    stem = os.path.basename(str(name or "").strip())
    if stem.lower().endswith(".json"):
        stem = stem[:-5]
    if not stem:
        raise ValueError("Give the recipe a name.")
    return os.path.join(_recipes_dir(), stem + ".json")


def list_recipes() -> list[str]:
    try:
        return sorted(f[:-5] for f in os.listdir(_recipes_dir()) if f.endswith(".json"))
    except Exception:
        return []


def save_recipe_handler(recipe_name: str, *values):
    """values: RECIPE_FIELDS in order, then MAX_LORAS dropdowns, then
    MAX_LORAS strengths, then the visible-row count."""
    try:
        scalars = values[: len(RECIPE_FIELDS)]
        rest = values[len(RECIPE_FIELDS):]
        dds, strengths, lora_count = rest[:MAX_LORAS], rest[MAX_LORAS:2 * MAX_LORAS], rest[2 * MAX_LORAS]

        recipe = {
            "_meta": {
                "extension": "sd-webui-merge-studio",
                "recipe_version": RECIPE_VERSION,
                "saved_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            },
            "settings": dict(zip(RECIPE_FIELDS, scalars)),
            "loras": [
                {"name": dd, "strength": float(st)}
                for dd, st in zip(dds, strengths)
                if dd and dd != NONE_LABEL
            ],
            "lora_slots": int(lora_count or 1),
        }
        path = _recipe_path(recipe_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recipe, f, indent=2, ensure_ascii=False)

        gr.Info(f"Recipe saved: {os.path.basename(path)}", duration=4)
        return (
            gr.update(choices=list_recipes(), value=os.path.basename(path)[:-5]),
            f"<div style='margin-top:6px;font-size:12px;color:#10b981;'>Saved to <code>{path}</code></div>",
        )
    except Exception as e:
        gr.Warning(str(e), duration=8)
        return gr.update(), _err_html(e)


def load_recipe_handler(recipe_name: str):
    """Returns updates for every merge-tab control, in the same order the
    outputs list is wired. Values naming a model or LoRA that isn't installed
    here are left alone and reported, so a recipe from another machine still
    restores everything else."""
    blank = (
        [gr.update() for _ in RECIPE_FIELDS]
        + [gr.update() for _ in range(2 * MAX_LORAS)]
        + [gr.update()]
        + [gr.update() for _ in range(MAX_LORAS)]
        + [gr.update(), gr.update()]
    )
    try:
        with open(_recipe_path(recipe_name), "r", encoding="utf-8") as f:
            recipe = json.load(f)
        settings = recipe.get("settings", {}) or {}
        loras = recipe.get("loras", []) or []

        checkpoints = set(sd_models.checkpoint_tiles())
        lora_names = set(_lora_choices())
        vae_names = set(_vae_choices())
        missing = []

        def pick(key, valid=None, label=None):
            if key not in settings:
                return gr.update()
            val = settings[key]
            if valid is not None and val and val not in valid:
                missing.append(f"{label or key}: {val}")
                return gr.update()
            return gr.update(value=val)

        scalar_updates = []
        for field in RECIPE_FIELDS:
            if field in _MODEL_SLOT_LABELS:
                scalar_updates.append(pick(field, checkpoints, _MODEL_SLOT_LABELS[field]))
            elif field == "bake_vae":
                scalar_updates.append(pick(field, vae_names, "Bake VAE"))
            else:
                scalar_updates.append(pick(field))

        dd_updates = [gr.update(value=NONE_LABEL) for _ in range(MAX_LORAS)]
        st_updates = [gr.update(value=1.0) for _ in range(MAX_LORAS)]
        for i, entry in enumerate(loras[:MAX_LORAS]):
            name = entry.get("name")
            if name and name not in lora_names:
                missing.append(f"LoRA: {name}")
                continue
            dd_updates[i] = gr.update(value=name)
            st_updates[i] = gr.update(value=float(entry.get("strength", 1.0)))

        slots = max(1, min(MAX_LORAS, int(recipe.get("lora_slots", 1) or 1), max(len(loras), 1)))
        row_updates = [gr.update(visible=(i < slots)) for i in range(MAX_LORAS)]

        method = INTERP_LABEL_TO_KEY.get(settings.get("interp"), settings.get("interp"))
        has_b = method != checkpoint_merge.INTERP_NO_INTERPOLATION
        has_c = method == checkpoint_merge.INTERP_ADD_DIFFERENCE

        if missing:
            gr.Warning("Not found on this install, left unchanged: " + "; ".join(missing), duration=10)
            note = (
                "<div style='margin-top:6px;font-size:12px;color:#f59e0b;'>Loaded, but not found here: "
                + "; ".join(missing)
                + "</div>"
            )
        else:
            gr.Info(f"Recipe loaded: {recipe_name}", duration=4)
            saved_at = (recipe.get("_meta") or {}).get("saved_at", "")
            note = f"<div style='margin-top:6px;font-size:12px;color:#10b981;'>Loaded <code>{recipe_name}</code>{' &middot; saved ' + saved_at if saved_at else ''}</div>"

        return scalar_updates + dd_updates + st_updates + [slots] + row_updates + [gr.update(visible=has_b), gr.update(visible=has_c)] + [note]
    except Exception as e:
        gr.Warning(str(e), duration=8)
        return blank + [_err_html(e)]


def merge_update_method(value: str):
    method = INTERP_LABEL_TO_KEY.get(value, value)
    has_b = method != checkpoint_merge.INTERP_NO_INTERPOLATION
    has_c = method == checkpoint_merge.INTERP_ADD_DIFFERENCE
    if has_c:
        config_choices = ["A", "B", "C"]
    elif has_b:
        config_choices = ["A", "B"]
    else:
        config_choices = ["A"]

    return [
        gr.update(visible=has_b),
        gr.update(visible=has_b),
        gr.update(visible=has_c),
        gr.update(info=INTERP_DESCRIPTIONS.get(method, "")),
        gr.update(choices=config_choices, value=config_choices),
    ]


def merge_handler(
    id_task,
    primary_name: str,
    secondary_name: str,
    tertiary_name: str,
    interp_label: str,
    multiplier: float,
    anima_extend_ratio: float = 0.0,
    save_mode_label: str = SAVE_MODE_CHOICES[0][0],
    device_label: str = DEVICE_CHOICES[0][0],
    output_filename: str = "",
    discard_regex: str = "",
    format_label: str = "same",
    clip_format_label: str = "same",
    vae_format_label: str = "same",
    save_metadata: bool = True,
    config_source: list[str] = None,
    add_merge_recipe: bool = True,
    bake_vae_label: str = ORIGINAL_VAE_LABEL,
    *lora_args,
):
    if not primary_name:
        gr.Warning("Select a Primary Model (A).")
        return gr.update(), "<div>Select a Primary Model (A).</div>"
    try:
        interp_method = INTERP_LABEL_TO_KEY.get(interp_label, checkpoint_merge.INTERP_NO_INTERPOLATION)
        output_format = FORMAT_LABEL_TO_KEY.get(format_label, "same")
        clip_output_format = FORMAT_LABEL_TO_KEY.get(clip_format_label, "same")
        vae_output_format = FORMAT_LABEL_TO_KEY.get(vae_format_label, "same")
        save_mode = SAVE_MODE_LABEL_TO_KEY.get(save_mode_label, "unet_only")
        device_choice = DEVICE_LABEL_TO_KEY.get(device_label, "auto")

        if bake_vae_label == ORIGINAL_VAE_LABEL:
            bake_vae = "original"
        elif bake_vae_label == NO_VAE_LABEL:
            bake_vae = "none"
        else:
            bake_vae = bake_vae_label

        loras_by_name = lora_bake.available_loras()
        selected_loras = []
        for i in range(0, len(lora_args), 2):
            if i + 1 < len(lora_args):
                name = lora_args[i]
                strength = lora_args[i + 1]
                if name and name != NONE_LABEL:
                    if name not in loras_by_name:
                        raise ValueError(f"LoRA not found: {name}")
                    try:
                        str_val = float(strength)
                    except Exception:
                        str_val = 1.0
                    selected_loras.append((loras_by_name[name], str_val))

        if not output_filename:
            raise ValueError("Enter an output filename.")
        if not output_filename.endswith(".safetensors"):
            output_filename += ".safetensors"
        output_path = os.path.join(sd_models.model_path, output_filename)

        progress_cb = _make_progress_cb("merge")

        result = checkpoint_merge.merge_checkpoints(
            primary_name,
            secondary_name or None,
            tertiary_name or None,
            interp_method,
            multiplier,
            output_path,
            output_format=output_format,
            clip_output_format=clip_output_format,
            vae_output_format=vae_output_format,
            discard_regex=discard_regex or "",
            save_metadata=save_metadata,
            config_source=tuple(config_source or ()),
            add_merge_recipe=add_merge_recipe,
            save_mode=save_mode,
            loras=selected_loras if selected_loras else None,
            device_choice=device_choice,
            bake_vae=bake_vae,
            anima_extend_ratio=float(anima_extend_ratio or 0.0),
            progress_cb=progress_cb,
        )

        sd_models.list_models()
        gr.Info("Merge completed successfully!", duration=5)
        skipped_total = sum(len(v) for v in result["skipped"].values())
        size_str = _format_size(result["output"])
        loras_desc = ", ".join(f"{a['name']} ({a['strength']})" for a in result.get("loras", [])) if result.get("loras") else ""
        baked_vae_desc = f"<b>Baked VAE:</b> <code>{result['baked_vae']}</code><br>" if result.get("baked_vae") else ""
        remap = result.get("anima_remap")
        remap_desc = (
            f"<b>Anima remap:</b> {remap['from_blocks']}-block &rarr; {remap['to_blocks']}-block "
            f"({remap['frozen_blocks']} shared blocks merged, {remap['inserted_blocks']} inserted blocks "
            + (
                f"blended at extend_ratio {remap['extend_ratio']})<br>"
                if remap.get("extend_ratio")
                else "kept from A)<br>"
            )
            if remap
            else ""
        )
        html = (
            f"<div style='margin-top: 10px; line-height: 1.6; font-size: 14px;'>"
            f"<b>Checkpoint saved to:</b> <code>{result['output']}</code><br>"
            f"<b>File size:</b> <span style='color: #10b981; font-weight: bold;'>{size_str}</span><br>"
            f"<b>Format:</b> <code>{result['output_format']}</code> (Mode: <code>{save_mode}</code>)<br>"
            f"<b>Merged layers:</b> UNet: {result['merged']['unet']}, CLIP: {result['merged']['clip']}, VAE: {result['merged']['vae']}<br>"
            + remap_desc
            + baked_vae_desc
            + (f"<b>Baked LoRAs:</b> {loras_desc}<br>" if loras_desc else "")
            + f"</div>"
        )
        llm_adapter_hits = [a["name"] for a in result.get("loras", []) if a.get("llm_adapter_warning")]
        if llm_adapter_hits:
            gr.Warning(f"LoRA(s) with LLM adapter weights baked in: {', '.join(llm_adapter_hits)}. Anima's own training guidance says never to train these alongside a LoRA.", duration=10)
            html += f"<div style='color:orange; margin-top: 6px;'>Warning: {', '.join(llm_adapter_hits)} contains LLM adapter weights — not recommended by Anima's own training guidance.</div>"
        if skipped_total:
            if remap and not remap.get("extend_ratio"):
                html += (
                    f"<div style='color:#6b7280; margin-top: 6px;'>{skipped_total} layer(s) kept from Model A "
                    f"(the {remap['inserted_blocks']} blocks Anima's expansion inserted have no counterpart in a "
                    f"{remap['from_blocks']}-block model). This is expected.</div>"
                )
            else:
                html += f"<div style='color:orange; margin-top: 6px;'>Warning: {skipped_total} layer(s) skipped (missing or incompatible between models).</div>"
        return gr.update(choices=sorted(sd_models.checkpoint_tiles())), html
    except checkpoint_merge.MergeError as e:
        gr.Warning(str(e), duration=8)
        return gr.update(), f"<div class='error'>{e}</div>"
    except Exception as e:
        gr.Warning("Failed to merge checkpoints.", duration=5)
        return gr.update(), _err_html(e)


# --- Quantize Checkpoint --------------------------------------------------


def quantize_handler(id_task, checkpoint_name: str, format_label: str, save_mode_label: str, output_filename: str):
    if not checkpoint_name:
        gr.Warning("Select a checkpoint.")
        return gr.update(), "<div>Select a checkpoint.</div>"
    try:
        checkpoint_path = _checkpoint_path(checkpoint_name)
        output_format = FORMAT_LABEL_TO_KEY.get(format_label, "same")
        save_mode = SAVE_MODE_LABEL_TO_KEY.get(save_mode_label, "unet_only")

        if not output_filename:
            raise ValueError("Enter an output filename.")
        if not output_filename.endswith(".safetensors"):
            output_filename += ".safetensors"
        output_path = os.path.join(sd_models.model_path, output_filename)

        progress_cb = _make_progress_cb("quantize")

        result = checkpoint_quantize.quantize_checkpoint(checkpoint_path, output_path, output_format, save_mode=save_mode, progress_cb=progress_cb)

        sd_models.list_models()
        gr.Info("Checkpoint converted successfully!", duration=5)
        size_str = _format_size(result["output"])
        html = (
            f"<div style='margin-top: 10px; line-height: 1.6; font-size: 14px;'>"
            f"<b>Checkpoint saved to:</b> <code>{result['output']}</code><br>"
            f"<b>File size:</b> <span style='color: #10b981; font-weight: bold;'>{size_str}</span><br>"
            f"<b>Converted layers:</b> {result['converted_layers']}<br>"
            f"<b>Format:</b> <code>{result['output_format']}</code> (Mode: <code>{save_mode}</code>)"
            f"</div>"
        )
        return gr.update(choices=sorted(sd_models.checkpoint_tiles())), html
    except checkpoint_quantize.QuantizeError as e:
        gr.Warning(str(e), duration=8)
        return gr.update(), f"<div class='error'>{e}</div>"
    except Exception as e:
        gr.Warning("Failed to convert checkpoint.", duration=5)
        return gr.update(), _err_html(e)


# --- Tab layout -----------------------------------------------------------


def create_merge_studio_tab():
    with gr.Blocks(analytics_enabled=False) as merge_studio_interface:
        gr.Markdown("## Merge Studio")

        # Shared invisible input: JS overwrites its value with a generated
        # task(...) id right before submit, which modules/call_queue.py's
        # wrap_gradio_gpu_call() picks up to register/track progress -- the
        # same mechanism the native Checkpoint Merger uses.
        dummy_component = gr.Textbox(visible=False)

        with gr.Tabs():
            with gr.Tab("Checkpoint Merge & Studio"):
                gr.Markdown(
                    "The all-in-one studio to merge checkpoints (Weighted Sum / Add Difference), convert or quantize "
                    "precision (No Interpolation), and bake LoRAs directly into model weights with full Anima and "
                    "quantization support."
                )
                gr.Markdown(COMPATIBLE_MODELS_NOTE)
                gr.Markdown("All models (A/B/C) must be checked individually — mixing a compatible A with an incompatible B/C will be rejected with a clear error.")
                with gr.Row():
                    with gr.Column():
                        merge_primary = gr.Dropdown(label="Primary Model (A)", choices=sorted(sd_models.checkpoint_tiles()))
                        primary_badge = gr.HTML("")
                    with gr.Column() as secondary_col:
                        merge_secondary = gr.Dropdown(label="Secondary Model (B)", choices=sorted(sd_models.checkpoint_tiles()))
                        secondary_badge = gr.HTML("")
                    with gr.Column(visible=False) as tertiary_col:
                        merge_tertiary = gr.Dropdown(label="Tertiary Model (C)", choices=sorted(sd_models.checkpoint_tiles()))
                        tertiary_badge = gr.HTML("")
                    def refresh_models_and_cache():
                        _checkpoint_info_cache.clear()
                        sd_models.list_models()

                    create_refresh_button(
                        [merge_primary, merge_secondary, merge_tertiary],
                        refresh_models_and_cache,
                        lambda: {"choices": sorted(sd_models.checkpoint_tiles())},
                        "merge_studio_refresh_models",
                    )

                merge_primary.change(
                    fn=primary_badge_handler,
                    inputs=[merge_primary, merge_secondary, merge_tertiary],
                    outputs=[primary_badge, secondary_badge, tertiary_badge],
                    show_progress=False,
                    queue=False,
                )
                merge_secondary.change(
                    fn=secondary_badge_handler,
                    inputs=[merge_primary, merge_secondary],
                    outputs=[secondary_badge],
                    show_progress=False,
                    queue=False,
                )
                merge_tertiary.change(
                    fn=tertiary_badge_handler,
                    inputs=[merge_primary, merge_tertiary],
                    outputs=[tertiary_badge],
                    show_progress=False,
                    queue=False,
                )

                with gr.Row():
                    merge_interp = gr.Radio(
                        choices=[label for label, _ in INTERP_CHOICES],
                        value=INTERP_CHOICES[1][0],
                        label="Interpolation Method",
                        info=INTERP_DESCRIPTIONS[checkpoint_merge.INTERP_WEIGHTED_SUM],
                    )
                    merge_multiplier = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Multiplier (M)")

                anima_extend_ratio = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.05,
                    label="Anima cross-generation: inserted-block blend (extend_ratio)",
                    info=(
                        "Only applies when merging two different Anima generations (28 / 40 / 52 blocks). "
                        "The newer generation's extra blocks have no counterpart in the older model, so at 0.0 they keep "
                        "Model A's weights. Above 0.0 they also blend in the block they were originally copied from. "
                        "If you plan to use LoRAs built for the OLDER generation, match this to the Multiplier: Forge remaps "
                        "such LoRAs onto the inserted blocks too, so an unblended base leaves those blocks reacting to a "
                        "LoRA trained against weights they don't have. Experimental — the inserted blocks diverged in training."
                    ),
                )

                with gr.Accordion("Bake LoRA(s) into Checkpoint (Optional)", open=False):
                    gr.Markdown("Optionally apply one or multiple LoRAs (e.g. Turbo LoRA, Style LoRAs) directly into the checkpoint weights.")
                    merge_lora_rows = []
                    merge_lora_row_layouts = []
                    merge_lora_del_btns = []
                    for i in range(1, MAX_LORAS + 1):
                        with gr.Row(visible=(i == 1)) as lora_row_layout:
                            merge_lora_dd = gr.Dropdown(label=f"LoRA {i}", choices=_lora_choices(), value=NONE_LABEL, scale=3)
                            merge_strength = gr.Slider(
                                label="Strength",
                                minimum=0.0,
                                maximum=2.0,
                                value=1.0,
                                step=0.05,
                                scale=2,
                                info="For Anima checkpoints, consider starting around 0.6-0.8 (e.g. 0.6 for Turbo)." if i == 1 else None,
                            )
                            create_refresh_button([merge_lora_dd], lambda: None, lambda: {"choices": _lora_choices()}, f"merge_studio_lora_refresh_{i}")
                            del_btn = gr.Button("X", elem_classes=["tool", "merge-studio-remove-lora-btn"], variant="stop")
                            merge_lora_rows.append((merge_lora_dd, merge_strength))
                            merge_lora_row_layouts.append(lora_row_layout)
                            merge_lora_del_btns.append(del_btn)

                    with gr.Row():
                        add_lora_btn = gr.Button("Add LoRA", variant="secondary")
                        clear_loras_btn = gr.Button("Clear All LoRAs", variant="secondary")

                    lora_count_state = gr.State(value=1)

                    def add_lora_slot(count):
                        new_count = min(count + 1, MAX_LORAS)
                        return [new_count] + [gr.update(visible=(i < new_count)) for i in range(MAX_LORAS)]

                    def clear_all_slots():
                        row_updates = [gr.update(visible=(i == 0)) for i in range(MAX_LORAS)]
                        val_updates = [gr.update(value=NONE_LABEL) for _ in range(MAX_LORAS)]
                        str_updates = [gr.update(value=1.0) for _ in range(MAX_LORAS)]
                        return [1] + row_updates + val_updates + str_updates

                    add_lora_btn.click(
                        fn=add_lora_slot,
                        inputs=[lora_count_state],
                        outputs=[lora_count_state] + merge_lora_row_layouts,
                        queue=False,
                    )
                    clear_loras_btn.click(
                        fn=clear_all_slots,
                        inputs=[],
                        outputs=[lora_count_state] + merge_lora_row_layouts + [dd for dd, _ in merge_lora_rows] + [st for _, st in merge_lora_rows],
                        queue=False,
                    )

                    def make_remove_lora_fn(k: int):
                        def remove_lora_at(count, *vals):
                            dds = list(vals[:MAX_LORAS])
                            sts = list(vals[MAX_LORAS:])
                            if count <= 1:
                                dds[0] = NONE_LABEL
                                sts[0] = 1.0
                                new_count = 1
                            else:
                                dds.pop(k)
                                sts.pop(k)
                                dds.append(NONE_LABEL)
                                sts.append(1.0)
                                new_count = max(1, count - 1)

                            row_updates = [gr.update(visible=(i < new_count)) for i in range(MAX_LORAS)]
                            dd_updates = [gr.update(value=dds[i]) for i in range(MAX_LORAS)]
                            st_updates = [gr.update(value=sts[i]) for i in range(MAX_LORAS)]
                            return [new_count] + row_updates + dd_updates + st_updates

                        return remove_lora_at

                    all_lora_inputs = [lora_count_state] + [dd for dd, _ in merge_lora_rows] + [st for _, st in merge_lora_rows]
                    all_lora_outputs = [lora_count_state] + merge_lora_row_layouts + [dd for dd, _ in merge_lora_rows] + [st for _, st in merge_lora_rows]

                    for idx, del_btn in enumerate(merge_lora_del_btns):
                        del_btn.click(
                            fn=make_remove_lora_fn(idx),
                            inputs=all_lora_inputs,
                            outputs=all_lora_outputs,
                            queue=False,
                        )

                merge_save_mode = gr.Radio(
                    choices=[label for label, _ in SAVE_MODE_CHOICES],
                    value=SAVE_MODE_CHOICES[0][0],
                    label="Save mode",
                    info="UNet Only produces a much smaller file (matches native 'Save UNet Only') -- use it if you already load this checkpoint's VAE/text encoder as separate files.",
                )

                with gr.Row(equal_height=True):
                    merge_device = gr.Dropdown(
                        label="Device",
                        choices=[label for label, _ in DEVICE_CHOICES],
                        value=DEVICE_CHOICES[0][0],
                    )
                    merge_output_name = gr.Textbox(label="Filename to save", placeholder="e.g. merged_anima.safetensors")
                    merge_discard = gr.Textbox(label="Layers to discard (regex)", placeholder="")

                with gr.Row(equal_height=True):
                    merge_format = _format_dropdown("Output format (diffusion model)")
                    merge_clip_format = _format_dropdown("Text encoder format (only for Full mode)")
                    merge_vae_format = _format_dropdown("VAE format (only for Full mode)")

                with gr.Row(equal_height=True):
                    merge_bake_vae = gr.Dropdown(
                        label="Bake VAE",
                        choices=_vae_choices(),
                        value=ORIGINAL_VAE_LABEL,
                        info="Choose a VAE to bake into the checkpoint. 'Original': keep source VAE. 'None': strip VAE.",
                    )
                    create_refresh_button(
                        [merge_bake_vae],
                        lambda: None,
                        lambda: {"choices": _vae_choices()},
                        "merge_studio_refresh_vae",
                    )

                merge_clip_format.visible = False
                merge_vae_format.visible = False

                merge_save_mode.change(
                    fn=lambda mode: [
                        gr.update(visible=(mode != SAVE_MODE_CHOICES[0][0])),
                        gr.update(visible=(mode != SAVE_MODE_CHOICES[0][0])),
                    ],
                    inputs=[merge_save_mode],
                    outputs=[merge_clip_format, merge_vae_format],
                    show_progress=False,
                    queue=False,
                )

                with InputAccordion(True, label="Save Metadata") as merge_save_metadata:
                    with FormRow():
                        merge_config_source = gr.CheckboxGroup(choices=["A", "B"], value=["A", "B"], label="Copy Metadata from")
                        merge_add_recipe = gr.Checkbox(True, label="Include Merge Recipe")

                    merge_metadata_preview = gr.TextArea(value=None, label="Metadata in JSON Format", visible=False)
                    merge_preview_btn = gr.Button("Preview Metadata from Models")

                    merge_preview_btn.click(fn=preview_metadata_handler, inputs=[merge_primary, merge_secondary, merge_tertiary], outputs=[merge_metadata_preview])

                merge_interp.change(
                    fn=merge_update_method,
                    inputs=[merge_interp],
                    outputs=[merge_multiplier, secondary_col, tertiary_col, merge_interp, merge_config_source],
                    show_progress=False,
                    queue=False,
                )

                with gr.Accordion("Save / Load Recipe", open=False):
                    gr.Markdown(
                        "Store every setting on this tab as a JSON file so a bake can be repeated or tweaked later. "
                        "Loading a recipe that names a model or LoRA you don't have leaves that field untouched and tells you which."
                    )
                    with gr.Row():
                        recipe_dropdown = gr.Dropdown(label="Saved recipes", choices=list_recipes(), value=None, scale=3)
                        create_refresh_button([recipe_dropdown], lambda: None, lambda: {"choices": list_recipes()}, "merge_studio_recipe_refresh")
                        recipe_load_btn = gr.Button("Load", variant="secondary", scale=1)
                    with gr.Row():
                        recipe_name = gr.Textbox(label="Save as", placeholder="e.g. anima-turbo-bake", scale=3)
                        recipe_save_btn = gr.Button("Save", variant="secondary", scale=1)
                    recipe_status = gr.HTML("")

                recipe_scalars = [
                    merge_primary, merge_secondary, merge_tertiary,
                    merge_interp, merge_multiplier, anima_extend_ratio,
                    merge_save_mode, merge_device, merge_output_name, merge_discard,
                    merge_format, merge_clip_format, merge_vae_format,
                    merge_save_metadata, merge_config_source, merge_add_recipe, merge_bake_vae,
                ]
                recipe_dds = [dd for dd, _ in merge_lora_rows]
                recipe_sts = [st for _, st in merge_lora_rows]

                recipe_save_btn.click(
                    fn=save_recipe_handler,
                    inputs=[recipe_name] + recipe_scalars + recipe_dds + recipe_sts + [lora_count_state],
                    outputs=[recipe_dropdown, recipe_status],
                    queue=False,
                )
                recipe_load_btn.click(
                    fn=load_recipe_handler,
                    inputs=[recipe_dropdown],
                    outputs=(
                        recipe_scalars + recipe_dds + recipe_sts
                        + [lora_count_state] + merge_lora_row_layouts
                        + [secondary_col, tertiary_col, recipe_status]
                    ),
                    queue=False,
                ).then(
                    fn=primary_badge_handler,
                    inputs=[merge_primary, merge_secondary, merge_tertiary],
                    outputs=[primary_badge, secondary_badge, tertiary_badge],
                    show_progress=False,
                    queue=False,
                )

                merge_btn = gr.Button("Merge / Process", variant="primary")
                with gr.Group(elem_id="checkpoint_doctor_merge_panel"):
                    merge_html = gr.HTML("")

                merge_btn.click(fn=lambda: "", outputs=[merge_html], queue=False, show_progress=False).then(
                    fn=call_queue.wrap_gradio_gpu_call(merge_handler, extra_outputs=lambda: [gr.skip()]),
                    js="checkpointDoctorMergeProgress",
                    inputs=[
                        dummy_component,
                        merge_primary,
                        merge_secondary,
                        merge_tertiary,
                        merge_interp,
                        merge_multiplier,
                        anima_extend_ratio,
                        merge_save_mode,
                        merge_device,
                        merge_output_name,
                        merge_discard,
                        merge_format,
                        merge_clip_format,
                        merge_vae_format,
                        merge_save_metadata,
                        merge_config_source,
                        merge_add_recipe,
                        merge_bake_vae,
                        *[c for pair in merge_lora_rows for c in pair],
                    ],
                    outputs=[merge_primary, merge_html],
                    show_progress=False,
                )

            with gr.Tab("Quant Format Doctor"):
                gr.Markdown(
                    "Some quantized checkpoints (e.g. community INT8 builds) have a metadata bug that causes "
                    "`ValueError: Unknown quantization format for layer ...` when loading. This tool diagnoses and fixes it."
                )
                with gr.Row():
                    doctor_checkpoint = gr.Dropdown(label="Checkpoint", choices=sorted(sd_models.checkpoint_tiles()))
                    create_refresh_button([doctor_checkpoint], sd_models.list_models, lambda: {"choices": sorted(sd_models.checkpoint_tiles())}, "merge_studio_refresh_doctor")

                doctor_diagnose_btn = gr.Button("Diagnose")
                doctor_diag_html = gr.HTML("")

                with gr.Row():
                    doctor_mode = gr.Radio(choices=[("Save as new file", "new"), ("Fix original file", "inplace")], value="new", label="Fix mode")
                    doctor_new_name = gr.Textbox(label="New filename", placeholder="e.g. model_fixed.safetensors")

                doctor_fix_btn = gr.Button("Fix", variant="primary")
                with gr.Group(elem_id="checkpoint_doctor_fix_panel"):
                    doctor_fix_html = gr.HTML("")

                doctor_diagnose_btn.click(fn=diagnose_handler, inputs=[doctor_checkpoint], outputs=[doctor_diag_html])
                doctor_fix_btn.click(fn=lambda: "", outputs=[doctor_fix_html], queue=False, show_progress=False).then(
                    fn=call_queue.wrap_gradio_gpu_call(repair_handler, extra_outputs=lambda: [gr.skip()]),
                    js="checkpointDoctorFixProgress",
                    inputs=[dummy_component, doctor_checkpoint, doctor_mode, doctor_new_name],
                    outputs=[doctor_checkpoint, doctor_fix_html],
                    show_progress=False,
                )

            with gr.Tab("Model Recipe & Inspector"):
                gr.Markdown(
                    "Inspect any model Forge can load, without reading weights into RAM/VRAM. "
                    "**Checkpoints:** components (UNet/DiT, Text Encoder, VAE), architecture, precision, and merge recipe provenance. "
                    "**LoRAs:** trigger word, which Anima generation it targets, rank, coverage, and whether it changes the LLM adapter. "
                    "**Text encoders / VAEs:** which one it actually is, so a checkpoint isn't paired with the wrong encoder."
                )
                with gr.Row():
                    inspector_checkpoint = gr.Dropdown(
                        label="Model",
                        choices=_inspect_choices(),
                        info="Checkpoints, LoRAs, text encoders and VAEs. Type to filter — e.g. 'lora'.",
                    )

                    def refresh_inspector_choices():
                        sd_models.list_models()
                        try:
                            from modules_forge import main_entry

                            main_entry.refresh_models()
                        except Exception:
                            pass

                    create_refresh_button(
                        [inspector_checkpoint],
                        refresh_inspector_choices,
                        lambda: {"choices": _inspect_choices()},
                        "merge_studio_refresh_inspector",
                    )
                inspector_btn = gr.Button("Inspect", variant="primary")
                inspector_html = gr.HTML(
                    "<div style='padding: 20px; color: #9ca3af;'>Select a model above to inspect it.</div>"
                )

                inspector_btn.click(
                    fn=run_inspection_handler,
                    inputs=[inspector_checkpoint],
                    outputs=[inspector_html],
                )
                inspector_checkpoint.change(
                    fn=run_inspection_handler,
                    inputs=[inspector_checkpoint],
                    outputs=[inspector_html],
                )

        for comp in (
            doctor_checkpoint,
            doctor_mode,
            doctor_new_name,
            merge_primary,
            merge_secondary,
            merge_tertiary,
            merge_interp,
            merge_multiplier,
            anima_extend_ratio,
            merge_save_mode,
            merge_device,
            merge_output_name,
            merge_discard,
            merge_format,
            merge_clip_format,
            merge_vae_format,
            merge_bake_vae,
            inspector_checkpoint,
            merge_save_metadata,
            merge_config_source,
            merge_add_recipe,
            merge_metadata_preview,
            *[c for pair in merge_lora_rows for c in pair],
        ):
            comp.do_not_save_to_config = True

    return [(merge_studio_interface, "Merge Studio", "merge_studio")]


script_callbacks.on_ui_tabs(create_merge_studio_tab)
