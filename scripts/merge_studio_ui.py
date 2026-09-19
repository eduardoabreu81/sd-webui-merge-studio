import datetime
import html
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
import elemental_weights  # noqa: E402
import merge_modes  # noqa: E402
import checkpoint_quantize  # noqa: E402
import lora_bake  # noqa: E402
import lora_extract  # noqa: E402
import quant_repair  # noqa: E402
import component_recipes
import component_ui
from checkpoint_inspector import (
    available_vaes,
    format_badges_html,
    format_recipe_dashboard_html,
    get_model_family,
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

# The modes, their labels and what each one needs are declared once, in
# `merge_modes.MERGE_MODES`. The picker, the field visibility and the
# recipe all read from there; none of them keeps its own list.
INTERP_CHOICES = [(mode.label, mode.key) for mode in merge_modes.MERGE_MODES]
INTERP_LABEL_TO_KEY = dict(INTERP_CHOICES)
INTERP_KEY_TO_LABEL = {key: label for label, key in INTERP_CHOICES}


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
    ("AIO (UNet + text encoder + VAE, self-contained, bigger)", "full"),
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


# --- Merge recipes (save / load the whole tab as JSON) -------------------

RECIPE_VERSION = component_recipes.RECIPE_VERSION
MAX_LORAS = 10

# Flux 1 is the widest architecture Forge declares: CLIP-L, T5XXL and a VAE.
# Rows are pre-allocated and toggled, the same way the LoRA rows are; how many
# are shown still comes from the architecture, never from this number.
MAX_COMPONENT_ROWS = 3

# Field order is the contract between save and load. Adding a field at the end
# stays backwards compatible: load() falls back to the component's current
# value for anything a older recipe doesn't carry.
RECIPE_FIELDS = (
    "primary", "secondary", "tertiary",
    "interp", "multiplier", "beta", "seed", "block_weights", "anima_extend_ratio",
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
        # secondary_col, tertiary_col, merge_beta, merge_seed,
        # merge_mode_html, block_weights_accordion, merge_block_profile. The
        # status HTML is appended by the caller, so it is not counted here.
        + [gr.update() for _ in range(7)]
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
        try:
            mode = merge_modes.merge_mode(method)
        except Exception:
            # A recipe naming a mode this build does not have. The rest of it
            # still loads; the picker keeps whatever is selected.
            mode = merge_modes.MERGE_MODES_BY_KEY[merge_modes.INTERP_WEIGHTED_SUM]

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

        return (
            scalar_updates + dd_updates + st_updates + [slots] + row_updates
            + [
                gr.update(visible=mode.needs_b),
                gr.update(visible=mode.needs_c),
                gr.update(visible=mode.needs_beta),
                gr.update(visible=mode.needs_seed),
                gr.update(value=merge_modes.merge_mode_panel(mode.key)),
                block_weights_visibility(settings.get("primary", "")),
                gr.update(
                    value=block_weights_preview(
                        settings.get("primary", ""),
                        settings.get("block_weights", ""),
                        settings.get("multiplier", 0.0),
                    )
                ),
            ]
            + [note]
        )
    except Exception as e:
        gr.Warning(str(e), duration=8)
        return blank + [_err_html(e)]


def merge_update_method(value: str):
    """The form follows the mode. Every answer comes from MERGE_MODES."""
    method = INTERP_LABEL_TO_KEY.get(value, value)
    try:
        mode = merge_modes.merge_mode(method)
    except Exception:
        mode = merge_modes.MERGE_MODES_BY_KEY[merge_modes.INTERP_WEIGHTED_SUM]

    config_choices = ["A"] + (["B"] if mode.needs_b else []) + (["C"] if mode.needs_c else [])

    return [
        gr.update(visible=mode.needs_b),                     # multiplier
        gr.update(visible=mode.needs_b),                     # Model B column
        gr.update(visible=mode.needs_c),                     # Model C column
        gr.update(visible=mode.needs_beta,
                  label=mode.beta_label or "Beta (β)"),      # beta
        gr.update(visible=mode.needs_seed),                  # seed
        gr.update(value=merge_modes.merge_mode_panel(method)),           # the panel
        gr.update(choices=config_choices, value=config_choices),
    ]


def _anima_block_count(checkpoint_name: str) -> int | None:
    """How many blocks Model A has, or None when it is not an Anima at all.

    The number comes from the checkpoint and never from a control. Anima ships
    as 28, 40 or 52; a free-running value would let someone write a rule for
    layer 40 of a 28-block model and watch a profile that cannot happen.
    """
    info = _get_cached_checkpoint_info(checkpoint_name)
    if not info or info.get("error"):
        return None
    if get_model_family(info.get("architecture", "")) != "anima":
        return None
    blocks = info.get("block_count")
    return blocks if isinstance(blocks, int) and blocks > 0 else None


def block_weights_visibility(primary_name: str):
    """Per-block weights are shown for Anima and absent for everything else.

    Not disabled with an explanation, not greyed out -- absent. `L00-L27`
    assumes one numbered stack of blocks, which SDXL's three sections and
    Flux's two parallel series are not, and offering a control that cannot
    work is worse than not offering it.
    """
    return gr.update(visible=_anima_block_count(primary_name) is not None)


def block_weights_preview(primary_name: str, text: str, multiplier: float):
    """The per-layer strip under the editor, redrawn as the rules are typed.

    Ten overlapping rules over a base of 0.0 is not something anyone can hold
    in their head, and it is trivial to look at. Same drawing the Inspector
    gives a finished merge, so what is on screen while typing is what the
    resulting checkpoint will report.
    """
    blocks = _anima_block_count(primary_name)
    spec = elemental_weights.spec_with_base(
        elemental_weights.parse_weight_spec(text or ""), float(multiplier or 0.0)
    )
    problems = elemental_weights.validate_spec(spec, blocks)
    return elemental_weights.format_weight_editor_html(spec, blocks, problems)


def import_block_weights(checkpoint_name: str):
    """Lift the per-block rules out of a checkpoint that already has them.

    Nobody writes ten rules from scratch; they start from something that
    worked. Reads the same fields the Inspector reads, so anything the
    Inspector can draw is something this can import.
    """
    info = _get_cached_checkpoint_info(checkpoint_name)
    recipe = (info or {}).get("recipe") or {}
    for field in ("block_weights", "alpha_raw", "multiplier_raw"):
        raw = recipe.get(field)
        if raw and elemental_weights.parse_weight_spec(str(raw)).rules:
            gr.Info(f"Imported per-block rules from {checkpoint_name}", duration=4)
            return gr.update(value=str(raw).strip())
    gr.Warning(f"{checkpoint_name or 'That checkpoint'} records no per-block rules.")
    return gr.update()


#: Shown in place of a filename when the checkpoint's own component is kept.
KEEP_EMBEDDED_LABEL = "Keep what is in the file"


def _inspect_installed_modules() -> dict:
    """Every installed text encoder and VAE, already classified.

    Keyed by the display name Forge shows, so a row's choices can be turned
    back into a path with the same lookup the rest of this file uses.
    """
    infos = {}
    for name in _module_choices():
        try:
            infos[name] = aux_inspector.inspect_module(_inspect_path(f"[module] {name}"))
        except Exception:
            continue
    return {name: info for name, info in infos.items() if "error" not in info}


def _component_row_states(primary_name: str, save_mode_label: str):
    """The rows this checkpoint needs, from its header alone.

    No engine is loaded here -- the form has to appear without waiting on a
    multi-gigabyte read. Whatever this guesses is replaced by the capability
    profile during the merge's preflight.
    """
    save_mode = SAVE_MODE_LABEL_TO_KEY.get(save_mode_label, "unet_only")
    if not primary_name or save_mode != "full":
        return ()
    try:
        info = inspect_checkpoint(_checkpoint_path(primary_name))
    except Exception:
        return ()
    if "error" in info:
        return ()
    return component_ui.build_component_rows(info, save_mode, _inspect_installed_modules())


def _refresh_component_rows(primary_name: str, save_mode_label: str):
    """Gradio updates for the pre-allocated component rows."""
    rows = _component_row_states(primary_name, save_mode_label)
    updates = []
    for i in range(MAX_COMPONENT_ROWS):
        row = rows[i] if i < len(rows) else None
        if row is None:
            updates += [gr.update(visible=False), gr.update(value=""),
                        gr.update(choices=[], value=None),
                        gr.update(choices=[c[0] for c in component_ui.COMPONENT_FORMAT_CHOICES],
                                  value=component_ui.COMPONENT_FORMAT_CHOICES[0][0]),
                        gr.update(value="")]
            continue
        choices = list(row.choices)
        if row.keep_embedded:
            choices.insert(0, KEEP_EMBEDDED_LABEL)
        note = row.status or row.warning
        updates += [
            gr.update(visible=True),
            gr.update(value=row.slot_id),
            gr.update(label=row.label, choices=choices, value=None),
            gr.update(choices=[c[0] for c in row.format_choices], value=row.format_choices[0][0]),
            gr.update(value=f"<div style='color:#f59e0b;font-size:0.85em;'>{note}</div>" if note else ""),
        ]
    return updates


def _parse_component_args(component_args) -> list[dict]:
    """The pre-allocated rows' values as component selections.

    An empty row contributes nothing, so the plan reports it as missing and the
    interface can name it rather than guessing something in.
    """
    rows = []
    for i in range(MAX_COMPONENT_ROWS):
        slot_id, value, fmt_label = component_args[i * 3: i * 3 + 3]
        if not slot_id or not value:
            continue
        if value == KEEP_EMBEDDED_LABEL:
            resolved = component_ui.KEEP_EMBEDDED
        else:
            try:
                resolved = _inspect_path(f"[module] {value}")
            except Exception:
                continue
        rows.append({
            "slot_id": slot_id,
            "value": resolved,
            "format": FORMAT_LABEL_TO_KEY.get(fmt_label, "same"),
        })
    return component_ui.parse_component_rows(rows)


def merge_handler(
    id_task,
    primary_name: str,
    secondary_name: str,
    tertiary_name: str,
    interp_label: str,
    multiplier: float,
    beta: float = 0.0,
    seed: int = 0,
    block_weights: str = "",
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
    *component_and_lora_args,
):
    # One variadic, split at a known boundary: the component rows come first
    # in the inputs list, the LoRA rows after. Two variadics is not a thing
    # Gradio can express.
    _split = MAX_COMPONENT_ROWS * 3
    component_args = component_and_lora_args[:_split]
    lora_args = component_and_lora_args[_split:]
    if not primary_name:
        gr.Warning("Select a Primary Model (A).")
        return gr.update(), "<div>Select a Primary Model (A).</div>"
    try:
        component_selections = _parse_component_args(component_args)
        declared = tuple(
            component_args[i * 3]
            for i in range(MAX_COMPONENT_ROWS)
            if component_args[i * 3]
        )
        filled = {s["slot_id"] for s in component_selections}
        missing = tuple(slot for slot in declared if slot not in filled)
        if missing:
            # Blocked here rather than after several gigabytes of merging.
            message = component_ui.missing_slot_message(missing, all_slots=declared)
            gr.Warning(message)
            return gr.update(), f"<div>{message}</div>"

        interp_method = INTERP_LABEL_TO_KEY.get(interp_label, merge_modes.INTERP_NO_INTERPOLATION)
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
            component_selections=component_selections,
            progress_cb=progress_cb,
            beta=float(beta or 0.0),
            block_weights=block_weights or "",
            seed=int(seed or 0),
        )

        sd_models.list_models()

        # An AIO is announced as finished only once it has reopened on its own.
        # Saying "completed" for a file that still needs external modules would
        # be the exact claim this feature exists to stop making.
        modular = result.get("modular_full")
        validated = result.get("validated", True)
        if not modular:
            gr.Info("Merge completed successfully!", duration=5)
        elif validated:
            gr.Info("AIO saved and verified: it reopens on its own.", duration=6)
        else:
            gr.Warning(
                "Saved, but NOT validated as an AIO: it did not reopen without "
                "external modules. The file was kept -- see the details below.",
                duration=12,
            )

        attached = result.get("components_attached") or []
        components_desc = ""
        if attached:
            listed = ", ".join(
                f"{html.escape(str(c.get('label') or c.get('slot')))}: "
                f"<code>{html.escape(str(c.get('name') or 'kept from source'))}</code>"
                for c in attached
            )
            components_desc = f"<b>Components:</b> {listed}<br>"

        validation_desc = ""
        if modular:
            if validated:
                validation_desc = (
                    "<b>Validated:</b> <span style='color:#10b981;font-weight:bold;'>"
                    "reopened with no external modules</span><br>"
                )
            else:
                reasons = "".join(
                    f"<li>{html.escape(str(e))}</li>"
                    for e in (result.get("validation").errors if result.get("validation") else ())
                )
                validation_desc = (
                    "<b style='color:#f59e0b;'>Not validated</b> &mdash; this file still "
                    "needs external modules to load:"
                    f"<ul style='margin:4px 0 8px 18px;'>{reasons}</ul>"
                )

        support = ""
        plan = result.get("component_plan")
        if plan is not None and str(getattr(plan, "support", "")).endswith("experimental"):
            support = (
                "<b style='color:#f59e0b;'>Experimental architecture</b> &mdash; the same "
                "checks ran, but this family has not been exercised against a real model.<br>"
            )

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
            + components_desc
            + validation_desc
            + support
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


# --- Extract LoRA ----------------------------------------------------------

#: Label -> safetensors dtype code, in the vocabulary lora_extract writes.
EXTRACT_DTYPE_CHOICES = [("BF16", "BF16"), ("FP16", "F16"), ("FP32", "F32")]


def _lora_output_dir() -> str:
    """Where Forge looks for LoRAs, so an extraction lands where it is usable.

    Asks the option first and falls back to the directory the loader is already
    reading from: a user who moved their LoRAs should not have to move the
    result too.
    """
    configured = getattr(shared.cmd_opts, "lora_dir", None)
    if configured and os.path.isdir(configured):
        return configured
    for path in lora_bake.available_loras().values():
        parent = os.path.dirname(path)
        if os.path.isdir(parent):
            return parent
    return os.path.join(getattr(shared, "models_path", "models"), "Lora")


def _suggested_extract_name(original: str, tuned: str, rank: int) -> str:
    """A filename that says what the file is without being opened."""
    def stem(name: str) -> str:
        base = os.path.splitext(os.path.basename(name or ""))[0]
        return "".join(c for c in base if c.isalnum() or c in "-_") or "model"

    return f"{stem(tuned)}-minus-{stem(original)}-r{int(rank)}.safetensors"


def _extract_plan(original: str, tuned: str, rank, conv_rank, dtype: str):
    if not original or not tuned:
        return None
    return lora_extract.plan_extraction(
        _checkpoint_path(original),
        _checkpoint_path(tuned),
        rank=int(rank),
        conv_rank=int(conv_rank),
        output_dtype=dtype,
    )


def extract_preview_handler(original: str, tuned: str, rank, conv_rank, dtype: str):
    """Badges, the plan, and a suggested filename -- all from headers.

    Runs on every change because it costs a header read: the point of the panel
    is that the consequences of a rank change are visible before the run, not
    after it.
    """
    o_info = _get_cached_checkpoint_info(original)
    t_info = _get_cached_checkpoint_info(tuned)
    o_html = format_badges_html(o_info) if o_info else ""
    t_html = format_badges_html(t_info, compatible_with_info=o_info) if t_info else ""

    try:
        plan = _extract_plan(original, tuned, rank, conv_rank, dtype)
    except Exception as e:
        return o_html, t_html, _err_html(e), gr.update()

    if plan is None:
        return o_html, t_html, "", gr.update()

    return (
        o_html,
        t_html,
        lora_extract.format_extraction_preview_html(plan),
        gr.update(placeholder=_suggested_extract_name(original, tuned, rank)),
    )


def extract_run_handler(
    id_task,
    original: str,
    tuned: str,
    rank,
    conv_rank,
    dtype: str,
    device: str,
    min_diff,
    filename: str,
):
    try:
        plan = _extract_plan(original, tuned, rank, conv_rank, dtype)
        if plan is None:
            raise ValueError("Select both an original and a tuned checkpoint.")
        if not plan.ok:
            # Already shown in the preview; repeated here because the user may
            # have pressed the button without reading it.
            return lora_extract.format_extraction_preview_html(plan)

        name = (filename or "").strip() or _suggested_extract_name(original, tuned, rank)
        name = os.path.basename(name)
        if not name.endswith(".safetensors"):
            name += ".safetensors"
        output_path = os.path.join(_lora_output_dir(), name)
        if os.path.exists(output_path):
            raise ValueError(f"{name} already exists; pick another name.")

        def progress_cb(done, total, key):
            if not total:
                return
            pct = 100 * done / total
            shared.state.textinfo = f"Extracting {name}: {pct:.0f}%"
            shared.state.sampling_steps = 100
            shared.state.sampling_step = int(pct)

        result = lora_extract.extract_lora(
            plan,
            output_path,
            min_diff=float(min_diff),
            device=device,
            progress_cb=progress_cb,
        )
    except Exception as e:
        return _err_html(e)

    size = lora_extract._human_bytes(result["bytes"])
    skipped = (
        f" &nbsp;&bull;&nbsp; {result['skipped']} modules below the difference floor were left out"
        if result["skipped"]
        else ""
    )
    return (
        "<div style='margin-top: 6px; padding: 8px 10px; border-radius: 4px; "
        "background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; font-size: 12px;'>"
        f"<b style='color: #6ee7b7;'>Saved {html.escape(name)}</b> &nbsp;&bull;&nbsp; "
        f"{result['tensors']} tensors &nbsp;&bull;&nbsp; {size}{skipped}"
        "<div style='color: #9ca3af; margin-top: 4px;'>Press the refresh button on any LoRA list to see it.</div>"
        "</div>"
    )


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

                # A Dropdown rather than a Radio: the mode list grows, and
                # Gradio filters a dropdown as you type where a Radio renders
                # as ragged rows of buttons. The panel below carries the
                # formula, which `info=` cannot format.
                with gr.Row():
                    merge_interp = gr.Dropdown(
                        choices=[label for label, _ in INTERP_CHOICES],
                        value=INTERP_KEY_TO_LABEL[merge_modes.INTERP_WEIGHTED_SUM],
                        label="Interpolation Method",
                        filterable=True,
                    )
                    merge_multiplier = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Multiplier (M) — α")
                    # The mode decides whether this exists at all. Sum Twice
                    # has a beta because of what Sum Twice is, not because the
                    # user asked for an advanced view.
                    merge_beta = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label=merge_modes.MERGE_MODES_BY_KEY[
                            merge_modes.INTERP_SUM_TWICE
                        ].beta_label,
                        visible=False,
                    )
                    # Only the stochastic modes have one. Recorded in the
                    # recipe, because a merge that cannot be repeated makes
                    # its own recipe a lie.
                    merge_seed = gr.Number(
                        value=0,
                        precision=0,
                        label="Seed — change it to draw a different subset",
                        visible=False,
                    )

                merge_mode_html = gr.HTML(
                    merge_modes.merge_mode_panel(merge_modes.INTERP_WEIGHTED_SUM)
                )

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

                # Anima only, and absent rather than disabled for everything
                # else -- see `block_weights_visibility`. Collapsed, because
                # the merge tab already has three accordions and a uniform
                # merge is what almost every run wants.
                with gr.Accordion(
                    "Per-block weights (Anima)", open=False, visible=False
                ) as block_weights_accordion:
                    gr.Markdown(
                        "Weight individual layers differently from the Multiplier. One rule per line, "
                        "written the way the recipes already record them: "
                        "`L05-L09:self_attn.q_proj self_attn.k_proj:0.08`  — a layer range, the module "
                        "names it applies to, and the weight. Layers no rule names merge at the "
                        "Multiplier. **Where two rules overlap, the later one wins.**"
                    )
                    merge_block_weights = gr.Textbox(
                        label="Rules",
                        lines=4,
                        max_lines=12,
                        placeholder="L05-L09:self_attn.q_proj self_attn.k_proj:0.08",
                    )
                    with gr.Row(equal_height=True):
                        block_weights_source = gr.Dropdown(
                            label="Start from a checkpoint that already has rules",
                            choices=sorted(sd_models.checkpoint_tiles()),
                            scale=4,
                        )
                        block_weights_import = gr.Button("Import rules", scale=1)
                        create_refresh_button(
                            block_weights_source,
                            refresh_models_and_cache,
                            lambda: {"choices": sorted(sd_models.checkpoint_tiles())},
                            "merge_studio_refresh_block_weight_source",
                        )
                    merge_block_profile = gr.HTML(
                        elemental_weights.format_weight_editor_html(
                            elemental_weights.parse_weight_spec(""), None, []
                        )
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
                            create_refresh_button([merge_lora_dd], lora_bake.reload_loras, lambda: {"choices": _lora_choices()}, f"merge_studio_lora_refresh_{i}")
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

                with gr.Accordion("AIO Components", open=True) as merge_components_accordion:
                    gr.Markdown(
                        "An AIO carries its own text encoder and VAE. Pick the file for each "
                        "component this architecture needs, or keep what the checkpoint already "
                        "has. Nothing is filled in for you, and the Additional Modules "
                        "configured in Forge are never used here."
                    )
                    merge_component_rows = []
                    merge_component_layouts = []
                    merge_component_notes = []
                    for i in range(MAX_COMPONENT_ROWS):
                        with gr.Row(visible=False) as component_row_layout:
                            component_slot = gr.Textbox(value="", visible=False)
                            component_file = gr.Dropdown(label="Component", choices=[], value=None, scale=3)
                            component_format = gr.Dropdown(
                                label="Precision",
                                choices=[c[0] for c in component_ui.COMPONENT_FORMAT_CHOICES],
                                value=component_ui.COMPONENT_FORMAT_CHOICES[0][0],
                                scale=2,
                            )
                        component_note = gr.HTML("")
                        merge_component_rows.append((component_slot, component_file, component_format))
                        merge_component_layouts.append(component_row_layout)
                        merge_component_notes.append(component_note)

                    _component_refresh_outputs = []
                    for layout, (slot, file_dd, fmt), note in zip(
                        merge_component_layouts, merge_component_rows, merge_component_notes
                    ):
                        _component_refresh_outputs += [layout, slot, file_dd, fmt, note]

                    for _trigger in (merge_primary, merge_save_mode):
                        _trigger.change(
                            fn=_refresh_component_rows,
                            inputs=[merge_primary, merge_save_mode],
                            outputs=_component_refresh_outputs,
                            show_progress=False,
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

                # Model A decides both whether the editor exists at all and
                # how many blocks the rules have to work with.
                merge_primary.change(
                    fn=block_weights_visibility,
                    inputs=[merge_primary],
                    outputs=[block_weights_accordion],
                    show_progress=False,
                    queue=False,
                )

                # The profile redraws on anything that changes what the rules
                # resolve to: the text, the base, or which model they apply to.
                for _control in (merge_block_weights, merge_multiplier, merge_primary):
                    _control.change(
                        fn=block_weights_preview,
                        inputs=[merge_primary, merge_block_weights, merge_multiplier],
                        outputs=[merge_block_profile],
                        show_progress=False,
                        queue=False,
                    )
                block_weights_import.click(
                    fn=import_block_weights,
                    inputs=[block_weights_source],
                    outputs=[merge_block_weights],
                    queue=False,
                )

                merge_interp.change(
                    fn=merge_update_method,
                    inputs=[merge_interp],
                    outputs=[
                        merge_multiplier, secondary_col, tertiary_col,
                        merge_beta, merge_seed, merge_mode_html, merge_config_source,
                    ],
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
                    merge_interp, merge_multiplier, merge_beta, merge_seed,
                    merge_block_weights, anima_extend_ratio,
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
                        + [
                            secondary_col, tertiary_col, merge_beta, merge_seed,
                            merge_mode_html, block_weights_accordion,
                            merge_block_profile, recipe_status,
                        ]
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
                        merge_beta,
                        merge_seed,
                        merge_block_weights,
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
                        *[c for row in merge_component_rows for c in row],
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

            with gr.Tab("Extract LoRA"):
                gr.Markdown(
                    "Turn the difference between two checkpoints into a LoRA. Pick the model a tune **started from** "
                    "and the tuned result; what comes out is an adapter that reproduces the change, at a fraction of "
                    "the size. The panel below shows exactly what would be subtracted **before** anything runs."
                )
                gr.Markdown(
                    "Both checkpoints must be the same architecture and stored in **FP32, FP16 or BF16** — "
                    "a difference between quantized weights is noise, not a difference, so those are refused."
                )

                with gr.Row():
                    with gr.Column():
                        extract_original = gr.Dropdown(
                            label="Original — what the tune started from",
                            choices=sorted(sd_models.checkpoint_tiles()),
                        )
                        extract_original_badge = gr.HTML("")
                    with gr.Column():
                        extract_tuned = gr.Dropdown(
                            label="Tuned — the finished model",
                            choices=sorted(sd_models.checkpoint_tiles()),
                        )
                        extract_tuned_badge = gr.HTML("")

                with gr.Row():
                    create_refresh_button(
                        [extract_original, extract_tuned],
                        sd_models.list_models,
                        lambda: {"choices": sorted(sd_models.checkpoint_tiles())},
                        "merge_studio_refresh_extract",
                    )

                extract_preview = gr.HTML("")

                with gr.Row():
                    extract_rank = gr.Slider(
                        label="Rank", minimum=4, maximum=256, step=4, value=64,
                        info="Higher keeps more of the difference and costs more disk. 64 is a sane default.",
                    )
                    extract_conv_rank = gr.Slider(
                        label="Conv rank", minimum=1, maximum=128, step=1, value=16,
                        info="Used only by UNet models (SD 1.5 / SDXL). Kernels need less rank than linear layers.",
                    )

                with gr.Row():
                    extract_dtype = gr.Dropdown(
                        label="Output precision", choices=EXTRACT_DTYPE_CHOICES, value="BF16"
                    )
                    extract_device = gr.Radio(
                        label="Compute on",
                        choices=[("Auto", "auto"), ("GPU", "cuda"), ("CPU", "cpu")],
                        value="auto",
                        info="The decomposition is many SVDs; on CPU expect minutes to tens of minutes.",
                    )
                    extract_min_diff = gr.Number(
                        label="Difference floor", value=1e-4,
                        info="Modules that moved less than this are left out instead of contributing noise.",
                    )

                extract_filename = gr.Textbox(
                    label="Save as", placeholder="picked automatically from the two model names"
                )
                extract_btn = gr.Button("Extract LoRA", variant="primary")
                extract_result = gr.HTML("")

                _extract_inputs = [
                    extract_original, extract_tuned, extract_rank, extract_conv_rank, extract_dtype,
                ]
                _extract_outputs = [
                    extract_original_badge, extract_tuned_badge, extract_preview, extract_filename,
                ]
                for _control in _extract_inputs:
                    _control.change(
                        fn=extract_preview_handler,
                        inputs=_extract_inputs,
                        outputs=_extract_outputs,
                        show_progress=False,
                    )

                extract_btn.click(fn=lambda: "", outputs=[extract_result], queue=False, show_progress=False).then(
                    fn=call_queue.wrap_gradio_gpu_call(extract_run_handler, extra_outputs=lambda: [gr.skip()]),
                    inputs=[
                        dummy_component, extract_original, extract_tuned, extract_rank,
                        extract_conv_rank, extract_dtype, extract_device, extract_min_diff,
                        extract_filename,
                    ],
                    outputs=[extract_result],
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
                        # The inspector lists LoRAs as well, so re-scanning
                        # only checkpoints and modules left a freshly
                        # downloaded one missing from a dropdown the button
                        # had just claimed to refresh.
                        try:
                            lora_bake.reload_loras()
                        except Exception:
                            pass
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
