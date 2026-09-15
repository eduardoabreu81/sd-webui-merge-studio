import json
import os
import sys
import traceback

import gradio as gr

from modules import call_queue, script_callbacks, sd_models, shared
from modules.ui_common import create_refresh_button
from modules.ui_components import FormRow, InputAccordion

_EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)

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


def checkpoint_badge_handler(name: str) -> str:
    if not name:
        return ""
    try:
        path = _checkpoint_path(name)
        info = inspect_checkpoint(path)
        return format_badges_html(info)
    except Exception:
        return ""


def run_inspection_handler(name: str) -> str:
    if not name:
        return "<div style='padding: 20px; color: #9ca3af;'>Select a checkpoint above to inspect.</div>"
    try:
        path = _checkpoint_path(name)
        info = inspect_checkpoint(path)
        return format_recipe_dashboard_html(info)
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
            progress_cb=progress_cb,
        )

        sd_models.list_models()
        gr.Info("Merge completed successfully!", duration=5)
        skipped_total = sum(len(v) for v in result["skipped"].values())
        size_str = _format_size(result["output"])
        loras_desc = ", ".join(f"{a['name']} ({a['strength']})" for a in result.get("loras", [])) if result.get("loras") else ""
        baked_vae_desc = f"<b>Baked VAE:</b> <code>{result['baked_vae']}</code><br>" if result.get("baked_vae") else ""
        html = (
            f"<div style='margin-top: 10px; line-height: 1.6; font-size: 14px;'>"
            f"<b>Checkpoint saved to:</b> <code>{result['output']}</code><br>"
            f"<b>File size:</b> <span style='color: #10b981; font-weight: bold;'>{size_str}</span><br>"
            f"<b>Format:</b> <code>{result['output_format']}</code> (Mode: <code>{save_mode}</code>)<br>"
            f"<b>Merged layers:</b> UNet: {result['merged']['unet']}, CLIP: {result['merged']['clip']}, VAE: {result['merged']['vae']}<br>"
            + baked_vae_desc
            + (f"<b>Baked LoRAs:</b> {loras_desc}<br>" if loras_desc else "")
            + f"</div>"
        )
        llm_adapter_hits = [a["name"] for a in result.get("loras", []) if a.get("llm_adapter_warning")]
        if llm_adapter_hits:
            gr.Warning(f"LoRA(s) with LLM adapter weights baked in: {', '.join(llm_adapter_hits)}. Anima's own training guidance says never to train these alongside a LoRA.", duration=10)
            html += f"<div style='color:orange; margin-top: 6px;'>Warning: {', '.join(llm_adapter_hits)} contains LLM adapter weights — not recommended by Anima's own training guidance.</div>"
        if skipped_total:
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
                    create_refresh_button(
                        [merge_primary, merge_secondary, merge_tertiary],
                        sd_models.list_models,
                        lambda: {"choices": sorted(sd_models.checkpoint_tiles())},
                        "merge_studio_refresh_models",
                    )

                merge_primary.change(fn=checkpoint_badge_handler, inputs=[merge_primary], outputs=[primary_badge], show_progress=False, queue=False)
                merge_secondary.change(fn=checkpoint_badge_handler, inputs=[merge_secondary], outputs=[secondary_badge], show_progress=False, queue=False)
                merge_tertiary.change(fn=checkpoint_badge_handler, inputs=[merge_tertiary], outputs=[tertiary_badge], show_progress=False, queue=False)

                with gr.Row():
                    merge_interp = gr.Radio(
                        choices=[label for label, _ in INTERP_CHOICES],
                        value=INTERP_CHOICES[1][0],
                        label="Interpolation Method",
                        info=INTERP_DESCRIPTIONS[checkpoint_merge.INTERP_WEIGHTED_SUM],
                    )
                    merge_multiplier = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Multiplier (M)")

                with gr.Accordion("Bake LoRA(s) into Checkpoint (Optional)", open=False):
                    gr.Markdown("Optionally apply one or multiple LoRAs (e.g. Turbo LoRA, Style LoRAs) directly into the checkpoint weights.")
                    MAX_LORAS = 10
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
                    "Inspect any checkpoint's internal components (UNet/DiT, Text Encoder, VAE), architecture, precision, "
                    "and full merge recipe provenance without loading weights into RAM/VRAM."
                )
                with gr.Row():
                    inspector_checkpoint = gr.Dropdown(label="Checkpoint", choices=sorted(sd_models.checkpoint_tiles()))
                    create_refresh_button(
                        [inspector_checkpoint],
                        sd_models.list_models,
                        lambda: {"choices": sorted(sd_models.checkpoint_tiles())},
                        "merge_studio_refresh_inspector",
                    )
                inspector_btn = gr.Button("Inspect Checkpoint", variant="primary")
                inspector_html = gr.HTML(
                    "<div style='padding: 20px; color: #9ca3af;'>Select a checkpoint above to inspect its components and merge recipe.</div>"
                )

                inspector_btn.click(fn=run_inspection_handler, inputs=[inspector_checkpoint], outputs=[inspector_html])
                inspector_checkpoint.change(fn=run_inspection_handler, inputs=[inspector_checkpoint], outputs=[inspector_html])

        for comp in (
            doctor_checkpoint,
            doctor_mode,
            doctor_new_name,
            merge_primary,
            merge_secondary,
            merge_tertiary,
            merge_interp,
            merge_multiplier,
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
