# 🔀 Merge Studio

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Model merging, conversion, LoRA baking and inspection for
[Stable Diffusion WebUI Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo).

Merge Studio brings the most useful model-management tools into a single Forge Neo tab. Merge compatible checkpoints, convert model precision, bake LoRAs and VAEs, inspect model metadata, save reusable recipes, and repair common quantization metadata issues without leaving the WebUI.

Unlike traditional checkpoint mergers, Merge Studio can work with the quantized models that Forge Neo already knows how to load — including FP8, INT8, NVFP4, and ConvRot formats.

---

## ✨ Why Merge Studio?

- **One workspace** — merge, convert, bake, inspect, and repair models from the same extension.
- **Quantization-aware** — process supported quantized checkpoints instead of being limited to FP16 and BF16 files.
- **Forge-native** — uses Forge Neo's own model and LoRA loading paths for consistent compatibility.
- **Reusable workflows** — save a complete setup as a recipe and load it again later.
- **Safer merges** — architecture checks and clear warnings help prevent incompatible combinations.
- **Fast inspection** — read model, LoRA, VAE, and text-encoder metadata without loading the full file into memory.

---

## 📦 Installation

1. Open Forge Neo WebUI.
2. Go to **Extensions** → **Install from URL**.
3. Paste:

```text
https://github.com/eduardoabreu81/sd-webui-merge-studio
```

4. Click **Install**.
5. Reload or restart the WebUI.
6. Open the new **Merge Studio** tab.

> [!IMPORTANT]
> Merge Studio is built for **Forge Neo** and follows the model support provided by Forge Neo itself.

---

## 🎯 What it does

| Feature | In short | Details |
| :--- | :--- | :--- |
| **Checkpoint merge** | Six modes, from a plain blend to DARE. The form shows only the models and sliders the chosen mode actually uses. | [Merge Modes](../../wiki/Merge-Modes) |
| **Convert & quantize** | Pick a precision per component — diffusion model, text encoder, VAE — and rebuild the file. | [Quantization and Formats](../../wiki/Quantization-and-Formats) |
| **AIO checkpoints** | Build a self-contained checkpoint that loads with no Additional Modules, and have it verified by reopening. | [AIO Checkpoints](../../wiki/AIO-Checkpoints) |
| **LoRA & VAE baking** | Bake up to 10 LoRAs and a custom VAE into a checkpoint, each at its own strength. | [LoRA Tools](../../wiki/LoRA-Tools) |
| **LoRA merge** | Combine up to six LoRAs into one adapter, exactly, on a tab of its own. | [LoRA Tools](../../wiki/LoRA-Tools) |
| **Anima merges** | Merge across Anima generations with automatic block mapping, and weight chosen layers differently from the rest. | [Anima Merges](../../wiki/Anima-Merges) |
| **Recipes** | Save the whole setup as JSON and load it back, including an AIO's components and a DARE seed. | [Recipes and Inspector](../../wiki/Recipes-and-Inspector) |
| **Inspector** | Read architecture, components, lineage, baked LoRAs and raw metadata — including merges made in other tools. | [Recipes and Inspector](../../wiki/Recipes-and-Inspector) |
| **Quant Format Doctor** | Diagnose and repair missing quantization metadata before it becomes a loading error. | [Quantization and Formats](../../wiki/Quantization-and-Formats) |

---

## 🚀 Quick Start

### Merge two checkpoints

1. Open **Merge Studio** → **Checkpoint Merge & Studio**.
2. Select the primary model in **Model A**.
3. Pick a merge mode and fill the models it asks for.
4. Set the output name, multiplier, precision, and save mode.
5. Click **Merge / Process**.

### Build an AIO (self-contained) checkpoint

1. Select the model in **Model A**.
2. Set **Save Mode** to **AIO**.
3. Fill every component row that appears — pick a file, or keep what the checkpoint already has.
4. Click **Merge / Process**.

The result is reported as an AIO only after it reopens with no external modules.

### Convert or quantize a model

1. Select the source checkpoint in **Model A**.
2. Choose **No Interpolation**.
3. Select the desired output precision for each component.
4. Enter an output name and click **Merge / Process**.

### Bake LoRAs

1. Expand **Bake LoRA(s) into Checkpoint**.
2. Select a LoRA and set its strength, using **Add LoRA** for more.
3. Click **Merge / Process**.

### Inspect a model

1. Open **Model Recipe & Inspector**.
2. Select any available checkpoint, LoRA, text encoder, or VAE.
3. Review its identity, components, lineage, and metadata.

### Repair quantization metadata

1. Open **Quant Format Doctor**.
2. Select the model and click **Diagnose**.
3. If a known issue is found, choose whether to repair in place or create a copy.

---

## ✅ Compatibility

Merge Studio supports the model families and file formats that Forge Neo can load through its standard checkpoint system.

| Workflow | Support |
| :--- | :--- |
| Convert, quantize, or bake a single model | Any compatible model Forge Neo can load |
| Merge checkpoints | Models with compatible architectures and tensor shapes |
| Weight layers individually | Anima checkpoints |
| Cross-generation Anima merge | Supported when the newer, larger model is Model A |
| Merge LoRAs into one LoRA | Plain LoRA / LoCon adapters of the same architecture |
| Read a merge made in another tool | Recipes written by other mergers, and ComfyUI workflows |
| Build an AIO checkpoint | Any architecture Forge Neo declares components for; verified by reopening |
| Inspect safetensors metadata | Checkpoints, LoRAs, text encoders, and VAEs |

**Output formats:** FP16, BF16, FP8 e4m3fn, FP8 e5m2, INT8, INT8 ConvRot, NVFP4, INT4 ConvRot W4A4, or same as the source.

**Not supported:** GGUF, Nunchaku / SVDQuant, NF4 and legacy bitsandbytes storage, and SD2 / SD3 — none of which Forge Neo's current model loader handles.

---

## ⚠️ Before you merge

- **Keep the originals** until you have generated with the result. A merge that loads is not the same as a merge that works.
- **Anima 3.8B v1.1 is refused.** It carries a Semantic Connector that Forge Neo drops on load, so the saved result would be missing it. The merge is blocked with a message naming what was found. Use the plain Anima 3.8B instead.
- **For cross-generation Anima merges, Model A must be the newer model with more blocks**, and Blend and Delta are not interchangeable at the same number — Delta moves roughly sixteen times less. See [Anima Merges](../../wiki/Anima-Merges).
- **Compatibility checks are not exhaustive.** Merge Studio blocks the mismatches it knows about, but it cannot safely guess every new or uncommon architecture.
- **Quantizing a VAE** affects image quality more than quantizing the diffusion model. FP16 or BF16 is the safest choice.
- **Some models keep the text encoder and VAE separately.** Make sure the required files are selected under Forge Neo's Additional Modules, even when saving only the diffusion model.

Things that went differently than expected are collected in
**[Troubleshooting](../../wiki/Troubleshooting)**.

---

## 📚 Documentation

This README covers installation and everyday use. The reference material —
merge methodology, architecture notes, model-specific behaviour, quantization
details and troubleshooting — lives in the
**[Wiki](https://github.com/eduardoabreu81/sd-webui-merge-studio/wiki)**.

---

## 🙏 Credits

- Built for **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02.
- Anima cross-generation mapping follows Forge Neo's own mapping tables.
- Additional Anima mapping research was informed by **[ComfyUI-Anima-Remap](https://github.com/shin131002/ComfyUI-Anima-Remap)** by shin131002.
- The **Delta** write rule for inserted blocks follows **[Anima Delta Mix](https://github.com/bestluner-create/Anima-Delta-Mix)** by bestluner-create, whose block maps were verified to agree with Forge Neo's.
- **Similarity Add Difference** is ported from **[meh](https://github.com/s1dlx/meh)** by s1dlx (MIT, Copyright © 2023 s1dlx).
- **DARE** is written from *[Language Models are Super Mario](https://arxiv.org/abs/2311.03099)* (Yu et al., arXiv:2311.03099); **[safetensors-merge-supermario](https://github.com/martyn/safetensors-merge-supermario)** (MIT) was the reference implementation consulted.
- Weighted Sum, Add Difference and Sum Twice are written from the formulas **[SuperMerger](https://github.com/hako-mikan/sd-webui-supermerger)** publishes in its README. No code was taken from it.
- Released under the [MIT License](LICENSE).

---

Found something it gets wrong, or a model it refuses that it should not?
[Open an issue](https://github.com/eduardoabreu81/sd-webui-merge-studio/issues)
— what the Inspector reports for the models involved is usually enough to
diagnose it.
