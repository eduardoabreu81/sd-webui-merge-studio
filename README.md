# 🔀 Merge Studio

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Model merging, conversion, LoRA baking, and inspection for [Stable Diffusion WebUI Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Merge Studio brings the most useful model-management tools into a single Forge Neo tab. Merge compatible checkpoints, convert model precision, bake LoRAs and VAEs, inspect model metadata, save reusable recipes, and repair common quantization metadata issues without leaving the WebUI.

Unlike traditional checkpoint mergers, Merge Studio can work with the quantized models that Forge Neo already knows how to load — including FP8, INT8, NVFP4, and ConvRot formats.

---

## 📋 Table of Contents

- [Why Merge Studio?](#-why-merge-studio)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Compatibility](#-compatibility)
- [Important Notes](#-important-notes)
- [Technical Documentation](#-technical-documentation)
- [Credits](#-credits)

---

## ✨ Why Merge Studio?

- **One workspace** — merge, convert, bake, inspect, and repair models from the same extension.
- **Quantization-aware** — process supported quantized checkpoints instead of being limited to FP16 and BF16 files.
- **Forge-native** — uses Forge Neo's own model and LoRA loading paths for consistent compatibility.
- **Reusable workflows** — save a complete setup as a recipe and load it again later.
- **Safer merges** — architecture checks and clear warnings help prevent incompatible combinations.
- **Fast inspection** — read model, LoRA, VAE, and text-encoder metadata without loading the full file into memory.

---

## 🎯 Features

### 🔀 Checkpoint Merge & Conversion

- Merge two compatible checkpoints with **Weighted Sum**.
- Transfer a model difference with **Add Difference**.
- Use **No Interpolation** to convert, re-quantize, or process a single model.
- Choose precision separately for the diffusion model, text encoder, and VAE.
- Merge compatible Anima generations with automatic block mapping.
- See the detected architecture and embedded components before processing.

### 🧩 LoRA & VAE Baking

- Bake up to **10 LoRAs** into one checkpoint.
- Set an individual strength for each LoRA.
- Keep the original VAE, remove it, or bake a custom VAE.
- Preserve available LoRA trigger declarations in the saved model recipe.

### 💾 Save & Load Recipes

- Save the complete merge setup as a JSON recipe.
- Restore models, merge settings, precision choices, VAE options, and LoRA selections.
- Keep recipes in Forge's data directory so they can survive extension updates.
- Load portable recipes even when some referenced files are not installed locally.

### 🔎 Model Recipe & Inspector

- Inspect checkpoints, LoRAs, text encoders, and VAEs.
- Identify model architecture and embedded components.
- Read merge lineage, parent hashes, baked LoRAs, and available trigger declarations.
- Recognize common LoRA and LyCORIS formats.
- View raw safetensors metadata when you need more detail.

### 🩺 Quant Format Doctor

- Diagnose missing quantization metadata without loading the model tensors.
- Repair the original file or create a corrected copy.
- Catch known metadata problems before they become model-loading errors.

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

## 🚀 Quick Start

### Merge Checkpoints

1. Open **Merge Studio** → **Checkpoint Merge & Studio**.
2. Select the primary model in **Model A**.
3. Choose **Weighted Sum** and add Model B, or choose **Add Difference** and add Models B and C.
4. Set the output name, multiplier, precision, and save mode.
5. Click **Merge / Process**.

### Convert or Quantize a Model

1. Select the source checkpoint in **Model A**.
2. Choose **No Interpolation**.
3. Select the desired output precision for each component.
4. Enter an output name and click **Merge / Process**.

### Bake LoRAs

1. Expand **Bake LoRA(s) into Checkpoint**.
2. Select a LoRA and set its strength.
3. Use **Add LoRA** to include more LoRAs if needed.
4. Click **Merge / Process** to create the baked checkpoint.

### Inspect a Model

1. Open **Model Recipe & Inspector**.
2. Select any available checkpoint, LoRA, text encoder, or VAE.
3. Review its identity, components, lineage, and metadata.

### Repair Quantization Metadata

1. Open **Quant Format Doctor**.
2. Select the model and click **Diagnose**.
3. If a known issue is found, choose whether to repair in place or create a copy.

---

## ✅ Compatibility

Merge Studio supports the model families and file formats that Forge Neo can load through its standard checkpoint system.

### Common Workflows

| Workflow | Support |
| :--- | :--- |
| Convert, quantize, or bake a single model | Any compatible model Forge Neo can load |
| Merge checkpoints | Models with compatible architectures and tensor shapes |
| Cross-generation Anima merge | Supported when the newer/larger model is Model A |
| Inspect safetensors metadata | Checkpoints, LoRAs, text encoders, and VAEs |

### Output Formats

- FP16 and BF16
- FP8 e4m3fn and e5m2
- INT8
- INT8 ConvRot
- NVFP4
- INT4 ConvRot W4A4
- Same as the source checkpoint

### Not Supported

- GGUF models
- Nunchaku / SVDQuant models
- NF4 or legacy bitsandbytes storage
- SD2 and SD3, which are not supported by Forge Neo's current model loader

---

## ⚠️ Important Notes

- Models used in the same merge must be compatible. Merge Studio blocks known mismatches, but it cannot safely guess every new or uncommon architecture.
- For cross-generation Anima merges, place the newer model with more blocks in **Model A**.
- Some models store the text encoder and VAE separately. Make sure the required files are selected under Forge Neo's **Additional Modules**, even when saving only the diffusion model.
- If Forge reports a LoRA mismatch in the console, the checkpoint can still be saved, but that LoRA may have been skipped.
- Quantizing a VAE can affect image quality more noticeably than quantizing the diffusion model. FP16 or BF16 is the safest choice for the VAE.
- Always keep the original models until you have tested the generated checkpoint.

---

## 📚 Technical Documentation

The README is intentionally focused on installation and everyday use. Architecture notes, model-specific behavior, merge methodology, quantization details, and advanced troubleshooting belong in the **[GitHub Wiki](https://github.com/eduardoabreu81/sd-webui-merge-studio/wiki)**.

The validation and research material already available in the repository can be found in **[Research Notes](docs/RESEARCH.md)**.

---

## 🙏 Credits

- Built for **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02.
- Anima cross-generation mapping follows Forge Neo's own mapping tables.
- Additional Anima mapping research was informed by **[ComfyUI-Anima-Remap](https://github.com/shin131002/ComfyUI-Anima-Remap)** by shin131002.
- Released under the [MIT License](LICENSE).

---

<div align="center">

Made for the Stable Diffusion and Forge Neo community.

**[Report Bug](https://github.com/eduardoabreu81/sd-webui-merge-studio/issues)** • **[Request Feature](https://github.com/eduardoabreu81/sd-webui-merge-studio/issues)** • **[Wiki](https://github.com/eduardoabreu81/sd-webui-merge-studio/wiki)**

</div>
