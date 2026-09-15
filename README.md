# Merge Studio

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Merge Studio is an all-in-one studio for Forge Neo that merges checkpoints, bakes multiple LoRAs, converts model precisions, inspects embedded recipes, and fixes quantized files.

Standard checkpoint mergers only work with raw unquantized tensors (FP16/BF16). If you try merging quantized checkpoints (FP8, INT8, ConvRot), standard mergers produce broken files or crash. Merge Studio solves this by dequantizing weights before computing merges, supporting modern architectures (SD1.5, SDXL, Pony, Illustrious, Flux, Anima, Wan2.1), and allowing you to bake up to 10 LoRAs and custom VAEs in a single pass.

---

## Table of Contents

- [Why Merge Studio?](#why-merge-studio)
- [Key Features](#key-features)
  - [1. Checkpoint Merge & Conversion](#1-checkpoint-merge--conversion)
  - [2. Dynamic LoRA Baking (Up to 10 LoRAs)](#2-dynamic-lora-baking-up-to-10-loras)
  - [3. Custom VAE Baking and Stripping](#3-custom-vae-baking-and-stripping)
  - [4. Model Recipe & Inspector](#4-model-recipe--inspector)
  - [5. Quant Format Doctor](#5-quant-format-doctor)
- [Quick Start Guides](#quick-start-guides)
  - [How to Merge Checkpoints](#how-to-merge-checkpoints)
  - [How to Bake LoRAs into a Checkpoint](#how-to-bake-loras-into-a-checkpoint)
  - [How to Inspect Checkpoint Lineage](#how-to-inspect-checkpoint-lineage)
  - [How to Fix a Broken Quantized Model](#how-to-fix-a-broken-quantized-model)
- [Supported Models & Formats](#supported-models--formats)
- [Installation](#installation)
- [Credits & License](#credits--license)

---

## Why Merge Studio?

- **Quantization-Aware**: Merges FP8, INT8, and ConvRot checkpoints accurately without tensor corruption.
- **Built for Modern DiT & SD Models**: Full support for Anima, Wan2.1, Flux, SDXL, Pony, Illustrious, and SD1.5.
- **Save Disk Space**: Bake Turbo LoRAs or style LoRAs directly into models, or strip unneeded VAEs to save gigabytes of storage.
- **Zero-RAM Recipe Inspector**: Check what components, base models, parent hashes, and LoRAs are inside any checkpoint in under 5 milliseconds.
- **Self-Healing**: Diagnose and repair common header metadata errors in community quantized checkpoints with one click.

---

## Key Features

### 1. Checkpoint Merge & Conversion

- **Weighted Sum (`A * (1 - M) + B * M`)**: Smoothly blend two checkpoints using an intuitive multiplier slider.
- **Add Difference (`A + (B - C) * M`)**: Extract unique stylistic or architectural differences between models B and C, and inject them into base model A.
- **No Interpolation (Format Converter)**: Convert or re-quantize a single checkpoint without merging. Switch between FP16, BF16, FP8 (e4m3fn / e5m2), and INT8.
- **Instant Component Badges**: Selecting Model A, B, or C immediately displays a badge showing component presence (`UNet/DiT`, `CLIP/T5/Qwen`, `VAE`), detected model family, precision, and file size.
- **Per-Component Precision**: Target different datatypes for diffusion models, text encoders, and VAEs independently.
- **Adaptive ConvRot Quantization**: Automatically selects optimal group sizes (256, 128, 64) for channel-sensitive architectures like SDXL and Illustrious to prevent shape mismatch crashes.

### 2. Dynamic LoRA Baking (Up to 10 LoRAs)

- **Dynamic Slots & Per-Row Removal**: Start with 1 slot and add more as needed with **Add LoRA** (up to 10 simultaneous LoRAs). Each row features its own dedicated red **X** button to delete that specific LoRA and automatically compact the list, plus a **Clear All LoRAs** button.
- **Native Forge Engine**: Uses Forge's native LoRA application pipeline rather than external approximations, ensuring identical results to loading LoRAs at generation time.
- **Preserved Trigger Words**: Activation text from LoRA metadata is automatically preserved in sidecar notes so you always know the required trigger words.
- **Anima & DiT Smart Warnings**: Normalizes trigger words to lowercase spacing and warns if a LoRA contains LLM (Qwen3) adapter weights that could destabilize Anima checkpoints.

### 3. Custom VAE Baking and Stripping

- **Original**: Keeps the VAE embedded in Model A.
- **None (Strip VAE)**: Completely removes the VAE from the output file, significantly reducing file size. Ideal for workflows where VAEs are loaded separately in Forge.
- **Custom VAE**: Pick any standalone VAE from your `models/VAE` folder and bake it directly into your merged checkpoint.
- **VAE Precision Casting**: Save your baked VAE in FP16, BF16, or FP8 e4m3fn.

### 4. Model Recipe & Inspector

- **Instant Header Analysis**: Reads `.safetensors` headers in <5ms with zero VRAM or RAM overhead.
- **Component Status**: Clearly identifies whether a checkpoint contains a UNet/DiT, CLIP/T5/Qwen text encoder, VAE, or LLM adapters.
- **Lineage & Recipe Detection**:
  - Automatically reads WebUI merge recipes (`sd_merge_recipe`).
  - Mechanically extracts ComfyUI workflows and prompt graphs (`UNETLoader`, `CLIPLoader`, `VAELoader`, `LoraLoaderModelOnly`).
  - Displays original base models, parent SHA-256 hashes, merge methods, and ratios.
- **Baked LoRA Discovery**: Lists any LoRAs previously baked into the checkpoint, along with bake weights and activation tags.
- **Raw Metadata Viewer**: Interactive expandable viewer for all raw metadata keys.

### 5. Quant Format Doctor

- **One-Click Diagnostic**: Checks whether a quantized checkpoint suffers from missing `format` metadata keys that cause `ValueError: Unknown quantization format for layer ...` during loading.
- **Fast Header Stream**: Analyzes file headers without loading large tensors.
- **Safe In-Place or Copy Repair**: Fix the file directly or output a repaired copy.

---

## Quick Start Guides

### How to Merge Checkpoints

1. Navigate to the **Merge Studio** tab and select the **Checkpoint Merge & Studio** sub-tab.
2. Choose your **Primary Model (A)** and **Secondary Model (B)**. The real-time badge underneath will confirm their architectures and components.
3. Select your **Interpolation Method** (e.g. *Weighted Sum*) and set the **Multiplier (M)**.
4. Set your desired output filename, target precision, and save mode (UNet Only or Full Checkpoint).
5. Click **Merge / Process**.

### How to Bake LoRAs into a Checkpoint

1. In the **Checkpoint Merge & Studio** tab, expand the **Bake LoRA(s) into Checkpoint (Optional)** accordion.
2. Select your LoRA from the dropdown and set the desired strength (for Anima Turbo LoRAs, a strength around `0.6 - 0.8` is recommended).
3. Click **Add LoRA** if you wish to apply additional LoRAs (e.g. a style LoRA or detailer).
4. Enter an output filename and click **Merge / Process**.

### How to Inspect Checkpoint Lineage

1. Open the **Model Recipe & Inspector** sub-tab.
2. Select any checkpoint from the dropdown.
3. Instantly view its architecture, component breakdown, base model parentage, ComfyUI workflow history, and previously baked LoRAs.

### How to Fix a Broken Quantized Model

1. Open the **Quant Format Doctor** sub-tab.
2. Select the problematic checkpoint and click **Diagnose**.
3. If missing quantization keys are detected, choose **Repair in-place** (or create a new copy) and click **Repair Checkpoint**.

---

## Supported Models & Formats

- **Supported Architectures**: SD1.5, SD2.1, SDXL, Pony, Illustrious, Flux, Anima, Wan2.1, SD3, and other standard Forge Neo diffusion architectures.
- **Supported Precision Types**: FP16, BF16, FP8 (e4m3fn / e5m2), INT8 (tensor-wise, ConvRot channel-wise), NVFP4, and INT4.
- **Not Supported**: GGUF, NF4, FP4 storage formats, and Nunchaku/SVDQuant files (these use incompatible tensor packing and are safely rejected with an informative error message).

---

## Installation

1. Open Forge Neo WebUI.
2. Go to the **Extensions** tab -> **Install from URL**.
3. Paste the repository URL:
   ```text
   https://github.com/eduardoabreu81/sd-webui-merge-studio
   ```
4. Click **Install**.
5. Reload or restart the WebUI, then navigate to the new **Merge Studio** tab.

---

## Credits & License

- Built for **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02.
- Adheres to standard Forge & WebUI merge conventions.
- Released under the [MIT License](LICENSE).

<div align="center">

Made for the Stable Diffusion and Forge Neo community.

**[Report Bug](https://github.com/eduardoabreu81/sd-webui-merge-studio/issues)** • **[Request Feature](https://github.com/eduardoabreu81/sd-webui-merge-studio/issues)**

</div>
