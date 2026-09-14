# 🧬 Merge Studio

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

A quantization-aware checkpoint merger for Forge Neo. Merges, converts, and bakes LoRAs
directly on the *dequantized* weights instead of interpolating raw tensors like the native
Checkpoint Merger — so it works correctly on quantized builds (fp8, int8, nvfp4, mxfp8,
convrot) and Anima checkpoints, not just plain fp16/bf16. Also ships a standalone doctor tool
that repairs a common metadata bug found in community int8 checkpoints.

---

## 📋 Table of Contents

- [Features](#-features)
- [Which Models Work Here](#-which-models-work-here)
- [Installation](#-installation)
- [Credits](#-credits)

---

## 🎯 Features

### 🔀 Checkpoint Merge & Studio
- **Weighted Sum** (`A * (1 - M) + B * M`), **Add Difference** (`A + (B - C) * M`), and
  **No Interpolation** (single-model format conversion) — same recipes as the native
  Checkpoint Merger, but computed in float space after dequantizing, so it stays correct on
  quantized sources
- Every source model resolved through the real Forge loader and re-quantized on save through
  its own weight-setting logic, instead of touching raw tensors directly
- **Bake up to 3 LoRAs** into the merge in the same pass, each with its own strength
- **UNet Only** or **Full Checkpoint** (UNet + CLIP + VAE) save modes
- Per-component output format: diffusion model, text encoder, and VAE can each target a
  different precision
- **Device control** — Auto, force GPU, or force CPU, with a safety margin check before
  picking GPU so large merges don't silently OOM
- **Discard layers by regex** and **metadata control** — copy metadata from A/B/C individually,
  attach a full merge recipe (interpolation method, multiplier, discarded layers, baked LoRAs,
  source model hashes) matching the native Checkpoint Merger's provenance convention, or
  preview the source models' existing metadata before merging

### 🧪 LoRA → Checkpoint Baking
- Bakes LoRAs permanently into a checkpoint's weights using Forge's own LoRA-application
  pipeline — not a reimplementation — so results match what you'd get applying the LoRA live
- Trigger words from a LoRA's "Activation Text" metadata are written into the baked
  checkpoint's sidecar notes as a reminder, since baked weights still expect the same prompt
- **Anima-aware**: activation text is normalized to Anima's lowercase, space-separated
  convention, and LoRAs carrying LLM (Qwen3) adapter weights are flagged with a warning —
  Anima's own training guidance says never to train those alongside a LoRA

### 🔄 Quantize / Convert Precision
- Standalone conversion of a checkpoint's diffusion model to a new precision with no merge or
  LoRA involved
- Supported targets: **FP16**, **BF16**, **FP8** (e4m3fn / e5m2), **INT8** (tensor-wise or
  per-channel + Hadamard rotation — "convrot", the scheme most community Anima int8 builds
  actually use), **NVFP4**, and **INT4** (convrot W4A4)

### 🩺 Quant Format Doctor
- Diagnoses and repairs `.safetensors` checkpoints whose `comfy_quant` metadata is missing the
  required `format` field — the cause of `ValueError: Unknown quantization format for layer
  ...` on otherwise-valid community quantized builds
- Streams the file header only (no weights loaded into RAM/GPU) to infer the correct format
  from each layer's dtype and auxiliary keys
- Fix in place or save a repaired copy; aborts safely instead of guessing when a layer's format
  can't be inferred

---

## ⚠️ Which Models Work Here

Anything Forge Neo can load as a normal checkpoint through the standard model loader (SD1,
SDXL, Flux, Anima, etc.) in plain precision (fp16/bf16) or quantized via the
MixedPrecisionOps system (fp8/int8/nvfp4/mxfp8/convrot).

**Does not work** for Nunchaku/SVDQuant checkpoints or GGUF/nf4/fp4 storage — those store
weights in a completely different way and are rejected with a clear error instead of
producing a broken file. In a 3-model merge, A/B/C are each checked individually; mixing a
compatible model with an incompatible one is also rejected.

---

## 📦 Installation

1. Open Forge Neo WebUI
2. Go to **Extensions** → **Install from URL**
3. Paste: `https://github.com/eduardoabreu81/sd-webui-merge-studio`
4. Click **Install** and reload the WebUI
5. Open the **Merge Studio** tab

---

## 📄 Credits

- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02
- Native **Checkpoint Merger** (`modules/extras.py`) — merge recipes, metadata/provenance
  convention, and progress UI this extension follows for quantization-aware merging

---

## 📜 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">

Made with ❤️ for the Stable Diffusion community

**[Report Bug](https://github.com/eduardoabreu81/sd-webui-merge-studio/issues)** • **[Request Feature](https://github.com/eduardoabreu81/sd-webui-merge-studio/issues)**

</div>
