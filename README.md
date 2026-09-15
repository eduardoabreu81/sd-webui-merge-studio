# Merge Studio

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

A quantization-aware checkpoint merger for Forge Neo. Merges, converts, and bakes LoRAs
directly on the *dequantized* weights instead of interpolating raw tensors like the native
Checkpoint Merger — so it works correctly on quantized builds (fp8, int8, nvfp4, mxfp8,
convrot) and Anima checkpoints, not just plain fp16/bf16. Also ships an instant model recipe
and component inspector, plus a standalone doctor tool that repairs a common metadata bug
found in community int8 checkpoints.

---

## Table of Contents

- [Features](#features)
  - [Checkpoint Merge & Studio](#checkpoint-merge--studio)
  - [Quant Format Doctor](#quant-format-doctor)
  - [Model Recipe & Inspector](#model-recipe--inspector)
- [Which Models Work Here](#which-models-work-here)
- [Installation](#installation)
- [Credits](#credits)
- [License](#license)

---

## Features

### Checkpoint Merge & Studio
- **Weighted Sum** (`A * (1 - M) + B * M`), **Add Difference** (`A + (B - C) * M`), and
  **No Interpolation** (single-model format conversion) — same recipes as the native
  Checkpoint Merger, but computed in float space after dequantizing, so it stays correct on
  quantized sources
- Every source model resolved through the real Forge loader and re-quantized on save through
  its own weight-setting logic, instead of touching raw tensors directly
- **Bake multiple LoRAs dynamically** into the merge in the same pass (up to 10 with addable/removable slots), each with its own strength, using
  Forge's own LoRA-application pipeline (not a reimplementation) so results match what you'd
  get applying the LoRA live
  - Trigger words from a LoRA's "Activation Text" metadata are written into the output
    checkpoint's sidecar notes as a reminder, since baked weights still expect the same prompt
  - **Anima-aware**: activation text is normalized to Anima's lowercase, space-separated
    convention, and LoRAs carrying LLM (Qwen3) adapter weights are flagged with a warning —
    Anima's own training guidance says never to train those alongside a LoRA
- **UNet Only** or **Full Checkpoint** (UNet + CLIP + VAE) save modes
- **Per-component output format**: diffusion model, text encoder, and VAE can each target a
  different precision — **FP16**, **BF16**, **FP8** (e4m3fn / e5m2), **INT8** (tensor-wise or
  per-channel + Hadamard rotation — "convrot"), **NVFP4**, or **INT4** (convrot W4A4). Pick
  **No Interpolation** with a single model to convert or quantize a checkpoint without merging
- **Adaptive ConvRot Quantization**: automatically scales group sizes (256, 128, or 64) to
  fit architectural channel dimensions (e.g. SDXL/Illustrious 640-channel layers) or falls
  back to tensor-wise int8, avoiding quantization shape mismatches and crashes
- **Bake Custom VAE or Strip VAE**:
  - **Original**: Retains the VAE embedded in Model A
  - **None**: Strips the VAE entirely from the output checkpoint to reduce file size
  - **Custom VAE**: Select any standalone VAE from `models/VAE` to bake directly into the
    output checkpoint (even when converting a UNet-only model into a complete checkpoint)
  - Dedicated precision casting for the saved VAE (FP16, BF16, or FP8 e4m3fn)
- **Real-Time Component Badges**: dynamic indicators under Model A, B, and C selectors
  display component presence (`[UNet: Present | CLIP: Present | VAE: Present/Missing]`),
  detected architecture family, weight precision, and file size immediately upon selection
  without loading weights into memory
- **Device control** — Auto, force GPU, or force CPU, with a safety margin check before
  picking GPU so large merges don't silently OOM
- **Discard layers by regex** and **metadata control** — copy metadata from A/B/C individually,
  attach a full merge recipe (interpolation method, multiplier, discarded layers, baked LoRAs,
  source model hashes) matching the native Checkpoint Merger's provenance convention, or
  preview the source models' existing metadata before merging

### Quant Format Doctor
- Diagnoses and repairs `.safetensors` checkpoints whose `comfy_quant` metadata is missing the
  required `format` field — the cause of `ValueError: Unknown quantization format for layer
  ...` on otherwise-valid community quantized builds
- Streams the file header only (no weights loaded into RAM/GPU) to infer the correct format
  from each layer's dtype and auxiliary keys
- Fix in place or save a repaired copy; aborts safely instead of guessing when a layer's format
  can't be inferred

### Model Recipe & Inspector
- **Instant Header Inspection**: parses `.safetensors` metadata in < 5ms with zero VRAM or RAM
  allocation
- **Component Status**: verifies whether the checkpoint contains a Diffusion Model (UNet/DiT),
  Text Encoders (CLIP, T5), VAE, or LLM adapters
- **Architecture & Precision**: identifies the model family (SD1.5, SDXL, Flux, Anima, Wan2.1,
  SD3, etc.) and primary tensor datatypes
- **Merge Provenance & Lineage**: reads embedded `sd_merge_recipe` (Forge/A1111) and mechanically
  parses ComfyUI node graphs (`prompt` / `workflow`) to display base models (with SHA-256 hashes),
  merge methods, interpolation ratios, and nested merge history
- **Baked LoRA Detection**: lists any LoRAs baked into the checkpoint along with bake strengths,
  SHA-256 parent hashes, and activation tags across both WebUI merge recipes and ComfyUI pipelines
- **Raw Metadata Explorer**: interactive formatted view of all header metadata keys (e.g.
  ComfyUI workflows, quantization configurations, training configs)

---

## Which Models Work Here

Anything Forge Neo can load as a normal checkpoint through the standard model loader (SD1,
SDXL, Pony, Illustrious, Flux, Anima, Wan2.1, SD3, etc.) in plain precision (fp16/bf16) or
quantized via the MixedPrecisionOps system (fp8/int8/nvfp4/mxfp8/convrot).

**Does not work** for Nunchaku/SVDQuant checkpoints or GGUF/nf4/fp4 storage — those store
weights in a completely different way and are rejected with a clear error instead of
producing a broken file. In a 3-model merge, A/B/C are each checked individually; mixing a
compatible model with an incompatible one is also rejected.

---

## Installation

1. Open Forge Neo WebUI
2. Go to **Extensions** -> **Install from URL**
3. Paste: `https://github.com/eduardoabreu81/sd-webui-merge-studio`
4. Click **Install** and reload the WebUI
5. Open the **Merge Studio** tab

---

## Credits

- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02
- Native **Checkpoint Merger** (`modules/extras.py`) — merge recipes, metadata/provenance
  convention, and progress UI this extension follows for quantization-aware merging

---

## License

MIT — see [LICENSE](LICENSE)

---

<div align="center">

Made for the Stable Diffusion community

**[Report Bug](https://github.com/eduardoabreu81/sd-webui-merge-studio/issues)** • **[Request Feature](https://github.com/eduardoabreu81/sd-webui-merge-studio/issues)**

</div>
