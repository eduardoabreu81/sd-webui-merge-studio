# Merge Studio

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Merge Studio is an all-in-one studio for Forge Neo that merges checkpoints, bakes multiple LoRAs, converts model precisions, inspects embedded recipes, and fixes quantized files.

Standard checkpoint mergers only work with raw unquantized tensors (FP16/BF16). If you try merging quantized checkpoints (FP8, INT8, ConvRot), standard mergers produce broken files or crash. Merge Studio solves this by dequantizing weights before computing merges, working with **whatever Forge Neo itself can load** through its standard checkpoint loader, and allowing you to bake up to 10 LoRAs and custom VAEs in a single pass.

---

## Table of Contents

- [Why Merge Studio?](#why-merge-studio)
- [Key Features](#key-features)
  - [1. Checkpoint Merge & Conversion](#1-checkpoint-merge--conversion)
  - [2. Save / Load Recipes](#2-save--load-recipes)
  - [3. Dynamic LoRA Baking (Up to 10 LoRAs)](#3-dynamic-lora-baking-up-to-10-loras)
  - [4. Custom VAE Baking and Stripping](#4-custom-vae-baking-and-stripping)
  - [5. Model Recipe & Inspector](#5-model-recipe--inspector)
  - [6. Quant Format Doctor](#6-quant-format-doctor)
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
- **Tracks Forge Neo, Not a Fixed List**: Compatibility is defined by Forge Neo's own loader, not by a hardcoded table here — Anima, Flux, Qwen-Image, Chroma, Z-Image, Lumina, Wan, SDXL (Pony / Illustrious) and SD1.5 all go through the same path. Architectures Forge Neo removed (**SD2** and **SD3**) are out of scope here as well.
- **Save Disk Space**: Bake Turbo LoRAs or style LoRAs directly into models, or strip unneeded VAEs to save gigabytes of storage.
- **Zero-RAM Recipe Inspector**: Check what components, base models, parent hashes, and LoRAs are inside any checkpoint in under 5 milliseconds.
- **Self-Healing**: Diagnose and repair common header metadata errors in community quantized checkpoints with one click.

---

## Key Features

### 1. Checkpoint Merge & Conversion

- **Weighted Sum (`A * (1 - M) + B * M`)**: Smoothly blend two checkpoints using an intuitive multiplier slider.
- **Add Difference (`A + (B - C) * M`)**: Extract unique stylistic or architectural differences between models B and C, and inject them into base model A.
- **No Interpolation (Format Converter)**: Convert or re-quantize a single checkpoint without merging. Switch between FP16, BF16, FP8 (e4m3fn / e5m2), and INT8.
- **Instant Present-Only Badges & Architecture Detection**: Selecting Model A, B, or C immediately names the model family from the file's own tensor keys (Anima, Illustrious, Pony, SDXL, Flux, Wan, SD1.5, with a generic `DiT / Diffusion Model` fallback for anything newer) and **only displays the components actually present in the file** (`DiT/UNet`, `Text Encoder`, `VAE`, `LLM Adapter`), keeping the interface clean with zero false-alarm "Missing" clutter.
- **Cross-Generation Anima Merging**: Anima's generations were each built by *inserting* new transformer blocks between the previous generation's (28 → 40 → 52), so a plain name-for-name merge lines up unrelated layers and yields noise. Merge Studio detects the block-count difference and remaps indices automatically, using Forge's own mapping tables — the same ones the LoRA path uses at generation time. Blocks that the expansion created have no counterpart in the older model and are kept at Model A's weights; the panel reports exactly how many were merged and how many were preserved.
- **Inserted-Block Blend (`extend_ratio`)**: The blocks a newer Anima generation inserted have no counterpart in the older model, so by default they keep Model A's weights. Raising `extend_ratio` also blends in the block each one was originally copied from. **Set this close to the Multiplier when you plan to use LoRAs built for the older generation** — Forge remaps such LoRAs onto the inserted blocks as well, so leaving those blocks unblended means a LoRA delta landing on weights it was never trained against. Experimental, and `0.0` is the right default otherwise.
- **Cross-Model Compatibility Protection**: Automatically warns if Model B or C belongs to an incompatible architecture family relative to Model A (e.g. attempting to mix Anima with SDXL or Flux), preventing corrupted merges both visually in the UI and via safety validation before engine loading.
- **Per-Component Precision**: Target different datatypes for diffusion models, text encoders, and VAEs independently.
- **Adaptive ConvRot Quantization**: Automatically selects optimal group sizes (256, 128, 64) for channel-sensitive architectures like SDXL and Illustrious to prevent shape mismatch crashes.

### 2. Save / Load Recipes

- **Repeatable Bakes**: Store every setting on the merge tab — models, method, multiplier, `extend_ratio`, precisions per component, save mode, VAE choice, metadata options, and the full LoRA list with strengths — as a single JSON file, then reload it to repeat or tweak a bake.
- **Survives Reinstalls**: Recipes are written under the WebUI's data directory when Forge exposes one, so updating or reinstalling the extension doesn't lose them.
- **Portable, and Honest About Gaps**: A recipe from another machine still loads. Fields naming a model or LoRA you don't have are left untouched rather than cleared, and the panel names exactly which ones were missing.

### 3. Dynamic LoRA Baking (Up to 10 LoRAs)

- **Dynamic Slots & Per-Row Removal**: Start with 1 slot and add more as needed with **Add LoRA** (up to 10 simultaneous LoRAs). Each row features its own dedicated red **X** button to delete that specific LoRA and automatically compact the list, plus a **Clear All LoRAs** button.
- **Native Forge Engine**: Uses Forge's native LoRA application pipeline (`networks.load_lora_for_models`) rather than external approximations, ensuring identical results to loading LoRAs at generation time — and, by the same token, supporting exactly the LoRA formats Forge Neo itself supports.
- **Mismatched LoRAs Are Skipped, Not Forced**: If more than half of a LoRA's keys don't map onto the checkpoint, Forge declines to apply it and logs `LoRA mismatch` to the console. The bake still completes and writes a valid file — just without that LoRA — so check the console when a result looks unchanged.
- **Preserved Trigger Words**: Activation text from LoRA metadata is automatically preserved in sidecar notes so you always know the required trigger words.
- **Anima & DiT Smart Warnings**: Normalizes trigger words to lowercase spacing and warns if a LoRA contains LLM (Qwen3) adapter weights that could destabilize Anima checkpoints.

### 4. Custom VAE Baking and Stripping

- **Original**: Keeps the VAE embedded in Model A.
- **None (Strip VAE)**: Completely removes the VAE from the output file, significantly reducing file size. Ideal for workflows where VAEs are loaded separately in Forge.
- **Custom VAE**: Pick any standalone VAE from your `models/VAE` folder and bake it directly into your merged checkpoint.
- **VAE Precision Casting**: Save your baked VAE in any of the output formats (FP16, BF16, FP8 e4m3fn / e5m2, INT8, NVFP4, INT4), or leave it `Same as source checkpoint`. VAEs are the most quality-sensitive component to quantize, so FP16/BF16 is the safe default.

### 5. Model Recipe & Inspector

Inspects **any** model Forge loads, not just checkpoints — pick the type and the dropdown follows.

- **Instant Header Analysis**: Reads `.safetensors` headers in <5ms with zero VRAM or RAM overhead.
- **Component Status**: Clearly identifies whether a checkpoint contains a UNet/DiT, CLIP/T5/Qwen text encoder, VAE, or LLM adapters (only showing embedded components).
- **Turbo & Anima Turbo 1.1 Detection**: Mechanically identifies whether a checkpoint contains a baked Turbo LoRA, was merged from an Anima Turbo 1.1 base model, or includes Turbo workflow nodes, highlighting lineage and bake strengths in a dedicated callout card.
- **Lineage & Recipe Detection**:
  - Automatically reads WebUI merge recipes (`sd_merge_recipe`).
  - Mechanically extracts ComfyUI workflows and prompt graphs (`UNETLoader`, `CLIPLoader`, `VAELoader`, `LoraLoaderModelOnly`).
  - Displays original base models, parent SHA-256 hashes, merge methods, and ratios.
- **Baked LoRA Discovery**: Lists any LoRAs previously baked into the checkpoint, along with bake weights and activation tags.
- **Raw Metadata Viewer**: Interactive expandable viewer for all raw metadata keys.

**LoRA inspection** answers what you actually need before baking one:

- **Trigger Word, Up Front**: Shows the activation text you must still type in the prompt — baking a LoRA never removes that need — or states plainly that none is required, which is the normal case for acceleration LoRAs.
- **Which Generation It Targets**: Reads the Anima generation (28 / 40 / 52 blocks) the same way Forge decides it at load time, and notes that Forge will remap it upward automatically.
- **Rank and Coverage**: Uniform rank or per-layer adaptive, how many of its generation's blocks it touches, and which parts of each block (self-attention, cross-attention, MLP, modulation).
- **LLM Adapter, by Magnitude**: Distinguishes a LoRA that genuinely trained Anima's LLM adapter from one that merely carries empty factors for it — which is what an SVD extraction produces for layers whose delta was zero. Only the former is flagged.
- **Trained or Extracted**: Recognises a LoRA produced by subtracting two checkpoints and shows how it was taken.

**Text encoder / VAE inspection** identifies the file by its actual layer shape rather than its filename, so you can confirm a checkpoint is paired with the encoder it was built for before merging — in Full Checkpoint mode the encoder loaded at merge time is the one baked into the output.

### 6. Quant Format Doctor

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

Merge Studio does not maintain its own compatibility list. It loads models through Forge Neo's standard checkpoint loader and operates on the resulting `MixedPrecisionOps` module tree, so **whatever Forge Neo can load, Merge Studio can open**. Whether a given operation is *useful* on that model is a separate question — the table below answers it per model. The authoritative list of what Forge Neo loads lives in the [Forge Neo README](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo#features-sep).

### Architectures

Two different questions hide behind "does it work here", and they have different answers:

- **Convert / quantize / bake LoRA** needs **one** checkpoint. The merge engine walks `named_modules()` generically and the save path uses the same `process_*_state_dict_for_saving()` hooks Forge's own `save_checkpoint()` uses, so this is architecture-agnostic.
- **Merge (Weighted Sum / Add Difference)** needs **two or three** checkpoints that share an architecture *and* a tensor shape. That is a much narrower condition, and for several Forge Neo models no second compatible checkpoint exists to merge with.

| Model | Convert / Quantize / Bake | Merge A+B | Architecture badge |
| :--- | :---: | :--- | :--- |
| SD1.5 | Yes | Yes — huge checkpoint ecosystem | `SD 1.5 / SD 2.1 (UNet)` |
| SDXL, Pony, Illustrious, Mugen | Yes | Yes — huge checkpoint ecosystem | `SDXL` / `Pony` / `Illustrious` |
| Flux, Flux Kontext, Chroma1-HD | Yes | Yes — finetunes are widely available | `Flux (MMDiT)` |
| Anima 2B / 2.9B / 3.8B | Yes | Yes, **including across generations** — 28 / 40 / 52 blocks are bridged automatically <sup>†‡</sup> | `Anima (DiT)` |

The block remapping was verified against 213 real Anima checkpoints: every one resolved to exactly 28, 40, or 52 blocks, across all three on-disk key prefixes in circulation (`model.diffusion_model.blocks.`, `net.blocks.`, and a bare `blocks.`). Every generation pair matched with **identical tensor shapes and zero mismatches** (52→40: 800 modules; 52→28: 560; 40→28: 560), with every inserted block resolving to the source it was copied from.

The mapping was then confirmed numerically against the official base checkpoints. Comparing `Anima-3.8B` (52 blocks) to `Anima-2.9B` (40 blocks) block-by-block, the best-matching source block agrees with the mapping for **all 52 targets**, with cosine similarity of **exactly 1.0000 on carried-over blocks** — they are bit-identical, confirming the frozen-stem design — and ~0.97 on inserted blocks against the block each was copied from, versus a 0.52 mean runner-up.
| Wan 2.2 (14B) | Yes | Only 14B with 14B (e.g. High Noise / Low Noise) | `Wan2.1 (DiT)` <sup>1</sup> |
| Z-Image, Krea 2, Ernie-Image | Yes | Plausible but untested — the only pairing is base + turbo | Misreported <sup>2</sup> |
| Lumina-Image-2.0 | Yes | Plausible but untested — Neta-Lumina / NetaYume-Lumina | Misreported <sup>2</sup> |
| Qwen-Image / Qwen-Image-Edit | Yes | Untested | Misreported <sup>2</sup> |
| Flux.2-Klein | Yes | **No practical pairing** — 4B and 9B are different sizes | Generic `DiT / Diffusion Model` |
| PiD 1.5 | Yes | **No** — the `sdxl` / `qwen` / `flux1` / `flux2` variants are four different transformers, and PiD is a refiner rather than a base model | Generic `DiT / Diffusion Model` |

> [!Note]
> The rows above were derived by reading Forge Neo's `model_list.py`, `loader.py`, and this extension's merge path — **not** by bench-testing every model. Rows marked *untested* are the ones where nothing in the code forbids the merge but no one has confirmed a good result. Treat "Yes" in the merge column as "the mechanics hold and compatible checkpoints exist", not as a quality guarantee.

<sup>†</sup> Merging a 28-block model into a 40- or 52-block one is supported, and the larger model must be **Primary Model (A)**. Putting the newer generation in B or C is caught twice: the badge flags it the moment you pick the model, and the merge refuses before loading any engine — collapsing a larger model onto a smaller one has no defined mapping. The inserted blocks are reported as preserved rather than as a skipped-layer warning, and `extend_ratio` optionally blends them.

<sup>1</sup> Not a typo. Forge Neo's own config class for Wan 2.2 14B is `WAN21_T2V` with `image_model: "wan2.1"` — the badge mirrors upstream naming.

<sup>‡</sup> Nested block stacks are excluded structurally, not by name. Anima checkpoints carry sub-modules with their own indexed stacks — `llm_adapter.blocks.<N>` in every generation, and `anima_v2_connector.semantic_resampler.blocks.<N>` in 3.8B v1.1's bundled Semantic Connector v2 — which a loose `.blocks.<N>.` search would happily rewrite. Only blocks at the DiT root are remapped, so a component upstream adds or renames later cannot silently start matching.

<sup>2</sup> The badge reports `Anima (DiT)` for these. Anima, Z-Image, Qwen-Image, and Krea 2 all bundle a Qwen-family text encoder under `text_encoders.`, and the detector treats any `qwen` key — or any `net.` prefix, which Lumina-style DiTs use — as an Anima signal. This affects the **label and the A-vs-B family guard only**; the merge math is untouched. UNet-only files (text encoder loaded as a separate module, the common case for these models) fall back to the generic label instead.

> [!Warning]
> The cross-model guard only blocks a mismatched Model B or C when it recognizes **both** families; an unrecognized or misreported family disables the check rather than guessing. For every DiT model below the Flux row, **you are responsible for picking a compatible B / C**. A genuine mismatch still degrades safely — mismatched layers are counted and reported as skipped rather than silently corrupting the output.

> [!Warning]
> **Anima-3.8B: match the text encoder the checkpoint was built for.** Some 52-block checkpoints are built against the native `qwen_3_06b` encoder and explicitly not against Qwen3.5-4B or the expanded adapter, because their inserted blocks were trained for that conditioning. Merge Studio loads whatever you have in **Additional Modules**, so the wrong encoder there gives you a misleading result — and in **Full Checkpoint** mode it gets baked into the output. Block remapping itself is unaffected: it never touches `llm_adapter` or the `semantic_attentions` / semantic-connector keys that Anima-3.8B v1.1+ may bundle into the DiT file.

> [!Important]
> **A UNet-only checkpoint still needs its text encoder and VAE reachable.** Forge builds the engine before Merge Studio touches anything, and engines like Z-Image, Krea 2, Qwen-Image, and Anima require `text_encoder`, `vae`, and `transformer` components to even construct — a missing one aborts with `You do not have {component} state dict!`. Merge Studio passes your configured **Settings → Additional Modules** through to the loader, so point those at the model's text encoder and VAE first. This applies **even when saving in UNet Only mode**, where neither component ends up in the output file.

> [!Important]
> **SD2 and SD3 are removed features in Forge Neo** and therefore cannot be merged here. The header inspector may still *label* an SD3 / SD3.5 (MMDiT) or SD2.1 file, because it reads `.safetensors` headers directly without the engine — but any merge, conversion, or LoRA bake will fail at load time, since Forge Neo has no loader for them.

### Precision & Quantization

- **Output formats**: FP16, BF16, FP8 (e4m3fn / e5m2), INT8 (tensor-wise single scale, or ConvRot per-channel + Hadamard rotation), NVFP4, and INT4 (ConvRot W4A4) — selectable independently for the diffusion model, text encoder, and VAE.
- **Readable inputs**: any plain FP16/BF16 checkpoint, plus Forge Neo's MixedPrecision family (`fp8_scaled`, `mxfp8`, `nvfp4`, `int8_convrot`, `convrot_w4a4`, and friends).

### Not Supported

- **Nunchaku / SVDQuant** — deprecated upstream and stores weights outside `MixedPrecisionOps`.
- **GGUF, NF4, FP4 storage** — different tensor packing; note that Forge Neo also dropped `bitsandbytes` support, which is what provided NF4/FP4 in the first place.

All of these are detected before any work starts and rejected with an explicit error instead of silently producing a broken file.

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
- Anima cross-generation block mapping follows Forge Neo's own `process_anima` tables. The approach — and the independent `expand_manifest` reconstruction that corroborates them — comes from **[ComfyUI-Anima-Remap](https://github.com/shin131002/ComfyUI-Anima-Remap)** by shin131002 (MIT).
- Adheres to standard Forge & WebUI merge conventions.
- Measurements behind these decisions, and a deferred proposal for difference extraction, are recorded in [docs/RESEARCH.md](docs/RESEARCH.md).
- Released under the [MIT License](LICENSE).

<div align="center">

Made for the Stable Diffusion and Forge Neo community.

**[Report Bug](https://github.com/eduardoabreu81/sd-webui-merge-studio/issues)** • **[Request Feature](https://github.com/eduardoabreu81/sd-webui-merge-studio/issues)**

</div>
