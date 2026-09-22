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
- [AIO Checkpoints](#-aio-checkpoints)
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

Six ways to combine models. Each one asks only for the models and sliders it
actually needs, so the form changes with your choice instead of showing fields
that do nothing.

| Mode | What it does | Models |
| :--- | :--- | :--- |
| **No Interpolation** | Passes one model straight through — for converting, re-quantizing, or rebuilding a single file. | A |
| **Weighted Sum** | Blends two models. The multiplier decides how much of each. | A + B |
| **Add Difference** | Takes what B learned on top of C and adds it to A. C has to be the model B was trained from. | A + B + C |
| **Sum Twice** | Blends A with B, then blends that result with C. Three models in one pass. | A + B + C |
| **Similarity Add Difference** | Add Difference that holds back where A and B already agree, and applies the full difference only where they disagree. | A + B + C |
| **DARE** | Keeps a random share of the difference and strengthens what survives — a lighter touch that still moves the model. | A + B |

- Choose precision separately for the diffusion model, text encoder, and VAE.
- Build a self-contained **AIO** checkpoint by choosing its text encoder and VAE explicitly.
- Merge compatible Anima generations with automatic block mapping.
- Weight some layers differently from the rest of the merge (Anima).
- See the detected architecture and embedded components before processing.

#### Weighting some layers differently (Anima)

A merge normally applies the same multiplier to the whole model. For Anima
checkpoints you can override that for a range of layers — which is how you keep
one model's composition while taking another's style.

Write one rule per line, naming the layers, the parts of them it applies to, and
the weight:

```text
L05-L09:self_attn.q_proj self_attn.k_proj:0.08
```

Anything no rule names merges at the multiplier, and where two rules overlap the
later one wins. The editor shows which layers each rule ends up touching, so you
can check it before merging, and rules can be imported from a checkpoint that
already carries them.

### 🧩 LoRA & VAE Baking

- Bake up to **10 LoRAs** into one checkpoint.
- Set an individual strength for each LoRA.
- Keep the original VAE, remove it, or bake a custom VAE.
- Preserve available LoRA trigger declarations in the saved model recipe.

### 🔗 LoRA Merge

Combine up to six LoRAs into a single adapter, on its own tab.

- Give each LoRA its own weight; a negative weight subtracts it.
- **Exact by default.** Adding the tensors of two adapters does not add their
  effects, so the factors are stacked instead — the result reproduces every
  input exactly, at a rank that is the sum of the input ranks.
- Set a target rank to compress that back down when the file matters more than
  the last decimal.
- Plain **LoRA / LoCon** only. LoHa, LoKr, OFT, GLoRA and DoRA store their
  change in a form that does not stack; they are refused by name, with the
  reason, rather than merged into something that loads and is wrong.
- LoRAs targeting different Anima generations are refused: their block indices
  do not line up, and the cross-generation remap is a checkpoint feature.

### 🧱 AIO Checkpoints

Modern models keep their text encoder and VAE in separate files. An **AIO** is a
checkpoint that carries its own, so it loads without anything configured in
Forge's Additional Modules.

Pick a model, choose **AIO** as the save mode, and a row appears for each
component that model needs:

```text
Text Encoder   ( ) Keep what is in the file
               (•) [ qwen_3_06b_base.safetensors  ▾ ]
VAE            (•) [ qwen_image_vae.safetensors   ▾ ]
```

- **How many rows** comes from the model. Anima needs one encoder and a VAE;
  Flux 1 needs two encoders and a VAE; SDXL needs only a VAE, because its
  encoders are always inside the checkpoint.
- **Keep what is in the file** appears only when the checkpoint already carries
  that component *and* Forge can actually read it.
- **Nothing is filled in for you.** The Additional Modules configured in Forge
  are never used here — what goes into the file is what you picked.
- **Every row has to be filled.** Leave one empty and the merge is blocked with
  a message naming it. If you did not want a self-contained file, use **UNet
  Only**.

Component files are read from `models/text_encoder/` and `models/VAE/`, the same
places Forge loads them from.

#### What each architecture needs

The rule is **one text encoder and one VAE**. Flux 1 is the only exception.

| Architecture | Text encoder(s) | VAE |
| :--- | :--- | :--- |
| **Flux 1** (Dev / Schnell / Kontext) | **2** — CLIP-L + T5XXL | ae |
| SDXL / Pony / Illustrious / NoobAI | built in, not selectable | sdxl-vae |
| SD 1.5 | built in, not selectable | vae-ft-mse |
| Anima (all generations) | Qwen3 0.6B | Qwen-Image VAE |
| Krea2 | Qwen3-VL 4B | Qwen-Image VAE |
| Qwen-Image / Edit | Qwen2.5-VL 7B | Qwen-Image VAE |
| Z-Image / Turbo | Qwen3 4B | ae |
| Flux.2-Klein 4B | Qwen3 4B | flux2-vae |
| Flux.2-Klein 9B | Qwen3 8B | flux2-vae |
| Wan 2.x | UMT5XXL | wan vae |
| Lumina Image 2 | Gemma2 2B | ae |
| Chroma | T5XXL | ae |
| Ernie-Image | Ministral3 3B | flux2-vae |

Despite the shared name, **Flux 2 Klein takes one encoder, not two.**

Only `.safetensors` components can be embedded. Encoders distributed as
`fp8_scaled` or `fp8mixed` are fine and are copied exactly as they are — they
cannot be converted to another precision, because that would mean unpacking the
quantization. GGUF, Nunchaku/SVDQ and NF4 cannot go into a checkpoint at all.

**Same as component source** reads the precision from the component file you
picked, not from the model you are merging.

#### Verified, or clearly not

After saving, Merge Studio reopens the file **with no external modules at all**.
Only then is it reported as an AIO.

This matters more than it sounds. A checkpoint can contain a perfectly good
encoder stored under a namespace its own architecture never reads — Forge
ignores it and quietly falls back to whatever is configured, so the file looks
self-contained and is not. Reopening is the only way to tell the two apart.

If the file does not reopen on its own it is **kept**, marked **Not validated**,
and the reasons are listed. It is still a usable checkpoint; it just needs its
external modules, like before.

The Model Recipe & Inspector reports the same thing for any checkpoint: which
components are inside, which namespace they use, and whether this architecture
will read them.

### 💾 Save & Load Recipes

- Save the complete merge setup as a JSON recipe.
- Restore models, merge settings, precision choices, VAE options, and LoRA selections.
- Keep recipes inside the extension, in its own `recipes/` folder.
- Record the components of an AIO, so the same composition can be rebuilt. A
  component that is not installed on the machine loading the recipe is
  reported and left empty, never swapped for something similar.
- Load portable recipes even when some referenced files are not installed locally.

### 🔎 Model Recipe & Inspector

- Inspect checkpoints, LoRAs, text encoders, and VAEs.
- Identify model architecture and embedded components.
- Read merge lineage, parent hashes, baked LoRAs, and available trigger declarations.
- Read merges made elsewhere. A checkpoint merged in another tool — or built in
  ComfyUI — records its own method, proportions and per-layer rules; Merge Studio
  reads those back, explains the method, and names the tool that wrote them
  instead of showing an empty recipe.
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
3. Pick a merge mode and fill the models it asks for — two for **Weighted Sum**
   and **DARE**, three for **Add Difference**, **Sum Twice** and **Similarity Add
   Difference**.
4. Set the output name, multiplier, precision, and save mode.
5. Click **Merge / Process**.

### Build an AIO (self-contained) Checkpoint

1. Select the model in **Model A**.
2. Set **Save Mode** to **AIO**.
3. Fill every component row that appears — pick a file, or keep what the
   checkpoint already has.
4. Click **Merge / Process**.

The result is reported as an AIO only after it reopens with no external modules.

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
| Weight layers individually | Anima checkpoints |
| Read a merge made in another tool | Recipes written by other mergers, and ComfyUI workflows |
| Merge LoRAs into one LoRA | Plain LoRA / LoCon adapters of the same architecture |
| Cross-generation Anima merge | Supported when the newer/larger model is Model A |
| Inspect safetensors metadata | Checkpoints, LoRAs, text encoders, and VAEs |
| Build an AIO checkpoint | Any architecture Forge Neo declares components for; verified by reopening |

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
- For cross-generation Anima merges, place the newer model with more blocks in **Model A**. The **inserted-block blend** slider and the **write rule** beside it appear only when Model A and Model B are Anima checkpoints of two different generations, because that is the only case where inserted blocks exist.
- The newer generation's extra blocks have no partner in the older model, so there are two ways to write them and you pick one. **Blend** mixes each extra block with the older block it grew out of — simple, but the two sit at different depths, so you get an average of two models inside one block. **Delta** adds only the distance Model B travelled away from its own neighbour, and never Model B's raw weights. Blend is the original behaviour and stays the default. At a blend of 0 neither runs and the extra blocks are left exactly as Model A had them.
- **The same number means very different things under the two rules.** Blend travels all the way to a block from another depth, so it moves a lot. Delta only adds how far Model B drifted from its own neighbour, and two finetunes of the same family are close relatives — so at the same setting Delta changes the block roughly sixteen times less. A good starting point for Delta is **0.35**, with the Multiplier at **0.0** so the shared blocks stay exactly as Model A published them. If a Delta merge looks like almost nothing happened, that is the rule's scale, not a failure — raise the number rather than assuming it did not run.
- **DARE** drops part of the difference at random, so it is the one mode whose result depends on a seed. The seed is saved in the merge recipe and shown again when the finished checkpoint is inspected: the same seed, models and ratios reproduce the same file. A DARE merge made by another tool that records no seed cannot be reproduced exactly, and the inspector says so.
- **Anima 3.8B v1.1 is not a safe model to merge here.** It carries an extra Semantic Connector that Forge Neo has no support for, so the connector is dropped when the file is loaded — before any merge begins — and the saved result would be missing it. Use the plain Anima 3.8B (the one that pairs with the native Qwen 0.6B text encoder) instead. Nothing warns about this yet.
- Some models store the text encoder and VAE separately. Make sure the required files are selected under Forge Neo's **Additional Modules**, even when saving only the diffusion model.
- A checkpoint that appears to contain a text encoder or VAE does not always load with it: if the components sit under a namespace the architecture does not read, Forge ignores them. The inspector says when that is the case.
- If Forge reports a LoRA mismatch in the console, the checkpoint can still be saved, but that LoRA may have been skipped.
- Quantizing a VAE can affect image quality more noticeably than quantizing the diffusion model. FP16 or BF16 is the safest choice for the VAE.
- Always keep the original models until you have tested the generated checkpoint.

---

## 📚 Technical Documentation

The README is intentionally focused on installation and everyday use. Architecture notes, model-specific behavior, merge methodology, quantization details, and advanced troubleshooting belong in the **[GitHub Wiki](https://github.com/eduardoabreu81/sd-webui-merge-studio/wiki)**.

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

<div align="center">

Made for the Stable Diffusion and Forge Neo community.

**[Report Bug](https://github.com/eduardoabreu81/sd-webui-merge-studio/issues)** • **[Request Feature](https://github.com/eduardoabreu81/sd-webui-merge-studio/issues)** • **[Wiki](https://github.com/eduardoabreu81/sd-webui-merge-studio/wiki)**

</div>
