# Extract LoRA from two checkpoints

Produce a LoRA from `tuned − original`. Today `lora_bake.py` only goes the other
way: it bakes an adapter *into* a checkpoint. This is the inverse, and it lets a
150 MB adapter stand in for a 6 GB checkpoint.

Status: **planned, not started.** Written 2026-09-19 against `HEAD` `690faa4`
(365 tests passing).

---

## The math

Per weight matrix, truncated SVD of the delta:

```
ΔW = W_tuned − W_original
U, S, Vh = svd(ΔW)
up   = U[:, :r] @ diag(S[:r])      # out × r
down = Vh[:r, :]                   # r × in
alpha = r                          # so the loader's alpha/rank scale is 1
```

That is the whole algorithm. `r = min(r, in_dim, out_dim)`. Compute in fp32 —
decomposing bf16 directly produces visible numerical noise — and store in bf16.

**Clamp at the 0.99 quantile.** Take `cat([U.flatten(), Vh.flatten()])`, find the
99th percentile, clamp both factors to `±hi`. Without it a single extreme
singular value becomes a visible artefact.

**Skip the near-zero deltas.** Below a floor on `mean(abs(ΔW))`, the module did
not really change and decomposing it just writes noise into the file.

**1-D tensors cannot be decomposed.** Bias and norm vectors have no rank to
truncate, and omitting them silently drops part of the delta. They are stored
raw — and the loader already has three slots for exactly this, confirmed in
ComfyUI's `load_lora` as vendored by Forge Neo: `.diff` for a weight, `.diff_b`
for a bias, `.set_weight` for a norm scale that is replaced rather than added.

## The naming question is one question, not fourteen

This is the part that decides whether the feature covers what the app promises.
The README commits to fourteen architectures. Writing a key-naming table for
each one would be the whole project, and it would rot on the next release.

We do not have to. `backend/patcher/lora.py` in Forge Neo re-exports ComfyUI's
`model_lora_keys_unet(model)` and `model_lora_keys_clip(clip)`, and those return
a **`dict[lora_key_name -> state_dict_key]` for whatever model is loaded**. Every
architecture Forge can load publishes its own LoRA naming through that map.
Inverting it gives us, for any target weight, the exact key name the loader will
look for. It is impossible to emit a name that does not load, because the map is
produced by the code that does the loading.

This is the same move kohya makes — it enumerates through its own `LoRANetwork`
rather than a table — except our source of truth is the loader that will
actually consume the file.

Two facts from that map worth knowing before writing the code:

- It always includes a **generic entry**, `key_map[k[:-len(".weight")]] = k`,
  described in the source as "generic lora format without any weird key names".
  That is why n-Arno's plain `diffusion_model.<path>` LoRAs load: the generic
  form is universal. It gives us an **offline fallback** that needs no loaded
  model, which is what keeps the tests runnable on this machine.
- Some entries map to a **slice, not a tensor**: Flux's fused QKV appears as
  `(key, (0, 0, hidden_size * 3))`. Subtracting naively by key name would
  produce an adapter that does not line up. The inverted map has to carry the
  offset or skip those modules explicitly.

## What actually varies per architecture

With naming handled, the real differences are few:

| Group | Architectures | What differs |
|---|---|---|
| UNet | SD 1.5, SDXL / Pony / Illustrious / NoobAI | **conv2d is mandatory** (3×3 and 1×1); text encoder is built in, so `model_lora_keys_clip` applies |
| DiT, external TE | Anima, Krea2, Qwen-Image / Edit, Z-Image, Flux.2-Klein 4B/9B, Wan 2.x, Lumina 2, Chroma, Ernie-Image | no conv in the blocks; TE delta only exists if both checkpoints embed the encoder |
| DiT, two TEs | Flux 1 (Dev / Schnell / Kontext) | CLIP-L **and** T5XXL; fused QKV slices in the key map |

So conv2d is back in scope — SDXL is a promise, not a maybe:
3×3 → `flatten(start_dim=1)`, 1×1 → `squeeze()`, reshape back after, with its own
`conv_dim`, because 3×3 kernels want a lower rank than the linear layers.

Anima's one special case is already handled upstream: Forge's `process_anima`
rewrites `diffusion_model.llm_adapter` → `text_encoders.qwen3_06b` and remaps
28/40/52 block counts by itself. We emit the natural names and it adapts. What we
must **not** copy from the Anima reference scripts is their hardcoded
`range(28)` — enumeration comes from the keys present, which
`anima_remap.block_count_from_keys` already does.

## What we must refuse

The app reads quantized checkpoints, and **a delta between two quantized
checkpoints is not a meaningful delta.** Subtracting INT8 or NVFP4 tensors
element-wise gives noise shaped like a tensor, and an SVD of noise is a
confident, useless file. The refusal has to be explicit and early, not a
surprise at save time.

`aux_inspector.detect_unsupported_storage` and
`checkpoint_inspector._read_quant_configs` / `_detect_precision` already identify
this. The rule: **both inputs must be unquantized floating point** (FP32 / FP16 /
BF16). Everything else is refused by name, the way the merge path already refuses
mismatched architectures.

Also refused: two checkpoints of different families, or the same family with
mismatched shapes. `get_model_family` plus a shape check on the intersected keys
covers it.

## What we already have

| Piece | Gives us |
|---|---|
| `checkpoint_inspector.read_safetensors_header` | keys and shapes without loading tensors |
| `checkpoint_inspector.get_model_family` / `infer_architecture_id` | family gate, and which group above applies |
| `checkpoint_inspector._detect_precision`, `aux_inspector.detect_unsupported_storage` | the quantization refusal |
| `anima_remap.block_count_from_keys` | Anima's real block count, 28/40/52 |
| `aux_inspector._extraction_info` | **already reads the metadata an extracted LoRA should carry** |
| `lora_bake._import_lora_networks` | the established way to reach Forge's LoRA stack from here |
| `tests/safetensors_helpers.py` | synthetic checkpoints for every family, no model files |

That fifth row settles the metadata schema — it is not ours to invent.
`_extraction_info` already looks for `format`, `mode: "svd"`, `base_prefix`,
`target_prefix`, `subtraction_dtype`, `output_dtype`, `rank`. Emit exactly those
and the existing inspector describes our output on day one.

## Why this is cheaper than the merge path

`checkpoint_merge._merge_module_tree` walks `named_modules()` — live `nn.Module`
trees, which means a loaded model, which means Forge. **Extraction does not need
that.** It is `state_dict` in, `state_dict` out: read both safetensors, intersect
the keys, subtract, decompose, write.

So the naming has two modes, and they agree:

- **offline** — emit the generic `diffusion_model.<path>` form, derived from the
  keys. Universal, per the generic entry above. No Forge.
- **with a loaded model** — invert `model_lora_keys_unet` / `_clip` and use the
  exact names, which also catches the fused-QKV slices.

Build offline first; the loaded-model path is a refinement that upgrades
precision, not a prerequisite.

And the test is **exact, not approximate**: build `orig`, build
`tuned = orig + (B @ A)` with a known rank-8 delta, extract at rank 8, assert the
recovered `up @ down` matches `B @ A` to fp32 tolerance. Truncated SVD is exact
when the target rank is at least the true rank, so there is a right answer rather
than a threshold. That holds per family, using `safetensors_helpers` to shape the
inputs — including a conv2d case for the UNet group.

## Proposed scope

- **In:** all three groups in the table. The work is the same loop; what changes
  is the conv branch and which TE map applies.
- **In:** conv2d, with its own `conv_dim`. SDXL is in the README.
- **In:** the 1-D `.diff` / `.diff_b` pass.
- **In:** the quantized-input refusal. Non-negotiable — it is the difference
  between a limitation and a bug report.
- **Out:** `.set_weight`. The loader supports it, but replacing a norm outright
  is not something a delta should decide on its own.
- **Out:** the `α·A − β·B` weighting SuperMerger offers. A scalar on the delta;
  add later if wanted.
- **Out:** dynamic rank by singular-value energy. Attractive, but it makes output
  size unpredictable and fixed-rank has to work first.

Shape: a new `lora_extract.py`, its tests, and a section on the tab — same
structure as `lora_bake.py` beside it.

## Licence

Three sources, three different permissions:

- **`kohya-ss/sd-scripts`** (`networks/extract_lora_from_models.py`) —
  **Apache-2.0, compatible with our MIT.** The `svd()` algorithm may be ported
  with the copyright notice kept.
- **ComfyUI's `comfy/lora.py`**, vendored by Forge at
  `modules_forge/packages/comfy/lora.py` — **GPL-3.0. Nothing is copied.** We
  *call* `model_lora_keys_unet` at runtime, which is ordinary use of the host's
  API, the same way `lora_bake.py` already reaches into Forge's LoRA stack.
- **`n-Arno/ANIMA_extract`** — Circlestone non-commercial licence, scripts
  "AS-IS". **Reference only.** What we take from it is factual: that Anima's TE
  is Qwen3-0.6B, that `.diff` / `.diff_b` works, that the round trip loads in
  Forge Neo.

## Open questions

**Default rank, per group.** n-Arno used 64 for Anima. kohya defaults to 4, a
training-era default far too low for a whole-checkpoint delta. A 12B Flux and a
2.6B SDXL probably do not want the same number. 64 is the informed starting
guess; worth one comparison each once it runs.

**alpha = rank, or not.** `alpha = rank` makes the loader's `alpha/rank` scale
exactly 1 and removes a class of off-by-a-factor bug. n-Arno instead used
rank 64 / alpha 32 and pre-multiplied the delta to compensate — same result, two
more places to get a factor wrong. Recommend `alpha = rank`.

**Cost on the large models.** `torch.linalg.svd` on CUDA is minutes; on CPU, tens
of minutes — and that is SDXL. Flux and Qwen-Image are several times that, on
matrices several times larger. Needs a device argument, a real progress callback,
and an honest estimate shown before the user commits to the run.

---

*Related: `2026-09-19-lora-merge.md` (LoRA + LoRA → LoRA) shares the quantile
clamp and the rank question. `2026-09-19-block-weighted-merge.md` was abandoned
as too large for the value.*
