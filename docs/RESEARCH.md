# Research notes

Measurements behind decisions in this extension, kept because they were
expensive to obtain and are not derivable from the code. Dated so a future
reader can tell what may have gone stale.

*Last updated: 2026-09-15*

---

## Anima generation structure

Each generation was built by **inserting** blocks into the previous one, not by
appending: 28 → 40 → 52. Index `N` therefore means a different layer in each
generation, which is why a name-for-name merge across generations produces
noise and why `anima_remap.py` exists.

**Verified against the official base checkpoints.** Comparing
`lylogummy/Anima-3.8B` (52 blocks) with `Gazingstars123/Anima-2.9B` (40 blocks)
block-by-block on `self_attn.q_proj.weight`:

| | result |
| :--- | :--- |
| best-matching source block agrees with the mapping | **52 / 52** |
| cosine on carried-over blocks | **1.0000** (min, not mean — bit-identical) |
| cosine on inserted blocks vs the block each was copied from | ~0.97 |
| mean runner-up | 0.52 |

The carried-over blocks being bit-identical confirms the "frozen stem" design
stated on community model cards. Inserted blocks sit just below their source,
the signature of LLaMA-Pro-style "copy an adjacent block, then fine-tune".

This is a third independent derivation of the same mapping, after Forge Neo's
hardcoded tables in `networks.py::process_anima` and the reconstructed
manifests in [ComfyUI-Anima-Remap](https://github.com/shin131002/ComfyUI-Anima-Remap).
All three agree exactly, including the 40→52 step, for which no official
mapping has ever been published.

### Key layouts seen in the wild

Three root prefixes are in circulation, all accepted by Forge's loader
(`backend/loader.py::preprocess_state_dict`):

| prefix | files (of 212 surveyed) |
| :--- | ---: |
| `model.diffusion_model.blocks.` | 176 |
| `net.blocks.` | 33 |
| `blocks.` (bare) | 3 |

Two sub-modules carry their **own** indexed block stacks and must never be
remapped:

- `llm_adapter.blocks.<N>` — present in every generation (118 tensors)
- `anima_v2_connector.semantic_resampler.blocks.<N>` — Anima-3.8B v1.1's
  bundled "Semantic Connector v2" (190 tensors, 84 of them matching a loose
  `.blocks.<N>.` search)

`anima_remap.py` excludes both by anchoring to the DiT root rather than
matching a list of known sub-module names. The name list came first and was
already wrong — it guessed `semantic_connector`, and the real prefix is
`anima_v2_connector`.

---

## What the turbo distillation actually changes

Delta between `anima-base-v1.0` and `anima-turbo-v1.1`, as `|dW| / |W|`,
sampled across blocks 0 / 13 / 27:

| module | relative delta |
| :--- | ---: |
| `adaln_modulation_mlp.2` | 0.054 |
| `self_attn.q_proj` | 0.026 |
| `mlp.layer1` | 0.017 |
| `cross_attn.k_proj` | 0.013 |
| `self_attn.output_proj` | 0.012 |
| `cross_attn.q_proj` | 0.009 |
| `mlp.layer2` | 0.004 |
| `adaln_modulation_self_attn.1` | 0.003 |
| **`llm_adapter`** | **0.000** |

The adapter is untouched by turbo training. That is why an SVD extraction of
this pair produces llm_adapter factors that are pure numerical noise, and why
`lora_bake.py::_lora_touches_llm_adapter` tests magnitude rather than the mere
presence of the keys.

### Ancestry

`anima-turbo-v1.1` is closest to `anima-base-v1.0`, confirming it as the right
`C` for an Add Difference or an extraction:

| pair | distance |
| :--- | ---: |
| turbo-v1.1 → base-v1.0 | **0.00665** |
| turbo-v1.1 → aesthetic-v1.1 | 0.00810 |
| turbo-v1.1 → turbo-v1.0 | 0.01123 |

### The delta is not low-rank

Singular value spectrum of `W_turbo − W_base`, as the fraction of delta energy
captured at a given rank:

| layer | r32 | r64 | r128 | rank for 95% |
| :--- | ---: | ---: | ---: | ---: |
| `blocks.13.self_attn.q_proj` | 67.5% | 83.5% | 93.0% | 231 |
| `blocks.13.mlp.layer1` | 70.8% | 85.0% | 93.5% | 229 |
| `blocks.13.cross_attn.k_proj` | 66.4% | 85.2% | 95.9% | 116 |
| `blocks.13.adaln_modulation_mlp.2` | **93.7%** | 96.9% | 98.6% | **40** |

Turbo concentrates a compact change in the modulation layers and spreads a
diffuse one through attention and MLP. A uniform rank-32 extraction discards
roughly a third of the delta energy — though energy is not the same thing as
perceptual quality, and the spectrum tail is often noise.

**Consequence for merging:** if the goal is a checkpoint, Add Difference
(`A + M × (turbo − base)`) applies the delta at full rank with no truncation
at all. Extracting a LoRA and baking it is a lossy detour. The LoRA is worth it
only for portability and per-generation strength.

---

## Deferred: Extract Difference

Not implemented. Recorded so the reasoning does not have to be rebuilt.

**What it would be.** The same `A − B` the merge path already computes, but
instead of adding it into a checkpoint, decomposing it by SVD and writing a
LoRA. Same dequantization, same module-tree walk, different output.

**Why it belongs here rather than in a trainer.** Extraction is arithmetic, not
training. A LoRA trainer learns a *new* concept from a dataset; extraction
repackages a difference that already exists between two files. Turbo in
particular cannot be trained that way — step distillation needs a
teacher/student setup and usually an adversarial loss, which is a research-scale
run, not a one-page trainer. Every public Anima turbo LoRA is an extraction.

**Why doing it through the Forge engine matters.** Loading both sides via
`forge_loader` normalises root prefix (`net.` vs `model.diffusion_model.`) and
storage dtype (BF16 / F16 / INT8) into one module tree. Both were real
stumbling blocks when working on raw state dicts — the existing
`anima-turbo-v11-rank32-bias` carries `base_prefix` / `target_prefix` metadata
precisely to work around the first.

**Uses beyond turbo**, in rough order of value:

1. **Cross-generation transport.** Of 212 Anima checkpoints surveyed, 208 are
   28-block, 4 are 40-block, none are 52-block. Extracting a style from a 2B
   finetune lets Forge's own LoRA remapping carry it onto 40- and 52-block
   models, which is otherwise impossible.
2. **Disk.** The local library is 864 GB across 220 checkpoints (3.93 GB mean).
   In a 16-file sample, 9 sat within 0.05 of the base — compact enough to
   express as a LoRA at roughly 1/14th the size.
3. **Adjustable strength.** A merge fixes its ratio at merge time.
4. **Auditing.** For checkpoints with no `sd_merge_recipe`, the delta against a
   base is the only way to see what actually changed, and where.

**Design notes if it gets built.** Fixed rank first, per-module-type rank later
(the spectrum table above is the recipe: ~32 for `adaln_modulation`, 128+ for
attention and MLP). A relative-magnitude threshold that skips unchanged layers
excludes `llm_adapter` on its own, with no exception list — the same reasoning
that led to structural anchoring in `anima_remap.py`.

**What it does not do.** It creates no new capability. It only repackages a
difference that already exists between two checkpoints.

---

## Method notes

- **HTTP Range beats downloading.** Civitai's signed URLs and HuggingFace
  `resolve/` URLs both honour `Range`, so a `.safetensors` header (~80–185 KB)
  and individual tensors can be pulled from a multi-GB remote file. The whole
  investigation above moved well under 30 MB.
- **Check dtype before decoding.** Anima checkpoints ship BF16, F16 and F32.
  Decoding an F16 file as BF16 silently yields garbage — it produced a
  meaningless "distance 1.00000" here before being caught.
- **Validate the instrument before trusting a negative.** A cosine comparison
  against a *merged* checkpoint suggested the 40→52 mapping was wrong. Running
  the same procedure against the known-good 28→40 mapping returned 40/40, which
  localised the fault to the specimen: merged models blend blocks, so
  correspondence has to be measured against the base.
