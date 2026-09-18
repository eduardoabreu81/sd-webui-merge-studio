# Modern AIO Components Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Revised 2026-09-17** against the Forge Neo source and the installed model library.
> Seven decisions changed the design; each is recorded in the spec's revision block and
> referenced here as D1–D7. Read the spec first — several steps below were wrong and are
> now corrected.
>
> | | Decision | Main effect on this plan |
> |---|---|---|
> | D1 | Quantized text encoders accepted as bit-exact passthrough | Task 1 gains storage policy; Task 7 gains passthrough |
> | D2 | Anima 3.8B Qwen3.5 4B is out of scope | `component_providers.py` is **not created**; Task 8 shrinks |
> | D3 | One Anima policy for all generations | No overlay anywhere |
> | D4 | Slots come from Forge's `clip_target`, not a copied matrix | Task 1 shrinks substantially |
> | D5 | Inspector reports real per-architecture fields | Task 3 gains scope |
> | D6 | AIO model: two options per row, no auto-fill, blocks when incomplete | Tasks 4, 5, 9 |
> | D7 | Follow Forge's rules; a wrong-namespace checkpoint is its author's problem | Task 4/5 |
>
> **Naming:** the self-contained output is now called **AIO**. "Traditional Full Checkpoint"
> refers to SD15/SDXL, whose encoders are always embedded.

**Goal:** Build deterministic, architecture-aware replacement and embedding of modern text encoders and VAEs into self-contained AIO checkpoints.

**Architecture:** A Forge capability adapter derives component targets and serialization behavior from the loaded `model_config`; the slot list itself comes from Forge's `clip_target` and `vae_key_prefix`, never from a local matrix. A small policy registry adds only what Forge does not express: readable labels, accepted signatures, accepted storage formats, and support state. Header inspection remains a fast provisional path, Forge preflight is authoritative, and the final output must reload with an empty external-module list before it is reported as validated.

**Tech Stack:** Python 3, `unittest`, PyTorch/Forge Neo runtime APIs, safetensors headers, Gradio.

**Spec:** `docs/superpowers/specs/2026-09-17-modern-full-checkpoint-components-design.md`

## Global Constraints

- The new component selector accepts only materialized `.safetensors` files.
- GGUF, Nunchaku, SVDQ, NF4, reference-only payloads, `.ckpt`, `.pt`, `.bin`, and every non-`.safetensors` extension are out of scope.
- **`fp8_scaled` and `fp8_mixed` text encoders ARE in scope (D1).** They are Forge Neo's standard distribution format for most architectures. They are accepted as bit-exact passthrough; `weight_scale`, `weight_scale_2`, and U8 blocks are never touched.
- External text encoders and VAEs are attached after diffusion-model merge math; they are never interpolated across A/B/C.
- **The slot list comes from Forge's `clip_target` and `vae_key_prefix` (D4).** The registry must not contain a per-architecture matrix of slots.
- **There is no `modular` / `not_applicable` split (D6).** Every architecture has a required set; only its size varies. SD1, SDXL, Illustrious, Pony, NoobAI, and Mugen keep a VAE slot — what they lack is a selectable text encoder, because `cond_stage_model.` ships inside the file.
- PiD is an i2i upscaler and is excluded by `latent_format == RGB`, which is unique to it among all 18 classes — never by name.
- Anima's LLM Adapter must be saved into the diffusion-model namespace at the DiT's precision. Note that at runtime Forge's `process_anima` **moves it into the text encoder**; the existing code already compensates on save.
- Existing Anima 28/40/52 cross-generation merge support is a regression baseline, not a feature to recreate or split into three architecture entries.
- **Anima is one policy (D3).** No `anima_semantic_v2` overlay, no Qwen3.5 4B slot, no provider module (D2). The bundled Semantic Connector v2 stays part of the diffusion model, and the legacy separate adapter is never a slot.
- Forge model-config class names must never be compatibility keys. Capability-equivalent renamed or future classes must continue to work without a registry edit. Forge's own compatibility key is `huggingface_repo`; aligning to it is acceptable and preferred.
- **Nothing is auto-filled (D6).** No guessing which installed file fills which slot, and no silent adoption of `shared.opts.forge_additional_modules`.
- **AIO blocks when a declared slot has no selection (D6)**, with a message naming what is missing and offering UNet only.
- `Same as component source` reads the selected component's own safetensors header, never Model A's header by proxy.
- Missing source keys keep the dtype already produced by Forge; they are not guessed or force-cast.
- The new path must not silently consume `shared.opts.forge_additional_modules`.
- No new third-party dependency is permitted.
- Existing behavior outside AIO component composition must remain unchanged.
- Baseline on 2026-09-17: `python -m unittest discover -s tests -v` passes 47 tests.

### Running the tests in this environment

`python -m unittest tests.<module>` **does not work here.** Another project on `sys.path`
(`billsandsavings`) ships a regular `tests` package that shadows this repository's namespace
`tests/` directory, so the focused command form fails with `ModuleNotFoundError` whether or
not the module exists. Use discovery instead:

```text
python -m unittest discover -s tests -p "test_<name>.py" -v     # focused
python -m unittest discover -s tests -v                          # full suite
```

Do **not** create `tests/__init__.py` or modify `tests/conftest.py` to work around this.

---

## Delivery and audit protocol

The implementation is executed externally, one phase at a time. The auditor releases only the current phase prompt.

For every phase, the executor must:

1. Read this plan, the design spec, and every file listed in the phase.
2. Record `git status --short` and the starting `HEAD` before editing.
3. Use test-driven development: red test, focused implementation, green focused test, full suite.
4. Modify only the files authorized by the phase.
5. Commit the phase with the exact commit subject listed in the phase.
6. Return the commit hash, changed-file list, focused test output, full-suite output, and any validation limitation.
7. Stop after that commit. Do not begin the next phase.

The auditor then checks:

- the complete diff from the recorded starting commit;
- scope against the phase and design spec;
- test quality, including adversarial cases rather than implementation-only happy paths;
- type/signature compatibility with earlier phases;
- absence of unrelated changes and hidden global Forge dependencies;
- focused and full test results reproduced locally where possible.

A failed audit produces a correction prompt for the same phase. The next phase is not released until the current phase passes.

## File map

### New production files

- `forge_capabilities.py`: normalized runtime capabilities derived from the actual Forge `model_config`, independent of class name.
- `component_registry.py`: immutable policy for signatures, labels, accepted storage formats, support state, and precision contracts. **Not a slot matrix** — slots come from `clip_target` (D4).
- ~~`component_providers.py`~~: **removed (D2).** Forge discards unlisted encoder keys, so an embedded Qwen3.5 4B is a no-op; and no structural evidence decides which encoder an Anima checkpoint wants.
- `component_bundle.py`: user selections, static validation, explicit Forge preflight, component precision, provenance, and final-output validation.
- `component_ui.py`: pure conversion between capability/bundle contracts and a variable-length component-row model.
- `component_recipes.py`: pure recipe v2 serialization, validation, and v1 migration.

### Existing production files

- `checkpoint_inspector.py`: stable architecture IDs and embedded-component evidence from safetensors headers.
- `aux_inspector.py`: standalone component signatures and unsupported-storage detection.
- `source_precision.py`: role-scoped source-key matching used by each selected component.
- `checkpoint_merge.py`: conditional modern composition, serialization, metadata, and output verification.
- `scripts/merge_studio_ui.py`: Gradio rows, handlers, result messaging, and recipe v2 wiring.
- `README.md`: user-facing Full Checkpoint component workflow and limitations.

### New test files

- `tests/safetensors_helpers.py`: tiny deterministic header/state-dict fixture writers.
- `tests/test_component_registry.py`: matrix and invariant tests.
- `tests/test_forge_capabilities.py`: capability discovery, class-renaming, future-model, and incomplete-contract tests.
- `tests/test_component_signatures.py`: encoder, VAE, and storage classification tests.
- `tests/test_modern_architecture_detection.py`: header and engine architecture resolution tests.
- `tests/test_component_bundle.py`: selection, compatibility, ordering, and preflight tests.
- `tests/test_component_precision.py`: per-source dtype, passthrough, and per-slot explicit conversion tests.
- `tests/test_full_checkpoint_validation.py`: namespace, dtype, and independent reload tests.
- ~~`tests/test_component_providers.py`~~: **not created (D2).**
- `tests/test_component_ui.py`: pure UI state and control parsing tests.
- `tests/test_component_recipes.py`: recipe v2 and v1 migration tests.

### Existing test files extended

- `tests/test_aux_inspector.py`: preserve existing LoRA behavior while component classification changes.
- `tests/test_source_precision.py`: preserve diffusion/VAE prefix behavior and add role isolation.
- `tests/test_checkpoint_recipe.py`: rendered provenance and escaping.

## Forge contracts the implementation must preserve

These were read from the `neo` branch source on 2026-09-17. They are measured, not assumed.

- Forge's `replace_state_dict` owns external-module remapping and replacement, including GGUF remapping and diffusers-format conversion.
- `split_state_dict` applies the ordered `additional_state_dicts`, then resolves `clip_target`, filters by prefix, and moves the Anima LLM Adapter.
- **`guess.clip_target = guess.clip_target(sd)`** — Forge replaces the method with a resolved dict during load. The capability layer must accept both forms.
- **`clip_target` is state_dict-conditional** for `Flux`, `Chroma`, `Lumina2`, and `QwenImage`, and is evaluated **after** the additional state dicts are merged. If a component is not supplied, Forge simply **does not declare that slot** — it does not raise. Missing slots are detected by comparing against policy, never by waiting for a Forge error.
- **Anything not matching a declared target is discarded.** `split_state_dict` collects the remainder into `state_dict["ignore"]` and deletes it. This is why an embedded Qwen3.5 4B is a no-op (D2).
- **`process_clip_state_dict` strips `text_encoder_key_prefix` with `filter_keys=True`**, returning a new dict containing only what matched. The VAE is filtered by `vae_key_prefix` with no strip.
- **`process_*_state_dict_for_saving` exist on `BASE` for every architecture** and merely apply `text_encoder_key_prefix[0]` / `vae_key_prefix[0]`. Only the legacy classes and Chroma override them. Their presence is therefore **not** a discriminating capability and must not become a failure condition.
- **`process_anima` moves `llm_adapter` keys from the transformer INTO the text encoder** at load time. The existing save path already moves them back into `model.diffusion_model.*` at the DiT's dtype.
- Forge dispatches on `huggingface_repo` (`if "Anima" in guess.huggingface_repo`) and even on the filename (`"kontext" in str(sd).lower()`). Class names are not its compatibility key either.
- `model_config.unet_config`, `huggingface_repo`, `clip_target`, key prefixes, and `latent_format` are the runtime capability surface. The implementation must inspect these values, not `type(model_config).__name__`.
- Anima remapping already probes Forge's `networks.process_anima` before using local fallback tables; preserve that behavior. Note that Forge's LoRA-side `process_anima` has a 40-entry `MAPPING` and does not yet cover 52 blocks (upstream issue #1431) — the local fallback tables matter.
- The local `(28,40)` and `(40,52)` fallback tables were verified against the official `expand_manifest.json` of Anima-2.9B and the embedded `inserted_to_source` metadata of Anima-3.8B. **They are correct; do not change them.**

Reference sources:

- `https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/backend/loader.py`
- `https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/backend/diffusion_engine/base.py`
- `https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/modules_forge/packages/huggingface_guess/detection.py`
- `https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/modules_forge/packages/huggingface_guess/__init__.py`
- `https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/modules_forge/packages/huggingface_guess/model_list.py`
- `https://huggingface.co/Gazingstars123/Anima-2.9B`
- `https://huggingface.co/lylogummy/Anima-3.8B`
- `https://github.com/GumGum10/forge-anima-3.8B`

---

### Task 1: Forge capability contract and policy

> **Revised.** The slot matrix is gone (D4), the storage policy is new (D1), and the
> save-processor gate is removed. This task is substantially smaller than first written.

**Files:**

- Create: `forge_capabilities.py`
- Create: `component_registry.py`
- Create: `tests/test_forge_capabilities.py`
- Create: `tests/test_component_registry.py`

**Interfaces:**

- Produces in `forge_capabilities.py`: `ForgeCapabilityError`, `ForgeComponentTarget`, `ForgeCapabilityProfile`, `capability_profile_from_engine(engine)` and `is_anima_profile(profile)`.
- Produces in `component_registry.py`: `SupportState`, `StorageKind`, `ComponentSlotPolicy`, `SLOT_POLICIES`, `get_slot_policy(forge_target)`, and `apply_policy(profile, checkpoint_info=None)`.
- Consumes only already-instantiated fake/real engine objects. It must not import Forge, Gradio, or touch the filesystem.

**Exact field names.** Later phases consume these; do not rename them.

```python
@dataclass(frozen=True)
class ForgeComponentTarget:
    forge_target: str                      # e.g. "qwen3_06b", key of clip_target
    internal_prefixes: tuple[str, ...]     # e.g. ("qwen3_06b.transformer.",)
    kind: str                              # "text_encoder" | "vae"


@dataclass(frozen=True)
class ForgeCapabilityProfile:
    family_hint: str
    text_targets: tuple[ForgeComponentTarget, ...]
    vae_target: ForgeComponentTarget | None
    text_encoder_key_prefix: tuple[str, ...]
    vae_key_prefix: tuple[str, ...]
    latent_format_name: str
    generic_discovery: bool
    semantic_fingerprint: str
    diagnostic_fingerprint: str
```

`SLOT_POLICIES` is keyed by `forge_target` (the slot role), **not** by architecture:

```python
@dataclass(frozen=True)
class ComponentSlotPolicy:
    forge_target: str                        # "qwen3_06b", "clip_l", "t5xxl", "vae"
    label: str                               # "Qwen3 0.6B", "CLIP-L", "T5XXL"
    accepted_signatures: tuple[str, ...]
    accepted_storage: tuple[StorageKind, ...]
    support: SupportState
```

`StorageKind` values: `PLAIN`, `FP8_SCALED`, `FP8_MIXED`, `GGUF`, `NUNCHAKU_SVDQ`, `NF4`.
The first three are accepted; the last three are always rejected (D1). This vocabulary is
defined here because Task 2 consumes it.

`SupportState` values: `SUPPORTED`, `EXPERIMENTAL`, `UNKNOWN`. There is no
`NOT_APPLICABLE` — every architecture has a required set (D6).

- [ ] **Step 1: Write failing capability-discovery tests**

Build minimal fake engines whose model configs expose Forge's functional attributes. Do not name the fakes after current Forge classes.

```python
import unittest

from forge_capabilities import capability_profile_from_engine


class ForgeCapabilityTests(unittest.TestCase):
    def test_class_name_is_not_part_of_the_contract(self):
        a = fake_engine(class_name="OldForgeName", image_model="flux", targets=("clip_l", "t5xxl"))
        b = fake_engine(class_name="RenamedByForge", image_model="flux", targets=("clip_l", "t5xxl"))
        self.assertEqual(
            capability_profile_from_engine(a).semantic_fingerprint,
            capability_profile_from_engine(b).semantic_fingerprint,
        )

    def test_future_complete_model_is_discovered_without_registry_entry(self):
        engine = fake_engine(
            class_name="NovaImageV7",
            image_model="nova_image",
            targets=("nova_text",),
            text_prefix=("text_encoder.",),
            vae_prefix=("vae.",),
            with_save_processors=True,
        )
        profile = capability_profile_from_engine(engine)
        self.assertEqual(("nova_text",), tuple(t.forge_target for t in profile.text_targets))
        self.assertTrue(profile.generic_discovery)
```

Add adversarial tests proving:

- missing `clip_target`, text prefixes, or VAE evidence produces `ForgeCapabilityError` naming the missing capability;
- **the presence of `process_*_state_dict_for_saving` is NOT checked.** `BASE` provides them
  for every architecture, so gating on them would either pass vacuously or reject everything.
  A fake engine lacking the override must still produce a usable profile;
- the diagnostic fingerprint may include the class module/name, but the semantic fingerprint and compatibility decision do not;
- callable and already-resolved/dict forms of `clip_target` are normalized safely — Forge
  itself replaces the method with a dict during `split_state_dict`;
- a state_dict-conditional `clip_target` that returns fewer targets is reported as fewer
  targets, not as an error;
- `unet_config["image_model"] == "anima"` identifies Anima even if the class name changes;
- `huggingface_repo` is usable as corroborating identity evidence, since Forge dispatches on
  it itself;
- Anima block counts 28, 40, and 52 all produce the same core Anima capability family;
- an SDXL-like config produces **two text targets and a VAE target** — it is a valid profile
  with embedded encoders, not an empty one (D6);
- a `latent_format` of RGB marks the profile as non-generative (PiD) regardless of its other
  capabilities;
- no function imports Forge at module-import time.

- [ ] **Step 2: Write failing policy tests**

**There is no architecture matrix (D4).** The slots come from the fake engine's
`clip_target`; the registry is asserted per slot role. Build fake engines whose
`clip_target` mirrors what Forge actually declares, and assert the derived targets:

```python
# what Forge's model_list.py actually returns, verbatim
CLIP_TARGETS = {
    "flux":        {"clip_l": "text_encoder", "t5xxl": "text_encoder_2"},
    "flux2_k4b":   {"qwen3_4b.transformer": "text_encoder"},
    "flux2_k9b":   {"qwen3_8b.transformer": "text_encoder"},
    "chroma":      {"t5xxl": "text_encoder"},
    "lumina2":     {"gemma2_2b.transformer": "text_encoder"},
    "zimage":      {"qwen3_4b.transformer": "text_encoder"},
    "anima":       {"qwen3_06b.transformer": "text_encoder"},
    "wan21":       {"umt5xxl": "text_encoder"},
    "qwen_image":  {"qwen25_7b.transformer": "text_encoder"},   # NOT qwen25vl_7b
    "krea2":       {"qwen3vl_4b.transformer": "text_encoder"},
    "sdxl":        {"clip_l": "text_encoder", "clip_g": "text_encoder_2"},
    "sd15":        {"clip_l": "text_encoder"},
    "base":        {},
}
```

Assert the slot policies, support states, and these invariants:

- an empty `clip_target` yields zero text slots and never raises;
- SDXL and SD15 yield text targets **and** a VAE target — they are not excluded, they simply
  have their encoders embedded (D6). A profile must not be marked "no slots" for them;
- PiD is excluded by `latent_format == RGB`, not by name, and no other class carries RGB;
- the modular convention is recognised by the `text_encoders.` prefix; `cond_stage_model.`-only
  configs are the traditional convention;
- Anima is a single policy with `qwen3_06b` (D3): no `qwen35_4b` slot, no overlay, no
  provider ownership field anywhere (D2);
- `qwen3vl_4b` and `qwen3_4b` are distinct slot roles even though the underlying models are
  dimensionally identical apart from the vision tower;
- `accepted_storage` includes `FP8_SCALED` and `FP8_MIXED` for text-encoder slots, and
  excludes `GGUF`, `NUNCHAKU_SVDQ`, `NF4` everywhere (D1);
- Chroma and Ernie-Image are `EXPERIMENTAL`;
- a generic future Forge target with no local policy still yields a usable slot with a
  generic label.

- [ ] **Step 3: Run focused tests and verify the red state**

Run:

```text
python -m unittest discover -s tests -p "test_forge_capabilities.py" -v
python -m unittest discover -s tests -p "test_component_registry.py" -v
```

Expected: import failures because the new modules do not exist.

Do not use `python -m unittest tests.<module>` — see **Running the tests in this
environment** above. It fails identically whether or not the module exists, which would
make the red state meaningless.

- [ ] **Step 4: Implement capability normalization and the small policy layer**

Use frozen dataclasses and string enums, with exactly the field names given under
**Interfaces** above. Normalize:

- `clip_target` in both its callable and resolved-dict forms, splitting each key into
  `forge_target` and `internal_prefixes`;
- `text_encoder_key_prefix` and `vae_key_prefix` as tuples, preserving order — Chroma has two
  text prefixes and Mugen has two VAE prefixes;
- `latent_format` by its class name, which is what distinguishes PiD (`RGB`);
- identity from `unet_config["image_model"]` plus `huggingface_repo`;
- a semantic fingerprint that excludes the class name, and a diagnostic fingerprint that may
  include it.

**Do not read `process_*_state_dict_for_saving`.** There are no `can_save_clip` /
`can_save_vae` fields, because `BASE` supplies those methods to every architecture and they
would always be `True`.

`component_registry.py` is keyed by slot role, not by architecture. It may recognize stable
semantic evidence such as `unet_config["image_model"]`, repository hints, target names, and
checkpoint metadata. It must not contain a tuple/dict of Forge class names, and it must not
contain a per-architecture slot matrix.

The policy layer may refine a capability profile — labels, accepted signatures, accepted
storage, support state — but **may never invent a target that Forge did not declare.**

- [ ] **Step 5: Run focused and full suites**

Run:

```text
python -m unittest discover -s tests -p "test_forge_capabilities.py" -v
python -m unittest discover -s tests -p "test_component_registry.py" -v
python -m unittest discover -s tests -v
```

Expected: capability and policy tests pass; the full suite passes with **at least 47
pre-existing tests** plus the new ones. A count below 47 means something regressed.

- [ ] **Step 6: Commit**

```text
git add forge_capabilities.py component_registry.py tests/test_forge_capabilities.py tests/test_component_registry.py
git commit -m "feat: discover forge component capabilities"
```

---

### Task 2: Header-only component classification

**Files:**

- Create: `tests/safetensors_helpers.py`
- Create: `tests/test_component_signatures.py`
- Modify: `aux_inspector.py`
- Modify: `tests/test_aux_inspector.py`

**Interfaces:**

- Consumes: slot signature IDs from `component_registry.py`.
- Produces: `classify_component_header(header) -> dict`, `detect_unsupported_storage(header, filepath) -> str | None`, and extended `inspect_module(filepath)` results.
- `inspect_module()` adds `signature_id`, `latent_channels`, `video_capable`, `storage_kind`, and `supported_storage` without removing its existing keys.

- [ ] **Step 1: Add deterministic safetensors fixture helpers**

Implement:

```python
def write_safetensors_header(path, tensors, metadata=None):
    """Write a minimal valid safetensors file from {name: (dtype, shape)}."""
```

The helper calculates byte offsets from dtype widths and writes zero payload bytes. It must support `F16`, `BF16`, `F32`, `I8`, and `U8` because storage-rejection tests need integer metadata tensors.

- [ ] **Step 2: Write failing signature and rejection tests**

Cover all encoder signatures with the same discriminator Forge uses:

```python
class ComponentSignatureTests(unittest.TestCase):
    def test_t5_is_classified_before_generic_encoder_decoder_rules(self):
        header = {
            "encoder.block.0.layer.0.SelfAttention.k.weight": info("BF16", [4096, 4096]),
            "shared.weight": info("BF16", [32128, 4096]),
        }
        self.assertEqual("t5xxl", classify_component_header(header)["signature_id"])

    def test_umt5_uses_the_forge_vocab_discriminator(self):
        header = {
            "encoder.block.0.layer.0.SelfAttention.k.weight": info("BF16", [4096, 4096]),
            "shared.weight": info("BF16", [256384, 4096]),
        }
        self.assertEqual("umt5xxl", classify_component_header(header)["signature_id"])
```

Add cases for `clip_l`, `clip_g`, `qwen3_06b`, `qwen3_4b`, `qwen3_8b`, `qwen3vl_4b`,
`qwen25_7b`, `gemma2_2b`, and `ministral3_3b`, using the key/shape decisions in Forge's
`replace_state_dict`. **Note the slot is `qwen25_7b`, not `qwen25vl_7b`** — that is what
Forge's `QwenImage.clip_target` returns. There is no `qwen35_4b` case (D2).

**Critical discriminator.** Krea2's `qwen3vl_4b` and Z-Image's `qwen3_4b` are dimensionally
**identical** — 36 layers, hidden 2560, vocab 151936, 32 heads, 8 kv heads, head_dim 128,
intermediate 9728. The only structural difference is the vision tower. Measured on a real
`qwen3vl_4b_fp8_scaled.safetensors`: 315 keys matching `model.visual.*`. Test both
directions — a VL file must not fill a plain Qwen3 4B slot, and vice versa.

Add a negative fixture for the legacy `Anima-3.8B-expanded_adapter.safetensors`: it must be classified as an unsupported legacy adapter for this feature, never as a text encoder. Add a bundled-v2 fixture proving keys under `net.anima_v2_connector.` remain diffusion evidence rather than standalone component evidence.

**VAE discrimination — the first draft of this step was wrong.** Do not assert
`latent_channels` from `decoder.conv_in.weight`: the Qwen-Image, Wan, and Anima VAEs use 3D
convolution and **have no such key** (theirs is `decoder.conv_in.conv.weight`). Measured on
real files:

```
qwen_image_vae / wan21-vae      194 tensors,  8 temporal keys, NO decoder.conv_in.weight
ae.safetensors (Flux AE)        244 tensors,  0 temporal,      conv_in [512, 16, 3, 3]
SDXL VAE                        248 tensors,  0 temporal,      conv_in [512,  4, 3, 3]
```

Use this hierarchy instead:

```
3D / temporal conv        -> Qwen-Image / Wan / Anima family   (z=16)
2D conv + conv_in 16ch    -> Flux AE family
2D conv +  conv_in 4ch    -> SD / SDXL
```

Note that `qwen_image_vae` and `wan21-vae` are **indistinguishable** by these measures, and
`latent_format` does not help — Anima, QwenImage, Krea2, and WAN21 all use `Wan21`. Treat
them as one VAE family unless a finer discriminator is found; record the decision in the test.

Add adversarial tests proving an encoder with `encoder.*` keys is not classified as a VAE.

Add storage tests covering every `StorageKind` from Task 1. `PLAIN`, `FP8_SCALED`, and
`FP8_MIXED` must classify as **supported** (D1); GGUF, Nunchaku/SVDQ, and NF4 must be
rejected. Fixtures for the scaled kinds, measured from real files:

```
fp8_scaled:  F8_E4M3 weights + F32 "<name>.weight_scale"   + U8 blocks
fp8_mixed:   F8_E4M3 weights + F32 "<name>.weight_scale_2" + U8 blocks
```

A `.pt` path must be rejected before header classification.

**Filenames are not evidence.** Add a named test using
`animaLLMLayerwiseFP8_v1.safetensors` as the case: its name says FP8 and its 310 tensors are
entirely F16. Classification must report F16.

- [ ] **Step 3: Run the focused test and confirm classification failures**

Run:

```text
python -m unittest discover -s tests -p "test_component_signatures.py" -v
python -m unittest discover -s tests -p "test_aux_inspector.py" -v
```

Expected: failure because the new public classifier and fields are absent; existing LoRA tests continue to import.

- [ ] **Step 4: Implement ordered classification**

Implement classification in this order:

1. reject unsupported storage;
2. match exact bundled prefixes;
3. match T5/UMT5 keys and vocab size;
4. match Qwen/Gemma/Ministral key-and-shape signatures;
5. match CLIP-L/CLIP-G hidden size;
6. match VAE layout;
7. return `unknown` with measured evidence, never a guessed architecture name.

Do not use filename keywords as component evidence. Change the current generic `encoder.`/`decoder.` VAE branch so it runs only after encoder signatures have been excluded.

- [ ] **Step 5: Run focused and full suites**

Run:

```text
python -m unittest discover -s tests -p "test_component_signatures.py" -v
python -m unittest discover -s tests -p "test_aux_inspector.py" -v
python -m unittest discover -s tests -v
```

- [ ] **Step 6: Commit**

```text
git add aux_inspector.py tests/safetensors_helpers.py tests/test_component_signatures.py tests/test_aux_inspector.py
git commit -m "feat: classify modern checkpoint components"
```

---

### Task 3: Stable header resolution and Anima regression bridge

**Files:**

- Create: `tests/test_modern_architecture_detection.py`
- Modify: `checkpoint_inspector.py`
- Modify: `anima_remap.py`
- Modify: `forge_capabilities.py`
- Modify: `component_registry.py`
- Modify: `tests/test_component_registry.py`
- Modify: `checkpoint_merge.py` and `lora_bake.py` — remove the two remaining
  `"SDXL" in type(...).__name__` VAE-prefix fallbacks. The no-hardcode constraint applies to
  them too; derive the prefix from `vae_key_prefix[0]` instead.

**Interfaces:**

- Produces: `infer_architecture_id(header, filename="") -> str`, `capability_profile_from_engine(engine) -> ForgeCapabilityProfile`, and `inspect_checkpoint(...)["architecture_id"]` as a provisional UI hint only.
- Produces: `inspect_checkpoint(...)["embedded_components"]` — for each component evidenced
  in the file: its slot role, the **namespace prefix it sits under**, and whether that
  namespace is one the resolved architecture actually declares (D5). Presence alone is not
  enough; see Step 1.
- Changes: `inspect_checkpoint(...)["components"]` from the fixed `unet` / `clip` / `vae` /
  `llm_adapter` dict to architecture-aware fields (D5). For an Anima file, `clip: True` is
  false labelling — it holds a Qwen3 0.6B.
- Changes: `is_anima_engine(engine)` delegates to Forge capability evidence rather than the model-config class name.
- Keeps: existing human-readable `architecture` output and badge behavior.

- [ ] **Step 1: Write failing architecture tests**

Use header fixtures for every modular family and explicit negative cases:

```python
class ModernArchitectureDetectionTests(unittest.TestCase):
    def test_qwen_encoder_keys_do_not_turn_an_unknown_diffusion_model_into_anima(self):
        header = {
            "model.diffusion_model.blocks.0.weight": info("BF16", [1]),
            "text_encoders.qwen3_4b.transformer.model.layers.0.weight": info("BF16", [1]),
        }
        self.assertNotEqual("anima", infer_architecture_id(header))

    def test_llm_adapter_is_positive_anima_evidence(self):
        header = {
            "model.diffusion_model.blocks.0.weight": info("BF16", [1]),
            "model.diffusion_model.llm_adapter.proj.weight": info("BF16", [1]),
        }
        self.assertEqual("anima", infer_architecture_id(header))
```

Reuse the capability-shaped fake engines from Task 1. Assert current known families resolve through structural capability evidence, not their Python class names. Rename every fake class and prove the result is unchanged. Assert PiD resolves as non-generative via `latent_format == RGB`, and that SD15/SDXL/Mugen resolve to valid traditional profiles **with a VAE slot** (D6).

**Namespace reporting (D5).** This is the highest-value new coverage, because it is where the
real library diverges from the spec. Measured across 222 checkpoints:

```
components under text_encoders. + vae.              2 files   <- what Forge actually reads
components under cond_stage_model. + first_stage_   18 files  <- Forge discards both
VAE only, under first_stage_model.                  12 files  <- Forge discards it
no embedded components at all                      190 files
```

An example key from one of the 18: `cond_stage_model.qwen3_06b.transformer.model.embed_tokens.weight`.
The inner path matches Anima's `clip_target`; only the outer prefix differs. Assert that
`embedded_components` reports the component, its namespace, **and** that Anima declares
`text_encoders.` so this one is not readable by Forge. Do not "fix" such files (D7) — report
them truthfully and move on.

Add explicit Anima regressions:

- 28-, 40-, and 52-root-block headers all resolve to the same `anima` family while retaining their exact `block_count`;
- headers with a **bare root** (`blocks.0...`, no `net.` or `model.diffusion_model.` prefix)
  still count correctly — three such files exist in the reference library;
- `is_anima_engine` remains true after renaming the fake model-config class when `unet_config["image_model"] == "anima"`;
- the current `_probe_target_to_source`/fallback behavior remains unchanged;
- nested `llm_adapter.blocks.*` and `net.anima_v2_connector.semantic_resampler.blocks.*` never affect the root block count;
- the connector prefix is diffusion evidence and creates **no** slot and **no** overlay (D2/D3);
- a generic 52-block model with no Anima structural evidence remains unknown.

**Anima structure, measured** — use these as fixture ground truth:

| | tensors | root blocks | `llm_adapter` blocks | `anima_v2_connector` |
|---|---|---|---|---|
| Anima 2B | — | 28 | 6 | 0 |
| Anima 2.9B | — | 40 | 6 | 0 |
| Anima-3.8B v1 | 1168 | 52 | **6** | **0** |
| Anima-3.8B v1.1 | 1358 | 52 | **6** | 190 keys, 6 blocks |

The LLM Adapter is 6 blocks in **every** generation. 28 -> 40 -> 52 is pure identity-block
insertion (the v1 metadata says `architecture_expansion: "LLaMA-Pro interleaved identity
blocks"`); only v1.1 adds a new module.

**Optional improvement, aligned with the no-hardcode constraint.** The expansion map ships
inside the official files: Anima-2.9B's `expand_manifest.json` carries the `(28,40)` map, and
`Anima-3.8B.safetensors` carries `inserted_to_source` in its safetensors metadata for
`(40,52)`. Both were verified position-by-position against `_FALLBACK_TARGET_TO_SOURCE` and
**match exactly**. Reading the map from metadata when present, falling back to the table when
absent, would let a future Anima expansion work without a code edit.

- [ ] **Step 2: Run the focused tests and verify red**

Run: `python -m unittest discover -s tests -p "test_modern_architecture_detection.py" -v`

- [ ] **Step 3: Implement conservative header inference and the capability bridge**

Rules:

- Prefer explicit embedded metadata and unique diffusion key/shape evidence.
- Use encoder keys only to report embedded components, never to decide the diffusion architecture.
- Filename evidence may refine an already-compatible family, but may not turn `unknown` into a supported architecture by itself.
- Return `unknown` for ambiguous Flux/Flux2, Lumina/Z-Image, or generic DiT evidence rather than selecting the wrong slot matrix.
- Runtime preflight always replaces the provisional header decision with the capability-derived profile and rejects incompatible disagreement.
- Do not add a class-name map. Delete **all three** literal class-name checks and route them
  through capability evidence:
  - `anima_remap.py` — `type(engine.model_config).__name__ == "Anima"` becomes
    `is_anima_profile(capability_profile_from_engine(engine))`;
  - `checkpoint_merge.py` and `lora_bake.py` — `"SDXL" in type(...).__name__ or "SD1" in ...`,
    used to pick a VAE prefix when `process_vae_state_dict_for_saving` raises, becomes
    `vae_key_prefix[0]` from the profile.
- Preserve `ANIMA_BLOCK_SIZES = (28, 40, 52)`, Forge probing, fallback tables, and root-anchored exclusion exactly unless a failing regression test proves a necessary change. The tables are verified correct against upstream (see Step 1) — a change to them is a regression, not a fix.

Add `architecture_id` and `embedded_components` to `inspect_checkpoint()`, and restructure
`components` per D5. The restructure is a breaking change to that key, so update every reader:
`checkpoint_inspector.py` itself uses it at two render sites, and
`tests/test_dashboard_escaping.py` may touch it.

- [ ] **Step 4: Run focused regression tests**

Run:

```text
python -m unittest discover -s tests -p "test_modern_architecture_detection.py" -v
python -m unittest discover -s tests -p "test_architecture_label.py" -v
python -m unittest discover -s tests -p "test_llm_adapter_precision.py" -v
python -m unittest discover -s tests -v
```

- [ ] **Step 5: Commit**

```text
git add checkpoint_inspector.py anima_remap.py checkpoint_merge.py lora_bake.py forge_capabilities.py component_registry.py tests/test_modern_architecture_detection.py tests/test_component_registry.py
git commit -m "feat: resolve modern model capabilities"
```

---

### Task 4: Pure component-plan validation

**Files:**

- Create: `component_bundle.py`
- Create: `tests/test_component_bundle.py`

**Interfaces:**

- Consumes: provisional checkpoint inspection, `ComponentSlotPolicy`, `ForgeCapabilityProfile` when available, and `inspect_module`.
- Produces: `ComponentSelection`, `ResolvedComponent`, `ComponentPlan`, `ComponentValidationError`, and `build_component_plan(...)`.

Use these stable contracts:

```python
@dataclass(frozen=True)
class ComponentSelection:
    slot_id: str
    source: str                 # "embedded" or "file"
    path: str | None = None
    output_format: str = "same"


@dataclass(frozen=True)
class ResolvedComponent:
    slot_id: str
    source: str
    path: str
    signature_id: str
    output_format: str
    source_precision: str


@dataclass(frozen=True)
class ComponentPlan:
    architecture_id: str
    support: SupportState
    components: tuple[ResolvedComponent, ...]
    additional_state_dicts: tuple[str, ...]
    capability_fingerprint: str | None
    missing_slots: tuple[str, ...]      # declared slots with no selection (D6)
```

There is no `required_providers` field (D2).

`ComponentSelection.source` is `"embedded"` or `"file"` — **there is no `"none"` in AIO mode**
(D6). "AIO without an encoder" is contradictory; a user who wants no VAE uses UNet only.
Existing `bake_vae: "none"` remains available only outside the AIO path.

`build_component_plan(architecture_id, primary_path, checkpoint_info, selections, capability_profile=None, inspect_fn=inspect_module) -> ComponentPlan` is pure apart from the injected inspector. A header-only plan is provisional and must be reconciled with the loaded profile in Task 5.

- [ ] **Step 1: Write failing validation tests**

Cover:

- a declared slot with no selection lands in `missing_slots` (D6) — the plan is still built,
  so the UI can name exactly what is missing;
- duplicate slot;
- unknown slot;
- non-`.safetensors` path;
- unsupported storage (`GGUF`, `NUNCHAKU_SVDQ`, `NF4`);
- **supported storage: `fp8_scaled` and `fp8_mixed` selections are accepted** (D1);
- signature mismatch such as Krea's `qwen3vl_4b` in Anima's `qwen3_06b` slot;
- `qwen3vl_4b` rejected from a `qwen3_4b` slot and vice versa — they differ only by the vision tower;
- T5 in UMT5 slot;
- **`source="embedded"` is rejected when the engine did not load that component** (D7). The
  header saying it exists is not enough: for the 18 reference files whose components sit under
  `cond_stage_model.`, Forge loads nothing, so "keep embedded" must not be selectable;
- exact registry order regardless of input order;
- deduplication of identical paths without hiding duplicate slot errors;
- traditional architectures accept a VAE selection and reject a text-encoder selection (D6) —
  they are not rejected wholesale;
- experimental plans retain `SupportState.EXPERIMENTAL`;
- Anima 28/40/52 use the same slots, with no overlay (D3);
- selecting `qwen35_4b` or the legacy Anima 3.8B adapter is rejected — neither is a slot (D2);
- a future generic Forge target is accepted only after an authoritative capability profile exists and its selected file has a reconcilable signature;

Example:

```python
class ComponentPlanTests(unittest.TestCase):
    def test_flux_plan_orders_files_by_registry_not_form_order(self):
        selections = [
            ComponentSelection("vae_flux", "file", "ae.safetensors"),
            ComponentSelection("t5xxl", "file", "t5.safetensors"),
            ComponentSelection("clip_l", "file", "clip.safetensors"),
        ]
        plan = build_component_plan(
            "flux1", "flux.safetensors", info, selections, inspect_fn=fake_inspect
        )
        self.assertEqual(
            ("clip.safetensors", "t5.safetensors", "ae.safetensors"),
            plan.additional_state_dicts,
        )
```

- [ ] **Step 2: Run focused tests and verify red**

Run: `python -m unittest discover -s tests -p "test_component_bundle.py" -v`

- [ ] **Step 3: Implement minimal immutable plan construction**

All validation errors must include the architecture label, slot label, selected basename, detected signature, and expected signatures when those values exist. Do not import Forge or Gradio in this phase.

For `source="embedded"`, store `primary_path` as the component path so source precision and provenance remain attributable to a physical safetensors file.

- [ ] **Step 4: Run focused and full suites**

Run:

```text
python -m unittest discover -s tests -p "test_component_bundle.py" -v
python -m unittest discover -s tests -v
```

- [ ] **Step 5: Commit**

```text
git add component_bundle.py tests/test_component_bundle.py
git commit -m "feat: validate explicit component plans"
```

---

### Task 5: Explicit Forge preflight

**Files:**

- Modify: `component_bundle.py`
- Modify: `component_registry.py`
- Modify: `tests/test_component_bundle.py`

**Interfaces:**

- Consumes: `ComponentPlan`, Forge's `forge_loader`, `capability_profile_from_engine`, and provider overlays.
- Produces: `preflight_component_plan(primary_path, plan, loader=None) -> object` returning the loaded engine on success. `loader=None` performs a lazy Forge import inside the function.
- Produces: `validate_loaded_components(engine, plan) -> None`.

- [ ] **Step 1: Write failing loader-boundary tests**

Use an injected fake loader; do not import a real Forge runtime in unit tests:

```python
class ComponentPreflightTests(unittest.TestCase):
    def test_preflight_passes_only_explicit_component_paths(self):
        calls = []
        engine = fake_engine("Flux", clip_targets={"clip_l", "t5xxl"}, has_vae=True)

        def loader(path, additional_state_dicts):
            calls.append((path, tuple(additional_state_dicts)))
            return engine

        returned = preflight_component_plan("flux.safetensors", flux_plan, loader=loader)
        self.assertIs(engine, returned)
        self.assertEqual(
            [("flux.safetensors", flux_plan.additional_state_dicts)], calls
        )
```

Add failures for:

- provisional header architecture disagrees with loaded engine architecture;
- a renamed capability-equivalent model-config class succeeds;
- a complete future model config succeeds without a class-name registry entry;
- an incomplete or drifted Forge capability contract fails with the diagnostic fingerprint and missing capability;
- required encoder target missing after load;
- VAE absent after load;
- **a slot that the reloaded `clip_target` stopped declaring is reported as missing, not as a
  Forge error.** `clip_target` is state_dict-conditional in Flux, Chroma, Lumina2, and
  QwenImage, and is evaluated *after* the additional state dicts are merged — supply no
  CLIP-L to a Flux plan and Forge simply returns one fewer target, silently. This is the
  single most likely way a bad AIO slips through;
- **an engine whose text-encoder bucket came back empty because the checkpoint used a foreign
  namespace** is reported as an unloadable embedded component, not as a present one (D7);
- loader exception wrapped with architecture, slot files, and original exception text;
- empty external list passed when every selected component is embedded;
- global Additional Modules never read.

- [ ] **Step 2: Run focused tests and verify red**

Run: `python -m unittest discover -s tests -p "test_component_bundle.py" -v`

- [ ] **Step 3: Implement preflight against Forge-owned targets**

Derive a fresh `ForgeCapabilityProfile` from the loaded engine, reconcile it with the provisional policy/plan, then validate the normalized text targets and VAE target against `engine.forge_objects.clip` and `engine.forge_objects.vae`. Never dispatch from the Python class name.

Compare the reloaded target set against the plan's slots and record any that disappeared in
`missing_slots`. Forge will not raise for them.

For Anima, the profile owns `qwen3_06b` and the VAE — that is the whole set (D2/D3).

Catch loader errors once and raise `ComponentValidationError` with contextual information. Preserve the original exception as `__cause__`.

Do not import `backend.loader` at module import time. The repository's unit suite runs outside an initialized Forge runtime, so Forge imports belong inside the `loader is None` branch only.

- [ ] **Step 4: Run focused and full suites**

Run:

```text
python -m unittest discover -s tests -p "test_component_bundle.py" -v
python -m unittest discover -s tests -v
```

- [ ] **Step 5: Commit**

```text
git add component_bundle.py component_registry.py tests/test_component_bundle.py
git commit -m "feat: preflight component bundles with forge"
```

---

### Task 6: Integrate modular composition into checkpoint merge

**Files:**

- Modify: `checkpoint_merge.py`
- Modify: `component_bundle.py`
- Create: `tests/test_modular_merge_flow.py`

**Interfaces:**

- Extends: `merge_checkpoints(..., component_selections: list[dict] | None = None) -> dict`.
- Changes: `_load_engine(checkpoint_path, additional_state_dicts=None, *, use_global_modules=True)` so the traditional path can retain current behavior while the modular path always supplies an explicit list and `use_global_modules=False`.
- Produces: result keys `component_plan`, `components_attached`, and `modular_full`.

- [ ] **Step 1: Write failing orchestration tests with fakes**

Patch engine loading, `_merge_module_tree`, state-dict extraction, and file saving. Assert:

- an AIO merge interpolates only the diffusion model;
- text encoder and VAE `_merge_module_tree` calls are absent on the AIO path;
- engine A loads with exactly the plan's component files;
- engines B/C load without external modules on the AIO path;
- selected components are serialized from composed engine A through its `model_config`;
- **Anima LLM Adapter ends up in the diffusion namespace at the DiT's dtype.** Remember that
  Forge's `process_anima` put it inside the text-encoder bucket at load; the existing code
  already moves it back. Assert the outcome, not the intermediate location;
- **the LLM Adapter keys are not attributed to a selected external encoder file.** When the
  encoder slot is filled from disk, those keys still came from Model A;
- Anima 28→40, 28→52, and 40→52 diffusion merges still invoke the existing remap translator and never remap nested LLM Adapter or Semantic Connector blocks;
- all Anima generations share one policy, with no overlay and no extra slot (D2/D3);
- the traditional SDXL path still executes its pre-existing CLIP/VAE merge behavior, and still accepts a VAE selection (D6);
- `unet_only` ignores component controls;
- `bake_vae` is migrated into the VAE slot for the AIO path, and its `"none"` value survives only outside AIO (D6).

Representative assertion:

```python
assert merge_calls == ["diffusion"]
assert engine_loads == [
    ("A.safetensors", ("qwen.safetensors", "vae.safetensors"), False),
    ("B.safetensors", (), False),
]
```

- [ ] **Step 2: Run the focused tests and verify red**

Run: `python -m unittest discover -s tests -p "test_modular_merge_flow.py" -v`

- [ ] **Step 3: Add the conditional modular path**

Build and preflight the plan before expensive merge work. Reuse the preflighted engine A rather than loading it twice. Keep all legacy branches byte-for-byte equivalent where possible.

When `save_mode != "full"`, reject non-empty component selections with a clear message instead of silently baking them into a UNet-only output.

- [ ] **Step 4: Run focused and full suites**

Run:

```text
python -m unittest discover -s tests -p "test_modular_merge_flow.py" -v
python -m unittest discover -s tests -v
```

- [ ] **Step 5: Commit**

```text
git add checkpoint_merge.py component_bundle.py tests/test_modular_merge_flow.py
git commit -m "feat: compose modern full checkpoints"
```

---

### Task 7: Per-component precision and provenance

**Files:**

- Modify: `source_precision.py`
- Modify: `component_bundle.py`
- Modify: `checkpoint_merge.py`
- Create: `tests/test_component_precision.py`
- Modify: `tests/test_source_precision.py`
- Modify: `tests/test_checkpoint_recipe.py`

**Interfaces:**

- Produces: `apply_component_precision(state_dict, component, slot, dtype_map) -> int`.
- Produces: `build_component_provenance(plan) -> list[dict]`.
- Extends: `_build_metadata(..., components: list[dict] | None = None)`.
- Keeps: existing `match_source_dtypes` behavior for diffusion and traditional VAE paths.

- [ ] **Step 1: Write failing precision-isolation tests**

Test each selected component against its own physical source:

```python
class ComponentPrecisionTests(unittest.TestCase):
    def test_same_uses_each_component_header_not_model_a(self):
        clip = resolved("clip_l", "clip.safetensors", "same")
        t5 = resolved("t5xxl", "t5.safetensors", "same")
        state = {
            "clip_l.transformer.weight": FakeTensor(fp16),
            "t5xxl.transformer.encoder.weight": FakeTensor(fp16),
        }
        apply_component_precision(state, clip, clip_slot, DTYPES)
        apply_component_precision(state, t5, t5_slot, DTYPES)
        self.assertIs(bf16, state["clip_l.transformer.weight"].dtype)
        self.assertIs(fp32, state["t5xxl.transformer.encoder.weight"].dtype)
```

Add tests proving:

- missing source keys retain their current dtype;
- dtype evidence never crosses slots with identical suffixes;
- explicit `fp16`, `bf16`, and `fp32` cast only floating tensors in the selected slot;
- embedded components use Model A only for that embedded slot;
- **Anima LLM Adapter follows diffusion precision, not encoder precision — including when
  the encoder slot is filled from an external file.** Those keys arrive inside the encoder
  bucket via Forge's `process_anima` but originate in Model A, so they must never be matched
  against the external encoder's header;
- Semantic Connector v2 follows diffusion precision and is never captured by the encoder prefix;
- provenance contains slot, basename, SHA-256, signature, source kind, physical source precision, requested output precision, and architecture support state.

**Quantized component passthrough (D1)** — its own test group:

- a `fp8_scaled` component with `output_format="same"` is copied **bit-exact**: no tensor
  changes dtype, including the `F8_E4M3` weights;
- `weight_scale`, `weight_scale_2`, and `U8` tensors are never touched in any mode;
- `fp16` / `bf16` / `fp32` are **not offered** for a scaled component, and requesting one is
  rejected before merge with an actionable message — converting would require dequantizing,
  which is out of scope;
- a `PLAIN` component still honours explicit format choices as before.

Fixture shapes, measured from real Forge-distributed encoders:

```
qwen3vl_4b_fp8_scaled : 252 F8_E4M3 + 252 F32 weight_scale   + 252 U8 + 461 BF16
qwen_3_4b_fp4_mixed   : 247 F8_E4M3 + 247 F32 weight_scale_2 + 436 U8 + 151 BF16
qwen_3_06b_base       : 310 BF16 (Anima's official encoder is plain)
```

- [ ] **Step 2: Run focused tests and verify red**

Run:

```text
python -m unittest discover -s tests -p "test_component_precision.py" -v
python -m unittest discover -s tests -p "test_source_precision.py" -v
python -m unittest discover -s tests -p "test_checkpoint_recipe.py" -v
```

- [ ] **Step 3: Implement role-scoped matching before namespace serialization**

Apply component precision to the internal `clip_sd`/`vae_sd` produced by Forge before `process_*_state_dict_for_saving`. Strip only the selected slot's declared internal prefix when matching a standalone source header. Never use a global suffix index across components.

Restrict component precision choices to `same`, `fp16`, `bf16`, and `fp32` for `PLAIN`
components. For `FP8_SCALED` and `FP8_MIXED` components the only choice is `same`, and it
means bit-exact copy (D1). Existing diffusion-model format choices remain unchanged.

Calculate SHA-256 by streaming the file in fixed-size chunks. Record embedded components against the primary checkpoint path and explicit components against their selected paths.

- [ ] **Step 4: Write provenance into `sd_merge_recipe`**

Use one stable field:

```json
{
  "components": [
    {
      "slot": "qwen3_06b",
      "name": "qwen_06b.safetensors",
      "sha256": "...",
      "signature": "qwen3_06b",
      "source": "file",
      "source_precision": "BF16",
      "output_precision": "same",
      "support": "supported"
    }
  ]
}
```

- [ ] **Step 5: Run focused and full suites**

Run:

```text
python -m unittest discover -s tests -p "test_component_precision.py" -v
python -m unittest discover -s tests -p "test_source_precision.py" -v
python -m unittest discover -s tests -p "test_checkpoint_recipe.py" -v
python -m unittest discover -s tests -v
```

- [ ] **Step 6: Commit**

```text
git add source_precision.py component_bundle.py checkpoint_merge.py tests/test_component_precision.py tests/test_source_precision.py tests/test_checkpoint_recipe.py
git commit -m "feat: preserve component precision and provenance"
```

---

### Task 8: Mandatory post-save validation

> **Revised.** The provider layer is gone (D2). This task is now only about output
> validation, and it is the most important task in the plan — see the note at the end.

**Files:**

- Modify: `component_bundle.py`
- Modify: `checkpoint_merge.py`
- Create: `tests/test_full_checkpoint_validation.py`

**Interfaces:**

- Produces: `OutputValidation(validated: bool, errors: tuple[str, ...], checked_slots: tuple[str, ...])`.
- Produces: `validate_aio_output(output_path, plan, loader=None) -> OutputValidation`. `loader=None` performs a lazy Forge import inside the function.
- Extends merge result with `validation` and `validated`.

- [ ] **Step 1: Write failing output-validation tests**

Cover:

- every declared saved namespace exists;
- **the expected namespaces come from the resolved profile** — `text_encoder_key_prefix[0]`
  and `vae_key_prefix[0]` — never from a local table. A fake profile declaring an unusual
  prefix must be honoured;
- a declared namespace missing prevents reload and returns a contextual error;
- explicit output dtype is checked from the final header;
- `same` verifies source-derived dtypes for keys that had source evidence;
- a scaled component's `weight_scale` / U8 tensors are present and unchanged in the final
  header (D1);
- missing source keys are allowed to retain Forge dtype;
- independent reload calls `loader(output_path, additional_state_dicts=[])` exactly;
- reload failure preserves the output file and returns `validated=False`;
- success requires header, slots, dtype, and reload checks all to pass;
- **an output written under a foreign namespace fails validation.** Build the adversarial
  fixture from the real-world shape: components under `cond_stage_model.` /
  `first_stage_model.` while the profile declares `text_encoders.` / `vae.`. Forge drops both
  into its discarded `ignore` bucket, so the reload comes back with empty buckets.

Example:

```python
class FullCheckpointValidationTests(unittest.TestCase):
    def test_reload_never_uses_external_modules(self):
        calls = []
        result = validate_full_checkpoint_output(
            str(output), plan,
            loader=lambda path, additional_state_dicts: calls.append(
                (path, tuple(additional_state_dicts))
            ) or fake_engine,
        )
        self.assertTrue(result.validated)
        self.assertEqual([(str(output), ())], calls)
```

- [ ] **Step 2: Run focused tests and verify red**

Run: `python -m unittest discover -s tests -p "test_full_checkpoint_validation.py" -v`

- [ ] **Step 3: Implement validation and merge integration**

Run validation immediately after `save_checkpoint_file`. Do not delete or overwrite a failed output. Return structured errors; do not reduce them to a boolean.

Traditional and UNet-only outputs retain their current completion semantics. Mandatory independent reload applies to the AIO path.

**Why this task carries the feature.** In the reference library of 222 checkpoints, only
**2** embed their components under the namespaces their own architecture declares. Thirty
embed them under a foreign namespace and therefore depend, in practice, on globally
configured Additional Modules while *appearing* self-contained. Reloading with
`additional_state_dicts=[]` is the only way to tell a real AIO from one that merely looks
like one. Do not weaken this step.

- [ ] **Step 4: Run focused and full suites**

Run:

```text
python -m unittest discover -s tests -p "test_full_checkpoint_validation.py" -v
python -m unittest discover -s tests -v
```

- [ ] **Step 5: Commit**

```text
git add component_bundle.py checkpoint_merge.py tests/test_full_checkpoint_validation.py
git commit -m "feat: validate aio outputs by independent reload"
```

---

### Task 9: Component UI without Forge-side effects

> **Revised to the D6 model.** Two options per row, no auto-fill, hard block when a declared
> slot has no selection. The row set is small: the realistic maximum is three (Flux).

**Files:**

- Create: `component_ui.py`
- Create: `tests/test_component_ui.py`
- Modify: `scripts/merge_studio_ui.py`

**Interfaces:**

- Produces: `ComponentRowState`, `build_component_rows(checkpoint_info, save_mode, module_infos)`, `parse_component_rows(rows)`, and `missing_slot_message(missing) -> str`.
- Produces an ordered row model derived from the resolved profile; the slot count is not hard-coded to a local architecture matrix.
- Each rendered row carries hidden `slot_id`, visible component selector, precision selector, and status HTML.

**The row model.** Each row offers exactly **two** kinds of choice in AIO mode:

```
Text Encoder   ( ) Keep what is in the file        <- only when the ENGINE loaded it
               (•) [ pick a file from the folder ▾ ]
```

There is no `None` entry in AIO mode (D6).

- [ ] **Step 1: Write failing pure UI-state tests**

Assert:

- Flux shows `CLIP-L`, `T5XXL`, and the VAE in registry order — the only three-row case;
- an Anima checkpoint of any generation shows exactly two rows, Qwen3 0.6B and the VAE (D3);
- no row, badge, or option anywhere mentions Qwen3.5 4B or the legacy adapter (D2);
- **SDXL shows one row — the VAE.** It is not excluded from AIO; its encoders are simply
  embedded and not selectable (D6);
- UNet only shows no rows at all;
- PiD shows no rows;
- an unknown header can request Forge capability resolution and then render generic discovered rows instead of remaining permanently stale;
- **"Keep what is in the file" appears only when the engine actually loaded that component**
  (D7). Given a checkpoint whose components sit under a foreign namespace, the option is
  absent even though the header shows them — the user picks files instead;
- **nothing is pre-selected.** A freshly rendered row for a diffusion-only checkpoint has no
  default file, and the globally configured Additional Modules are neither selected nor
  suggested (D6);
- incompatible modules are absent from a row's choices;
- experimental families include a visible warning;
- a declared slot with no selection produces the exact blocking message, naming what is
  missing and offering UNet only:

```python
def test_missing_slot_message_names_what_is_missing(self):
    self.assertEqual(
        "Select a text encoder to save as AIO — or switch to UNet only.",
        missing_slot_message(("qwen3_06b",)),
    )
    self.assertEqual(
        "Select a text encoder and a VAE to save as AIO — or switch to UNet only.",
        missing_slot_message(("qwen3_06b", "vae")),
    )
```

- for Flux, the message names which of the two encoders is missing;
- parsing produces the exact `ComponentSelection` dictionaries accepted by `merge_checkpoints`.

- [ ] **Step 2: Run focused tests and verify red**

Run: `python -m unittest discover -s tests -p "test_component_ui.py" -v`

- [ ] **Step 3: Implement pure UI-state functions**

`component_ui.py` must not import Gradio or Forge. It receives already-inspected module dictionaries and returns plain dataclasses/dicts.

Use these precision choices for modular components:

```python
COMPONENT_FORMAT_CHOICES = (
    ("Same as component source", "same"),
    ("FP16", "fp16"),
    ("BF16", "bf16"),
    ("FP32", "fp32"),
)
```

- [ ] **Step 4: Wire capability-driven Gradio rows**

In `create_merge_studio_tab()`:

- add an `AIO Components` group rendered from the row state;
- rename the `Full Checkpoint` save mode to **AIO** in the UI;
- refresh the group when Model A, Save Mode, or module inventory changes;
- use fast header policy for known families and an explicit cached Forge capability probe for unknown/new families; cache by checkpoint path, size, and mtime, and surface probe failures without guessing;
- resolve module display names to physical paths using Forge's `main_entry.module_list`;
- migrate `Bake VAE` into the VAE row; keep its `none` value and the global text-encoder / VAE format controls only on the non-AIO paths;
- pass parsed selections to `merge_handler` and then `merge_checkpoints`;
- **block the handler when the plan reports `missing_slots`**, showing `missing_slot_message`;
- never automatically select Forge global Additional Modules.

**On the rendering approach.** The earlier draft claimed the Gradio dynamic-render API was
"already required by this project". It is not — `@gr.render` appears nowhere in
`scripts/merge_studio_ui.py`, which uses pre-allocated rows with visibility toggles
(`MAX_LORAS = 10`). `gr.render` *is* available in the installed Gradio 4.40.0, so either
approach works.

Prefer the codebase's existing idiom — pre-allocated rows toggled by visibility — unless a
short spike shows `@gr.render` behaves well inside a Forge tab. The realistic maximum is
three rows, and Forge persists UI state by `elem_id`, which dynamically created components
may not have. **Run that spike before committing to the dynamic API**, and record the result.
Either way, the row set is derived from the profile; a fixed ceiling in the widget pool is an
implementation detail, not a cap on the profile.

- [ ] **Step 5: Run focused and full suites**

Run:

```text
python -m unittest discover -s tests -p "test_component_ui.py" -v
python -m unittest discover -s tests -v
```

Perform a Forge UI smoke check and report whether it was possible. The minimum evidence is: tab loads, changing Model A updates rows, traditional SDXL hides rows, and a missing required slot prevents merge.

- [ ] **Step 6: Commit**

```text
git add component_ui.py scripts/merge_studio_ui.py tests/test_component_ui.py
git commit -m "feat: add full checkpoint component selectors"
```

---

### Task 10: Recipe v2, user messaging, and acceptance closure

**Files:**

- Modify: `scripts/merge_studio_ui.py`
- Modify: `checkpoint_inspector.py`
- Modify: `README.md`
- Create: `component_recipes.py`
- Create: `tests/test_component_recipes.py`
- Modify: `tests/test_checkpoint_recipe.py`
- Modify: `tests/test_dashboard_escaping.py`

**Interfaces:**

- Changes: `RECIPE_VERSION = 2`.
- Produces: `serialize_component_recipe(rows)`, `restore_component_recipe(recipe, checkpoint_info, module_infos)`, and `migrate_v1_components(settings, checkpoint_info, module_infos)` in the pure `component_recipes.py` module.
- Recipe v2 adds top-level `components` entries with `slot_id`, source kind, installed module name/path reference, and output format.
- Keeps: recipe v1 loading and all existing LoRA fields.

- [ ] **Step 1: Write failing recipe and rendering tests**

Cover:

- v2 round-trip of all modular component rows;
- a missing local component is reported and left unresolved rather than silently replaced;
- v1 traditional `bake_vae` loads unchanged;
- v1 modular recipe with an explicit Bake VAE maps that VAE into the architecture's VAE slot when compatible, while required text encoders remain visibly unresolved;
- no v1 recipe invents a text encoder;
- component provenance shown by the checkpoint dashboard is labeled as recipe provenance, not fresh inference;
- every component filename, slot label, precision, and error is HTML-escaped.

- [ ] **Step 2: Run focused tests and verify red**

Run:

```text
python -m unittest discover -s tests -p "test_component_recipes.py" -v
python -m unittest discover -s tests -p "test_checkpoint_recipe.py" -v
python -m unittest discover -s tests -p "test_dashboard_escaping.py" -v
```

- [ ] **Step 3: Implement recipe v2 and migration**

Keep `RECIPE_FIELDS` backward-compatible for scalar controls. Implement serialization and migration in `component_recipes.py`, which must not import Gradio or Forge. Store modular rows in the top-level `components` list rather than adding a variable number of positional Gradio fields to `settings`.

On load, validate installed module names against the current Forge module inventory. Missing files remain `Not selected — required` and are included in the warning summary.

- [ ] **Step 4: Implement completion messaging**

The merge handler must:

- call `gr.Info` only when `result["validated"]` is true for a modular Full output;
- call `gr.Warning` and render `Not validated` plus all structured validation errors when reload fails;
- show the attached component slots and source basenames;
- show an `Experimental architecture` warning for Chroma and Ernie-Image;
- keep the saved file path visible in both validated and non-validated outcomes.

- [ ] **Step 5: Document the user workflow**

Documentation is a first-class deliverable of this work, not a footnote. Write it **after**
the behaviour is implemented and tested, and lead with the walkthrough rather than the rules
— the two-screenshot form in the spec's §7.1 is the model to follow.

Add a README section covering:

- what an AIO is, and how it differs from a traditional Full Checkpoint;
- the walkthrough: pick a checkpoint, see the rows, fill them, what happens if you do not;
- **how many components each architecture needs** — reproduce the table from the spec's §5.
  The rule is 1 text encoder + 1 VAE; Flux is the only exception, with 2;
- where the component files must live (`models/text_encoder/`, `models/VAE/`);
- `.safetensors`-only rule and the accepted storage formats, including that `fp8_scaled`
  encoders are fine;
- `Same as component source` semantics, and that a quantized component is copied as-is;
- independent reload validation, and what "not validated" means when it appears;
- that a checkpoint may look like it has embedded components while its namespace makes them
  unreadable — with the note that this is a property of the file, not of this tool;
- existing Anima 28/40/52 merge support versus the new component layer;
- that Anima uses `qwen3_06b` for every generation, and that Qwen3.5 4B is not embeddable.

- [ ] **Step 6: Run complete acceptance verification**

Run:

```text
python -m unittest discover -s tests -v
python -m compileall -q .
git diff --check HEAD^
```

If a Forge runtime and representative local files are available, execute one no-interpolation AIO smoke test per supported architecture. Record exact filenames, output paths, validation results, and unavailable families.

**Be honest about coverage.** The available reference library is essentially Anima — 222 of
222 checkpoints — plus component files for Krea2, Z-Image and Flux but no checkpoints of
those families. End-to-end runtime validation is therefore only possible for Anima. Every
other architecture is covered by synthetic fixtures. **Do not claim runtime validation for a
family that was not actually loaded and reopened.**

One runtime check is worth more than the rest and should be done first: take a checkpoint
whose components sit under a foreign namespace, clear Additional Modules entirely, and try to
load it. If it fails to generate, the whole feature is justified and the reload-based
validation is proven necessary.

- [ ] **Step 7: Commit**

```text
git add component_recipes.py scripts/merge_studio_ui.py checkpoint_inspector.py README.md tests/test_component_recipes.py tests/test_checkpoint_recipe.py tests/test_dashboard_escaping.py
git commit -m "feat: complete modular full checkpoint workflow"
```

---

## Final audit checklist

- [ ] The slot list comes from Forge's `clip_target`; the registry contains no per-architecture slot matrix.
- [ ] No Forge class name appears as a compatibility key anywhere, including the two former `"SDXL" in type(...).__name__` fallbacks.
- [ ] A capability-equivalent renamed Forge class passes without a policy edit.
- [ ] A future complete fake Forge model config is discovered generically; an incomplete one fails with an actionable reason.
- [ ] The presence of `process_*_state_dict_for_saving` is never used as a gate.
- [ ] Chroma and Ernie-Image are visibly experimental.
- [ ] PiD is excluded by `latent_format == RGB`, not by name.
- [ ] SD15 and SDXL keep a VAE slot; they are not excluded from AIO.
- [ ] Traditional model behavior is covered by regression tests.
- [ ] Only `.safetensors` component sources are accepted.
- [ ] `fp8_scaled` and `fp8_mixed` encoders are accepted and copied bit-exact; `weight_scale` and U8 tensors are untouched.
- [ ] No AIO load reads global Additional Modules, and nothing is auto-filled.
- [ ] External components never enter A/B/C interpolation.
- [ ] Component precision is sourced per physical file and isolated by slot.
- [ ] Anima LLM Adapter is saved into the diffusion namespace at the DiT dtype, even when the encoder slot is filled from an external file.
- [ ] Existing Anima 28/40/52 remapping remains green, with the fallback tables unchanged, under a single Anima policy.
- [ ] No slot, overlay, provider, or UI element for Qwen3.5 4B or the legacy Anima adapter exists.
- [ ] `component_providers.py` and `tests/test_component_providers.py` were never created.
- [ ] A declared slot with no selection blocks AIO with a message naming it and offering UNet only.
- [ ] "Keep what is in the file" is offered only when the engine loaded the component, never from header evidence alone.
- [ ] The inspector reports components per architecture, with namespace and readability.
- [ ] Recipe provenance includes component hashes and is safely rendered.
- [ ] Failed reload preserves the output and is never reported as success.
- [ ] A successful AIO output reopens with `additional_state_dicts=[]`.
- [ ] Every runtime-validation claim names the architecture that was actually loaded and reopened.
- [ ] The full test suite and compile check pass from a clean worktree, with at least 47 pre-existing tests still green.

## Phase-prompt release rule

The auditor generates the concrete executor prompt for Task 1 first. Each later prompt is generated only after the previous task's commit passes audit, and it includes the accepted commit hash as its baseline. This prevents later prompts from encoding assumptions invalidated by implementation findings.
