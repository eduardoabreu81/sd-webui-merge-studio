# Modern Full Checkpoint Components Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic, architecture-aware replacement and embedding of modern text encoders and VAEs into self-contained Full Checkpoints.

**Architecture:** A pure registry defines the component slots for each supported model family. Header inspection provides fast, conservative classification; Forge remains the runtime authority through an explicit `additional_state_dicts` preflight, and the final output must reload with an empty external-module list before it is reported as validated.

**Tech Stack:** Python 3, `unittest`, PyTorch/Forge Neo runtime APIs, safetensors headers, Gradio.

**Spec:** `docs/superpowers/specs/2026-09-17-modern-full-checkpoint-components-design.md`

## Global Constraints

- The new component selector accepts only materialized `.safetensors` files.
- GGUF, Nunchaku, SVDQ, NF4, reference-only payloads, `.ckpt`, `.pt`, `.bin`, and every non-`.safetensors` extension are out of scope.
- External text encoders and VAEs are attached after diffusion-model merge math; they are never interpolated across A/B/C.
- SD1, SDXL, Illustrious, Pony, NoobAI, and Mugen preserve the existing traditional Full Checkpoint flow.
- PiD is an i2i upscaler and must not appear in the modular component registry or modular UI.
- Anima's LLM Adapter remains part of the diffusion model and must not become a replaceable text-encoder slot.
- `Same as component source` reads the selected component's own safetensors header, never Model A's header by proxy.
- Missing source keys keep the dtype already produced by Forge; they are not guessed or force-cast.
- The new path must not silently consume `shared.opts.forge_additional_modules`.
- No new third-party dependency is permitted.
- Existing behavior outside Full Checkpoint modular composition must remain unchanged.
- Baseline on 2026-09-17: `python -m unittest discover -s tests -v` passes 47 tests.

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

- `component_registry.py`: immutable architecture, slot, prefix, support-state, and precision contracts.
- `component_bundle.py`: user selections, static validation, explicit Forge preflight, component precision, provenance, and final-output validation.
- `component_ui.py`: pure conversion between registry/bundle contracts and the fixed Gradio component rows.
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
- `tests/test_component_signatures.py`: encoder, VAE, and storage classification tests.
- `tests/test_modern_architecture_detection.py`: header and engine architecture resolution tests.
- `tests/test_component_bundle.py`: selection, compatibility, ordering, and preflight tests.
- `tests/test_component_precision.py`: per-source dtype and per-slot explicit conversion tests.
- `tests/test_full_checkpoint_validation.py`: namespace, dtype, and independent reload tests.
- `tests/test_component_ui.py`: pure UI state and control parsing tests.
- `tests/test_component_recipes.py`: recipe v2 and v1 migration tests.

### Existing test files extended

- `tests/test_aux_inspector.py`: preserve existing LoRA behavior while component classification changes.
- `tests/test_source_precision.py`: preserve diffusion/VAE prefix behavior and add role isolation.
- `tests/test_checkpoint_recipe.py`: rendered provenance and escaping.

## Forge contracts the implementation must preserve

- Forge's `replace_state_dict` owns external-module remapping and replacement.
- Forge's `split_state_dict` applies the ordered `additional_state_dicts`, then resolves `clip_target`, VAE processing, and Anima LLM Adapter movement.
- Forge model configs own `process_clip_state_dict_for_saving` and `process_vae_state_dict_for_saving`.
- Forge's engine save path serializes diffusion model, clip, and VAE through those model-config methods.

Reference sources:

- `https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/backend/loader.py`
- `https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/backend/diffusion_engine/base.py`
- `https://raw.githubusercontent.com/Haoming02/sd-webui-forge-classic/neo/modules_forge/packages/huggingface_guess/model_list.py`

---

### Task 1: Immutable architecture and slot registry

**Files:**

- Create: `component_registry.py`
- Create: `tests/test_component_registry.py`

**Interfaces:**

- Produces: `SupportState`, `ComponentSlotSpec`, `ArchitectureSpec`, `ARCHITECTURES`, `get_architecture_spec(architecture_id)`, `modular_architecture_ids()`.
- Consumes: no Forge, Gradio, or filesystem API.

- [ ] **Step 1: Write the failing registry tests**

Create table-driven tests that assert the exact approved matrix and invariants:

```python
import unittest

from component_registry import SupportState, get_architecture_spec, modular_architecture_ids


class ComponentRegistryTests(unittest.TestCase):
    def test_flux1_requires_two_encoders_and_flux_vae(self):
        spec = get_architecture_spec("flux1")
        self.assertIs(spec.support, SupportState.SUPPORTED)
        self.assertEqual(
            ("clip_l", "t5xxl", "vae_flux"),
            tuple(slot.slot_id for slot in spec.slots),
        )
        self.assertTrue(all(slot.required for slot in spec.slots))

    def test_experimental_families_are_explicit(self):
        self.assertIs(get_architecture_spec("chroma").support, SupportState.EXPERIMENTAL)
        self.assertIs(get_architecture_spec("ernie_image").support, SupportState.EXPERIMENTAL)

    def test_pid_and_traditional_families_are_not_modular_entries(self):
        ids = modular_architecture_ids()
        self.assertNotIn("pid", ids)
        self.assertNotIn("sdxl", ids)
        self.assertNotIn("mugen", ids)
```

Add one assertion for every matrix row in the spec:

```python
EXPECTED = {
    "flux1": ("clip_l", "t5xxl", "vae_flux"),
    "flux2_klein_4b": ("qwen3_4b", "vae_flux2"),
    "flux2_klein_9b": ("qwen3_8b", "vae_flux2"),
    "wan2": ("umt5xxl", "vae_wan"),
    "qwen_image": ("qwen25vl_7b", "vae_qwen_image"),
    "anima": ("qwen3_06b", "vae_qwen_image"),
    "krea2": ("qwen3vl_4b", "vae_qwen_image"),
    "zimage": ("qwen3_4b", "vae_flux"),
    "lumina2": ("gemma2_2b", "vae_flux"),
    "chroma": ("t5xxl", "vae_flux"),
    "ernie_image": ("ministral3_3b", "vae_flux2"),
}
```

- [ ] **Step 2: Run the focused tests and verify the red state**

Run: `python -m unittest tests.test_component_registry -v`

Expected: import failure because `component_registry.py` does not exist.

- [ ] **Step 3: Implement the immutable registry**

Use frozen dataclasses and string enums:

```python
from dataclasses import dataclass
from enum import Enum


class SupportState(str, Enum):
    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ComponentSlotSpec:
    slot_id: str
    label: str
    kind: str
    accepted_signatures: tuple[str, ...]
    internal_prefixes: tuple[str, ...]
    saved_prefixes: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class ArchitectureSpec:
    architecture_id: str
    label: str
    support: SupportState
    forge_model_configs: tuple[str, ...]
    slots: tuple[ComponentSlotSpec, ...]
```

Populate `ARCHITECTURES` with the exact `EXPECTED` matrix. Encoder slots accept only their matching signature. VAE slots accept only their matching VAE-family signature. Store Forge's internal role prefixes and final saved prefixes in the slot so later phases do not hard-code them in merge/UI code.

`get_architecture_spec()` must return `None` for unknown, PiD, and traditional IDs instead of manufacturing a modular spec.

- [ ] **Step 4: Run focused and full suites**

Run:

```text
python -m unittest tests.test_component_registry -v
python -m unittest discover -s tests -v
```

Expected: registry tests pass; the full suite passes with at least 47 pre-existing tests plus the new registry tests.

- [ ] **Step 5: Commit**

```text
git add component_registry.py tests/test_component_registry.py
git commit -m "feat: add modern component registry"
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

Add cases for `clip_l`, `qwen3_06b`, `qwen3_4b`, `qwen3_8b`, `qwen3vl_4b`, `qwen25vl_7b`, `gemma2_2b`, and `ministral3_3b`, using the key/shape decisions in Forge's `replace_state_dict`.

Add VAE cases that assert `latent_channels` from `decoder.conv_in.weight` and `video_capable` from temporal/video convolution markers. Add adversarial tests proving an encoder with `encoder.*` keys is not classified as a VAE.

Add storage tests for safetensors metadata/key evidence of GGUF, Nunchaku/SVDQ, and NF4. A plain FP16/BF16/FP32 component must remain supported. A `.pt` path must be rejected before header classification.

- [ ] **Step 3: Run the focused test and confirm classification failures**

Run: `python -m unittest tests.test_component_signatures tests.test_aux_inspector -v`

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
python -m unittest tests.test_component_signatures tests.test_aux_inspector -v
python -m unittest discover -s tests -v
```

- [ ] **Step 6: Commit**

```text
git add aux_inspector.py tests/safetensors_helpers.py tests/test_component_signatures.py tests/test_aux_inspector.py
git commit -m "feat: classify modern checkpoint components"
```

---

### Task 3: Stable modern architecture resolution

**Files:**

- Create: `tests/test_modern_architecture_detection.py`
- Modify: `checkpoint_inspector.py`
- Modify: `component_registry.py`
- Modify: `tests/test_component_registry.py`

**Interfaces:**

- Produces: `infer_architecture_id(header, filename="") -> str`, `architecture_id_from_engine(engine) -> str`, and `inspect_checkpoint(...)["architecture_id"]`.
- Produces: `inspect_checkpoint(...)["embedded_signatures"]` as a tuple/list of exact slot signature IDs evidenced in the file.
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

Create small fake engines whose `type(engine.model_config).__name__` matches Forge model configs. Assert mappings for `Flux`, `Flux2K4B`, `Flux2K9B`, `WAN21_T2V`, `WAN21_I2V`, `QwenImage`, `Anima`, `Krea2`, `ZImage`, `Lumina2`, `Chroma`, and `ErnieImage`. Assert `SD15`, `SDXL`, `Mugen`, and `PiD` return `not_applicable`, not a modular registry ID.

- [ ] **Step 2: Run the focused tests and verify red**

Run: `python -m unittest tests.test_modern_architecture_detection -v`

- [ ] **Step 3: Implement conservative header inference and authoritative engine mapping**

Rules:

- Prefer explicit embedded metadata and unique diffusion key/shape evidence.
- Use encoder keys only to report embedded components, never to decide the diffusion architecture.
- Filename evidence may refine an already-compatible family, but may not turn `unknown` into a supported architecture by itself.
- Return `unknown` for ambiguous Flux/Flux2, Lumina/Z-Image, or generic DiT evidence rather than selecting the wrong slot matrix.
- Runtime preflight always replaces the provisional header ID with `architecture_id_from_engine(engine)` and rejects disagreement.

Add `architecture_id` and `embedded_signatures` to `inspect_checkpoint()` without changing existing fields.

- [ ] **Step 4: Run focused regression tests**

Run:

```text
python -m unittest tests.test_modern_architecture_detection tests.test_architecture_label -v
python -m unittest discover -s tests -v
```

- [ ] **Step 5: Commit**

```text
git add checkpoint_inspector.py component_registry.py tests/test_modern_architecture_detection.py tests/test_component_registry.py
git commit -m "feat: resolve modern model architectures"
```

---

### Task 4: Pure component-plan validation

**Files:**

- Create: `component_bundle.py`
- Create: `tests/test_component_bundle.py`

**Interfaces:**

- Consumes: `ArchitectureSpec`, `inspect_module`, and checkpoint inspection dictionaries.
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
```

`build_component_plan(architecture_id, primary_path, checkpoint_info, selections, inspect_fn=inspect_module) -> ComponentPlan` is pure apart from the injected inspector.

- [ ] **Step 1: Write failing validation tests**

Cover:

- required slot missing;
- duplicate slot;
- unknown slot;
- non-`.safetensors` path;
- unsupported storage;
- signature mismatch such as Krea Qwen3-VL in Anima's Qwen3 0.6B slot;
- T5 in UMT5 slot;
- embedded selected without embedded evidence;
- exact registry order regardless of input order;
- deduplication of identical paths without hiding duplicate slot errors;
- `unknown` and traditional architectures reject modular selections;
- experimental plans retain `SupportState.EXPERIMENTAL`.

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

Run: `python -m unittest tests.test_component_bundle -v`

- [ ] **Step 3: Implement minimal immutable plan construction**

All validation errors must include the architecture label, slot label, selected basename, detected signature, and expected signatures when those values exist. Do not import Forge or Gradio in this phase.

For `source="embedded"`, store `primary_path` as the component path so source precision and provenance remain attributable to a physical safetensors file.

- [ ] **Step 4: Run focused and full suites**

Run:

```text
python -m unittest tests.test_component_bundle -v
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

- Consumes: `ComponentPlan`, Forge's `forge_loader`, and `architecture_id_from_engine`.
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
- required encoder target missing after load;
- VAE absent after load;
- loader exception wrapped with architecture, slot files, and original exception text;
- empty external list passed when every selected component is embedded;
- global Additional Modules never read.

- [ ] **Step 2: Run focused tests and verify red**

Run: `python -m unittest tests.test_component_bundle -v`

- [ ] **Step 3: Implement preflight against Forge-owned targets**

Validate `engine.model_config.clip_target` keys using the registry slot's internal prefixes/Forge target name. Validate `engine.forge_objects.clip` and `engine.forge_objects.vae` before returning.

Catch loader errors once and raise `ComponentValidationError` with contextual information. Preserve the original exception as `__cause__`.

Do not import `backend.loader` at module import time. The repository's unit suite runs outside an initialized Forge runtime, so Forge imports belong inside the `loader is None` branch only.

- [ ] **Step 4: Run focused and full suites**

Run:

```text
python -m unittest tests.test_component_bundle -v
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

- a modular Full merge interpolates only the diffusion model;
- text encoder and VAE `_merge_module_tree` calls are absent on the modular path;
- engine A loads with exactly the plan's component files;
- engines B/C load without external modules on the modular path;
- selected components are serialized from composed engine A through its `model_config`;
- Anima LLM Adapter remains assigned to diffusion output precision/namespace;
- the traditional SDXL path still executes its pre-existing CLIP/VAE merge behavior;
- `unet_only` ignores modular component controls;
- the legacy `bake_vae` path remains available only to the traditional flow.

Representative assertion:

```python
assert merge_calls == ["diffusion"]
assert engine_loads == [
    ("A.safetensors", ("qwen.safetensors", "vae.safetensors"), False),
    ("B.safetensors", (), False),
]
```

- [ ] **Step 2: Run the focused tests and verify red**

Run: `python -m unittest tests.test_modular_merge_flow -v`

- [ ] **Step 3: Add the conditional modular path**

Build and preflight the plan before expensive merge work. Reuse the preflighted engine A rather than loading it twice. Keep all legacy branches byte-for-byte equivalent where possible.

When `save_mode != "full"`, reject non-empty component selections with a clear message instead of silently baking them into a UNet-only output.

- [ ] **Step 4: Run focused and full suites**

Run:

```text
python -m unittest tests.test_modular_merge_flow -v
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
- unsupported/quantized component output formats are rejected before merge;
- embedded components use Model A only for that embedded slot;
- Anima LLM Adapter follows diffusion precision, not Qwen precision;
- provenance contains slot, basename, SHA-256, signature, source kind, physical source precision, requested output precision, and architecture support state.

- [ ] **Step 2: Run focused tests and verify red**

Run:

```text
python -m unittest tests.test_component_precision tests.test_source_precision tests.test_checkpoint_recipe -v
```

- [ ] **Step 3: Implement role-scoped matching before namespace serialization**

Apply component precision to the internal `clip_sd`/`vae_sd` produced by Forge before `process_*_state_dict_for_saving`. Strip only the selected slot's declared internal prefix when matching a standalone source header. Never use a global suffix index across components.

Restrict modular component precision choices to `same`, `fp16`, `bf16`, and `fp32`. Existing diffusion-model format choices remain unchanged.

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
python -m unittest tests.test_component_precision tests.test_source_precision tests.test_checkpoint_recipe -v
python -m unittest discover -s tests -v
```

- [ ] **Step 6: Commit**

```text
git add source_precision.py component_bundle.py checkpoint_merge.py tests/test_component_precision.py tests/test_source_precision.py tests/test_checkpoint_recipe.py
git commit -m "feat: preserve component precision and provenance"
```

---

### Task 8: Mandatory post-save validation

**Files:**

- Modify: `component_bundle.py`
- Modify: `checkpoint_merge.py`
- Create: `tests/test_full_checkpoint_validation.py`

**Interfaces:**

- Produces: `OutputValidation(validated: bool, errors: tuple[str, ...], checked_slots: tuple[str, ...])`.
- Produces: `validate_full_checkpoint_output(output_path, plan, loader=None) -> OutputValidation`. `loader=None` performs a lazy Forge import inside the function.
- Extends merge result with `validation` and `validated`.

- [ ] **Step 1: Write failing output-validation tests**

Cover:

- every required saved namespace exists;
- a required namespace missing prevents reload and returns a contextual error;
- explicit output dtype is checked from the final header;
- `same` verifies source-derived dtypes for keys that had source evidence;
- missing source keys are allowed to retain Forge dtype;
- independent reload calls `loader(output_path, additional_state_dicts=[])` exactly;
- reload failure preserves the output file and returns `validated=False`;
- success requires header, slots, dtype, and reload checks all to pass.

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

Run: `python -m unittest tests.test_full_checkpoint_validation -v`

- [ ] **Step 3: Implement validation and merge integration**

Run validation immediately after `save_checkpoint_file`. Do not delete or overwrite a failed output. Return structured errors; do not reduce them to a boolean.

Traditional and UNet-only outputs retain their current completion semantics. Mandatory independent reload applies to the new modular Full path.

- [ ] **Step 4: Run focused and full suites**

Run:

```text
python -m unittest tests.test_full_checkpoint_validation -v
python -m unittest discover -s tests -v
```

- [ ] **Step 5: Commit**

```text
git add component_bundle.py checkpoint_merge.py tests/test_full_checkpoint_validation.py
git commit -m "feat: validate full checkpoints after save"
```

---

### Task 9: Dynamic component UI without Forge-side effects

**Files:**

- Create: `component_ui.py`
- Create: `tests/test_component_ui.py`
- Modify: `scripts/merge_studio_ui.py`

**Interfaces:**

- Produces: `ComponentRowState`, `build_component_rows(checkpoint_info, save_mode, module_infos)`, and `parse_component_rows(rows)`.
- Uses three fixed Gradio rows because Flux 1 is the largest approved matrix: two text encoders plus one VAE.
- Each row carries hidden `slot_id`, visible component selector, precision selector, and status HTML.

- [ ] **Step 1: Write failing pure UI-state tests**

Assert:

- Flux 1 shows `CLIP-L`, `T5XXL`, and `Flux AE` in registry order;
- Anima shows only Qwen3 0.6B and Qwen Image VAE;
- an embedded option appears only when `embedded_signatures` proves it exists;
- a missing required component defaults to `Not selected — required` and disables execution;
- incompatible modules are absent from a row's choices;
- experimental families include a visible warning;
- traditional, unknown, PiD, and UNet-only states show no modular rows;
- parsing produces the exact `ComponentSelection` dictionaries accepted by `merge_checkpoints`.

- [ ] **Step 2: Run focused tests and verify red**

Run: `python -m unittest tests.test_component_ui -v`

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

- [ ] **Step 4: Wire the fixed Gradio rows**

In `create_merge_studio_tab()`:

- add a `Full Checkpoint Components` group with three pre-created rows;
- refresh it when Model A, Save Mode, or module inventory changes;
- resolve module display names to physical paths using Forge's `main_entry.module_list`;
- keep `Bake VAE`, global text-encoder format, and global VAE format visible only for the traditional path;
- pass parsed modular selections to `merge_handler` and then `merge_checkpoints`;
- prevent the handler from starting when a required modular row is unresolved;
- never automatically select Forge global Additional Modules.

- [ ] **Step 5: Run focused and full suites**

Run:

```text
python -m unittest tests.test_component_ui -v
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
python -m unittest tests.test_component_recipes tests.test_checkpoint_recipe tests.test_dashboard_escaping -v
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

Add a concise README section covering:

- when component slots appear;
- supported and experimental matrices;
- `.safetensors`-only rule;
- `Same as component source` semantics;
- the difference between traditional Full Checkpoint and modern modular Full Checkpoint;
- independent reload validation;
- unsupported storage formats;
- PiD exclusion.

- [ ] **Step 6: Run complete acceptance verification**

Run:

```text
python -m unittest discover -s tests -v
python -m compileall -q .
git diff --check HEAD^
```

If a Forge runtime and representative local files are available, execute one no-interpolation modular Full smoke test per supported architecture. Record exact filenames, output paths, validation results, and unavailable families. Do not claim runtime validation for a family that was not actually loaded and reopened.

- [ ] **Step 7: Commit**

```text
git add component_recipes.py scripts/merge_studio_ui.py checkpoint_inspector.py README.md tests/test_component_recipes.py tests/test_checkpoint_recipe.py tests/test_dashboard_escaping.py
git commit -m "feat: complete modular full checkpoint workflow"
```

---

## Final audit checklist

- [ ] All approved architecture rows exist and have exact slots.
- [ ] Chroma and Ernie-Image are visibly experimental.
- [ ] PiD exists in neither modular registry nor modular UI.
- [ ] Traditional model behavior is covered by regression tests.
- [ ] Only `.safetensors` component sources are accepted.
- [ ] No modular load reads global Additional Modules.
- [ ] External components never enter A/B/C interpolation.
- [ ] Component precision is sourced per physical file and isolated by slot.
- [ ] Anima LLM Adapter remains a diffusion-model concern.
- [ ] Recipe provenance includes component hashes and is safely rendered.
- [ ] Failed reload preserves the output and is never reported as success.
- [ ] Successful modular Full output reopens with `additional_state_dicts=[]`.
- [ ] The full test suite and compile check pass from a clean worktree.

## Phase-prompt release rule

The auditor generates the concrete executor prompt for Task 1 first. Each later prompt is generated only after the previous task's commit passes audit, and it includes the accepted commit hash as its baseline. This prevents later prompts from encoding assumptions invalidated by implementation findings.
