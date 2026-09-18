# Runtime verification: AIO checkpoints

Everything in the AIO feature was built and tested without a running Forge. The
suite covers the logic — which slots exist, which file fits which, what
precision survives, what namespace is expected. It cannot cover what only
happens with tensors in memory.

This document lists what is still unproven and exactly how to prove it. It
exists because the machine the work was done on has no Forge; these steps need
the one that does.

*Written 2026-09-18, against commit `c7ae9f6`. 365 tests pass.*

---

## What is already proven

- 365 automated tests, run on every commit of the work.
- Component classification checked against the 11 real component files in the
  reference install — every signature, storage format and VAE family correct.
- Architecture detection and namespace reporting checked against 222 real
  checkpoints.
- The Anima block-remap tables verified against the official expansion
  manifests shipped by Anima-2.9B and Anima-3.8B; they match position for
  position.

## What is not proven, and cannot be from here

| | Why |
|---|---|
| The merge's tensor path | `checkpoint_merge.py` imports torch, `backend` and `modules`; it is not importable outside Forge |
| The interface | `scripts/merge_studio_ui.py` additionally imports `modules.ui_components`; **it has no test coverage at all** |
| That an AIO reopens on its own | Requires a real load |
| That the LLM Adapter lands in the diffusion namespace | Requires a real save |

Nothing below may be called "validated" until it has actually been run.

---

## Test 1 — the one that justifies the whole feature

**Claim under test:** a checkpoint whose components sit under a namespace its
own architecture does not declare is *not* self-contained, even though its
header shows a text encoder and a VAE. Forge discards them and silently falls
back to Additional Modules.

This was read out of Forge's `split_state_dict`, not observed. If it is wrong,
the case for mandatory reload validation weakens considerably and Task 8 should
be revisited.

**Steps**

1. In Forge, clear **Additional Modules** completely — no text encoder, no VAE.
2. Load one of these, which carry their components under `cond_stage_model.`
   and `first_stage_model.`:
   - `Anima/aenokamikage_v01.safetensors`
   - `Anima/animaV1TurboAIO_v31.safetensors`
   - `Anima/animaMokubapoint_v90.safetensors`
3. Try to generate a single image.

**Reading the result**

- **Fails to load or generate** → the claim holds. The feature is justified and
  the reload check is doing real work.
- **Generates normally** → the claim is wrong. Forge reads those namespaces
  after all, and `validate_aio_output` is rejecting valid files. Say so; Task 8
  needs correcting before this ships.

**Control:** repeat with `Anima/royalBlendXLAnima_animaV10.safetensors`, which
uses `text_encoders.` / `vae.`. It should load and generate with Additional
Modules still empty. If *that* one also fails, the problem is the empty module
list itself, not the namespaces.

---

## Test 2 — build an AIO end to end

1. Open **Merge Studio → Checkpoint Merge & Studio**.
2. Model A: `Anima/Auralis_v4.safetensors` (28 blocks, diffusion model only).
3. Merge method: **No Interpolation**.
4. Save Mode: **AIO**.
5. Two rows should appear:
   - `Qwen3 0.6B` — pick `qwen_3_06b_base.safetensors`
   - `VAE` — pick `qwen_image_vae.safetensors`
6. Name the output and run.

**What to check**

- Both rows appeared, and only those two.
- The VAE row offered **only** the volumetric VAEs — `qwen_image_vae`,
  `wan21-vae`, `hdrVAEAnimaKrea2QWEN`. If `ae.safetensors` appears there, the
  latent-format filter is not working.
- Nothing was pre-selected.
- The result says **AIO saved and verified: it reopens on its own.**
- Load the output with Additional Modules empty, and generate.

## Test 3 — the block

Same as Test 2, but leave the VAE row empty and press Merge.

Expected: blocked before anything loads, with

> Select a VAE to save as AIO — or switch to UNet only.

## Test 4 — nothing regressed

The traditional path must be untouched. With **UNet Only**:

- no component rows appear at all;
- Bake VAE still works as before;
- the merge still reads Additional Modules, as it always did.

Then a **Full/AIO** merge of two Anima checkpoints of different generations
(28 → 40) to confirm the cross-generation remap still runs.

---

## If the interface does not even open

`scripts/merge_studio_ui.py` was written without being able to import it once.
It parses, every helper exists, no name is used before it is defined, and the
`inputs` list matches the handler's arity — but that is not the same as working.

If the tab fails to load, the error will name a line. The component work is
confined to:

- the `AIO Components` accordion and its refresh handler;
- `_inspect_installed_modules`, `_component_row_states`,
  `_refresh_component_rows`, `_parse_component_args`;
- the component block at the top of `merge_handler`.

Reverting those restores the previous interface without touching any of the
tested modules.

---

## Coverage the tests cannot give, whatever happens

The reference library is Anima end to end — 222 of 222 checkpoints. Component
files exist for Krea2, Z-Image and Flux, but no checkpoints of those families,
so even a full pass here validates **Anima only**.

Every other architecture rests on Forge's own `clip_target` declarations plus
synthetic fixtures. That is a reasonable footing, and it is not the same as
having been run. Do not report an architecture as working without loading and
reopening one.
