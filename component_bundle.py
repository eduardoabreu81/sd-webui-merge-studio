"""Turning selected component files into a plan, or into a clear refusal.

This is the layer that gives AIO its meaning: every slot the architecture
declares has to be filled, from a file or from what the checkpoint already
carries. A slot left empty is reported rather than raised, because the
interface needs to name exactly what is missing and offer UNet only instead.

Everything here is pure apart from the injected inspector. No Forge, no engine,
no filesystem -- the loader boundary is a separate concern, and this module has
to stay testable outside a running Forge.

The slot list is the architecture's, never ours. When a capability profile is
available it is authoritative; without one the header's architecture id gives a
provisional guess, good enough to render a form and never good enough to save
on. Task 5's preflight replaces it with the real thing.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

from checkpoint_inspector import read_safetensors_header
from component_registry import SupportState, apply_policy, get_slot_policy
from forge_capabilities import ForgeCapabilityError

#: Only what a header can pin down on its own. Everything else waits for the
#: capability probe -- see `checkpoint_inspector.infer_architecture_id`, which
#: deliberately returns "unknown" for anything ambiguous.
#:
#: This is a hint for rendering, not a compatibility table. The moment an
#: engine is loaded, its `clip_target` replaces whatever is guessed here.
PROVISIONAL_SLOTS: dict[str, tuple[str, ...]] = {
    "anima": ("qwen3_06b", "vae"),
    # Traditional families keep their VAE slot; their encoders ship inside the
    # checkpoint and are not selectable.
    "sdxl": ("vae",),
    "sd15": ("vae",),
}
# Every id here must be one Forge can actually load. SD3 is deliberately
# absent: its headers are recognisable, but Forge Neo ships no model config for
# it, so there is no engine to preflight against and no AIO to build.

#: A component is either picked from a folder or kept from the checkpoint.
#: There is no "none" in an AIO plan: an AIO without an encoder is a
#: contradiction, and UNet only already covers wanting a leaner file.
VALID_SOURCES = ("file", "embedded")


class ComponentValidationError(Exception):
    """A selection cannot be turned into a plan, with the reason a user needs."""


@dataclass(frozen=True)
class ComponentSelection:
    slot_id: str
    source: str
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
    #: The namespace this component occupies inside the engine. Carried on
    #: the component so precision matching can be scoped to it without
    #: reaching back for the capability profile.
    internal_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComponentPlan:
    architecture_id: str
    support: SupportState
    components: tuple[ResolvedComponent, ...] = ()
    additional_state_dicts: tuple[str, ...] = ()
    capability_fingerprint: str | None = None
    #: Slots the architecture declares that nobody filled. The user needs to
    #: pick a file for each.
    missing_slots: tuple[str, ...] = ()
    #: Slots that *were* filled and that the reloaded engine no longer
    #: declares -- the selected file was handed to Forge and did not take.
    #: Kept apart from `missing_slots` because "select a text encoder" is the
    #: wrong advice for someone who already did.
    dropped_slots: tuple[str, ...] = ()
    #: The namespaces this architecture writes its components under, taken
    #: from the loaded engine. Output validation checks against these rather
    #: than a table of its own -- a future architecture must not need an edit
    #: here to be validated.
    text_namespace: str = ""
    vae_namespace: str = ""
    slot_order: tuple[str, ...] = field(default=())

    @property
    def is_complete(self) -> bool:
        """Whether this composition can be saved as an AIO."""
        return not self.missing_slots and not self.dropped_slots


def _declared_slots(architecture_id: str, capability_profile):
    """``(selectable_slots, embedded_only_slots, support)``.

    With a profile these come from Forge. Without one they are the provisional
    guess, and anything the header could not identify offers nothing at all --
    better an empty form than the wrong slots.
    """
    if capability_profile is not None:
        policy = apply_policy(capability_profile)
        selectable = tuple(s.forge_target for s in policy.slots if not s.embedded_only)
        embedded_only = tuple(s.forge_target for s in policy.slots if s.embedded_only)
        prefixes = {s.forge_target: s.internal_prefixes for s in policy.slots}
        return selectable, embedded_only, policy.support, prefixes

    slots = PROVISIONAL_SLOTS.get(architecture_id)
    if slots is None:
        return (), (), SupportState.UNKNOWN, {}
    # Without an engine the namespaces are unknown too; precision matching
    # waits for the preflight rather than guessing them.
    return slots, (), SupportState.SUPPORTED, {}


def _readable_embedded(checkpoint_info) -> dict[str, dict]:
    """Embedded components keyed by role, as the inspector reported them."""
    found = {}
    for component in (checkpoint_info or {}).get("embedded_components") or ():
        role = component.get("role")
        if role:
            found[role] = component
    return found


def _label(slot_id: str) -> str:
    return get_slot_policy(slot_id).label


def _resolve_file(selection, slot_id, inspect_fn, internal_prefixes=()):
    path = selection.path
    if not path:
        raise ComponentValidationError(
            f"{_label(slot_id)}: no file was selected."
        )
    if not str(path).lower().endswith(".safetensors"):
        suffix = os.path.splitext(str(path))[1] or "(no extension)"
        raise ComponentValidationError(
            f"{_label(slot_id)}: only .safetensors components can be embedded, "
            f"but {os.path.basename(path)} is {suffix}."
        )

    info = inspect_fn(path) or {}
    if "error" in info:
        raise ComponentValidationError(
            f"{_label(slot_id)}: could not read {os.path.basename(path)} "
            f"({info['error']})."
        )

    storage = info.get("storage_kind", "plain")
    if not info.get("supported_storage", True):
        raise ComponentValidationError(
            f"{_label(slot_id)}: {os.path.basename(path)} is stored as {storage}, "
            "which cannot be embedded into a safetensors checkpoint."
        )

    # A component stored with scales can only be copied as it is. Converting it
    # means dequantizing, which this path does not do -- and casting the scale
    # tensors alongside the weights would quietly wreck the component.
    if storage != "plain" and selection.output_format != "same":
        raise ComponentValidationError(
            f"{_label(slot_id)}: {os.path.basename(path)} is stored as {storage}, "
            f"so it can only be embedded as-is. {selection.output_format!r} would "
            "require dequantizing it first, which this path does not do."
        )

    signature = info.get("signature_id")
    accepted = get_slot_policy(slot_id).accepted_signatures
    if signature is None:
        raise ComponentValidationError(
            f"{_label(slot_id)}: {os.path.basename(path)} does not match any "
            f"component this tool recognises, so it cannot be assigned to a slot. "
            f"Expected one of: {', '.join(accepted)}."
        )
    if signature not in accepted:
        raise ComponentValidationError(
            f"{_label(slot_id)}: {os.path.basename(path)} is a {signature}, "
            f"but this slot expected {', '.join(accepted)}."
        )

    return ResolvedComponent(
        slot_id=slot_id,
        source="file",
        path=str(path),
        signature_id=signature,
        output_format=selection.output_format,
        source_precision=str(info.get("precision", "")),
        internal_prefixes=tuple(internal_prefixes),
    )


def _resolve_embedded(selection, slot_id, primary_path, embedded, internal_prefixes=()):
    """Keep what the checkpoint carries -- if the loader can actually see it.

    A component sitting under a namespace this architecture never reads is in
    the file and invisible to Forge, which then silently falls back to whatever
    is configured globally. Offering to "keep" it would promise something the
    output cannot deliver.
    """
    component = embedded.get(slot_id)
    if component is None:
        raise ComponentValidationError(
            f"{_label(slot_id)}: this checkpoint has no embedded component for "
            "that slot, so there is nothing to keep. Select a file instead."
        )
    if component.get("readable") is False:
        namespace = component.get("namespace", "an unread namespace")
        expected = component.get("expected_namespace") or "the architecture's namespace"
        raise ComponentValidationError(
            f"{_label(slot_id)}: this checkpoint stores it under {namespace}, "
            f"but the architecture reads {expected}, so the loader never sees it. "
            "Select a file instead."
        )

    return ResolvedComponent(
        slot_id=slot_id,
        source="embedded",
        # Attributed to the checkpoint it came from, so source precision and
        # provenance stay pinned to a real physical file.
        path=str(primary_path),
        signature_id=slot_id,
        output_format=selection.output_format,
        source_precision="",
        internal_prefixes=tuple(internal_prefixes),
    )


def build_component_plan(
    architecture_id: str,
    primary_path: str,
    checkpoint_info,
    selections,
    capability_profile=None,
    inspect_fn=None,
) -> ComponentPlan:
    """Validate selections against the slots the architecture declares.

    Raises `ComponentValidationError` for a selection that cannot work.
    Reports -- rather than raises -- a declared slot nobody filled, so the
    caller can name it and offer UNet only.
    """
    if inspect_fn is None:  # pragma: no cover - the real inspector by default
        from aux_inspector import inspect_module as inspect_fn

    selectable, embedded_only, support, prefixes = _declared_slots(
        architecture_id, capability_profile
    )
    known = set(selectable) | set(embedded_only)
    embedded = _readable_embedded(checkpoint_info)

    resolved: dict[str, ResolvedComponent] = {}
    for selection in selections or ():
        slot_id = selection.slot_id

        if selection.source not in VALID_SOURCES:
            raise ComponentValidationError(
                f"{_label(slot_id)}: {selection.source!r} is not a component source. "
                "An AIO needs every slot filled -- switch to UNet only for a "
                "checkpoint without one."
            )

        if slot_id not in known:
            # Named by its raw id rather than a prettified label: an unknown
            # slot has no label worth printing, and the id is what the caller
            # sent and what a bug report needs to quote.
            if not known:
                raise ComponentValidationError(
                    f"{slot_id}: this checkpoint's architecture has not been "
                    "resolved yet, so no component slots are available."
                )
            raise ComponentValidationError(
                f"{slot_id}: this architecture does not use that component. "
                f"It declares: {', '.join(sorted(known))}."
            )

        if slot_id in embedded_only:
            raise ComponentValidationError(
                f"{_label(slot_id)}: this architecture always embeds its own "
                "text encoder, so it cannot be replaced from a file."
            )

        if slot_id in resolved:
            raise ComponentValidationError(
                f"{_label(slot_id)}: selected twice. Each slot takes one component."
            )

        if selection.source == "embedded":
            resolved[slot_id] = _resolve_embedded(
                selection, slot_id, primary_path, embedded,
                prefixes.get(slot_id, ()),
            )
        else:
            resolved[slot_id] = _resolve_file(
                selection, slot_id, inspect_fn, prefixes.get(slot_id, ())
            )

    ordered = tuple(slot for slot in selectable if slot in resolved)
    components = tuple(resolved[slot] for slot in ordered)

    return ComponentPlan(
        architecture_id=architecture_id,
        support=support,
        components=components,
        additional_state_dicts=tuple(
            resolved[slot].path for slot in ordered if resolved[slot].source == "file"
        ),
        capability_fingerprint=(
            capability_profile.semantic_fingerprint if capability_profile else None
        ),
        missing_slots=tuple(slot for slot in selectable if slot not in resolved),
        slot_order=tuple(selectable),
    )


@dataclass(frozen=True)
class MergeComposition:
    """How a merge loads and combines, decided before a single tensor moves.

    Extracted so it can be tested. `checkpoint_merge` imports torch, Forge's
    backend and the WebUI's `modules` at module scope, none of which exist in
    the unit environment, so the orchestration decisions live here and the
    tensor work stays there.
    """

    modular: bool
    #: What engine A loads with. Empty on the traditional path, where the
    #: global module list is still consulted.
    primary_files: tuple[str, ...] = ()
    use_global_modules: bool = True
    #: B and C only ever contribute their diffusion model, so on the modular
    #: path they load bare -- inheriting globals there would pull components
    #: into memory that nothing reads.
    secondary_uses_global_modules: bool = True
    merge_text_encoder: bool = False
    merge_vae: bool = False
    allow_bake_vae: bool = True
    plan: ComponentPlan | None = None


def plan_merge_composition(
    save_mode: str, component_selections=None, plan: ComponentPlan | None = None
) -> MergeComposition:
    """Decide how this merge composes, from the save mode and the selections.

    The modular path takes over only when components were actually chosen for
    a full save. Everything else keeps the traditional behaviour untouched,
    including its use of the global module list -- changing that quietly would
    alter outputs nobody asked to change.
    """
    selections = tuple(component_selections or ())

    if selections and save_mode != "full":
        raise ComponentValidationError(
            "Component selections only apply when saving a full AIO checkpoint. "
            f"This merge is set to {save_mode!r}, which writes the diffusion "
            "model alone -- the selected components would be silently discarded."
        )

    if not selections:
        return MergeComposition(
            modular=False,
            merge_text_encoder=save_mode == "full",
            merge_vae=save_mode == "full",
            allow_bake_vae=True,
        )

    if plan is None:
        raise ComponentValidationError(
            "Components were selected but no component plan was built for them."
        )
    if not plan.is_complete:
        raise ComponentValidationError(
            _incomplete_plan_message(plan)
        )

    return MergeComposition(
        modular=True,
        primary_files=plan.additional_state_dicts,
        use_global_modules=False,
        secondary_uses_global_modules=False,
        # External components are attached whole, after the diffusion merge.
        # Interpolating them across A/B/C is what this path exists to avoid.
        merge_text_encoder=False,
        merge_vae=False,
        # The VAE slot replaces Bake VAE here; having both would be two ways to
        # set the same thing.
        allow_bake_vae=False,
        plan=plan,
    )


def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a component file, read in chunks.

    Components run to several gigabytes, so the file is streamed rather than
    read whole. An unreadable file yields an empty hash: provenance without a
    hash is still useful, and failing a finished merge over it is not.
    """
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def component_provenance(plan: ComponentPlan | None, *, with_hashes: bool = False) -> list[dict]:
    """What actually went into the output, for the recipe and the result line.

    Records the slot, the file it came from and the precision that file
    carried, so a saved AIO can be traced back to its inputs without guessing.

    Hashing is opt-in because it reads every byte of every component. The
    interface wants a quick summary; the recipe written into the checkpoint
    wants the hashes.
    """
    if plan is None:
        return []
    return [
        {
            "slot": component.slot_id,
            "label": _label(component.slot_id),
            "name": os.path.basename(component.path) if component.path else "",
            "sha256": file_sha256(component.path) if with_hashes and component.path else "",
            "source": component.source,
            "signature": component.signature_id,
            "source_precision": component.source_precision,
            "output_precision": component.output_format,
            "support": str(plan.support),
        }
        for component in plan.components
    ]


def _incomplete_plan_message(plan: ComponentPlan) -> str:
    parts = []
    if plan.missing_slots:
        names = ", ".join(_label(s) for s in plan.missing_slots)
        parts.append(f"nothing was selected for {names}")
    if plan.dropped_slots:
        names = ", ".join(_label(s) for s in plan.dropped_slots)
        parts.append(f"Forge did not accept the file chosen for {names}")
    return (
        "This checkpoint cannot be saved as an AIO: "
        + "; and ".join(parts)
        + ". Fill every component, or switch to UNet only."
    )


@dataclass(frozen=True)
class OutputValidation:
    validated: bool
    errors: tuple[str, ...] = ()
    checked_slots: tuple[str, ...] = ()


#: Requested output formats, as they appear in the final header.
_FORMAT_HEADER_CODES = {"fp16": "F16", "bf16": "BF16", "fp32": "F32"}


def _namespace_for(plan: ComponentPlan, component: ResolvedComponent) -> str:
    return plan.vae_namespace if component.slot_id == "vae" else plan.text_namespace


def _component_keys(header, plan, component) -> list[str]:
    """The saved keys belonging to one component.

    Both halves of the namespace come from the loaded engine, never a table
    here -- but they nest differently for the two kinds, and assuming otherwise
    doubles the prefix. A text encoder is saved under the architecture's outer
    prefix *plus* its own slot namespace from `clip_target`
    (``text_encoders.`` + ``qwen3_06b.transformer.``), while a VAE's slot
    namespace already is the architecture's (``vae.``), because it came from
    `vae_key_prefix` in the first place.
    """
    outer = _namespace_for(plan, component)
    candidates = {
        inner if outer and inner.startswith(outer) else outer + inner
        for inner in component.internal_prefixes or ()
    }
    if not candidates:
        candidates = {outer}
    return [
        key
        for key in header
        if key != "__metadata__" and key.startswith(tuple(candidates))
    ]


def validate_aio_output(
    output_path: str, plan: ComponentPlan | None, loader=None
) -> OutputValidation:
    """Prove a saved file is an AIO, rather than assume it.

    Four checks, in order, and the last one is the point: the file is reopened
    with an **empty** external-module list. Across the reference library only
    two checkpoints in 222 embed their components where their own architecture
    looks for them, so a file that merely contains the right tensors is not
    evidence of anything. Reloading is the only way to tell a real AIO from one
    that quietly leans on whatever is configured globally.

    A failed output is never deleted or overwritten -- it is kept, and the
    reasons come back structured rather than flattened into a boolean.
    """
    if plan is None:
        # Traditional and UNet-only outputs keep their existing semantics.
        return OutputValidation(validated=True)

    errors: list[str] = []

    try:
        header, _ = read_safetensors_header(output_path)
    except Exception as exc:
        return OutputValidation(
            validated=False,
            errors=(f"Could not read the saved file back: {exc}",),
            checked_slots=tuple(c.slot_id for c in plan.components),
        )

    for component in plan.components:
        label = _label(component.slot_id)
        keys = _component_keys(header, plan, component)
        if not keys:
            expected = _namespace_for(plan, component) or "the architecture's namespace"
            errors.append(
                f"{label}: nothing was written under {expected}, so this "
                "architecture will not find it when the file is loaded."
            )
            continue
        errors.extend(_dtype_errors(header, keys, component, label))

    if errors:
        # A file missing a whole namespace cannot reload; attempting it anyway
        # only buys a confusing exception stacked on a clear one.
        return OutputValidation(
            validated=False,
            errors=tuple(errors),
            checked_slots=tuple(c.slot_id for c in plan.components),
        )

    if loader is None:  # pragma: no cover - requires a running Forge
        from backend.loader import forge_loader as loader

    try:
        loader(output_path, additional_state_dicts=[])
    except Exception as exc:
        errors.append(
            f"The saved file did not reopen on its own: {exc}. It still needs "
            "external modules, so it is not a self-contained AIO."
        )

    return OutputValidation(
        validated=not errors,
        errors=tuple(errors),
        checked_slots=tuple(c.slot_id for c in plan.components),
    )


def _dtype_errors(header, keys, component: ResolvedComponent, label: str) -> list[str]:
    """Whether the saved dtypes match what was asked for.

    `same` copies whatever the source carried, so without the source header in
    hand there is nothing to check -- inventing a rule there would fail valid
    outputs. What `same` *does* guarantee for a scaled component is that its
    scales came through untouched, and that is checked.
    """
    errors: list[str] = []
    dtypes = {
        header[key].get("dtype")
        for key in keys
        if isinstance(header.get(key), dict)
    }

    if component.output_format != "same":
        wanted = _FORMAT_HEADER_CODES.get(component.output_format)
        if wanted:
            wrong = {d for d in dtypes if d and d != wanted and d not in ("U8", "I8")}
            if wrong:
                errors.append(
                    f"{label}: asked for {component.output_format}, but the saved "
                    f"file holds {', '.join(sorted(wrong))}."
                )
        return errors

    if "scaled" in (component.source_precision or "") or "mixed" in (
        component.source_precision or ""
    ):
        has_scales = any(
            key.endswith(("weight_scale", "weight_scale_2")) for key in keys
        )
        if not has_scales:
            errors.append(
                f"{label}: this component was stored with quantization scales, "
                "but none were written to the output. Its weights cannot be "
                "read back without them."
            )
    return errors


def preflight_component_plan(primary_path: str, plan: ComponentPlan, loader=None):
    """Load the checkpoint with exactly the plan's component files.

    Nothing else goes in. `shared.opts.forge_additional_modules` is never read
    here -- that implicit dependency is the problem this whole feature exists to
    remove, and a preflight that quietly inherited it would validate a
    composition the output cannot reproduce.

    Returns the loaded engine. The Forge import happens inside the
    `loader is None` branch on purpose: the unit suite runs outside an
    initialised Forge, and importing the loader at module scope would make this
    module untestable.
    """
    if loader is None:  # pragma: no cover - requires a running Forge
        from backend.loader import forge_loader as loader

    files = list(plan.additional_state_dicts)
    try:
        return loader(primary_path, additional_state_dicts=files)
    except Exception as exc:
        selected = ", ".join(os.path.basename(f) for f in files) or "no external files"
        raise ComponentValidationError(
            f"Forge could not load this {plan.architecture_id} composition "
            f"({selected}): {exc}"
        ) from exc


def _loaded_object(engine, name: str):
    objects = getattr(engine, "forge_objects", None)
    if objects is None:
        raise ComponentValidationError(
            "The loaded engine exposes no 'forge_objects', so its components "
            "cannot be verified. This is not an engine this path can validate."
        )
    return getattr(objects, name, None)


def validate_loaded_components(engine, plan: ComponentPlan) -> ComponentPlan:
    """Reconcile a provisional plan against what Forge actually resolved.

    The loaded engine is the authority. Whatever the header guessed is replaced
    here, and two failure shapes are told apart deliberately:

    * a capability the config no longer exposes, or an engine that came back as
      a different architecture, is an error -- something drifted;
    * a slot the reloaded `clip_target` simply stopped declaring is *missing*,
      not an error. Forge does not raise when a component was not supplied; it
      returns one target fewer, in silence. That silence is the most likely way
      a broken AIO would slip through, so it is turned into data the caller can
      act on.

    Returns the reconciled plan. (The first draft of this contract returned
    None and raised on everything; missing slots are data, so they need to come
    back rather than blow up.)
    """
    from forge_capabilities import capability_profile_from_engine, is_anima_profile

    try:
        profile = capability_profile_from_engine(engine)
    except ForgeCapabilityError as exc:
        raise ComponentValidationError(
            f"The loaded engine no longer exposes a capability this composition "
            f"needs: {exc}"
        ) from exc

    if plan.architecture_id == "anima" and not is_anima_profile(profile):
        raise ComponentValidationError(
            f"The header read this checkpoint as Anima, but the engine Forge "
            f"loaded reports {profile.family_hint or 'another architecture'} "
            f"({profile.diagnostic_fingerprint}). Refusing to compose against a "
            "mismatched architecture."
        )

    policy = apply_policy(profile)
    declared = tuple(s.forge_target for s in policy.slots if not s.embedded_only)

    if any(s.kind == "text_encoder" for s in policy.slots):
        if _loaded_object(engine, "clip") is None:
            raise ComponentValidationError(
                "Forge loaded no text encoder for this composition. The selected "
                "file may sit under a namespace this architecture does not read."
            )
    if profile.vae_target is not None and _loaded_object(engine, "vae") is None:
        raise ComponentValidationError(
            "Forge loaded no VAE for this composition. The selected file may sit "
            "under a namespace this architecture does not read."
        )

    # Only components whose slot survived the reload stay in the plan. A slot
    # that vanished took the user's file with it, which Forge did not announce.
    kept = tuple(c for c in plan.components if c.slot_id in declared)
    order = {slot: i for i, slot in enumerate(declared)}
    kept = tuple(sorted(kept, key=lambda c: order[c.slot_id]))
    filled = {c.slot_id for c in kept}

    return ComponentPlan(
        architecture_id=plan.architecture_id,
        support=policy.support,
        components=kept,
        additional_state_dicts=tuple(
            c.path for c in kept if c.source == "file"
        ),
        capability_fingerprint=profile.semantic_fingerprint,
        missing_slots=tuple(slot for slot in declared if slot not in filled),
        dropped_slots=tuple(
            c.slot_id for c in plan.components if c.slot_id not in declared
        ),
        text_namespace=(
            profile.text_encoder_key_prefix[0]
            if profile.text_encoder_key_prefix
            else ""
        ),
        vae_namespace=profile.vae_key_prefix[0] if profile.vae_key_prefix else "",
        slot_order=declared,
    )
