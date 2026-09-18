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

import os
from dataclasses import dataclass, field

from component_registry import SupportState, apply_policy, get_slot_policy

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


@dataclass(frozen=True)
class ComponentPlan:
    architecture_id: str
    support: SupportState
    components: tuple[ResolvedComponent, ...] = ()
    additional_state_dicts: tuple[str, ...] = ()
    capability_fingerprint: str | None = None
    missing_slots: tuple[str, ...] = ()
    slot_order: tuple[str, ...] = field(default=())


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
        return selectable, embedded_only, policy.support

    slots = PROVISIONAL_SLOTS.get(architecture_id)
    if slots is None:
        return (), (), SupportState.UNKNOWN
    return slots, (), SupportState.SUPPORTED


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


def _resolve_file(selection, slot_id, inspect_fn):
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
    )


def _resolve_embedded(selection, slot_id, primary_path, embedded):
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

    selectable, embedded_only, support = _declared_slots(
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
                selection, slot_id, primary_path, embedded
            )
        else:
            resolved[slot_id] = _resolve_file(selection, slot_id, inspect_fn)

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
