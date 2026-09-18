"""The shape of the AIO component form, decided without drawing it.

`scripts/merge_studio_ui.py` renders what this module returns and does nothing
else with it. Keeping the decisions here means they can be tested: that file
imports Gradio and the WebUI's `modules`, neither of which exists outside a
running Forge.

Two rules run through everything below:

* **How many rows is the architecture's call**, taken from Forge's
  `clip_target`. Never a table here.
* **Nothing is filled in automatically.** Not from scanning the folder, not
  from whatever is set in Additional Modules. Someone who wants to embed
  components has the knowledge to choose them, and a guess would put something
  in the checkpoint that nobody picked.
"""

from __future__ import annotations

from dataclasses import dataclass

from component_bundle import PROVISIONAL_SLOTS
from component_registry import (
    SupportState,
    apply_policy,
    get_slot_policy,
    vae_fits_latent_format,
)

#: The sentinel a row carries when the user keeps what the checkpoint already
#: has. Not a filename, so it cannot collide with one.
KEEP_EMBEDDED = "__keep_embedded__"

#: Offered for a component stored plainly. A component stored with scales gets
#: `same` alone -- converting it would mean dequantizing.
COMPONENT_FORMAT_CHOICES = (
    ("Same as component source", "same"),
    ("FP16", "fp16"),
    ("BF16", "bf16"),
    ("FP32", "fp32"),
)
SAME_ONLY = (COMPONENT_FORMAT_CHOICES[0],)


@dataclass(frozen=True)
class ComponentRowState:
    slot_id: str
    label: str
    kind: str
    #: Installed files whose signature fits this slot, by display name.
    choices: tuple[str, ...] = ()
    #: Whether "keep what is in the file" may be offered -- only when the
    #: loader can actually see it.
    keep_embedded: bool = False
    selected: str | None = None
    format_choices: tuple[tuple[str, str], ...] = COMPONENT_FORMAT_CHOICES
    warning: str = ""
    status: str = ""


def _embedded_by_role(checkpoint_info) -> dict:
    found = {}
    for component in (checkpoint_info or {}).get("embedded_components") or ():
        role = component.get("role")
        if role:
            found[role] = component
    return found


def _slots_for(checkpoint_info, capability_profile):
    """`[(slot_id, label, kind)]`, from Forge when an engine has been loaded.

    Before that, the header's architecture id gives a provisional guess for the
    few families a header can pin down. An architecture that could not be
    identified offers nothing: an empty form beats the wrong one.
    """
    if capability_profile is not None:
        policy = apply_policy(capability_profile)
        return (
            [
                (slot.forge_target, slot.label, slot.kind)
                for slot in policy.slots
                if not slot.embedded_only
            ],
            policy.support,
        )

    architecture_id = (checkpoint_info or {}).get("architecture_id") or "unknown"
    slots = PROVISIONAL_SLOTS.get(architecture_id)
    if not slots:
        return [], SupportState.UNKNOWN
    return (
        [
            (slot_id, get_slot_policy(slot_id).label, "vae" if slot_id == "vae" else "text_encoder")
            for slot_id in slots
        ],
        SupportState.SUPPORTED,
    )


def _fitting_modules(slot_id: str, module_infos, latent_format_name: str = "") -> tuple[str, ...]:
    """Installed files that can actually fill this slot.

    For a VAE that means more than "is a VAE": the latent space has to match.
    Offering a Flux AE for an Anima checkpoint builds a file that only fails
    at the post-save reload, gigabytes later.
    """
    accepted = get_slot_policy(slot_id).accepted_signatures
    fitting = []
    for name, info in sorted((module_infos or {}).items()):
        if not info.get("supported_storage", True):
            continue
        if info.get("signature_id") not in accepted:
            continue
        if slot_id == "vae" and not vae_fits_latent_format(info, latent_format_name):
            continue
        fitting.append(name)
    return tuple(fitting)


def _format_choices_for(selected_name: str | None, module_infos):
    if not selected_name or selected_name == KEEP_EMBEDDED:
        return COMPONENT_FORMAT_CHOICES
    info = (module_infos or {}).get(selected_name) or {}
    if info.get("storage_kind", "plain") != "plain":
        return SAME_ONLY
    return COMPONENT_FORMAT_CHOICES


def build_component_rows(
    checkpoint_info,
    save_mode: str,
    module_infos,
    capability_profile=None,
    selections=None,
) -> tuple[ComponentRowState, ...]:
    """One row per component this architecture needs, or nothing at all.

    Rows appear only for an AIO save. UNet only writes the diffusion model
    alone, so a component form there would promise something the output does
    not deliver.
    """
    if save_mode != "full":
        return ()

    slots, support = _slots_for(checkpoint_info, capability_profile)
    if not slots:
        return ()

    latent_format = getattr(capability_profile, "latent_format_name", "") or ""
    embedded = _embedded_by_role(checkpoint_info)
    chosen = selections or {}
    warning = (
        "This architecture is experimental here: the same checks apply, but it "
        "has not been exercised against a real model."
        if support == SupportState.EXPERIMENTAL
        else ""
    )

    built = []
    for slot_id, label, kind in slots:
        component = embedded.get(slot_id)
        readable = bool(component and component.get("readable"))

        status = ""
        if component and not readable:
            namespace = component.get("namespace", "an unread namespace")
            expected = component.get("expected_namespace") or "this architecture's namespace"
            status = (
                f"This checkpoint carries a {label} under {namespace}, but "
                f"{expected} is what gets loaded, so it cannot be kept. "
                "Select a file instead."
            )

        selected = chosen.get(slot_id)
        built.append(
            ComponentRowState(
                slot_id=slot_id,
                label=label,
                kind=kind,
                choices=_fitting_modules(slot_id, module_infos, latent_format),
                keep_embedded=readable,
                selected=selected,
                format_choices=_format_choices_for(selected, module_infos),
                warning=warning,
                status=status,
            )
        )
    return tuple(built)


def missing_slot_message(missing, *, all_slots=()) -> str:
    """Name what is missing, and offer the way out.

    A lone encoder is described by its role rather than its model name, which
    reads better and says everything needed. Flux has two, so there the label
    is the only thing that says which one.
    """
    missing = tuple(missing or ())
    if not missing:
        return ""

    encoders = [slot for slot in missing if slot != "vae"]
    wants_vae = "vae" in missing
    total_encoders = len([s for s in (all_slots or missing) if s != "vae"])

    parts = []
    if encoders:
        if total_encoders > 1:
            parts.extend(get_slot_policy(s).label for s in encoders)
        else:
            parts.append("a text encoder")
    if wants_vae:
        parts.append("a VAE")

    if len(parts) > 2:
        listed = ", ".join(parts[:-1]) + f" and {parts[-1]}"
    else:
        listed = " and ".join(parts)
    return f"Select {listed} to save as AIO — or switch to UNet only."


def parse_component_rows(rows) -> list[dict]:
    """Row values as selections `merge_checkpoints` accepts.

    An unfilled row contributes nothing rather than an empty selection, so the
    plan reports it as missing and the interface can name it.
    """
    parsed = []
    for row in rows or ():
        value = row.get("value")
        if not value:
            continue
        slot_id = row.get("slot_id")
        if not slot_id:
            continue
        embedded = value == KEEP_EMBEDDED
        parsed.append(
            {
                "slot_id": slot_id,
                "source": "embedded" if embedded else "file",
                "path": None if embedded else value,
                "output_format": row.get("format", "same"),
            }
        )
    return parsed
