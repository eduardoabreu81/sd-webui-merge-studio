"""Saving an AIO composition, and loading one back honestly.

A recipe records what was chosen so the same checkpoint can be built again.
That only holds if restoring is truthful: a component that is no longer
installed leaves its slot empty and says so. Quietly reaching for a similar
file would produce a different checkpoint under the same recipe name, which is
worse than refusing to fill it.

v1 recipes predate component slots. They carry a Bake VAE and nothing else, so
that is the most they may restore -- inventing a text encoder for one would put
a component in the output that the original merge never had.

Pure: no Gradio, no Forge, no filesystem.
"""

from __future__ import annotations

import os

from .component_registry import get_slot_policy

#: Bumped from 1 when component slots arrived. v1 recipes still load.
RECIPE_VERSION = 2

#: Bake VAE values that mean "no file", not a component.
_NON_COMPONENT_VAE = {"original", "none", ""}


def serialize_component_recipe(rows) -> list[dict]:
    """Component rows as recipe entries.

    Only the basename is stored. A recipe should survive being moved between
    machines whose model folders sit in different places, and the name is what
    the module list is keyed by anyway.
    """
    stored = []
    for row in rows or ():
        slot_id = row.get("slot_id")
        source = row.get("source", "file")
        name = row.get("name") or ""
        if not slot_id:
            continue
        if source != "embedded" and not name:
            continue
        stored.append(
            {
                "slot_id": slot_id,
                "name": os.path.basename(name),
                "source": source,
                "format": row.get("format", "same"),
            }
        )
    return stored


def _fits(slot_id: str, name: str, module_infos) -> bool:
    info = (module_infos or {}).get(name)
    if info is None:
        return False
    return info.get("signature_id") in get_slot_policy(slot_id).accepted_signatures


def restore_component_recipe(recipe, slot_ids, module_infos):
    """`(selections_by_slot, warnings)` for a v2 recipe.

    A slot is filled only when the recorded file is installed *and* still fits
    it. Anything else is reported and left empty, so the form shows what has to
    be picked again instead of hiding a substitution.
    """
    entries = (recipe or {}).get("components") or []
    slots = tuple(slot_ids or ())
    restored: dict[str, dict] = {}
    warnings: list[str] = []

    for entry in entries:
        slot_id = entry.get("slot_id")
        if not slot_id:
            continue
        label = get_slot_policy(slot_id).label
        name = entry.get("name") or ""

        if slots and slot_id not in slots:
            warnings.append(
                f"This recipe sets {label} ({slot_id}), which this checkpoint's "
                "architecture does not use. Ignored."
            )
            continue

        if entry.get("source") == "embedded":
            restored[slot_id] = {"source": "embedded", "name": "", "format": entry.get("format", "same")}
            continue

        if not _fits(slot_id, name, module_infos):
            installed = name in (module_infos or {})
            warnings.append(
                f"{label}: {name} "
                + ("no longer fits this slot" if installed else "is not installed here")
                + ". Select a component for it."
            )
            continue

        restored[slot_id] = {
            "source": "file",
            "name": name,
            "format": entry.get("format", "same"),
        }

    return restored, warnings


def migrate_v1_components(settings, slot_ids, module_infos):
    """What a v1 recipe can contribute to the component form.

    At most its Bake VAE, and only when that file is installed and fits the
    architecture's VAE slot. Every other slot stays empty and is reported: the
    merge this recipe describes did not set them, and guessing would change
    what it produces.
    """
    slots = tuple(slot_ids or ())
    restored: dict[str, dict] = {}
    warnings: list[str] = []

    if not slots:
        # A traditional architecture with no component form: nothing to migrate.
        return restored, warnings

    baked = (settings or {}).get("bake_vae")
    name = os.path.basename(str(baked)) if baked else ""

    if name and name.lower() not in _NON_COMPONENT_VAE:
        if "vae" in slots and _fits("vae", name, module_infos):
            restored["vae"] = {"source": "file", "name": name, "format": "same"}
        else:
            warnings.append(
                f"VAE: {name} was baked by this recipe but is not available here. "
                "Select a component for it."
            )

    for slot_id in slots:
        if slot_id in restored:
            continue
        label = get_slot_policy(slot_id).label
        warnings.append(
            f"{label} ({slot_id}): this recipe predates component slots, so it "
            "does not say which file to use. Select one."
        )

    return restored, warnings
