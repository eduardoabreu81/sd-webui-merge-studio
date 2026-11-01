"""Fill the LoRA slots from a pasted prompt.

A prompt names its LoRAs the way Forge's extra-networks syntax does:

    <lora:MOSS_BASE_DMD_8STEP_QKV:1.0>, <lora:moss_style_v1-comfy:0.7>

`parse` pulls out every tag, in order, with its weight; `resolve` maps each
tag's name to the name the dropdowns use, the same way Forge would when the
prompt runs: by name, then by alias (`ss_output_name`), ignoring case. Tags
that match nothing are handed back so the UI can say which ones were skipped.
"""

from __future__ import annotations

import re

# <lora:name>, <lora:name:0.7>, <lora:name:0.7:0.5>, <lyco:...>. The first
# argument after the name is the weight Forge applies to the model.
_TAG = re.compile(r"<\s*(?:lora|lyco)\s*:([^>]*)>", re.IGNORECASE)


def _weight(arg: str) -> float:
    arg = arg.strip()
    if "=" in arg:
        key, _, value = arg.partition("=")
        if key.strip().lower() not in ("unet", "w"):
            return 1.0
        arg = value
    try:
        return float(arg)
    except ValueError:
        return 1.0


def parse(text: str) -> list[tuple[str, float]]:
    """(name, weight) for every LoRA tag in `text`, in the order written."""
    out = []
    for match in _TAG.finditer(text or ""):
        parts = match.group(1).split(":")
        name = parts[0].strip()
        if not name:
            continue
        weight = _weight(parts[1]) if len(parts) > 1 and parts[1].strip() else 1.0
        out.append((name, weight))
    return out


def _candidates(name: str) -> list[str]:
    """The tag name as written, then without a folder or file extension,
    for prompts copied from tools that keep either."""
    base = re.split(r"[\\/]", name)[-1]
    stem = re.sub(r"\.(safetensors|ckpt|pt|bin)$", "", base, flags=re.IGNORECASE)
    return list(dict.fromkeys([name, base, stem]))


def resolve(
    tags: list[tuple[str, float]],
    names,
    aliases: dict[str, str] | None = None,
) -> tuple[list[tuple[str, float]], list[str]]:
    """Map tag names to dropdown names.

    `names` are the names the dropdowns hold; `aliases` maps other names a
    LoRA answers to (Forge's alias table) onto one of them. Returns the
    matched (name, weight) pairs and the tag names nothing matched.
    """
    exact = {n: n for n in names}
    folded = {n.casefold(): n for n in names}
    alias_exact = dict(aliases or {})
    alias_folded = {a.casefold(): n for a, n in alias_exact.items()}

    found, missing = [], []
    for name, weight in tags:
        hit = None
        for candidate in _candidates(name):
            hit = (
                exact.get(candidate)
                or alias_exact.get(candidate)
                or folded.get(candidate.casefold())
                or alias_folded.get(candidate.casefold())
            )
            if hit:
                break
        if hit:
            found.append((hit, weight))
        else:
            missing.append(name)
    return found, missing
