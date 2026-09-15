"""Cross-generation Anima block remapping for checkpoint merges.

Each Anima generation was produced by *inserting* new transformer blocks in
between the previous generation's blocks (28 -> 40 -> 52), rather than by
appending them. A name-for-name merge therefore lines up `blocks.5` of a 28
block model with `blocks.5` of a 40 block model, which are different layers --
the result is noise.

Forge already solves this for LoRAs in
`extensions-builtin/sd_forge_lora/networks.py::process_anima`, but only there;
model merges get no such treatment. This module reuses Forge's own mapping
tables so a merge can bridge two generations.

The tables are not module-level constants in Forge -- they're locals inside
process_anima -- so instead of copying them (and risking drift when a new
generation lands) we *execute* process_anima against a synthetic LoRA whose
tensors carry their own block index, then read the mapping back out. The
vendored tables below are only a fallback for when that probe fails.

Mapping terminology, matching the manifests published by
https://github.com/shin131002/ComfyUI-Anima-Remap (MIT), whose 28/40, 40/52
and 28/52 tables were verified to agree with Forge's exactly:

  target  an index in the LARGER (newer) model -- the merge output's base
  source  an index in the SMALLER (older) model
  frozen  a target block carried over from the older generation; it has a
          real source counterpart, so it can be merged
  inserted a target block created by the expansion; it has no counterpart in
          the older model and is left at the base model's own weights
"""

from __future__ import annotations

import re

ANIMA_BLOCK_SIZES = (28, 40, 52)

# Fallback only -- see module docstring. Index = target block, value = source
# block. Copied from networks.py::process_anima.
_FALLBACK_TARGET_TO_SOURCE: dict[tuple[int, int], list[int]] = {
    (28, 40): [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 14, 14, 15, 16, 16, 17, 18, 18, 19, 20, 20, 21, 22, 22, 23, 24, 24, 25, 26, 27],
    (28, 52): [0, 1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 7, 7, 8, 9, 9, 9, 10, 11, 11, 11, 12, 13, 14, 14, 14, 15, 16, 16, 16, 17, 18, 18, 18, 19, 20, 20, 20, 21, 22, 22, 22, 23, 24, 24, 24, 25, 26, 27],
    (40, 52): [0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 14, 15, 16, 17, 17, 18, 19, 20, 20, 21, 22, 23, 23, 24, 25, 26, 26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36, 37, 38, 39],
}

# The main transformer blocks sit at the ROOT of the DiT: "blocks.<N>..." in a
# module path relative to diffusion_model, or "<root>.blocks.<N>..." on disk,
# where <root> is one of the prefixes Forge's loader accepts (backend/loader.py
# ::preprocess_state_dict normalises to "model.diffusion_model." / "net.").
#
# Anchoring matters. Anima checkpoints carry other sub-modules that have their
# own, unrelated indexed block stacks:
#
#   model.diffusion_model.llm_adapter.blocks.<N>....             (all generations)
#   net.anima_v2_connector.semantic_resampler.blocks.<N>....     (3.8B v1.1+)
#
# Both match a loose ".blocks.<N>." search, and remapping either would corrupt a
# component that the 28/40/52 expansion never touched. Anchoring to the root
# excludes them structurally, which is why this does NOT use a list of known
# sub-module names -- that list would silently go stale the next time upstream
# adds or renames a component, as happened with the v1.1 "Semantic Connector v2".
_BLOCK_RE = re.compile(r"^(blocks\.)(\d+)(\.|$)")
_DISK_BLOCK_RE = re.compile(r"^(?:net\.|model\.diffusion_model\.)?blocks\.(\d+)\.")


def _main_block_index(name: str) -> int | None:
    """Main-stack block index for a module path relative to diffusion_model,
    or None if this path isn't a root-level block."""
    m = _BLOCK_RE.match(name)
    return int(m.group(2)) if m else None



def block_count_from_keys(keys) -> int | None:
    """Highest main transformer block index + 1, read straight from
    .safetensors key names -- no engine, no tensor loads. Returns None when
    the checkpoint has no block-list structure.

    Only root-level blocks count: nested stacks such as the llm_adapter's or
    the v1.1 semantic connector's are structurally excluded by the anchor."""
    best = -1
    for k in keys:
        m = _DISK_BLOCK_RE.match(k)
        if m:
            idx = int(m.group(1))
            if idx > best:
                best = idx
    return best + 1 if best >= 0 else None


class AnimaRemapError(RuntimeError):
    pass


def _probe_target_to_source(src_blocks: int, dst_blocks: int) -> list[int] | None:
    """Derives target -> source by running Forge's own process_anima over a
    synthetic LoRA. Each tensor's single element is its source block index,
    so reading element t back after the call yields the source that Forge
    would have mapped onto target block t. Returns None if the probe can't
    run, in which case the caller falls back to the vendored table."""
    try:
        import torch

        import networks  # type: ignore

        process_anima = getattr(networks, "process_anima", None)
        if process_anima is None:
            return None

        probe = {f"diffusion_model.blocks.{s}.weight": torch.tensor([float(s)]) for s in range(src_blocks)}
        if not process_anima(probe, dst_blocks):
            return None

        mapping: list[int] = []
        for t in range(dst_blocks):
            entry = probe.get(f"diffusion_model.blocks.{t}.weight")
            if entry is None:
                return None
            mapping.append(int(entry.item()))
        return mapping
    except Exception:
        return None


def target_to_source(src_blocks: int, dst_blocks: int) -> list[int]:
    """target block index -> source block index, for every target block."""
    if src_blocks == dst_blocks:
        return list(range(dst_blocks))
    if src_blocks > dst_blocks:
        raise AnimaRemapError(
            f"Cannot remap a larger Anima model onto a smaller one ({src_blocks} -> {dst_blocks} blocks). "
            "Use the model with more blocks as Primary Model (A)."
        )

    mapping = _probe_target_to_source(src_blocks, dst_blocks)
    if mapping is None:
        mapping = _FALLBACK_TARGET_TO_SOURCE.get((src_blocks, dst_blocks))
    if mapping is None or len(mapping) != dst_blocks:
        raise AnimaRemapError(f"No Anima block mapping is known for {src_blocks} -> {dst_blocks} blocks.")
    return mapping


def split_frozen_inserted(mapping: list[int]) -> tuple[dict[int, int], dict[int, int]]:
    """Splits target -> source into the frozen blocks (first occurrence of
    each source: a genuine one-to-one counterpart) and the inserted blocks
    (every later occurrence: a block the expansion created by copying an
    existing one, with no counterpart of its own)."""
    frozen: dict[int, int] = {}
    inserted: dict[int, int] = {}
    seen: set[int] = set()
    for target, source in enumerate(mapping):
        if source in seen:
            inserted[target] = source
        else:
            frozen[target] = source
            seen.add(source)
    return frozen, inserted


def block_count(diffusion_model) -> int | None:
    """Number of main transformer blocks, or None if this isn't a block-list
    architecture."""
    blocks = getattr(diffusion_model, "blocks", None)
    try:
        return len(blocks) if blocks is not None else None
    except TypeError:
        return None


def is_anima_engine(engine) -> bool:
    return type(engine.model_config).__name__ == "Anima"


def make_extend_translator(inserted: dict[int, int]):
    """Returns f(name_in_A) -> the B-side name of the block this inserted
    block was originally copied from, or None for anything that isn't an
    inserted block.

    Each generation created its new blocks by deep-copying an existing one at
    initialization, so that source block is the closest thing the older model
    has to a counterpart -- see `inserted_to_source` in the Anima-Remap
    manifests. The two have since diverged in training, which is why blending
    them is opt-in rather than the default."""

    def extend(name: str) -> str | None:
        target = _main_block_index(name)
        if target is None:
            return None
        source = inserted.get(target)
        if source is None:
            return None
        m = _BLOCK_RE.match(name)
        return f"blocks.{source}" + name[m.end(2) :]

    return extend


def make_name_translator(frozen: dict[int, int]):
    """Returns f(name_in_A) -> name_in_B, or None when the A-side module has
    no counterpart in B (an inserted block) and must be left alone.

    Anything that isn't a root-level block -- the llm_adapter, the v1.1
    semantic connector, embedders, final_layer -- is returned untouched."""

    def translate(name: str) -> str | None:
        target = _main_block_index(name)
        if target is None:
            return name
        source = frozen.get(target)
        if source is None:
            return None
        m = _BLOCK_RE.match(name)
        return f"blocks.{source}" + name[m.end(2) :]

    return translate
