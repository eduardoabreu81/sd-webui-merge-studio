"""Merge two or more LoRAs into a single LoRA.

`lora_bake.py` bakes an adapter into a checkpoint and `lora_extract.py` pulls
one out of a difference. Neither combines adapters, which is what this does.

Adding LoRAs is not adding their tensors. A LoRA stores `down` and `up`, and
the weight delta it applies is `up @ down * (alpha / rank)`; the sum of two
deltas is therefore

    up1 @ down1 + up2 @ down2   !=   (up1 + up2) @ (down1 + down2)

Tools that add the tensors produce a file that loads, runs, and is quietly
wrong. The identity that does hold is concatenation: stacking the factors side
by side gives a product that is exactly the sum, at the cost of a rank that is
the sum of the input ranks. Compressing that back down is a separate, optional
step -- so `target_rank=0` is not a fallback, it is the exact answer.

Like `lora_extract`, planning works from safetensors headers alone, so the
interface can say what a merge would do the moment the files are picked. And
like it, torch is imported inside the function that needs it rather than at
module scope: the maths here is written against an array module passed in, so
it can be tested without a Forge install.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .aux_inspector import _lora_algorithm, _lora_block_info
from .checkpoint_inspector import _esc, read_safetensors_header

#: The two spellings of the same pair. kohya-style files write
#: `lora_down`/`lora_up`; diffusers-style ones write `lora_A`/`lora_B`. Both
#: mean "the two factors of one module", and Forge loads either.
_FACTOR_PAIRS = (
    ("lora_down.weight", "lora_up.weight"),
    ("lora_A.weight", "lora_B.weight"),
)

#: Bytes per element, for estimating the output before writing it.
_DTYPE_WIDTHS = {"BF16": 2, "F16": 2, "F32": 4}

#: Algorithms whose maths is not a product of two factors, so concatenation
#: does not apply to them at all. Named individually because "unsupported" on
#: its own sends someone looking for a setting that would fix it.
_UNMERGEABLE = (
    ("hada_w", "LoHa", "its delta is a Hadamard product of two low-rank pairs"),
    ("lokr_w", "LoKr", "its delta is a Kronecker product, not a matrix product"),
    ("oft_R.", "OFTv2", "it rotates the weight instead of adding to it"),
    ("oft_blocks", "OFT / BOFT", "it rotates the weight instead of adding to it"),
    (".a1.weight", "GLoRA", "it carries four factors per module, not two"),
    (".diff.", "a plain difference patch", "it stores whole tensors, not factors"),
)


def split_factor_key(key: str) -> tuple[str, str] | None:
    """``(module, "down"|"up")`` for a factor tensor, or None for anything else.

    The module name is everything before the suffix, which is what makes the
    same module in two files line up regardless of how deep the prefix is.
    """
    for down_suffix, up_suffix in _FACTOR_PAIRS:
        for suffix, side in ((down_suffix, "down"), (up_suffix, "up")):
            ending = "." + suffix
            if key.endswith(ending):
                return key[: -len(ending)], side
    return None


def _suffixes_for(key: str) -> tuple[str, str]:
    """The pair of suffixes the file used, so the output keeps its spelling."""
    for down_suffix, up_suffix in _FACTOR_PAIRS:
        if key.endswith("." + up_suffix):
            return down_suffix, up_suffix
    return _FACTOR_PAIRS[0]


def _matrix_shape(shape) -> tuple[int, int]:
    """A factor's 2-D shape. Conv factors carry their kernel in the trailing
    dimensions; flattening them is what lets a LoCon module concatenate with
    the same code as a linear one."""
    if len(shape) <= 1:
        return int(shape[0]), 1
    rows = int(shape[0])
    columns = 1
    for dimension in shape[1:]:
        columns *= int(dimension)
    return rows, columns


@dataclass(frozen=True)
class SourceLora:
    """One input file, described from its header."""

    path: str
    name: str
    weight: float
    algorithm: str
    generation: int | None
    modules: int
    ranks: dict[int, int]
    refusals: list[str] = field(default_factory=list)

    @property
    def uniform_rank(self) -> int | None:
        return next(iter(self.ranks)) if len(self.ranks) == 1 else None


@dataclass(frozen=True)
class ModuleMerge:
    """One module, and what the merge would do with it."""

    name: str
    sources: list[int]
    #: Rank if nothing is compressed: the ranks of every source, added.
    accumulated_rank: int
    out_features: int
    in_features: int
    down_suffix: str
    up_suffix: str
    #: The `down` shape to write, kernel dimensions included.
    down_shape: tuple[int, ...]

    def output_rank(self, target_rank: int) -> int:
        """Compression cannot raise rank, and cannot exceed the module's own
        dimensions -- a rank above `min(out, in)` describes directions the
        matrix does not have."""
        ceiling = min(self.accumulated_rank, self.out_features, self.in_features)
        if target_rank <= 0:
            return ceiling
        return min(int(target_rank), ceiling)


@dataclass(frozen=True)
class MergePlan:
    sources: list[SourceLora]
    modules: list[ModuleMerge]
    target_rank: int
    output_dtype: str
    refusals: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.refusals and bool(self.modules)

    @property
    def shared_modules(self) -> int:
        return sum(1 for m in self.modules if len(m.sources) > 1)

    @property
    def exclusive_modules(self) -> int:
        return sum(1 for m in self.modules if len(m.sources) == 1)

    def estimated_bytes(self) -> int:
        width = _DTYPE_WIDTHS.get(self.output_dtype, 2)
        total = 0
        for module in self.modules:
            rank = module.output_rank(self.target_rank)
            total += (module.out_features * rank + rank * module.in_features) * width
            total += 4  # the alpha scalar
        return total


def _describe_source(path: str, weight: float) -> SourceLora:
    """Everything the plan needs about one file, from its header."""
    name = os.path.basename(path)
    try:
        header, _ = read_safetensors_header(path)
    except Exception as e:
        return SourceLora(
            path=path, name=name, weight=weight, algorithm="unreadable",
            generation=None, modules=0, ranks={},
            refusals=[f"{name}: header could not be read ({e})"],
        )

    keys = [k for k in header if k != "__metadata__"]
    joined = "\n".join(keys)
    refusals = []

    for marker, algorithm_name, because in _UNMERGEABLE:
        if marker in joined:
            refusals.append(
                f"{name} is {algorithm_name}, which cannot be merged this way: {because}."
            )
            break

    if "dora_scale" in joined:
        refusals.append(
            f"{name} carries DoRA magnitude vectors. They scale the merged weight per "
            "column rather than adding to it, so concatenating the factors underneath "
            "them changes what they scale."
        )

    modules: dict[str, dict[str, Any]] = {}
    for key in keys:
        split = split_factor_key(key)
        if split is not None:
            modules.setdefault(split[0], {})[split[1]] = key

    paired = {m: sides for m, sides in modules.items() if "down" in sides and "up" in sides}
    if not paired and not refusals:
        refusals.append(f"{name} has no LoRA factor pairs in it.")

    ranks: dict[int, int] = {}
    for sides in paired.values():
        shape = (header.get(sides["down"]) or {}).get("shape") or []
        if shape:
            ranks[int(shape[0])] = ranks.get(int(shape[0]), 0) + 1

    return SourceLora(
        path=path,
        name=name,
        weight=float(weight),
        algorithm=_lora_algorithm(keys),
        generation=_lora_block_info(keys)["generation"],
        modules=len(paired),
        ranks=ranks,
        refusals=refusals,
    )


def _module_table(path: str) -> dict[str, dict[str, Any]]:
    """{module: {down, up, alpha, shapes}} for one file."""
    header, _ = read_safetensors_header(path)
    out: dict[str, dict[str, Any]] = {}
    for key, value in header.items():
        if key == "__metadata__" or not isinstance(value, dict):
            continue
        split = split_factor_key(key)
        if split is None:
            continue
        module, side = split
        entry = out.setdefault(module, {})
        entry[side] = key
        entry[side + "_shape"] = tuple(int(d) for d in (value.get("shape") or []))
    return {m: e for m, e in out.items() if "down" in e and "up" in e}


def plan_merge(
    paths: list[str],
    weights: list[float],
    *,
    target_rank: int = 0,
    output_dtype: str = "BF16",
) -> MergePlan:
    """What merging these files would produce, decided from headers alone.

    Modules present in only some of the inputs are kept, not dropped: the
    result is the sum of the deltas, and a module only one adapter touches is
    still part of that sum. They are counted separately in the plan because a
    pair of LoRAs with nothing in common is usually a mistake worth seeing
    before the file is written.
    """
    sources = [
        _describe_source(path, weight)
        for path, weight in zip(paths, weights)
    ]

    refusals: list[str] = []
    warnings: list[str] = []
    for source in sources:
        refusals.extend(source.refusals)

    if len(sources) < 2:
        refusals.append("A merge needs at least two LoRAs.")

    generations = {s.generation for s in sources if s.generation is not None}
    if len(generations) > 1:
        listed = ", ".join(f"{s.name} ({s.generation})" for s in sources if s.generation)
        refusals.append(
            "These LoRAs target different Anima generations, whose block indices do not "
            f"line up: {listed}. The cross-generation remap exists for checkpoint merges, "
            "not for adapters."
        )

    if refusals:
        return MergePlan(
            sources=sources, modules=[], target_rank=int(target_rank),
            output_dtype=output_dtype, refusals=refusals, warnings=warnings,
        )

    tables = [_module_table(s.path) for s in sources]
    ordered: list[str] = []
    seen: set[str] = set()
    for table in tables:
        for module in table:
            if module not in seen:
                seen.add(module)
                ordered.append(module)

    modules: list[ModuleMerge] = []
    for module in ordered:
        present = [i for i, table in enumerate(tables) if module in table]
        first = tables[present[0]][module]
        out_features = _matrix_shape(first["up_shape"])[0]
        in_features = _matrix_shape(first["down_shape"])[1]

        mismatched = [
            sources[i].name
            for i in present[1:]
            if _matrix_shape(tables[i][module]["up_shape"])[0] != out_features
            or _matrix_shape(tables[i][module]["down_shape"])[1] != in_features
        ]
        if mismatched:
            warnings.append(
                f"{module} has a different shape in {', '.join(mismatched)} and was left "
                "at the first file's version."
            )
            present = [present[0]]

        accumulated = sum(
            int(tables[i][module]["down_shape"][0]) for i in present
        )
        down_suffix, up_suffix = _suffixes_for(first["up"])
        modules.append(
            ModuleMerge(
                name=module,
                sources=present,
                accumulated_rank=accumulated,
                out_features=out_features,
                in_features=in_features,
                down_suffix=down_suffix,
                up_suffix=up_suffix,
                down_shape=first["down_shape"],
            )
        )

    if not any(len(m.sources) > 1 for m in modules):
        warnings.append(
            "These LoRAs have no module in common. The result is valid -- it applies "
            "both -- but nothing is actually being combined."
        )

    return MergePlan(
        sources=sources,
        modules=modules,
        target_rank=int(target_rank),
        output_dtype=output_dtype,
        refusals=refusals,
        warnings=warnings,
    )


# --- The maths -------------------------------------------------------------


def fold_weight(up, down, weight: float, alpha: float, rank: int, xp):
    """Fold a source's own scale into both factors, symmetrically.

    A LoRA's delta is `up @ down * (alpha / rank)`, and the user's weight
    multiplies that. All of it has to end up inside the factors before they are
    concatenated, because after concatenation there is one shared scale for the
    whole stack and no way to give one slice a different one.

    The square root goes on both sides rather than all of it on one: the QR
    step that follows is stable when the two factors have comparable
    magnitudes, and lopsided by a factor of `scale` when they do not. The sign
    cannot be split that way, so it goes on `down` alone -- a negative weight
    subtracts the adapter, which is a thing people do deliberately.
    """
    scale = float(weight) * (float(alpha) / float(rank)) if rank else float(weight)
    magnitude = abs(scale) ** 0.5
    sign = -1.0 if scale < 0 else 1.0
    return up * magnitude, down * (magnitude * sign)


def concat_factors(accumulated, up, down, xp):
    """``(U, V)`` with one more pair stacked on.

    `U` grows along its columns and `V` along its rows, so `U @ V` stays the
    same shape and equals the sum of every pair put into it. This is exact --
    there is no approximation anywhere in this function.
    """
    if accumulated is None:
        return up, down
    big_u, big_v = accumulated
    return (
        xp.concatenate([big_u, up], axis=1),
        xp.concatenate([big_v, down], axis=0),
    )


def compress_factors(big_u, big_v, rank: int, xp, clamp_quantile: float | None = None):
    """Refactor ``(U, V)`` down to `rank` without forming ``U @ V``.

    QR on each side, then one SVD of the small square that is left:

        U = Qu Ru,  V^T = Qv Rv   =>   U V = Qu (Ru Rv^T) Qv^T

    `Ru Rv^T` is `k x k`, where k is the accumulated rank -- 128 x 128 for two
    rank-64 adapters, against the 1280 x 1280 the dense product would be. The
    saving is the whole reason this is affordable on a CPU.

    `clamp_quantile` defaults to **None**, unlike the extractor's 0.99. There
    the input is a checkpoint difference, which can carry a handful of enormous
    singular directions that a trained adapter never would. Here the inputs are
    trained adapters, already bounded by their own training; clamping them
    would quietly alter files the user did not ask to have altered.
    """
    q_u, r_u = xp.linalg.qr(big_u)
    q_v, r_v = xp.linalg.qr(big_v.T)

    small = r_u @ r_v.T
    u, s, vh = xp.linalg.svd(small, full_matrices=False)

    keep = max(1, min(int(rank), int(s.shape[0])))
    up = (q_u @ u[:, :keep]) * s[:keep]
    down = vh[:keep, :] @ q_v.T

    if clamp_quantile:
        spread = xp.concatenate([up.flatten(), down.flatten()])
        hi = xp.quantile(spread, clamp_quantile)
        up = xp.clip(up, -hi, hi)
        down = xp.clip(down, -hi, hi)

    return up, down


def merge_metadata(plan: MergePlan) -> dict[str, str]:
    """Provenance for the file about to be written.

    `format` says `merged-lora` rather than borrowing the extractor's
    `delta-lora`: a merged adapter is not a subtraction, and
    `aux_inspector._extraction_info` reads those fields to describe one. A file
    that claimed to be an extraction would be described as a difference between
    two checkpoints it never saw.
    """
    metadata = {
        "format": "merged-lora",
        "mode": "concat+svd" if plan.target_rank > 0 else "concat",
        "sources": " | ".join(f"{s.name}:{s.weight:g}" for s in plan.sources),
        "source_count": str(len(plan.sources)),
        "target_rank": str(plan.target_rank) if plan.target_rank > 0 else "exact",
        "modules": str(len(plan.modules)),
        "shared_modules": str(plan.shared_modules),
        "output_dtype": plan.output_dtype,
    }
    generations = {s.generation for s in plan.sources if s.generation}
    if len(generations) == 1:
        metadata["anima_generation"] = str(next(iter(generations)))
    return {k: str(v) for k, v in metadata.items()}


def merge_loras(
    plan: MergePlan,
    output_path: str,
    *,
    clamp_quantile: float | None = None,
    device: str = "auto",
    progress_cb=None,
) -> dict[str, Any]:
    """Write the LoRA the plan describes. Requires torch; runs inside Forge.

    One module at a time, through safetensors' lazy reader, so peak memory is a
    few tensors rather than every input file at once.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if not plan.ok:
        raise ValueError("; ".join(plan.refusals) or "Nothing to merge.")

    torch_dtype = {
        "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32
    }[plan.output_dtype]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    readers = [safe_open(source.path, framework="pt", device="cpu") for source in plan.sources]
    alphas = [
        {k: readers[i].get_tensor(k) for k in readers[i].keys() if k.endswith(".alpha")}
        for i in range(len(plan.sources))
    ]
    # Once per file, not once per module: the header is the same all the way
    # through, and re-reading it a few hundred times is minutes of nothing.
    tables = [_module_table(source.path) for source in plan.sources]

    out: dict[str, Any] = {}
    compressed = 0
    for done, module in enumerate(plan.modules):
        accumulated = None
        for index in module.sources:
            reader = readers[index]
            table = tables[index][module.name]
            up = reader.get_tensor(table["up"]).to(device=device, dtype=torch.float32)
            down = reader.get_tensor(table["down"]).to(device=device, dtype=torch.float32)

            rank = int(down.shape[0])
            alpha_tensor = alphas[index].get(module.name + ".alpha")
            alpha = float(alpha_tensor.item()) if alpha_tensor is not None else float(rank)

            up = up.reshape(up.shape[0], -1)
            down = down.reshape(rank, -1)
            up, down = fold_weight(up, down, plan.sources[index].weight, alpha, rank, torch)
            accumulated = concat_factors(accumulated, up, down, torch)

        big_u, big_v = accumulated
        rank_out = module.output_rank(plan.target_rank)
        if plan.target_rank > 0 and rank_out < int(big_v.shape[0]):
            big_u, big_v = compress_factors(
                big_u, big_v, rank_out, torch, clamp_quantile=clamp_quantile
            )
            compressed += 1

        final_rank = int(big_v.shape[0])
        # The scale is already inside the factors, so the loader's alpha/rank
        # factor has to come out as exactly 1.
        out[module.name + "." + module.up_suffix] = big_u.to(torch_dtype).cpu()
        out[module.name + "." + module.down_suffix] = (
            big_v.reshape((final_rank,) + tuple(module.down_shape[1:])).to(torch_dtype).cpu()
        )
        out[module.name + ".alpha"] = torch.tensor(float(final_rank), dtype=torch.float32)

        if progress_cb:
            progress_cb(done + 1, len(plan.modules), module.name)

    metadata = merge_metadata(plan)
    save_file(out, output_path, metadata=metadata)

    return {
        "output_path": output_path,
        "modules": len(plan.modules),
        "compressed_modules": compressed,
        "tensors": len(out),
        "file_size": os.path.getsize(output_path),
        "metadata": metadata,
    }


# --- Presentation ----------------------------------------------------------


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def format_merge_preview_html(plan: MergePlan | None) -> str:
    """What the merge would do, before it is run."""
    if plan is None:
        return ""

    if plan.refusals:
        items = "".join(f"<li>{_esc(r)}</li>" for r in plan.refusals)
        return (
            "<div style='border:1px solid #7f1d1d;background:#1f1416;border-radius:8px;"
            "padding:12px;margin-top:8px;'>"
            "<div style='color:#fca5a5;font-weight:600;margin-bottom:6px;'>This merge cannot run</div>"
            f"<ul style='margin:0 0 0 18px;color:#fecaca;font-size:13px;'>{items}</ul></div>"
        )

    rows = "".join(
        "<tr>"
        f"<td style='padding:3px 10px 3px 0;color:#e5e7eb;'>{_esc(s.name)}</td>"
        f"<td style='padding:3px 10px 3px 0;color:#f97316;'>x {s.weight:g}</td>"
        f"<td style='padding:3px 10px 3px 0;color:#9ca3af;'>{_esc(s.algorithm)}</td>"
        f"<td style='padding:3px 0;color:#9ca3af;'>{s.modules} modules, "
        f"rank {s.uniform_rank if s.uniform_rank else 'mixed'}</td>"
        "</tr>"
        for s in plan.sources
    )

    ranks = [m.output_rank(plan.target_rank) for m in plan.modules]
    rank_text = (
        f"{min(ranks)}-{max(ranks)}" if ranks and min(ranks) != max(ranks)
        else (str(ranks[0]) if ranks else "0")
    )
    mode = (
        f"compressed to rank {plan.target_rank}" if plan.target_rank > 0
        else "exact, no compression"
    )

    warnings = ""
    if plan.warnings:
        items = "".join(f"<li>{_esc(w)}</li>" for w in plan.warnings)
        warnings = (
            "<ul style='margin:8px 0 0 18px;color:#fcd34d;font-size:12px;'>"
            f"{items}</ul>"
        )

    return (
        "<div style='border:1px solid #374151;background:#111827;border-radius:8px;"
        "padding:12px;margin-top:8px;'>"
        f"<table style='font-size:13px;border-collapse:collapse;'>{rows}</table>"
        "<div style='margin-top:10px;font-size:13px;color:#d1d5db;'>"
        f"<b>{len(plan.modules)}</b> modules out "
        f"(<b>{plan.shared_modules}</b> in more than one file, "
        f"<b>{plan.exclusive_modules}</b> from a single file) &middot; "
        f"output rank <b>{_esc(rank_text)}</b> &mdash; {mode} &middot; "
        f"about <b>{_human_bytes(plan.estimated_bytes())}</b> in {plan.output_dtype}"
        "</div>"
        f"{warnings}"
        "</div>"
    )
