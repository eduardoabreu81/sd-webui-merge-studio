"""Extract a LoRA from the difference between two checkpoints.

`lora_bake.py` bakes an adapter into a checkpoint; this is the inverse. The
subtraction itself is trivial -- what decides whether the result is usable is
everything around it: which tensors can be decomposed at all, which pairs are
comparable, and which inputs make the whole operation meaningless.

This module answers that before any tensor is read. `plan_extraction` works
from safetensors headers alone, so the interface can show what a run would do
the moment two checkpoints are picked, rather than after minutes of I/O. A
plan that comes back not-ok names its reasons; it never guesses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .aux_inspector import detect_unsupported_storage
from .checkpoint_inspector import (
    ARCH_UNKNOWN,
    _esc,
    infer_architecture_id,
    read_safetensors_header,
)

#: Bytes per element for the dtypes an extraction can write.
_OUTPUT_DTYPE_WIDTHS = {"BF16": 2, "F16": 2, "F32": 4}

#: Storage dtypes a delta can be taken in. Everything else is either a
#: quantized payload or an index, and subtracting it element-wise produces
#: noise rather than a difference.
_FLOAT_DTYPES = frozenset({"F32", "F16", "BF16", "F64"})

KIND_LINEAR = "linear"
KIND_CONV = "conv"
KIND_VECTOR = "vector"
KIND_SKIP = "skip"

#: Namespaces a diffusion model is stored under, longest first. The read side
#: accepts any of them; the pairs are documented in full in
#: `source_precision.KEY_PREFIX_GROUPS`.
_DIFFUSION_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "net.")

#: Components an extraction does not touch. A VAE has no LoRA convention worth
#: emitting, and a text encoder's does not go through the UNet key map -- so
#: rather than emit names nothing loads, these are counted and reported as left
#: out. Anima's llm_adapter is the exception that proves it: it lives under the
#: diffusion namespace, and Forge itself moves it to `text_encoders.qwen3_06b`
#: at load time, so we emit its natural name and let the loader relocate it.
_OUT_OF_SCOPE_PREFIXES = (
    "first_stage_model.",
    "vae.",
    "cond_stage_model.",
    "conditioner.embedders.",
    "text_encoders.",
)


def _strip_diffusion_prefix(key: str) -> str:
    """A key without whichever diffusion namespace it happens to use."""
    for prefix in _DIFFUSION_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _by_stripped_name(entries) -> dict[str, str]:
    """`{name under the prefix: the key as written}`.

    Shorter than `entries` when a file carries the same tensor under two
    prefixes, which the caller treats as a refusal rather than picking one.
    """
    return {_strip_diffusion_prefix(key): key for key in entries}


def lora_key_base(key: str) -> str | None:
    """The name Forge's loader will look for, or None if out of scope.

    Forge's LoRA adapter tries seven naming conventions per module and
    `<base>.lora_up.weight` is the first of them, so that is what we emit. The
    `<base>` itself is the loader's own universal form -- ComfyUI's key map
    always registers `diffusion_model.<path>` alongside whatever exotic names
    an architecture also accepts, which is why a plainly-named adapter loads
    everywhere.
    """
    for prefix in _OUT_OF_SCOPE_PREFIXES:
        if key.startswith(prefix):
            return None

    name = key
    for prefix in _DIFFUSION_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    for suffix in (".weight", ".bias"):
        if name.endswith(suffix):
            return "diffusion_model." + name[: -len(suffix)]
    return None


def classify_shape(shape) -> str:
    """What can be done with a tensor of this shape.

    A 2-D weight decomposes directly. A 4-D conv kernel decomposes once it is
    flattened. A 1-D bias or norm has no rank to truncate, so it can only be
    carried as a raw difference. Anything else -- scalars, 3-D oddities -- has
    no established LoRA representation and is left alone rather than guessed at.
    """
    dims = len(tuple(shape))
    if dims == 2:
        return KIND_LINEAR
    if dims == 4:
        return KIND_CONV
    if dims == 1:
        return KIND_VECTOR
    return KIND_SKIP


def _matrix_dims(shape) -> tuple[int, int]:
    """``(out, in)`` for the 2-D matrix a weight decomposes as.

    A conv kernel is flattened the way the SVD path flattens it: the output
    channels stay, everything else folds into the input side.
    """
    shape = tuple(shape)
    out_dim = shape[0]
    in_dim = 1
    for dim in shape[1:]:
        in_dim *= dim
    return out_dim, in_dim


def effective_rank(shape, *, rank: int, conv_rank: int) -> int:
    """The rank actually used for this tensor.

    A rank above either dimension of the matrix is not just wasteful, it is
    undefined: the SVD cannot return more singular values than the smaller
    dimension. Conv kernels take their own rank, because a 3x3 kernel carries
    far less independent signal than a wide linear layer of the same footprint.
    """
    kind = classify_shape(shape)
    requested = conv_rank if kind == KIND_CONV else rank
    out_dim, in_dim = _matrix_dims(shape)
    return max(1, min(int(requested), out_dim, in_dim))


def factor_elements(shape, rank: int) -> int:
    """How many elements the ``up`` and ``down`` factors hold together."""
    out_dim, in_dim = _matrix_dims(shape)
    return out_dim * rank + rank * in_dim


def _numel(shape) -> int:
    count = 1
    for dim in tuple(shape):
        count *= dim
    return count


def _default_xp():
    """The array library to decompose with.

    torch inside Forge, numpy anywhere else. Both expose `linalg.svd`, `diag`,
    `quantile`, `clip` and `concatenate` under the same names with the same
    argument order, so the decomposition is written once and runs on either --
    which is what lets the maths be tested outside a Forge install.
    """
    try:
        import torch

        return torch
    except ImportError:
        import numpy

        return numpy


def svd_factors(matrix, rank: int, *, clamp_quantile: float | None = 0.99, xp=None):
    """``(up, down)`` for a delta matrix, by truncated SVD.

    All of the singular scale goes into `up`, which is what makes `alpha = rank`
    correct and the loader's `alpha / rank` factor exactly 1.

    `clamp_quantile` bounds both factors at that percentile of their combined
    values. A checkpoint delta is not a trained adapter: it can carry a handful
    of enormous singular directions that a trained LoRA never would, and
    reproducing them faithfully is how an extraction ends up with burnt
    highlights. Pass None to keep the decomposition exact -- the tests do, in
    order to assert a known delta comes back unchanged.
    """
    if xp is None:
        xp = _default_xp()

    rank = max(1, min(int(rank), int(matrix.shape[0]), int(matrix.shape[1])))
    u, s, vh = xp.linalg.svd(matrix, full_matrices=False)

    up = u[:, :rank] @ xp.diag(s[:rank])
    down = vh[:rank, :]

    if clamp_quantile:
        spread = xp.concatenate([up.flatten(), down.flatten()])
        hi = xp.quantile(spread, clamp_quantile)
        up = xp.clip(up, -hi, hi)
        down = xp.clip(down, -hi, hi)

    return up, down


@dataclass(frozen=True)
class ModulePlan:
    """One tensor, and what the extraction would do with it."""

    key: str
    kind: str
    shape: tuple[int, ...]
    rank: int
    elements: int
    #: The tuned file's name for the same tensor. Empty when the two agree,
    #: which they do not when one side writes `net.` and the other
    #: `model.diffusion_model.` -- the Anima base releases against their own
    #: republished finetunes.
    key_tuned: str = ""


@dataclass
class ExtractionPlan:
    """What a run would produce, and why it might not run at all."""

    original_path: str = ""
    tuned_path: str = ""
    architecture_original: str = ARCH_UNKNOWN
    architecture_tuned: str = ARCH_UNKNOWN
    prefix_original: str = ""
    prefix_tuned: str = ""
    rank: int = 0
    conv_rank: int = 0
    output_dtype: str = "BF16"

    total_original: int = 0
    total_tuned: int = 0
    shared: int = 0
    only_original: int = 0
    only_tuned: int = 0
    shape_mismatch: int = 0
    out_of_scope: int = 0

    modules: list[ModulePlan] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.refusals

    @property
    def linear_count(self) -> int:
        return sum(1 for m in self.modules if m.kind == KIND_LINEAR)

    @property
    def conv_count(self) -> int:
        return sum(1 for m in self.modules if m.kind == KIND_CONV)

    @property
    def vector_count(self) -> int:
        return sum(1 for m in self.modules if m.kind == KIND_VECTOR)

    @property
    def decomposable_count(self) -> int:
        return self.linear_count + self.conv_count

    @property
    def estimated_bytes(self) -> int:
        """Size of the resulting file, from the shapes alone.

        Exact for the tensors, give or take the header: a factor pair's size is
        fixed by its rank, and a raw difference is the size of the tensor it
        came from. Shown before the run because rank is the one knob whose cost
        the user cannot estimate by eye.
        """
        width = _OUTPUT_DTYPE_WIDTHS.get(self.output_dtype, 2)
        return sum(m.elements for m in self.modules) * width


def _tensor_entries(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        k: v
        for k, v in header.items()
        if k != "__metadata__" and isinstance(v, dict) and "shape" in v
    }


def _dominant_prefix(entries: dict[str, dict[str, Any]]) -> str:
    """Which diffusion namespace this file stores its model under.

    Recorded in the output's metadata because it is not cosmetic: the same
    Anima generation ships as `net.` from one publisher and
    `model.diffusion_model.` from another, and knowing which side a delta was
    taken from is what makes it reproducible later.
    """
    counts = {prefix: 0 for prefix in _DIFFUSION_PREFIXES}
    for key in entries:
        for prefix in _DIFFUSION_PREFIXES:
            if key.startswith(prefix):
                counts[prefix] += 1
                break
    best = max(counts, key=lambda p: counts[p])
    return best if counts[best] else ""


def _quantization_refusal(entries: dict[str, dict[str, Any]], label: str) -> str | None:
    """Why this file's weights cannot be subtracted, or None.

    Checked here rather than through the inspector's precision label because
    the question is narrower than "how is this stored": a delta needs floating
    point on both sides. An INT8 or FP8 payload subtracted element-wise yields
    noise shaped like a tensor, and an SVD of noise is a confident, useless
    file -- the worst possible failure, because it looks like it worked.
    """
    if any(k.endswith(".comfy_quant") for k in entries):
        return f"{label} is quantized (comfy_quant); a delta needs unquantized weights on both sides."

    offenders = {
        v.get("dtype", "?")
        for k, v in entries.items()
        if len(v.get("shape", ())) >= 2 and v.get("dtype") not in _FLOAT_DTYPES
    }
    if offenders:
        found = ", ".join(sorted(offenders))
        return f"{label} stores weights as {found}; a delta needs FP32, FP16 or BF16."
    return None


def plan_extraction(
    original_path: str,
    tuned_path: str,
    *,
    rank: int = 64,
    conv_rank: int | None = None,
    output_dtype: str = "BF16",
) -> ExtractionPlan:
    """What extracting ``tuned - original`` would do, read from headers only.

    Never raises on bad input: an unreadable or incompatible pair comes back as
    a plan whose ``refusals`` say why, so the interface can show the reason in
    the same place it would have shown the result.
    """
    if conv_rank is None:
        conv_rank = max(1, min(rank, 32))

    plan = ExtractionPlan(
        original_path=original_path,
        tuned_path=tuned_path,
        rank=int(rank),
        conv_rank=int(conv_rank),
        output_dtype=output_dtype if output_dtype in _OUTPUT_DTYPE_WIDTHS else "BF16",
    )

    if os.path.abspath(original_path) == os.path.abspath(tuned_path):
        plan.refusals.append("Both sides are the same file; the difference would be empty.")
        return plan

    headers = {}
    for label, path in (("Original", original_path), ("Tuned", tuned_path)):
        try:
            header, _ = read_safetensors_header(path)
        except Exception as exc:
            plan.refusals.append(f"{label} could not be read: {exc}")
            continue
        headers[label] = header

        blocked = detect_unsupported_storage(header, path)
        if blocked:
            plan.refusals.append(f"{label}: {blocked}")

    if len(headers) < 2:
        return plan

    entries = {label: _tensor_entries(h) for label, h in headers.items()}
    for label in ("Original", "Tuned"):
        refusal = _quantization_refusal(entries[label], label)
        if refusal:
            plan.refusals.append(refusal)

    plan.architecture_original = infer_architecture_id(
        headers["Original"], os.path.basename(original_path)
    )
    plan.architecture_tuned = infer_architecture_id(
        headers["Tuned"], os.path.basename(tuned_path)
    )
    named = {plan.architecture_original, plan.architecture_tuned} - {ARCH_UNKNOWN}
    if len(named) > 1:
        plan.refusals.append(
            f"Different architectures: {plan.architecture_original} and {plan.architecture_tuned}."
        )

    original_entries, tuned_entries = entries["Original"], entries["Tuned"]
    plan.total_original = len(original_entries)
    plan.total_tuned = len(tuned_entries)
    plan.prefix_original = _dominant_prefix(original_entries)
    plan.prefix_tuned = _dominant_prefix(tuned_entries)

    # Matched on the name under the diffusion prefix, not on the literal key.
    # `anima_baseV10` writes `net.blocks.0...` and `anima_turboV11` writes
    # `model.diffusion_model.blocks.0...` for the same 685 tensors; a literal
    # intersection of those is empty, and the extraction refused two files
    # that are structurally identical.
    original_by_base = _by_stripped_name(original_entries)
    tuned_by_base = _by_stripped_name(tuned_entries)
    for label, mapping, source in (
        ("Original", original_by_base, original_entries),
        ("Tuned", tuned_by_base, tuned_entries),
    ):
        if len(mapping) < len(source):
            plan.refusals.append(
                f"{label} carries the same tensor under more than one diffusion "
                "prefix, so there is no single name to match on."
            )

    shared_bases = original_by_base.keys() & tuned_by_base.keys()
    plan.shared = len(shared_bases)
    plan.only_original = plan.total_original - plan.shared
    plan.only_tuned = plan.total_tuned - plan.shared

    for base_name in sorted(shared_bases):
        key = original_by_base[base_name]
        key_tuned = tuned_by_base[base_name]
        if lora_key_base(key) is None:
            # A VAE tensor, a text encoder outside the diffusion namespace, or
            # a key with no weight/bias suffix. Counted so the preview can say
            # so, rather than extracted under a name nothing would load.
            plan.out_of_scope += 1
            continue

        shape_a = tuple(original_entries[key].get("shape", ()))
        shape_b = tuple(tuned_entries[key_tuned].get("shape", ()))
        if shape_a != shape_b:
            # A tensor that changed shape is not a tuned version of the same
            # weight, whatever its name says. Counted so the preview can show
            # it, never extracted.
            plan.shape_mismatch += 1
            continue

        kind = classify_shape(shape_a)
        if kind == KIND_SKIP:
            continue
        if kind == KIND_VECTOR:
            plan.modules.append(
                ModulePlan(
                    key=key, kind=kind, shape=shape_a, rank=0,
                    elements=_numel(shape_a), key_tuned=key_tuned,
                )
            )
            continue

        r = effective_rank(shape_a, rank=plan.rank, conv_rank=plan.conv_rank)
        plan.modules.append(
            ModulePlan(
                key=key, kind=kind, shape=shape_a, rank=r,
                elements=factor_elements(shape_a, r), key_tuned=key_tuned,
            )
        )

    if plan.decomposable_count == 0 and not plan.refusals:
        plan.refusals.append(
            "No weight matrices in common; these two files have nothing comparable to subtract."
        )

    if plan.shared and plan.shape_mismatch:
        plan.warnings.append(
            f"{plan.shape_mismatch} shared tensors differ in shape and will be left out."
        )
    if plan.shared and plan.total_original and plan.shared / plan.total_original < 0.5:
        plan.warnings.append(
            f"Only {plan.shared} of {plan.total_original} tensors are shared; the result will be partial."
        )

    return plan


# --- Writing ---------------------------------------------------------------


def extraction_metadata(
    plan: ExtractionPlan,
    *,
    min_diff: float = 0.0,
    clamp_quantile: float | None = 0.99,
) -> dict[str, str]:
    """Provenance for the file we are about to write.

    The field names are not ours to choose: `aux_inspector._extraction_info`
    already recognises a LoRA that was subtracted rather than trained, and reads
    exactly these. An extraction that used different names would open in our own
    inspector as an ordinary trained adapter -- the one thing it is not.
    """
    metadata = {
        "format": "delta-lora",
        "mode": "svd",
        "base_prefix": plan.prefix_original,
        "target_prefix": plan.prefix_tuned,
        "subtraction_dtype": "F32",
        "output_dtype": plan.output_dtype,
        "rank": str(plan.rank),
        "conv_rank": str(plan.conv_rank),
        "base_model": os.path.basename(plan.original_path),
        "target_model": os.path.basename(plan.tuned_path),
        "architecture": plan.architecture_tuned,
        "modules": str(len(plan.modules)),
        "min_diff": str(min_diff),
    }
    if clamp_quantile:
        metadata["clamp_quantile"] = str(clamp_quantile)
    return {k: str(v) for k, v in metadata.items()}


#: Default difference floor, as a share of the weight's own magnitude.
#:
#: Chosen from a measurement rather than picked. Across all 685 modules of
#: `anima_turboV11 - anima_baseV10`: 177 are bit-identical, 60 more move by
#: around 1e-10 of their own magnitude -- BF16 round-trip noise -- and the
#: remaining 448 move by a median of 0.6%, the largest by 2.6%. Between
#: 3e-10 and 6e-3 there is nothing at all, so anything inside that gap
#: separates "untouched" from "tuned" cleanly. 1e-5 sits four orders above
#: the noise and three below the signal.
DEFAULT_MIN_DIFF = 1e-5


def extract_lora(
    plan: ExtractionPlan,
    output_path: str,
    *,
    min_diff: float = DEFAULT_MIN_DIFF,
    clamp_quantile: float | None = 0.99,
    device: str = "auto",
    progress_cb=None,
) -> dict[str, Any]:
    """Write the LoRA the plan describes. Requires torch; runs inside Forge.

    Reads one tensor pair at a time through safetensors' lazy reader rather
    than loading both checkpoints. Peak memory is therefore a couple of layers,
    not two models -- which is the difference between this running beside a
    loaded Forge and the 64 GB the established tools ask for.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if not plan.ok:
        raise ValueError("; ".join(plan.refusals))

    torch_dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[
        plan.output_dtype
    ]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out: dict[str, torch.Tensor] = {}
    skipped = 0
    total = len(plan.modules)

    with torch.no_grad():
        with safe_open(plan.original_path, framework="pt") as f_orig, safe_open(
            plan.tuned_path, framework="pt"
        ) as f_tuned:
            for index, module in enumerate(plan.modules):
                base = lora_key_base(module.key)
                if base is None:
                    continue

                original = f_orig.get_tensor(module.key).to(torch.float32)
                delta = (
                    f_tuned.get_tensor(module.key_tuned or module.key).to(torch.float32)
                    - original
                )

                # A module the tuning barely moved contributes noise, not
                # signal. Decomposing it anyway is how an extraction ends up
                # larger and worse than one that left it alone.
                #
                # Relative to the weight, not absolute. An absolute floor is a
                # statement about a magnitude scale, and architectures do not
                # share one: measured on `anima_turboV11 - anima_baseV10`, the
                # largest mean absolute delta in the whole model is 1.4e-4, so
                # the old default of 1e-4 discarded 622 of 685 modules and
                # 3e-4 would have discarded every one of them. As a share of
                # the weight the same deltas are a median 0.3%, which is a
                # number that means the same thing on any model.
                reference = float(original.abs().mean())
                del original
                moved = float(delta.abs().mean())
                if moved <= 0.0 or (reference > 0.0 and moved < min_diff * reference):
                    skipped += 1
                    del delta
                    if progress_cb:
                        progress_cb(index + 1, total, module.key)
                    continue

                if module.kind == KIND_VECTOR:
                    suffix = ".diff_b" if module.key.endswith(".bias") else ".diff"
                    out[base + suffix] = delta.to(torch_dtype).contiguous()
                    del delta
                    if progress_cb:
                        progress_cb(index + 1, total, module.key)
                    continue

                original_shape = delta.shape
                matrix = delta.flatten(start_dim=1) if module.kind == KIND_CONV else delta
                up, down = svd_factors(
                    matrix.to(device), module.rank, clamp_quantile=clamp_quantile, xp=torch
                )

                if module.kind == KIND_CONV:
                    # The loader expects conv factors shaped as convolutions:
                    # up is a 1x1 over the rank, down keeps the real kernel.
                    up = up.reshape(original_shape[0], module.rank, 1, 1)
                    down = down.reshape(module.rank, *original_shape[1:])

                out[base + ".lora_up.weight"] = up.to("cpu", torch_dtype).contiguous()
                out[base + ".lora_down.weight"] = down.to("cpu", torch_dtype).contiguous()
                out[base + ".alpha"] = torch.tensor(float(module.rank), dtype=torch_dtype)

                del delta, matrix, up, down
                if progress_cb:
                    progress_cb(index + 1, total, module.key)

    metadata = extraction_metadata(plan, min_diff=min_diff, clamp_quantile=clamp_quantile)
    metadata["skipped_below_min_diff"] = str(skipped)
    save_file(out, output_path, metadata=metadata)

    return {
        "path": output_path,
        "tensors": len(out),
        "skipped": skipped,
        "bytes": os.path.getsize(output_path),
    }


# --- Presentation ----------------------------------------------------------
#
# The preview exists because a checkpoint difference is invisible: two files go
# in, one file comes out, and nothing on screen says which parts of the model
# the operation actually touched. Established merge tools ask for a rank and a
# button press and reveal the answer only in the output. Showing the plan first
# is the difference between a tool and a slot machine.


def _human_bytes(n: int) -> str:
    """Size for a file that may be tens of KB or several GB.

    `aux_inspector._human_size` starts at MB, which reads as "0.0 MB" for a
    small adapter -- exactly the range a low-rank extraction lands in.
    """
    for unit, size in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if n >= size:
            return f"{n / size:,.1f} {unit}"
    return f"{n} B"


def _group_for(key: str) -> str:
    """The part of the model a tensor belongs to, for the preview's breakdown.

    Block indices collapse: forty entries for `blocks.N` say nothing that one
    line reading "blocks, 40 modules" does not. What must not collapse is a
    separate component sharing the word -- Anima's `llm_adapter.blocks.N` is
    the text-encoder side and is counted apart from the DiT's own blocks.
    """
    name = key
    for prefix in _DIFFUSION_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    for part in name.split("."):
        if part and not part.isdigit():
            return part
    return "other"


@dataclass(frozen=True)
class GroupSummary:
    """One region of the model, and how much of the output it accounts for."""

    name: str
    count: int
    elements: int


def group_summary(modules: list[ModulePlan]) -> list[GroupSummary]:
    """Modules folded into their regions, largest first."""
    totals: dict[str, list[int]] = {}
    for module in modules:
        entry = totals.setdefault(_group_for(module.key), [0, 0])
        entry[0] += 1
        entry[1] += module.elements
    return sorted(
        (GroupSummary(name, count, elements) for name, (count, elements) in totals.items()),
        key=lambda g: (-g.elements, g.name),
    )


def _panel(border: str, background: str, body: str) -> str:
    return (
        f"<div style='margin-top: 6px; padding: 8px 10px; border-radius: 4px; "
        f"background: {background}; border: 1px solid {border}; font-size: 12px; line-height: 1.6;'>"
        f"{body}</div>"
    )


def _stat(label: str, value: str, color: str = "#e5e7eb") -> str:
    return (
        f"<span style='display: inline-block; margin-right: 18px;'>"
        f"<span style='color: #9ca3af;'>{label}</span> "
        f"<b style='color: {color};'>{value}</b></span>"
    )


def format_extraction_preview_html(plan: ExtractionPlan) -> str:
    """What the run would do, or why it will not run.

    Every value here comes out of a downloaded .safetensors header, so it is
    escaped on the way in -- the same rule the recipe and LoRA dashboards
    follow.
    """
    if not plan.ok:
        reasons = "".join(f"<li>{_esc(r)}</li>" for r in plan.refusals)
        return _panel(
            "#ef4444",
            "rgba(239, 68, 68, 0.15)",
            f"<b style='color: #fca5a5;'>Cannot extract</b>"
            f"<ul style='margin: 4px 0 0 0; padding-left: 18px; color: #fca5a5;'>{reasons}</ul>",
        )

    arch_original = _esc(plan.architecture_original)
    arch_tuned = _esc(plan.architecture_tuned)
    architecture = arch_original if arch_original == arch_tuned else f"{arch_original} &rarr; {arch_tuned}"

    head = (
        f"<div style='margin-bottom: 6px;'>"
        f"<b style='color: #10b981;'>Ready to extract</b> &nbsp;&bull;&nbsp; "
        f"<b>{architecture}</b> &nbsp;&bull;&nbsp; "
        f"<span style='color: #9ca3af;'>{plan.shared} of {plan.total_tuned} tensors shared</span>"
        f"</div>"
    )

    stats = (
        "<div style='margin-bottom: 6px;'>"
        + _stat("Decomposed", f"{plan.linear_count + plan.conv_count}", "#38bdf8")
        + (_stat("of which conv", f"{plan.conv_count}", "#38bdf8") if plan.conv_count else "")
        + _stat("Raw differences", f"{plan.vector_count}")
        + _stat("Rank", f"{plan.rank}")
        + _stat("Estimated size", _human_bytes(plan.estimated_bytes), "#10b981")
        + "</div>"
    )

    # Rendered as rows of spans rather than a <table>: Gradio's stylesheet puts
    # borders and cell padding on every table in the page, which turns a
    # three-line breakdown into something that reads like a spreadsheet.
    width = _OUTPUT_DTYPE_WIDTHS.get(plan.output_dtype, 2)
    rows = "".join(
        f"<div style='display: flex; gap: 12px; font-size: 11.5px; line-height: 1.7;'>"
        f"<span style='color: #e5e7eb; min-width: 110px;'>{_esc(g.name)}</span>"
        f"<span style='color: #9ca3af; min-width: 90px;'>{g.count} modules</span>"
        f"<span style='color: #9ca3af;'>{_human_bytes(g.elements * width)}</span>"
        f"</div>"
        for g in group_summary(plan.modules)
    )
    breakdown = (
        "<div style='color: #9ca3af; margin: 4px 0 2px 0;'>What is being subtracted</div>"
        f"{rows}"
    )

    excluded = []
    if plan.only_tuned:
        excluded.append(f"{plan.only_tuned} only in the tuned model")
    if plan.only_original:
        excluded.append(f"{plan.only_original} only in the original")
    if plan.shape_mismatch:
        excluded.append(f"{plan.shape_mismatch} with mismatched shapes")
    if plan.out_of_scope:
        excluded.append(f"{plan.out_of_scope} outside the diffusion model (VAE / text encoder)")
    excluded_line = (
        f"<div style='margin-top: 6px; color: #9ca3af;'>Left out: {_esc(', '.join(excluded))}</div>"
        if excluded
        else ""
    )

    html = _panel("rgba(16, 185, 129, 0.4)", "rgba(16, 185, 129, 0.08)", head + stats + breakdown + excluded_line)

    for warning in plan.warnings:
        html += _panel(
            "#f59e0b",
            "rgba(245, 158, 11, 0.15)",
            f"<span style='color: #fcd34d;'>{_esc(warning)}</span>",
        )
    return html
