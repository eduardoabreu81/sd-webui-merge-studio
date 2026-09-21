"""The merge modes: what each one is, what it needs, and its arithmetic.

Separated from `checkpoint_merge` because that module imports torch, Forge's
backend and the WebUI's `modules` at import time, none of which exist outside a
running Forge -- the same reason `plan_merge_composition` lives in
`component_bundle`. Everything here is declarations and arithmetic over
whatever tensor-like objects it is handed, so it can be tested on this side.

One table, three consumers. The merge loop asks which models a mode needs, the
interface asks which fields to show, and the inspector asks what to call it.
They used to answer separately and drifted apart -- which is how a Sum Twice
merge came to display as the name of the tool that wrote it.
"""

from __future__ import annotations

import contextlib
import html
from dataclasses import dataclass
from functools import lru_cache


class MergeError(RuntimeError):
    pass


INTERP_NO_INTERPOLATION = "no_interpolation"
INTERP_WEIGHTED_SUM = "weighted_sum"
INTERP_ADD_DIFFERENCE = "add_difference"
INTERP_SUM_TWICE = "sum_twice"
INTERP_SIMILARITY_ADD_DIFFERENCE = "similarity_add_difference"
INTERP_DARE = "dare"


@dataclass(frozen=True)
class MergeMode:
    """One merge mode, defined once for everything that needs to know it.

    Three consumers used to answer these questions separately: the merge loop
    asked "does this need C", the interface asked "which fields do I show",
    and the inspector asked "what do I call this". They drifted -- which is how
    a Sum Twice merge came to display as the name of the tool that wrote it.

    `formula` is written in the notation SuperMerger publishes in its README.
    A published equation is not code and carries no licence; the implementation
    below is written from the equation, not from theirs.
    """

    key: str
    label: str
    formula: str
    description: str
    needs_b: bool
    needs_c: bool
    needs_beta: bool
    beta_label: str = ""
    #: Whether this mode draws random numbers. A merge that cannot be
    #: repeated makes the recipe a lie, so the modes that do get a seed and
    #: record it.
    needs_seed: bool = False


MERGE_MODES: tuple[MergeMode, ...] = (
    MergeMode(
        key=INTERP_NO_INTERPOLATION,
        label="No Interpolation",
        formula="A",
        description="Passes Model A straight through. For converting or quantizing precision.",
        needs_b=False,
        needs_c=False,
        needs_beta=False,
    ),
    MergeMode(
        key=INTERP_WEIGHTED_SUM,
        label="Weighted Sum",
        formula="(1 - α)A + αB",
        description="Blends two models. At α = 0 you get A, at α = 1 you get B.",
        needs_b=True,
        needs_c=False,
        needs_beta=False,
    ),
    MergeMode(
        key=INTERP_ADD_DIFFERENCE,
        label="Add Difference",
        formula="A + α(B - C)",
        description=(
            "Adds what B learned on top of C onto A. C is the base both were "
            "trained from; get it wrong and you add the difference between two "
            "unrelated models."
        ),
        needs_b=True,
        needs_c=True,
        needs_beta=False,
    ),
    MergeMode(
        key=INTERP_SUM_TWICE,
        label="Sum Twice",
        formula="(1 - β)((1 - α)A + αB) + βC",
        description=(
            "Blends A with B, then blends that result with C. Three models in "
            "one pass, with α setting the first blend and β the second."
        ),
        needs_b=True,
        needs_c=True,
        needs_beta=True,
        beta_label="Second blend (β) — how much of Model C",
    ),
    MergeMode(
        key=INTERP_SIMILARITY_ADD_DIFFERENCE,
        label="Similarity Add Difference",
        formula="(1 - s)(A + α(B - C)) + s((1 - α/2)A + (α/2)B),  s = β · cos(A, B)",
        description=(
            "Add Difference that holds back where A and B already agree. Each "
            "tensor is judged on its own: where the two models point the same "
            "way it leans towards a plain average, and where they diverge it "
            "adds the full difference. β sets how strong that pull is."
        ),
        needs_b=True,
        needs_c=True,
        needs_beta=True,
        beta_label="Similarity pull (β) — how much agreement softens the add",
    ),
    MergeMode(
        key=INTERP_DARE,
        label="DARE",
        formula="A + α · (mask ⊙ (B - A)) / (1 - β),  mask ~ Bernoulli(1 - β)",
        description=(
            "Drops a share of the difference at random and rescales what "
            "survives, so the result keeps the magnitude of the change with "
            "fewer of the individual edits. β is the drop rate: at 0.9 nine "
            "tenths of the difference is thrown away and the rest multiplied "
            "by ten."
        ),
        needs_b=True,
        needs_c=False,
        needs_beta=True,
        beta_label="Drop rate (β) — share of the difference thrown away",
        needs_seed=True,
    ),
)

MERGE_MODES_BY_KEY: dict[str, MergeMode] = {mode.key: mode for mode in MERGE_MODES}


def merge_mode(interp_method: str) -> MergeMode:
    """The mode record for a key, or a `MergeError` naming what was asked for."""
    try:
        return MERGE_MODES_BY_KEY[interp_method]
    except KeyError:
        raise MergeError(f"Unknown interpolation method: {interp_method}") from None



def _lerp(w_a, w_b, weight):
    """`(1 - t)a + tb`, using the tensor's own `lerp` when it has one.

    torch's `lerp` writes one output where the arithmetic writes three, which
    on a 15 GB diffusion model is the difference between a merge fitting in
    memory and not. Anything without it -- numpy, a test stand-in -- gets the
    arithmetic.
    """
    native = getattr(w_a, "lerp", None)
    if native is not None:
        return native(w_b, weight)
    return w_a + weight * (w_b - w_a)


def make_seeded_rand(seed, xp, device=None):
    """A `rand_like` drawing from a generator this seed owns.

    Every other mode is deterministic and a recipe exists to repeat a merge,
    so a DARE merge that came out differently each run would make its own
    recipe a lie. One generator per merge, advanced tensor by tensor in the
    order `named_modules()` yields -- which is itself deterministic -- so the
    same seed reproduces the whole file.

    `None` back for a `None` seed, which means "use the module's global
    stream" and gives a different merge every time.
    """
    if seed is None:
        return None

    if hasattr(xp, "Generator"):  # torch
        generator = xp.Generator(device=device or "cpu").manual_seed(int(seed))

        def rand(tensor):
            return xp.rand(
                tensor.shape,
                generator=generator,
                device=generator.device,
                dtype=xp.float32,
            ).to(tensor.device)

        return rand

    stream = xp.random.default_rng(int(seed))
    return lambda tensor: stream.random(tensor.shape)


def _keep_mask(tensor, drop_rate: float, xp, rand=None):
    """A Bernoulli keep-mask the shape and dtype of `tensor`.

    1 where the element survives, 0 where it is dropped, so it multiplies
    straight through. Drawn once per tensor, which is what makes two DARE
    merges differ unless they share a seed.
    """
    if rand is not None:
        draw = rand(tensor)
    elif hasattr(xp, "rand_like"):
        draw = xp.rand_like(tensor)
    else:
        draw = xp.random.random(tensor.shape)
    mask = draw >= drop_rate
    cast = getattr(mask, "to", None)
    return cast(tensor.dtype) if cast is not None else mask.astype(tensor.dtype)


def delta_write(w_a, w_b, w_kept, ratio: float):
    """The Delta Mix write rule for one tensor: `insert += r * (donor - kept)`.

    Not a merge mode, and deliberately not in MERGE_MODES: a mode answers how
    two models blend across the whole checkpoint, while this answers what may
    be written into a block that exists in only one of them. It is reached
    solely through the cross-generation Anima path in `_merge_module_tree`.

    What it is NOT is `lerp(insert, donor, r)`. The donor block lives at a
    different depth in a smaller model, so averaging the two puts a foreign
    layer's absolute weights into the niche. Subtracting `kept` -- the floor
    of Model A that shares this insert's origin -- leaves only the direction
    the donor moved during its own finetune, which is a quantity the two
    models do agree on.

    `None` back means there is no floor of the matching shape to subtract, and
    the caller leaves the block at A's weights. A missing floor is not a zero
    floor: dropping the subtraction would write the donor's absolute weights,
    the exact failure the rule exists to prevent.
    """
    if w_kept is None or getattr(w_kept, "shape", None) != getattr(w_a, "shape", None):
        return None
    return w_a + ratio * (w_b - w_kept)


def blend_tensors(
    interp_method: str,
    alpha: float,
    beta: float,
    w_a,
    w_b,
    w_c=None,
    *,
    xp=None,
    rand=None,
):
    """One tensor's worth of merge math, for every mode.

    Weights and biases used to carry their own copy of this, which is how the
    bias path came to support two modes while the weight path supported two
    different ones. `None` back means the mode wanted a third model and this
    module has none -- the caller skips it rather than merging half of it.

    `rand` is a seeded `rand_like` from `make_seeded_rand`, for the modes that
    draw random numbers. Without one they use the module's global stream and
    the merge is different every run.

    `xp` is the array module -- torch inside Forge, numpy in a test. The same
    parameterisation `lora_extract.svd_factors` uses, and for the same reason:
    torch is not installed on the development machine, so a formula that can
    only run inside Forge is a formula nothing checks.

    Where each formula comes from, because two of them are ports:

    * Weighted Sum, Add Difference, Sum Twice -- SuperMerger publishes these
      in LaTeX in its README. An equation is not an implementation and carries
      no licence; these lines are written from the equations.
    * Similarity Add Difference -- ported from `s1dlx/meh`
      (`sd_meh/merge_methods.py::similarity_add_difference`), MIT,
      Copyright (c) 2023 s1dlx.
    * DARE -- "Language Models are Super Mario" (arXiv:2311.03099), by way of
      `martyn/safetensors-merge-supermario`, MIT. **Written from the paper,
      not copied from that repo, because the two disagree:** the paper drops
      with probability `p` and rescales the survivors by `1 / (1 - p)`, while
      `merge.py::merge_tensors` builds its mask with `binomial(1, p)` -- which
      *keeps* with probability `p` -- and then rescales by `1 / (1 - p)`
      anyway. That is only unbiased at `p = 0.5`. This follows the paper, so
      beta is the drop rate and beta = 0 is a plain Add Difference against A.
    """
    mode = merge_mode(interp_method)
    if mode.needs_c and w_c is None:
        return None

    if interp_method == INTERP_WEIGHTED_SUM:
        return _lerp(w_a, w_b, alpha)

    if interp_method == INTERP_ADD_DIFFERENCE:
        return w_a + alpha * (w_b - w_c)

    if interp_method == INTERP_SUM_TWICE:
        # (1 - beta)((1 - alpha)A + alpha B) + beta C -- the inner blend is
        # exactly a Weighted Sum, which is why this reads as two lerps.
        return _lerp(_lerp(w_a, w_b, alpha), w_c, beta)

    if interp_method == INTERP_SIMILARITY_ADD_DIFFERENCE:
        xp = xp or _array_module()
        # `threshold` is the larger magnitude of the pair, so `a*b/threshold^2`
        # is a cosine-like agreement in [-1, 1] per element; the +1)/2 maps it
        # to [0, 1]. Where both are zero it is 0/0, and `nan_to_num` sends
        # that to beta -- two zeros agree completely.
        threshold = xp.maximum(xp.abs(w_a), xp.abs(w_b))
        # The 0/0 is the point, not an accident, so the warning numpy raises
        # for it is noise. torch has no `errstate` and does not warn.
        errstate = getattr(xp, "errstate", None)
        guard = (
            errstate(invalid="ignore", divide="ignore")
            if errstate is not None
            else contextlib.nullcontext()
        )
        with guard:
            similarity = ((w_a * w_b / threshold ** 2) + 1) / 2
        similarity = xp.nan_to_num(similarity * beta, nan=beta)

        ab_diff = w_a + alpha * (w_b - w_c)
        ab_sum = (1 - alpha / 2) * w_a + (alpha / 2) * w_b
        return (1 - similarity) * ab_diff + similarity * ab_sum

    if interp_method == INTERP_DARE:
        xp = xp or _array_module()
        if beta >= 1.0:
            # Everything dropped. The rescale would divide by zero, and the
            # answer is Model A untouched.
            return w_a * 1
        delta = w_b - w_a
        # Keep with probability (1 - beta) and rescale the survivors, so the
        # expected delta is unchanged however much was thrown away.
        keep = _keep_mask(delta, beta, xp, rand)
        return w_a + alpha * (keep * delta) / (1 - beta)

    raise MergeError(f"No merge math for interpolation method: {interp_method}")


@lru_cache(maxsize=1)
def _array_module():
    """torch when it is importable, numpy otherwise.

    Only the two modes that need real array operations reach for it, and the
    callers inside Forge pass `xp` themselves -- this is the fallback that
    lets the same formulas run on this side.
    """
    try:
        import torch

        return torch
    except ImportError:
        import numpy

        return numpy



def merge_mode_panel(method: str) -> str:
    """The formula, the sentence and the requirement badges for a mode.

    Replaces Gradio's `info=`, which is plain text: `(1-b)((1-a)A + aB) + bC`
    set in running prose is unreadable, and the formula is the thing that makes
    a second ratio slider explain itself. Everything here comes from
    `merge_modes.MERGE_MODES`, so a mode added to the merge loop describes
    itself on screen without a second edit.
    """
    try:
        mode = merge_mode(method)
    except MergeError:
        # A recipe naming a mode this build does not have. An empty panel is
        # the honest answer; inventing a description for it is not.
        return ""

    models = "A"
    if mode.needs_b:
        models += ", B"
    if mode.needs_c:
        models += ", C"
    ratios = "α and β" if mode.needs_beta else ("α" if mode.needs_b else "none")

    def badge(text: str) -> str:
        return (
            "<span style='background: rgba(255,255,255,0.06); border: 1px solid "
            "rgba(255,255,255,0.12); border-radius: 4px; padding: 1px 7px; "
            f"font-size: 11.5px; color: #d1d5db;'>{html.escape(text)}</span>"
        )

    return (
        "<div style='margin: -4px 0 8px; padding: 10px 12px; border-radius: 6px; "
        "background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);'>"
        "<div style='font-family: ui-monospace, SFMono-Regular, Menlo, monospace; "
        f"font-size: 13px; color: #f97316;'>{html.escape(mode.formula)}</div>"
        "<div style='margin-top: 6px; font-size: 12.5px; color: #9ca3af; "
        f"line-height: 1.5;'>{html.escape(mode.description)}</div>"
        "<div style='margin-top: 8px; display: flex; gap: 6px; flex-wrap: wrap;'>"
        f"{badge('Models: ' + models)}{badge('Ratios: ' + ratios)}"
        "</div></div>"
    )
