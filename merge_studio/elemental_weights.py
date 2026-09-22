"""Read the elemental weight syntax that merge recipes are written in.

A checkpoint merged with per-block weights records its ratio not as a number but
as a base plus a list of overrides::

    0,L05-L09:self_attn.q_proj self_attn.k_proj:0.08,L15-L27:mlp.layer1:0.50

The base applies everywhere; each rule replaces it for a range of layers and a
set of module names. Ten rules in one merge is ordinary, several of them
overlapping, which is why a recipe like this is unreadable as text and has to be
resolved and drawn.

One parser, both directions. It is what the Inspector needs to describe recipes
it used to render blank, and what the merge loop resolves against to apply
them -- so a merge this app writes is drawn by the same code that draws one it
downloaded, and the strip under the editor is the strip the finished checkpoint
will show.

Written by hand as well as read, so it takes a rules-only text with the base
coming from the Multiplier slider, and one rule per line as readily as one per
comma.

The syntax is not ours: it is what the tool that produced these checkpoints
writes. Adopting it is interoperability -- a second syntax meaning the same
thing would be a tax on the only person who has to type it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .checkpoint_inspector import _esc

#: ``L<start>-L<end>:<module> <module>:<weight>``
_RULE = re.compile(
    r"^\s*L\s*(\d+)\s*-\s*L\s*(\d+)\s*:\s*(.+?)\s*:\s*([-+]?\d*\.?\d+)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ElementalRule:
    """One override: these modules, in these layers, at this weight."""

    start: int
    end: int
    targets: tuple[str, ...]
    weight: float
    raw: str = ""

    def covers(self, block: int) -> bool:
        """Ranges are inclusive at both ends, as written."""
        return self.start <= block <= self.end

    def matches(self, module_path: str) -> bool:
        """Whether a module name falls under this rule.

        Targets are written as bare module paths (``mlp.layer1``) while the
        thing being weighted may arrive as a full state-dict key
        (``net.blocks.20.mlp.layer1.weight``). Matching on containment keeps
        both callers working without either having to normalise first.
        """
        return any(t and t in module_path for t in self.targets)


@dataclass
class WeightSpec:
    """A parsed ratio: a base, its overrides, and anything unreadable."""

    base: float | None = None
    text_base: str = ""
    rules: tuple[ElementalRule, ...] = ()
    errors: list[str] = field(default_factory=list)

    @property
    def is_uniform(self) -> bool:
        return not self.rules

    @property
    def effective_base(self) -> float:
        return 0.0 if self.base is None else self.base


def parse_weight_spec(text: str) -> WeightSpec:
    """Parse a ratio string. Never raises: bad input comes back in ``errors``.

    These strings arrive from downloaded checkpoint headers, so a malformed
    rule is an ordinary occurrence, not an exception. One unreadable rule
    should not cost the other nine.
    """
    spec = WeightSpec()
    if not text or not str(text).strip():
        return spec

    # Recipes write one line separated by commas; a person typing into a text
    # area writes one rule per line. Both mean the same thing.
    parts = str(text).replace("\r", "\n").replace("\n", ",").split(",")
    head, tail = parts[0].strip(), parts[1:]

    if _RULE.match(head):
        # A recipe always leads with its base, but someone typing rules into
        # the editor has no reason to -- the base is the multiplier slider.
        # A rule is unmistakable, so a leading one is read as a rule and the
        # base is left for `spec_with_base` to fill in.
        tail = parts
    else:
        try:
            spec.base = float(head)
        except ValueError:
            # "Save Components" writes a component list where a ratio would
            # go, so a non-numeric head is valid input, not a mistake.
            spec.text_base = head

    rules = []
    for part in tail:
        if not part.strip():
            continue
        match = _RULE.match(part)
        if not match:
            spec.errors.append(f"Could not read rule: {part.strip()}")
            continue
        start, end, targets, weight = match.groups()
        start, end = int(start), int(end)
        if start > end:
            start, end = end, start
        rules.append(
            ElementalRule(
                start=start,
                end=end,
                targets=tuple(t for t in targets.split() if t),
                weight=float(weight),
                raw=part.strip(),
            )
        )

    spec.rules = tuple(rules)
    return spec


def resolve_weight(spec: WeightSpec, block: int, module_path: str) -> float:
    """The weight in force for one module of one block.

    **The last matching rule wins, and this is verified rather than assumed.**
    Ranges in these recipes genuinely overlap -- `nova3DCGAM_v20` weights
    `self_attn.v_proj` at 0.10 over L06-L11 and at 0.22 over L10-L16, so
    layers 10 and 11 match both.

    Confirmed against the behaviour of the tool that wrote these files:
    `Utils.py::elementals2` loops over every rule and **assigns** on each
    match with no early exit, so the final match is what survives. Matching is
    a substring test against a composite of the key, the block tag and an
    extra tag, which is why containment is the right reading here too. That
    repository has no licence and nothing was copied from it -- reading code
    to learn what a program does is not taking its expression, and the loop
    below was written before this was checked.

    `overlaps()` still reports every such case. The precedence is settled;
    whether the author *meant* the third rule to bury the first is not.
    """
    weight = spec.effective_base
    for rule in spec.rules:
        if rule.covers(block) and rule.matches(module_path):
            weight = rule.weight
    return weight


def spec_with_base(spec: WeightSpec, base: float) -> WeightSpec:
    """The same rules, with a base supplied from outside.

    A merge has two possible sources for the base: the multiplier slider, and
    a leading number in the rule text itself. Pasting a ratio straight out of
    a recipe -- `0,L05-L09:...` -- brings its own base, and that one wins
    because it is what the recipe meant. A rules-only text takes the slider's.

    Returns a new spec rather than mutating: the same parsed rules get resolved
    against different bases in the same session, once per module tree.
    """
    if spec.base is not None or spec.text_base:
        # A text base is "Save Components" writing a component list where a
        # ratio would go. Dropping a number on top of it would turn a mode
        # marker into a blend ratio.
        return spec
    return WeightSpec(
        base=float(base),
        text_base=spec.text_base,
        rules=spec.rules,
        errors=list(spec.errors),
    )


def validate_spec(spec: WeightSpec, blocks: int | None) -> list[str]:
    """Everything wrong with a spec, in the words the person typing it needs.

    Two kinds of problem. A rule that did not parse is already in
    `spec.errors`. A rule that parsed but names layers this checkpoint does
    not have is worse, because it is silent: it simply never fires, and the
    merge comes out uniform where the author expected a weighted band.

    `blocks` comes from the checkpoint, never from a control -- Anima ships as
    28, 40 or 52, and a free-running number would let someone write a rule for
    layer 40 of a 28-block model and see a profile that cannot happen.
    """
    problems = list(spec.errors)
    if not blocks:
        return problems
    for rule in spec.rules:
        if rule.start >= blocks:
            problems.append(
                f"L{rule.start:02d}-L{rule.end:02d} is past the end of this "
                f"model: it has {blocks} blocks, numbered L00-L{blocks - 1:02d}."
            )
        elif rule.end >= blocks:
            problems.append(
                f"L{rule.start:02d}-L{rule.end:02d} runs past L{blocks - 1:02d}; "
                f"this model has {blocks} blocks. The rule still applies up to "
                f"L{blocks - 1:02d}."
            )
    return problems


@dataclass(frozen=True)
class Overlap:
    """Two rules competing for the same modules in the same layers."""

    target: str
    layers: tuple[int, int]
    first: ElementalRule
    second: ElementalRule

    @property
    def weights(self) -> tuple[float, float]:
        return self.first.weight, self.second.weight


def overlaps(spec: WeightSpec) -> list[Overlap]:
    """Every pair of rules that fight over the same module and layers.

    Worth surfacing because an overlap is usually unintentional: the author
    added a later rule for a wider range and forgot the earlier narrow one.
    The merge still resolves deterministically; the author just may not have
    meant what it resolves to.
    """
    found: list[Overlap] = []
    for i, first in enumerate(spec.rules):
        for second in spec.rules[i + 1 :]:
            shared = set(first.targets) & set(second.targets)
            low, high = max(first.start, second.start), min(first.end, second.end)
            if not shared or low > high:
                continue
            for target in sorted(shared):
                found.append(
                    Overlap(target=target, layers=(low, high), first=first, second=second)
                )
    return found


@dataclass(frozen=True)
class ProfileRow:
    """One module target, and its weight in every block."""

    target: str
    weights: list[float]

    @property
    def is_flat(self) -> bool:
        return len(set(self.weights)) <= 1


def weight_profile(spec: WeightSpec, blocks: int) -> list[ProfileRow]:
    """The resolved weight of every target in every block.

    This is what makes the recipe legible. Ten rules with a base of 0.0 is not
    something anyone can hold in their head, but as one row per module and one
    cell per layer it reads at a glance -- which module the merge touched, over
    which depth of the network, and how hard.
    """
    targets = sorted({t for rule in spec.rules for t in rule.targets})
    if not targets:
        return [ProfileRow(target="(all modules)", weights=[spec.effective_base] * blocks)]

    return [
        ProfileRow(
            target=target,
            weights=[resolve_weight(spec, block, target) for block in range(blocks)],
        )
        for target in targets
    ]


# --- Presentation ----------------------------------------------------------


def _cell(weight: float) -> str:
    """One layer of one module, shaded by how much of B it takes.

    Colour rather than a number, because the grid can be twelve rows of
    twenty-eight and the question it answers is "where does this merge act",
    not "what is cell 17". The exact value is in the tooltip.
    """
    if weight <= 0:
        return (
            "<span style='display:inline-block;width:11px;height:14px;"
            "background:rgba(255,255,255,0.04);border-radius:1px;' title='0'></span>"
        )
    # Clamped so a rule above 1.0 still reads as "full" instead of overflowing.
    intensity = min(abs(weight), 1.0)
    alpha = 0.18 + 0.82 * intensity
    colour = "249,115,22" if weight > 0 else "239,68,68"
    return (
        f"<span style='display:inline-block;width:11px;height:14px;"
        f"background:rgba({colour},{alpha:.2f});border-radius:1px;' "
        f"title='{weight:g}'></span>"
    )


def format_weight_profile_html(spec: WeightSpec, blocks: int) -> str:
    """The recipe's ratio, drawn.

    Everything here came out of a downloaded header, so it is escaped on the
    way in -- the rule the recipe and LoRA dashboards already follow.
    """
    # Built before the early returns: rules that all failed to parse leave
    # `rules` empty, and reporting that as "uniform" would tell someone who
    # wrote ten broken rules that everything is fine.
    error_html = ""
    if spec.errors:
        items = "".join(f"<li>{_esc(e)}</li>" for e in spec.errors)
        error_html = (
            "<div style='margin-top:8px;padding:8px 10px;border-radius:4px;"
            "background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.4);"
            "color:#fca5a5;font-size:11.5px;'>"
            f"<ul style='margin:0;padding-left:18px;'>{items}</ul></div>"
        )

    if spec.text_base and not spec.rules:
        return (
            "<div style='font-size:12px;color:#9ca3af;'>Ratio field holds "
            f"<code>{_esc(spec.text_base)}</code> rather than a number.</div>"
        ) + error_html

    if spec.is_uniform:
        base = spec.effective_base
        return (
            "<div style='font-size:12px;color:#d1d5db;'>Uniform ratio of "
            f"<b style='color:#f97316;'>{base:g}</b> across every block.</div>"
        ) + error_html

    rows = "".join(
        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:2px;'>"
        f"<span style='min-width:190px;font-size:11px;color:#e5e7eb;'>{_esc(row.target)}</span>"
        f"<span style='display:flex;gap:1px;'>{''.join(_cell(w) for w in row.weights)}</span>"
        "</div>"
        for row in weight_profile(spec, blocks)
    )

    found = overlaps(spec)
    overlap_html = ""
    if found:
        items = "".join(
            f"<li><code>{_esc(o.target)}</code> in layers {o.layers[0]}&ndash;{o.layers[1]}: "
            f"{o.weights[0]:g} overridden by {o.weights[1]:g}</li>"
            for o in found
        )
        overlap_html = (
            "<div style='margin-top:8px;padding:8px 10px;border-radius:4px;"
            "background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.35);"
            "color:#fbbf24;font-size:11.5px;line-height:1.5;'>"
            f"<b>{len(found)} overlapping rule{'s' if len(found) > 1 else ''}</b> "
            "&mdash; a later rule covers layers an earlier one already set. The "
            "later value is the one that applies."
            f"<ul style='margin:4px 0 0 0;padding-left:18px;'>{items}</ul></div>"
        )

    scale = (
        "<div style='display:flex;align-items:center;gap:6px;margin-top:8px;"
        "font-size:10.5px;color:#6b7280;'>"
        "<span>0</span>"
        + "".join(_cell(i / 8) for i in range(9))
        + "<span>1</span>"
        "<span style='margin-left:10px;'>each cell is one layer, left to right</span>"
        "</div>"
    )

    header = (
        "<div style='font-size:11.5px;color:#9ca3af;margin-bottom:6px;'>"
        f"Base <b style='color:#f97316;'>{spec.effective_base:g}</b>, "
        f"overridden by {len(spec.rules)} rule{'s' if len(spec.rules) != 1 else ''} "
        f"across {blocks} blocks</div>"
    )

    return header + rows + scale + overlap_html + error_html


def format_weight_editor_html(
    spec: WeightSpec, blocks: int | None, problems: list[str]
) -> str:
    """The live profile under the editor: what the rules will do, and what is
    wrong with them.

    The same drawing the Inspector shows for a finished merge, so what you see
    while typing is what the resulting checkpoint will report. The problem list
    is separate from `spec.errors` because it needs the block count, which the
    parser does not have -- a rule for L40 of a 28-block model parses perfectly
    and still cannot fire.
    """
    if not blocks:
        return (
            "<div style='font-size:12px;color:#9ca3af;'>Select a Primary Model "
            "(A) to see how many blocks these rules have to work with.</div>"
        )

    problem_html = ""
    if problems:
        items = "".join(f"<li>{_esc(p)}</li>" for p in problems)
        problem_html = (
            "<div style='margin-bottom:8px;padding:8px 10px;border-radius:4px;"
            "background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.4);"
            f"color:#fca5a5;font-size:11.5px;'><ul style='margin:0;padding-left:18px;'>"
            f"{items}</ul></div>"
        )

    if not spec.rules:
        return problem_html + (
            "<div style='font-size:12px;color:#9ca3af;'>No rules yet &mdash; every "
            "layer merges at the Multiplier. One rule per line or comma, written as "
            "<code style='color:#f97316;'>L05-L09:self_attn.q_proj self_attn.k_proj:0.08</code>"
            "</div>"
        )

    return problem_html + format_weight_profile_html(spec, blocks)


#: An Anima 28-block checkpoint carries 28 DiT blocks plus 6
#: `llm_adapter.blocks`, and every `alpha_weights` array in the reference
#: library is exactly 34 long -- measured across 139 recipes on 2026-09-19.
#: The correlation is perfect but rests on one combination: all 139 came from
#: 28-block merges, and the 40-block checkpoints in the library were made in
#: ComfyUI and carry no recipe at all. So this is the reading, not a proof.
def split_block_array(weights, blocks: int | None) -> tuple[list, list]:
    """A tool's block-weight array, split into the main stack and the tail.

    The tail is whatever the array carries past the diffusion blocks -- for
    Anima that is the LLM adapter's own six. Labelling them apart matters
    because a reader counting cells against `L00-L27` would otherwise find
    six too many and conclude the drawing is wrong.
    """
    values = list(weights or ())
    if not blocks or blocks >= len(values):
        return values, []
    return values[:blocks], values[blocks:]


def is_constant(weights) -> bool:
    """Whether an array says anything a single number does not.

    All 139 `alpha_weights` arrays in the reference library are constant --
    135 all-zero, 4 all-one. The tool writes the base ratio into every slot
    and puts the real per-block variation in the elemental rules instead, so
    drawing these as a profile would be 34 identical cells claiming to be
    information.
    """
    values = list(weights or ())
    return not values or all(v == values[0] for v in values)


def format_block_array_html(weights, blocks: int | None, label: str = "") -> str:
    """A tool's per-block array, drawn -- or nothing when it is constant."""
    values = list(weights or ())
    if is_constant(values):
        return ""

    main, tail = split_block_array(values, blocks)

    def strip(row_label: str, row: list) -> str:
        cells = "".join(_cell(float(v)) for v in row)
        return (
            "<div style='display:flex;align-items:center;gap:8px;margin-top:4px;'>"
            f"<span style='min-width:150px;font-size:11px;color:#9ca3af;'>{_esc(row_label)}</span>"
            f"<span style='display:flex;gap:1px;'>{cells}</span></div>"
        )

    rows = strip(f"L00-L{len(main) - 1:02d}", main)
    if tail:
        rows += strip(f"LLM adapter ({len(tail)})", tail)

    heading = (
        "<div style='font-size:11.5px;color:#9ca3af;margin-bottom:6px;'>"
        f"{_esc(label)}{' &mdash; ' if label else ''}"
        f"{len(values)} values across {len(main)} blocks"
        f"{f' plus {len(tail)} adapter blocks' if tail else ''}</div>"
    )
    return heading + rows
