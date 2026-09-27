"""The merge history a checkpoint carries in `sd_merge_models`.

WebUI-style mergers write every ancestor they know of into `sd_merge_models`,
keyed by sha256, each with its own `sd_merge_recipe`. Following the primary /
secondary / tertiary hashes from the file's own recipe walks the whole family
tree. Community checkpoints are often merges of merges -- novaRealityAM_v10
records 46 ancestors -- so the two direct parents say little about what is in
the file.

Everything here comes out of a downloaded file's header: names are escaped at
interpolation, and the walk is bounded against cycles and runaway depth.
"""

from __future__ import annotations

import html as html_lib
from dataclasses import dataclass
from typing import Any

_ROLES = (("A", "primary_model_hash"), ("B", "secondary_model_hash"), ("C", "tertiary_model_hash"))
# Real trees run about 15 deep; this only stops a malformed one.
_MAX_DEPTH = 64
_RAW_PREVIEW_CHARS = 90


@dataclass(frozen=True)
class LineageNode:
    role: str
    name: str
    method: str
    ratio: str
    children: tuple["LineageNode", ...] = ()
    # An ancestor already drawn above, whose own history is not repeated.
    repeat: bool = False


def _esc(value: Any) -> str:
    return html_lib.escape(str(value), quote=True)


def _entries(models: Any) -> dict[str, dict]:
    """The hash-keyed entries only. Some writers leave a stray record's own
    fields at the top level of the same dict (see `stray_records`)."""
    if not isinstance(models, dict):
        return {}
    return {k: v for k, v in models.items() if isinstance(v, dict)}


def _ratio_text(recipe: dict) -> str:
    raw = recipe.get("alpha_raw")
    if raw not in (None, ""):
        text = str(raw)
        return "α " + (text if len(text) <= _RAW_PREVIEW_CHARS else text[:_RAW_PREVIEW_CHARS] + "…")
    for field in ("multiplier", "alpha"):
        if isinstance(recipe.get(field), (int, float)) and not isinstance(recipe.get(field), bool):
            return f"M = {recipe[field]:g}"
    if recipe.get("block_weights") is True:
        return "per-block weights, values not recorded"
    if recipe.get("alpha_info") not in (None, ""):
        return f"α = {recipe['alpha_info']}"
    return ""


def _method(recipe: dict) -> str:
    return str(recipe.get("merge_method") or recipe.get("interp_method") or recipe.get("type") or "")


def build_lineage(recipe: dict, models: Any) -> LineageNode:
    """The file's recipe as a tree of its ancestors.

    Each ancestor with a recipe is expanded the first time it is met; later
    references are marked `repeat` and left unexpanded, which also ends any
    cycle a malformed header might contain."""
    entries = _entries(models)
    expanded: set[str] = set()

    def parents(rec: dict, depth: int) -> tuple[LineageNode, ...]:
        out = []
        for role, field in _ROLES:
            h = rec.get(field)
            if isinstance(h, str) and h:
                out.append(node(role, h, depth))
        return tuple(out)

    def node(role: str, h: str, depth: int) -> LineageNode:
        entry = entries.get(h)
        if entry is None:
            return LineageNode(role, f"unknown ({h[:12]})", "", "")
        name = str(entry.get("name") or h[:12])
        rec = entry.get("sd_merge_recipe")
        if not (isinstance(rec, dict) and rec):
            return LineageNode(role, name, "", "")
        if h in expanded or depth >= _MAX_DEPTH:
            return LineageNode(role, name, _method(rec), "", repeat=True)
        expanded.add(h)
        return LineageNode(role, name, _method(rec), _ratio_text(rec), parents(rec, depth + 1))

    root_name = str(recipe.get("output_name") or "this file")
    return LineageNode("", root_name, _method(recipe), _ratio_text(recipe), parents(recipe, 1))


def stray_records(models: Any) -> list[dict[str, Any]]:
    """A LoRA-bake recipe written into `sd_merge_models` as loose top-level
    keys rather than under a hash.

    chattiori's `pluslora` does this: `type`, `checkpoint_hash`, `lora_hash`
    and `alpha_info` land beside the hash entries. The record still says which
    LoRAs were baked and how strongly, which nothing else in the file does."""
    if not isinstance(models, dict):
        return []
    kind = models.get("type")
    if not (isinstance(kind, str) and "lora" in kind.lower()):
        return []

    def split(field: str) -> list[str]:
        value = models.get(field)
        return [p.strip() for p in value.split(",") if p.strip()] if isinstance(value, str) else []

    return [
        {
            "type": kind,
            "output_name": str(models.get("output_name") or ""),
            "checkpoint_hash": str(models.get("checkpoint_hash") or ""),
            "loras": split("lora_hash"),
            "strengths": split("alpha_info"),
        }
    ]


def _node_rows(node: LineageNode, depth: int) -> str:
    role = f"<code style='color:#f97316;'>{_esc(node.role)}</code> " if node.role else ""
    detail = " &middot; ".join(_esc(p) for p in (node.method, node.ratio) if p)
    if node.repeat:
        detail = (detail + " &middot; " if detail else "") + "history shown above"
    row = (
        f"<div style='padding:2px 0 2px {depth * 18}px;font-size:12.5px;color:#d1d5db;'>"
        f"{role}<b>{_esc(node.name)}</b>"
        + (f" <span style='color:#9ca3af;'>&mdash; {detail}</span>" if detail else "")
        + "</div>"
    )
    return row + "".join(_node_rows(child, depth + 1) for child in node.children)


def _stray_html(record: dict[str, Any]) -> str:
    strengths = ", ".join(record["strengths"]) or "not recorded"
    return (
        "<div style='margin-top:8px;font-size:12px;color:#9ca3af;'>"
        f"<b style='color:#d1d5db;'>Loose LoRA-bake record</b> (<code>{_esc(record['type'])}</code>): "
        f"<b style='color:#d1d5db;'>{_esc(record['output_name'] or 'unnamed')}</b> was baked from checkpoint "
        f"<code>{_esc(record['checkpoint_hash'][:12])}</code> with {len(record['loras'])} LoRA(s) at strengths "
        f"<code style='color:#10b981;'>{_esc(strengths)}</code>. The writer left it outside the hash entries, "
        "so which ancestor of that name it belongs to is not recorded."
        "</div>"
    )


def render_lineage_html(recipe: dict, models: Any) -> str:
    """A collapsed section with the whole tree, or "" when there is no
    history beyond what the parent cards already show."""
    entries = _entries(models)
    strays = stray_records(models)
    if not entries and not strays:
        return ""
    tree = build_lineage(recipe, models)
    rows = "".join(_node_rows(child, 0) for child in tree.children)
    return (
        "<details style='margin-top:14px;'>"
        "<summary style='cursor:pointer;font-size:12px;font-weight:600;text-transform:uppercase;color:#9ca3af;'>"
        f"Merge history &mdash; {len(entries)} ancestor(s) recorded</summary>"
        f"<div style='margin-top:8px;padding:8px 10px;border-radius:6px;background:rgba(0,0,0,0.2);"
        f"border:1px solid rgba(255,255,255,0.06);'>{rows}</div>"
        + "".join(_stray_html(r) for r in strays)
        + "<div style='margin-top:6px;color:#6b7280;font-size:11px;'>"
        "As declared by each merger in the file header, not read back from the tensors.</div>"
        "</details>"
    )
