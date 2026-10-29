"""Readable names for the LoRA dropdowns.

Forge keys its LoRA catalog by filename stem, and that stem is all the
dropdowns used to show -- `xyz_v3-000012` says nothing about which LoRA it is.
This module builds the text shown next to it, while the dropdown value stays
the stem, so recipes, handlers and `available_loras()` lookups are unchanged:

    Detail Tweaker XL · v2.0 · xl  —  add-detail-xl

The name comes from the first source that has one:

1. The CivitAI sidecars written next to the file: `<stem>.api_info.json`
   (CivitAI Browser Neo) or `<stem>.civitai.info` (Civitai Helper). Both hold
   a CivitAI API response, either for the model or for one version.
2. `ss_output_name` from the safetensors header, the name the trainer gave it.

The architecture label (`xl`, `flux`, `anima`, ...) comes from `lora_arch`,
which reads the same header. Sidecars are only ever read, never written.

Everything is cached per file by size and modification time, sidecars
included, so rebuilding the choices on a refresh costs a `stat` per LoRA.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

from merge_studio import lora_arch

SEPARATOR = " · "
NAME_SEPARATOR = "  —  "

# Detector class names from `architecture_guess._DETECTOR_LABELS`, mapped to
# the labels `lora_arch` gives LoRAs. A class missing here (Iris has no LoRA
# fingerprint yet) means "cannot compare", never "mismatch".
CHECKPOINT_TO_LORA_LABEL = {
    "SD15": "sd",
    "SDXL": "xl",
    "SDXLRefiner": "xl",
    "Mugen": "xl",
    "Flux": "flux",
    "FluxSchnell": "flux",
    "Chroma": "flux",
    "Flux2K4B": "klein",
    "Flux2K9B": "klein",
    "Lumina2": "lumina",
    "ZImage": "zit",
    "Anima": "anima",
    "WAN21_T2V": "wan",
    "WAN21_I2V": "wan",
    "QwenImage": "qwen",
    "QwenImage21": "qwen21",
    "MOSSImage": "moss",
    "MossImage": "moss",
    "Krea2": "krea",
    "ErnieImage": "ernie",
    "PiD": "pid",
    "Flux3": "flux3",
    "Fibo2": "fibo2",
}


@dataclass(frozen=True)
class LoraDescription:
    name: str
    title: str = ""
    version: str = ""
    architectures: frozenset = frozenset()

    @property
    def label(self) -> str:
        parts = [p for p in (self.title, self.version) if p]
        if self.architectures:
            parts.append("/".join(sorted(self.architectures)))
        if not self.title:
            # Nothing better than the stem: show it once, with the tags.
            return SEPARATOR.join([self.name] + parts)
        return SEPARATOR.join(parts) + NAME_SEPARATOR + self.name


_cache: dict[str, tuple[tuple, LoraDescription]] = {}


def _read_json(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _sidecar_paths(path: str) -> tuple[str, str, str]:
    base = os.path.splitext(path)[0]
    civitai_info = base + ".civitai.info"
    if not os.path.isfile(civitai_info) and os.path.isfile(path + ".civitai.info"):
        civitai_info = path + ".civitai.info"
    return base + ".json", base + ".api_info.json", civitai_info


def _same_id(a, b) -> bool:
    """A sidecar for another model must not name this one. Missing IDs pass,
    since older sidecars do not all record them."""
    if a in (None, "") or b in (None, ""):
        return True
    return str(a).strip() == str(b).strip()


def civitai_names(data: dict[str, Any], expected_model_id=None, expected_version_id=None) -> tuple[str, str]:
    """(model name, version name) from a CivitAI API response, or ("", "").

    A version response carries the model under "model"; a model response
    carries its versions under "modelVersions", and the version is picked by
    id, or taken when there is only one.
    """
    if not isinstance(data, dict):
        return "", ""
    versions = data.get("modelVersions")
    if isinstance(versions, list):
        if not _same_id(data.get("id"), expected_model_id):
            return "", ""
        version = {}
        if expected_version_id not in (None, ""):
            version = next(
                (v for v in versions if isinstance(v, dict) and _same_id(v.get("id"), expected_version_id)),
                {},
            )
        elif len(versions) == 1 and isinstance(versions[0], dict):
            version = versions[0]
        return str(data.get("name") or "").strip(), str(version.get("name") or "").strip()

    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    if not _same_id(data.get("modelId", model.get("id")), expected_model_id):
        return "", ""
    if not _same_id(data.get("id"), expected_version_id):
        return "", ""
    return str(model.get("name") or "").strip(), str(data.get("name") or "").strip()


def _mtime(path: str):
    try:
        st = os.stat(path)
        return st.st_size, st.st_mtime_ns
    except OSError:
        return None


def describe_lora(name: str, path: str) -> LoraDescription:
    """Name, version and architectures for one LoRA file. Never raises."""
    sidecars = _sidecar_paths(path)
    key = (_mtime(path),) + tuple(_mtime(p) for p in sidecars)
    cached = _cache.get(path)
    if cached is not None and cached[0] == key:
        return cached[1]

    browser_json, api_info, civitai_info = (
        _read_json(p) if os.path.isfile(p) else {} for p in sidecars
    )

    title = version = ""
    for data in (api_info, civitai_info):
        title, version = civitai_names(
            data, browser_json.get("modelId"), browser_json.get("modelVersionId")
        )
        if title:
            break
    if version and version.casefold() in title.casefold():
        version = ""

    architectures = frozenset()
    if path.lower().endswith(".safetensors"):
        try:
            tensors, metadata = lora_arch.read_safetensors_header(path)
            architectures = lora_arch.classify(tensors, metadata, browser_json.get("sd version"))
            if not title:
                output_name = str(metadata.get("ss_output_name") or "").strip()
                if output_name and output_name.casefold() != name.casefold():
                    title = output_name
        except Exception:
            pass

    description = LoraDescription(name=name, title=title, version=version, architectures=architectures)
    _cache[path] = (key, description)
    return description


def lora_choices(loras: dict[str, str]) -> list[tuple[str, str]]:
    """(label, name) pairs for a Gradio dropdown, sorted by what is shown."""
    pairs = [(describe_lora(name, path).label, name) for name, path in loras.items()]
    return sorted(pairs, key=lambda pair: pair[0].casefold())


def architecture_mismatches(
    checkpoint_detector: str | None, loras: Iterable[tuple[str, str]]
) -> list[tuple[str, frozenset]]:
    """LoRAs made for another architecture than the checkpoint's.

    `checkpoint_detector` is the detector class name `guess_architecture`
    reports. LoRAs whose architecture is unknown are never listed, and nothing
    is when the checkpoint's architecture has no LoRA label.
    """
    wanted = CHECKPOINT_TO_LORA_LABEL.get(checkpoint_detector or "")
    if wanted is None:
        return []
    out = []
    for name, path in loras:
        archs = describe_lora(name, path).architectures
        if archs and wanted not in archs:
            out.append((name, archs))
    return out


def disjoint_architectures(loras: Iterable[tuple[str, str]]) -> list[tuple[str, frozenset]]:
    """For LoRA merging: the known architectures of each source, when no
    single architecture fits all of them. Empty when they agree or are unknown."""
    known = [(name, describe_lora(name, path).architectures) for name, path in loras]
    known = [(name, archs) for name, archs in known if archs]
    if len(known) < 2:
        return []
    common = frozenset.intersection(*(archs for _, archs in known))
    return [] if common else known
