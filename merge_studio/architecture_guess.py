"""Names a checkpoint's architecture by asking Forge's own detector.

`backend/loader.py` calls `huggingface_guess.guess(state_dict)` to decide which
diffusion engine to build, so that package -- vendored at
`modules_forge/packages/huggingface_guess/` and importable by plain name inside
Forge -- is the authority on what a file is. It knows eighteen architectures by
their official repo names, where the pattern-matching in
`checkpoint_inspector._detect_architecture` knows six families and calls
everything else "Diffusion Model".

Nearly every path through it reads only keys and `.shape`, never a tensor
value, so it can be driven from a safetensors header: a 15 GB checkpoint costs
a header read. `tools/guess_architecture.py` is the command-line form of the
same thing.

Measured over the reference library: 225 of 225 checkpoints named from the
header, plus all seven base models, including the FP8 and INT8 ones.

**Quantization does not break this, and an earlier note saying it did was
wrong.** The failure that note described -- `'ShapeOnly' object has no
attribute 'view'` on an FP8 Krea2 -- came from the torch stub below, not from
the file: building a matched config also builds its latent format, and
`latent.py` reshapes two constants in `Wan21.__init__`. Every Anima file failed
the same way. The stub answers `view` now and both detect. The real pipeline
does convert the quantization before guessing, so a future format that changes
shapes rather than dtypes could still need the real path -- but nothing in the
library does.

Two limits that are real:

* **A few paths read values.** `detection.py` decides Lumina2's `allow_fp16`
  from `torch.std(...)` of a real tensor, so a header-only call has to tolerate
  failure rather than assume it cannot happen.
* **Fine-tunes are not architectures.** Pony, Illustrious and NoobAI are
  structurally identical SDXL -- the detector calls all three SDXL, correctly.
  Naming them is the filename's job, so a fine-tune label the caller already
  worked out survives a detector result of the same family.

Every failure falls back to the label the caller already had, so this can only
add names, never take one away.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any, Callable


class ShapeOnly:
    """Stands in for a tensor. Carries a shape and a dtype, and nothing else.

    Enough for the detector, which branches on which keys exist and how wide
    they are. A path that wants an actual value raises `AttributeError`, which
    is exactly the signal to fall back.

    `view` and `reshape` are here because building a matched config also builds
    its latent format, and `latent.py` reshapes two constant tensors in
    `Wan21.__init__` -- the format Anima uses. Without them every Anima file
    failed with `'ShapeOnly' object has no attribute 'view'`, which looks
    exactly like the error a quantized checkpoint produces and is nothing to do
    with the checkpoint at all. Inside Forge the real torch is already imported
    and none of this runs.
    """

    __slots__ = ("shape", "dtype")

    def __init__(self, shape, dtype: str = "float16"):
        self.shape = tuple(shape)
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.shape)

    def view(self, *shape) -> "ShapeOnly":
        return ShapeOnly(shape[0] if len(shape) == 1 and not isinstance(shape[0], int) else shape, self.dtype)

    reshape = view

    def to(self, *a, **k) -> "ShapeOnly":
        return self


def install_torch_stub() -> bool:
    """Puts a torch-shaped module in `sys.modules` if torch is missing.

    `huggingface_guess.detection` imports torch at module scope but only
    touches it on paths a header-only guess does not reach. Inside Forge torch
    is always already imported and this does nothing; outside it, the stub is
    what lets the detector be exercised at all.

    Opt-in, never automatic: a library that quietly installs a fake torch would
    break the next thing in the process that tries to import the real one.
    """
    if "torch" in sys.modules:
        return False
    torch = types.ModuleType("torch")
    for name in (
        "float32", "float16", "bfloat16", "float64",
        "uint8", "int8", "int32", "int64",
        "float8_e4m3fn", "float8_e5m2",
    ):
        setattr(torch, name, name)
    torch.Tensor = ShapeOnly
    torch.tensor = lambda data, *a, **k: ShapeOnly(
        (len(data),) if isinstance(data, (list, tuple)) else ()
    )
    torch.zeros = lambda *shape, **k: ShapeOnly(shape or ())
    sys.modules["torch"] = torch
    return True


# `backend/loader.py::preprocess_state_dict` puts every key under the diffusion
# namespace when nothing is there already. Skipping it costs three of the seven
# reference checkpoints -- Ernie, Z-Image and Krea2 all store their weights at
# the root and are reported unrecognised without it. Not optional.
_DIFFUSION_NAMESPACES = ("model.diffusion_model.", "net.")


def state_dict_from_header(header: dict[str, Any]) -> dict[str, ShapeOnly]:
    """Header entries as shape-carrying stand-ins, namespaced as Forge does."""
    sd = {
        key: ShapeOnly(entry["shape"], entry.get("dtype", "F16"))
        for key, entry in header.items()
        if key != "__metadata__" and isinstance(entry, dict) and "shape" in entry
    }
    if not any(k.startswith(_DIFFUSION_NAMESPACES) for k in sd):
        sd = {f"model.diffusion_model.{k}": v for k, v in sd.items()}
    return sd


# The detector's class names, in this library's display vocabulary. Keeping the
# family word ("SDXL", "Flux", "Wan", "Anima") inside the label matters:
# `checkpoint_inspector.get_model_family` reads these strings to decide whether
# two checkpoints may be merged at all.
_DETECTOR_LABELS = {
    "SD15": "SD 1.5 / SD 2.1 (UNet)",
    "SDXL": "SDXL (UNet)",
    "SDXLRefiner": "SDXL Refiner (UNet)",
    "Mugen": "Mugen (SDXL)",
    "Flux": "Flux.1 dev (MMDiT)",
    "FluxSchnell": "Flux.1 schnell (MMDiT)",
    "Flux2K4B": "Flux.2 Klein 4B (MMDiT)",
    "Flux2K9B": "Flux.2 Klein 9B (MMDiT)",
    "Chroma": "Chroma (MMDiT)",
    "Lumina2": "Lumina 2 (DiT)",
    "ZImage": "Z-Image (DiT)",
    "Anima": "Anima (DiT)",
    "WAN21_T2V": "Wan2.1 T2V (DiT)",
    "WAN21_I2V": "Wan2.1 I2V (DiT)",
    "QwenImage": "Qwen-Image (MMDiT)",
    "Krea2": "Krea 2 (DiT)",
    "ErnieImage": "ERNIE-Image (DiT)",
    "PiD": "PiD (DiT)",
}

# Labels that name a fine-tune rather than an architecture. The detector cannot
# produce these -- Pony and Illustrious are byte-for-byte SDXL in structure --
# so when the caller's own filename check already found one, it is the more
# informative of the two answers and it wins.
_FINETUNE_LABELS = {
    "Pony (SDXL)": "SDXL",
    "Illustrious (SDXL)": "SDXL",
}


@dataclass(frozen=True)
class ArchitectureGuess:
    """What the file is, and who said so.

    `detector` is the class name the detector returned, or `None` when the
    answer came from the caller's fallback. Anything reading this should treat
    a `None` detector as "pattern matching, not the loader's own opinion".
    """

    label: str
    repo: str | None = None
    detector: str | None = None
    config: dict[str, Any] | None = None
    error: str | None = None

    @property
    def from_detector(self) -> bool:
        return self.detector is not None


def _load_guesser() -> Callable[[dict], Any]:
    import huggingface_guess

    return huggingface_guess.guess


def guess_architecture(
    header: dict[str, Any],
    fallback: str,
    *,
    guesser: Callable[[dict], Any] | None = None,
) -> ArchitectureGuess:
    """The best available name for the architecture in `header`.

    `fallback` is the label the caller already has, and it is what comes back
    whenever the detector is unavailable, raises, or returns a class this
    module has no display name for. `guesser` is injectable so the mapping can
    be tested without Forge on the path.
    """
    try:
        guess = (guesser or _load_guesser())(state_dict_from_header(header))
    except Exception as exc:  # unrecognised, quantized, value-reading, absent
        return ArchitectureGuess(label=fallback, error=f"{type(exc).__name__}: {exc}")

    name = type(guess).__name__
    label = _DETECTOR_LABELS.get(name)
    if label is None:
        # A class the detector gained and this table has not. Reporting the
        # bare class name would put an unknown string in front of
        # `get_model_family`; the fallback is the safer of the two.
        return ArchitectureGuess(
            label=fallback, error=f"no display name for {name!r}"
        )

    if _FINETUNE_LABELS.get(fallback) == name:
        label = fallback

    config = getattr(guess, "unet_config", None) or {}
    return ArchitectureGuess(
        label=label,
        repo=getattr(guess, "huggingface_repo", None),
        detector=name,
        config={k: v for k, v in config.items() if isinstance(v, (int, float, str))},
    )


# A1111 and Forge write these as zero-length marker tensors at the root of the
# file. They say nothing about the architecture and everything about how the
# model has to be sampled, which is why they are worth surfacing: a v-prediction
# checkpoint run on an epsilon schedule produces noise, and nothing on screen
# says so today. NoobAI carries both.
_PREDICTION_MARKERS = {
    "v_pred": "v_prediction",
    "ztsnr": "ztsnr",
}


def detect_prediction_markers(header: dict[str, Any]) -> dict[str, bool]:
    """Sampling markers carried in the header, as `{marker: True}`.

    Empty when the file carries none, which is the overwhelmingly common case.
    """
    found: dict[str, bool] = {}
    for key in header:
        if key == "__metadata__":
            continue
        leaf = key.rsplit(".", 1)[-1]
        if leaf in _PREDICTION_MARKERS:
            found[_PREDICTION_MARKERS[leaf]] = True
    return found


# Anima ships in three depths and the depth is the generation. Verified
# block-by-block against the reference library (cosine 1.0000 on carried-over
# blocks) and recorded in `docs/RESEARCH.md`: the detector returns only "Anima",
# so this naming stays ours.
#
# Trap for anyone tempted to shorten this into "28 blocks means Anima": Krea2
# also has `blocks=28` at the root, and `anima_remap.block_count_from_keys`
# treats the prefix as optional, so it returns 28 for a Krea2 file. The count
# only names a generation once the architecture is already known to be Anima.
ANIMA_GENERATIONS = {
    28: ("", "circlestone-labs/Anima"),
    40: ("2.9B", "Gazingstars123/Anima-2.9B"),
    52: ("3.8B", "lylogummy/Anima-3.8B"),
}


def anima_generation(block_count: int | None) -> str:
    """The generation name for an Anima depth, or `""` for the base and for
    any depth that is not one of the three known ones."""
    if block_count is None:
        return ""
    return ANIMA_GENERATIONS.get(block_count, ("", None))[0]
