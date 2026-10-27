"""Which layers a quantization from full precision converts, per architecture.

Only consulted when the source carries no quantization of its own: a quantized
source is mirrored layer by layer, which already reproduces whatever its
builder chose. From BF16 there is nothing to mirror, and the generic heuristic
(`norm` / `embed` / `modulation` skipped by name) misses the layers the
official builds keep at full precision: Krea 2's `first`, `last`, `tmlp` and
`tproj`, FLUX 3's `txt_in` and `time_in`, Qwen-Image 2.1's `img_in` and
`proj_out`, among others.

Every rule below was read from the headers of published builds on 2026-10-18
(Comfy-Org, Black Forest Labs, and the Iris 3B and MOSS-Image releases), and
the module paths were checked against the classes that build each model inside
Forge -- Forge Neo's own for Krea 2 and Qwen-Image 2.1, the extensions' for
FLUX 3, MOSS-Image and Iris 3B. They coincide with the file's keys.

Where the official builds of one model differ by format, the profile says so.
A format with no measured build gets the profile's most conservative set.

No torch here: the profile is chosen from module names alone, so it can be
tested without Forge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Output-format keys (quant_utils.OUTPUT_FORMAT_CHOICES) by the family a
# profile's measurements were taken in.
FORMAT_GROUPS = {
    "float8_e4m3fn": "fp8",
    "float8_e5m2": "fp8",
    "mxfp8": "fp8",
    "int8_tensorwise": "int8",
    "int8_tensorwise_convrot": "int8",
    "w6a8_int8": "int8",
    "nvfp4": "fp4",
    "convrot_w4a4": "int4",
    "asym_w4a8_int8": "int4",
}


@dataclass(frozen=True)
class QuantProfile:
    """One architecture's selection.

    `marks` are module paths that must all exist for the profile to apply:
    identity by structure, so a renamed class still matches and a text encoder
    or VAE converted through the same function never does. `quantize` lists
    the path prefixes converted by default; `by_group` replaces that list for
    a format family whose official build selects differently. `keep` are
    patterns that stay at full precision even inside a selected prefix.
    """

    name: str
    marks: tuple[str, ...]
    quantize: tuple[str, ...]
    by_group: dict[str, tuple[str, ...]] = field(default_factory=dict)
    keep: tuple[str, ...] = ()

    def prefixes_for(self, output_format: str) -> tuple[str, ...]:
        return self.by_group.get(FORMAT_GROUPS.get(output_format, ""), self.quantize)

    def selects(self, module_path: str, output_format: str) -> bool:
        if any(re.search(pattern, module_path) for pattern in self.keep):
            return False
        return module_path.startswith(self.prefixes_for(output_format))


PROFILES: tuple[QuantProfile, ...] = (
    QuantProfile(
        # Comfy-Org: INT8 ConvRot converts the 28 main blocks only; FP8 scaled,
        # MXFP8 and NVFP4 also convert the text-fusion blocks. All four keep
        # `first`, `last`, `tmlp`, `tproj`, `txtmlp` and `txtfusion.projector`.
        name="krea2",
        marks=("first", "tproj", "txtmlp", "txtfusion", "blocks"),
        quantize=("blocks.",),
        by_group={
            "fp8": ("blocks.", "txtfusion.layerwise_blocks.", "txtfusion.refiner_blocks."),
            "fp4": ("blocks.", "txtfusion.layerwise_blocks.", "txtfusion.refiner_blocks."),
        },
    ),
    QuantProfile(
        # Comfy-Org INT8 ConvRot and NVFP4 (BennyDaBall's agrees): the
        # transformer blocks only. `img_in`, `txt_in`, `modulation`,
        # `norm_out`, `proj_out` and the timestep embedder stay.
        name="qwen_image21",
        marks=("transformer_blocks", "txt_in.text_norm", "time_text_embed", "proj_out"),
        quantize=("transformer_blocks.",),
    ),
    QuantProfile(
        # Black Forest Labs: INT8 ConvRot converts every block, the text-mode
        # ones included; NVFP4 leaves `txt_mode_blocks` alone. Both keep
        # `txt_in`, `time_in`, `ref_time_in`, `emb_in`, `final_layer` and the
        # stream modulations.
        name="flux3",
        marks=("content_mode_blocks", "txt_mode_blocks", "single_blocks", "txt_in"),
        quantize=("content_mode_blocks.", "single_blocks."),
        by_group={
            "int8": ("content_mode_blocks.", "txt_mode_blocks.", "single_blocks."),
        },
    ),
    QuantProfile(
        # Comfy-Org, Abiray and toxicdog INT8 ConvRot agree: the attention and
        # MLP projections of the 42 layers. The text embedder, the shared and
        # head modulations and `final_linear_video` stay.
        name="moss_image",
        marks=("layers", "text_embedder", "shared_block_modulation", "final_linear_video"),
        quantize=("layers.",),
    ),
    QuantProfile(
        # The INT8 ConvRot build converts the 24 blocks and the pixel blocks
        # except their MLP; the W4A8 build converts the blocks and the text
        # adapter, and leaves the pixel stage whole. Both keep the embedders,
        # `modulation_cores`, `y_embedder.layer_pool` and `final_layer`.
        # Unmeasured formats get the blocks alone, which both agree on.
        name="iris",
        marks=("y_embedder", "pixel_blocks", "modulation_cores", "blocks"),
        quantize=("blocks.",),
        by_group={
            "int8": ("blocks.", "pixel_blocks."),
            "int4": ("blocks.", "y_embedder.layer_blocks.", "y_embedder.refiner."),
        },
        keep=(r"^pixel_blocks\.\d+\.mlp\.", r"^y_embedder\.layer_pool$"),
    ),
    QuantProfile(
        # The dev and edit-turbo INT8 ConvRot builds agree: every projection
        # of the 44 layers, the noise refiner and the perceiver. Each block's
        # `adaLN_modulation`, `final_layer`, `t_embedder` and `x_embedder`
        # stay; the name heuristic would have converted `final_layer.linear`.
        # Measured 2026-10-27.
        name="fibo2",
        marks=("perceiver", "noise_refiner", "final_layer", "layers"),
        quantize=("layers.", "noise_refiner.", "perceiver.layers."),
        keep=(r"\.adaLN_modulation\.",),
    ),
)


def profile_for(module_paths) -> QuantProfile | None:
    """The profile whose marks all appear among `module_paths`, or None."""
    present = set(module_paths)
    for profile in PROFILES:
        if all(mark in present for mark in profile.marks):
            return profile
    return None
