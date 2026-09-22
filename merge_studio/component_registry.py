"""Policy that fills the gaps Forge leaves.

Forge declares *which* components an architecture consumes. It does not say
what to call them in a form a person reads, which files may legitimately fill a
role, which storage formats are workable, or how confident we are in a family.
That is all this module holds, and nothing more.

Two things this module deliberately is not:

* **Not a slot matrix.** There is no per-architecture table of components. The
  slots come from `clip_target`, so an architecture Forge adds tomorrow shows
  up without an edit here. `SLOT_POLICIES` is keyed by *role*
  ("qwen3_06b", "clip_l", "vae"), which is shared across architectures.
* **Not a class-name map.** No Forge class name appears anywhere below.

Storage note: `fp8_scaled` is the format Forge's own wiki distributes for most
architectures' text encoders, so it is accepted rather than treated as exotic.
What is refused is a container that cannot be embedded in a safetensors
checkpoint at all: GGUF, Nunchaku/SVDQ and NF4.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SupportState(str, Enum):
    """How much the composition path has been exercised for a family.

    There is no NOT_APPLICABLE: every architecture has a required component
    set, and its size is what varies. SD15 and SDXL keep a VAE slot; what they
    lack is a *selectable* text encoder, because theirs ships inside the file.
    """

    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    UNKNOWN = "unknown"


class StorageKind(str, Enum):
    PLAIN = "plain"
    FP8_SCALED = "fp8_scaled"
    FP8_MIXED = "fp8_mixed"
    GGUF = "gguf"
    NUNCHAKU_SVDQ = "nunchaku_svdq"
    NF4 = "nf4"


#: Every storage kind a materialised component may use. A scaled component is
#: copied bit-exact; its `weight_scale` and U8 blocks are never touched.
EMBEDDABLE_STORAGE = (StorageKind.PLAIN, StorageKind.FP8_SCALED, StorageKind.FP8_MIXED)

#: A VAE is small and always distributed dense, so the scaled kinds do not
#: arise in practice; plain keeps the contract honest.
VAE_STORAGE = (StorageKind.PLAIN,)


@dataclass(frozen=True)
class ComponentSlotPolicy:
    forge_target: str
    label: str
    kind: str
    accepted_signatures: tuple[str, ...]
    accepted_storage: tuple[StorageKind, ...]


@dataclass(frozen=True)
class ResolvedSlot:
    """A slot Forge declared, dressed with local policy."""

    forge_target: str
    kind: str
    label: str
    internal_prefixes: tuple[str, ...]
    accepted_signatures: tuple[str, ...]
    accepted_storage: tuple[StorageKind, ...]
    embedded_only: bool


@dataclass(frozen=True)
class ComponentPolicy:
    slots: tuple[ResolvedSlot, ...]
    support: SupportState
    is_generative: bool
    uses_modular_convention: bool


def _encoder(forge_target: str, label: str, *signatures: str) -> ComponentSlotPolicy:
    return ComponentSlotPolicy(
        forge_target=forge_target,
        label=label,
        kind="text_encoder",
        accepted_signatures=signatures or (forge_target,),
        accepted_storage=EMBEDDABLE_STORAGE,
    )


#: Keyed by role, not by architecture. Several architectures share a role.
SLOT_POLICIES: dict[str, ComponentSlotPolicy] = {
    "clip_l": _encoder("clip_l", "CLIP-L"),
    "clip_g": _encoder("clip_g", "CLIP-G"),
    "t5xxl": _encoder("t5xxl", "T5XXL"),
    "umt5xxl": _encoder("umt5xxl", "UMT5XXL"),
    "qwen3_06b": _encoder("qwen3_06b", "Qwen3 0.6B"),
    "qwen3_4b": _encoder("qwen3_4b", "Qwen3 4B"),
    "qwen3_8b": _encoder("qwen3_8b", "Qwen3 8B"),
    # Dimensionally identical to qwen3_4b apart from the vision tower, so it
    # stays a separate role and the signature sets must not overlap.
    "qwen3vl_4b": _encoder("qwen3vl_4b", "Qwen3-VL 4B"),
    "qwen25_7b": _encoder("qwen25_7b", "Qwen2.5-VL 7B"),
    "gemma2_2b": _encoder("gemma2_2b", "Gemma2 2B"),
    "ministral3_3b": _encoder("ministral3_3b", "Ministral3 3B"),
    "vae": ComponentSlotPolicy(
        forge_target="vae",
        label="VAE",
        kind="vae",
        accepted_signatures=("vae",),
        accepted_storage=VAE_STORAGE,
    ),
}

#: Keyed by Forge's own `unet_config["image_model"]`. A family absent from both
#: sets is UNKNOWN, which permits composition while saying so.
SUPPORTED_FAMILIES = frozenset(
    {"flux", "flux2", "anima", "krea2", "lumina2", "wan2.1", "qwen_image"}
)
EXPERIMENTAL_FAMILIES = frozenset({"chroma", "ernie"})


#: Which autoencoder an architecture's latent space needs, keyed by the
#: `latent_format` Forge itself declares. This is capability, not identity:
#: the latent format is what actually decides whether a VAE can decode the
#: model's output, and an unlisted format simply does not filter rather than
#: guessing.
#:
#: The traits are measured. The Qwen-Image, Wan and Anima autoencoder uses 3D
#: convolution; the Flux AE and the SD/SDXL VAE are 2D and differ only in
#: latent channel count -- 16 against 4. Latent channels alone do not separate
#: Flux from Qwen-Image, since both are 16.
VAE_TRAITS_BY_LATENT_FORMAT: dict[str, dict] = {
    "Wan21": {"video_capable": True},
    "Flux": {"video_capable": False, "latent_channels": 16},
    "SDXL": {"video_capable": False, "latent_channels": 4},
    "SD15": {"video_capable": False, "latent_channels": 4},
}


def vae_fits_latent_format(module_info, latent_format_name: str) -> bool:
    """Whether this autoencoder can serve that latent space.

    Offering a Flux AE for an Anima checkpoint builds a file that only fails at
    the post-save reload, several gigabytes later. An unknown latent format
    accepts anything: better to allow a combination that turns out wrong than
    to hide the only VAE a new architecture can use.
    """
    traits = VAE_TRAITS_BY_LATENT_FORMAT.get(latent_format_name)
    if not traits:
        return True
    for key, expected in traits.items():
        actual = module_info.get(key)
        if actual is None:
            continue
        if actual != expected:
            return False
    return True


def get_slot_policy(forge_target: str) -> ComponentSlotPolicy:
    """Policy for a role, inventing a usable generic one when unknown.

    An unrecognised role is not an error: Forge declared it, so it exists. It
    gets a readable label derived from its own name and the default storage
    contract, and later signature reconciliation decides whether a chosen file
    actually fits.
    """
    known = SLOT_POLICIES.get(forge_target)
    if known is not None:
        return known
    return _encoder(forge_target, forge_target.replace("_", " ").upper())


def _support_for(profile) -> SupportState:
    if profile.family_hint in EXPERIMENTAL_FAMILIES:
        return SupportState.EXPERIMENTAL
    if profile.family_hint in SUPPORTED_FAMILIES:
        return SupportState.SUPPORTED
    if not profile.uses_modular_convention:
        # The traditional path predates this feature and already works.
        return SupportState.SUPPORTED
    return SupportState.UNKNOWN


def apply_policy(profile, checkpoint_info=None) -> ComponentPolicy:
    """Dress the slots Forge declared with local policy.

    `checkpoint_info` is accepted for callers that have already inspected the
    file. It cannot add or remove slots -- only Forge decides those -- so
    evidence like an Anima Semantic Connector prefix changes nothing here: that
    connector belongs to the diffusion model, and Anima consumes exactly one
    text encoder in every generation.
    """
    if not profile.is_generative:
        # An i2i upscaler has no checkpoint composition to offer.
        return ComponentPolicy(
            slots=(),
            support=SupportState.UNKNOWN,
            is_generative=False,
            uses_modular_convention=profile.uses_modular_convention,
        )

    # An encoder that only ever ships inside the checkpoint cannot be selected
    # from a folder; the row exists but offers no file picker.
    embedded_only = not profile.uses_modular_convention

    slots = []
    for target in profile.text_targets:
        policy = get_slot_policy(target.forge_target)
        slots.append(
            ResolvedSlot(
                forge_target=target.forge_target,
                kind=target.kind,
                label=policy.label,
                internal_prefixes=target.internal_prefixes,
                accepted_signatures=policy.accepted_signatures,
                accepted_storage=policy.accepted_storage,
                embedded_only=embedded_only,
            )
        )

    if profile.vae_target is not None:
        policy = get_slot_policy("vae")
        slots.append(
            ResolvedSlot(
                forge_target="vae",
                kind="vae",
                label=policy.label,
                internal_prefixes=profile.vae_target.internal_prefixes,
                accepted_signatures=policy.accepted_signatures,
                accepted_storage=policy.accepted_storage,
                embedded_only=False,
            )
        )

    return ComponentPolicy(
        slots=tuple(slots),
        support=_support_for(profile),
        is_generative=True,
        uses_modular_convention=profile.uses_modular_convention,
    )
