"""What a loaded Forge engine says it can do, normalised.

Forge's `model_config` already declares everything needed to compose a
checkpoint: which text encoders the architecture consumes (`clip_target`), the
namespaces its components live under (`text_encoder_key_prefix`,
`vae_key_prefix`), and what kind of latent space it works in
(`latent_format`). This module reads those values and nothing else.

It deliberately does NOT look at `type(model_config).__name__`. Forge renames
and adds classes; a capability-equivalent config under a new name has to keep
working without an edit here. Forge's own dispatch uses `huggingface_repo`
(`if "Anima" in guess.huggingface_repo`), so that string is fair game as
corroborating evidence -- it fails exactly when Forge fails, never earlier.

Two details of Forge's loader shape this module:

1. `clip_target` is a *method* until `split_state_dict` runs
   `guess.clip_target = guess.clip_target(sd)`, after which it is a dict.
   Both forms are normalised here.
2. `clip_target` is state-dict-conditional for Flux, Chroma, Lumina2 and
   QwenImage, and is evaluated after the additional state dicts are merged.
   Supply no CLIP-L to a Flux load and Forge simply returns one target fewer,
   silently. A shorter list is therefore a fact to report, never an error --
   detecting a *missing* slot is the caller's job, by comparing against policy.

Nothing here imports Forge, Gradio, or touches the filesystem, so the whole
module is testable against small fakes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# Values Forge puts in `unet_config["image_model"]`. This is a hint for
# presentation and for the `generic_discovery` flag -- never a compatibility
# key. A family missing from here still produces a fully usable profile,
# because the slots come from `clip_target`.
KNOWN_FAMILIES = frozenset(
    {
        "flux",
        "flux2",
        "chroma",
        "lumina2",
        "anima",
        "wan2.1",
        "qwen_image",
        "krea2",
        "ernie",
        "pid",
    }
)

# The prefix that marks the modern convention of shipping components as
# separate files. Declared by Forge itself, so it survives renames.
MODULAR_TEXT_PREFIX = "text_encoders."

# PiD is an i2i upscaler: it works in pixel space, not a latent one. It is the
# only class carrying this latent format, which makes it excludable by
# capability rather than by name.
NON_GENERATIVE_LATENT_FORMAT = "RGB"


class ForgeCapabilityError(Exception):
    """A Forge config did not expose a capability the composition needs."""


@dataclass(frozen=True)
class ForgeComponentTarget:
    """One component the architecture consumes.

    `forge_target` is the role ("qwen3_06b", "clip_l", "vae").
    `internal_prefixes` are the namespaces its tensors occupy inside the
    engine, already terminated with a dot.
    """

    forge_target: str
    internal_prefixes: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class ForgeCapabilityProfile:
    family_hint: str
    huggingface_repo: str
    text_targets: tuple[ForgeComponentTarget, ...]
    vae_target: ForgeComponentTarget | None
    text_encoder_key_prefix: tuple[str, ...]
    vae_key_prefix: tuple[str, ...]
    latent_format_name: str
    uses_modular_convention: bool
    is_generative: bool
    generic_discovery: bool
    semantic_fingerprint: str
    diagnostic_fingerprint: str


def _require(config, attribute: str):
    value = getattr(config, attribute, None)
    if value is None:
        raise ForgeCapabilityError(
            f"The loaded Forge model config does not expose {attribute!r}, which is "
            f"required to resolve components. Observed config: {_describe(config)}."
        )
    return value


def _describe(config) -> str:
    cls = type(config)
    return f"{getattr(cls, '__module__', '?')}.{getattr(cls, '__name__', '?')}"


def _resolve_clip_target(config) -> dict:
    """`clip_target` as a dict, whether it is still a method or already resolved.

    Forge mutates the attribute in place during `split_state_dict`, so which
    form is present depends on whether the engine has been loaded yet.
    """
    if not hasattr(config, "clip_target"):
        raise ForgeCapabilityError(
            f"The loaded Forge model config does not expose 'clip_target', which is "
            f"required to resolve text encoders. Observed config: {_describe(config)}."
        )

    target = config.clip_target
    if callable(target):
        try:
            target = target({})
        except TypeError:
            # Some configs declare clip_target(self, state_dict) without a
            # default; a bound method still needs the argument.
            target = target(state_dict={})

    if not isinstance(target, dict):
        raise ForgeCapabilityError(
            f"'clip_target' resolved to {type(target).__name__}, not a mapping of "
            f"component namespaces. Observed config: {_describe(config)}."
        )
    return target


def _split_target(raw: str) -> tuple[str, tuple[str, ...]]:
    """`"qwen3_06b.transformer"` -> `("qwen3_06b", ("qwen3_06b.transformer.",))`.

    The role is the first segment; the whole key is the namespace Forge filters
    on, which it does as `key + "."`.
    """
    role = raw.split(".", 1)[0]
    return role, (raw if raw.endswith(".") else raw + ".",)


def _as_prefix_tuple(value) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _latent_format_name(config) -> str:
    latent = getattr(config, "latent_format", None)
    if latent is None:
        return ""
    # Forge assigns the class itself (`latent_format = latent.Wan21`), but an
    # instance would be just as meaningful.
    return getattr(latent, "__name__", None) or type(latent).__name__


def _fingerprint(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]


def capability_profile_from_engine(engine) -> ForgeCapabilityProfile:
    """Read an already-instantiated Forge engine into a normalised profile.

    Raises `ForgeCapabilityError` naming the exact capability that is missing,
    so a future Forge that drops one fails loudly instead of silently falling
    back to a stale assumption.
    """
    config = getattr(engine, "model_config", None)
    if config is None:
        raise ForgeCapabilityError(
            "The object passed in has no 'model_config'; it does not look like a "
            "loaded Forge engine."
        )

    clip_target = _resolve_clip_target(config)
    text_prefix = _as_prefix_tuple(_require(config, "text_encoder_key_prefix"))
    vae_prefix = _as_prefix_tuple(_require(config, "vae_key_prefix"))

    text_targets = []
    for raw, bucket in clip_target.items():
        role, prefixes = _split_target(str(raw))
        text_targets.append(
            ForgeComponentTarget(
                forge_target=role,
                internal_prefixes=prefixes,
                kind="text_encoder",
            )
        )

    vae_target = ForgeComponentTarget(
        forge_target="vae",
        internal_prefixes=vae_prefix,
        kind="vae",
    )

    unet_config = getattr(config, "unet_config", None) or {}
    family_hint = str(unet_config.get("image_model") or "")
    repo = str(getattr(config, "huggingface_repo", "") or "")
    latent_name = _latent_format_name(config)

    semantic = _fingerprint(
        family_hint,
        repo,
        "|".join(f"{t.forge_target}:{'/'.join(t.internal_prefixes)}" for t in text_targets),
        "/".join(vae_prefix),
        "/".join(text_prefix),
        latent_name,
    )
    diagnostic = _fingerprint(semantic, _describe(config))

    return ForgeCapabilityProfile(
        family_hint=family_hint,
        huggingface_repo=repo,
        text_targets=tuple(text_targets),
        vae_target=vae_target,
        text_encoder_key_prefix=text_prefix,
        vae_key_prefix=vae_prefix,
        latent_format_name=latent_name,
        uses_modular_convention=MODULAR_TEXT_PREFIX in text_prefix,
        is_generative=latent_name != NON_GENERATIVE_LATENT_FORMAT,
        generic_discovery=family_hint not in KNOWN_FAMILIES,
        semantic_fingerprint=semantic,
        diagnostic_fingerprint=diagnostic,
    )


def is_anima_profile(profile: ForgeCapabilityProfile) -> bool:
    """Anima by structural evidence, never by Python class name.

    Forge itself decides this with `"Anima" in guess.huggingface_repo`, so both
    signals are accepted: the declared family and the declared repository.
    """
    if profile.family_hint == "anima":
        return True
    return "anima" in profile.huggingface_repo.lower()
