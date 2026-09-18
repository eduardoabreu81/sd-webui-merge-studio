"""Policy that complements what Forge declares.

The registry must never decide *which* slots exist -- that comes from Forge's
``clip_target``. It only adds what Forge does not express: a readable label,
which signatures may fill a role, which storage formats are acceptable, and the
support level.

The ``CLIP_TARGETS`` table below is what Forge's ``model_list.py`` actually
returns, copied verbatim so these tests fail if the assumption drifts.
"""

import unittest

from component_registry import (
    SupportState,
    StorageKind,
    apply_policy,
    get_slot_policy,
    SLOT_POLICIES,
)
from forge_capabilities import capability_profile_from_engine

CLIP_TARGETS = {
    "flux": {"clip_l": "text_encoder", "t5xxl": "text_encoder_2"},
    "flux2_k4b": {"qwen3_4b.transformer": "text_encoder"},
    "flux2_k9b": {"qwen3_8b.transformer": "text_encoder"},
    "chroma": {"t5xxl": "text_encoder"},
    "lumina2": {"gemma2_2b.transformer": "text_encoder"},
    "zimage": {"qwen3_4b.transformer": "text_encoder"},
    "anima": {"qwen3_06b.transformer": "text_encoder"},
    "wan21": {"umt5xxl": "text_encoder"},
    "qwen_image": {"qwen25_7b.transformer": "text_encoder"},
    "krea2": {"qwen3vl_4b.transformer": "text_encoder"},
    "ernie": {"ministral3_3b.transformer": "text_encoder"},
    "sdxl": {"clip_l": "text_encoder", "clip_g": "text_encoder_2"},
    "sd15": {"clip_l": "text_encoder"},
    "base": {},
}

TRADITIONAL = {"sdxl", "sd15", "base"}

# What Forge actually puts in unet_config["image_model"]. Note that ZImage and
# Lumina2 share "lumina2", and both Klein sizes share "flux2" -- the family hint
# is not a unique architecture id, which is exactly why the slots come from
# clip_target instead.
#
# The class hierarchy is no better a guide: Flux2K4B and Flux2K9B subclass Flux
# yet consume one encoder instead of two, and Chroma subclasses FluxSchnell and
# drops to a single T5XXL. Only clip_target is authoritative.
IMAGE_MODELS = {
    "flux": "flux",
    "flux2_k4b": "flux2",
    "flux2_k9b": "flux2",
    "chroma": "chroma",
    "lumina2": "lumina2",
    "zimage": "lumina2",
    "anima": "anima",
    "wan21": "wan2.1",
    "qwen_image": "qwen_image",
    "krea2": "krea2",
    "ernie": "ernie",
    "sdxl": None,
    "sd15": None,
    "base": None,
}

LATENT_FORMATS = {
    "flux": "Flux",
    "flux2_k4b": "Flux2",
    "flux2_k9b": "Flux2",
    "chroma": "Flux",
    "lumina2": "Flux",
    "zimage": "Flux",
    "anima": "Wan21",
    "wan21": "Wan21",
    "qwen_image": "Wan21",
    "krea2": "Wan21",
    "ernie": "Flux2",
    "sdxl": "SDXL",
    "sd15": "SD15",
    "base": "SD15",
}


def make_engine(key, *, class_name=None, image_model=None, repo=None):
    traditional = key in TRADITIONAL
    hint = image_model if image_model is not None else IMAGE_MODELS[key]
    body = {
        "unet_config": {"image_model": hint} if hint else {},
        "huggingface_repo": repo or f"vendor/{key}",
        "clip_target": dict(CLIP_TARGETS[key]),
        "text_encoder_key_prefix": (
            ["cond_stage_model."] if traditional else ["text_encoders."]
        ),
        "vae_key_prefix": ["first_stage_model."] if traditional else ["vae."],
        "latent_format": type(LATENT_FORMATS[key], (), {}),
    }
    config = type(class_name or f"{key}Config", (), body)()
    return type("FakeEngine", (), {"model_config": config})()


def policy_for(key, **kw):
    return apply_policy(capability_profile_from_engine(make_engine(key, **kw)))


def roles(policy):
    return tuple(slot.forge_target for slot in policy.slots)


class SlotsComeFromForgeTests(unittest.TestCase):
    def test_every_declared_target_becomes_a_slot_plus_the_vae(self):
        for key, targets in CLIP_TARGETS.items():
            with self.subTest(architecture=key):
                expected = tuple(t.split(".")[0] for t in targets) + ("vae",)
                self.assertEqual(expected, roles(policy_for(key)))

    def test_flux_1_is_the_only_family_with_two_text_encoders_plus_vae(self):
        """Dev, Schnell and Kontext -- Kontext runs through the Flux class and
        is told apart by filename, so it inherits the same two encoders."""
        self.assertEqual(("clip_l", "t5xxl", "vae"), roles(policy_for("flux")))
        self.assertEqual(3, len(policy_for("flux").slots))

    def test_flux_2_klein_has_one_encoder_despite_the_shared_name(self):
        """Flux2K4B and Flux2K9B subclass Forge's Flux but override
        clip_target, and Chroma does the same from FluxSchnell. Class
        hierarchy says nothing about how many encoders a family consumes."""
        self.assertEqual(("qwen3_4b", "vae"), roles(policy_for("flux2_k4b")))
        self.assertEqual(("qwen3_8b", "vae"), roles(policy_for("flux2_k9b")))
        self.assertEqual(("t5xxl", "vae"), roles(policy_for("chroma")))

    def test_the_common_case_is_one_encoder_and_one_vae(self):
        for key in ("anima", "krea2", "zimage", "wan21", "qwen_image", "chroma"):
            with self.subTest(architecture=key):
                self.assertEqual(2, len(policy_for(key).slots))

    def test_qwen_image_uses_the_name_forge_uses(self):
        self.assertIn("qwen25_7b", roles(policy_for("qwen_image")))
        self.assertNotIn("qwen25vl_7b", roles(policy_for("qwen_image")))

    def test_the_registry_is_keyed_by_role_not_by_architecture(self):
        for key in CLIP_TARGETS:
            self.assertNotIn(key, SLOT_POLICIES)
        self.assertIn("qwen3_06b", SLOT_POLICIES)
        self.assertIn("vae", SLOT_POLICIES)

    def test_no_forge_class_name_appears_in_the_registry(self):
        blob = repr(SLOT_POLICIES)
        for class_name in ("Flux2K4B", "Krea2", "ZImage", "QwenImage", "ErnieImage", "PiD"):
            self.assertNotIn(class_name, blob)

    def test_policy_never_invents_a_target_forge_did_not_declare(self):
        declared = set(roles(policy_for("anima")))
        self.assertEqual({"qwen3_06b", "vae"}, declared)


class TraditionalArchitecturesKeepAVaeSlotTests(unittest.TestCase):
    """SD15 and SDXL are not excluded from AIO. Their encoders are embedded."""

    def test_sdxl_has_a_vae_slot(self):
        self.assertIn("vae", roles(policy_for("sdxl")))

    def test_sdxl_text_encoders_are_marked_embedded_only(self):
        policy = policy_for("sdxl")
        encoders = [s for s in policy.slots if s.kind == "text_encoder"]
        self.assertEqual(2, len(encoders))
        self.assertTrue(all(s.embedded_only for s in encoders))

    def test_a_modern_encoder_is_selectable(self):
        (encoder,) = [s for s in policy_for("anima").slots if s.kind == "text_encoder"]
        self.assertFalse(encoder.embedded_only)

    def test_an_empty_clip_target_still_yields_a_vae_slot(self):
        self.assertEqual(("vae",), roles(policy_for("base")))


class PiDIsExcludedByCapabilityTests(unittest.TestCase):
    def test_an_rgb_latent_format_is_not_a_checkpoint_architecture(self):
        engine = make_engine("lumina2", class_name="SomethingElse")
        engine.model_config.latent_format = type("RGB", (), {})
        policy = apply_policy(capability_profile_from_engine(engine))
        self.assertFalse(policy.is_generative)
        self.assertEqual((), policy.slots)

    def test_no_other_architecture_uses_rgb(self):
        for key in CLIP_TARGETS:
            with self.subTest(architecture=key):
                self.assertTrue(policy_for(key).is_generative)


class AnimaHasOneSinglePolicyTests(unittest.TestCase):
    def test_anima_declares_exactly_qwen3_06b_and_a_vae(self):
        self.assertEqual(("qwen3_06b", "vae"), roles(policy_for("anima")))

    def test_there_is_no_qwen35_slot_anywhere_in_the_registry(self):
        self.assertNotIn("qwen35_4b", SLOT_POLICIES)
        self.assertNotIn("qwen35_4b", repr(SLOT_POLICIES))

    def test_no_slot_records_provider_ownership(self):
        for policy in SLOT_POLICIES.values():
            self.assertFalse(hasattr(policy, "provider"))
            self.assertFalse(hasattr(policy, "required_provider"))

    def test_connector_evidence_does_not_add_a_slot(self):
        info = {"embedded_prefixes": ("net.anima_v2_connector.",)}
        profile = capability_profile_from_engine(make_engine("anima"))
        self.assertEqual(
            ("qwen3_06b", "vae"),
            roles(apply_policy(profile, checkpoint_info=info)),
        )

    def test_the_legacy_adapter_is_not_a_slot(self):
        self.assertNotIn("expanded_adapter", SLOT_POLICIES)

    def test_renaming_the_class_does_not_change_the_slots(self):
        a = roles(policy_for("anima"))
        b = roles(policy_for("anima", class_name="AnimaRenamedUpstream"))
        self.assertEqual(a, b)


class DistinctQwenRolesTests(unittest.TestCase):
    """Krea2's qwen3vl_4b and Z-Image's qwen3_4b are dimensionally identical
    apart from the vision tower, so they must stay separate roles."""

    def test_krea2_and_zimage_do_not_share_a_slot(self):
        self.assertIn("qwen3vl_4b", roles(policy_for("krea2")))
        self.assertIn("qwen3_4b", roles(policy_for("zimage")))
        self.assertNotIn("qwen3_4b", roles(policy_for("krea2")))
        self.assertNotIn("qwen3vl_4b", roles(policy_for("zimage")))

    def test_their_accepted_signatures_do_not_overlap(self):
        vl = set(get_slot_policy("qwen3vl_4b").accepted_signatures)
        plain = set(get_slot_policy("qwen3_4b").accepted_signatures)
        self.assertEqual(set(), vl & plain)

    def test_qwen3_sizes_are_separate_roles(self):
        for role in ("qwen3_06b", "qwen3_4b", "qwen3_8b"):
            self.assertIn(role, SLOT_POLICIES)


class StorageTests(unittest.TestCase):
    def test_scaled_encoders_are_accepted(self):
        """fp8_scaled is Forge's standard distribution format, not an oddity."""
        for role in (
            "clip_l",
            "t5xxl",
            "umt5xxl",
            "qwen3_06b",
            "qwen3_4b",
            "qwen3_8b",
            "qwen3vl_4b",
            "qwen25_7b",
            "gemma2_2b",
            "ministral3_3b",
        ):
            with self.subTest(slot=role):
                accepted = get_slot_policy(role).accepted_storage
                self.assertIn(StorageKind.PLAIN, accepted)
                self.assertIn(StorageKind.FP8_SCALED, accepted)
                self.assertIn(StorageKind.FP8_MIXED, accepted)

    def test_container_formats_are_refused_everywhere(self):
        for role, policy in SLOT_POLICIES.items():
            with self.subTest(slot=role):
                for banned in (
                    StorageKind.GGUF,
                    StorageKind.NUNCHAKU_SVDQ,
                    StorageKind.NF4,
                ):
                    self.assertNotIn(banned, policy.accepted_storage)

    def test_every_slot_accepts_plain_storage(self):
        for role, policy in SLOT_POLICIES.items():
            with self.subTest(slot=role):
                self.assertIn(StorageKind.PLAIN, policy.accepted_storage)


class SupportStateTests(unittest.TestCase):
    def test_chroma_and_ernie_are_experimental(self):
        for key in ("chroma", "ernie"):
            with self.subTest(architecture=key):
                self.assertEqual(SupportState.EXPERIMENTAL, policy_for(key).support)

    def test_established_families_are_supported(self):
        for key in ("flux", "anima", "krea2", "zimage", "wan21", "qwen_image"):
            with self.subTest(architecture=key):
                self.assertEqual(SupportState.SUPPORTED, policy_for(key).support)

    def test_an_unknown_family_is_unknown_not_rejected(self):
        engine = make_engine("lumina2", class_name="NovaImageV7", image_model="nova_image")
        engine.model_config.clip_target = {"nova_text.transformer": "text_encoder"}
        policy = apply_policy(capability_profile_from_engine(engine))
        self.assertEqual(SupportState.UNKNOWN, policy.support)
        self.assertEqual(("nova_text", "vae"), roles(policy))

    def test_there_is_no_not_applicable_state(self):
        self.assertFalse(hasattr(SupportState, "NOT_APPLICABLE"))


class GenericSlotsRemainUsableTests(unittest.TestCase):
    def test_a_slot_with_no_local_policy_gets_a_generic_label(self):
        engine = make_engine("lumina2", class_name="NovaImageV7")
        engine.model_config.clip_target = {"nova_text.transformer": "text_encoder"}
        (slot,) = [
            s
            for s in apply_policy(capability_profile_from_engine(engine)).slots
            if s.kind == "text_encoder"
        ]
        self.assertEqual("nova_text", slot.forge_target)
        self.assertTrue(slot.label)
        self.assertIn(StorageKind.PLAIN, slot.accepted_storage)

    def test_known_slots_get_readable_labels(self):
        self.assertEqual("CLIP-L", get_slot_policy("clip_l").label)
        self.assertEqual("T5XXL", get_slot_policy("t5xxl").label)
        self.assertEqual("Qwen3 0.6B", get_slot_policy("qwen3_06b").label)


class ModuleHygieneTests(unittest.TestCase):
    def test_importing_the_module_does_not_pull_in_forge_or_gradio(self):
        import sys

        for banned in ("backend", "backend.loader", "gradio", "modules", "modules_forge"):
            self.assertNotIn(banned, sys.modules)

    def test_slot_policies_are_immutable(self):
        policy = get_slot_policy("clip_l")
        with self.assertRaises(Exception):
            policy.label = "tampered"


if __name__ == "__main__":
    unittest.main()
