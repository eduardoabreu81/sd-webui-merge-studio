"""Capability discovery from a loaded Forge engine.

The fakes here mirror the *shape* of Forge's model configs, never their names.
Every class name used is deliberately not one Forge ships, because a renamed or
future config with the same functional contract must keep working.
"""

import unittest

from merge_studio.forge_capabilities import (
    ForgeCapabilityError,
    capability_profile_from_engine,
    is_anima_profile,
)

MISSING = object()


def make_config(
    class_name,
    *,
    image_model="flux",
    huggingface_repo="vendor/Model",
    clip_target=MISSING,
    text_prefix=("text_encoders.",),
    vae_prefix=("vae.",),
    latent_format="Flux",
    with_save_processors=True,
):
    """A model config shaped like Forge's, under an arbitrary class name.

    ``clip_target`` accepts a dict (already resolved, as Forge leaves it after
    ``split_state_dict``) or a callable (as it is declared before the load).
    Passing ``MISSING`` omits the attribute entirely.
    """
    body = {}

    if image_model is not None:
        body["unet_config"] = {"image_model": image_model}
    if huggingface_repo is not None:
        body["huggingface_repo"] = huggingface_repo
    if text_prefix is not None:
        body["text_encoder_key_prefix"] = list(text_prefix)
    if vae_prefix is not None:
        body["vae_key_prefix"] = list(vae_prefix)
    if latent_format is not None:
        body["latent_format"] = type(latent_format, (), {})
    if clip_target is not MISSING:
        body["clip_target"] = clip_target
    if with_save_processors:
        body["process_clip_state_dict_for_saving"] = lambda self, sd: sd
        body["process_vae_state_dict_for_saving"] = lambda self, sd: sd

    return type(class_name, (), body)()


def make_engine(config):
    return type("FakeEngine", (), {"model_config": config})()


def flux_like(class_name, **kw):
    kw.setdefault(
        "clip_target", {"clip_l": "text_encoder", "t5xxl": "text_encoder_2"}
    )
    return make_engine(make_config(class_name, **kw))


def anima_like(class_name, **kw):
    kw.setdefault("image_model", "anima")
    kw.setdefault("huggingface_repo", "circlestone-labs/Anima")
    kw.setdefault("clip_target", {"qwen3_06b.transformer": "text_encoder"})
    kw.setdefault("latent_format", "Wan21")
    return make_engine(make_config(class_name, **kw))


class ClassNameIsNotTheContractTests(unittest.TestCase):
    def test_two_names_with_identical_capabilities_agree_semantically(self):
        a = capability_profile_from_engine(flux_like("OldForgeName"))
        b = capability_profile_from_engine(flux_like("RenamedByForge"))
        self.assertEqual(a.semantic_fingerprint, b.semantic_fingerprint)

    def test_diagnostic_fingerprint_may_tell_them_apart(self):
        a = capability_profile_from_engine(flux_like("OldForgeName"))
        b = capability_profile_from_engine(flux_like("RenamedByForge"))
        self.assertNotEqual(a.diagnostic_fingerprint, b.diagnostic_fingerprint)

    def test_semantic_fingerprint_does_not_contain_the_class_name(self):
        profile = capability_profile_from_engine(flux_like("VeryDistinctiveName"))
        self.assertNotIn("VeryDistinctiveName", profile.semantic_fingerprint)


class TargetNormalizationTests(unittest.TestCase):
    def test_dotted_target_splits_into_role_and_internal_prefix(self):
        profile = capability_profile_from_engine(anima_like("Whatever"))
        (target,) = profile.text_targets
        self.assertEqual("qwen3_06b", target.forge_target)
        self.assertEqual(("qwen3_06b.transformer.",), target.internal_prefixes)
        self.assertEqual("text_encoder", target.kind)

    def test_undotted_target_keeps_its_own_name_as_prefix(self):
        engine = make_engine(
            make_config("Whatever", clip_target={"umt5xxl": "text_encoder"})
        )
        (target,) = capability_profile_from_engine(engine).text_targets
        self.assertEqual("umt5xxl", target.forge_target)
        self.assertEqual(("umt5xxl.",), target.internal_prefixes)

    def test_targets_keep_declaration_order(self):
        profile = capability_profile_from_engine(flux_like("Whatever"))
        self.assertEqual(
            ("clip_l", "t5xxl"),
            tuple(t.forge_target for t in profile.text_targets),
        )

    def test_vae_target_comes_from_the_vae_key_prefix(self):
        profile = capability_profile_from_engine(flux_like("Whatever"))
        self.assertIsNotNone(profile.vae_target)
        self.assertEqual("vae", profile.vae_target.kind)
        self.assertEqual(("vae.",), profile.vae_target.internal_prefixes)


class ClipTargetFormTests(unittest.TestCase):
    """Forge replaces the method with a dict during ``split_state_dict``."""

    def test_callable_form_is_resolved(self):
        engine = make_engine(
            make_config(
                "Whatever",
                clip_target=lambda self, state_dict=None: {
                    "qwen3_4b.transformer": "text_encoder"
                },
            )
        )
        (target,) = capability_profile_from_engine(engine).text_targets
        self.assertEqual("qwen3_4b", target.forge_target)

    def test_resolved_dict_form_is_accepted_unchanged(self):
        engine = make_engine(
            make_config("Whatever", clip_target={"qwen3_4b.transformer": "text_encoder"})
        )
        (target,) = capability_profile_from_engine(engine).text_targets
        self.assertEqual("qwen3_4b", target.forge_target)

    def test_both_forms_produce_the_same_semantic_fingerprint(self):
        as_dict = make_engine(
            make_config("A", clip_target={"qwen3_4b.transformer": "text_encoder"})
        )
        as_call = make_engine(
            make_config(
                "A",
                clip_target=lambda self, state_dict=None: {
                    "qwen3_4b.transformer": "text_encoder"
                },
            )
        )
        self.assertEqual(
            capability_profile_from_engine(as_dict).semantic_fingerprint,
            capability_profile_from_engine(as_call).semantic_fingerprint,
        )

    def test_a_conditional_target_that_returns_less_is_reported_as_less(self):
        """Forge does not raise when a component was not supplied -- the slot
        simply stops being declared. That must surface as a shorter target
        list, never as an error."""
        engine = make_engine(
            make_config(
                "Whatever",
                clip_target=lambda self, state_dict=None: {"t5xxl": "text_encoder"},
            )
        )
        profile = capability_profile_from_engine(engine)
        self.assertEqual(("t5xxl",), tuple(t.forge_target for t in profile.text_targets))

    def test_empty_clip_target_yields_no_text_targets_and_does_not_raise(self):
        engine = make_engine(make_config("BaseLike", clip_target={}))
        profile = capability_profile_from_engine(engine)
        self.assertEqual((), profile.text_targets)
        self.assertIsNotNone(profile.vae_target)


class IncompleteContractTests(unittest.TestCase):
    def test_absent_clip_target_names_the_missing_capability(self):
        engine = make_engine(make_config("Whatever"))
        with self.assertRaises(ForgeCapabilityError) as ctx:
            capability_profile_from_engine(engine)
        self.assertIn("clip_target", str(ctx.exception))

    def test_absent_text_prefix_names_the_missing_capability(self):
        engine = make_engine(
            make_config("Whatever", clip_target={"clip_l": "text_encoder"}, text_prefix=None)
        )
        with self.assertRaises(ForgeCapabilityError) as ctx:
            capability_profile_from_engine(engine)
        self.assertIn("text_encoder_key_prefix", str(ctx.exception))

    def test_absent_vae_prefix_names_the_missing_capability(self):
        engine = make_engine(
            make_config("Whatever", clip_target={"clip_l": "text_encoder"}, vae_prefix=None)
        )
        with self.assertRaises(ForgeCapabilityError) as ctx:
            capability_profile_from_engine(engine)
        self.assertIn("vae_key_prefix", str(ctx.exception))

    def test_an_engine_without_a_model_config_is_rejected_clearly(self):
        with self.assertRaises(ForgeCapabilityError) as ctx:
            capability_profile_from_engine(object())
        self.assertIn("model_config", str(ctx.exception))

    def test_save_processors_are_not_part_of_the_contract(self):
        """BASE supplies them to every architecture, so gating on them would
        either pass vacuously or reject everything. A config without the
        override must still produce a usable profile."""
        engine = flux_like("Whatever", with_save_processors=False)
        profile = capability_profile_from_engine(engine)
        self.assertEqual(2, len(profile.text_targets))
        self.assertIsNotNone(profile.vae_target)

    def test_profile_exposes_no_save_capability_flags(self):
        profile = capability_profile_from_engine(flux_like("Whatever"))
        self.assertFalse(hasattr(profile, "can_save_clip"))
        self.assertFalse(hasattr(profile, "can_save_vae"))


class PrefixTests(unittest.TestCase):
    def test_multiple_text_prefixes_are_preserved_in_order(self):
        """Chroma declares both ``text_encoders.`` and ``cond_stage_model.``."""
        engine = make_engine(
            make_config(
                "ChromaLike",
                clip_target={"t5xxl": "text_encoder"},
                text_prefix=("text_encoders.", "cond_stage_model."),
            )
        )
        profile = capability_profile_from_engine(engine)
        self.assertEqual(
            ("text_encoders.", "cond_stage_model."), profile.text_encoder_key_prefix
        )

    def test_multiple_vae_prefixes_are_preserved_in_order(self):
        """Mugen declares both ``vae.`` and ``first_stage_model.``."""
        engine = make_engine(
            make_config(
                "MugenLike",
                clip_target={"clip_l": "text_encoder"},
                vae_prefix=("vae.", "first_stage_model."),
            )
        )
        profile = capability_profile_from_engine(engine)
        self.assertEqual(("vae.", "first_stage_model."), profile.vae_key_prefix)

    def test_the_modern_convention_is_the_text_encoders_prefix(self):
        modern = capability_profile_from_engine(flux_like("Whatever"))
        self.assertTrue(modern.uses_modular_convention)

    def test_the_traditional_convention_is_cond_stage_model_only(self):
        engine = make_engine(
            make_config(
                "SDXLLike",
                image_model=None,
                clip_target={"clip_l": "text_encoder", "clip_g": "text_encoder_2"},
                text_prefix=("cond_stage_model.",),
                vae_prefix=("first_stage_model.",),
                latent_format="SDXL",
            )
        )
        profile = capability_profile_from_engine(engine)
        self.assertFalse(profile.uses_modular_convention)


class TraditionalArchitecturesAreStillProfilesTests(unittest.TestCase):
    """SDXL is not excluded from AIO -- its encoders are simply embedded."""

    def test_sdxl_like_declares_two_text_targets_and_a_vae(self):
        engine = make_engine(
            make_config(
                "SDXLLike",
                image_model=None,
                clip_target={"clip_l": "text_encoder", "clip_g": "text_encoder_2"},
                text_prefix=("cond_stage_model.",),
                vae_prefix=("first_stage_model.",),
                latent_format="SDXL",
            )
        )
        profile = capability_profile_from_engine(engine)
        self.assertEqual(
            ("clip_l", "clip_g"), tuple(t.forge_target for t in profile.text_targets)
        )
        self.assertIsNotNone(profile.vae_target)


class GenerativeEvidenceTests(unittest.TestCase):
    def test_rgb_latent_format_marks_a_non_generative_profile(self):
        """PiD is the only class with an RGB latent format -- it is an i2i
        upscaler, not a checkpoint architecture. Identified by capability."""
        engine = make_engine(
            make_config(
                "SomeUpscaler",
                image_model="pid",
                clip_target={"gemma2_2b": "text_encoder"},
                latent_format="RGB",
            )
        )
        profile = capability_profile_from_engine(engine)
        self.assertFalse(profile.is_generative)

    def test_a_latent_space_format_is_generative(self):
        self.assertTrue(capability_profile_from_engine(flux_like("W")).is_generative)


class AnimaIdentityTests(unittest.TestCase):
    def test_image_model_identifies_anima_after_a_class_rename(self):
        for name in ("Anima", "AnimaRenamedUpstream", "CompletelyDifferent"):
            with self.subTest(class_name=name):
                profile = capability_profile_from_engine(anima_like(name))
                self.assertTrue(is_anima_profile(profile))

    def test_repository_evidence_alone_also_identifies_anima(self):
        engine = anima_like("X", image_model="something_else")
        self.assertTrue(is_anima_profile(capability_profile_from_engine(engine)))

    def test_a_qwen_encoder_does_not_make_an_arbitrary_model_anima(self):
        engine = make_engine(
            make_config(
                "ZImageLike",
                image_model="lumina2",
                huggingface_repo="Tongyi-MAI/Z-Image-Turbo",
                clip_target={"qwen3_4b.transformer": "text_encoder"},
            )
        )
        self.assertFalse(is_anima_profile(capability_profile_from_engine(engine)))

    def test_every_anima_generation_shares_one_capability_family(self):
        """28, 40 and 52 blocks are one architecture at different depths."""
        prints = {
            capability_profile_from_engine(
                anima_like(f"Anima{blocks}")
            ).semantic_fingerprint
            for blocks in (28, 40, 52)
        }
        self.assertEqual(1, len(prints))


class GenericDiscoveryTests(unittest.TestCase):
    def test_a_future_complete_config_is_discovered_without_a_registry_entry(self):
        engine = make_engine(
            make_config(
                "NovaImageV7",
                image_model="nova_image",
                huggingface_repo="nova/Nova-Image-V7",
                clip_target={"nova_text.transformer": "text_encoder"},
            )
        )
        profile = capability_profile_from_engine(engine)
        self.assertEqual(
            ("nova_text",), tuple(t.forge_target for t in profile.text_targets)
        )
        self.assertTrue(profile.generic_discovery)

    def test_a_known_family_is_not_flagged_as_generic(self):
        self.assertFalse(capability_profile_from_engine(anima_like("X")).generic_discovery)


class ModuleHygieneTests(unittest.TestCase):
    def test_importing_the_module_does_not_pull_in_forge_or_gradio(self):
        import sys

        for banned in ("backend", "backend.loader", "gradio", "modules", "modules_forge"):
            self.assertNotIn(
                banned, sys.modules, f"{banned} must not be imported at module scope"
            )

    def test_the_profile_is_immutable(self):
        profile = capability_profile_from_engine(flux_like("W"))
        with self.assertRaises(Exception):
            profile.family_hint = "tampered"


if __name__ == "__main__":
    unittest.main()
