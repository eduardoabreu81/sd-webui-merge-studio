"""Turning what the user picked into a plan, or into a clear refusal.

Everything here is pure: no Forge, no engine, no real files. The component
inspector is injected, so a signature mismatch can be staged exactly.

The rule this enforces is the one that gives AIO its meaning -- every slot the
architecture declares has to be filled. A slot left empty is not an error to
raise, it is a fact to report, so the interface can name what is missing and
offer UNet only instead.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from component_bundle import (  # noqa: E402
    ComponentPlan,
    ComponentSelection,
    ComponentValidationError,
    build_component_plan,
)
from component_registry import SupportState  # noqa: E402
from forge_capabilities import capability_profile_from_engine  # noqa: E402


def module_info(signature_id, *, storage="plain", supported=True, precision="BF16", kind=None):
    """Shaped like `aux_inspector.inspect_module`'s return value."""
    return {
        "kind": kind or ("vae" if signature_id == "vae" else "text_encoder"),
        "signature_id": signature_id,
        "storage_kind": storage,
        "supported_storage": supported,
        "precision": precision,
        "latent_channels": 16 if signature_id == "vae" else None,
        "video_capable": signature_id == "vae",
    }


def inspector(mapping):
    """An inspect_fn that answers from a path -> info table."""

    def inspect(path):
        try:
            return mapping[os.path.basename(path)]
        except KeyError:
            return {"error": f"no fixture for {path}"}

    return inspect


def engine(clip_target, *, image_model="anima", text_prefix=("text_encoders.",),
           vae_prefix=("vae.",), latent="Wan21", repo="circlestone-labs/Anima"):
    config = type(
        "Config",
        (),
        {
            "unet_config": {"image_model": image_model} if image_model else {},
            "huggingface_repo": repo,
            "clip_target": dict(clip_target),
            "text_encoder_key_prefix": list(text_prefix),
            "vae_key_prefix": list(vae_prefix),
            "latent_format": type(latent, (), {}),
        },
    )()
    return type("FakeEngine", (), {"model_config": config})()


ANIMA = {"qwen3_06b.transformer": "text_encoder"}
FLUX = {"clip_l": "text_encoder", "t5xxl": "text_encoder_2"}


def anima_profile():
    return capability_profile_from_engine(engine(ANIMA))


def flux_profile():
    return capability_profile_from_engine(
        engine(FLUX, image_model="flux", latent="Flux", repo="black-forest-labs/FLUX.1-dev")
    )


def plan(selections, *, profile=None, files=None, info=None, architecture_id="anima"):
    return build_component_plan(
        architecture_id,
        "A.safetensors",
        info or {},
        selections,
        capability_profile=profile or anima_profile(),
        inspect_fn=inspector(files or {}),
    )


ANIMA_FILES = {
    "qwen.safetensors": module_info("qwen3_06b"),
    "vae.safetensors": module_info("vae"),
}


def anima_complete():
    return [
        ComponentSelection("qwen3_06b", "file", "qwen.safetensors"),
        ComponentSelection("vae", "file", "vae.safetensors"),
    ]


class CompletePlanTests(unittest.TestCase):
    def test_a_filled_plan_has_nothing_missing(self):
        result = plan(anima_complete(), files=ANIMA_FILES)
        self.assertIsInstance(result, ComponentPlan)
        self.assertEqual((), result.missing_slots)
        self.assertEqual(("qwen3_06b", "vae"), tuple(c.slot_id for c in result.components))

    def test_the_resolved_component_records_its_physical_source_precision(self):
        files = {
            "qwen.safetensors": module_info("qwen3_06b", precision="FP16"),
            "vae.safetensors": module_info("vae", precision="F32"),
        }
        result = plan(anima_complete(), files=files)
        by_slot = {c.slot_id: c for c in result.components}
        self.assertEqual("FP16", by_slot["qwen3_06b"].source_precision)
        self.assertEqual("F32", by_slot["vae"].source_precision)

    def test_the_default_output_format_is_same(self):
        result = plan(anima_complete(), files=ANIMA_FILES)
        self.assertTrue(all(c.output_format == "same" for c in result.components))

    def test_the_plan_carries_the_capability_fingerprint(self):
        profile = anima_profile()
        result = plan(anima_complete(), profile=profile, files=ANIMA_FILES)
        self.assertEqual(profile.semantic_fingerprint, result.capability_fingerprint)

    def test_the_plan_is_immutable(self):
        result = plan(anima_complete(), files=ANIMA_FILES)
        with self.assertRaises(Exception):
            result.missing_slots = ("tampered",)


class MissingSlotsAreReportedNotRaisedTests(unittest.TestCase):
    """The interface needs to name what is missing, so an incomplete plan is
    still built. Blocking is the caller's job."""

    def test_an_empty_selection_reports_every_declared_slot(self):
        result = plan([], files=ANIMA_FILES)
        self.assertEqual(("qwen3_06b", "vae"), result.missing_slots)

    def test_a_partial_selection_reports_only_what_is_left(self):
        result = plan(
            [ComponentSelection("qwen3_06b", "file", "qwen.safetensors")],
            files=ANIMA_FILES,
        )
        self.assertEqual(("vae",), result.missing_slots)

    def test_flux_reports_whichever_encoder_is_missing_by_name(self):
        files = {
            "clip.safetensors": module_info("clip_l"),
            "ae.safetensors": module_info("vae"),
        }
        result = plan(
            [
                ComponentSelection("clip_l", "file", "clip.safetensors"),
                ComponentSelection("vae", "file", "ae.safetensors"),
            ],
            profile=flux_profile(),
            files=files,
            architecture_id="unknown",
        )
        self.assertEqual(("t5xxl",), result.missing_slots)

    def test_missing_slots_keep_registry_order(self):
        result = plan([], profile=flux_profile(), files={}, architecture_id="unknown")
        self.assertEqual(("clip_l", "t5xxl", "vae"), result.missing_slots)


class OrderingTests(unittest.TestCase):
    def test_files_are_ordered_by_the_architecture_not_by_the_form(self):
        files = {
            "clip.safetensors": module_info("clip_l"),
            "t5.safetensors": module_info("t5xxl"),
            "ae.safetensors": module_info("vae"),
        }
        selections = [
            ComponentSelection("vae", "file", "ae.safetensors"),
            ComponentSelection("t5xxl", "file", "t5.safetensors"),
            ComponentSelection("clip_l", "file", "clip.safetensors"),
        ]
        result = plan(
            selections, profile=flux_profile(), files=files, architecture_id="unknown"
        )
        self.assertEqual(
            ("clip.safetensors", "t5.safetensors", "ae.safetensors"),
            result.additional_state_dicts,
        )

    def test_embedded_components_contribute_no_external_file(self):
        info = {"embedded_components": ({"kind": "text_encoder", "role": "qwen3_06b", "readable": True},)}
        selections = [
            ComponentSelection("qwen3_06b", "embedded"),
            ComponentSelection("vae", "file", "vae.safetensors"),
        ]
        result = plan(selections, files=ANIMA_FILES, info=info)
        self.assertEqual(("vae.safetensors",), result.additional_state_dicts)

    def test_an_all_embedded_plan_passes_an_empty_external_list(self):
        info = {
            "embedded_components": (
                {"kind": "text_encoder", "role": "qwen3_06b", "readable": True},
                {"kind": "vae", "role": "vae", "readable": True},
            )
        }
        selections = [
            ComponentSelection("qwen3_06b", "embedded"),
            ComponentSelection("vae", "embedded"),
        ]
        result = plan(selections, files={}, info=info)
        self.assertEqual((), result.additional_state_dicts)
        self.assertEqual((), result.missing_slots)


class SelectionValidationTests(unittest.TestCase):
    def assert_refuses(self, selections, fragment, **kw):
        with self.assertRaises(ComponentValidationError) as ctx:
            plan(selections, **kw)
        self.assertIn(fragment, str(ctx.exception).lower())

    def test_a_slot_the_architecture_never_declared_is_refused(self):
        self.assert_refuses(
            [ComponentSelection("t5xxl", "file", "t5.safetensors")],
            "t5xxl",
            files={"t5.safetensors": module_info("t5xxl")},
        )

    def test_the_same_slot_twice_is_refused(self):
        self.assert_refuses(
            [
                ComponentSelection("qwen3_06b", "file", "qwen.safetensors"),
                ComponentSelection("qwen3_06b", "file", "qwen.safetensors"),
            ],
            "twice",
            files=ANIMA_FILES,
        )

    def test_a_non_safetensors_path_is_refused(self):
        self.assert_refuses(
            [ComponentSelection("qwen3_06b", "file", "qwen.pt")],
            "safetensors",
            files={"qwen.pt": module_info("qwen3_06b")},
        )

    def test_a_file_selection_without_a_path_is_refused(self):
        self.assert_refuses([ComponentSelection("qwen3_06b", "file")], "no file")

    def test_an_unreadable_file_is_refused_with_the_reason(self):
        self.assert_refuses(
            [ComponentSelection("qwen3_06b", "file", "absent.safetensors")],
            "absent.safetensors",
            files={},
        )

    def test_none_is_not_a_source_in_an_aio_plan(self):
        """AIO without an encoder is a contradiction; UNet only covers that."""
        self.assert_refuses(
            [ComponentSelection("qwen3_06b", "none")], "none", files=ANIMA_FILES
        )


class SignatureCompatibilityTests(unittest.TestCase):
    def assert_refuses(self, slot, signature, fragment="expected"):
        files = {"picked.safetensors": module_info(signature)}
        with self.assertRaises(ComponentValidationError) as ctx:
            plan([ComponentSelection(slot, "file", "picked.safetensors")], files=files,
                 profile=self.profile)
        message = str(ctx.exception)
        self.assertIn(signature, message)
        self.assertIn(fragment, message.lower())

    def setUp(self):
        self.profile = anima_profile()

    def test_kreas_vision_encoder_does_not_fit_animas_slot(self):
        self.assert_refuses("qwen3_06b", "qwen3vl_4b")

    def test_a_vae_does_not_fit_an_encoder_slot(self):
        self.assert_refuses("qwen3_06b", "vae")

    def test_an_unrecognised_file_is_refused_rather_than_assumed(self):
        files = {"mystery.safetensors": module_info(None)}
        with self.assertRaises(ComponentValidationError):
            plan([ComponentSelection("qwen3_06b", "file", "mystery.safetensors")], files=files)

    def test_qwen3_4b_and_qwen3vl_4b_do_not_substitute_for_each_other(self):
        """Identical at 36 layers and hidden 2560 apart from the vision tower,
        so nothing but the signature separates them."""
        zimage = capability_profile_from_engine(
            engine({"qwen3_4b.transformer": "text_encoder"}, image_model="lumina2",
                   latent="Flux", repo="Tongyi-MAI/Z-Image-Turbo")
        )
        krea = capability_profile_from_engine(
            engine({"qwen3vl_4b.transformer": "text_encoder"}, image_model="krea2",
                   repo="krea/Krea-2-Raw")
        )
        with self.assertRaises(ComponentValidationError):
            plan([ComponentSelection("qwen3_4b", "file", "vl.safetensors")],
                 profile=zimage, files={"vl.safetensors": module_info("qwen3vl_4b")},
                 architecture_id="unknown")
        with self.assertRaises(ComponentValidationError):
            plan([ComponentSelection("qwen3vl_4b", "file", "plain.safetensors")],
                 profile=krea, files={"plain.safetensors": module_info("qwen3_4b")},
                 architecture_id="unknown")

    def test_t5_does_not_fit_a_umt5_slot(self):
        wan = capability_profile_from_engine(
            engine({"umt5xxl": "text_encoder"}, image_model="wan2.1",
                   repo="Wan-AI/Wan2.1-T2V-14B")
        )
        with self.assertRaises(ComponentValidationError):
            plan([ComponentSelection("umt5xxl", "file", "t5.safetensors")],
                 profile=wan, files={"t5.safetensors": module_info("t5xxl")},
                 architecture_id="unknown")


class StorageTests(unittest.TestCase):
    def test_a_scaled_encoder_is_accepted(self):
        files = {
            "qwen.safetensors": module_info("qwen3_06b", storage="fp8_scaled"),
            "vae.safetensors": module_info("vae"),
        }
        result = plan(anima_complete(), files=files)
        self.assertEqual((), result.missing_slots)

    def test_a_mixed_encoder_is_accepted(self):
        files = {
            "qwen.safetensors": module_info("qwen3_06b", storage="fp8_mixed"),
            "vae.safetensors": module_info("vae"),
        }
        self.assertEqual((), plan(anima_complete(), files=files).missing_slots)

    def test_a_container_format_is_refused_with_its_name(self):
        for storage in ("gguf", "nunchaku_svdq", "nf4"):
            with self.subTest(storage=storage):
                files = {
                    "qwen.safetensors": module_info(
                        "qwen3_06b", storage=storage, supported=False
                    )
                }
                with self.assertRaises(ComponentValidationError) as ctx:
                    plan(
                        [ComponentSelection("qwen3_06b", "file", "qwen.safetensors")],
                        files=files,
                    )
                self.assertIn(storage, str(ctx.exception))


class EmbeddedSourceTests(unittest.TestCase):
    """"Keep what is in the file" has to mean the loader actually has it.

    Eighteen reference checkpoints carry an encoder under a namespace their own
    architecture never reads, so header presence is not evidence."""

    def test_embedded_is_refused_when_nothing_was_loaded(self):
        with self.assertRaises(ComponentValidationError) as ctx:
            plan([ComponentSelection("qwen3_06b", "embedded")], info={}, files={})
        self.assertIn("embedded", str(ctx.exception).lower())

    def test_embedded_is_refused_when_the_namespace_is_not_read(self):
        info = {
            "embedded_components": (
                {"kind": "text_encoder", "role": "qwen3_06b", "readable": False,
                 "namespace": "cond_stage_model."},
            )
        }
        with self.assertRaises(ComponentValidationError) as ctx:
            plan([ComponentSelection("qwen3_06b", "embedded")], info=info, files={})
        self.assertIn("cond_stage_model.", str(ctx.exception))

    def test_embedded_is_accepted_when_the_namespace_is_read(self):
        info = {
            "embedded_components": (
                {"kind": "text_encoder", "role": "qwen3_06b", "readable": True,
                 "namespace": "text_encoders."},
            )
        }
        result = plan(
            [
                ComponentSelection("qwen3_06b", "embedded"),
                ComponentSelection("vae", "file", "vae.safetensors"),
            ],
            info=info,
            files=ANIMA_FILES,
        )
        self.assertEqual((), result.missing_slots)

    def test_an_embedded_component_is_attributed_to_the_primary_file(self):
        info = {
            "embedded_components": (
                {"kind": "text_encoder", "role": "qwen3_06b", "readable": True},
            )
        }
        result = plan(
            [
                ComponentSelection("qwen3_06b", "embedded"),
                ComponentSelection("vae", "file", "vae.safetensors"),
            ],
            info=info,
            files=ANIMA_FILES,
        )
        embedded = next(c for c in result.components if c.source == "embedded")
        self.assertEqual("A.safetensors", embedded.path)


class TraditionalArchitectureTests(unittest.TestCase):
    """SDXL is not excluded from AIO. Its encoders are embedded, so only the
    VAE row is selectable."""

    def setUp(self):
        self.profile = capability_profile_from_engine(
            engine(
                {"clip_l": "text_encoder", "clip_g": "text_encoder_2"},
                image_model=None,
                text_prefix=("cond_stage_model.",),
                vae_prefix=("first_stage_model.",),
                latent="SDXL",
                repo="stabilityai/stable-diffusion-xl-base-1.0",
            )
        )

    def test_a_vae_selection_is_accepted(self):
        result = plan(
            [ComponentSelection("vae", "file", "vae.safetensors")],
            profile=self.profile,
            files=ANIMA_FILES,
            architecture_id="sdxl",
        )
        self.assertEqual((), result.missing_slots)

    def test_an_embedded_encoder_is_not_a_missing_slot(self):
        result = plan(
            [ComponentSelection("vae", "file", "vae.safetensors")],
            profile=self.profile,
            files=ANIMA_FILES,
            architecture_id="sdxl",
        )
        self.assertNotIn("clip_l", result.missing_slots)

    def test_selecting_an_embedded_only_encoder_from_a_file_is_refused(self):
        with self.assertRaises(ComponentValidationError) as ctx:
            plan(
                [ComponentSelection("clip_l", "file", "clip.safetensors")],
                profile=self.profile,
                files={"clip.safetensors": module_info("clip_l")},
                architecture_id="sdxl",
            )
        self.assertIn("cannot be replaced", str(ctx.exception).lower())


class AnimaPolicyTests(unittest.TestCase):
    def test_every_generation_uses_the_same_two_slots(self):
        for blocks in (28, 40, 52):
            with self.subTest(blocks=blocks):
                result = plan(
                    anima_complete(),
                    files=ANIMA_FILES,
                    info={"block_count": blocks},
                )
                self.assertEqual(
                    ("qwen3_06b", "vae"), tuple(c.slot_id for c in result.components)
                )

    def test_a_connector_bundle_does_not_gain_a_slot(self):
        info = {"block_count": 52, "embedded_prefixes": ("net.anima_v2_connector.",)}
        result = plan(anima_complete(), files=ANIMA_FILES, info=info)
        self.assertEqual(2, len(result.components))

    def test_selecting_qwen35_is_refused_because_no_slot_exists(self):
        with self.assertRaises(ComponentValidationError) as ctx:
            plan(
                [ComponentSelection("qwen35_4b", "file", "qwen35.safetensors")],
                files={"qwen35.safetensors": module_info("qwen35_4b")},
            )
        self.assertIn("qwen35_4b", str(ctx.exception))


class SupportStateTests(unittest.TestCase):
    def test_a_supported_family_says_so(self):
        self.assertEqual(
            SupportState.SUPPORTED, plan(anima_complete(), files=ANIMA_FILES).support
        )

    def test_an_experimental_family_keeps_its_warning(self):
        chroma = capability_profile_from_engine(
            engine({"t5xxl": "text_encoder"}, image_model="chroma",
                   latent="Flux", repo="Chroma")
        )
        files = {"t5.safetensors": module_info("t5xxl"), "ae.safetensors": module_info("vae")}
        result = plan(
            [
                ComponentSelection("t5xxl", "file", "t5.safetensors"),
                ComponentSelection("vae", "file", "ae.safetensors"),
            ],
            profile=chroma,
            files=files,
            architecture_id="unknown",
        )
        self.assertEqual(SupportState.EXPERIMENTAL, result.support)


class ProvisionalPlanTests(unittest.TestCase):
    """Without a loaded engine the slots are a header-derived guess, good
    enough to render a form and never good enough to save on."""

    def test_anima_gets_provisional_slots_from_its_header_id(self):
        result = build_component_plan(
            "anima", "A.safetensors", {}, [], inspect_fn=inspector({})
        )
        self.assertEqual(("qwen3_06b", "vae"), result.missing_slots)
        self.assertIsNone(result.capability_fingerprint)

    def test_an_unknown_architecture_offers_no_slots_until_forge_resolves_it(self):
        result = build_component_plan(
            "unknown", "A.safetensors", {}, [], inspect_fn=inspector({})
        )
        self.assertEqual((), result.missing_slots)
        self.assertEqual(SupportState.UNKNOWN, result.support)

    def test_a_selection_against_an_unresolved_architecture_is_refused(self):
        with self.assertRaises(ComponentValidationError):
            build_component_plan(
                "unknown",
                "A.safetensors",
                {},
                [ComponentSelection("qwen3_06b", "file", "qwen.safetensors")],
                inspect_fn=inspector(ANIMA_FILES),
            )


class ModuleHygieneTests(unittest.TestCase):
    def test_importing_the_module_does_not_pull_in_forge_or_gradio(self):
        for banned in ("backend", "backend.loader", "gradio", "modules_forge"):
            self.assertNotIn(banned, sys.modules)


if __name__ == "__main__":
    unittest.main()
