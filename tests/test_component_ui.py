"""What the AIO form offers, decided without touching Gradio or Forge.

The row model is pure so it can be pinned down here; `scripts/merge_studio_ui.py`
only renders what these functions return. It imports Gradio and the WebUI's
`modules`, neither usable outside a running Forge, so the rendering itself is
verified by the smoke test in Task 10 rather than here.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from merge_studio.component_ui import (  # noqa: E402
    COMPONENT_FORMAT_CHOICES,
    KEEP_EMBEDDED,
    build_component_rows,
    missing_slot_message,
    parse_component_rows,
)
from merge_studio.forge_capabilities import capability_profile_from_engine  # noqa: E402


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


def profile(clip_target, **kw):
    return capability_profile_from_engine(engine(clip_target, **kw))


ANIMA = profile({"qwen3_06b.transformer": "text_encoder"})
FLUX = profile(
    {"clip_l": "text_encoder", "t5xxl": "text_encoder_2"},
    image_model="flux",
    latent="Flux",
    repo="black-forest-labs/FLUX.1-dev",
)
SDXL = profile(
    {"clip_l": "text_encoder", "clip_g": "text_encoder_2"},
    image_model=None,
    text_prefix=("cond_stage_model.",),
    vae_prefix=("first_stage_model.",),
    latent="SDXL",
    repo="stabilityai/stable-diffusion-xl-base-1.0",
)


def module(signature_id, *, storage="plain", supported=True):
    return {
        "kind": "vae" if signature_id == "vae" else "text_encoder",
        "signature_id": signature_id,
        "storage_kind": storage,
        "supported_storage": supported,
        "precision": "BF16",
    }


MODULES = {
    "qwen_3_06b_base.safetensors": module("qwen3_06b"),
    "qwen3vl_4b_fp8_scaled.safetensors": module("qwen3vl_4b", storage="fp8_scaled"),
    "t5xxl_fp16.safetensors": module("t5xxl"),
    "clip_l.safetensors": module("clip_l"),
    "qwen_image_vae.safetensors": module("vae"),
    "broken.gguf.safetensors": module("qwen3_06b", storage="gguf", supported=False),
}


def rows(info=None, save_mode="full", modules=None, prof=ANIMA):
    return build_component_rows(
        info or {}, save_mode, modules if modules is not None else MODULES, prof
    )


class RowCountFollowsTheArchitectureTests(unittest.TestCase):
    def test_anima_offers_an_encoder_and_a_vae(self):
        self.assertEqual(
            ["qwen3_06b", "vae"], [r.slot_id for r in rows()]
        )

    def test_flux_1_is_the_only_three_row_case(self):
        self.assertEqual(
            ["clip_l", "t5xxl", "vae"], [r.slot_id for r in rows(prof=FLUX)]
        )

    def test_sdxl_offers_only_the_vae(self):
        """Its encoders ship inside the checkpoint, so they are not selectable.
        That is not the same as SDXL being excluded from AIO."""
        self.assertEqual(["vae"], [r.slot_id for r in rows(prof=SDXL)])

    def test_unet_only_offers_no_rows_at_all(self):
        self.assertEqual((), rows(save_mode="unet_only"))

    def test_an_unresolved_architecture_offers_no_rows(self):
        self.assertEqual((), rows(prof=None))

    def test_rows_carry_a_readable_label(self):
        self.assertEqual(
            ["Qwen3 0.6B", "VAE"], [r.label for r in rows()]
        )


class ChoicesAreFilteredBySignatureTests(unittest.TestCase):
    def test_only_compatible_files_are_offered(self):
        encoder_row = rows()[0]
        self.assertIn("qwen_3_06b_base.safetensors", encoder_row.choices)
        self.assertNotIn("qwen3vl_4b_fp8_scaled.safetensors", encoder_row.choices)
        self.assertNotIn("qwen_image_vae.safetensors", encoder_row.choices)

    def test_the_vae_row_offers_vaes(self):
        vae_row = rows()[1]
        self.assertEqual(("qwen_image_vae.safetensors",), vae_row.choices)

    def test_an_unembeddable_file_is_never_offered(self):
        for row in rows():
            self.assertNotIn("broken.gguf.safetensors", row.choices)

    def test_a_scaled_encoder_is_offered_where_it_fits(self):
        """fp8_scaled is how Forge distributes most encoders."""
        krea = profile(
            {"qwen3vl_4b.transformer": "text_encoder"},
            image_model="krea2",
            repo="krea/Krea-2-Raw",
        )
        self.assertIn(
            "qwen3vl_4b_fp8_scaled.safetensors", rows(prof=krea)[0].choices
        )

    def test_an_empty_folder_leaves_the_row_with_no_choices(self):
        self.assertEqual((), rows(modules={})[0].choices)


class VaeMustFitTheLatentSpaceTests(unittest.TestCase):
    """Being a VAE is not enough. Offering a Flux AE for an Anima checkpoint
    builds a file that only fails at the post-save reload, gigabytes later."""

    VAES = {
        "qwen_image_vae.safetensors": {
            "kind": "vae", "signature_id": "vae", "supported_storage": True,
            "storage_kind": "plain", "video_capable": True, "latent_channels": 16,
        },
        "ae.safetensors": {
            "kind": "vae", "signature_id": "vae", "supported_storage": True,
            "storage_kind": "plain", "video_capable": False, "latent_channels": 16,
        },
        "sdxl_vae.safetensors": {
            "kind": "vae", "signature_id": "vae", "supported_storage": True,
            "storage_kind": "plain", "video_capable": False, "latent_channels": 4,
        },
    }

    def vae_choices(self, prof):
        return build_component_rows({}, "full", self.VAES, prof)[-1].choices

    def test_a_wan21_latent_space_takes_only_the_volumetric_autoencoder(self):
        self.assertEqual(("qwen_image_vae.safetensors",), self.vae_choices(ANIMA))

    def test_a_flux_latent_space_takes_the_flux_autoencoder(self):
        self.assertEqual(("ae.safetensors",), self.vae_choices(FLUX))

    def test_an_sdxl_latent_space_takes_the_four_channel_autoencoder(self):
        self.assertEqual(("sdxl_vae.safetensors",), self.vae_choices(SDXL))

    def test_latent_channels_alone_would_not_separate_flux_from_anima(self):
        """Both are 16. Only the convolution rank tells them apart."""
        self.assertEqual(
            self.VAES["ae.safetensors"]["latent_channels"],
            self.VAES["qwen_image_vae.safetensors"]["latent_channels"],
        )
        self.assertNotEqual(self.vae_choices(ANIMA), self.vae_choices(FLUX))

    def test_an_unknown_latent_format_does_not_hide_everything(self):
        """Better to allow a combination that turns out wrong than to leave a
        new architecture with no VAE it can use."""
        future = profile(
            {"nova_text.transformer": "text_encoder"},
            image_model="nova_image",
            latent="NovaLatent",
            repo="nova/Nova-Image-V7",
        )
        self.assertEqual(3, len(self.vae_choices(future)))


class NothingIsPreSelectedTests(unittest.TestCase):
    """Whoever wants to embed has the knowledge to choose. Guessing from the
    folder, or adopting whatever is set in Additional Modules, would put
    something in the file that nobody picked."""

    def test_a_fresh_row_has_no_selection(self):
        self.assertTrue(all(r.selected is None for r in rows()))

    def test_a_single_matching_file_is_still_not_auto_selected(self):
        only_one = {"qwen_3_06b_base.safetensors": module("qwen3_06b")}
        self.assertIsNone(rows(modules=only_one)[0].selected)


class KeepEmbeddedTests(unittest.TestCase):
    """Offered only when the loader can actually see it."""

    def test_a_diffusion_only_checkpoint_cannot_keep_anything(self):
        self.assertFalse(any(r.keep_embedded for r in rows()))

    def test_a_readable_embedded_component_can_be_kept(self):
        info = {
            "embedded_components": (
                {"kind": "text_encoder", "role": "qwen3_06b", "readable": True},
            )
        }
        self.assertTrue(rows(info)[0].keep_embedded)
        self.assertFalse(rows(info)[1].keep_embedded)

    def test_a_component_under_a_foreign_namespace_cannot_be_kept(self):
        """Eighteen reference checkpoints look like this: the header shows the
        encoder, and Forge discards it."""
        info = {
            "embedded_components": (
                {
                    "kind": "text_encoder",
                    "role": "qwen3_06b",
                    "readable": False,
                    "namespace": "cond_stage_model.",
                },
            )
        }
        row = rows(info)[0]
        self.assertFalse(row.keep_embedded)
        self.assertIn("cond_stage_model.", row.status)


class FormatChoicesTests(unittest.TestCase):
    def test_a_plain_row_offers_every_format(self):
        self.assertEqual(COMPONENT_FORMAT_CHOICES, rows()[0].format_choices)

    def test_same_is_always_the_default(self):
        self.assertEqual("same", rows()[0].format_choices[0][1])

    def test_a_scaled_selection_offers_only_same(self):
        """Converting it would mean dequantizing, which this path does not do."""
        krea = profile(
            {"qwen3vl_4b.transformer": "text_encoder"},
            image_model="krea2",
            repo="krea/Krea-2-Raw",
        )
        row = build_component_rows(
            {},
            "full",
            MODULES,
            krea,
            selections={"qwen3vl_4b": "qwen3vl_4b_fp8_scaled.safetensors"},
        )[0]
        self.assertEqual((("Same as component source", "same"),), row.format_choices)


class ExperimentalWarningTests(unittest.TestCase):
    def test_an_experimental_family_says_so(self):
        chroma = profile(
            {"t5xxl": "text_encoder"},
            image_model="chroma",
            latent="Flux",
            repo="Chroma",
        )
        self.assertTrue(any(r.warning for r in rows(prof=chroma)))

    def test_a_supported_family_carries_no_warning(self):
        self.assertFalse(any(r.warning for r in rows()))


class MissingSlotMessageTests(unittest.TestCase):
    def test_a_single_encoder_is_described_generically(self):
        self.assertEqual(
            "Select a text encoder to save as AIO — or switch to UNet only.",
            missing_slot_message(("qwen3_06b",)),
        )

    def test_a_missing_vae_says_vae(self):
        self.assertEqual(
            "Select a VAE to save as AIO — or switch to UNet only.",
            missing_slot_message(("vae",)),
        )

    def test_both_are_joined(self):
        self.assertEqual(
            "Select a text encoder and a VAE to save as AIO — or switch to UNet only.",
            missing_slot_message(("qwen3_06b", "vae")),
        )

    def test_flux_names_which_encoder_is_missing(self):
        """With two encoder rows, "a text encoder" would not say which one."""
        message = missing_slot_message(
            ("t5xxl",), all_slots=("clip_l", "t5xxl", "vae")
        )
        self.assertIn("T5XXL", message)
        self.assertNotIn("a text encoder", message)

    def test_nothing_missing_means_no_message(self):
        self.assertEqual("", missing_slot_message(()))

    def test_every_message_offers_the_way_out(self):
        for missing in (("vae",), ("qwen3_06b",), ("qwen3_06b", "vae")):
            with self.subTest(missing=missing):
                self.assertIn("UNet only", missing_slot_message(missing))


class ParsingTests(unittest.TestCase):
    def test_a_chosen_file_becomes_a_file_selection(self):
        parsed = parse_component_rows(
            [{"slot_id": "qwen3_06b", "value": "/models/qwen.safetensors", "format": "same"}]
        )
        self.assertEqual(
            [{"slot_id": "qwen3_06b", "source": "file",
              "path": "/models/qwen.safetensors", "output_format": "same"}],
            parsed,
        )

    def test_keeping_the_embedded_component_becomes_an_embedded_selection(self):
        parsed = parse_component_rows(
            [{"slot_id": "qwen3_06b", "value": KEEP_EMBEDDED, "format": "same"}]
        )
        self.assertEqual("embedded", parsed[0]["source"])
        self.assertIsNone(parsed[0]["path"])

    def test_an_unfilled_row_contributes_nothing(self):
        self.assertEqual(
            [], parse_component_rows([{"slot_id": "vae", "value": None, "format": "same"}])
        )

    def test_a_blank_value_contributes_nothing(self):
        self.assertEqual(
            [], parse_component_rows([{"slot_id": "vae", "value": "", "format": "same"}])
        )

    def test_the_requested_format_is_carried_through(self):
        parsed = parse_component_rows(
            [{"slot_id": "vae", "value": "/models/ae.safetensors", "format": "fp16"}]
        )
        self.assertEqual("fp16", parsed[0]["output_format"])

    def test_rows_keep_their_order(self):
        parsed = parse_component_rows(
            [
                {"slot_id": "clip_l", "value": "/a.safetensors", "format": "same"},
                {"slot_id": "t5xxl", "value": "/b.safetensors", "format": "same"},
            ]
        )
        self.assertEqual(["clip_l", "t5xxl"], [p["slot_id"] for p in parsed])


class ModuleHygieneTests(unittest.TestCase):
    def test_the_row_model_pulls_in_neither_gradio_nor_forge(self):
        for banned in ("gradio", "backend", "backend.loader", "modules_forge"):
            self.assertNotIn(banned, sys.modules)


if __name__ == "__main__":
    unittest.main()
