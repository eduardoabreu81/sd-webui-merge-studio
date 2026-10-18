"""Telling one component file from another by its header alone.

Every shape asserted here was measured from a real file in a Forge Neo install,
or taken from the model's published config where no local copy exists. The
comments say which, because a guessed signature that happens to pass is worse
than no signature at all.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safetensors_helpers import (  # noqa: E402
    as_fp8_mixed,
    as_fp8_scaled,
    build_header,
    clip_encoder,
    qwen_encoder,
    t5_encoder,
    vae_2d,
    vae_3d,
    write_safetensors_header,
)

from merge_studio.aux_inspector import (  # noqa: E402
    classify_component_header,
    detect_unsupported_storage,
    inspect_module,
)


def classify(tensors, metadata=None):
    return classify_component_header(build_header(tensors, metadata))


class QwenFamilyTests(unittest.TestCase):
    """Measured: 0.6B is 28 layers at hidden 1024; the 4B pair is 36 at 2560."""

    def test_qwen3_06b(self):
        result = classify(qwen_encoder(28, 1024))
        self.assertEqual("qwen3_06b", result["signature_id"])
        self.assertEqual("text_encoder", result["kind"])

    def test_qwen3_4b_has_no_vision_tower(self):
        result = classify(qwen_encoder(36, 2560))
        self.assertEqual("qwen3_4b", result["signature_id"])
        self.assertFalse(result["has_vision"])

    def test_qwen3vl_4b_is_the_same_shape_plus_a_vision_tower(self):
        result = classify(qwen_encoder(36, 2560, vision_blocks=24))
        self.assertEqual("qwen3vl_4b", result["signature_id"])
        self.assertTrue(result["has_vision"])

    def test_the_vision_tower_is_the_only_thing_separating_them(self):
        """Measured on real files: both are 36 layers, hidden 2560, vocab
        151936. If the tower were ever stripped they would be identical."""
        plain = classify(qwen_encoder(36, 2560))
        vl = classify(qwen_encoder(36, 2560, vision_blocks=24))
        self.assertEqual(plain["layers"], vl["layers"])
        self.assertEqual(plain["hidden"], vl["hidden"])
        self.assertNotEqual(plain["signature_id"], vl["signature_id"])

    def test_a_vl_file_is_not_offered_as_a_plain_qwen3_4b(self):
        self.assertNotEqual(
            "qwen3_4b", classify(qwen_encoder(36, 2560, vision_blocks=24))["signature_id"]
        )

    def test_vision_blocks_do_not_inflate_the_layer_count(self):
        result = classify(qwen_encoder(36, 2560, vision_blocks=24))
        self.assertEqual(36, result["layers"])


class LargerQwenEncoderTests(unittest.TestCase):
    """Measured 2026-10-18 from Comfy-Org's bf16, int8 convrot and w4a8 files:
    Qwen3-VL 8B (Qwen-Image 2.1) is 36 layers at hidden 4096 with a vision
    tower; Qwen3.5 9B (MOSS-Image) is 32 at 4096, vocab 248320, also with one."""

    def test_qwen3vl_8b(self):
        self.assertEqual(
            "qwen3vl_8b", classify(qwen_encoder(36, 4096, vision_blocks=27))["signature_id"]
        )

    def test_qwen3vl_8b_is_not_offered_as_flux2_klein_9bs_qwen3_8b(self):
        # Same depth and width; Klein 9B's encoder has no vision tower.
        self.assertEqual("qwen3_8b", classify(qwen_encoder(36, 4096))["signature_id"])
        self.assertNotEqual(
            "qwen3_8b", classify(qwen_encoder(36, 4096, vision_blocks=27))["signature_id"]
        )

    def test_qwen35_9b_in_its_own_layout(self):
        result = classify(
            qwen_encoder(32, 4096, vision_blocks=27, vocab=248320,
                         language_prefix="model.language_model.", mtp_layers=1)
        )
        self.assertEqual("qwen35_9b", result["signature_id"])
        # The prediction head's single layer does not add to the depth.
        self.assertEqual(32, result["layers"])


class T5FamilyTests(unittest.TestCase):
    """Forge's own discriminator is the vocabulary size."""

    def test_t5xxl_is_classified_before_generic_encoder_decoder_rules(self):
        result = classify(t5_encoder(vocab=32128))
        self.assertEqual("t5xxl", result["signature_id"])
        self.assertEqual("text_encoder", result["kind"])

    def test_umt5_uses_the_forge_vocab_discriminator(self):
        self.assertEqual("umt5xxl", classify(t5_encoder(vocab=256384))["signature_id"])

    def test_a_t5_is_not_a_vae_despite_its_encoder_keys(self):
        """``encoder.block.*`` must not trip the generic encoder/decoder rule
        that used to decide a file was an autoencoder."""
        self.assertNotEqual("vae", classify(t5_encoder(vocab=32128))["kind"])


class ClipFamilyTests(unittest.TestCase):
    """Hidden sizes from the published CLIP configs."""

    def test_clip_l(self):
        self.assertEqual("clip_l", classify(clip_encoder(hidden=768))["signature_id"])

    def test_clip_g(self):
        self.assertEqual(
            "clip_g", classify(clip_encoder(hidden=1280, layers=32))["signature_id"]
        )


class VaeTests(unittest.TestCase):
    """The 3D families have no ``decoder.conv_in.weight`` at all."""

    def test_the_qwen_image_wan_anima_family_is_video_capable(self):
        result = classify(vae_3d())
        self.assertEqual("vae", result["signature_id"])
        self.assertEqual("vae", result["kind"])
        self.assertTrue(result["video_capable"])

    def test_latent_channels_are_read_without_decoder_conv_in_weight(self):
        tensors = vae_3d(latent_channels=16)
        self.assertNotIn("decoder.conv_in.weight", tensors)
        self.assertEqual(16, classify(tensors)["latent_channels"])

    def test_the_flux_autoencoder_is_two_dimensional_with_sixteen_channels(self):
        result = classify(vae_2d(latent_channels=16))
        self.assertEqual("vae", result["kind"])
        self.assertFalse(result["video_capable"])
        self.assertEqual(16, result["latent_channels"])

    def test_the_sd_autoencoder_has_four_channels(self):
        result = classify(vae_2d(latent_channels=4))
        self.assertFalse(result["video_capable"])
        self.assertEqual(4, result["latent_channels"])

    def test_latent_channels_alone_do_not_separate_flux_from_qwen_image(self):
        """Both are 16. Only the convolution rank tells them apart, which is
        why video_capable carries the distinction."""
        flux = classify(vae_2d(latent_channels=16))
        qwen = classify(vae_3d(latent_channels=16))
        self.assertEqual(flux["latent_channels"], qwen["latent_channels"])
        self.assertNotEqual(flux["video_capable"], qwen["video_capable"])

    def test_an_encoder_is_never_classified_as_a_vae(self):
        for name, tensors in (
            ("t5xxl", t5_encoder(vocab=32128)),
            ("qwen3_06b", qwen_encoder(28, 1024)),
            ("clip_l", clip_encoder(hidden=768)),
        ):
            with self.subTest(component=name):
                self.assertEqual("text_encoder", classify(tensors)["kind"])


class VaeFitsLatentFormatTests(unittest.TestCase):
    """Which autoencoder each latent space is offered. Run against every VAE
    in the reference library on 2026-10-18: each one matched only its own
    family, and none that was offered before stopped being offered."""

    VAES = {
        "flux_ae": vae_2d(latent_channels=16),
        "flux2_ae": vae_2d(latent_channels=32),
        "wan_family": vae_3d(latent_channels=16),
        "qwen_image21": vae_3d(latent_channels=64),
        "sd": vae_2d(latent_channels=4),
    }
    EXPECTED = {
        "Flux": "flux_ae",
        # FLUX.2 Klein, ERNIE-Image, and the FLUX 3 and MOSS-Image extensions.
        "Flux2": "flux2_ae",
        "Wan21": "wan_family",
        "QwenImage21": "qwen_image21",
        "QwenImage21Latent": "qwen_image21",
        "SDXL": "sd",
    }

    def test_each_latent_space_accepts_only_its_own_autoencoder(self):
        from merge_studio.component_registry import vae_fits_latent_format

        for latent, wanted in self.EXPECTED.items():
            for name, tensors in self.VAES.items():
                with self.subTest(latent=latent, vae=name):
                    self.assertEqual(
                        name == wanted,
                        vae_fits_latent_format(classify(tensors), latent),
                    )

    def test_qwen_image_21s_3d_vae_is_not_offered_to_the_wan_family(self):
        # Krea 2, Anima and Qwen-Image all read Wan21 latents; this VAE is 3D
        # like theirs but carries 64 channels.
        from merge_studio.component_registry import vae_fits_latent_format

        self.assertFalse(vae_fits_latent_format(classify(vae_3d(latent_channels=64)), "Wan21"))


class StorageTests(unittest.TestCase):
    def test_a_plain_component_is_supported(self):
        result = classify(qwen_encoder(28, 1024))
        self.assertEqual("plain", result["storage_kind"])
        self.assertTrue(result["supported_storage"])

    def test_a_scaled_fp8_component_is_supported(self):
        """fp8_scaled is how Forge distributes most encoders, not an oddity."""
        result = classify(as_fp8_scaled(qwen_encoder(36, 2560)))
        self.assertEqual("fp8_scaled", result["storage_kind"])
        self.assertTrue(result["supported_storage"])

    def test_a_mixed_fp8_component_is_supported(self):
        result = classify(as_fp8_mixed(qwen_encoder(36, 2560)))
        self.assertEqual("fp8_mixed", result["storage_kind"])
        self.assertTrue(result["supported_storage"])

    def test_scaling_does_not_change_what_the_component_is(self):
        self.assertEqual(
            classify(qwen_encoder(36, 2560))["signature_id"],
            classify(as_fp8_scaled(qwen_encoder(36, 2560)))["signature_id"],
        )

    def test_nf4_is_rejected(self):
        tensors = dict(qwen_encoder(28, 1024))
        tensors["model.layers.0.self_attn.q_proj.weight.absmax"] = ("U8", [64])
        tensors["model.layers.0.self_attn.q_proj.weight.quant_state.bitsandbytes__nf4"] = (
            "U8",
            [16],
        )
        result = classify(tensors)
        self.assertEqual("nf4", result["storage_kind"])
        self.assertFalse(result["supported_storage"])

    def test_gguf_evidence_is_rejected(self):
        result = classify(qwen_encoder(28, 1024), metadata={"quantization": "gguf"})
        self.assertEqual("gguf", result["storage_kind"])
        self.assertFalse(result["supported_storage"])

    def test_nunchaku_svdq_is_rejected(self):
        tensors = dict(qwen_encoder(28, 1024))
        tensors["model.layers.0.self_attn.q_proj.qweight"] = ("I8", [1024, 512])
        tensors["model.layers.0.self_attn.q_proj.wtscale"] = ("F32", [1])
        tensors["model.layers.0.self_attn.q_proj.smooth_factor"] = ("F16", [1024])
        result = classify(tensors)
        self.assertEqual("nunchaku_svdq", result["storage_kind"])
        self.assertFalse(result["supported_storage"])


class FilenameIsNotEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def test_a_file_named_fp8_that_holds_f16_reports_f16(self):
        """``animaLLMLayerwiseFP8_v1.safetensors`` is a real file in the
        reference library: its name says FP8 and its 310 tensors are all F16.
        It is a Qwen3 0.6B stored densely, and must read as exactly that."""
        path = write_safetensors_header(
            self.path("animaLLMLayerwiseFP8_v1.safetensors"),
            qwen_encoder(28, 1024, dtype="F16"),
        )
        info = inspect_module(path)
        self.assertEqual("qwen3_06b", info["signature_id"])
        self.assertEqual("plain", info["storage_kind"])
        self.assertTrue(info["supported_storage"])
        self.assertNotIn("F8", info["precision"].upper())

    def test_a_scaled_file_named_plainly_still_reads_as_scaled(self):
        path = write_safetensors_header(
            self.path("just_an_encoder.safetensors"),
            as_fp8_scaled(qwen_encoder(36, 2560)),
        )
        self.assertEqual("fp8_scaled", inspect_module(path)["storage_kind"])

    def test_a_pt_path_is_refused_before_the_header_is_read(self):
        path = self.path("encoder.pt")
        with open(path, "wb") as f:
            f.write(b"not a safetensors file at all")
        self.assertIsNotNone(detect_unsupported_storage({}, path))
        self.assertIn("error", inspect_module(path))

    def test_a_gguf_path_is_refused(self):
        self.assertIsNotNone(detect_unsupported_storage({}, "encoder.gguf"))


class AnimaNegativeFixtureTests(unittest.TestCase):
    def test_the_legacy_expanded_adapter_is_not_a_text_encoder(self):
        """A cross-attention adapter, not an encoder. It fills no slot."""
        tensors = {
            "adapter.blocks.0.cross_attn.q_proj.weight": ("BF16", [2048, 2048]),
            "adapter.blocks.0.cross_attn.k_proj.weight": ("BF16", [2048, 2560]),
            "adapter.proj_in.weight": ("BF16", [2048, 2560]),
        }
        self.assertNotEqual("text_encoder", classify(tensors)["kind"])

    def test_connector_keys_are_diffusion_evidence_not_a_component(self):
        """``net.anima_v2_connector.`` lives inside the diffusion model; a file
        carrying it is not a standalone component."""
        tensors = {
            "net.anima_v2_connector.quality_anchor.layer_mix_logits": ("BF16", [12]),
            "net.anima_v2_connector.semantic_resampler.blocks.0.attn.q.weight": (
                "BF16",
                [1024, 1024],
            ),
        }
        result = classify(tensors)
        self.assertNotEqual("text_encoder", result["kind"])
        self.assertNotEqual("vae", result["kind"])


class UnknownStaysUnknownTests(unittest.TestCase):
    def test_an_unrecognised_layout_reports_measured_evidence_not_a_guess(self):
        result = classify(qwen_encoder(99, 7777))
        self.assertIsNone(result["signature_id"])
        self.assertEqual(99, result["layers"])
        self.assertEqual(7777, result["hidden"])
        self.assertIn("99", result["description"])

    def test_an_empty_header_is_unknown_and_does_not_raise(self):
        result = classify({})
        self.assertIsNone(result["signature_id"])
        self.assertEqual("unknown", result["kind"])


class InspectModuleKeepsItsExistingShapeTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = write_safetensors_header(
            os.path.join(self.dir.name, "qwen_3_06b_base.safetensors"),
            qwen_encoder(28, 1024),
        )

    def test_the_pre_existing_keys_are_all_still_there(self):
        info = inspect_module(self.path)
        for key in (
            "kind",
            "description",
            "filename",
            "filepath",
            "file_size",
            "size_str",
            "total_tensors",
            "precision",
            "raw_metadata",
        ):
            with self.subTest(key=key):
                self.assertIn(key, info)

    def test_the_new_keys_are_added(self):
        info = inspect_module(self.path)
        for key in (
            "signature_id",
            "latent_channels",
            "video_capable",
            "storage_kind",
            "supported_storage",
        ):
            with self.subTest(key=key):
                self.assertIn(key, info)


if __name__ == "__main__":
    unittest.main()
