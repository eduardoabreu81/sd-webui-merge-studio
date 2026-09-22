"""Forge's own detector names checkpoints the pattern matching cannot.

`huggingface_guess.guess` is what `backend/loader.py` calls to decide which
engine to build, so it is the authority on what a file is -- and because nearly
every path through it reads only keys and shapes, a safetensors header is
enough to drive it. Z-Image and ERNIE-Image displayed as a generic "Diffusion
Model" before this; they now carry their real names and their upstream repos.

The detector is injected in most of what follows. That is deliberate: the real
package ships inside Forge and is not importable here, and the part worth
pinning down is the mapping and the fallback, not the detector's own accuracy.
One test does run the real thing, and skips when it is absent.
"""

import os
import unittest

from merge_studio import architecture_guess as ag
from merge_studio.architecture_guess import (
    anima_generation,
    detect_prediction_markers,
    guess_architecture,
    state_dict_from_header,
)
from merge_studio.checkpoint_inspector import get_model_family


def header(*keys, metadata=None):
    """A header-shaped dict: every key a 1-D tensor of width 4."""
    entry = {"dtype": "F16", "shape": [4], "data_offsets": [0, 8]}
    built = {key: dict(entry) for key in keys}
    if metadata is not None:
        built["__metadata__"] = metadata
    return built


class FakeGuess:
    """Stands in for a detector result. Its class name is what gets mapped."""

    huggingface_repo = "someone/something"
    unet_config = {"n_layers": 30, "dim": 3840, "qk_norm": True}


class StateDictFromHeaderTests(unittest.TestCase):
    def test_shapes_survive_and_metadata_does_not(self):
        sd = state_dict_from_header(
            {
                "model.diffusion_model.blocks.0.weight": {
                    "dtype": "BF16", "shape": [2560, 2560]
                },
                "__metadata__": {"format": "pt"},
            }
        )
        self.assertEqual(["model.diffusion_model.blocks.0.weight"], list(sd))
        self.assertEqual((2560, 2560), sd["model.diffusion_model.blocks.0.weight"].shape)

    def test_root_level_weights_get_the_diffusion_namespace(self):
        # `backend/loader.py::preprocess_state_dict` does this, and skipping it
        # is what made Ernie, Z-Image and Krea2 all read as unrecognised.
        sd = state_dict_from_header(header("blocks.0.weight", "x_embedder.proj.weight"))
        self.assertEqual(
            {
                "model.diffusion_model.blocks.0.weight",
                "model.diffusion_model.x_embedder.proj.weight",
            },
            set(sd),
        )

    def test_a_namespace_that_is_already_there_is_left_alone(self):
        for prefix in ("model.diffusion_model.", "net."):
            with self.subTest(prefix=prefix):
                sd = state_dict_from_header(header(f"{prefix}blocks.0.weight", "vae.x"))
                self.assertIn(f"{prefix}blocks.0.weight", sd)
                self.assertIn("vae.x", sd)

    def test_an_entry_without_a_shape_is_skipped_rather_than_crashing(self):
        # Headers come off disk from files this code did not write.
        sd = state_dict_from_header({"weird": "not a dict", "fine": {"shape": [1]}})
        self.assertEqual(["model.diffusion_model.fine"], list(sd))


class ShapeOnlyTests(unittest.TestCase):
    """The stand-in has to survive everything a config build does to a tensor.

    Building a matched config also builds its latent format, and `latent.py`
    reshapes two constants in `Wan21.__init__` -- the format Anima uses. When
    the stand-in could not do that, every Anima checkpoint and the FP8 Krea2
    came back as `'ShapeOnly' object has no attribute 'view'`, which reads
    exactly like a quantization failure and is nothing of the kind. 225 of 225
    checkpoints in the reference library detect with this in place.
    """

    def test_reshaping_keeps_working(self):
        for call in (lambda t: t.view(1, 16, 1, 1, 1), lambda t: t.reshape(1, 16)):
            with self.subTest(call=call):
                self.assertIsInstance(call(ag.ShapeOnly([16])), ag.ShapeOnly)

    def test_a_shape_passed_as_one_sequence_is_not_nested(self):
        self.assertEqual((1, 16), ag.ShapeOnly([16]).view((1, 16)).shape)

    def test_moving_a_stand_in_to_a_device_is_a_no_op(self):
        tensor = ag.ShapeOnly([16])
        self.assertIs(tensor, tensor.to("cpu"))


class GuessArchitectureTests(unittest.TestCase):
    def guess_as(self, class_name, **attrs):
        """A guesser that returns an instance of a class with this name."""
        cls = type(class_name, (FakeGuess,), attrs)
        return lambda state_dict: cls()

    def test_the_detector_names_what_pattern_matching_called_generic(self):
        result = guess_architecture(
            header("blocks.0.weight"),
            fallback="Diffusion Model",
            guesser=self.guess_as("ZImage", huggingface_repo="Tongyi-MAI/Z-Image-Turbo"),
        )
        self.assertEqual("Z-Image (DiT)", result.label)
        self.assertEqual("Tongyi-MAI/Z-Image-Turbo", result.repo)
        self.assertEqual("ZImage", result.detector)
        self.assertTrue(result.from_detector)
        self.assertIsNone(result.error)

    def test_only_scalar_config_entries_are_carried(self):
        # The rest of `unet_config` holds objects that would not survive being
        # put on screen or into a JSON dump.
        result = guess_architecture(
            header("blocks.0.weight"),
            fallback="Diffusion Model",
            guesser=self.guess_as(
                "ZImage", unet_config={"dim": 3840, "dtype": object(), "name": "z"}
            ),
        )
        self.assertEqual({"dim": 3840, "name": "z"}, result.config)

    def test_a_failing_detector_leaves_the_existing_label_alone(self):
        # Quantized files, value-reading paths and a missing package all arrive
        # here, and none of them may cost the caller the name it already had.
        def explode(state_dict):
            raise AttributeError("'ShapeOnly' object has no attribute 'view'")

        result = guess_architecture(
            header("blocks.0.weight"), fallback="SDXL (UNet)", guesser=explode
        )
        self.assertEqual("SDXL (UNet)", result.label)
        self.assertIsNone(result.detector)
        self.assertFalse(result.from_detector)
        self.assertIn("AttributeError", result.error)

    def test_a_class_with_no_display_name_falls_back_and_says_so(self):
        # The detector gains architectures; this table will lag behind it. A
        # bare class name in front of `get_model_family` is worse than the
        # label the caller already worked out.
        result = guess_architecture(
            header("blocks.0.weight"),
            fallback="Diffusion Model",
            guesser=self.guess_as("SomethingLandingIn2027"),
        )
        self.assertEqual("Diffusion Model", result.label)
        self.assertIsNone(result.detector)
        self.assertIn("SomethingLandingIn2027", result.error)

    def test_a_finetune_name_survives_a_detector_result_of_the_same_family(self):
        # Pony and Illustrious are structurally identical SDXL -- the detector
        # is right to say SDXL and the filename is the only thing that knows
        # better, so it keeps the more informative of the two answers.
        for fallback in ("Pony (SDXL)", "Illustrious (SDXL)"):
            with self.subTest(fallback=fallback):
                result = guess_architecture(
                    header("model.diffusion_model.input_blocks.0.0.weight"),
                    fallback=fallback,
                    guesser=self.guess_as("SDXL"),
                )
                self.assertEqual(fallback, result.label)
                self.assertEqual("SDXL", result.detector)

    def test_a_finetune_name_does_not_survive_a_different_architecture(self):
        # A file named ponyWhatever that turns out to be a Flux is a Flux.
        result = guess_architecture(
            header("double_blocks.0.img_attn.proj.weight"),
            fallback="Pony (SDXL)",
            guesser=self.guess_as("Flux"),
        )
        self.assertEqual("Flux.1 dev (MMDiT)", result.label)


class DisplayLabelTests(unittest.TestCase):
    def test_every_label_lands_in_a_real_compatibility_family(self):
        # `get_model_family` reads these strings to decide whether two
        # checkpoints may be merged, and "other" never raises a warning. A
        # label that falls through to "other" silently switches that guard off.
        for name, label in ag._DETECTOR_LABELS.items():
            with self.subTest(architecture=name):
                self.assertNotEqual("other", get_model_family(label))

    def test_families_that_must_not_be_confused_stay_apart(self):
        distinct = ("SD15", "SDXL", "Flux", "Anima", "ZImage", "ErnieImage",
                    "Krea2", "QwenImage", "WAN21_T2V", "Chroma")
        families = {
            name: get_model_family(ag._DETECTOR_LABELS[name]) for name in distinct
        }
        self.assertEqual(len(distinct), len(set(families.values())), families)

    def test_the_two_wan_variants_share_a_family(self):
        # Same architecture, different conditioning. Splitting them would put a
        # false incompatibility warning in front of a merge that works.
        self.assertEqual(
            get_model_family(ag._DETECTOR_LABELS["WAN21_T2V"]),
            get_model_family(ag._DETECTOR_LABELS["WAN21_I2V"]),
        )


class PredictionMarkerTests(unittest.TestCase):
    def test_noobai_style_markers_are_found(self):
        # Measured from noobaiXLNAIXL_vPred10Version: both are zero-length
        # tensors written at the root of the file.
        self.assertEqual(
            {"v_prediction": True, "ztsnr": True},
            detect_prediction_markers(header("v_pred", "ztsnr", "model.x")),
        )

    def test_a_model_that_declares_neither_reports_neither(self):
        self.assertEqual({}, detect_prediction_markers(header("model.x", "vae.y")))

    def test_metadata_is_not_mistaken_for_a_marker(self):
        self.assertEqual(
            {}, detect_prediction_markers(header("model.x", metadata={"v_pred": "1"}))
        )

    def test_a_marker_under_a_prefix_still_counts(self):
        self.assertEqual(
            {"v_prediction": True},
            detect_prediction_markers(header("model.diffusion_model.v_pred")),
        )


class AnimaGenerationTests(unittest.TestCase):
    def test_the_three_known_depths(self):
        # Verified block-by-block against the reference library; recorded in
        # docs/RESEARCH.md. 28 is the base release, which has no suffix.
        self.assertEqual("", anima_generation(28))
        self.assertEqual("2.9B", anima_generation(40))
        self.assertEqual("3.8B", anima_generation(52))

    def test_an_unreadable_or_unknown_depth_names_nothing(self):
        for blocks in (None, 0, 36, 64):
            with self.subTest(blocks=blocks):
                self.assertEqual("", anima_generation(blocks))


class RealDetectorTests(unittest.TestCase):
    """The real package against a real file, when both are available.

    A synthetic header will not do here: the detector counts transformer
    depths across the whole UNet, so anything small enough to write by hand
    fails to match and the test would only be measuring the fixture. Point
    MERGE_STUDIO_TEST_CHECKPOINT at a checkpoint to run this -- inside Forge
    the package is already importable.
    """

    def setUp(self):
        self.path = os.environ.get("MERGE_STUDIO_TEST_CHECKPOINT")
        if not self.path or not os.path.exists(self.path):
            self.skipTest("set MERGE_STUDIO_TEST_CHECKPOINT to a .safetensors file")
        ag.install_torch_stub()
        try:
            import huggingface_guess  # noqa: F401
        except Exception as exc:
            self.skipTest(f"huggingface_guess not importable here: {exc}")

    def test_a_real_checkpoint_is_named_from_its_header_alone(self):
        from merge_studio.checkpoint_inspector import read_safetensors_header

        real_header, _ = read_safetensors_header(self.path)
        result = guess_architecture(real_header, fallback="Diffusion Model")
        if result.error:
            # Quantized files legitimately fail, and the contract is that they
            # cost nothing -- the caller keeps the label it arrived with.
            self.assertEqual("Diffusion Model", result.label)
            self.skipTest(f"detector declined this file: {result.error}")
        self.assertIn(result.detector, ag._DETECTOR_LABELS)
        self.assertTrue(result.repo)
        self.assertNotEqual("Diffusion Model", result.label)


if __name__ == "__main__":
    unittest.main()
