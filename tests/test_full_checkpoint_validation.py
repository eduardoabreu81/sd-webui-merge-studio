"""Proving a saved file is an AIO, rather than assuming it.

This is the step that carries the feature. Across the 222 checkpoints of the
reference library, only **two** embed their components under the namespace
their own architecture declares; thirty carry them under a foreign prefix that
Forge filters out and discards, and those files look self-contained while
depending on whatever is configured globally.

Reopening with an empty external-module list is the only way to tell the two
apart, so nothing here may be softened into a header check.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safetensors_helpers import write_safetensors_header  # noqa: E402

from component_bundle import (  # noqa: E402
    ComponentPlan,
    OutputValidation,
    ResolvedComponent,
    validate_aio_output,
)
from component_registry import SupportState  # noqa: E402


def component(slot_id, *, output_format="same", precision="BF16", path="src.safetensors",
              prefixes=None):
    return ResolvedComponent(
        slot_id=slot_id,
        source="file",
        path=path,
        signature_id=slot_id,
        output_format=output_format,
        source_precision=precision,
        internal_prefixes=prefixes or (f"{slot_id}.transformer.",),
    )


def anima_plan(**kw):
    defaults = dict(
        architecture_id="anima",
        support=SupportState.SUPPORTED,
        components=(
            component("qwen3_06b"),
            component("vae", prefixes=("vae.",)),
        ),
        text_namespace="text_encoders.",
        vae_namespace="vae.",
        slot_order=("qwen3_06b", "vae"),
    )
    defaults.update(kw)
    return ComponentPlan(**defaults)


def aio_tensors(*, text_ns="text_encoders.", vae_ns="vae.", dtype="BF16"):
    return {
        "model.diffusion_model.blocks.0.weight": ("BF16", [4, 4]),
        f"{text_ns}qwen3_06b.transformer.model.embed_tokens.weight": (dtype, [8, 4]),
        f"{vae_ns}decoder.conv1.weight": (dtype, [4, 4]),
    }


def accepting_loader(calls):
    def loader(path, additional_state_dicts):
        calls.append((path, tuple(additional_state_dicts)))
        return object()

    return loader


class ReloadIsMandatoryTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.output = write_safetensors_header(
            os.path.join(self.dir.name, "out.safetensors"), aio_tensors()
        )

    def test_a_good_output_validates(self):
        result = validate_aio_output(self.output, anima_plan(), loader=accepting_loader([]))
        self.assertIsInstance(result, OutputValidation)
        self.assertTrue(result.validated)
        self.assertEqual((), result.errors)
        self.assertEqual(("qwen3_06b", "vae"), result.checked_slots)

    def test_the_reload_passes_no_external_modules_at_all(self):
        """Inheriting even one would validate a file that cannot stand alone."""
        calls = []
        validate_aio_output(self.output, anima_plan(), loader=accepting_loader(calls))
        self.assertEqual([(self.output, ())], calls)

    def test_a_reload_failure_fails_validation_without_touching_the_file(self):
        def loader(path, additional_state_dicts):
            raise RuntimeError("missing text encoder weights")

        result = validate_aio_output(self.output, anima_plan(), loader=loader)
        self.assertFalse(result.validated)
        self.assertTrue(any("missing text encoder weights" in e for e in result.errors))
        self.assertTrue(os.path.exists(self.output))

    def test_a_failed_output_is_preserved_for_inspection(self):
        size_before = os.path.getsize(self.output)

        def loader(path, additional_state_dicts):
            raise RuntimeError("nope")

        validate_aio_output(self.output, anima_plan(), loader=loader)
        self.assertEqual(size_before, os.path.getsize(self.output))


class NamespaceTests(unittest.TestCase):
    """The expected namespaces come from the loaded engine, never from a table
    here -- a future architecture must not need an edit to be validated."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def write(self, name, tensors):
        return write_safetensors_header(os.path.join(self.dir.name, name), tensors)

    def test_an_output_under_a_foreign_namespace_is_refused(self):
        """The real-world shape: eighteen reference checkpoints look like this.
        Forge drops both components into its discarded bucket."""
        output = self.write(
            "sd_style.safetensors",
            aio_tensors(text_ns="cond_stage_model.", vae_ns="first_stage_model."),
        )
        result = validate_aio_output(output, anima_plan(), loader=accepting_loader([]))
        self.assertFalse(result.validated)
        joined = " ".join(result.errors)
        self.assertIn("text_encoders.", joined)
        self.assertIn("vae.", joined)

    def test_a_missing_text_encoder_namespace_is_named(self):
        output = self.write(
            "no_encoder.safetensors",
            {
                "model.diffusion_model.blocks.0.weight": ("BF16", [4, 4]),
                "vae.decoder.conv1.weight": ("BF16", [4, 4]),
            },
        )
        result = validate_aio_output(output, anima_plan(), loader=accepting_loader([]))
        self.assertFalse(result.validated)
        self.assertTrue(any("Qwen3 0.6B" in e for e in result.errors))

    def test_an_unusual_namespace_is_honoured_when_the_engine_declares_it(self):
        """A VAE's slot namespace *is* the architecture's, since both come from
        `vae_key_prefix`. A text encoder's nests inside it. An engine declaring
        unfamiliar prefixes must validate without an edit here."""
        plan = anima_plan(
            text_namespace="conditioner.embedders.",
            vae_namespace="ae.",
            components=(
                component("qwen3_06b"),
                component("vae", prefixes=("ae.",)),
            ),
        )
        output = self.write(
            "custom.safetensors",
            aio_tensors(text_ns="conditioner.embedders.", vae_ns="ae."),
        )
        self.assertTrue(
            validate_aio_output(output, plan, loader=accepting_loader([])).validated
        )

    def test_the_reload_is_not_attempted_when_the_header_already_fails(self):
        """A file missing a whole namespace cannot reload; trying anyway just
        buys a confusing exception on top of a clear one."""
        calls = []
        output = self.write(
            "broken.safetensors",
            {"model.diffusion_model.blocks.0.weight": ("BF16", [4, 4])},
        )
        validate_aio_output(output, anima_plan(), loader=accepting_loader(calls))
        self.assertEqual([], calls)


class DtypeTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def write(self, name, tensors):
        return write_safetensors_header(os.path.join(self.dir.name, name), tensors)

    def test_an_explicit_format_is_checked_against_the_final_header(self):
        output = self.write("out.safetensors", aio_tensors(dtype="BF16"))
        plan = anima_plan(
            components=(
                component("qwen3_06b", output_format="fp16"),
                component("vae", prefixes=("vae.",)),
            )
        )
        result = validate_aio_output(output, plan, loader=accepting_loader([]))
        self.assertFalse(result.validated)
        self.assertTrue(any("fp16" in e for e in result.errors))

    def test_an_honoured_explicit_format_validates(self):
        output = self.write("out.safetensors", aio_tensors(dtype="F16"))
        plan = anima_plan(
            components=(
                component("qwen3_06b", output_format="fp16"),
                component("vae", output_format="fp16", prefixes=("vae.",)),
            )
        )
        self.assertTrue(
            validate_aio_output(output, plan, loader=accepting_loader([])).validated
        )

    def test_same_does_not_demand_a_dtype_it_was_never_told(self):
        """`same` copies whatever the source had; without source evidence in
        hand there is nothing to check, and inventing a rule would fail valid
        outputs."""
        output = self.write("out.safetensors", aio_tensors(dtype="F32"))
        self.assertTrue(
            validate_aio_output(output, anima_plan(), loader=accepting_loader([])).validated
        )

    def test_a_scaled_components_scales_must_survive_intact(self):
        tensors = {
            "model.diffusion_model.blocks.0.weight": ("BF16", [4, 4]),
            "text_encoders.qwen3_06b.transformer.a.weight": ("F8_E4M3", [8, 4]),
            "text_encoders.qwen3_06b.transformer.a.weight_scale": ("F32", [8, 1]),
            "vae.decoder.conv1.weight": ("BF16", [4, 4]),
        }
        output = self.write("scaled.safetensors", tensors)
        plan = anima_plan(
            components=(
                component("qwen3_06b", precision="fp8_scaled"),
                component("vae", prefixes=("vae.",)),
            )
        )
        self.assertTrue(
            validate_aio_output(output, plan, loader=accepting_loader([])).validated
        )

    def test_a_scaled_component_that_lost_its_scales_is_refused(self):
        tensors = {
            "model.diffusion_model.blocks.0.weight": ("BF16", [4, 4]),
            "text_encoders.qwen3_06b.transformer.a.weight": ("F16", [8, 4]),
            "vae.decoder.conv1.weight": ("BF16", [4, 4]),
        }
        output = self.write("dequantised.safetensors", tensors)
        plan = anima_plan(
            components=(
                component("qwen3_06b", precision="fp8_scaled"),
                component("vae", prefixes=("vae.",)),
            )
        )
        result = validate_aio_output(output, plan, loader=accepting_loader([]))
        self.assertFalse(result.validated)
        self.assertTrue(any("scale" in e.lower() for e in result.errors))


class ErrorShapeTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_errors_are_structured_not_reduced_to_a_boolean(self):
        output = write_safetensors_header(
            os.path.join(self.dir.name, "broken.safetensors"),
            {"model.diffusion_model.blocks.0.weight": ("BF16", [4, 4])},
        )
        result = validate_aio_output(output, anima_plan(), loader=accepting_loader([]))
        self.assertGreaterEqual(len(result.errors), 2)
        self.assertTrue(all(isinstance(e, str) and e for e in result.errors))

    def test_an_unreadable_output_fails_rather_than_raises(self):
        result = validate_aio_output(
            os.path.join(self.dir.name, "absent.safetensors"),
            anima_plan(),
            loader=accepting_loader([]),
        )
        self.assertFalse(result.validated)
        self.assertTrue(result.errors)

    def test_the_result_is_immutable(self):
        result = OutputValidation(validated=True)
        with self.assertRaises(Exception):
            result.validated = False


class NonModularOutputsAreNotJudgedTests(unittest.TestCase):
    def test_no_plan_means_nothing_to_validate(self):
        """Traditional and UNet-only outputs keep their existing completion
        semantics; this check belongs to the AIO path alone."""
        result = validate_aio_output("whatever.safetensors", None, loader=accepting_loader([]))
        self.assertTrue(result.validated)
        self.assertEqual((), result.checked_slots)


if __name__ == "__main__":
    unittest.main()
