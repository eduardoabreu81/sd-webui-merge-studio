"""Planning a LoRA extraction: what gets subtracted, and what gets refused.

Everything here reads headers only. No tensor is decoded and no model is
loaded, which is what lets the preview appear the moment two checkpoints are
picked rather than after a multi-minute read.
"""

import tempfile
import unittest
from pathlib import Path

from safetensors_helpers import write_safetensors_header


def dit_blocks(count=4, hidden=64, *, prefix="net.", dtype="BF16"):
    """A DiT-shaped diffusion model: linear weights, biases and norms."""
    tensors = {}
    for i in range(count):
        tensors[f"{prefix}blocks.{i}.self_attn.q_proj.weight"] = (dtype, (hidden, hidden))
        tensors[f"{prefix}blocks.{i}.self_attn.q_proj.bias"] = (dtype, (hidden,))
        tensors[f"{prefix}blocks.{i}.mlp.layer1.weight"] = (dtype, (hidden * 4, hidden))
        tensors[f"{prefix}blocks.{i}.norm.weight"] = (dtype, (hidden,))
    return tensors


def unet_blocks(count=2, channels=32, *, dtype="BF16"):
    """A UNet-shaped model, which is the only group carrying conv kernels."""
    prefix = "model.diffusion_model."
    tensors = {"conditioner.embedders.0.transformer.weight": (dtype, (channels, channels))}
    for i in range(count):
        tensors[f"{prefix}input_blocks.{i}.0.in_layers.2.weight"] = (dtype, (channels, channels, 3, 3))
        tensors[f"{prefix}input_blocks.{i}.0.skip_connection.weight"] = (dtype, (channels, channels, 1, 1))
        tensors[f"{prefix}input_blocks.{i}.1.proj_in.weight"] = (dtype, (channels, channels))
    return tensors


class ClassifyShapeTests(unittest.TestCase):
    def test_two_dimensional_weight_is_decomposable(self):
        from lora_extract import KIND_LINEAR, classify_shape

        self.assertEqual(classify_shape((128, 64)), KIND_LINEAR)

    def test_four_dimensional_weight_is_a_conv_kernel(self):
        from lora_extract import KIND_CONV, classify_shape

        self.assertEqual(classify_shape((64, 32, 3, 3)), KIND_CONV)

    def test_one_dimensional_weight_cannot_be_decomposed(self):
        from lora_extract import KIND_VECTOR, classify_shape

        self.assertEqual(classify_shape((128,)), KIND_VECTOR)

    def test_scalar_and_three_dimensional_are_skipped(self):
        from lora_extract import KIND_SKIP, classify_shape

        self.assertEqual(classify_shape(()), KIND_SKIP)
        self.assertEqual(classify_shape((4, 8, 16)), KIND_SKIP)


class EffectiveRankTests(unittest.TestCase):
    def test_rank_is_capped_by_the_smaller_dimension(self):
        from lora_extract import effective_rank

        self.assertEqual(effective_rank((320, 16), rank=64, conv_rank=32), 16)

    def test_rank_passes_through_when_the_matrix_is_large_enough(self):
        from lora_extract import effective_rank

        self.assertEqual(effective_rank((1280, 1280), rank=64, conv_rank=32), 64)

    def test_conv_kernels_use_their_own_rank(self):
        from lora_extract import effective_rank

        # A 3x3 kernel flattens to 64 x (64*3*3), so conv_rank is not capped.
        self.assertEqual(effective_rank((64, 64, 3, 3), rank=64, conv_rank=32), 32)


class FactorElementsTests(unittest.TestCase):
    def test_linear_factors_are_out_by_r_plus_r_by_in(self):
        from lora_extract import factor_elements

        self.assertEqual(factor_elements((128, 64), 8), 128 * 8 + 8 * 64)

    def test_conv_factors_account_for_the_flattened_kernel(self):
        from lora_extract import factor_elements

        # down covers in*kh*kw once flattened.
        self.assertEqual(factor_elements((64, 32, 3, 3), 4), 64 * 4 + 4 * 32 * 3 * 3)


class PlanExtractionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, name, tensors, metadata=None):
        return write_safetensors_header(self.tmp / name, tensors, metadata)

    def test_matching_models_plan_every_decomposable_module(self):
        from lora_extract import plan_extraction

        tensors = dit_blocks(count=4)
        original = self.write("orig.safetensors", tensors)
        tuned = self.write("tuned.safetensors", tensors)

        plan = plan_extraction(original, tuned, rank=8)

        self.assertTrue(plan.ok, plan.refusals)
        # Per block: q_proj.weight and mlp.layer1.weight decompose.
        self.assertEqual(plan.linear_count, 8)
        # Per block: q_proj.bias and norm.weight are stored raw.
        self.assertEqual(plan.vector_count, 8)
        self.assertEqual(plan.conv_count, 0)
        self.assertGreater(plan.estimated_bytes, 0)

    def test_keys_present_in_only_one_model_are_reported_not_extracted(self):
        from lora_extract import plan_extraction

        base = dit_blocks(count=2)
        extra = dict(base)
        extra["net.blocks.9.self_attn.q_proj.weight"] = ("BF16", (64, 64))

        original = self.write("orig.safetensors", base)
        tuned = self.write("tuned.safetensors", extra)

        plan = plan_extraction(original, tuned, rank=8)

        self.assertEqual(plan.only_tuned, 1)
        self.assertEqual(plan.only_original, 0)
        self.assertEqual(plan.linear_count, 4)

    def test_shared_key_with_different_shapes_is_excluded(self):
        from lora_extract import plan_extraction

        base = dit_blocks(count=1)
        widened = dict(base)
        widened["net.blocks.0.mlp.layer1.weight"] = ("BF16", (512, 64))

        original = self.write("orig.safetensors", base)
        tuned = self.write("tuned.safetensors", widened)

        plan = plan_extraction(original, tuned, rank=8)

        self.assertEqual(plan.shape_mismatch, 1)
        self.assertEqual(plan.linear_count, 1)

    def test_conv_kernels_are_planned_for_unet_models(self):
        from lora_extract import plan_extraction

        tensors = unet_blocks(count=2)
        original = self.write("orig.safetensors", tensors)
        tuned = self.write("tuned.safetensors", tensors)

        plan = plan_extraction(original, tuned, rank=8, conv_rank=4)

        self.assertEqual(plan.conv_count, 4)
        self.assertTrue(plan.ok, plan.refusals)

    def test_estimated_size_grows_with_rank(self):
        from lora_extract import plan_extraction

        tensors = dit_blocks(count=4)
        original = self.write("orig.safetensors", tensors)
        tuned = self.write("tuned.safetensors", tensors)

        small = plan_extraction(original, tuned, rank=4)
        large = plan_extraction(original, tuned, rank=32)

        self.assertLess(small.estimated_bytes, large.estimated_bytes)


class RefusalTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, name, tensors, metadata=None):
        return write_safetensors_header(self.tmp / name, tensors, metadata)

    def test_quantized_input_is_refused(self):
        from lora_extract import plan_extraction

        plain = dit_blocks(count=2)
        quantized = dict(plain)
        quantized["net.blocks.0.self_attn.q_proj.weight"] = ("I8", (64, 64))
        quantized["net.blocks.0.self_attn.q_proj.comfy_quant"] = ("U8", (16,))

        original = self.write("orig.safetensors", plain)
        tuned = self.write("tuned.safetensors", quantized)

        plan = plan_extraction(original, tuned, rank=8)

        self.assertFalse(plan.ok)
        self.assertTrue(
            any("quant" in r.lower() for r in plan.refusals),
            plan.refusals,
        )

    def test_different_architectures_are_refused(self):
        from lora_extract import plan_extraction

        anima = dit_blocks(count=2)
        anima["net.llm_adapter.blocks.0.self_attn.q_proj.weight"] = ("BF16", (64, 64))

        original = self.write("anima.safetensors", anima)
        tuned = self.write("sdxl.safetensors", unet_blocks(count=2))

        plan = plan_extraction(original, tuned, rank=8)

        self.assertFalse(plan.ok)
        self.assertTrue(
            any("architect" in r.lower() for r in plan.refusals),
            plan.refusals,
        )

    def test_models_with_nothing_in_common_are_refused(self):
        from lora_extract import plan_extraction

        original = self.write("a.safetensors", dit_blocks(count=2, prefix="net."))
        tuned = self.write("b.safetensors", {"totally.unrelated.weight": ("BF16", (8, 8))})

        plan = plan_extraction(original, tuned, rank=8)

        self.assertFalse(plan.ok)

    def test_identical_file_on_both_sides_is_refused(self):
        from lora_extract import plan_extraction

        path = self.write("same.safetensors", dit_blocks(count=2))

        plan = plan_extraction(path, path, rank=8)

        self.assertFalse(plan.ok)
        self.assertTrue(any("same file" in r.lower() for r in plan.refusals), plan.refusals)


class GroupSummaryTests(unittest.TestCase):
    def test_blocks_collapse_into_one_group_regardless_of_index(self):
        from lora_extract import ModulePlan, group_summary

        modules = [
            ModulePlan(f"net.blocks.{i}.self_attn.q_proj.weight", "linear", (64, 64), 8, 1024)
            for i in range(4)
        ]
        groups = group_summary(modules)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].name, "blocks")
        self.assertEqual(groups[0].count, 4)

    def test_llm_adapter_is_its_own_group(self):
        from lora_extract import ModulePlan, group_summary

        modules = [
            ModulePlan("net.blocks.0.mlp.layer1.weight", "linear", (64, 64), 8, 1024),
            ModulePlan("net.llm_adapter.blocks.0.mlp.0.weight", "linear", (64, 64), 8, 1024),
        ]
        names = [g.name for g in group_summary(modules)]

        self.assertIn("llm_adapter", names)
        self.assertIn("blocks", names)

    def test_groups_are_ordered_by_size(self):
        from lora_extract import ModulePlan, group_summary

        modules = [
            ModulePlan("net.small.0.weight", "linear", (8, 8), 4, 64),
            ModulePlan("net.big.0.weight", "linear", (64, 64), 8, 4096),
        ]
        groups = group_summary(modules)

        self.assertEqual(groups[0].name, "big")


class PreviewHtmlTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_a_workable_plan_shows_what_it_would_produce(self):
        from lora_extract import format_extraction_preview_html, plan_extraction

        tensors = dit_blocks(count=4)
        original = write_safetensors_header(self.tmp / "orig.safetensors", tensors)
        tuned = write_safetensors_header(self.tmp / "tuned.safetensors", tensors)

        html = format_extraction_preview_html(plan_extraction(original, tuned, rank=8))

        self.assertIn("8", html)
        self.assertIn("KB", html.upper().replace("KIB", "KB"))
        self.assertNotIn("Cannot extract", html)

    def test_a_refused_plan_leads_with_the_reason(self):
        from lora_extract import ExtractionPlan, format_extraction_preview_html

        plan = ExtractionPlan(refusals=["Both sides are the same file."])
        html = format_extraction_preview_html(plan)

        self.assertIn("Both sides are the same file.", html)
        self.assertIn("Cannot extract", html)

    def test_hostile_text_from_a_header_is_escaped(self):
        from lora_extract import ExtractionPlan, format_extraction_preview_html

        payload = "<img src=x onerror=alert(1)>"
        plan = ExtractionPlan(
            original_path=payload,
            tuned_path=payload,
            architecture_original=payload,
            refusals=[payload],
        )
        html = format_extraction_preview_html(plan)

        self.assertNotIn("<img src=x", html)
        self.assertIn("&lt;img src=x", html)


class LoraKeyNamingTests(unittest.TestCase):
    """The emitted names must be ones Forge's loader looks for.

    Its LoRA adapter tries `<x>.lora_up.weight` first of seven conventions, and
    `<x>` comes from the loader's own key map, whose universal entry is the
    plain `diffusion_model.<path>` form.
    """

    def test_net_prefix_becomes_diffusion_model(self):
        from lora_extract import lora_key_base

        self.assertEqual(
            lora_key_base("net.blocks.0.self_attn.q_proj.weight"),
            "diffusion_model.blocks.0.self_attn.q_proj",
        )

    def test_full_diffusion_prefix_is_normalised(self):
        from lora_extract import lora_key_base

        self.assertEqual(
            lora_key_base("model.diffusion_model.input_blocks.4.0.weight"),
            "diffusion_model.input_blocks.4.0",
        )

    def test_llm_adapter_keeps_its_path(self):
        from lora_extract import lora_key_base

        # Forge rewrites this to text_encoders.qwen3_06b on load; we emit the
        # natural name and let it do that.
        self.assertEqual(
            lora_key_base("net.llm_adapter.blocks.0.mlp.0.weight"),
            "diffusion_model.llm_adapter.blocks.0.mlp.0",
        )

    def test_bias_resolves_to_the_same_base(self):
        from lora_extract import lora_key_base

        self.assertEqual(
            lora_key_base("net.blocks.0.mlp.layer1.bias"),
            "diffusion_model.blocks.0.mlp.layer1",
        )

    def test_vae_and_text_encoder_are_out_of_scope(self):
        from lora_extract import lora_key_base

        self.assertIsNone(lora_key_base("first_stage_model.decoder.conv_in.weight"))
        self.assertIsNone(lora_key_base("conditioner.embedders.0.transformer.weight"))
        self.assertIsNone(lora_key_base("vae.encoder.conv.weight"))


class SvdFactorsTests(unittest.TestCase):
    """The decomposition itself, checked against a delta of known rank.

    A truncated SVD is *exact* when the target rank is at least the true rank,
    so these assertions have a right answer rather than a tolerance pulled out
    of the air. Run on numpy here; production runs the same code on torch,
    which exposes `linalg.svd`, `diag`, `quantile`, `clip` and `concatenate`
    under the same names.
    """

    def setUp(self):
        import numpy as np

        self.np = np
        self.rng = np.random.default_rng(0)

    def test_a_rank_8_delta_is_recovered_exactly(self):
        from lora_extract import svd_factors

        np = self.np
        b = self.rng.standard_normal((64, 8))
        a = self.rng.standard_normal((8, 32))
        delta = b @ a

        up, down = svd_factors(delta, 8, clamp_quantile=None, xp=np)

        self.assertEqual(up.shape, (64, 8))
        self.assertEqual(down.shape, (8, 32))
        np.testing.assert_allclose(up @ down, delta, atol=1e-9)

    def test_truncating_below_the_true_rank_loses_signal(self):
        from lora_extract import svd_factors

        np = self.np
        delta = self.rng.standard_normal((64, 8)) @ self.rng.standard_normal((8, 32))

        up, down = svd_factors(delta, 2, clamp_quantile=None, xp=np)

        self.assertEqual(up.shape, (64, 2))
        self.assertGreater(np.abs(up @ down - delta).max(), 1e-3)

    def test_rank_cannot_exceed_the_smaller_dimension(self):
        from lora_extract import svd_factors

        np = self.np
        delta = self.rng.standard_normal((64, 8))

        up, down = svd_factors(delta, 32, clamp_quantile=None, xp=np)

        self.assertEqual(up.shape[1], 8)
        self.assertEqual(down.shape[0], 8)

    def test_clamping_bounds_the_factors(self):
        from lora_extract import svd_factors

        np = self.np
        delta = self.rng.standard_normal((32, 32))
        delta[0, 0] = 500.0  # one extreme value

        clamped, _ = svd_factors(delta, 8, clamp_quantile=0.99, xp=np)
        loose, _ = svd_factors(delta, 8, clamp_quantile=None, xp=np)

        self.assertLess(np.abs(clamped).max(), np.abs(loose).max())

    def test_conv_kernels_round_trip_through_the_flattened_form(self):
        from lora_extract import svd_factors

        np = self.np
        kernel = self.rng.standard_normal((16, 8, 3, 3))
        flat = kernel.reshape(16, -1)

        up, down = svd_factors(flat, 16, clamp_quantile=None, xp=np)

        np.testing.assert_allclose(up @ down, flat, atol=1e-9)
        self.assertEqual(down.reshape(16, 8, 3, 3).shape, kernel.shape)


class ExtractionMetadataTests(unittest.TestCase):
    """What we write must be what the inspector already knows how to read.

    `aux_inspector._extraction_info` predates this feature: it recognises a
    LoRA that was subtracted rather than trained, and names the fields it wants.
    Emitting anything else would mean an extracted LoRA opening in our own
    inspector as an ordinary trained one.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _plan(self):
        from lora_extract import plan_extraction

        tensors = dit_blocks(count=2)
        original = write_safetensors_header(self.tmp / "orig.safetensors", tensors)
        tuned = write_safetensors_header(self.tmp / "tuned.safetensors", tensors)
        return plan_extraction(original, tuned, rank=16)

    def test_every_field_is_a_string(self):
        from lora_extract import extraction_metadata

        metadata = extraction_metadata(self._plan())

        for key, value in metadata.items():
            self.assertIsInstance(value, str, f"{key} is not a string")

    def test_the_inspector_recognises_it_as_an_extraction(self):
        from aux_inspector import _extraction_info
        from lora_extract import extraction_metadata

        info = _extraction_info(extraction_metadata(self._plan()))

        self.assertIsNotNone(info)
        self.assertEqual(info["mode"], "svd")
        self.assertEqual(info["rank"], "16")

    def test_the_source_prefixes_are_recorded(self):
        from lora_extract import extraction_metadata

        metadata = extraction_metadata(self._plan())

        self.assertEqual(metadata["base_prefix"], "net.")
        self.assertEqual(metadata["target_prefix"], "net.")


if __name__ == "__main__":
    unittest.main()
