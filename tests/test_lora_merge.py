"""Merging LoRAs: the algebra that makes it correct, and what gets refused.

The maths half runs on numpy, because the identity being tested is the point
of the module and has nothing to do with torch: concatenating factors gives a
product that is *exactly* the sum of the deltas, where adding the tensors --
which is what a naive merge does -- gives something else entirely.

The planning half reads headers only, so a pair of LoRAs can be described the
moment they are picked.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from merge_studio import lora_merge
from merge_studio.lora_merge import (
    compress_factors,
    concat_factors,
    fold_weight,
    merge_metadata,
    plan_merge,
    split_factor_key,
)
from safetensors_helpers import write_safetensors_header


def lora_tensors(
    modules=("self_attn.q_proj", "mlp.layer1"),
    *,
    blocks=(0, 1),
    rank=4,
    hidden=16,
    prefix="diffusion_model.blocks.",
    dtype="BF16",
    style="kohya",
    extra=None,
):
    """A LoRA-shaped header: a factor pair and an alpha per module."""
    down_name, up_name = (
        ("lora_down.weight", "lora_up.weight") if style == "kohya"
        else ("lora_A.weight", "lora_B.weight")
    )
    tensors = {}
    for block in blocks:
        for module in modules:
            base = f"{prefix}{block}.{module}"
            tensors[f"{base}.{down_name}"] = (dtype, (rank, hidden))
            tensors[f"{base}.{up_name}"] = (dtype, (hidden, rank))
            tensors[f"{base}.alpha"] = ("F32", ())
    if extra:
        tensors.update(extra)
    return tensors


class KeySplittingTests(unittest.TestCase):
    def test_both_spellings_of_a_factor_pair_are_recognised(self):
        # kohya writes lora_down/lora_up, diffusers writes lora_A/lora_B, and
        # Forge loads either. A merge that only knew one would refuse half the
        # library as "no factor pairs".
        self.assertEqual(
            ("blocks.0.q", "down"), split_factor_key("blocks.0.q.lora_down.weight")
        )
        self.assertEqual(
            ("blocks.0.q", "up"), split_factor_key("blocks.0.q.lora_B.weight")
        )

    def test_anything_else_is_not_a_factor(self):
        self.assertIsNone(split_factor_key("blocks.0.q.alpha"))
        self.assertIsNone(split_factor_key("blocks.0.q.weight"))


class ConcatenationIsExactTests(unittest.TestCase):
    """The identity the whole module rests on."""

    def setUp(self):
        rng = np.random.default_rng(11)
        self.up1, self.down1 = rng.normal(size=(12, 3)), rng.normal(size=(3, 8))
        self.up2, self.down2 = rng.normal(size=(12, 5)), rng.normal(size=(5, 8))

    def test_the_product_of_the_stack_is_the_sum_of_the_deltas(self):
        acc = concat_factors(None, self.up1, self.down1, np)
        acc = concat_factors(acc, self.up2, self.down2, np)
        big_u, big_v = acc

        np.testing.assert_allclose(
            big_u @ big_v, self.up1 @ self.down1 + self.up2 @ self.down2
        )

    def test_the_rank_is_the_sum_of_the_ranks(self):
        acc = concat_factors(None, self.up1, self.down1, np)
        acc = concat_factors(acc, self.up2, self.down2, np)
        self.assertEqual((12, 8), (acc[0].shape[0], acc[1].shape[1]))
        self.assertEqual(8, acc[1].shape[0])

    def test_adding_the_tensors_instead_would_be_wrong(self):
        # Not a property of our code -- a guard on the reason it exists. Two
        # rank-3 adapters, added tensor-wise, do not give the sum of their
        # deltas, and nothing about the resulting file says so.
        rng = np.random.default_rng(3)
        up1, down1 = rng.normal(size=(6, 3)), rng.normal(size=(3, 6))
        up2, down2 = rng.normal(size=(6, 3)), rng.normal(size=(3, 6))

        naive = (up1 + up2) @ (down1 + down2)
        correct = up1 @ down1 + up2 @ down2

        self.assertFalse(np.allclose(naive, correct))


class FoldWeightTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.up, self.down = rng.normal(size=(9, 4)), rng.normal(size=(4, 7))

    def test_the_whole_scale_ends_up_in_the_product(self):
        # alpha/rank is the loader's own factor and the weight is the user's.
        # Both have to be inside the factors before concatenation, because
        # afterwards there is one shared alpha for the entire stack.
        up, down = fold_weight(self.up, self.down, 0.5, 2.0, 4, np)

        np.testing.assert_allclose(up @ down, 0.5 * (2.0 / 4) * (self.up @ self.down))

    def test_a_negative_weight_subtracts_the_adapter(self):
        up, down = fold_weight(self.up, self.down, -1.0, 4.0, 4, np)

        np.testing.assert_allclose(up @ down, -(self.up @ self.down))

    def test_the_two_factors_stay_comparable_in_magnitude(self):
        # The square root is split across both sides on purpose: the QR step
        # that follows is stable when they match and lopsided when they do not.
        up, down = fold_weight(self.up, self.down, 100.0, 4.0, 4, np)

        ratio = np.abs(up).mean() / np.abs(down).mean()
        original = np.abs(self.up).mean() / np.abs(self.down).mean()
        self.assertAlmostEqual(ratio, original, places=6)


class CompressionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.up1, self.down1 = rng.normal(size=(20, 4)), rng.normal(size=(4, 16))
        self.up2, self.down2 = rng.normal(size=(20, 4)), rng.normal(size=(4, 16))
        acc = concat_factors(None, self.up1, self.down1, np)
        self.big_u, self.big_v = concat_factors(acc, self.up2, self.down2, np)
        self.target = self.up1 @ self.down1 + self.up2 @ self.down2

    def test_full_rank_compression_changes_nothing(self):
        # Refactorisation, not approximation: at the accumulated rank the
        # product is the same matrix in fewer columns.
        up, down = compress_factors(self.big_u, self.big_v, 8, np)

        self.assertEqual((20, 8), up.shape)
        self.assertEqual((8, 16), down.shape)
        np.testing.assert_allclose(up @ down, self.target, atol=1e-10)

    def test_a_lower_rank_costs_accuracy_and_says_so_in_the_shape(self):
        up, down = compress_factors(self.big_u, self.big_v, 4, np)

        self.assertEqual(4, up.shape[1])
        self.assertFalse(np.allclose(up @ down, self.target))

    def test_more_rank_is_closer(self):
        errors = [
            np.linalg.norm(
                np.subtract(*(lambda p: (p[0] @ p[1], self.target))(
                    compress_factors(self.big_u, self.big_v, r, np)
                ))
            )
            for r in (2, 4, 6)
        ]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])

    def test_it_never_forms_the_dense_product(self):
        # The saving is the whole reason this runs on a CPU: the SVD is over
        # the accumulated rank, not over the layer.
        calls = []
        real_svd = np.linalg.svd

        class Spy:
            linalg = type(
                "L", (), {
                    "qr": staticmethod(np.linalg.qr),
                    "svd": staticmethod(
                        lambda m, **kw: (calls.append(m.shape), real_svd(m, **kw))[1]
                    ),
                }
            )
            concatenate = staticmethod(np.concatenate)

        compress_factors(self.big_u, self.big_v, 4, Spy)

        self.assertEqual([(8, 8)], calls)

    def test_clamping_is_off_by_default(self):
        # The extractor clamps at 0.99 because a checkpoint difference can
        # carry singular directions a trained adapter never would. These
        # inputs are trained adapters; clamping them would alter files the
        # user did not ask to have altered.
        import inspect

        default = inspect.signature(compress_factors).parameters["clamp_quantile"].default
        self.assertIsNone(default)


class PlanMergeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, name, tensors, metadata=None):
        return write_safetensors_header(self.tmp / name, tensors, metadata)

    def two_loras(self, **kwargs):
        return [
            self.write("a.safetensors", lora_tensors(**kwargs)),
            self.write("b.safetensors", lora_tensors(**kwargs)),
        ]

    def test_a_matching_pair_plans_every_module(self):
        plan = plan_merge(self.two_loras(), [1.0, 1.0])

        self.assertTrue(plan.ok)
        self.assertEqual(4, len(plan.modules))
        self.assertEqual(4, plan.shared_modules)
        self.assertEqual(0, plan.exclusive_modules)

    def test_the_accumulated_rank_is_the_sum_of_the_inputs(self):
        plan = plan_merge(self.two_loras(rank=4), [1.0, 1.0])

        self.assertEqual(8, plan.modules[0].accumulated_rank)
        self.assertEqual(8, plan.modules[0].output_rank(0))
        self.assertEqual(6, plan.modules[0].output_rank(6))

    def test_a_target_rank_above_the_layer_is_clipped_to_it(self):
        # A rank above min(out, in) asks for directions the matrix does not
        # have; the SVD would return them as zeros and the file would carry
        # padding that does nothing.
        plan = plan_merge(self.two_loras(rank=4, hidden=6), [1.0, 1.0])

        self.assertEqual(6, plan.modules[0].output_rank(64))

    def test_modules_only_one_file_touches_are_kept(self):
        a = self.write("a.safetensors", lora_tensors(modules=("self_attn.q_proj",)))
        b = self.write("b.safetensors", lora_tensors(modules=("mlp.layer1",)))

        plan = plan_merge([a, b], [1.0, 1.0])

        # The result is the sum of the deltas, and a module only one adapter
        # touches is part of that sum.
        self.assertEqual(4, len(plan.modules))
        self.assertEqual(0, plan.shared_modules)
        self.assertTrue(plan.ok)

    def test_nothing_in_common_is_worth_saying_out_loud(self):
        a = self.write("a.safetensors", lora_tensors(modules=("self_attn.q_proj",)))
        b = self.write("b.safetensors", lora_tensors(modules=("mlp.layer1",)))

        plan = plan_merge([a, b], [1.0, 1.0])

        self.assertTrue(any("no module in common" in w for w in plan.warnings))

    def test_one_file_is_not_a_merge(self):
        plan = plan_merge([self.two_loras()[0]], [1.0])

        self.assertFalse(plan.ok)
        self.assertTrue(any("at least two" in r for r in plan.refusals))

    def test_loha_is_refused_by_name_and_reason(self):
        loha = self.write(
            "loha.safetensors",
            {"diffusion_model.blocks.0.q.hada_w1_a": ("BF16", (4, 16))},
        )
        plan = plan_merge([self.two_loras()[0], loha], [1.0, 1.0])

        self.assertFalse(plan.ok)
        joined = " ".join(plan.refusals)
        self.assertIn("LoHa", joined)
        self.assertIn("Hadamard", joined)

    def test_lokr_is_refused(self):
        lokr = self.write(
            "lokr.safetensors",
            {"diffusion_model.blocks.0.q.lokr_w1": ("BF16", (4, 4))},
        )
        plan = plan_merge([self.two_loras()[0], lokr], [1.0, 1.0])

        self.assertFalse(plan.ok)
        self.assertIn("LoKr", " ".join(plan.refusals))

    def test_dora_is_refused_because_it_scales_what_it_sits_on(self):
        dora = self.write(
            "dora.safetensors",
            lora_tensors(extra={"diffusion_model.blocks.0.q.dora_scale": ("F32", (16,))}),
        )
        plan = plan_merge([self.two_loras()[0], dora], [1.0, 1.0])

        self.assertFalse(plan.ok)
        self.assertIn("DoRA", " ".join(plan.refusals))

    def test_different_anima_generations_are_refused(self):
        small = self.write("small.safetensors", lora_tensors(blocks=(0, 27)))
        large = self.write("large.safetensors", lora_tensors(blocks=(0, 39)))

        plan = plan_merge([small, large], [1.0, 1.0])

        self.assertFalse(plan.ok)
        joined = " ".join(plan.refusals)
        self.assertIn("generation", joined)
        # The checkpoint-side remap must not read as an option here.
        self.assertIn("not for adapters", joined)

    def test_the_same_generation_is_fine(self):
        a = self.write("a.safetensors", lora_tensors(blocks=(0, 27)))
        b = self.write("b.safetensors", lora_tensors(blocks=(5, 20)))

        self.assertTrue(plan_merge([a, b], [1.0, 1.0]).ok)

    def test_a_shape_mismatch_warns_instead_of_writing_nonsense(self):
        a = self.write("a.safetensors", lora_tensors(hidden=16))
        b = self.write("b.safetensors", lora_tensors(hidden=32))

        plan = plan_merge([a, b], [1.0, 1.0])

        self.assertTrue(plan.warnings)
        self.assertTrue(all(len(m.sources) == 1 for m in plan.modules))

    def test_the_estimate_follows_the_rank(self):
        paths = self.two_loras(rank=4, hidden=64)

        small = plan_merge(paths, [1.0, 1.0], target_rank=2).estimated_bytes()
        large = plan_merge(paths, [1.0, 1.0], target_rank=8).estimated_bytes()

        self.assertLess(small, large)

    def test_an_unreadable_file_is_named(self):
        broken = self.tmp / "broken.safetensors"
        broken.write_bytes(b"not a safetensors file")

        plan = plan_merge([self.two_loras()[0], str(broken)], [1.0, 1.0])

        self.assertFalse(plan.ok)
        self.assertIn("broken.safetensors", " ".join(plan.refusals))


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.paths = [
            write_safetensors_header(self.tmp / "a.safetensors", lora_tensors()),
            write_safetensors_header(self.tmp / "b.safetensors", lora_tensors()),
        ]

    def test_a_merge_does_not_claim_to_be_an_extraction(self):
        # `aux_inspector._extraction_info` reads `format: delta-lora` to
        # describe a LoRA subtracted from two checkpoints. A merged adapter
        # borrowing that field would be described as a difference between two
        # models it never saw.
        metadata = merge_metadata(plan_merge(self.paths, [1.0, 1.0]))

        self.assertEqual("merged-lora", metadata["format"])

    def test_the_sources_and_their_weights_are_recorded(self):
        metadata = merge_metadata(plan_merge(self.paths, [1.0, 0.5]))

        self.assertIn("a.safetensors:1", metadata["sources"])
        self.assertIn("b.safetensors:0.5", metadata["sources"])

    def test_exact_and_compressed_runs_are_distinguishable(self):
        exact = merge_metadata(plan_merge(self.paths, [1.0, 1.0], target_rank=0))
        compressed = merge_metadata(plan_merge(self.paths, [1.0, 1.0], target_rank=8))

        self.assertEqual("exact", exact["target_rank"])
        self.assertEqual("concat", exact["mode"])
        self.assertEqual("concat+svd", compressed["mode"])


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_a_refusal_is_shown_rather_than_an_empty_panel(self):
        loha = write_safetensors_header(
            self.tmp / "loha.safetensors",
            {"diffusion_model.blocks.0.q.hada_w1_a": ("BF16", (4, 16))},
        )
        ok = write_safetensors_header(self.tmp / "a.safetensors", lora_tensors())

        html = lora_merge.format_merge_preview_html(plan_merge([ok, loha], [1.0, 1.0]))

        self.assertIn("cannot run", html)
        self.assertIn("LoHa", html)

    def test_a_filename_cannot_inject_markup(self):
        # The panel prints names that came off disk. There is a dedicated test
        # file for this class of bug: tests/test_dashboard_escaping.py.
        #
        # The name is put into the plan rather than onto disk: Windows refuses
        # to create this file, and the panel cannot tell where the string came
        # from. A LoRA downloaded on Linux and synced over is the real path in.
        import dataclasses

        a = write_safetensors_header(self.tmp / "a.safetensors", lora_tensors())
        b = write_safetensors_header(self.tmp / "b.safetensors", lora_tensors())
        plan = plan_merge([a, b], [1.0, 1.0])
        plan = dataclasses.replace(
            plan,
            sources=[
                dataclasses.replace(
                    plan.sources[0], name="<img src=x onerror=alert(1)>.safetensors"
                ),
                plan.sources[1],
            ],
        )

        html = lora_merge.format_merge_preview_html(plan)

        self.assertNotIn("<img", html)
        self.assertIn("&lt;img", html)

    def test_nothing_selected_draws_nothing(self):
        self.assertEqual("", lora_merge.format_merge_preview_html(None))


if __name__ == "__main__":
    unittest.main()
