"""Three things the Krea 2 merge exposed.

A hidden Model B was still loaded, the parameters that are not a `.weight` were
never blended, and an FP32 bake held the whole model several times over at
once. On a 32 GB machine the last one took Forge down mid-bake.
"""

import ast
import gc
import os
import unittest
import weakref

import torch
from torch import nn

from merge_studio.loose_parameters import merge_loose_parameters
from merge_studio.merge_modes import INTERP_NO_INTERPOLATION, INTERP_WEIGHTED_SUM, merge_mode
from merge_studio.merge_precision import consume_to_cpu_state_dict, discard_patch_backups, live_state_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


class Norm(nn.Module):
    """Stores its value as `scale`, like Krea 2's RMSNorm."""

    def __init__(self, value, size=4):
        super().__init__()
        self.scale = nn.Parameter(torch.full((size,), float(value), dtype=torch.float32))


class Block(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.norm = Norm(value)
        self.linear = nn.Linear(2, 2)
        self.lin = nn.Parameter(torch.full((3,), float(value)))


def named(module):
    return dict(module.named_modules())


class LooseParameterTests(unittest.TestCase):
    def merge(self, a, b, **kwargs):
        arguments = dict(ratio_for=lambda name: 0.5, beta=0.0, interp_method=INTERP_WEIGHTED_SUM)
        arguments.update(kwargs)
        return merge_loose_parameters(named(a), named(b), {}, **arguments)

    def test_a_weighted_sum_reaches_the_midpoint(self):
        a, b = Block(1.0), Block(3.0)
        count, skipped = self.merge(a, b)
        self.assertEqual(2, count)
        self.assertEqual([], skipped)
        self.assertTrue(torch.allclose(a.norm.scale, torch.full((4,), 2.0)))
        self.assertTrue(torch.allclose(a.lin, torch.full((3,), 2.0)))

    def test_weights_and_biases_are_left_to_the_module_walk(self):
        a, b = Block(1.0), Block(3.0)
        before = a.linear.weight.data.clone(), a.linear.bias.data.clone()
        self.merge(a, b)
        self.assertTrue(torch.equal(before[0], a.linear.weight.data))
        self.assertTrue(torch.equal(before[1], a.linear.bias.data))

    def test_the_ratio_is_asked_per_module(self):
        a, b = Block(0.0), Block(1.0)
        self.merge(a, b, ratio_for=lambda name: 1.0 if name == "norm" else 0.0)
        self.assertTrue(torch.allclose(a.norm.scale, torch.ones(4)))
        self.assertTrue(torch.allclose(a.lin, torch.zeros(3)))

    def test_the_dtype_of_a_stays(self):
        a, b = Norm(1.0), Norm(3.0)
        a.scale = nn.Parameter(a.scale.data.to(torch.bfloat16), requires_grad=False)
        self.merge(a, b)
        self.assertEqual(torch.bfloat16, a.scale.dtype)

    def test_a_shape_mismatch_keeps_a_and_is_reported(self):
        a, b = Norm(1.0, size=4), Norm(3.0, size=5)
        count, skipped = self.merge(a, b)
        self.assertEqual(0, count)
        self.assertEqual(["scale"], skipped)
        self.assertTrue(torch.allclose(a.scale, torch.ones(4)))

    def test_a_module_with_no_counterpart_keeps_a_silently(self):
        a, b = Block(1.0), Block(3.0)
        count, skipped = self.merge(a, b, translate_name=lambda name: None)
        self.assertEqual((0, []), (count, skipped))
        self.assertTrue(torch.allclose(a.lin, torch.ones(3)))

    def test_a_skipped_module_is_not_touched(self):
        a, b = Norm(1.0), Norm(3.0)
        count, _ = self.merge(a, b, skip_module=lambda module: True)
        self.assertEqual(0, count)
        self.assertTrue(torch.allclose(a.scale, torch.ones(4)))

    def test_the_merge_calls_it_after_the_weights_and_counts_it(self):
        merge = source("merge_studio", "checkpoint_merge.py")
        tree = merge[merge.index("def _merge_module_tree("):merge.index("def _per_block_spec(")]
        call = tree.index("merge_loose_parameters(")
        self.assertGreater(call, tree.index("for i, name in enumerate(weighted_names):"))
        self.assertIn("merged_count += loose_merged", tree[call:])
        self.assertIn("skipped.extend(loose_skipped)", tree[call:])

    def test_no_interpolation_blends_nothing(self):
        # It keeps A as it is, so the walk returns before this runs.
        merge = source("merge_studio", "checkpoint_merge.py")
        tree = merge[merge.index("def _merge_module_tree("):merge.index("def _per_block_spec(")]
        guard = tree.index("if interp_method != INTERP_NO_INTERPOLATION:")
        self.assertLess(guard, tree.index("merge_loose_parameters("))


class LoadedModelTests(unittest.TestCase):
    def test_the_loaded_checkpoint_goes_before_any_model_loads(self):
        merge = source("merge_studio", "checkpoint_merge.py")
        function = merge[merge.index("def merge_checkpoints("):]
        unload = function.index("sd_models.unload_model_weights()")
        self.assertLess(unload, function.index("if composition.modular:"))
        self.assertLess(unload, function.index("_load_engine(primary_info.filename"))
        # After the checks that can refuse, so a refused merge leaves it loaded.
        self.assertLess(function.index("_guard_unsupported_submodules(p_insp"), unload)


class ForgeGoneNoticeTests(unittest.TestCase):
    def setUp(self):
        self.script = source("javascript", "merge_studio.js")

    def test_every_job_asks_forge_when_its_bar_ends(self):
        at_end = self.script[self.script.index("const atEnd = function () {"):]
        at_end = at_end[: at_end.index("const onProgress")]
        self.assertIn("_mergeStudioForgeIsAnswering()", at_end)
        self.assertIn("MERGE_STUDIO_FORGE_GONE", at_end)

    def test_it_does_not_give_up_on_one_slow_answer(self):
        probe = self.script[self.script.index("async function _mergeStudioForgeIsAnswering"):]
        self.assertIn("attempt < 3", probe[:300])
        self.assertIn('fetch("./internal/ping"', probe[:400])


class UnusedModelTests(unittest.TestCase):
    def test_a_mode_only_loads_the_models_it_reads(self):
        merge = source("merge_studio", "checkpoint_merge.py")
        function = merge[merge.index("def merge_checkpoints("):]
        clear_b = function.index("secondary_name = None")
        clear_c = function.index("tertiary_name = None")
        first_use = function.index("_checkpoint_info(secondary_name)")
        self.assertLess(clear_b, first_use)
        self.assertLess(clear_c, first_use)
        self.assertIn("if not mode.needs_b:", function[:clear_b])
        self.assertIn("if not mode.needs_c:", function[:clear_c])

    def test_no_interpolation_declares_it_reads_only_a(self):
        mode = merge_mode(INTERP_NO_INTERPOLATION)
        self.assertFalse(mode.needs_b)
        self.assertFalse(mode.needs_c)


class Patcher:
    """The part of Forge's ModelPatcher a bake goes through
    (backend/patcher/base.py::patch_weight_to_device)."""

    def __init__(self, weight):
        self.weight = weight
        self.backup = {}

    def patch(self):
        if "w" not in self.backup:
            self.backup["w"] = self.weight
        self.weight = self.weight + 1


class PatchBackupTests(unittest.TestCase):
    def test_forge_keeps_the_old_weight_without_it(self):
        old = torch.zeros(4)
        reference = weakref.ref(old)
        patcher = Patcher(old)
        del old
        patcher.patch()
        gc.collect()
        self.assertIsNotNone(reference())

    def test_with_it_the_old_weight_is_released(self):
        old = torch.zeros(4)
        reference = weakref.ref(old)
        patcher = Patcher(old)
        del old
        discard_patch_backups(patcher)
        patcher.patch()
        gc.collect()
        self.assertIsNone(reference())
        self.assertTrue(torch.equal(patcher.weight, torch.ones(4)))

    def test_it_stays_empty_for_the_rest_of_the_patcher(self):
        patcher = Patcher(torch.zeros(4))
        discard_patch_backups(patcher)
        patcher.patch()
        patcher.patch()
        self.assertNotIn("w", patcher.backup)
        self.assertIsNone(patcher.backup.get("w", None))
        self.assertEqual([], list(patcher.backup.keys()))
        patcher.backup.clear()

    def test_the_merge_discards_them_before_patching(self):
        merge = source("merge_studio", "checkpoint_merge.py")
        block = merge[merge.index('progress_cb("Materializing patched weights'):]
        block = block[: block.index("unet_overrides = {}")]
        self.assertLess(block.index("discard_patch_backups(unet)"), block.index("unet.patch_model("))
        self.assertIn("discard_patch_backups(clip.patcher)", block)


class StateDictCopyTests(unittest.TestCase):
    def test_each_original_is_released_as_it_is_copied(self):
        tensors = {"a": torch.zeros(2), "b": torch.ones(2)}
        references = {k: weakref.ref(v) for k, v in tensors.items()}
        seen = []

        class Watching(dict):
            def pop(self, key):
                seen.append({k: r() is not None for k, r in references.items()})
                return super().pop(key)

        source_dict = Watching(tensors)
        del tensors
        out = consume_to_cpu_state_dict(source_dict)
        gc.collect()
        self.assertEqual({"a", "b"}, set(out))
        self.assertEqual({}, dict(source_dict))
        # When "b" came up, "a" had already gone.
        self.assertFalse(seen[1]["a"])
        self.assertIsNone(references["a"]())
        self.assertIsNone(references["b"]())

    def test_the_copies_own_their_memory_and_keep_their_values(self):
        original = torch.arange(4.0)
        out = consume_to_cpu_state_dict({"a": original})
        self.assertTrue(torch.equal(original, out["a"]))
        self.assertIsNot(original.untyped_storage()._cdata, out["a"].untyped_storage()._cdata)
        self.assertEqual(torch.device("cpu"), out["a"].device)

    def test_the_models_are_dropped_before_the_copy(self):
        # Copying first holds the whole model twice; on an FP32 Krea 2 that
        # alone is about 96 GB.
        merge = source("merge_studio", "checkpoint_merge.py")
        drop = merge.index("engine_a = engine_b = engine_c = None", merge.index("rounded = round_float32_rest_to_bf16"))
        copy = merge.index("sd = to_cpu_contiguous_state_dict(sd)")
        self.assertLess(drop, copy)
        between = merge[drop:copy]
        for name in ("diffusion_a", "unet_sd", "processed_unet", "unet_overrides", "unet"):
            self.assertIn(name, between)

    def test_nothing_after_the_drop_reads_what_was_dropped(self):
        # The result dict reported a baked VAE's size from a name this now
        # clears, which would only have surfaced at the end of a long merge.
        merge = source("merge_studio", "checkpoint_merge.py")
        start = merge.index("sd = to_cpu_contiguous_state_dict(sd)")
        rest = merge[start:merge.index("    finally:", start)]
        for name in ("custom_vae_sd", "processed_unet", "unet_sd", "diffusion_a", "unet_overrides"):
            self.assertNotIn(name, rest)

    def test_the_state_dict_shares_the_model_memory(self):
        model = nn.Linear(4, 4)
        sd = live_state_dict(model)
        self.assertEqual(model.weight.data_ptr(), sd["weight"].data_ptr())
        self.assertEqual({"weight", "bias"}, set(sd))

    def test_the_whole_way_out_never_holds_two_models(self):
        model = nn.Linear(4, 4)
        pointer_gone = weakref.ref(model.weight)
        expected = model.weight.detach().clone()
        sd = live_state_dict(model)
        del model
        gc.collect()
        self.assertIsNone(pointer_gone())
        out = consume_to_cpu_state_dict(sd)
        self.assertEqual({}, sd)
        self.assertTrue(torch.equal(expected, out["weight"]))

    def test_the_merge_no_longer_clones_through_forge(self):
        merge = source("merge_studio", "checkpoint_merge.py")
        self.assertNotIn("get_state_dict_after_quant", merge)
        self.assertEqual(4, merge.count("= live_state_dict("))

    def test_the_shared_function_is_the_consuming_one(self):
        utils = source("merge_studio", "quant_utils.py")
        functions = {n.name: n for n in ast.walk(ast.parse(utils)) if isinstance(n, ast.FunctionDef)}
        body = ast.get_source_segment(utils, functions["to_cpu_contiguous_state_dict"])
        self.assertIn("consume_to_cpu_state_dict(sd)", body)


if __name__ == "__main__":
    unittest.main()
