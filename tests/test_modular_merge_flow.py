"""How a merge decides to load and combine, before any tensor moves.

`checkpoint_merge` cannot be imported here: it pulls in torch, Forge's backend
and the WebUI's `modules` at module scope, none of which exist outside a
running Forge. So the orchestration decisions were extracted into
`plan_merge_composition`, and those are what these tests pin down.

What that leaves untested is stated plainly rather than implied: the tensor
path itself -- the actual serialisation through `model_config`, and the LLM
Adapter's journey back into the diffusion namespace -- is exercised only by a
real merge inside Forge. See the runtime smoke test in Task 10.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from component_bundle import (  # noqa: E402
    ComponentPlan,
    ComponentSelection,
    ComponentValidationError,
    MergeComposition,
    plan_merge_composition,
)
from component_registry import SupportState  # noqa: E402


def complete_plan(**kw):
    defaults = dict(
        architecture_id="anima",
        support=SupportState.SUPPORTED,
        additional_state_dicts=("qwen.safetensors", "vae.safetensors"),
        slot_order=("qwen3_06b", "vae"),
    )
    defaults.update(kw)
    return ComponentPlan(**defaults)


ANY_SELECTION = [ComponentSelection("qwen3_06b", "file", "qwen.safetensors")]


class TraditionalPathIsUntouchedTests(unittest.TestCase):
    """No selections means nothing changes, including the reliance on the
    global module list. Altering that silently would change outputs nobody
    asked to change."""

    def test_a_full_save_without_selections_still_merges_encoder_and_vae(self):
        composition = plan_merge_composition("full", None)
        self.assertFalse(composition.modular)
        self.assertTrue(composition.merge_text_encoder)
        self.assertTrue(composition.merge_vae)

    def test_the_traditional_path_still_reads_the_global_module_list(self):
        composition = plan_merge_composition("full", None)
        self.assertTrue(composition.use_global_modules)
        self.assertTrue(composition.secondary_uses_global_modules)
        self.assertEqual((), composition.primary_files)

    def test_bake_vae_survives_outside_the_modular_path(self):
        self.assertTrue(plan_merge_composition("full", None).allow_bake_vae)
        self.assertTrue(plan_merge_composition("unet_only", None).allow_bake_vae)

    def test_unet_only_merges_neither_component(self):
        composition = plan_merge_composition("unet_only", None)
        self.assertFalse(composition.merge_text_encoder)
        self.assertFalse(composition.merge_vae)

    def test_an_empty_selection_list_is_not_a_modular_merge(self):
        self.assertFalse(plan_merge_composition("full", []).modular)


class ModularPathTests(unittest.TestCase):
    def setUp(self):
        self.composition = plan_merge_composition("full", ANY_SELECTION, complete_plan())

    def test_only_the_diffusion_model_is_interpolated(self):
        """External components are attached whole, after the merge math.
        Interpolating them across A/B/C is what this path exists to avoid."""
        self.assertFalse(self.composition.merge_text_encoder)
        self.assertFalse(self.composition.merge_vae)

    def test_engine_a_loads_with_exactly_the_plans_files(self):
        self.assertEqual(
            ("qwen.safetensors", "vae.safetensors"), self.composition.primary_files
        )

    def test_engine_a_does_not_inherit_the_global_module_list(self):
        self.assertFalse(self.composition.use_global_modules)

    def test_engines_b_and_c_load_bare(self):
        """They only contribute a diffusion model; inheriting globals there
        would pull components into memory that nothing reads."""
        self.assertFalse(self.composition.secondary_uses_global_modules)

    def test_bake_vae_is_replaced_by_the_vae_slot(self):
        self.assertFalse(self.composition.allow_bake_vae)

    def test_the_plan_travels_with_the_composition(self):
        self.assertEqual("anima", self.composition.plan.architecture_id)

    def test_the_composition_is_immutable(self):
        with self.assertRaises(Exception):
            self.composition.modular = False


class SaveModeConflictTests(unittest.TestCase):
    def test_selections_are_refused_for_a_unet_only_save(self):
        """Silently baking them into a diffusion-only file would produce an
        output that does not match what was asked for."""
        with self.assertRaises(ComponentValidationError) as ctx:
            plan_merge_composition("unet_only", ANY_SELECTION, complete_plan())
        message = str(ctx.exception)
        self.assertIn("unet_only", message)
        self.assertIn("discarded", message)

    def test_the_refusal_covers_every_non_full_mode(self):
        for mode in ("unet_only", "diffusion_only", "anything_else"):
            with self.subTest(save_mode=mode):
                with self.assertRaises(ComponentValidationError):
                    plan_merge_composition(mode, ANY_SELECTION, complete_plan())


class IncompletePlansAreRefusedTests(unittest.TestCase):
    """A merge is expensive. An AIO that cannot be completed is refused before
    any of it starts, not after the file is written."""

    def test_an_unfilled_slot_names_itself(self):
        plan = complete_plan(missing_slots=("vae",))
        with self.assertRaises(ComponentValidationError) as ctx:
            plan_merge_composition("full", ANY_SELECTION, plan)
        self.assertIn("VAE", str(ctx.exception))
        self.assertIn("nothing was selected", str(ctx.exception))

    def test_a_dropped_slot_says_forge_refused_it_not_that_it_is_empty(self):
        plan = complete_plan(dropped_slots=("qwen3_06b",))
        with self.assertRaises(ComponentValidationError) as ctx:
            plan_merge_composition("full", ANY_SELECTION, plan)
        message = str(ctx.exception)
        self.assertIn("Qwen3 0.6B", message)
        self.assertIn("did not accept", message)
        self.assertNotIn("nothing was selected", message)

    def test_both_kinds_are_reported_together(self):
        plan = complete_plan(missing_slots=("vae",), dropped_slots=("qwen3_06b",))
        with self.assertRaises(ComponentValidationError) as ctx:
            plan_merge_composition("full", ANY_SELECTION, plan)
        message = str(ctx.exception)
        self.assertIn("VAE", message)
        self.assertIn("Qwen3 0.6B", message)

    def test_the_refusal_offers_the_way_out(self):
        plan = complete_plan(missing_slots=("vae",))
        with self.assertRaises(ComponentValidationError) as ctx:
            plan_merge_composition("full", ANY_SELECTION, plan)
        self.assertIn("UNet only", str(ctx.exception))

    def test_selections_without_a_plan_are_refused(self):
        with self.assertRaises(ComponentValidationError):
            plan_merge_composition("full", ANY_SELECTION, None)


class AnimaCompositionTests(unittest.TestCase):
    def test_every_generation_composes_the_same_way(self):
        """28, 40 and 52 blocks share one component policy; the block count is
        a diffusion-model detail the composition never sees."""
        for blocks in (28, 40, 52):
            with self.subTest(blocks=blocks):
                composition = plan_merge_composition(
                    "full", ANY_SELECTION, complete_plan()
                )
                self.assertEqual(
                    ("qwen3_06b", "vae"), composition.plan.slot_order
                )

    def test_a_connector_bundle_composes_like_any_other_anima(self):
        composition = plan_merge_composition("full", ANY_SELECTION, complete_plan())
        self.assertEqual(2, len(composition.plan.slot_order))


class ReturnShapeTests(unittest.TestCase):
    def test_the_result_is_a_merge_composition(self):
        self.assertIsInstance(plan_merge_composition("full", None), MergeComposition)

    def test_the_traditional_path_carries_no_plan(self):
        self.assertIsNone(plan_merge_composition("full", None).plan)


if __name__ == "__main__":
    unittest.main()
