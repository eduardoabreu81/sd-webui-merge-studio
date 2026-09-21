"""The merge tab is wired by position, and position is easy to get wrong.

Gradio passes a `click`'s inputs to the handler positionally, and the recipe
saver zips `RECIPE_FIELDS` against a list of controls. Insert a field in one
place and forget the other and nothing raises: a value simply lands in the
wrong parameter. That is how adding a beta slider could quietly make the
multiplier arrive as the block-weight text.

`scripts/merge_studio_ui.py` cannot be imported here -- it pulls in gradio and
the WebUI's `modules` at module scope -- so these read the source. A count is
a weaker check than calling the thing, and it is the one that catches the
mistake this file exists for.
"""

import ast
import os
import unittest

UI_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "merge_studio_ui.py",
)


def ui_source() -> str:
    with open(UI_PATH, encoding="utf-8") as f:
        return f.read()


def bracketed(source: str, opening: str) -> list[str]:
    """The comma-separated entries of a bracketed list, by its opening text."""
    start = source.index(opening) + len(opening)
    depth = 1
    for i in range(start, len(source)):
        if source[i] == "[":
            depth += 1
        elif source[i] == "]":
            depth -= 1
            if depth == 0:
                body = source[start:i]
                break
    else:  # pragma: no cover - a malformed file would fail earlier
        raise AssertionError(f"unbalanced brackets after {opening!r}")
    # Entries are bare names here; a nested call would need real parsing.
    return [part.strip() for part in body.split(",") if part.strip()]


class RecipeFieldsTests(unittest.TestCase):
    def setUp(self):
        self.source = ui_source()

    def recipe_fields(self) -> list[str]:
        for node in ast.walk(ast.parse(self.source)):
            if (
                isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "RECIPE_FIELDS"
            ):
                return [element.value for element in node.value.elts]
        raise AssertionError("RECIPE_FIELDS not found")

    def test_every_field_has_exactly_one_control(self):
        # `save_recipe_handler` zips these two together. A field without a
        # control shifts every field after it by one.
        fields = self.recipe_fields()
        controls = bracketed(self.source, "recipe_scalars = [")
        self.assertEqual(len(fields), len(controls), f"{fields}\n{controls}")

    def test_the_fields_are_all_distinct(self):
        fields = self.recipe_fields()
        self.assertEqual(len(fields), len(set(fields)))

    def test_the_settings_a_merge_needs_are_all_saved(self):
        # A recipe that cannot reproduce the merge it was saved from is worse
        # than no recipe, because it looks like it can.
        for field in ("interp", "multiplier", "beta", "block_weights"):
            self.assertIn(field, self.recipe_fields())


class MergeHandlerInputsTests(unittest.TestCase):
    def setUp(self):
        self.source = ui_source()

    def test_the_click_passes_exactly_what_the_handler_declares(self):
        handler = next(
            node
            for node in ast.walk(ast.parse(self.source))
            if isinstance(node, ast.FunctionDef) and node.name == "merge_handler"
        )
        declared = [a.arg for a in handler.args.args]
        self.assertTrue(handler.args.vararg, "the component/LoRA tail is variadic")

        # The click's inputs up to the first starred entry, which is where the
        # variadic tail begins.
        inputs = bracketed(self.source, "js=\"checkpointDoctorMergeProgress\",\n                    inputs=[")
        positional = [name for name in inputs if not name.startswith("*")]
        self.assertEqual(
            len(declared),
            len(positional),
            f"handler declares {declared}\nclick passes {positional}",
        )

    def test_the_handler_forwards_every_ratio_it_receives(self):
        # Adding a parameter and forgetting to pass it on is silent: the merge
        # runs with the default and the recipe records the default too.
        call = self.source[self.source.index("checkpoint_merge.merge_checkpoints("):]
        call = call[: call.index("\n        )")]
        for argument in (
            "beta=", "block_weights=", "anima_extend_ratio=", "anima_extend_rule=",
        ):
            self.assertIn(argument, call)


class PerBlockEditorIsAnimaOnlyTests(unittest.TestCase):
    def test_the_accordion_starts_hidden(self):
        # It is absent for anything that is not Anima -- not disabled, not
        # greyed out with an explanation. `L00-L27` assumes one numbered stack
        # of blocks, which SDXL's three sections are not.
        source = ui_source()
        accordion = source[source.index('gr.Accordion(\n                    "Per-block weights'):]
        self.assertIn("visible=False", accordion[:400])

    def visibility_handler(self) -> str:
        source = ui_source()
        handler = source[source.index("def block_weights_visibility("):]
        return handler[: handler.index("\n\n\ndef ")]

    def test_visibility_is_decided_by_the_architecture_and_not_a_name(self):
        self.assertIn("_anima_block_count", self.visibility_handler())

    def test_the_mode_also_has_to_be_one_that_blends(self):
        # The rules replace the Multiplier per layer, and No Interpolation has
        # no multiplier to replace -- it copies Model A through. Leaving the
        # editor on screen there means offering rules that do nothing.
        self.assertIn("needs_b", self.visibility_handler())

    def test_both_model_a_and_the_mode_drive_the_accordion(self):
        source = ui_source()
        binding = source[source.index("for _control in (merge_primary, merge_interp):"):]
        binding = binding[: binding.index("\n\n")]
        self.assertIn("block_weights_visibility", binding)
        self.assertIn("inputs=[merge_primary, merge_interp]", binding)


class AnimaExtendRatioIsCrossGenerationOnlyTests(unittest.TestCase):
    """The inserted-block blend applies to one pairing and no other.

    It weights blocks the newer Anima generation inserted. Two models of the
    same generation have none, and a non-Anima pair has no block list at all,
    so on screen with anything else selected it is a control that reads like
    it does something.
    """

    def test_the_slider_starts_hidden(self):
        source = ui_source()
        slider = source[source.index("anima_extend_ratio = gr.Slider("):]
        self.assertIn("visible=False", slider[:400])

    def visibility_handler(self) -> str:
        source = ui_source()
        handler = source[source.index("def anima_extend_ratio_visibility("):]
        return handler[: handler.index("\n\n\ndef ")]

    def test_both_models_are_consulted(self):
        # Cross-generation is a property of the pair. A alone cannot answer it.
        handler = self.visibility_handler()
        self.assertIn("_anima_block_count(primary_name)", handler)
        self.assertIn("_anima_block_count(secondary_name)", handler)

    def test_the_generations_have_to_differ(self):
        # Same block count means plain name matching: no inserted blocks, and
        # nothing for the slider to weight.
        self.assertIn("blocks_a != blocks_b", self.visibility_handler())

    def test_a_mode_that_never_reads_model_b_hides_it(self):
        self.assertIn("needs_b", self.visibility_handler())

    def test_all_three_controls_drive_it(self):
        source = ui_source()
        binding = source[
            source.index("for _control in (merge_primary, merge_secondary, merge_interp):"):
        ]
        binding = binding[: binding.index("\n\n")]
        self.assertIn("anima_extend_ratio_visibility", binding)
        # Both halves of the inserted-block decision, or the radio outlives a
        # pair that has no inserted blocks for it to govern.
        self.assertIn("outputs=[anima_extend_ratio, anima_extend_rule]", binding)

    def test_the_write_rule_starts_hidden_too(self):
        # It travels with the slider: at extend_ratio 0 no inserted block is
        # written, so a rule for writing them governs nothing.
        source = ui_source()
        radio = source[source.index("anima_extend_rule = gr.Radio("):]
        self.assertIn("visible=False", radio[:400])

    def test_the_write_rule_defaults_to_the_old_behaviour(self):
        # Every recipe saved before the rule existed merged with the lerp and
        # carries no rule of its own, so it loads with the radio at whatever
        # sits first in the choices. Reordering them re-interprets those
        # recipes without touching a single saved file.
        source = ui_source()
        choices = source[source.index("ANIMA_EXTEND_RULE_CHOICES = ["):]
        choices = choices[: choices.index("]")]
        self.assertLess(
            choices.index("EXTEND_RULE_BLEND"), choices.index("EXTEND_RULE_DELTA")
        )
        radio = source[source.index("anima_extend_rule = gr.Radio("):]
        self.assertIn("value=ANIMA_EXTEND_RULE_CHOICES[0][0]", radio[:400])

    def test_loading_a_recipe_re_decides_it(self):
        # The models arrive from the backend there, so nothing the user
        # touched fires and the slider would keep the previous pair's answer.
        source = ui_source()
        chain = source[source.index("recipe_load_btn.click("):]
        chain = chain[: chain.index("merge_btn = gr.Button")]
        self.assertIn("anima_extend_ratio_visibility", chain)


class LoraMergeTabTests(unittest.TestCase):
    """The LoRA Merge tab is wired positionally too, and worse: its handlers
    unpack one flat list by index rather than by name.

    `lora_merge_preview_handler(*args)` slices names, weights, rank and dtype
    out of `args` by position. Add a control to the list without moving the
    slice and a weight arrives where a rank was expected -- silently, because
    every one of them is a number.
    """

    def test_the_tab_is_offered(self):
        # Unlike Extract LoRA, which ships hidden behind a switch.
        self.assertIn('with gr.Tab("LoRA Merge"):', ui_source())

    def test_it_is_not_gated_on_a_switch(self):
        source = ui_source()
        tab = source[source.index('with gr.Tab("LoRA Merge"'):]
        self.assertNotIn("visible=", tab[: tab.index(":")])

    def shared_inputs(self) -> str:
        source = ui_source()
        block = source[source.index("_merge_source_inputs = ("):]
        return block[: block.index('\n                )')]

    def test_the_shared_list_is_dropdowns_then_weights_then_the_two_settings(self):
        # The order the handlers slice by. Weights before dropdowns would put
        # a weight where a LoRA name belongs.
        shared = self.shared_inputs()
        self.assertLess(
            shared.index("[dd for dd, _ in lora_merge_rows]"),
            shared.index("[w for _, w in lora_merge_rows]"),
        )
        self.assertLess(
            shared.index("[w for _, w in lora_merge_rows]"),
            shared.index("lora_merge_rank, lora_merge_dtype"),
        )

    def test_both_handlers_slice_by_the_same_constant(self):
        # One of them using a literal 6 is how these drift apart.
        source = ui_source()
        for handler in ("def lora_merge_preview_handler", "def lora_merge_run_handler"):
            body = source[source.index(handler):]
            body = body[: body.index("\n\n\ndef ")]
            self.assertIn("MAX_MERGE_SOURCES", body)

    def test_the_run_handler_gets_device_and_filename_after_the_shared_list(self):
        # The handler reads them at 2*MAX+2 and 2*MAX+3, in that order.
        source = ui_source()
        click = source[source.index("lora_merge_btn.click("):]
        # The first call only clears the panel; the run is in the .then().
        click = click[click.index(".then("):]
        click = click[: click.index("outputs=[lora_merge_result]")]
        self.assertIn("[dummy_component] + _merge_source_inputs", click)
        self.assertIn("[lora_merge_device, lora_merge_filename]", click)

    def test_the_slot_rows_and_the_handlers_agree_on_the_maximum(self):
        self.assertIn("for i in range(1, MAX_MERGE_SOURCES + 1):", ui_source())

    def test_the_module_stays_importable_outside_forge(self):
        # lora_bake imports torch at module scope, which is why it cannot be
        # imported here. lora_merge must not follow it, or its tests stop
        # running in this environment.
        import lora_merge

        self.assertTrue(hasattr(lora_merge, "plan_merge"))
        self.assertTrue(hasattr(lora_merge, "merge_loras"))


class ExtractTabIsOffByDefaultTests(unittest.TestCase):
    """Hidden, not removed.

    Nine minutes on a GPU for the smallest Anima at rank 64, which still
    leaves 41% of the delta behind. The module and its tests stay, because it
    works and because turning it back on should be one line rather than a
    revert.
    """

    def test_the_tab_is_gated_on_a_named_switch(self):
        source = ui_source()
        self.assertIn('gr.Tab("Extract LoRA", visible=SHOW_EXTRACT_TAB)', source)

    def test_the_switch_is_off(self):
        source = ui_source()
        self.assertIn("SHOW_EXTRACT_TAB = False", source)

    def test_the_module_is_still_there(self):
        # Hiding the tab must not quietly take the code with it.
        import lora_extract

        self.assertTrue(hasattr(lora_extract, "plan_extraction"))
        self.assertTrue(hasattr(lora_extract, "extract_lora"))


if __name__ == "__main__":
    unittest.main()
