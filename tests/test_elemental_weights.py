"""Reading the elemental weight syntax your existing recipes are written in.

Every merge in this library's ancestry carries an `alpha_raw` like:

    0,L05-L09:self_attn.q_proj self_attn.k_proj:0.08,L15-L27:mlp.layer1:0.50

A base ratio, then rules that override it for a range of layers and a set of
module names. Ten overlapping rules in one merge is normal, which is why this
has to be parsed and shown rather than left as a wall of text.
"""

import unittest


class ParseTests(unittest.TestCase):
    def test_a_bare_number_is_a_base_with_no_rules(self):
        from elemental_weights import parse_weight_spec

        spec = parse_weight_spec("0.5")

        self.assertEqual(spec.base, 0.5)
        self.assertEqual(spec.rules, ())
        self.assertFalse(spec.errors)

    def test_zero_base_is_not_confused_with_absent(self):
        from elemental_weights import parse_weight_spec

        spec = parse_weight_spec("0")

        self.assertEqual(spec.base, 0.0)
        self.assertIsNotNone(spec.base)

    def test_a_rule_carries_its_range_targets_and_weight(self):
        from elemental_weights import parse_weight_spec

        spec = parse_weight_spec("0,L05-L09:self_attn.q_proj self_attn.k_proj:0.08")

        self.assertEqual(len(spec.rules), 1)
        rule = spec.rules[0]
        self.assertEqual((rule.start, rule.end), (5, 9))
        self.assertEqual(rule.targets, ("self_attn.q_proj", "self_attn.k_proj"))
        self.assertEqual(rule.weight, 0.08)

    def test_several_rules_keep_their_written_order(self):
        from elemental_weights import parse_weight_spec

        spec = parse_weight_spec(
            "0,L06-L11:self_attn.v_proj:0.10,L10-L16:self_attn.v_proj:0.22"
        )

        self.assertEqual([r.weight for r in spec.rules], [0.10, 0.22])

    def test_an_integer_weight_parses(self):
        from elemental_weights import parse_weight_spec

        spec = parse_weight_spec("0,L10-L13:self_attn.q_proj:1")

        self.assertEqual(spec.rules[0].weight, 1.0)

    def test_whitespace_around_parts_is_tolerated(self):
        from elemental_weights import parse_weight_spec

        spec = parse_weight_spec(" 0 , L05-L09 : mlp.layer1  mlp.layer2 : 0.20 ")

        self.assertEqual(spec.base, 0.0)
        self.assertEqual(spec.rules[0].targets, ("mlp.layer1", "mlp.layer2"))
        self.assertEqual(spec.rules[0].weight, 0.20)

    def test_a_non_numeric_base_is_kept_as_text(self):
        from elemental_weights import parse_weight_spec

        # "Save Components" writes the component list where a ratio would go.
        spec = parse_weight_spec("unet")

        self.assertIsNone(spec.base)
        self.assertEqual(spec.text_base, "unet")
        self.assertFalse(spec.errors)

    def test_a_malformed_rule_is_reported_not_raised(self):
        from elemental_weights import parse_weight_spec

        spec = parse_weight_spec("0,L05-L09:self_attn.q_proj,L10-L12:mlp.layer1:0.2")

        self.assertEqual(len(spec.rules), 1)
        self.assertTrue(spec.errors)

    def test_empty_input_is_an_empty_spec(self):
        from elemental_weights import parse_weight_spec

        spec = parse_weight_spec("")

        self.assertIsNone(spec.base)
        self.assertEqual(spec.rules, ())


class ResolveTests(unittest.TestCase):
    def test_a_module_no_rule_matches_gets_the_base(self):
        from elemental_weights import parse_weight_spec, resolve_weight

        spec = parse_weight_spec("0.3,L05-L09:mlp.layer1:0.8")

        self.assertEqual(resolve_weight(spec, 2, "self_attn.q_proj"), 0.3)

    def test_a_matching_rule_overrides_the_base(self):
        from elemental_weights import parse_weight_spec, resolve_weight

        spec = parse_weight_spec("0,L05-L09:mlp.layer1 mlp.layer2:0.8")

        self.assertEqual(resolve_weight(spec, 7, "mlp.layer1"), 0.8)

    def test_a_rule_does_not_apply_outside_its_range(self):
        from elemental_weights import parse_weight_spec, resolve_weight

        spec = parse_weight_spec("0,L05-L09:mlp.layer1:0.8")

        self.assertEqual(resolve_weight(spec, 10, "mlp.layer1"), 0.0)

    def test_ranges_are_inclusive_at_both_ends(self):
        from elemental_weights import parse_weight_spec, resolve_weight

        spec = parse_weight_spec("0,L05-L09:mlp.layer1:0.8")

        self.assertEqual(resolve_weight(spec, 5, "mlp.layer1"), 0.8)
        self.assertEqual(resolve_weight(spec, 9, "mlp.layer1"), 0.8)

    def test_the_last_matching_rule_wins(self):
        from elemental_weights import parse_weight_spec, resolve_weight

        # Real overlap, taken from nova3DCGAM_v20: layers 10-11 match both.
        spec = parse_weight_spec(
            "0,L06-L11:self_attn.v_proj:0.10,L10-L16:self_attn.v_proj:0.22"
        )

        self.assertEqual(resolve_weight(spec, 8, "self_attn.v_proj"), 0.10)
        self.assertEqual(resolve_weight(spec, 10, "self_attn.v_proj"), 0.22)

    def test_a_full_state_dict_key_resolves_by_its_module_path(self):
        from elemental_weights import parse_weight_spec, resolve_weight

        spec = parse_weight_spec("0,L15-L27:mlp.layer1:0.5")

        self.assertEqual(
            resolve_weight(spec, 20, "net.blocks.20.mlp.layer1.weight"), 0.5
        )


class OverlapTests(unittest.TestCase):
    def test_overlapping_rules_are_reported(self):
        from elemental_weights import overlaps, parse_weight_spec

        spec = parse_weight_spec(
            "0,L06-L11:self_attn.v_proj:0.10,L10-L16:self_attn.v_proj:0.22"
        )
        found = overlaps(spec)

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].layers, (10, 11))
        self.assertEqual(found[0].target, "self_attn.v_proj")

    def test_rules_on_different_targets_do_not_overlap(self):
        from elemental_weights import overlaps, parse_weight_spec

        spec = parse_weight_spec(
            "0,L06-L11:self_attn.v_proj:0.10,L06-L11:mlp.layer1:0.22"
        )

        self.assertEqual(overlaps(spec), [])

    def test_disjoint_ranges_on_the_same_target_do_not_overlap(self):
        from elemental_weights import overlaps, parse_weight_spec

        spec = parse_weight_spec(
            "0,L08-L14:mlp.layer1:0.18,L15-L27:mlp.layer1:0.50"
        )

        self.assertEqual(overlaps(spec), [])


class ProfileTests(unittest.TestCase):
    def test_the_profile_has_one_row_per_target(self):
        from elemental_weights import parse_weight_spec, weight_profile

        spec = parse_weight_spec(
            "0,L05-L09:mlp.layer1:0.5,L10-L14:self_attn.v_proj:0.3"
        )
        profile = weight_profile(spec, blocks=28)

        self.assertEqual({row.target for row in profile}, {"mlp.layer1", "self_attn.v_proj"})

    def test_each_row_has_one_weight_per_block(self):
        from elemental_weights import parse_weight_spec, weight_profile

        spec = parse_weight_spec("0,L05-L09:mlp.layer1:0.5")
        row = weight_profile(spec, blocks=28)[0]

        self.assertEqual(len(row.weights), 28)
        self.assertEqual(row.weights[4], 0.0)
        self.assertEqual(row.weights[5], 0.5)
        self.assertEqual(row.weights[9], 0.5)
        self.assertEqual(row.weights[10], 0.0)

    def test_a_spec_with_no_rules_profiles_as_the_flat_base(self):
        from elemental_weights import parse_weight_spec, weight_profile

        profile = weight_profile(parse_weight_spec("0.4"), blocks=4)

        self.assertEqual(len(profile), 1)
        self.assertEqual(profile[0].weights, [0.4, 0.4, 0.4, 0.4])


#: The alpha of nova3DCGAM_v20, verbatim from its header. Ten rules, two of
#: which overlap. Kept whole so a parser change that breaks a real recipe
#: fails here rather than in the interface.
REAL_ALPHA = (
    "0,L05-L09:self_attn.q_proj self_attn.k_proj:0.08"
    ",L05-L09:cross_attn.q_proj cross_attn.k_proj:0.06"
    ",L06-L11:self_attn.v_proj self_attn.output_proj:0.10"
    ",L08-L14:mlp.layer1 mlp.layer2:0.18"
    ",L10-L16:self_attn.v_proj self_attn.output_proj:0.22"
    ",L10-L16:cross_attn.v_proj cross_attn.output_proj:0.20"
    ",L15-L27:mlp.layer1 mlp.layer2:0.50"
    ",L16-L27:adaln_modulation_mlp.1 adaln_modulation_mlp.2:0.46"
    ",L17-L27:self_attn.v_proj self_attn.output_proj:0.44"
    ",L18-L27:cross_attn.v_proj cross_attn.output_proj:0.48"
)


class RealRecipeTests(unittest.TestCase):
    def test_every_rule_of_a_real_recipe_parses(self):
        from elemental_weights import parse_weight_spec

        spec = parse_weight_spec(REAL_ALPHA)

        self.assertEqual(spec.base, 0.0)
        self.assertEqual(len(spec.rules), 10)
        self.assertEqual(spec.errors, [])

    def test_the_real_overlap_is_found(self):
        from elemental_weights import overlaps, parse_weight_spec

        found = overlaps(parse_weight_spec(REAL_ALPHA))

        self.assertEqual(
            {(o.target, o.layers) for o in found},
            {("self_attn.v_proj", (10, 11)), ("self_attn.output_proj", (10, 11))},
        )

    def test_the_merge_deepens_towards_the_output(self):
        from elemental_weights import parse_weight_spec, resolve_weight

        spec = parse_weight_spec(REAL_ALPHA)

        # Untouched early, heavily weighted late: the shape of the recipe.
        self.assertEqual(resolve_weight(spec, 1, "mlp.layer1"), 0.0)
        self.assertEqual(resolve_weight(spec, 10, "mlp.layer1"), 0.18)
        self.assertEqual(resolve_weight(spec, 24, "mlp.layer1"), 0.50)


class ProfileHtmlTests(unittest.TestCase):
    def test_the_targets_are_listed(self):
        from elemental_weights import format_weight_profile_html, parse_weight_spec

        html = format_weight_profile_html(parse_weight_spec(REAL_ALPHA), blocks=28)

        self.assertIn("mlp.layer1", html)
        self.assertIn("self_attn.v_proj", html)

    def test_overlaps_are_flagged(self):
        from elemental_weights import format_weight_profile_html, parse_weight_spec

        html = format_weight_profile_html(parse_weight_spec(REAL_ALPHA), blocks=28)

        self.assertIn("overlap", html.lower())

    def test_parse_errors_are_shown(self):
        from elemental_weights import format_weight_profile_html, parse_weight_spec

        html = format_weight_profile_html(parse_weight_spec("0,L05-L09:broken"), blocks=8)

        self.assertIn("Could not read", html)

    def test_a_uniform_spec_says_so_instead_of_drawing_a_grid(self):
        from elemental_weights import format_weight_profile_html, parse_weight_spec

        html = format_weight_profile_html(parse_weight_spec("0.5"), blocks=28)

        self.assertIn("0.5", html)
        self.assertNotIn("<table", html)

    def test_hostile_text_from_a_header_is_escaped(self):
        from elemental_weights import format_weight_profile_html, parse_weight_spec

        # No spaces in the payload: targets are space-separated, so a payload
        # containing one would be split into several and could not execute
        # anyway. This is the shape that actually has to be escaped.
        spec = parse_weight_spec("0,L00-L02:<img/onerror=alert(1)>:0.5")
        html = format_weight_profile_html(spec, blocks=4)

        self.assertNotIn("<img/onerror", html)
        self.assertIn("&lt;img/onerror", html)


class WritingSideTests(unittest.TestCase):
    """What the merge editor needs that reading a finished recipe did not.

    A recipe always leads with its base and puts everything on one line. A
    person typing rules has a Multiplier slider for the base and presses Enter
    between rules, and the same parser has to take both.
    """

    def test_a_text_of_rules_alone_leaves_the_base_to_the_slider(self):
        from elemental_weights import parse_weight_spec, spec_with_base

        spec = parse_weight_spec("L05-L09:mlp.layer1:0.08")
        self.assertIsNone(spec.base)
        self.assertEqual(1, len(spec.rules))
        self.assertEqual(0.5, spec_with_base(spec, 0.5).base)

    def test_a_base_written_into_the_text_beats_the_slider(self):
        # Pasting a ratio straight out of a recipe brings its own base, and
        # that is what the recipe meant.
        from elemental_weights import parse_weight_spec, spec_with_base

        spec = parse_weight_spec("0,L05-L09:mlp.layer1:0.08")
        self.assertEqual(0.0, spec_with_base(spec, 0.5).base)

    def test_a_component_list_is_not_given_a_numeric_base(self):
        # "Save Components" writes a component list where a ratio would go.
        # Dropping a number on it would turn a mode marker into a blend ratio.
        from elemental_weights import parse_weight_spec, spec_with_base

        spec = spec_with_base(parse_weight_spec("save_components"), 0.5)
        self.assertIsNone(spec.base)
        self.assertEqual("save_components", spec.text_base)

    def test_one_rule_per_line_means_the_same_as_one_per_comma(self):
        from elemental_weights import parse_weight_spec

        by_line = parse_weight_spec(
            """L05-L09:mlp.a:0.1
L15-L27:mlp.b:0.5"""
        )
        by_comma = parse_weight_spec("L05-L09:mlp.a:0.1,L15-L27:mlp.b:0.5")
        self.assertEqual(by_comma.rules, by_line.rules)
        self.assertEqual([], by_line.errors)

    def test_the_resolved_weights_follow_the_rule_and_the_base_elsewhere(self):
        from elemental_weights import parse_weight_spec, resolve_weight, spec_with_base

        spec = spec_with_base(parse_weight_spec("L05-L07:mlp.layer1:0.08"), 0.5)
        weights = [
            resolve_weight(spec, block, f"blocks.{block}.mlp.layer1")
            for block in range(10)
        ]
        self.assertEqual([0.5, 0.5, 0.5, 0.5, 0.5, 0.08, 0.08, 0.08, 0.5, 0.5], weights)

    def test_a_module_the_rule_does_not_name_keeps_the_base(self):
        from elemental_weights import parse_weight_spec, resolve_weight, spec_with_base

        spec = spec_with_base(parse_weight_spec("L05-L07:mlp.layer1:0.08"), 0.5)
        self.assertEqual(0.5, resolve_weight(spec, 6, "blocks.6.self_attn.q_proj"))


class ValidationTests(unittest.TestCase):
    """A rule for a layer the model does not have parses fine and never fires.

    That is the failure worth catching: the merge comes out uniform under a
    name that says it was weighted, and nothing on screen said otherwise.
    """

    def test_a_rule_entirely_past_the_end_is_reported(self):
        from elemental_weights import parse_weight_spec, validate_spec

        problems = validate_spec(parse_weight_spec("L40-L44:mlp.a:0.3"), 28)
        self.assertEqual(1, len(problems))
        self.assertIn("28 blocks", problems[0])

    def test_a_rule_that_runs_off_the_end_says_it_still_partly_applies(self):
        from elemental_weights import parse_weight_spec, validate_spec

        problems = validate_spec(parse_weight_spec("L26-L33:mlp.a:0.3"), 28)
        self.assertEqual(1, len(problems))
        self.assertIn("still applies", problems[0])

    def test_the_same_rule_is_fine_on_a_deeper_generation(self):
        # 28 / 40 / 52 are real Anima depths; a rule valid on one is not
        # valid on another, which is why the count comes from the checkpoint.
        from elemental_weights import parse_weight_spec, validate_spec

        self.assertEqual([], validate_spec(parse_weight_spec("L26-L33:mlp.a:0.3"), 40))

    def test_unparseable_rules_are_carried_through(self):
        from elemental_weights import parse_weight_spec, validate_spec

        problems = validate_spec(parse_weight_spec("not a rule at all"), 28)
        self.assertEqual([], problems)  # a non-numeric head is a text base

        problems = validate_spec(parse_weight_spec("0,not a rule at all"), 28)
        self.assertEqual(1, len(problems))
        self.assertIn("Could not read", problems[0])

    def test_without_a_block_count_only_parse_errors_are_reported(self):
        from elemental_weights import parse_weight_spec, validate_spec

        self.assertEqual([], validate_spec(parse_weight_spec("L40-L44:mlp.a:0.3"), None))


class EditorHtmlTests(unittest.TestCase):
    def test_no_model_selected_asks_for_one(self):
        from elemental_weights import format_weight_editor_html, parse_weight_spec

        html = format_weight_editor_html(parse_weight_spec(""), None, [])
        self.assertIn("Primary Model", html)

    def test_no_rules_explains_the_syntax_rather_than_drawing_a_flat_strip(self):
        from elemental_weights import format_weight_editor_html, parse_weight_spec

        html = format_weight_editor_html(parse_weight_spec(""), 28, [])
        self.assertIn("L05-L09", html)

    def test_problems_are_shown_above_the_profile(self):
        from elemental_weights import (
            format_weight_editor_html,
            parse_weight_spec,
            spec_with_base,
            validate_spec,
        )

        spec = spec_with_base(parse_weight_spec("L40-L44:mlp.a:0.3"), 0.5)
        html = format_weight_editor_html(spec, 28, validate_spec(spec, 28))
        self.assertIn("past the end", html)

    def test_a_problem_from_a_downloaded_recipe_is_escaped(self):
        # These strings reach the editor through "import from a checkpoint".
        from elemental_weights import format_weight_editor_html, parse_weight_spec

        html = format_weight_editor_html(
            parse_weight_spec(""), 28, ["<img/onerror=alert(1)>"]
        )
        self.assertNotIn("<img/onerror", html)
        self.assertIn("&lt;img/onerror", html)


class BlockArrayTests(unittest.TestCase):
    """`alpha_weights` has 34 entries and Anima has 28 blocks.

    Measured across all 139 recipes in the reference library on 2026-09-19:
    every array is exactly 34 long, and an Anima 28-block checkpoint carries
    28 DiT blocks plus 6 `llm_adapter.blocks`. 28 + 6 = 34. The correlation is
    perfect but rests on one combination -- every one of the 139 came from a
    28-block merge, and the library's 40-block checkpoints were made in
    ComfyUI and carry no recipe at all.

    The practical half is settled regardless: all 139 are constant (135
    all-zero, 4 all-one), so the array never held a profile in the first
    place. The variation lives in the elemental rules.
    """

    def test_the_tail_past_the_diffusion_blocks_is_split_off(self):
        from elemental_weights import split_block_array

        main, tail = split_block_array([0.0] * 34, 28)
        self.assertEqual(28, len(main))
        self.assertEqual(6, len(tail))

    def test_an_array_no_longer_than_the_stack_has_no_tail(self):
        from elemental_weights import split_block_array

        main, tail = split_block_array([0.0] * 28, 28)
        self.assertEqual(28, len(main))
        self.assertEqual([], tail)

    def test_an_unknown_block_count_leaves_the_array_whole(self):
        from elemental_weights import split_block_array

        main, tail = split_block_array([0.0] * 34, None)
        self.assertEqual(34, len(main))
        self.assertEqual([], tail)

    def test_the_arrays_the_library_actually_contains_are_constant(self):
        from elemental_weights import is_constant

        self.assertTrue(is_constant([0.0] * 34))
        self.assertTrue(is_constant([1.0] * 34))
        self.assertTrue(is_constant([]))
        self.assertTrue(is_constant(None))
        self.assertFalse(is_constant([0.0] * 33 + [1.0]))

    def test_a_constant_array_is_not_drawn_at_all(self):
        # 34 identical cells claiming to be information. The base value is
        # already on the formula line above.
        from elemental_weights import format_block_array_html

        self.assertEqual("", format_block_array_html([0.0] * 34, 28, "Alpha"))

    def test_a_real_profile_is_drawn_with_the_adapter_labelled_apart(self):
        # A reader counting cells against L00-L27 would otherwise find six
        # too many and conclude the drawing is wrong.
        from elemental_weights import format_block_array_html

        html = format_block_array_html([0.2] * 28 + [1.0] * 6, 28, "Alpha")
        self.assertIn("L00-L27", html)
        self.assertIn("LLM adapter", html)
        self.assertIn("28 blocks", html)

    def test_a_label_from_a_downloaded_header_is_escaped(self):
        from elemental_weights import format_block_array_html

        html = format_block_array_html([0.0] * 33 + [1.0], 28, "<img/onerror=x>")
        self.assertNotIn("<img/onerror", html)


if __name__ == "__main__":
    unittest.main()
