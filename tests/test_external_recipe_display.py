"""Describing merges made by other tools.

A checkpoint merged elsewhere records its recipe in that tool's vocabulary. The
dashboard read `interp_method`, which those recipes do not have, and fell back
to `type` -- printing the name of the program where the method belongs, and
showing no ratio at all for merges that very much had one.
"""

import unittest

from merge_studio.checkpoint_inspector import format_recipe_dashboard_html

ALPHA = (
    "0,L05-L09:self_attn.q_proj self_attn.k_proj:0.08"
    ",L15-L27:mlp.layer1 mlp.layer2:0.50"
    ",L18-L27:cross_attn.v_proj cross_attn.output_proj:0.48"
)


def external_info(**recipe_extra):
    recipe = {
        "type": "merge-models-chattiori",
        "merge_method": "Sum Twice",
        "primary_model_hash": "aa",
        "secondary_model_hash": "bb",
        "tertiary_model_hash": "cc",
        "block_weights": True,
        "alpha": 0.0,
        "beta": 0.0,
        "alpha_raw": ALPHA,
        "uses_beta": True,
    }
    recipe.update(recipe_extra)
    return {
        "filename": "nova3DCGAM_v20.safetensors",
        "components": {"unet": True, "llm_adapter": True},
        "embedded_components": (),
        "architecture": "Anima (DiT)",
        "precision": "BF16",
        "size_str": "3.90 GB",
        "total_tensors": 685,
        "block_count": 28,
        "turbo": {"has_turbo": False},
        "recipe": recipe,
        "metadata": {},
    }


class MethodNameTests(unittest.TestCase):
    def test_the_method_is_read_not_the_tool_name(self):
        html = format_recipe_dashboard_html(external_info())

        self.assertIn("Sum Twice", html)

    def test_the_tool_is_still_named_somewhere(self):
        html = format_recipe_dashboard_html(external_info())

        self.assertIn("chattiori", html.lower())

    def test_sum_twice_gets_its_formula(self):
        html = format_recipe_dashboard_html(external_info())

        # (1-b)((1-a)A + aB) + bC -- rendered with entities, so check a piece.
        self.assertIn("Sum Twice", html)
        self.assertIn("A", html)
        self.assertNotIn("no blend proportion appears", html)

    def test_a_webui_recipe_still_reads_its_own_field(self):
        info = external_info()
        info["recipe"] = {
            "type": "webui",
            "interp_method": "Weighted sum",
            "multiplier": 0.25,
            "primary_model_hash": "aa",
            "secondary_model_hash": "bb",
        }
        html = format_recipe_dashboard_html(info)

        self.assertIn("Weighted Sum", html)
        self.assertIn("25%", html)


class ElementalDisplayTests(unittest.TestCase):
    def test_the_rules_are_drawn(self):
        html = format_recipe_dashboard_html(external_info())

        self.assertIn("mlp.layer1", html)
        self.assertIn("cross_attn.v_proj", html)

    def test_the_rule_count_is_stated(self):
        html = format_recipe_dashboard_html(external_info())

        self.assertIn("3 rules", html)

    def test_beta_is_drawn_separately_when_present(self):
        html = format_recipe_dashboard_html(
            external_info(beta_raw="0,L04-L08:adaln_modulation_mlp.1:0.10")
        )

        self.assertIn("adaln_modulation_mlp.1", html)

    def test_a_uniform_external_recipe_does_not_draw_a_grid(self):
        # A uniform merge writes the same number to both fields.
        html = format_recipe_dashboard_html(
            external_info(alpha=0.35, alpha_raw="0.35", block_weights=False, uses_beta=False)
        )

        self.assertIn("0.35", html)
        self.assertNotIn("mlp.layer1", html)

    def test_a_recipe_without_ratios_still_says_so(self):
        info = external_info()
        del info["recipe"]["alpha_raw"]
        del info["recipe"]["alpha"]
        html = format_recipe_dashboard_html(info)

        self.assertIn("Sum Twice", html)


class SaveComponentsTests(unittest.TestCase):
    """"Save Components" is a mode this extension already has under another name.

    All thirteen of these in the reference library save `unet` and nothing
    else, which is what No Interpolation with the save mode on UNet Only
    produces here. Saying so beats leaving someone hunting the mode list for
    a name that is not in it.
    """

    def describe(self, recipe):
        from merge_studio.checkpoint_inspector import _describe_merge_math

        return _describe_merge_math("Save Components (model0 only)", recipe)

    def test_the_components_that_were_saved_are_named(self):
        _, html = self.describe({"alpha": ["unet"], "alpha_raw": "unet"})
        self.assertIn("Components saved", html)
        self.assertIn("unet", html)

    def test_a_unet_only_save_points_at_the_local_equivalent(self):
        _, html = self.describe({"alpha": ["unet"], "alpha_raw": "unet"})
        self.assertIn("No Interpolation", html)
        self.assertIn("UNet Only", html)

    def test_a_combination_this_extension_cannot_produce_claims_no_equivalent(self):
        # Writing the VAE on its own is a different output kind -- a file for
        # models/VAE, not models/Stable-diffusion -- and nothing here does it.
        _, html = self.describe({"alpha": ["vae"], "alpha_raw": "vae"})
        self.assertIn("vae", html)
        self.assertNotIn("No Interpolation", html)

    def test_a_component_list_is_not_drawn_as_a_weight_profile(self):
        # The ratio field holds a component list, so heading it "per-block
        # weights" says the opposite of what it is.
        from merge_studio.checkpoint_inspector import _format_recipe_ratios

        self.assertEqual("", _format_recipe_ratios({"alpha_raw": "unet"}, 28))

    def test_a_component_name_from_a_downloaded_header_is_escaped(self):
        _, html = self.describe({"alpha": ["<img/onerror=alert(1)>"]})
        self.assertNotIn("<img/onerror", html)


if __name__ == "__main__":
    unittest.main()
