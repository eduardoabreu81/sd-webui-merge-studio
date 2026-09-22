import unittest

from merge_studio.checkpoint_inspector import format_recipe_dashboard_html


class CheckpointRecipeProvenanceTests(unittest.TestCase):
    def test_baked_trigger_is_labelled_as_recipe_record_not_file_inference(self):
        info = {
            "filename": "baked.safetensors",
            "components": {"unet": True},
            "architecture": "Anima (DiT)",
            "precision": "BF16",
            "size_str": "1 MB",
            "total_tensors": 1,
            "turbo": {"has_turbo": False},
            "recipe": {
                "type": "CheckpointDoctor-LoRABake",
                "loras": [
                    {
                        "name": "concept.safetensors",
                        "strength": 0.7,
                        "activation_text": "declared-token",
                        "activation_text_source": "modelspec.trigger_phrase",
                    }
                ],
            },
            "raw_metadata": {},
        }

        html = format_recipe_dashboard_html(info)

        self.assertIn("declared trigger: declared-token", html)
        self.assertIn("recorded in embedded merge recipe", html)


class StochasticMergeShowsItsSeedTests(unittest.TestCase):
    """DARE draws its mask at random, so the seed is the difference between a
    recipe that can be run again and one that only looks like it can.

    The merge records it; this is the other half -- reading it back off a
    finished file.
    """

    def dare_info(self, **recipe_extra):
        recipe = {
            "type": "MergeStudio-AnimaMerge",
            "interp_method": "dare",
            "multiplier": 0.5,
        }
        recipe.update(recipe_extra)
        return {
            "filename": "dare.safetensors",
            "components": {"unet": True},
            "architecture": "Anima (DiT)",
            "precision": "BF16",
            "size_str": "1 MB",
            "total_tensors": 1,
            "turbo": {"has_turbo": False},
            "recipe": recipe,
            "raw_metadata": {},
        }

    def test_the_seed_is_shown(self):
        html = format_recipe_dashboard_html(self.dare_info(seed=1234))

        self.assertIn("Seed:", html)
        self.assertIn("1234", html)

    def test_seed_zero_is_a_seed_and_not_a_missing_one(self):
        # It is the default, so it is the value most files will carry.
        html = format_recipe_dashboard_html(self.dare_info(seed=0))

        self.assertIn(">0<", html)
        self.assertNotIn("not recorded", html)

    def test_a_missing_seed_is_called_out(self):
        # An older file, or another tool's DARE. Saying nothing would read as
        # if the merge were deterministic.
        html = format_recipe_dashboard_html(self.dare_info())

        self.assertIn("not recorded", html)
        self.assertIn("cannot be reproduced", html)

    def test_a_deterministic_mode_says_nothing_about_seeds(self):
        info = self.dare_info()
        info["recipe"]["interp_method"] = "weighted_sum"

        self.assertNotIn("Seed:", format_recipe_dashboard_html(info))


if __name__ == "__main__":
    unittest.main()
