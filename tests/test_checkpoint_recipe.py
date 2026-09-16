import unittest

from checkpoint_inspector import format_recipe_dashboard_html


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


if __name__ == "__main__":
    unittest.main()
