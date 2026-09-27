"""A ModelMergeSimple ratio is model1's share, not model2's.

ComfyUI's node body is

    m = model1.clone()
    m.add_patches(model2_patches, 1.0 - ratio, ratio)   # (patch strength, model strength)

so the output is `ratio * model1 + (1 - ratio) * model2`, and the node's
default ratio of 1.0 returns model1 untouched. realDream_animaV5 carries
`ratio: 0.04` with v3 on model1 and v4.2 on model2: the file is 96% v4.2.
The inspector used to print "Ratio: 0.04" beside two identical "Base Model"
cards, which reads as the opposite.
"""

import json
import unittest

from merge_studio.checkpoint_inspector import _parse_comfy_recipe, format_recipe_dashboard_html

# Verbatim graph from realDream_animaV5.safetensors' `prompt` metadata.
REALDREAM_V5 = {
    "1": {
        "inputs": {"ratio": 0.04, "model1": ["2", 0], "model2": ["3", 0]},
        "class_type": "ModelMergeSimple",
    },
    "2": {"inputs": {"unet_name": "real-dream-v3-anima-bf16.safetensors"}, "class_type": "UNETLoader"},
    "3": {"inputs": {"unet_name": "real-dream-v4.2-anima-bf16.safetensors"}, "class_type": "UNETLoader"},
    "4": {"inputs": {"filename_prefix": "diffusion_models/ComfyUI", "model": ["1", 0]}, "class_type": "ModelSave"},
}


def recipe(prompt):
    return _parse_comfy_recipe({"prompt": json.dumps(prompt)})


class ModelMergeSimpleShares(unittest.TestCase):
    def test_ratio_is_model1s_share(self):
        (merge,) = recipe(REALDREAM_V5)["merges"]

        self.assertEqual(
            [("model1", "real-dream-v3-anima-bf16.safetensors", 0.04),
             ("model2", "real-dream-v4.2-anima-bf16.safetensors", 0.96)],
            [(i["slot"], i["name"], round(i["share"], 6)) for i in merge["inputs"]],
        )

    def test_a_lora_between_loader_and_merge_still_names_the_model(self):
        prompt = dict(REALDREAM_V5)
        prompt["5"] = {
            "inputs": {"lora_name": "x.safetensors", "strength_model": 1.0, "model": ["3", 0]},
            "class_type": "LoraLoaderModelOnly",
        }
        prompt["1"] = {
            "inputs": {"ratio": 0.3, "model1": ["2", 0], "model2": ["5", 0]},
            "class_type": "ModelMergeSimple",
        }
        (merge,) = recipe(prompt)["merges"]

        self.assertEqual("real-dream-v4.2-anima-bf16.safetensors", merge["inputs"][1]["name"])
        self.assertAlmostEqual(0.7, merge["inputs"][1]["share"])

    def test_a_merge_fed_by_another_merge_is_not_named_as_one_model(self):
        prompt = dict(REALDREAM_V5)
        prompt["6"] = {
            "inputs": {"ratio": 0.5, "model1": ["1", 0], "model2": ["2", 0]},
            "class_type": "ModelMergeSimple",
        }
        merges = recipe(prompt)["merges"]
        outer = next(m for m in merges if m["inputs"][0]["name"] is None)

        self.assertEqual(0.5, outer["inputs"][0]["share"])

    def test_other_merge_nodes_get_no_invented_shares(self):
        # ModelMergeAdd / ModelMergeSubtract have no ratio-as-share meaning.
        prompt = {
            "1": {"inputs": {"model1": ["2", 0], "model2": ["3", 0]}, "class_type": "ModelMergeAdd"},
            "2": REALDREAM_V5["2"],
            "3": REALDREAM_V5["3"],
        }
        (merge,) = recipe(prompt)["merges"]

        self.assertTrue(all(i["share"] is None for i in merge["inputs"]))


class DashboardShowsShares(unittest.TestCase):
    def test_the_card_says_who_gets_what(self):
        html = format_recipe_dashboard_html(
            {
                "filename": "realDream_animaV5.safetensors",
                "architecture": "Anima (DiT)",
                "precision": "BF16",
                "size_str": "4 GB",
                "components": {"unet": True},
                "embedded_components": (),
                "comfy_recipe": recipe(REALDREAM_V5),
                "raw_metadata": {},
            }
        )

        self.assertIn("4%", html)
        self.assertIn("96%", html)
        self.assertIn("model2", html)


if __name__ == "__main__":
    unittest.main()
