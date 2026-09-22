"""A Turbo reference in a workflow is not the same as Turbo in the file.

Measured on divingAnima_v70.safetensors, which the inspector badged as
"Turbo Acceleration Detected" while its published description says nothing
about Turbo -- and was right to say nothing.

Its embedded graph does carry `Anima/anima-turbo-lora-v0.2.safetensors`, on a
`LoraLoaderModelOnly` at `mode: 4`, which is ComfyUI's bypass. The saved model
comes from a `ModelSave` fed through that node and one other bypassed LoRA, so
what reached the file is the `ModelMergeSimple` output untouched. ComfyUI proves
it in the metadata itself: the `prompt` field -- the graph that actually ran --
contains neither LoRA node, and `ModelSave` takes its input straight from the
merge.

`_parse_comfy_recipe` reads `prompt`, so it already reported the file correctly.
What badged it was the raw-metadata fallback in `detect_turbo`, which searched
the whole `__metadata__` blob as one string and so read the `workflow` field
too, bypassed nodes included.
"""

import json
import unittest

from merge_studio.checkpoint_inspector import detect_turbo

# The nodes that matter, in ComfyUI's own shape. `mode: 4` is bypass.
WORKFLOW = {
    "nodes": [
        {
            "id": 63,
            "type": "LoraLoaderModelOnly",
            "mode": 4,
            "widgets_values": ["Anima/anima-turbo-lora-v0.2.safetensors", 1],
        },
        {
            "id": 68,
            "type": "ModelMergeSimple",
            "mode": 0,
            "widgets_values": [0.9],
        },
        {"id": 62, "type": "ModelSave", "mode": 0, "widgets_values": ["diffusion_models/Anima"]},
    ]
}

# The executed graph. ComfyUI drops the bypassed nodes when it builds this, so
# 63 is absent and ModelSave takes the merge output directly.
PROMPT = {
    "62": {"inputs": {"filename_prefix": "diffusion_models/Anima", "model": ["68", 0]}, "class_type": "ModelSave"},
    "65": {"inputs": {"unet_name": "Anima/Diving-Anima_V7.safetensors"}, "class_type": "UNETLoader"},
    "68": {"inputs": {"ratio": 0.9, "model1": ["65", 0], "model2": ["69", 0]}, "class_type": "ModelMergeSimple"},
    "69": {
        "inputs": {"unet_name": "Anima/BASBetterAnimeStylePlusAnima_baseV10.safetensors"},
        "class_type": "UNETLoader",
    },
}


def info(**overrides):
    base = {
        "filename": "divingAnima_v70.safetensors",
        "architecture": "Anima (DiT), 28-block",
        "raw_metadata": {
            "workflow": json.dumps(WORKFLOW),
            "prompt": json.dumps(PROMPT),
        },
        "comfy_recipe": {
            "source": "ComfyUI Workflow",
            "base_models": [
                {"name": "Diving-Anima_V7.safetensors", "node": "UNETLoader", "hash": ""},
                {"name": "BASBetterAnimeStylePlusAnima_baseV10.safetensors", "node": "UNETLoader", "hash": ""},
            ],
            "loras": [],
            "merges": [{"type": "ModelMergeSimple", "ratio": 0.9}],
        },
        "recipe": None,
        "models": {},
    }
    base.update(overrides)
    return base


class BypassedNodeTests(unittest.TestCase):
    def test_a_bypassed_turbo_lora_is_not_reported_as_turbo(self):
        self.assertFalse(detect_turbo(info())["has_turbo"])

    def test_the_parsed_graph_wins_over_the_raw_blob(self):
        """The name is in the metadata either way; only the executed graph decides."""
        self.assertIn("anima-turbo", json.dumps(info()["raw_metadata"]).lower())
        self.assertFalse(detect_turbo(info())["has_turbo"])

    def test_an_active_turbo_lora_is_still_reported(self):
        recipe = dict(info()["comfy_recipe"])
        recipe["loras"] = [{"name": "anima-turbo-lora-v0.2.safetensors", "strength": 1.0, "hash": ""}]

        result = detect_turbo(info(comfy_recipe=recipe))

        self.assertTrue(result["has_turbo"])
        self.assertEqual(result["kind"], "Baked Turbo LoRA")

    def test_a_turbo_merge_parent_is_still_reported(self):
        recipe = dict(info()["comfy_recipe"])
        recipe["base_models"] = [{"name": "anima-turbo-v1.0.safetensors", "node": "UNETLoader", "hash": ""}]

        result = detect_turbo(info(comfy_recipe=recipe))

        self.assertTrue(result["has_turbo"])
        self.assertEqual(result["kind"], "Merged from Turbo Checkpoint")

    def test_the_raw_fallback_still_runs_without_a_parsed_recipe(self):
        """A checkpoint with no recipe anything can parse keeps the old behaviour."""
        result = detect_turbo(info(comfy_recipe=None))

        self.assertTrue(result["has_turbo"])
        self.assertEqual(result["kind"], "Turbo Acceleration Detected")


class ZeroStrengthTests(unittest.TestCase):
    def test_a_turbo_lora_at_zero_strength_changed_nothing(self):
        recipe = dict(info()["comfy_recipe"])
        recipe["loras"] = [{"name": "anima-turbo-lora-v0.2.safetensors", "strength": 0.0, "hash": ""}]

        self.assertFalse(detect_turbo(info(comfy_recipe=recipe))["has_turbo"])

    def test_a_webui_baked_turbo_lora_at_zero_strength_changed_nothing(self):
        recipe = {"baked_loras": [{"name": "Turbo-ANIMA-v1.5.safetensors", "strength": 0.0}]}

        self.assertFalse(detect_turbo(info(recipe=recipe, comfy_recipe=None, raw_metadata={}))["has_turbo"])

    def test_an_unreadable_strength_is_not_treated_as_zero(self):
        recipe = {"baked_loras": [{"name": "Turbo-ANIMA-v1.5.safetensors", "strength": "1.0x"}]}

        self.assertTrue(detect_turbo(info(recipe=recipe, comfy_recipe=None, raw_metadata={}))["has_turbo"])


if __name__ == "__main__":
    unittest.main()
