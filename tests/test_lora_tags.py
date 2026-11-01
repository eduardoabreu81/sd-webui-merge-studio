"""A pasted prompt fills the LoRA rows: each <lora:name:weight> tag becomes a
row with its weight, matched to the library the way Forge matches it."""

import unittest

from merge_studio import lora_tags

PROMPT = (
    "<lora:MOSS_BASE_DMD_8STEP_QKV:1.0>,<lora:MOSS-Image-Beauty-LoRA-ComfyUI:0.7>,"
    "<lora:MOSS-Image-NSFW-Uncensored-LoRA-ComfyUI:1.0>,<lora:moss_style_v1-comfy:0.7>,"
    "<lora:Moss-RealismEnhancer-V1:0.7>,<lora:phone_photography_2025_moss:0.7>,"
)


class ParseTests(unittest.TestCase):
    def test_splits_every_tag_with_its_weight_in_order(self):
        self.assertEqual(
            lora_tags.parse(PROMPT),
            [
                ("MOSS_BASE_DMD_8STEP_QKV", 1.0),
                ("MOSS-Image-Beauty-LoRA-ComfyUI", 0.7),
                ("MOSS-Image-NSFW-Uncensored-LoRA-ComfyUI", 1.0),
                ("moss_style_v1-comfy", 0.7),
                ("Moss-RealismEnhancer-V1", 0.7),
                ("phone_photography_2025_moss", 0.7),
            ],
        )

    def test_tags_inside_a_full_prompt(self):
        text = "a cat on a sofa, <lora:cat_style:0.6> soft light, <LORA: detail : 1.2 >"
        self.assertEqual(lora_tags.parse(text), [("cat_style", 0.6), ("detail", 1.2)])

    def test_missing_or_odd_weight_defaults_to_one(self):
        self.assertEqual(lora_tags.parse("<lora:a> <lora:b:> <lora:c:abc>"), [("a", 1.0), ("b", 1.0), ("c", 1.0)])

    def test_first_argument_is_the_model_weight(self):
        self.assertEqual(lora_tags.parse("<lora:a:0.8:0.3> <lora:b:unet=0.5> <lyco:c:-0.4>"),
                         [("a", 0.8), ("b", 0.5), ("c", -0.4)])

    def test_incomplete_tag_is_ignored(self):
        self.assertEqual(lora_tags.parse("<lora:still_typing:0."), [])


class ResolveTests(unittest.TestCase):
    NAMES = ["MOSS_BASE_DMD_8STEP_QKV", "moss_style_v1-comfy", "detail-xl"]

    def test_exact_and_case_insensitive(self):
        found, missing = lora_tags.resolve(
            [("MOSS_BASE_DMD_8STEP_QKV", 1.0), ("Moss_Style_V1-Comfy", 0.7)], self.NAMES
        )
        self.assertEqual(found, [("MOSS_BASE_DMD_8STEP_QKV", 1.0), ("moss_style_v1-comfy", 0.7)])
        self.assertEqual(missing, [])

    def test_alias_maps_to_the_file_name(self):
        found, _ = lora_tags.resolve([("Detail Tweaker", 0.5)], self.NAMES, {"Detail Tweaker": "detail-xl"})
        self.assertEqual(found, [("detail-xl", 0.5)])

    def test_folder_and_extension_are_dropped(self):
        found, _ = lora_tags.resolve([("styles\\detail-xl.safetensors", 1.0)], self.NAMES)
        self.assertEqual(found, [("detail-xl", 1.0)])

    def test_unknown_names_are_reported(self):
        found, missing = lora_tags.resolve([("nope", 1.0), ("detail-xl", 0.3)], self.NAMES)
        self.assertEqual(found, [("detail-xl", 0.3)])
        self.assertEqual(missing, ["nope"])


if __name__ == "__main__":
    unittest.main()
