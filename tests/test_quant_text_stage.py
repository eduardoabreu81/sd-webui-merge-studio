"""The option that also quantizes Krea 2's text stage.

Published INT4 builds of Krea 2 disagree: some convert the 28 main blocks only
(224 layers), others the text-fusion blocks too (256). The default stays the
smaller selection, and the option has to reach the other one exactly, without
touching any layer the official builds keep or any other architecture.
"""

import ast
import os
import unittest

from merge_studio.quant_profiles import FORMAT_GROUPS, offers_text_stage, profile_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINEARS = ("attn.gate", "attn.wk", "attn.wo", "attn.wq", "attn.wv", "mlp.down", "mlp.gate", "mlp.up")
KEPT = ("first", "last.linear", "tmlp.0", "tmlp.2", "tproj.1", "txtmlp.1", "txtmlp.3", "txtfusion.projector")


def krea2_linears():
    """Every linear of a Krea 2 model, as its checkpoint headers list them."""
    blocks = [f"blocks.{i}.{name}" for i in range(28) for name in LINEARS]
    text = [
        f"txtfusion.{stage}.{i}.{name}"
        for stage in ("layerwise_blocks", "refiner_blocks")
        for i in range(2)
        for name in LINEARS
    ]
    return blocks, text


def with_parents(paths):
    out = set()
    for path in paths:
        parts = path.split(".")
        for i in range(1, len(parts) + 1):
            out.add(".".join(parts[:i]))
    return out


def source(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


class Krea2SelectionTests(unittest.TestCase):
    def setUp(self):
        self.blocks, self.text = krea2_linears()
        self.profile = profile_for(with_parents(self.blocks + self.text + list(KEPT)))
        self.assertEqual("krea2", self.profile.name)

    def selected(self, output_format, **kwargs):
        return [p for p in self.blocks + self.text + list(KEPT) if self.profile.selects(p, output_format, **kwargs)]

    def test_int4_converts_224_layers_by_default(self):
        self.assertEqual(self.blocks, self.selected("convrot_w4a4"))
        self.assertEqual(224, len(self.selected("convrot_w4a4")))

    def test_the_option_adds_exactly_the_32_text_stage_layers(self):
        selected = self.selected("convrot_w4a4", text_stage=True)
        self.assertEqual(self.blocks + self.text, selected)
        self.assertEqual(256, len(selected))

    def test_the_layers_every_build_keeps_stay_kept(self):
        for output_format in ("convrot_w4a4", "asym_w4a8_int8", "w6a8_int8", "int8_tensorwise_convrot"):
            selected = self.selected(output_format, text_stage=True)
            for path in KEPT:
                with self.subTest(format=output_format, layer=path):
                    self.assertNotIn(path, selected)

    def test_formats_that_already_convert_it_do_not_change(self):
        for output_format in ("float8_e4m3fn", "float8_e5m2", "mxfp8", "nvfp4"):
            with self.subTest(format=output_format):
                self.assertEqual(
                    self.selected(output_format),
                    self.selected(output_format, text_stage=True),
                )


class OtherArchitecturesTests(unittest.TestCase):
    def test_a_profile_without_a_text_stage_ignores_the_option(self):
        flux3 = profile_for(with_parents([
            "content_mode_blocks.image.0.qkv", "txt_mode_blocks.0.qkv", "single_blocks.0.qkv", "txt_in",
        ]))
        self.assertEqual("flux3", flux3.name)
        for path in ("txt_mode_blocks.0.qkv", "single_blocks.0.qkv", "txt_in"):
            for output_format in FORMAT_GROUPS:
                with self.subTest(layer=path, format=output_format):
                    self.assertEqual(
                        flux3.selects(path, output_format),
                        flux3.selects(path, output_format, text_stage=True),
                    )


class OfferedTests(unittest.TestCase):
    def test_offered_only_where_it_changes_the_result(self):
        for output_format in ("convrot_w4a4", "asym_w4a8_int8", "w6a8_int8", "int8_tensorwise", "int8_tensorwise_convrot"):
            with self.subTest(format=output_format):
                self.assertTrue(offers_text_stage("krea2", output_format))
        for output_format in ("float8_e4m3fn", "float8_e5m2", "mxfp8", "nvfp4", "same", "bf16", "fp16", "fp32"):
            with self.subTest(format=output_format):
                self.assertFalse(offers_text_stage("krea2", output_format))

    def test_not_offered_for_a_model_without_one(self):
        self.assertFalse(offers_text_stage("flux3", "convrot_w4a4"))
        self.assertFalse(offers_text_stage("not_a_profile", "convrot_w4a4"))


class WiringTests(unittest.TestCase):
    def test_each_function_on_the_way_declares_the_parameter(self):
        # A call passing a keyword its target does not declare only fails when
        # someone runs the conversion, and quant_utils needs torch to import.
        declared = {
            ("merge_studio", "quant_utils.py"): ("_fresh_targets", "convert_module_tree_precision"),
            ("merge_studio", "checkpoint_merge.py"): ("_build_metadata", "merge_checkpoints"),
        }
        for path, names in declared.items():
            functions = {
                node.name: node
                for node in ast.walk(ast.parse(source(*path)))
                if isinstance(node, ast.FunctionDef)
            }
            for name in names:
                with self.subTest(function=name):
                    self.assertIn("quantize_text_stage", [a.arg for a in functions[name].args.args])

    def test_the_conversion_hands_the_choice_to_the_selection(self):
        utils = source("merge_studio", "quant_utils.py")
        self.assertIn("_fresh_targets(root_module, output_format, quantize_text_stage=quantize_text_stage)", utils)
        self.assertIn("profile.selects(name, output_format, text_stage=quantize_text_stage)", utils)

    def test_the_merge_forwards_it_and_records_it(self):
        merge = source("merge_studio", "checkpoint_merge.py")
        call = merge[merge.index("_, unet_overrides = convert_module_tree_precision("):]
        self.assertIn("quantize_text_stage=quantize_text_stage", call[:400])
        self.assertIn('merge_recipe["quantize_text_stage"] = True', merge)
        build = merge[merge.index("_build_metadata(\n                primary_info"):]
        self.assertIn("quantize_text_stage=quantize_text_stage", build[:900])

    def test_the_merge_tab_forwards_it(self):
        ui = source("scripts", "merge_studio_ui.py")
        call = ui[ui.index("checkpoint_merge.merge_checkpoints("):]
        self.assertIn("quantize_text_stage=", call[: call.index("\n        )")])

    def test_the_control_sits_right_after_merge_precision_everywhere(self):
        # Gradio wires these by position.
        ui = source("scripts", "merge_studio_ui.py")
        self.assertIn('"anima_extend_rule", "merge_precision", "quantize_text_stage",', ui)
        self.assertIn("anima_extend_rule, merge_precision, merge_text_stage,", ui)
        self.assertIn("merge_precision,\n                        merge_text_stage,\n                        *[c for row in merge_component_rows", ui)
        handler = ui[ui.index("def merge_handler("):]
        self.assertIn("merge_precision_label: str = MERGE_PRECISION_CHOICES[0][0],\n    quantize_text_stage: bool = False,\n    *component_and_lora_args", handler)

    def test_it_starts_off_and_hidden(self):
        ui = source("scripts", "merge_studio_ui.py")
        control = ui[ui.index("merge_text_stage = gr.Checkbox("):]
        control = control[: control.index("\n                )")]
        self.assertIn("value=False", control)
        self.assertIn("visible=False", control)

    def test_the_model_and_the_format_both_drive_it(self):
        ui = source("scripts", "merge_studio_ui.py")
        binding = ui[ui.index("for _control in (merge_primary, merge_format):"):]
        binding = binding[: binding.index("\n\n")]
        self.assertIn("text_stage_visibility", binding)
        self.assertIn("inputs=[merge_primary, merge_format]", binding)
        self.assertIn("outputs=[merge_text_stage]", binding)


if __name__ == "__main__":
    unittest.main()
