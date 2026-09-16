"""Checkpoint and LoRA headers are attacker-shaped text.

Every dashboard here renders fields copied straight out of a downloaded
.safetensors header, so anything that is not escaped becomes markup in the
Gradio page.
"""

import unittest

from aux_inspector import format_lora_dashboard_html, format_module_dashboard_html
from checkpoint_inspector import format_recipe_dashboard_html

PAYLOAD = "<img src=x onerror=alert(1)>"
ESCAPED = "&lt;img src=x onerror=alert(1)&gt;"


class RecipeDashboardEscapingTests(unittest.TestCase):
    def _info(self) -> dict:
        return {
            "filename": PAYLOAD,
            "components": {"unet": True},
            "architecture": PAYLOAD,
            "precision": PAYLOAD,
            "size_str": "1 MB",
            "total_tensors": 1,
            "recipe": {
                "interp_method": PAYLOAD,
                "primary_model_hash": "aa",
                "baked_loras": [
                    {"name": PAYLOAD, "strength": PAYLOAD, "activation_text": PAYLOAD}
                ],
            },
            "models": {"aa": {"name": PAYLOAD}},
            "comfy_recipe": {
                "base_models": [{"name": PAYLOAD, "hash": PAYLOAD, "node": PAYLOAD}],
                "loras": [{"name": PAYLOAD, "strength": PAYLOAD, "hash": PAYLOAD}],
                "merges": [{"type": PAYLOAD, "ratio": PAYLOAD}],
                "encoders": [PAYLOAD],
                "vaes": [PAYLOAD],
            },
            "raw_metadata": {"note": PAYLOAD},
        }

    def test_no_header_field_reaches_the_page_as_markup(self):
        html = format_recipe_dashboard_html(self._info())

        self.assertNotIn(PAYLOAD, html)
        self.assertIn(ESCAPED, html)

    def test_raw_metadata_dump_cannot_close_its_own_pre_block(self):
        info = self._info()
        info["raw_metadata"] = {"note": "</pre>" + PAYLOAD}

        html = format_recipe_dashboard_html(info)

        self.assertNotIn("</pre><img", html)

    def test_inspection_error_text_is_escaped(self):
        html = format_recipe_dashboard_html({"error": PAYLOAD})

        self.assertNotIn(PAYLOAD, html)
        self.assertIn(ESCAPED, html)


class LoraDashboardEscapingTests(unittest.TestCase):
    def test_trainer_written_lora_fields_are_escaped(self):
        info = {
            "filename": PAYLOAD,
            "size_str": "1 MB",
            "total_tensors": 1,
            "precision": PAYLOAD,
            "key_convention": PAYLOAD,
            "activation_text": PAYLOAD,
            "activation_text_source": PAYLOAD,
            "foreign_layout": PAYLOAD,
            "extraction": {
                "format": PAYLOAD,
                "mode": PAYLOAD,
                "base_prefix": PAYLOAD,
                "target_prefix": PAYLOAD,
                "subtraction_dtype": PAYLOAD,
            },
        }

        html = format_lora_dashboard_html(info)

        self.assertNotIn(PAYLOAD, html)
        self.assertIn(ESCAPED, html)

    def test_module_dashboard_escapes_declared_precision(self):
        html = format_module_dashboard_html(
            {
                "filename": PAYLOAD,
                "size_str": "1 MB",
                "total_tensors": 1,
                "kind": "vae",
                "precision": PAYLOAD,
                "description": "VAE",
            }
        )

        self.assertNotIn(PAYLOAD, html)
        self.assertIn(ESCAPED, html)


if __name__ == "__main__":
    unittest.main()
