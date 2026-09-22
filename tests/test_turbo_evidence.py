"""A Turbo LoRA carries no evidence of being one except its name.

Measured on the library: Turbo-ANIMA-v1.5.safetensors ships with an empty
__metadata__ (0 keys), while a concept LoRA like ANIMA_Tan_Lines carries 90.
There is no tensor or header signature that separates an acceleration LoRA from
any other, so the badge names the file name as its source -- the same way
detect_turbo already labels filename evidence on the checkpoint side
("Checkpoint filename identifies as Turbo (...)").
"""

import unittest

from merge_studio.aux_inspector import format_lora_dashboard_html


def lora(**overrides):
    info = {
        "filename": "Turbo-ANIMA-v1.5.safetensors",
        "size_str": "143 MB",
        "total_tensors": 912,
        "precision": "BF16",
        "generation": 28,
        "blocks_touched": 28,
        "covers_all_blocks": True,
        "algorithm": "LoRA",
        "key_convention": "diffusion_model",
    }
    info.update(overrides)
    return info


class TurboEvidenceTests(unittest.TestCase):
    def test_badge_names_the_file_name_as_its_source(self):
        html = format_lora_dashboard_html(lora(turbo_in_name=True))

        self.assertIn("Turbo (by name)", html)

    def test_no_unqualified_turbo_claim_is_made(self):
        html = format_lora_dashboard_html(lora(turbo_in_name=True))

        self.assertNotIn(">Turbo<", html)

    def test_a_lora_without_turbo_in_its_name_gets_no_badge(self):
        html = format_lora_dashboard_html(
            lora(filename="ANIMA_Tan_Lines.safetensors", turbo_in_name=False)
        )

        self.assertNotIn("Turbo", html)


if __name__ == "__main__":
    unittest.main()
