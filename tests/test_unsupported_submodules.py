"""A component Forge cannot load makes the merge a lie, so the merge is refused.

Anima-3.8B v1.1 bundles a "Semantic Connector v2". Forge Neo has no support for
it and drops it when the file is opened, before any merge begins -- so the merge
reads a model the file does not describe and saves a result missing the part
that made it different, with nothing raising on the way.

Measured on the two files side by side:

    Anima-3.8B.safetensors       1168 tensors
    Anima-3.8B-v1.1.safetensors  1358 tensors

The 1358 are a strict superset -- there is no key the plain model has that v1.1
lacks -- and all 190 extra ones sit under the single segment
`net.anima_v2_connector`. That is the signal this detects. Not the file name:
a republished finetune keeps the sub-module and loses the name.
"""

import unittest

from merge_studio.checkpoint_inspector import (
    detect_unsupported_submodules,
    unsupported_submodule_message,
)

PLAIN_KEYS = [
    "net.blocks.0.self_attn.q_proj.weight",
    "net.blocks.27.mlp.layer1.weight",
    "net.llm_adapter.blocks.0.self_attn.q_proj.weight",
]

CONNECTOR_KEYS = PLAIN_KEYS + [
    "net.anima_v2_connector.semantic_resampler.blocks.0.source_norm.weight",
    "net.anima_v2_connector.semantic_resampler.blocks.1.time_modulation.weight",
    "net.anima_v2_connector.v2_attentions.0.q_proj.weight",
    "net.anima_v2_connector.quality_anchor.semantic_attentions.0.k_proj.weight",
]

# Republished finetunes write the diffusion model under `model.diffusion_model.`
# instead of `net.`. The sub-module travels with it.
REPUBLISHED_KEYS = [
    "model.diffusion_model.blocks.0.self_attn.q_proj.weight",
    "model.diffusion_model.anima_v2_connector.semantic_resampler.blocks.0.source_norm.weight",
]


class DetectionTests(unittest.TestCase):
    def test_a_plain_checkpoint_carries_nothing_unsupported(self):
        self.assertEqual(detect_unsupported_submodules(PLAIN_KEYS), [])

    def test_the_connector_is_found_and_named(self):
        found = detect_unsupported_submodules(CONNECTOR_KEYS)

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["segment"], "anima_v2_connector")
        self.assertEqual(found[0]["label"], "Semantic Connector v2")

    def test_the_tensor_count_is_reported_so_it_can_be_checked(self):
        self.assertEqual(detect_unsupported_submodules(CONNECTOR_KEYS)[0]["tensors"], 4)

    def test_the_llm_adapter_is_not_mistaken_for_it(self):
        """Every generation carries an llm_adapter, and Forge loads it fine."""
        self.assertEqual(detect_unsupported_submodules(["net.llm_adapter.blocks.0.q.weight"]), [])

    def test_it_is_found_under_the_other_namespace_too(self):
        found = detect_unsupported_submodules(REPUBLISHED_KEYS)

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["segment"], "anima_v2_connector")

    def test_a_name_that_merely_contains_the_segment_is_not_a_match(self):
        """Matched as a whole dotted segment, so a longer name is a different module."""
        self.assertEqual(
            detect_unsupported_submodules(["net.anima_v2_connector_probe.blocks.0.weight"]), []
        )


class RefusalTests(unittest.TestCase):
    """`checkpoint_merge._guard_unsupported_submodules` raises whatever this returns."""

    def inspected(self, keys):
        return {"unsupported_submodules": detect_unsupported_submodules(keys)}

    def test_a_clean_checkpoint_gives_no_reason_to_refuse(self):
        self.assertIsNone(unsupported_submodule_message(self.inspected(PLAIN_KEYS), "Primary Model (A)"))

    def test_a_checkpoint_missing_the_field_gives_no_reason_to_refuse(self):
        """An inspection that failed must not become a refusal to merge."""
        self.assertIsNone(unsupported_submodule_message({}, "Primary Model (A)"))

    def test_the_connector_gives_a_reason(self):
        message = unsupported_submodule_message(self.inspected(CONNECTOR_KEYS), "Primary Model (A)")

        self.assertIsNotNone(message)
        self.assertIn("Primary Model (A)", message)

    def test_the_reason_says_what_was_found_and_what_to_use_instead(self):
        message = unsupported_submodule_message(self.inspected(CONNECTOR_KEYS), "Secondary Model (B)")

        self.assertIn("Semantic Connector v2", message)
        self.assertIn("4 tensors", message)
        self.assertIn("Anima-3.8B", message)

    def test_it_names_the_slot_the_bad_model_is_in(self):
        insp = self.inspected(CONNECTOR_KEYS)

        for label in ("Primary Model (A)", "Secondary Model (B)", "Tertiary Model (C)"):
            self.assertIn(label, unsupported_submodule_message(insp, label))


if __name__ == "__main__":
    unittest.main()
