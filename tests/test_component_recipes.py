"""Saving an AIO composition, and loading one back honestly.

A recipe records what was chosen. Restoring it on a machine that lacks one of
those files must say so, not quietly pick something else -- a recipe that
silently swaps a component produces a different checkpoint under the same name.

v1 recipes predate component slots entirely. They carry a Bake VAE and nothing
else, so that is all they may restore: inventing a text encoder for them would
put a component in the output that the original merge never had.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from component_recipes import (  # noqa: E402
    RECIPE_VERSION,
    migrate_v1_components,
    restore_component_recipe,
    serialize_component_recipe,
)


def row(slot_id, name, *, source="file", fmt="same"):
    return {"slot_id": slot_id, "name": name, "source": source, "format": fmt}


INSTALLED = {
    "qwen_3_06b_base.safetensors": {"signature_id": "qwen3_06b", "kind": "text_encoder"},
    "qwen_image_vae.safetensors": {"signature_id": "vae", "kind": "vae"},
    "ae.safetensors": {"signature_id": "vae", "kind": "vae"},
}

ANIMA_SLOTS = ("qwen3_06b", "vae")


class VersionTests(unittest.TestCase):
    def test_the_recipe_version_moved_to_two(self):
        self.assertEqual(2, RECIPE_VERSION)


class SerialisationTests(unittest.TestCase):
    def test_a_row_round_trips(self):
        rows = [
            row("qwen3_06b", "qwen_3_06b_base.safetensors"),
            row("vae", "qwen_image_vae.safetensors", fmt="fp16"),
        ]
        stored = serialize_component_recipe(rows)
        restored, warnings = restore_component_recipe(
            {"components": stored}, ANIMA_SLOTS, INSTALLED
        )
        self.assertEqual([], warnings)
        self.assertEqual("qwen_3_06b_base.safetensors", restored["qwen3_06b"]["name"])
        self.assertEqual("fp16", restored["vae"]["format"])

    def test_keeping_the_embedded_component_round_trips(self):
        stored = serialize_component_recipe([row("qwen3_06b", "", source="embedded")])
        restored, _ = restore_component_recipe(
            {"components": stored}, ANIMA_SLOTS, INSTALLED
        )
        self.assertEqual("embedded", restored["qwen3_06b"]["source"])

    def test_an_unfilled_row_is_not_stored(self):
        self.assertEqual([], serialize_component_recipe([row("vae", "")]))

    def test_only_the_basename_is_stored(self):
        stored = serialize_component_recipe(
            [row("vae", "/models/VAE/qwen_image_vae.safetensors")]
        )
        self.assertEqual("qwen_image_vae.safetensors", stored[0]["name"])

    def test_order_is_preserved(self):
        stored = serialize_component_recipe(
            [row("qwen3_06b", "a.safetensors"), row("vae", "b.safetensors")]
        )
        self.assertEqual(["qwen3_06b", "vae"], [s["slot_id"] for s in stored])


class MissingComponentsAreReportedTests(unittest.TestCase):
    """Never silently replaced. A recipe that swaps a component produces a
    different checkpoint under the same name."""

    def test_a_file_that_is_not_installed_leaves_its_slot_empty(self):
        recipe = {"components": [{"slot_id": "vae", "name": "absent.safetensors",
                                  "source": "file", "format": "same"}]}
        restored, warnings = restore_component_recipe(recipe, ANIMA_SLOTS, INSTALLED)
        self.assertNotIn("vae", restored)
        self.assertTrue(any("absent.safetensors" in w for w in warnings))

    def test_a_similar_installed_file_is_not_substituted(self):
        recipe = {"components": [{"slot_id": "vae", "name": "some_other_vae.safetensors",
                                  "source": "file", "format": "same"}]}
        restored, _ = restore_component_recipe(recipe, ANIMA_SLOTS, INSTALLED)
        self.assertEqual({}, restored)

    def test_a_slot_this_architecture_does_not_have_is_reported(self):
        recipe = {"components": [{"slot_id": "t5xxl", "name": "t5.safetensors",
                                  "source": "file", "format": "same"}]}
        restored, warnings = restore_component_recipe(recipe, ANIMA_SLOTS, INSTALLED)
        self.assertEqual({}, restored)
        self.assertTrue(any("t5xxl" in w for w in warnings))

    def test_a_file_that_no_longer_fits_its_slot_is_reported(self):
        recipe = {"components": [{"slot_id": "qwen3_06b", "name": "qwen_image_vae.safetensors",
                                  "source": "file", "format": "same"}]}
        restored, warnings = restore_component_recipe(recipe, ANIMA_SLOTS, INSTALLED)
        self.assertEqual({}, restored)
        self.assertTrue(warnings)

    def test_a_recipe_without_components_restores_nothing_quietly(self):
        restored, warnings = restore_component_recipe({}, ANIMA_SLOTS, INSTALLED)
        self.assertEqual({}, restored)
        self.assertEqual([], warnings)


class V1MigrationTests(unittest.TestCase):
    """v1 recipes carry a Bake VAE and nothing else."""

    def test_a_traditional_bake_vae_is_left_alone(self):
        settings = {"bake_vae": "qwen_image_vae.safetensors"}
        restored, warnings = migrate_v1_components(settings, (), INSTALLED)
        self.assertEqual({}, restored)
        self.assertEqual([], warnings)

    def test_a_bake_vae_maps_into_the_vae_slot_when_the_file_fits(self):
        settings = {"bake_vae": "qwen_image_vae.safetensors"}
        restored, _ = migrate_v1_components(settings, ANIMA_SLOTS, INSTALLED)
        self.assertEqual("qwen_image_vae.safetensors", restored["vae"]["name"])

    def test_no_text_encoder_is_ever_invented(self):
        """The original merge had none, so the migrated one must not either."""
        settings = {"bake_vae": "qwen_image_vae.safetensors"}
        restored, _ = migrate_v1_components(settings, ANIMA_SLOTS, INSTALLED)
        self.assertNotIn("qwen3_06b", restored)

    def test_the_unfilled_encoder_is_reported_as_still_needed(self):
        settings = {"bake_vae": "qwen_image_vae.safetensors"}
        _, warnings = migrate_v1_components(settings, ANIMA_SLOTS, INSTALLED)
        self.assertTrue(any("qwen3_06b" in w or "text encoder" in w.lower() for w in warnings))

    def test_original_and_none_are_not_components(self):
        for value in ("original", "none", "", None):
            with self.subTest(bake_vae=value):
                restored, _ = migrate_v1_components(
                    {"bake_vae": value}, ANIMA_SLOTS, INSTALLED
                )
                self.assertEqual({}, restored)

    def test_a_bake_vae_that_is_gone_is_reported(self):
        settings = {"bake_vae": "vanished.safetensors"}
        restored, warnings = migrate_v1_components(settings, ANIMA_SLOTS, INSTALLED)
        self.assertEqual({}, restored)
        self.assertTrue(any("vanished.safetensors" in w for w in warnings))


class EscapingTests(unittest.TestCase):
    def test_a_hostile_filename_survives_serialisation_as_data(self):
        """Escaping belongs to whatever renders it; the recipe stores the name
        as it is, so a round trip does not corrupt a legitimate one."""
        nasty = "<img src=x onerror=alert(1)>.safetensors"
        stored = serialize_component_recipe([row("vae", nasty)])
        self.assertEqual(nasty, stored[0]["name"])


if __name__ == "__main__":
    unittest.main()
