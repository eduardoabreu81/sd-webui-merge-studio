"""The LoRA dropdowns show a readable name, and the value stays Forge's name.

Forge keys LoRAs by filename stem, so a training checkpoint such as
`xyz_v3-000012` was all the dropdown showed. The label now carries the CivitAI
model and version name (or the trainer's `ss_output_name`) and the architecture,
while the value is still the stem every handler and recipe looks up.
"""

import json
import os
import sys
import tempfile
import unittest

from merge_studio import lora_labels

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safetensors_helpers import write_safetensors_header  # noqa: E402

XL = {"ss_base_model_version": "sdxl_base_v1-0"}


class LoraLabelTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        lora_labels._cache.clear()

    def lora(self, stem, metadata=None, **sidecars):
        path = os.path.join(self.dir, stem + ".safetensors")
        write_safetensors_header(path, {}, metadata)
        suffixes = {"json": ".json", "api_info_json": ".api_info.json", "civitai_info": ".civitai.info"}
        for key, data in sidecars.items():
            with open(os.path.join(self.dir, stem + suffixes[key]), "w", encoding="utf-8") as f:
                json.dump(data, f)
        return path

    def test_civitai_info_names_model_and_version(self):
        path = self.lora("ad-xl", XL, civitai_info={
            "id": 2, "modelId": 1, "name": "v2.0", "model": {"name": "Detail Tweaker XL"},
        })
        self.assertEqual(
            "Detail Tweaker XL · v2.0 · xl  —  ad-xl",
            lora_labels.describe_lora("ad-xl", path).label,
        )

    def test_api_info_model_response_picks_the_recorded_version(self):
        path = self.lora(
            "style", XL,
            json={"modelId": 7, "modelVersionId": 72},
            api_info_json={"id": 7, "name": "Ink Style", "modelVersions": [
                {"id": 71, "name": "v1"}, {"id": 72, "name": "v2"},
            ]},
        )
        self.assertEqual("Ink Style · v2 · xl  —  style", lora_labels.describe_lora("style", path).label)

    def test_sidecar_for_another_model_is_ignored(self):
        # The Browser Neo .json says model 5; an api_info for model 9 belongs to
        # some other file and must not rename this one.
        path = self.lora(
            "char-000012", {"ss_output_name": "mychar", **XL},
            json={"modelId": 5},
            api_info_json={"id": 99, "modelId": 9, "name": "v1", "model": {"name": "Someone Else"}},
        )
        self.assertEqual("mychar · xl  —  char-000012", lora_labels.describe_lora("char-000012", path).label)

    def test_output_name_equal_to_the_stem_adds_nothing(self):
        path = self.lora("plain", {"ss_output_name": "Plain"})
        self.assertEqual("plain", lora_labels.describe_lora("plain", path).label)

    def test_version_already_in_the_name_is_not_repeated(self):
        path = self.lora("x", None, civitai_info={"name": "v3", "model": {"name": "Thing v3"}})
        self.assertEqual("Thing v3  —  x", lora_labels.describe_lora("x", path).label)

    def test_user_sd_version_wins(self):
        path = self.lora("f", XL, json={"sd version": "Flux"})
        self.assertEqual(frozenset({"flux"}), lora_labels.describe_lora("f", path).architectures)

    def test_a_new_sidecar_is_picked_up(self):
        path = self.lora("later", XL)
        self.assertEqual("later · xl", lora_labels.describe_lora("later", path).label)
        with open(os.path.join(self.dir, "later.civitai.info"), "w", encoding="utf-8") as f:
            json.dump({"name": "v1", "model": {"name": "Downloaded"}}, f)
        self.assertEqual("Downloaded · v1 · xl  —  later", lora_labels.describe_lora("later", path).label)

    def test_choices_keep_the_forge_name_as_value(self):
        loras = {
            "zz": self.lora("zz", None, civitai_info={"name": "v1", "model": {"name": "Alpha"}}),
            "aa": self.lora("aa"),
        }
        self.assertEqual(
            [("aa", "aa"), ("Alpha · v1  —  zz", "zz")],
            lora_labels.lora_choices(loras),
        )

    def test_unreadable_file_still_gets_its_name(self):
        path = os.path.join(self.dir, "broken.safetensors")
        with open(path, "wb") as f:
            f.write(b"not a header")
        self.assertEqual("broken", lora_labels.describe_lora("broken", path).label)


class ArchitectureWarningTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        lora_labels._cache.clear()

    def lora(self, stem, sd_version=None):
        path = os.path.join(self.dir, stem + ".safetensors")
        write_safetensors_header(path, {}, XL if sd_version is None else None)
        if sd_version:
            with open(os.path.join(self.dir, stem + ".json"), "w", encoding="utf-8") as f:
                json.dump({"sd version": sd_version}, f)
        return stem, path

    def test_flux_lora_on_an_sdxl_checkpoint_is_named(self):
        loras = [self.lora("xl_one"), self.lora("flux_one", "Flux")]
        mismatched = lora_labels.architecture_mismatches("SDXL", loras)
        self.assertEqual([("flux_one", frozenset({"flux"}))], mismatched)

    def test_unknown_lora_and_unmapped_checkpoint_never_warn(self):
        unknown = os.path.join(self.dir, "unknown.safetensors")
        write_safetensors_header(unknown, {})
        self.assertEqual([], lora_labels.architecture_mismatches("SDXL", [("unknown", unknown)]))
        self.assertEqual([], lora_labels.architecture_mismatches("Iris", [self.lora("flux_two", "Flux")]))
        self.assertEqual([], lora_labels.architecture_mismatches(None, [self.lora("flux_three", "Flux")]))

    def test_disjoint_sources_for_lora_merge(self):
        self.assertEqual([], lora_labels.disjoint_architectures([self.lora("a"), self.lora("b")]))
        split = lora_labels.disjoint_architectures([self.lora("c"), self.lora("d", "Flux")])
        self.assertEqual({"c", "d"}, {name for name, _ in split})


if __name__ == "__main__":
    unittest.main()
