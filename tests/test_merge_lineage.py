"""The merge history a checkpoint carries, drawn rather than dropped.

novaRealityAM_v10.safetensors records 46 ancestors in `sd_merge_models`, each
with its own recipe, and the dashboard showed two names: the direct parents.
The same file also shows two things the reader has to be told rather than
left to infer:

- every recipe says `block_weights: true` with `alpha_info: "0.0"` and no
  weight values, so the proportions were used but never written down;
- a `pluslora-chattiori` LoRA-bake record sits loose in the models dict (its
  fields are top-level keys, not a hash entry), naming three LoRAs at
  0.4 / 1.0 / 0.4 that appear nowhere else.

The fixture below is trimmed from that file's metadata, hashes shortened.
"""

import unittest

from merge_studio.checkpoint_inspector import format_recipe_dashboard_html
from merge_studio.merge_lineage import build_lineage, render_lineage_html, stray_records


def chattiori(method, a, b, c=None, name="X"):
    return {
        "type": "merge-models-chattiori",
        "primary_model_hash": a,
        "secondary_model_hash": b,
        "tertiary_model_hash": c,
        "merge_method": method,
        "block_weights": True,
        "alpha_info": "0.0",
        "beta_info": None,
        "output_name": name,
    }


TOP = chattiori("Weighted Sum", "micro", "nova30", name="NovaRealityAMV10")

MODELS = {
    "micro": {"name": "REAL_L_MICRO_SKIN_LOCK", "sd_merge_recipe": chattiori("Weighted Sum", "shape", "unnamed")},
    "shape": {"name": "REAL_K_REAL_SHAPE_LOCK", "sd_merge_recipe": chattiori("Add Difference", "cyber", "nova30", "base")},
    "nova30": {"name": "NovaAnimeAMV30", "sd_merge_recipe": chattiori("Weighted Sum", "tempg", "base")},
    "tempg": {"name": "TEMP_G", "sd_merge_recipe": {}},
    "cyber": {"name": "cyberrealisticAnima_v20", "sd_merge_recipe": {}},
    "unnamed": {"name": "unnamedaANIMA_v10", "sd_merge_recipe": {}},
    "base": {"name": "anima_baseV10", "sd_merge_recipe": {}},
    # Verbatim shape: a LoRA-bake recipe's own fields written into the dict.
    "type": "pluslora-chattiori",
    "checkpoint_hash": "48d72ae37d1e",
    "lora_hash": "361b7a7cec0b,1ad743f64fef,b330b46df1c7",
    "alpha_info": "0.4,1.0,0.4",
    "output_name": "TEMP_D",
}


def flatten(node, depth=0):
    yield depth, node
    for child in node.children:
        yield from flatten(child, depth + 1)


class BuildLineageTests(unittest.TestCase):
    def test_it_walks_every_generation(self):
        names = [n.name for _, n in flatten(build_lineage(TOP, MODELS))]

        for expected in ("REAL_K_REAL_SHAPE_LOCK", "cyberrealisticAnima_v20", "TEMP_G"):
            self.assertIn(expected, names)

    def test_each_parent_keeps_its_role(self):
        root = build_lineage(TOP, MODELS)

        self.assertEqual(["A", "B"], [c.role for c in root.children])
        shape = root.children[0].children[0]
        self.assertEqual(["A", "B", "C"], [c.role for c in shape.children])

    def test_a_model_seen_twice_is_expanded_once(self):
        nodes = [n for _, n in flatten(build_lineage(TOP, MODELS)) if n.name == "NovaAnimeAMV30"]

        self.assertEqual(2, len(nodes))
        self.assertEqual(1, sum(1 for n in nodes if n.children))
        self.assertEqual(1, sum(1 for n in nodes if n.repeat))

    def test_a_cycle_ends(self):
        looped = dict(MODELS)
        looped["tempg"] = {"name": "TEMP_G", "sd_merge_recipe": chattiori("Weighted Sum", "micro", "base")}

        names = [n.name for _, n in flatten(build_lineage(TOP, looped))]

        self.assertLess(len(names), 50)

    def test_an_unknown_hash_is_named_by_its_hash(self):
        root = build_lineage(chattiori("Weighted Sum", "micro", "deadbeefcafe"), MODELS)

        self.assertIn("deadbeefcafe", root.children[1].name)


class StrayRecordTests(unittest.TestCase):
    def test_the_loose_lora_bake_is_found(self):
        (record,) = stray_records(MODELS)

        self.assertEqual("TEMP_D", record["output_name"])
        self.assertEqual(["0.4", "1.0", "0.4"], record["strengths"])
        self.assertEqual(3, len(record["loras"]))

    def test_a_clean_dict_has_none(self):
        clean = {k: v for k, v in MODELS.items() if isinstance(v, dict)}

        self.assertEqual([], stray_records(clean))


class RenderTests(unittest.TestCase):
    def test_hostile_names_are_escaped(self):
        hostile = dict(MODELS)
        hostile["cyber"] = {"name": "<script>x</script>", "sd_merge_recipe": {}}

        html = render_lineage_html(TOP, hostile)

        self.assertNotIn("<script>", html)

    def test_nothing_to_draw_draws_nothing(self):
        self.assertEqual("", render_lineage_html(TOP, {}))


def dashboard(recipe=TOP, models=MODELS):
    return format_recipe_dashboard_html(
        {
            "filename": "novaRealityAM_v10.safetensors",
            "components": {"unet": True},
            "embedded_components": (),
            "architecture": "Anima (DiT)",
            "precision": "BF16",
            "size_str": "3.90 GB",
            "block_count": 28,
            "recipe": recipe,
            "models": models,
            "metadata": {},
        }
    )


class DashboardTests(unittest.TestCase):
    def test_the_history_reaches_the_grandparents(self):
        html = dashboard()

        self.assertIn("cyberrealisticAnima_v20", html)
        self.assertIn("Merge history", html)

    def test_the_loose_lora_bake_is_shown(self):
        html = dashboard()

        self.assertIn("TEMP_D", html)
        self.assertIn("0.4, 1.0, 0.4", html)

    def test_unrecorded_block_weights_are_said_out_loud(self):
        html = dashboard()

        self.assertIn("not recorded", html)
        # 0.0 is only the base; reading it as the blend would claim 100% A.
        self.assertNotIn("100% Model A", html)

    def test_recorded_block_weights_get_no_such_warning(self):
        recipe = dict(TOP, alpha_raw="0,L08-L13:cross_attn.q_proj:0.20")

        self.assertNotIn("were not recorded", dashboard(recipe))


if __name__ == "__main__":
    unittest.main()
