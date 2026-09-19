"""Deciding what a checkpoint is, and what it actually carries inside.

Two things are kept apart here on purpose:

* the **diffusion architecture**, decided only from diffusion-model evidence;
* the **embedded components**, which are reported but never allowed to vote on
  the architecture. A Qwen encoder sitting inside a file says nothing about
  which denoiser the file holds.

The block counts and namespaces asserted below were measured across the 222
checkpoints of the reference library.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safetensors_helpers import build_header  # noqa: E402

from anima_remap import (  # noqa: E402
    block_count_from_keys,
    is_anima_engine,
    target_to_source,
    target_to_source_from_manifest,
)
from checkpoint_inspector import (  # noqa: E402
    embedded_components_from_keys,
    infer_architecture_id,
)


def header(*names, shapes=None):
    shapes = shapes or {}
    return build_header({n: ("BF16", shapes.get(n, [1])) for n in names})


def anima_keys(blocks=28, root="model.diffusion_model.", adapter_blocks=6, connector=False):
    keys = [f"{root}blocks.{i}.self_attn.q_proj.weight" for i in range(blocks)]
    keys += [f"{root}llm_adapter.blocks.{i}.attn.q.weight" for i in range(adapter_blocks)]
    if connector:
        keys += [
            "net.anima_v2_connector.quality_anchor.layer_mix_logits",
            "net.anima_v2_connector.semantic_resampler.blocks.0.attn.q.weight",
        ]
    return keys


def fake_engine(class_name, *, image_model="anima", repo="circlestone-labs/Anima"):
    config = type(
        class_name,
        (),
        {
            "unet_config": {"image_model": image_model} if image_model else {},
            "huggingface_repo": repo,
            "clip_target": {"qwen3_06b.transformer": "text_encoder"},
            "text_encoder_key_prefix": ["text_encoders."],
            "vae_key_prefix": ["vae."],
            "latent_format": type("Wan21", (), {}),
        },
    )()
    return type("FakeEngine", (), {"model_config": config})()


class ArchitectureIsDecidedByTheDenoiserTests(unittest.TestCase):
    def test_qwen_encoder_keys_do_not_turn_an_unknown_model_into_anima(self):
        head = header(
            "model.diffusion_model.blocks.0.weight",
            "text_encoders.qwen3_4b.transformer.model.layers.0.weight",
        )
        self.assertNotEqual("anima", infer_architecture_id(head))

    def test_llm_adapter_is_positive_anima_evidence(self):
        head = header(*anima_keys())
        self.assertEqual("anima", infer_architecture_id(head))

    def test_a_bundled_qwen_encoder_does_not_change_the_verdict(self):
        head = header(
            *anima_keys(),
            "text_encoders.qwen3_06b.transformer.model.embed_tokens.weight",
        )
        self.assertEqual("anima", infer_architecture_id(head))

    def test_sdxl_is_recognised_by_its_conditioner(self):
        head = header(
            "model.diffusion_model.input_blocks.0.0.weight",
            "conditioner.embedders.0.transformer.text_model.final_layer_norm.weight",
        )
        self.assertEqual("sdxl", infer_architecture_id(head))

    def test_sd3_gets_no_architecture_id_because_forge_cannot_load_it(self):
        """Its joint_blocks are recognisable, but Forge Neo ships no model
        config for SD3. Naming it would offer component slots for an engine
        that can never be loaded, so it stays unknown."""
        head = header("model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight")
        self.assertEqual("unknown", infer_architecture_id(head))


class AmbiguityStaysUnknownTests(unittest.TestCase):
    """Choosing the wrong family picks the wrong slots, so the header declines
    rather than guesses. The runtime capability probe resolves it properly."""

    def test_a_flux_family_denoiser_is_not_pinned_to_a_generation(self):
        head = header(
            "model.diffusion_model.double_blocks.0.img_attn.qkv.weight",
            "model.diffusion_model.single_blocks.0.linear1.weight",
        )
        self.assertEqual("unknown", infer_architecture_id(head))

    def test_a_generic_dit_is_unknown(self):
        head = header("model.diffusion_model.blocks.0.attn.qkv.weight")
        self.assertEqual("unknown", infer_architecture_id(head))

    def test_a_52_block_model_without_anima_evidence_stays_unknown(self):
        head = header(*[f"net.blocks.{i}.attn.q.weight" for i in range(52)])
        self.assertEqual("unknown", infer_architecture_id(head))

    def test_a_filename_cannot_promote_unknown_to_a_named_architecture(self):
        head = header("model.diffusion_model.blocks.0.attn.qkv.weight")
        self.assertEqual("unknown", infer_architecture_id(head, filename="anima_v1.safetensors"))


class EmbeddedComponentsTests(unittest.TestCase):
    """Presence is not the whole truth. Only 2 of 222 reference checkpoints
    embed components under the namespace their own architecture declares."""

    def test_a_modern_namespace_is_reported_as_readable(self):
        keys = anima_keys() + [
            "text_encoders.qwen3_06b.transformer.model.embed_tokens.weight",
            "vae.decoder.conv1.weight",
        ]
        found = {c["kind"]: c for c in embedded_components_from_keys(keys, "anima")}
        self.assertTrue(found["text_encoder"]["readable"])
        self.assertTrue(found["vae"]["readable"])
        self.assertEqual("text_encoders.", found["text_encoder"]["namespace"])

    def test_an_sd_style_namespace_in_an_anima_file_is_reported_unreadable(self):
        """Eighteen reference files look like this. Forge filters on
        ``text_encoders.``, finds nothing, and silently drops the lot."""
        keys = anima_keys() + [
            "cond_stage_model.qwen3_06b.transformer.model.embed_tokens.weight",
            "first_stage_model.decoder.conv1.weight",
        ]
        found = {c["kind"]: c for c in embedded_components_from_keys(keys, "anima")}
        self.assertFalse(found["text_encoder"]["readable"])
        self.assertFalse(found["vae"]["readable"])
        self.assertEqual("cond_stage_model.", found["text_encoder"]["namespace"])
        self.assertEqual("text_encoders.", found["text_encoder"]["expected_namespace"])

    def test_the_inner_role_is_reported_even_under_a_foreign_namespace(self):
        keys = anima_keys() + [
            "cond_stage_model.qwen3_06b.transformer.model.embed_tokens.weight"
        ]
        (component,) = [
            c for c in embedded_components_from_keys(keys, "anima")
            if c["kind"] == "text_encoder"
        ]
        self.assertEqual("qwen3_06b", component["role"])

    def test_sdxl_reads_its_own_traditional_namespaces(self):
        keys = [
            "model.diffusion_model.input_blocks.0.0.weight",
            "conditioner.embedders.0.transformer.text_model.final_layer_norm.weight",
            "first_stage_model.decoder.conv_in.weight",
        ]
        found = {c["kind"]: c for c in embedded_components_from_keys(keys, "sdxl")}
        self.assertTrue(found["vae"]["readable"])

    def test_readability_is_undecided_when_the_architecture_is_unknown(self):
        keys = ["model.diffusion_model.blocks.0.weight", "vae.decoder.conv1.weight"]
        (component,) = [
            c for c in embedded_components_from_keys(keys, "unknown")
            if c["kind"] == "vae"
        ]
        self.assertIsNone(component["readable"])

    def test_a_diffusion_only_checkpoint_reports_nothing_embedded(self):
        self.assertEqual((), embedded_components_from_keys(anima_keys(), "anima"))

    def test_the_connector_is_not_an_embedded_component(self):
        keys = anima_keys(blocks=52, root="net.", connector=True)
        self.assertEqual((), embedded_components_from_keys(keys, "anima"))


class AnimaBlockCountRegressionTests(unittest.TestCase):
    """Measured: 215 of 222 reference checkpoints are 28 blocks, four are 40,
    and three use a bare root with no prefix at all."""

    def test_every_generation_counts_its_root_blocks(self):
        for blocks in (28, 40, 52):
            with self.subTest(blocks=blocks):
                self.assertEqual(blocks, block_count_from_keys(anima_keys(blocks)))

    def test_all_three_generations_resolve_to_the_same_architecture(self):
        ids = {
            infer_architecture_id(header(*anima_keys(b))) for b in (28, 40, 52)
        }
        self.assertEqual({"anima"}, ids)

    def test_the_net_root_counts_the_same_as_the_model_root(self):
        self.assertEqual(
            block_count_from_keys(anima_keys(40, root="net.")),
            block_count_from_keys(anima_keys(40, root="model.diffusion_model.")),
        )

    def test_the_llm_adapter_stack_never_inflates_the_root_count(self):
        """Six adapter blocks in every generation, measured from the official
        2.9B manifest and both 3.8B releases."""
        self.assertEqual(28, block_count_from_keys(anima_keys(28, adapter_blocks=6)))
        self.assertEqual(28, block_count_from_keys(anima_keys(28, adapter_blocks=99)))

    def test_the_semantic_connector_never_inflates_the_root_count(self):
        keys = anima_keys(52, root="net.", connector=True)
        self.assertEqual(52, block_count_from_keys(keys))


class IsAnimaEngineTests(unittest.TestCase):
    def test_the_verdict_survives_a_class_rename(self):
        for name in ("Anima", "AnimaRenamedUpstream", "SomethingEntirelyElse"):
            with self.subTest(class_name=name):
                self.assertTrue(is_anima_engine(fake_engine(name)))

    def test_a_different_architecture_is_not_anima(self):
        engine = fake_engine("X", image_model="lumina2", repo="Tongyi-MAI/Z-Image-Turbo")
        self.assertFalse(is_anima_engine(engine))

    def test_an_engine_without_a_usable_config_is_not_anima_and_does_not_raise(self):
        self.assertFalse(is_anima_engine(object()))


class ExpansionManifestTests(unittest.TestCase):
    """The official releases ship their own expansion map. Reading it means a
    future generation works without editing the vendored tables."""

    #: Verbatim from Anima-3.8B.safetensors' safetensors metadata.
    MANIFEST_40_TO_52 = {
        "3": 2, "7": 5, "11": 8, "15": 11, "19": 14, "23": 17,
        "27": 20, "31": 23, "35": 26, "39": 29, "43": 32, "47": 35,
    }
    #: Verbatim from Anima-2.9B's expand_manifest.json.
    MANIFEST_28_TO_40 = {
        "2": 1, "5": 3, "8": 5, "11": 7, "14": 9, "17": 11,
        "21": 14, "24": 16, "27": 18, "30": 20, "33": 22, "36": 24,
    }

    def test_the_manifest_reproduces_the_vendored_40_to_52_table(self):
        self.assertEqual(
            target_to_source(40, 52),
            target_to_source_from_manifest(self.MANIFEST_40_TO_52, 40, 52),
        )

    def test_the_manifest_reproduces_the_vendored_28_to_40_table(self):
        self.assertEqual(
            target_to_source(28, 40),
            target_to_source_from_manifest(self.MANIFEST_28_TO_40, 28, 40),
        )

    def test_integer_keys_are_accepted_too(self):
        as_ints = {int(k): v for k, v in self.MANIFEST_40_TO_52.items()}
        self.assertEqual(
            target_to_source(40, 52),
            target_to_source_from_manifest(as_ints, 40, 52),
        )

    def test_a_manifest_of_the_wrong_size_is_refused(self):
        self.assertIsNone(target_to_source_from_manifest({"3": 2}, 40, 52))

    def test_an_unusable_manifest_is_refused_rather_than_patched(self):
        for broken in (None, {}, "not a mapping", {"oops": "nope"}):
            with self.subTest(manifest=broken):
                self.assertIsNone(target_to_source_from_manifest(broken, 40, 52))


class VendoredTablesAreUnchangedTests(unittest.TestCase):
    """Both tables were verified position-by-position against the official
    manifests. A change here is a regression, not an improvement."""

    def test_28_to_40_keeps_its_twelve_insertions(self):
        mapping = target_to_source(28, 40)
        self.assertEqual(40, len(mapping))
        self.assertEqual(27, max(mapping))

    def test_40_to_52_keeps_its_twelve_insertions(self):
        mapping = target_to_source(40, 52)
        self.assertEqual(52, len(mapping))
        self.assertEqual(39, max(mapping))

    def test_an_equal_count_is_the_identity(self):
        self.assertEqual(list(range(28)), target_to_source(28, 28))


class AnimaShipsUnderTwoNamespacesTests(unittest.TestCase):
    """Anima checkpoints do not agree on where to put their weights.

    Measured across nine reference checkpoints on 2026-09-19. `net.`:
    `anima_baseV10` (28), `Anima-2.9B-preview-v1` (40), `Anima-3.8B-v1.1`
    (52). `model.diffusion_model.`: `anima-turbo-v1.0`,
    `anima-aesthetic-v1.0`, `anima-aesthetic-v1.0b`, `anima_turboV11`,
    `anima_aestheticV11` (all 28).

    **The split is not chronological** -- the two newest releases use `net.`
    and the oldest derivatives use `model.diffusion_model.` What it lines up
    with is base versus published derivative. The six 28-block files are
    structurally identical once the prefix is stripped: same 685 keys, same
    shapes, same dtypes, and byte-identical payload sizes.

    So the two-prefix handling in `anima_remap._DISK_BLOCK_RE` and in
    `architecture_guess.state_dict_from_header` is not defensive coding
    against a hypothetical. Dropping either branch breaks half the base
    models in the library.
    """

    def test_both_namespaces_give_the_same_architecture_and_depth(self):
        from anima_remap import block_count_from_keys

        # Every depth in the library appears under `net.`, and the 28-block
        # one appears under both.
        for root in ("net.", "model.diffusion_model."):
            for blocks in (28, 40, 52):
                with self.subTest(root=root, blocks=blocks):
                    keys = anima_keys(blocks=blocks, root=root)
                    head = header(*keys)
                    self.assertEqual("anima", infer_architecture_id(head))
                    self.assertEqual(blocks, block_count_from_keys(keys))

    def test_the_header_driven_detector_leaves_both_alone(self):
        # `state_dict_from_header` only adds the diffusion namespace when
        # neither is present. Adding it on top of `net.` would bury the keys
        # the detector matches on.
        from architecture_guess import state_dict_from_header

        for root in ("net.", "model.diffusion_model."):
            with self.subTest(root=root):
                head = header(*anima_keys(blocks=4, root=root))
                built = state_dict_from_header(head)
                self.assertTrue(all(k.startswith(root) for k in built), sorted(built)[:3])


if __name__ == "__main__":
    unittest.main()
