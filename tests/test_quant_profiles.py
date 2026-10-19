"""Quantization from full precision picks what the official builds picked.

Each case lists layers read from a published build's header on 2026-10-18:
`QUANT` were quantized in that build, `KEEP` were left at full precision. The
profile has to reproduce both sides for that build's format.
"""

import unittest

from merge_studio.quant_profiles import FORMAT_GROUPS, profile_for


def with_parents(paths):
    """Module paths as `named_modules()` yields them: every ancestor too."""
    out = set()
    for path in paths:
        parts = path.split(".")
        for i in range(1, len(parts) + 1):
            out.add(".".join(parts[:i]))
    return out


KREA2_BLOCKS = ["blocks.0.attn.wq", "blocks.0.attn.gate", "blocks.27.mlp.down"]
KREA2_TEXT_FUSION = [
    "txtfusion.layerwise_blocks.0.attn.wq",
    "txtfusion.refiner_blocks.1.mlp.up",
]
KREA2_KEEP = ["first", "last.linear", "tmlp.0", "tproj.1", "txtmlp.0", "txtfusion.projector"]

QWEN21_QUANT = [
    "transformer_blocks.0.attn.to_q",
    "transformer_blocks.0.attn.to_out.0",
    "transformer_blocks.31.img_mlp.gate_up",
    "transformer_blocks.31.img_mlp.out",
]
QWEN21_KEEP = [
    "img_in", "txt_in.in_layer", "txt_in.out_layer", "modulation.1", "norm_out.linear",
    "proj_out", "time_text_embed.timestep_embedder.linear_1",
]

FLUX3_CONTENT = ["content_mode_blocks.image.0.qkv", "content_mode_blocks.image_cond.4.mlp_out",
                 "single_blocks.27.attn_out", "single_blocks.0.mlp_in"]
FLUX3_TEXT = ["txt_mode_blocks.0.qkv", "txt_mode_blocks.4.mlp_out"]
FLUX3_KEEP = ["txt_in", "time_in.in_layer", "ref_time_in.out_layer", "emb_in.image",
              "final_layer.image.linear", "single_stream_modulations.txt.lin",
              "early_stream_modulations.image.lin"]

MOSS_QUANT = ["layers.0.proj.qkv", "layers.41.proj.o", "layers.3.proj.ffn_up_gate", "layers.3.proj.ffn_down"]
MOSS_KEEP = ["final_linear_video", "shared_block_modulation", "video_head_modulation",
             "video_embedder", "time_embedding.0", "text_embedder.proj",
             "text_embedder.blocks.0.qkv", "text_embedder.blocks.1.ffn_down"]

IRIS_BLOCKS = ["blocks.0.attn.q_proj_x", "blocks.0.mlp_y.w1", "blocks.23.q_proj", "blocks.23.attn_gate"]
IRIS_PIXEL = ["pixel_blocks.0.adaln", "pixel_blocks.0.compress", "pixel_blocks.3.attn.qkv"]
IRIS_PIXEL_MLP = ["pixel_blocks.0.mlp.fc1", "pixel_blocks.3.mlp.fc2"]
IRIS_ADAPTER = ["y_embedder.layer_blocks.0.attn.qkv", "y_embedder.refiner.blocks.0.mlp.w2",
                "y_embedder.refiner.proj"]
IRIS_KEEP = ["final_layer.linear", "s_embedder.proj", "t_embedder.mlp.0", "pixel_embedder.proj",
             "modulation_cores.adaln_img", "y_embedder.layer_pool"]

MODELS = {
    "krea2": KREA2_BLOCKS + KREA2_TEXT_FUSION + KREA2_KEEP,
    "qwen_image21": QWEN21_QUANT + QWEN21_KEEP + ["txt_in.text_norm"],
    "flux3": FLUX3_CONTENT + FLUX3_TEXT + FLUX3_KEEP,
    "moss_image": MOSS_QUANT + MOSS_KEEP,
    "iris": IRIS_BLOCKS + IRIS_PIXEL + IRIS_PIXEL_MLP + IRIS_ADAPTER + IRIS_KEEP,
}

# (model, output format, quantized in the official build, kept there)
OFFICIAL_BUILDS = [
    ("krea2", "int8_tensorwise_convrot", KREA2_BLOCKS, KREA2_TEXT_FUSION + KREA2_KEEP),
    ("krea2", "float8_e4m3fn", KREA2_BLOCKS + KREA2_TEXT_FUSION, KREA2_KEEP),
    ("krea2", "mxfp8", KREA2_BLOCKS + KREA2_TEXT_FUSION, KREA2_KEEP),
    ("krea2", "nvfp4", KREA2_BLOCKS + KREA2_TEXT_FUSION, KREA2_KEEP),
    ("qwen_image21", "int8_tensorwise_convrot", QWEN21_QUANT, QWEN21_KEEP),
    ("qwen_image21", "nvfp4", QWEN21_QUANT, QWEN21_KEEP),
    ("flux3", "int8_tensorwise_convrot", FLUX3_CONTENT + FLUX3_TEXT, FLUX3_KEEP),
    ("flux3", "nvfp4", FLUX3_CONTENT, FLUX3_TEXT + FLUX3_KEEP),
    ("moss_image", "int8_tensorwise_convrot", MOSS_QUANT, MOSS_KEEP),
    ("iris", "int8_tensorwise_convrot", IRIS_BLOCKS + IRIS_PIXEL,
     IRIS_PIXEL_MLP + IRIS_ADAPTER + IRIS_KEEP),
    ("iris", "asym_w4a8_int8", IRIS_BLOCKS + IRIS_ADAPTER,
     IRIS_PIXEL + IRIS_PIXEL_MLP + IRIS_KEEP),
]


class OfficialBuildTests(unittest.TestCase):
    def test_each_profile_reproduces_the_official_selection(self):
        for model, fmt, quantized, kept in OFFICIAL_BUILDS:
            profile = profile_for(with_parents(MODELS[model]))
            self.assertIsNotNone(profile, model)
            self.assertEqual(model, profile.name)
            for path in quantized:
                with self.subTest(model=model, format=fmt, layer=path):
                    self.assertTrue(profile.selects(path, fmt))
            for path in kept:
                with self.subTest(model=model, format=fmt, layer=path):
                    self.assertFalse(profile.selects(path, fmt))

    def test_an_unmeasured_format_gets_the_conservative_set(self):
        # No FP8 build of FLUX 3 was found: its text-mode blocks stay, as in
        # NVFP4, rather than following the wider INT8 selection.
        flux3 = profile_for(with_parents(MODELS["flux3"]))
        self.assertFalse(flux3.selects("txt_mode_blocks.0.qkv", "float8_e4m3fn"))
        self.assertTrue(flux3.selects("single_blocks.0.qkv", "float8_e4m3fn"))
        iris = profile_for(with_parents(MODELS["iris"]))
        self.assertFalse(iris.selects("pixel_blocks.0.compress", "nvfp4"))
        self.assertFalse(iris.selects("y_embedder.refiner.proj", "nvfp4"))

    def test_every_offered_quantized_format_has_a_group(self):
        from_ui = {"float8_e4m3fn", "float8_e5m2", "int8_tensorwise", "int8_tensorwise_convrot",
                   "nvfp4", "mxfp8", "convrot_w4a4", "asym_w4a8_int8", "w6a8_int8"}
        self.assertEqual(from_ui, set(FORMAT_GROUPS))


class IdentityTests(unittest.TestCase):
    """A profile must never claim a model it was not measured on."""

    OTHERS = {
        "anima": ["blocks.0.self_attn.q_proj", "blocks.0.mlp.layer1", "t_embedder.1.linear_1",
                  "x_embedder.proj.1", "final_layer.linear", "llm_adapter.blocks.0.self_attn.q_proj"],
        "qwen_image": ["transformer_blocks.0.attn.to_q", "transformer_blocks.0.img_mod.1",
                       "txt_in", "img_in", "proj_out", "time_text_embed.timestep_embedder.linear_1"],
        "flux1": ["double_blocks.0.img_attn.qkv", "single_blocks.0.linear1", "txt_in", "img_in",
                  "time_in.in_layer", "final_layer.linear"],
        "zimage": ["layers.0.attention.qkv", "context_refiner.0.attention.qkv",
                   "noise_refiner.0.feed_forward.w1", "cap_embedder.1", "x_embedder"],
        "qwen3vl_text_encoder": ["model.layers.0.self_attn.q_proj", "model.visual.blocks.0.attn.qkv",
                                 "model.embed_tokens"],
        "wan_vae": ["decoder.conv1", "encoder.downsamples.0.residual.0", "conv2"],
    }

    def test_no_profile_matches_another_architecture(self):
        for name, paths in self.OTHERS.items():
            with self.subTest(architecture=name):
                self.assertIsNone(profile_for(with_parents(paths)))

    def test_qwen_image_21_is_not_qwen_image(self):
        # Both have transformer_blocks, txt_in and proj_out; only 2.1's text
        # projection has a norm of its own.
        self.assertIsNone(profile_for(with_parents(self.OTHERS["qwen_image"])))
        self.assertEqual(
            "qwen_image21",
            profile_for(with_parents(MODELS["qwen_image21"])).name,
        )


if __name__ == "__main__":
    unittest.main()
