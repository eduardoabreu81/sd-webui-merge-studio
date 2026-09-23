import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

from merge_studio.aux_inspector import (
    detect_file_kind,
    format_lora_dashboard_html,
    inspect_lora,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safetensors_helpers import (  # noqa: E402
    qwen_encoder,
    t5_encoder,
    vae_2d,
    vae_3d,
    write_safetensors_header,
)


def write_header_only_safetensors(path: Path, metadata: dict[str, str]) -> None:
    header = json.dumps({"__metadata__": metadata}).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header)


class LoraMetadataBoundaryTests(unittest.TestCase):
    def test_inspector_uses_embedded_trigger_instead_of_conflicting_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora = Path(tmp) / "concept.safetensors"
            write_header_only_safetensors(
                lora, {"modelspec.trigger_phrase": "inside-file"}
            )
            lora.with_suffix(".json").write_text(
                json.dumps({"activation text": "from-sidecar"}), encoding="utf-8"
            )

            info = inspect_lora(str(lora))

            self.assertEqual("inside-file", info["activation_text"])
            self.assertEqual(
                "modelspec.trigger_phrase", info["activation_text_source"]
            )

    def test_inspector_does_not_treat_sidecar_only_value_as_file_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora = Path(tmp) / "turbo-concept.safetensors"
            write_header_only_safetensors(lora, {})
            lora.with_suffix(".json").write_text(
                json.dumps({"activation text": "from-sidecar"}), encoding="utf-8"
            )

            info = inspect_lora(str(lora))
            html = format_lora_dashboard_html(info)

            self.assertEqual("", info["activation_text"])
            self.assertIn("NO TRIGGER DECLARED IN THIS FILE", html)
            self.assertNotIn("from-sidecar", html)
            self.assertNotIn("NO TRIGGER WORD</div>", html)


class FileKindSurvivesTheClassifierReorderTests(unittest.TestCase):
    """Component classification was reordered so encoder signatures win before
    any autoencoder rule. `detect_file_kind` reads the same classifier, so
    these guard that the reorder did not move a file into the wrong bucket."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def write(self, name, tensors):
        return write_safetensors_header(
            os.path.join(self.dir.name, name), tensors
        )

    def test_a_lora_is_still_a_lora(self):
        path = self.write(
            "style.safetensors",
            {
                "lora_unet_blocks_0.lora_down.weight": ("BF16", [32, 2048]),
                "lora_unet_blocks_0.lora_up.weight": ("BF16", [2048, 32]),
            },
        )
        self.assertEqual("lora", detect_file_kind(path))

    def test_a_diffusion_model_is_still_a_checkpoint(self):
        path = self.write(
            "model.safetensors",
            {"model.diffusion_model.blocks.0.attn.qkv.weight": ("BF16", [6144, 2048])},
        )
        self.assertEqual("checkpoint", detect_file_kind(path))

    def test_encoders_and_vaes_are_both_modules(self):
        cases = {
            "t5.safetensors": t5_encoder(vocab=32128),
            "qwen.safetensors": qwen_encoder(28, 1024),
            "ae.safetensors": vae_2d(latent_channels=16),
            "qwen_image_vae.safetensors": vae_3d(),
        }
        for name, tensors in cases.items():
            with self.subTest(component=name):
                self.assertEqual("module", detect_file_kind(self.write(name, tensors)))

    def test_a_standalone_t5_is_an_encoder_not_an_autoencoder(self):
        """Its ``encoder.block.*`` keys used to satisfy the generic
        encoder/decoder rule and land it in the VAE bucket."""
        from merge_studio.aux_inspector import inspect_module

        info = inspect_module(self.write("t5xxl.safetensors", t5_encoder(vocab=32128)))
        self.assertEqual("text_encoder", info["kind"])
        self.assertEqual("t5xxl", info["signature_id"])


class FrozenAdapterModulesAreNotAChangeTests(unittest.TestCase):
    """PEFT starts lora_A random and lora_B at zero. A LoRA whose trainer froze
    the llm_adapter still ships its modules -- Turbo-ANIMA-v2.9 carries 186
    such tensors -- with lora_A as large as any trained one and lora_B at 0,
    so the change it applies there is nil. Measuring every factor saw 1x the
    main blocks and flagged it."""

    def write_lora(self, adapter_b_scale: float) -> str:
        import numpy as np
        from safetensors.numpy import save_file

        rng = np.random.default_rng(0)
        tensors = {}
        for i in range(2):
            main = f"diffusion_model.blocks.{i}.self_attn.q_proj"
            tensors[f"{main}.lora_A.weight"] = rng.normal(0, 2e-2, (16, 64)).astype(np.float32)
            tensors[f"{main}.lora_B.weight"] = rng.normal(0, 4e-4, (64, 16)).astype(np.float32)
            adapter = f"diffusion_model.llm_adapter.blocks.{i}.self_attn.q_proj"
            tensors[f"{adapter}.lora_A.weight"] = rng.normal(0, 2e-2, (16, 64)).astype(np.float32)
            tensors[f"{adapter}.lora_B.weight"] = (
                rng.normal(0, 4e-4, (64, 16)) * adapter_b_scale
            ).astype(np.float32)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "turbo.safetensors")
        save_file(tensors, path)
        return path

    def test_an_adapter_with_zero_lora_b_is_not_significant(self):
        info = inspect_lora(self.write_lora(adapter_b_scale=0.0))

        self.assertFalse(info["llm_adapter_significant"])

    def test_an_adapter_that_was_trained_is_still_flagged(self):
        info = inspect_lora(self.write_lora(adapter_b_scale=1.0))

        self.assertTrue(info["llm_adapter_significant"])


if __name__ == "__main__":
    unittest.main()
