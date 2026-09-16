import json
import struct
import tempfile
import unittest
from pathlib import Path

from aux_inspector import format_lora_dashboard_html, inspect_lora


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


if __name__ == "__main__":
    unittest.main()
