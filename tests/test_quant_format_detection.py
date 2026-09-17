"""A quantized checkpoint's format lives in bytes, not in the header.

Each quantized layer carries a `comfy_quant` tensor that is not weights: it is
the JSON config Forge wrote for that layer, stored as uint8. The header alone
gives only its size, so every comfy_quant checkpoint used to be reported as a
flat "INT8 (comfy_quant)".

The blob below is the real one, read out of
Anima-2.9B-preview-v1_int8_convrot.safetensors at its recorded offset:

    {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}

The distinction matters because it is one this extension already asks the user
to make: "INT8 (tensor-wise, single scale)" and "INT8 (convrot: per-channel +
rotation, recommended)" are separate entries in its own output-format dropdown.
Of the 17 quantized checkpoints surveyed in a real library, 14 were convrot and
3 were not -- and all 17 were reported identically.
"""

import json
import struct
import tempfile
import unittest
from pathlib import Path

from checkpoint_inspector import inspect_checkpoint

CONVROT = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
TENSORWISE = {"format": "int8_tensorwise", "per_row": True}


def write_checkpoint(path: Path, layer_configs: list[dict] | None) -> None:
    """A minimal Anima-shaped checkpoint, optionally quantized."""
    header: dict = {}
    blobs: list[bytes] = []
    offset = 0

    def add(name: str, dtype: str, payload: bytes, shape: list[int]) -> None:
        nonlocal offset
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        blobs.append(payload)
        offset += len(payload)

    if layer_configs is None:
        add("model.diffusion_model.blocks.0.mlp.weight", "BF16", b"\0" * 8, [2, 2])
    else:
        for i, conf in enumerate(layer_configs):
            prefix = f"model.diffusion_model.blocks.{i}.mlp"
            add(f"{prefix}.weight", "I8", b"\0" * 4, [2, 2])
            add(f"{prefix}.weight_scale", "F32", b"\0" * 4, [1])
            encoded = json.dumps(conf).encode("utf-8")
            add(f"{prefix}.comfy_quant", "U8", encoded, [len(encoded)])

    encoded_header = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + b"".join(blobs))


class QuantFormatDetectionTests(unittest.TestCase):
    def _precision(self, layer_configs) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            write_checkpoint(path, layer_configs)
            return inspect_checkpoint(str(path)).get("precision")

    def test_convrot_is_named(self):
        self.assertEqual("INT8 convrot (comfy_quant)", self._precision([CONVROT] * 3))

    def test_plain_tensorwise_is_not_called_convrot(self):
        self.assertEqual(
            "INT8 tensor-wise (comfy_quant)", self._precision([TENSORWISE] * 3)
        )

    def test_a_checkpoint_quantized_two_ways_says_so(self):
        precision = self._precision([CONVROT, TENSORWISE])

        self.assertTrue(precision.startswith("Mixed: "), precision)
        self.assertIn("INT8 convrot", precision)
        self.assertIn("INT8 tensor-wise", precision)

    def test_fp8_layers_are_named_by_their_format(self):
        self.assertEqual(
            "FP8 (e4m3fn) (comfy_quant)",
            self._precision([{"format": "float8_e4m3fn"}] * 2),
        )

    def test_an_unknown_format_is_reported_verbatim(self):
        self.assertEqual(
            "some_future_format (comfy_quant)",
            self._precision([{"format": "some_future_format"}] * 2),
        )

    def test_an_unreadable_config_falls_back_instead_of_failing(self):
        # Truncated JSON: the family is still named from the weight dtypes.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            write_checkpoint(path, [CONVROT])
            raw = bytearray(path.read_bytes())
            raw[-10:] = b"0" * 10
            path.write_bytes(bytes(raw))

            self.assertEqual(
                "INT8 (comfy_quant)", inspect_checkpoint(str(path)).get("precision")
            )

    def test_an_unquantized_checkpoint_is_untouched(self):
        self.assertEqual("BF16", self._precision(None))


if __name__ == "__main__":
    unittest.main()
