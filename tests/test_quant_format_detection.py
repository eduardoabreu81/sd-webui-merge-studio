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

from merge_studio.checkpoint_inspector import inspect_checkpoint

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


W4A4 = {"format": "convrot_w4a4", "convrot_groupsize": 256, "quant_group_size": 64}


def write_declared_checkpoint(
    path: Path, declared: dict | None, blobs: dict[str, dict] | None = None, raw: str | None = None
) -> None:
    """Integer weights next to a BF16 norm, with the formats declared in
    `__metadata__._quantization_metadata` instead of (or as well as) blobs.

    Shaped after krea2TurboNSFWV10: 256 layers declared as convrot_w4a4, no
    comfy_quant tensor at all, names without the model prefix -- and read by
    this inspector as BF16, which is what its norms are stored in.
    """
    header: dict = {}
    payloads: list[bytes] = []
    offset = 0

    def add(name: str, dtype: str, payload: bytes, shape: list[int]) -> None:
        nonlocal offset
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(payload)]}
        payloads.append(payload)
        offset += len(payload)

    add("blocks.0.norm.weight", "BF16", b"\0" * 4, [2])
    for layer in ("blocks.0.attn.gate", "blocks.1.attn.gate"):
        add(f"{layer}.weight", "I8", b"\0" * 4, [2, 2])
        add(f"{layer}.weight_scale", "F32", b"\0" * 4, [1])
        if blobs and layer in blobs:
            encoded = json.dumps(blobs[layer]).encode("utf-8")
            add(f"{layer}.comfy_quant", "U8", encoded, [len(encoded)])
    if raw is not None:
        header["__metadata__"] = {"_quantization_metadata": raw}
    elif declared is not None:
        header["__metadata__"] = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": declared})}

    encoded_header = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + b"".join(payloads))


class DeclaredQuantFormatTests(unittest.TestCase):
    """Formats stated in the metadata are formats, the way Forge reads them."""

    def _precision(self, declared, blobs=None, raw=None) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            write_declared_checkpoint(path, declared, blobs, raw)
            return inspect_checkpoint(str(path)).get("precision")

    def test_a_declared_only_build_is_not_called_bf16(self):
        declared = {"blocks.0.attn.gate": W4A4, "blocks.1.attn.gate": W4A4}

        self.assertEqual("INT4 W4A4 (quantization metadata)", self._precision(declared))

    def test_prefixed_declarations_are_read_too(self):
        # moodyMossMix names its layers with the model prefix.
        declared = {f"model.diffusion_model.blocks.{i}.attn.gate": CONVROT for i in (0, 1)}

        self.assertEqual("INT8 convrot (quantization metadata)", self._precision(declared))

    def test_the_declaration_wins_over_the_blob_it_replaces(self):
        # Forge writes the declared config over the file's own blob.
        declared = {"blocks.0.attn.gate": CONVROT, "blocks.1.attn.gate": CONVROT}
        blobs = {"blocks.0.attn.gate": TENSORWISE, "blocks.1.attn.gate": TENSORWISE}

        self.assertEqual("INT8 convrot (comfy_quant)", self._precision(declared, blobs))

    def test_blobs_the_declaration_leaves_out_still_count(self):
        declared = {"blocks.0.attn.gate": {"format": "nvfp4"}}
        blobs = {"blocks.1.attn.gate": CONVROT}

        precision = self._precision(declared, blobs)

        self.assertEqual("Mixed: INT8 convrot, NVFP4 (comfy_quant)", precision)

    def test_unreadable_metadata_falls_back_to_the_dtypes(self):
        self.assertEqual("BF16", self._precision(None, raw="{not json"))

    def test_the_header_alone_is_enough(self):
        # Component inspectors pass no file path; the declaration needs none.
        from merge_studio.checkpoint_inspector import _detect_precision

        header = {"__metadata__": {"_quantization_metadata": json.dumps({"layers": {"a": W4A4}})}}

        self.assertEqual("INT4 W4A4 (quantization metadata)", _detect_precision(header))


if __name__ == "__main__":
    unittest.main()
