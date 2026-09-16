import json
import struct
import tempfile
import unittest
from pathlib import Path

def write_source_header(path: Path, tensors: dict[str, str]) -> None:
    offset = 0
    header = {}
    for name, dtype in tensors.items():
        byte_width = {"F16": 2, "BF16": 2, "F32": 4}[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": [1],
            "data_offsets": [offset, offset + byte_width],
        }
        offset += byte_width
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + (b"\0" * offset))


class SourcePrecisionTests(unittest.TestCase):
    def test_same_restores_bf16_dtype_recorded_in_source_header(self):
        try:
            from source_precision import match_source_dtypes
        except ModuleNotFoundError:
            self.fail("source-aware precision support is missing")

        key = "model.diffusion_model.block.weight"

        class FakeDtype:
            def __init__(self, name: str):
                self.name = name
                self.is_floating_point = True

        class FakeTensor:
            def __init__(self, dtype: FakeDtype):
                self.dtype = dtype

            def to(self, dtype: FakeDtype):
                return FakeTensor(dtype)

        fp16 = FakeDtype("F16")
        bf16 = FakeDtype("BF16")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "base.safetensors"
            write_source_header(source, {key: "BF16"})
            state_dict = {key: FakeTensor(fp16)}

            changed = match_source_dtypes(
                state_dict, str(source), {key}, {"BF16": bf16}
            )

            self.assertEqual(1, changed)
            self.assertIs(bf16, state_dict[key].dtype)


class SourcePrecisionFallbackTests(unittest.TestCase):
    """A source with no readable header must not destroy a finished merge."""

    def test_non_safetensors_source_is_reported_not_raised(self):
        from source_precision import try_match_source_dtypes

        key = "model.diffusion_model.block.weight"
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "legacy.ckpt"
            source.write_bytes(b"not a safetensors file")
            state_dict = {key: object()}

            changed, warning = try_match_source_dtypes(
                state_dict, str(source), {key}, {}
            )

            self.assertEqual(0, changed)
            self.assertIn("legacy.ckpt", warning)
            self.assertIn("Keeping the precision", warning)

    def test_missing_source_file_is_reported_not_raised(self):
        from source_precision import try_match_source_dtypes

        changed, warning = try_match_source_dtypes(
            {}, "no-such-model.safetensors", set(), {}
        )

        self.assertEqual(0, changed)
        self.assertIn("no-such-model.safetensors", warning)

    def test_readable_source_reports_no_warning(self):
        from source_precision import try_match_source_dtypes

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "base.safetensors"
            write_source_header(source, {"a.weight": "F16"})

            changed, warning = try_match_source_dtypes({}, str(source), set(), {})

            self.assertEqual(0, changed)
            self.assertEqual("", warning)


if __name__ == "__main__":
    unittest.main()
