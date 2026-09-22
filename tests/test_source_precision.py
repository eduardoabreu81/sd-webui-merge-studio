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
            from merge_studio.source_precision import match_source_dtypes
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
        from merge_studio.source_precision import try_match_source_dtypes

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
        from merge_studio.source_precision import try_match_source_dtypes

        changed, warning = try_match_source_dtypes(
            {}, "no-such-model.safetensors", set(), {}
        )

        self.assertEqual(0, changed)
        self.assertIn("no-such-model.safetensors", warning)

    def test_readable_source_reports_no_warning(self):
        from merge_studio.source_precision import try_match_source_dtypes

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "base.safetensors"
            write_source_header(source, {"a.weight": "F16"})

            changed, warning = try_match_source_dtypes({}, str(source), set(), {})

            self.assertEqual(0, changed)
            self.assertEqual("", warning)


class DiffusionPrefixMatchingTests(unittest.TestCase):
    """The diffusion model does not ship under one prefix.

    Measured on the official Anima releases: anima-base-v1.0 and the previews
    ship as "net.", while anima-aesthetic and anima-turbo ship as
    "model.diffusion_model.". The save path always writes the latter, so exact
    key matching silently restored nothing for the whole Base line.
    """

    def _dtypes(self):
        class FakeDtype:
            def __init__(self, name):
                self.name = name
                self.is_floating_point = True

        class FakeTensor:
            def __init__(self, dtype):
                self.dtype = dtype

            def to(self, dtype):
                return FakeTensor(dtype)

        return FakeDtype, FakeTensor

    def test_net_prefixed_source_matches_saved_diffusion_keys(self):
        from merge_studio.source_precision import match_source_dtypes

        FakeDtype, FakeTensor = self._dtypes()
        fp16, bf16 = FakeDtype("F16"), FakeDtype("BF16")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "base.safetensors"
            write_source_header(source, {"net.blocks.0.mlp.weight": "BF16"})
            state_dict = {"model.diffusion_model.blocks.0.mlp.weight": FakeTensor(fp16)}

            changed = match_source_dtypes(
                state_dict, str(source), set(state_dict), {"BF16": bf16}
            )

            self.assertEqual(1, changed)
            self.assertIs(bf16, state_dict["model.diffusion_model.blocks.0.mlp.weight"].dtype)

    def test_other_components_do_not_match_a_diffusion_key(self):
        from merge_studio.source_precision import match_source_dtypes

        FakeDtype, FakeTensor = self._dtypes()
        fp16, bf16 = FakeDtype("F16"), FakeDtype("BF16")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "base.safetensors"
            write_source_header(source, {"net.blocks.0.mlp.weight": "BF16"})
            # Same trailing name, different component: a dtype must not cross
            # from the diffusion model into the autoencoder.
            state_dict = {"vae.blocks.0.mlp.weight": FakeTensor(fp16)}

            changed = match_source_dtypes(
                state_dict, str(source), set(state_dict), {"BF16": bf16}
            )

            self.assertEqual(0, changed)
            self.assertIs(fp16, state_dict["vae.blocks.0.mlp.weight"].dtype)

    def test_first_stage_model_source_matches_saved_vae_keys(self):
        """Mugen accepts both autoencoder prefixes on load but saves "vae.",
        and Chroma renames "first_stage_model." to "vae." on load. Either way
        the source header and the output disagree on the prefix."""
        from merge_studio.source_precision import match_source_dtypes

        FakeDtype, FakeTensor = self._dtypes()
        fp16, bf16 = FakeDtype("F16"), FakeDtype("BF16")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "mugen.safetensors"
            write_source_header(source, {"first_stage_model.decoder.conv_in.weight": "BF16"})
            state_dict = {"vae.decoder.conv_in.weight": FakeTensor(fp16)}

            changed = match_source_dtypes(
                state_dict, str(source), set(state_dict), {"BF16": bf16}
            )

            self.assertEqual(1, changed)
            self.assertIs(bf16, state_dict["vae.decoder.conv_in.weight"].dtype)

    def test_a_vae_key_does_not_borrow_from_the_diffusion_model(self):
        from merge_studio.source_precision import match_source_dtypes

        FakeDtype, FakeTensor = self._dtypes()
        fp16, bf16 = FakeDtype("F16"), FakeDtype("BF16")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "mixed.safetensors"
            write_source_header(source, {"first_stage_model.decoder.conv_in.weight": "BF16"})
            state_dict = {"model.diffusion_model.decoder.conv_in.weight": FakeTensor(fp16)}

            changed = match_source_dtypes(
                state_dict, str(source), set(state_dict), {"BF16": bf16}
            )

            self.assertEqual(0, changed)

    def test_a_name_two_source_keys_share_is_dropped_not_guessed(self):
        from merge_studio.source_precision import match_source_dtypes

        FakeDtype, FakeTensor = self._dtypes()
        fp16, bf16 = FakeDtype("F16"), FakeDtype("BF16")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "both.safetensors"
            write_source_header(
                source,
                {
                    "net.blocks.0.mlp.weight": "BF16",
                    "model.diffusion_model.blocks.0.mlp.weight": "F32",
                },
            )
            state_dict = {"model.diffusion_model.blocks.0.mlp.weight": FakeTensor(fp16)}

            # The exact key is present, so it wins outright and nothing is guessed.
            changed = match_source_dtypes(
                state_dict, str(source), set(state_dict), {"F32": bf16}
            )

            self.assertEqual(1, changed)

    def test_ambiguous_stripped_name_is_not_used_as_a_fallback(self):
        from merge_studio.source_precision import match_source_dtypes

        FakeDtype, FakeTensor = self._dtypes()
        fp16, bf16 = FakeDtype("F16"), FakeDtype("BF16")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "both.safetensors"
            write_source_header(
                source,
                {
                    "net.blocks.0.mlp.weight": "BF16",
                    "model.diffusion_model.blocks.0.mlp.weight": "BF16",
                },
            )
            # No exact match, and the stripped name maps to two source tensors.
            state_dict = {"blocks.0.mlp.weight": FakeTensor(fp16)}

            changed = match_source_dtypes(
                state_dict, str(source), set(state_dict), {"BF16": bf16}
            )

            self.assertEqual(0, changed)
            self.assertIs(fp16, state_dict["blocks.0.mlp.weight"].dtype)


if __name__ == "__main__":
    unittest.main()
