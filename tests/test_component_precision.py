"""Each component keeps the precision of the file it came from.

Not Model A's, and not its neighbour's. Two encoders in one state dict have
keys that reduce to the same suffix, so matching has to be scoped to the slot's
own namespace -- a global suffix index would cast one component to the other's
dtype and nothing downstream would notice.

The Anima LLM Adapter is the case that makes scoping load-bearing. Forge's
`process_anima` moves it into the text encoder's bucket at load time, so it
travels alongside the encoder while belonging to the diffusion model. When the
encoder slot is filled from an external file, those keys still came from Model
A and must not be matched against that file's header.
"""

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from component_bundle import ResolvedComponent, component_provenance  # noqa: E402
from source_precision import apply_component_precision  # noqa: E402


class FakeDtype:
    def __init__(self, name, floating=True):
        self.name = name
        self.is_floating_point = floating

    def __repr__(self):
        return f"<{self.name}>"


class FakeTensor:
    def __init__(self, dtype):
        self.dtype = dtype

    def to(self, dtype):
        return FakeTensor(dtype)


F16 = FakeDtype("F16")
BF16 = FakeDtype("BF16")
F32 = FakeDtype("F32")
F8 = FakeDtype("F8_E4M3")
U8 = FakeDtype("U8", floating=False)

DTYPES = {"F16": F16, "BF16": BF16, "F32": F32, "F8_E4M3": F8}


def write_header(path, tensors):
    """`{name: dtype_code}` -> a safetensors file with a matching header."""
    widths = {"F16": 2, "BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}
    offset, header = 0, {}
    for name, dtype in tensors.items():
        width = widths[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": [1],
            "data_offsets": [offset, offset + width],
        }
        offset += width
    raw = json.dumps(header).encode("utf-8")
    Path(path).write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)
    return str(path)


class Slot:
    """Stands in for the resolved slot's declared namespace."""

    def __init__(self, *prefixes):
        self.internal_prefixes = prefixes


def component(slot_id, path, output_format="same", source="file", storage="plain"):
    return ResolvedComponent(
        slot_id=slot_id,
        source=source,
        path=path,
        signature_id=slot_id,
        output_format=output_format,
        source_precision=storage,
    )


class SameReadsEachComponentsOwnHeaderTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def test_two_encoders_keep_their_own_source_dtypes(self):
        clip = write_header(
            self.path("clip.safetensors"), {"text_model.weight": "BF16"}
        )
        t5 = write_header(
            self.path("t5.safetensors"), {"encoder.weight": "F32"}
        )
        state = {
            "clip_l.transformer.text_model.weight": FakeTensor(F16),
            "t5xxl.transformer.encoder.weight": FakeTensor(F16),
        }

        apply_component_precision(
            state, component("clip_l", clip), Slot("clip_l.transformer."), DTYPES
        )
        apply_component_precision(
            state, component("t5xxl", t5), Slot("t5xxl.transformer."), DTYPES
        )

        self.assertIs(BF16, state["clip_l.transformer.text_model.weight"].dtype)
        self.assertIs(F32, state["t5xxl.transformer.encoder.weight"].dtype)

    def test_a_slot_never_reads_its_neighbours_header(self):
        """Both keys reduce to the same suffix. Only the prefix keeps them
        apart, so a global suffix index would cross the two."""
        clip = write_header(self.path("clip.safetensors"), {"shared.weight": "BF16"})
        state = {
            "clip_l.transformer.shared.weight": FakeTensor(F16),
            "t5xxl.transformer.shared.weight": FakeTensor(F16),
        }
        apply_component_precision(
            state, component("clip_l", clip), Slot("clip_l.transformer."), DTYPES
        )
        self.assertIs(BF16, state["clip_l.transformer.shared.weight"].dtype)
        self.assertIs(F16, state["t5xxl.transformer.shared.weight"].dtype)

    def test_a_key_absent_from_the_source_keeps_the_dtype_forge_produced(self):
        src = write_header(self.path("enc.safetensors"), {"a.weight": "BF16"})
        state = {
            "qwen3_06b.transformer.a.weight": FakeTensor(F16),
            "qwen3_06b.transformer.b.weight": FakeTensor(F16),
        }
        apply_component_precision(
            state, component("qwen3_06b", src), Slot("qwen3_06b.transformer."), DTYPES
        )
        self.assertIs(BF16, state["qwen3_06b.transformer.a.weight"].dtype)
        self.assertIs(F16, state["qwen3_06b.transformer.b.weight"].dtype)

    def test_the_count_of_changed_tensors_is_returned(self):
        src = write_header(
            self.path("enc.safetensors"), {"a.weight": "BF16", "b.weight": "BF16"}
        )
        state = {
            "qwen3_06b.transformer.a.weight": FakeTensor(F16),
            "qwen3_06b.transformer.b.weight": FakeTensor(BF16),
        }
        changed = apply_component_precision(
            state, component("qwen3_06b", src), Slot("qwen3_06b.transformer."), DTYPES
        )
        self.assertEqual(1, changed)

    def test_an_unreadable_source_leaves_everything_alone(self):
        state = {"qwen3_06b.transformer.a.weight": FakeTensor(F16)}
        changed = apply_component_precision(
            state,
            component("qwen3_06b", self.path("absent.safetensors")),
            Slot("qwen3_06b.transformer."),
            DTYPES,
        )
        self.assertEqual(0, changed)
        self.assertIs(F16, state["qwen3_06b.transformer.a.weight"].dtype)


class TheLlmAdapterBelongsToTheDiffusionModelTests(unittest.TestCase):
    """Forge moves it into the encoder's bucket at load. It came from Model A."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_the_adapter_is_untouched_by_an_external_encoder_selection(self):
        external = write_header(
            os.path.join(self.dir.name, "qwen.safetensors"),
            {"model.embed_tokens.weight": "F32", "proj.weight": "F32"},
        )
        state = {
            "qwen3_06b.transformer.model.embed_tokens.weight": FakeTensor(F16),
            # Sitting in the same bucket, belonging to the DiT.
            "llm_adapter.proj.weight": FakeTensor(BF16),
        }
        apply_component_precision(
            state,
            component("qwen3_06b", external),
            Slot("qwen3_06b.transformer."),
            DTYPES,
        )
        self.assertIs(F32, state["qwen3_06b.transformer.model.embed_tokens.weight"].dtype)
        self.assertIs(BF16, state["llm_adapter.proj.weight"].dtype)

    def test_an_explicit_format_does_not_reach_the_adapter_either(self):
        state = {
            "qwen3_06b.transformer.a.weight": FakeTensor(F32),
            "llm_adapter.proj.weight": FakeTensor(BF16),
        }
        apply_component_precision(
            state,
            component("qwen3_06b", "unused", output_format="fp16"),
            Slot("qwen3_06b.transformer."),
            DTYPES,
        )
        self.assertIs(F16, state["qwen3_06b.transformer.a.weight"].dtype)
        self.assertIs(BF16, state["llm_adapter.proj.weight"].dtype)

    def test_the_semantic_connector_is_out_of_reach_as_well(self):
        state = {
            "qwen3_06b.transformer.a.weight": FakeTensor(F32),
            "net.anima_v2_connector.quality_anchor.weight": FakeTensor(BF16),
        }
        apply_component_precision(
            state,
            component("qwen3_06b", "unused", output_format="fp16"),
            Slot("qwen3_06b.transformer."),
            DTYPES,
        )
        self.assertIs(BF16, state["net.anima_v2_connector.quality_anchor.weight"].dtype)


class ExplicitFormatTests(unittest.TestCase):
    def test_only_the_selected_slot_is_cast(self):
        state = {
            "clip_l.transformer.a.weight": FakeTensor(F32),
            "t5xxl.transformer.a.weight": FakeTensor(F32),
        }
        apply_component_precision(
            state,
            component("clip_l", "unused", output_format="bf16"),
            Slot("clip_l.transformer."),
            DTYPES,
        )
        self.assertIs(BF16, state["clip_l.transformer.a.weight"].dtype)
        self.assertIs(F32, state["t5xxl.transformer.a.weight"].dtype)

    def test_every_plain_format_is_honoured(self):
        for fmt, expected in (("fp16", F16), ("bf16", BF16), ("fp32", F32)):
            with self.subTest(output_format=fmt):
                state = {"vae.a.weight": FakeTensor(F8)}
                apply_component_precision(
                    state,
                    component("vae", "unused", output_format=fmt),
                    Slot("vae."),
                    DTYPES,
                )
                self.assertIs(expected, state["vae.a.weight"].dtype)

    def test_an_integer_tensor_is_never_cast(self):
        state = {"vae.quant.weight": FakeTensor(U8)}
        apply_component_precision(
            state,
            component("vae", "unused", output_format="fp16"),
            Slot("vae."),
            DTYPES,
        )
        self.assertIs(U8, state["vae.quant.weight"].dtype)


class ScaledComponentsAreCopiedBitExactTests(unittest.TestCase):
    """fp8_scaled is how Forge distributes most encoders. Converting one would
    mean dequantising, which is out of scope, so `same` means untouched."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.state = {
            "qwen3vl_4b.transformer.a.weight": FakeTensor(F8),
            "qwen3vl_4b.transformer.a.weight_scale": FakeTensor(F32),
            "qwen3vl_4b.transformer.a.weight_block": FakeTensor(U8),
        }

    def test_same_changes_nothing_at_all(self):
        src = write_header(
            os.path.join(self.dir.name, "vl.safetensors"),
            {"a.weight": "F8_E4M3", "a.weight_scale": "F32", "a.weight_block": "U8"},
        )
        changed = apply_component_precision(
            self.state,
            component("qwen3vl_4b", src, storage="fp8_scaled"),
            Slot("qwen3vl_4b.transformer."),
            DTYPES,
        )
        self.assertEqual(0, changed)
        self.assertIs(F8, self.state["qwen3vl_4b.transformer.a.weight"].dtype)
        self.assertIs(F32, self.state["qwen3vl_4b.transformer.a.weight_scale"].dtype)
        self.assertIs(U8, self.state["qwen3vl_4b.transformer.a.weight_block"].dtype)

    def test_a_scale_tensor_survives_an_explicit_format(self):
        """Casting the scales to fp16 would quietly wreck the encoder. Even
        though this combination is refused before the merge, the guard stays."""
        apply_component_precision(
            self.state,
            component("qwen3vl_4b", "unused", output_format="fp16"),
            Slot("qwen3vl_4b.transformer."),
            DTYPES,
        )
        self.assertIs(F32, self.state["qwen3vl_4b.transformer.a.weight_scale"].dtype)
        self.assertIs(U8, self.state["qwen3vl_4b.transformer.a.weight_block"].dtype)

    def test_a_second_level_scale_is_protected_too(self):
        state = {"q.transformer.a.weight_scale_2": FakeTensor(F32)}
        apply_component_precision(
            state,
            component("q", "unused", output_format="bf16"),
            Slot("q.transformer."),
            DTYPES,
        )
        self.assertIs(F32, state["q.transformer.a.weight_scale_2"].dtype)


class EmbeddedComponentTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_an_embedded_component_reads_the_checkpoint_it_came_from(self):
        primary = write_header(
            os.path.join(self.dir.name, "A.safetensors"),
            {"text_encoders.qwen3_06b.transformer.a.weight": "BF16"},
        )
        state = {"qwen3_06b.transformer.a.weight": FakeTensor(F16)}
        apply_component_precision(
            state,
            component("qwen3_06b", primary, source="embedded"),
            Slot("qwen3_06b.transformer."),
            DTYPES,
        )
        self.assertIs(BF16, state["qwen3_06b.transformer.a.weight"].dtype)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        from component_bundle import ComponentPlan
        from component_registry import SupportState

        self.plan = ComponentPlan(
            architecture_id="anima",
            support=SupportState.SUPPORTED,
            components=(
                component("qwen3_06b", "/models/qwen_3_06b_base.safetensors"),
                component("vae", "/models/qwen_image_vae.safetensors", "fp16"),
            ),
            slot_order=("qwen3_06b", "vae"),
        )

    def test_each_component_records_what_it_is_and_where_it_came_from(self):
        entry = component_provenance(self.plan)[0]
        self.assertEqual("qwen3_06b", entry["slot"])
        self.assertEqual("Qwen3 0.6B", entry["label"])
        self.assertEqual("qwen_3_06b_base.safetensors", entry["name"])
        self.assertEqual("file", entry["source"])
        self.assertEqual("qwen3_06b", entry["signature"])

    def test_the_requested_output_precision_is_recorded_separately(self):
        entries = component_provenance(self.plan)
        self.assertEqual("same", entries[0]["output_precision"])
        self.assertEqual("fp16", entries[1]["output_precision"])

    def test_only_the_basename_is_recorded_not_the_whole_path(self):
        for entry in component_provenance(self.plan):
            self.assertNotIn("/", entry["name"])

    def test_the_support_state_travels_with_the_provenance(self):
        self.assertTrue(all(e["support"] for e in component_provenance(self.plan)))

    def test_no_plan_means_no_provenance_rather_than_an_error(self):
        self.assertEqual([], component_provenance(None))

    def test_the_order_follows_the_plan(self):
        self.assertEqual(
            ["qwen3_06b", "vae"], [e["slot"] for e in component_provenance(self.plan)]
        )


if __name__ == "__main__":
    unittest.main()
