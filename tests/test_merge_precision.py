"""FP32 merges keep the changes a BF16 round trip throws away.

Forge rounds a patched weight back to its own dtype, so the only way to keep a
delta smaller than half a BF16 step is to have the weight be float32 while it
is patched, and to make Forge's LoRA math float32 too (it is float16 on most
GPUs). These check the two helpers that do that, and that the merge tab
actually carries the choice through to `merge_checkpoints`.
"""

import os
import tempfile
import types
import unittest

import torch
from torch import nn

from merge_studio.merge_precision import (
    MERGE_PRECISION_CHOICES,
    MERGE_PRECISION_LABEL_TO_KEY,
    float32_source_storage,
    forced_lora_compute_dtype,
    forced_unet_storage_dtype,
    fp32_notes,
    upcast_to_float32,
)
from merge_studio.source_precision import diffusion_storage_code
from safetensors_helpers import write_safetensors_header

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


class UpcastTests(unittest.TestCase):
    def model(self):
        root = nn.Sequential(nn.Linear(8, 8), nn.Sequential(nn.Linear(8, 4)))
        root[0].to(torch.bfloat16)
        root[1][0].to(torch.float16)
        return root

    def test_every_floating_parameter_ends_up_float32(self):
        root = self.model()
        upcast_to_float32(root)
        self.assertEqual({p.dtype for p in root.parameters()}, {torch.float32})

    def test_the_count_is_the_parameters_that_changed(self):
        root = self.model()
        root[1][0].to(torch.float32)
        # weight + bias of the BF16 layer only.
        self.assertEqual(upcast_to_float32(root), 2)

    def test_values_survive_the_cast(self):
        root = self.model()
        before = root[0].weight.detach().float().clone()
        upcast_to_float32(root)
        self.assertTrue(torch.equal(root[0].weight.detach(), before))

    def test_a_delta_below_half_a_bf16_step_is_kept(self):
        # The point of the mode: the same tiny update that vanishes on a BF16
        # weight is still there on the upcast one.
        root = self.model()
        weight = root[0].weight
        delta = weight.detach().float().abs() * 2**-10
        on_bf16 = (weight.detach() + delta.to(torch.bfloat16)).to(torch.bfloat16)
        self.assertTrue(torch.equal(on_bf16, weight.detach()))

        upcast_to_float32(root)
        patched = root[0].weight.detach() + delta
        self.assertFalse(torch.equal(patched, root[0].weight.detach()))

    def test_integer_parameters_are_left_alone(self):
        module = nn.Module()
        module.codes = nn.Parameter(torch.zeros(4, dtype=torch.int8), requires_grad=False)
        self.assertEqual(upcast_to_float32(module), 0)
        self.assertEqual(module.codes.dtype, torch.int8)

    def test_skipped_parameters_keep_their_dtype(self):
        # How a quantized layer is kept out: `.float()` on a QuantizedTensor
        # would dequantize it, and Forge already patches those in float32.
        root = self.model()
        quantized = root[0].weight
        upcast_to_float32(root, skip=lambda p: p is quantized)
        self.assertEqual(root[0].weight.dtype, torch.bfloat16)
        self.assertEqual(root[0].bias.dtype, torch.float32)
        self.assertEqual(root[1][0].weight.dtype, torch.float32)


class NotesTests(unittest.TestCase):
    PLAIN = ("fp16", "bf16", "fp32")

    def test_a_quantized_base_is_allowed_and_explained(self):
        # Often the only build a user has: FP32 still runs, and says what it
        # cannot recover.
        notes = fp32_notes(True, "same", "Model A", self.PLAIN)
        self.assertEqual(len(notes), 1)
        self.assertIn("already quantized", notes[0])

    def test_a_quantized_base_into_another_quantized_format_warns_twice(self):
        notes = fp32_notes(True, "int8_tensorwise_convrot", "Model A", self.PLAIN)
        self.assertEqual(len(notes), 2)
        self.assertIn("second time", notes[1])

    def test_a_quantized_base_into_a_plain_format_has_no_second_quantization(self):
        self.assertEqual(len(fp32_notes(True, "fp32", "Model A", self.PLAIN)), 1)

    def test_a_plain_base_saved_as_same_is_told_it_rounds_back(self):
        notes = fp32_notes(False, "same", "Model A", self.PLAIN)
        self.assertEqual(len(notes), 1)
        self.assertIn("rounds away", notes[0])

    def test_an_f32_base_saved_as_same_keeps_float32(self):
        # Loaded as float32, 'Same as source' writes F32: nothing rounds away.
        self.assertEqual(fp32_notes(False, "same", "Model A", self.PLAIN, base_float32=True), [])

    def test_a_plain_base_into_fp32_or_a_quantized_format_needs_no_note(self):
        for output in ("fp32", "int8_tensorwise_convrot", "w6a8_int8"):
            self.assertEqual(fp32_notes(False, output, "Model A", self.PLAIN), [], output)


class ForcedComputeDtypeTests(unittest.TestCase):
    def fake_memory_management(self):
        return types.SimpleNamespace(lora_compute_dtype=lambda device: torch.float16)

    def test_every_device_gets_the_forced_dtype(self):
        mm = self.fake_memory_management()
        with forced_lora_compute_dtype(mm, torch.float32):
            self.assertIs(mm.lora_compute_dtype(torch.device("cuda")), torch.float32)
            self.assertIs(mm.lora_compute_dtype(torch.device("cpu")), torch.float32)

    def test_the_original_comes_back(self):
        mm = self.fake_memory_management()
        original = mm.lora_compute_dtype
        with forced_lora_compute_dtype(mm, torch.float32):
            pass
        self.assertIs(mm.lora_compute_dtype, original)

    def test_the_original_comes_back_after_a_failure(self):
        # A merge that dies halfway must not leave generation computing LoRAs
        # in float32 for the rest of the session.
        mm = self.fake_memory_management()
        original = mm.lora_compute_dtype
        with self.assertRaises(RuntimeError):
            with forced_lora_compute_dtype(mm, torch.float32):
                raise RuntimeError("out of memory")
        self.assertIs(mm.lora_compute_dtype, original)


class StorageCodeTests(unittest.TestCase):
    def write(self, tensors):
        handle = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return write_safetensors_header(handle.name, tensors)

    def test_an_fp32_merge_reads_as_f32(self):
        path = self.write({
            "model.diffusion_model.layers.0.proj.qkv.weight": ("F32", (64, 32)),
            "model.diffusion_model.layers.0.norm.weight": ("F32", (32,)),
        })
        self.assertEqual(diffusion_storage_code(path), "F32")

    def test_counted_by_element_not_by_tensor(self):
        # A BF16 build keeps many small F32 norms; the weights decide.
        tensors = {f"model.diffusion_model.layers.{i}.norm.weight": ("F32", (32,)) for i in range(8)}
        tensors["model.diffusion_model.layers.0.proj.qkv.weight"] = ("BF16", (64, 32))
        self.assertEqual(diffusion_storage_code(self.write(tensors)), "BF16")

    def test_only_the_diffusion_model_counts(self):
        # An AIO file whose text encoder outweighs its DiT.
        path = self.write({
            "model.diffusion_model.layers.0.proj.qkv.weight": ("F32", (16, 16)),
            "text_encoders.qwen3.transformer.embed.weight": ("BF16", (512, 64)),
            "vae.decoder.conv.weight": ("BF16", (64, 64)),
        })
        self.assertEqual(diffusion_storage_code(path), "F32")

    def test_a_bare_diffusion_file_counts(self):
        self.assertEqual(diffusion_storage_code(self.write({"layers.0.qkv.weight": ("F32", (8, 8))})), "F32")

    def test_an_unreadable_file_is_none(self):
        self.assertIsNone(diffusion_storage_code(os.path.join(ROOT, "missing.safetensors")))


class ForcedStorageDtypeTests(unittest.TestCase):
    def fake_dynamic_args(self, value=None):
        return types.SimpleNamespace(forge_unet_storage_dtype=value)

    def test_the_override_holds_inside_and_comes_back(self):
        # The user's own "Diffusion in Low Bits" choice survives the load.
        args = self.fake_dynamic_args(torch.float8_e4m3fn)
        with forced_unet_storage_dtype(args, torch.float32):
            self.assertIs(args.forge_unet_storage_dtype, torch.float32)
        self.assertIs(args.forge_unet_storage_dtype, torch.float8_e4m3fn)

    def test_the_original_comes_back_after_a_failure(self):
        args = self.fake_dynamic_args()
        with self.assertRaises(RuntimeError):
            with forced_unet_storage_dtype(args, torch.float32):
                raise RuntimeError("out of memory")
        self.assertIsNone(args.forge_unet_storage_dtype)

    def test_only_an_f32_file_is_loaded_as_float32(self):
        handle = tempfile.TemporaryDirectory()
        self.addCleanup(handle.cleanup)
        f32 = write_safetensors_header(os.path.join(handle.name, "a.safetensors"), {"layers.0.w": ("F32", (4, 4))})
        bf16 = write_safetensors_header(os.path.join(handle.name, "b.safetensors"), {"layers.0.w": ("BF16", (4, 4))})
        args = self.fake_dynamic_args()
        with float32_source_storage(args, f32):
            self.assertIs(args.forge_unet_storage_dtype, torch.float32)
        with float32_source_storage(args, bf16):
            self.assertIsNone(args.forge_unet_storage_dtype)


class ChoiceTests(unittest.TestCase):
    def test_native_is_the_default(self):
        # Every recipe saved before this option loads with the radio at its
        # first choice; that has to be the behaviour those recipes ran with.
        self.assertEqual(MERGE_PRECISION_CHOICES[0][1], "native")

    def test_both_keys_are_offered(self):
        self.assertEqual(set(MERGE_PRECISION_LABEL_TO_KEY.values()), {"native", "fp32"})


class WiringTests(unittest.TestCase):
    def test_the_merge_tab_forwards_the_choice(self):
        ui = source("scripts", "merge_studio_ui.py")
        call = ui[ui.index("checkpoint_merge.merge_checkpoints("):]
        call = call[: call.index("\n        )")]
        self.assertIn("merge_precision=", call)

    def test_the_merge_patches_inside_the_forced_dtype(self):
        merge = source("merge_studio", "checkpoint_merge.py")
        block = merge[merge.index("forced_lora_compute_dtype(memory_management"):]
        self.assertIn("unet.patch_model(", block[:400])

    def test_a_quantized_base_is_never_refused(self):
        for path in (("merge_studio", "checkpoint_merge.py"), ("merge_studio", "lora_bake.py")):
            text = source(*path)
            self.assertNotIn("FP32 merge needs a plain", text, path)
            self.assertIn("skip=is_quantized_tensor", text, path)

    def test_quantizing_reads_the_float32_result(self):
        for path in (("merge_studio", "checkpoint_merge.py"), ("merge_studio", "lora_bake.py")):
            self.assertIn("quant_input_dtype=torch.float32 if fp32", source(*path), path)

    def test_every_loader_keeps_an_f32_file_float32(self):
        quantize = source("merge_studio", "checkpoint_quantize.py")
        self.assertIn("float32_source_storage(backend.args.dynamic_args, checkpoint_path)", quantize)
        self.assertIn("quant_input_dtype=torch.float32 if float32_source", quantize)
        bake = source("merge_studio", "lora_bake.py")
        self.assertIn("float32_source_storage(backend.args.dynamic_args, checkpoint_path) if fp32", bake)
        merge = source("merge_studio", "checkpoint_merge.py")
        self.assertEqual(merge.count("keep_float32=fp32"), 3)
        self.assertIn("float32_source_storage(backend.args.dynamic_args, primary_info.filename) if fp32", merge)

    def test_fp32_is_a_plain_output_format(self):
        quant = source("merge_studio", "quant_utils.py")
        self.assertIn('"fp32": torch.float32', quant)
        self.assertIn('"fp32"),', quant[quant.index("OUTPUT_FORMAT_CHOICES = ["):])


if __name__ == "__main__":
    unittest.main()
