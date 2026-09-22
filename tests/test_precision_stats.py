"""A quantized checkpoint's scales must not outvote its weights.

The numbers here are the ones measured in
Anima-2.9B-preview-v1_int8_convrot.safetensors: 640 F32 scale tensors holding
1.8M elements between them, against 288 BF16 tensors holding 153.3M. Counting
tensors reads that as F32; counting elements reads it as BF16, which is what the
model's unquantized weights are actually in.
"""

import unittest

from merge_studio.precision_stats import dominant_float_dtype, match_dtype


class FakeDtype:
    def __init__(self, name, is_floating_point=True):
        self.name = name
        self.is_floating_point = is_floating_point

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, FakeDtype) and other.name == self.name

    def __hash__(self):
        return hash(self.name)


class FakeTensor:
    def __init__(self, dtype, elements=1):
        self.dtype = dtype
        self._elements = elements

    def numel(self):
        return self._elements

    def to(self, dtype):
        return FakeTensor(dtype, self._elements)


BF16 = FakeDtype("BF16")
F32 = FakeDtype("F32")
I8 = FakeDtype("I8", is_floating_point=False)


class DominantFloatDtypeTests(unittest.TestCase):
    def _int8_build(self):
        sd = {}
        # 640 per-channel scales, ~2864 elements each -> 1.8M total.
        for i in range(640):
            sd[f"blocks.{i}.weight_scale"] = FakeTensor(F32, 2864)
            sd[f"blocks.{i}.weight"] = FakeTensor(I8, 4325376)
        # 288 plain tensors, but they carry 153.3M elements.
        for i in range(288):
            sd[f"llm_adapter.blocks.{i}.weight"] = FakeTensor(BF16, 532316)
        return sd

    def test_scales_do_not_outvote_weights_they_describe(self):
        self.assertEqual(BF16, dominant_float_dtype(self._int8_build()))

    def test_tensor_count_alone_would_have_answered_f32(self):
        sd = self._int8_build()
        counts = {}
        for tensor in sd.values():
            if tensor.dtype.is_floating_point:
                counts[tensor.dtype] = counts.get(tensor.dtype, 0) + 1
        # Guards the premise: the two rules really do disagree on this shape.
        self.assertEqual(F32, max(counts, key=counts.get))

    def test_quantized_payloads_are_not_a_precision(self):
        sd = {"w": FakeTensor(I8, 10**9), "n": FakeTensor(BF16, 1)}
        self.assertEqual(BF16, dominant_float_dtype(sd))

    def test_returns_none_when_nothing_floats(self):
        self.assertIsNone(dominant_float_dtype({"w": FakeTensor(I8, 5)}))
        self.assertIsNone(dominant_float_dtype({}))

    def test_falls_back_to_shape_when_numel_is_unavailable(self):
        class ShapeOnly:
            def __init__(self, dtype, shape):
                self.dtype = dtype
                self.shape = shape

        sd = {"big": ShapeOnly(BF16, (1024, 1024)), "small": ShapeOnly(F32, (8,))}
        self.assertEqual(BF16, dominant_float_dtype(sd))


class MatchDtypeTests(unittest.TestCase):
    def test_casts_a_floating_tensor(self):
        self.assertEqual(BF16, match_dtype(FakeTensor(F32, 4), BF16).dtype)

    def test_leaves_a_quantized_payload_alone(self):
        tensor = FakeTensor(I8, 4)
        self.assertIs(tensor, match_dtype(tensor, BF16))

    def test_leaves_a_tensor_already_in_the_target_dtype(self):
        tensor = FakeTensor(BF16, 4)
        self.assertIs(tensor, match_dtype(tensor, BF16))

    def test_no_target_dtype_is_not_a_cast(self):
        tensor = FakeTensor(F32, 4)
        self.assertIs(tensor, match_dtype(tensor, None))

    def test_a_cast_that_fails_keeps_the_original(self):
        class Unconvertible(FakeTensor):
            def to(self, dtype):
                raise RuntimeError("no such conversion")

        tensor = Unconvertible(F32, 4)
        self.assertIs(tensor, match_dtype(tensor, BF16))


if __name__ == "__main__":
    unittest.main()
