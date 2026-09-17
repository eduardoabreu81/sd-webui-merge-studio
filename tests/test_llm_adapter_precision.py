"""The LLM adapter takes the DiT's precision, never the text encoder's.

Forge's loader moves Anima's LLMAdapter into the text encoder at load time
(backend/loader.py::process_anima), but on disk it is part of the DiT. That
made the "Text encoder format" dropdown silently govern it. circlestone-labs'
README is explicit that the adapter "has an outsized influence on the generated
images" and "is easy to degrade", and the community's own INT8 release agrees in
practice: Anima-2.9B-preview-v1_int8_convrot quantizes 640 layers and leaves all
118 adapter tensors in plain BF16.

Measured on anima_baseV10/anima_turboV11: 118 adapter keys, 134.7M params, of
which 61 Linears (101.7M params) pass the quantizer's own eligibility filter.
"""

import unittest

from precision_stats import dominant_float_dtype, match_dtype


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
F8 = FakeDtype("F8_E4M3")
I8 = FakeDtype("I8", is_floating_point=False)

ADAPTER_KEY = "model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight"


def assemble(processed_unet, adapter):
    """The rule the merge applies once the state dict is assembled."""
    sd = dict(processed_unet)
    sd.update(adapter)
    dit_dtype = dominant_float_dtype(processed_unet)
    for key in adapter:
        sd[key] = match_dtype(sd[key], dit_dtype)
    return sd


class AdapterFollowsTheDitTests(unittest.TestCase):
    def test_encoder_precision_does_not_leak_into_the_adapter(self):
        # DiT kept at source precision, encoder converted to FP8: before, the
        # adapter came back carrying the encoder's dtype.
        unet = {"model.diffusion_model.blocks.0.mlp.weight": FakeTensor(BF16, 4325376)}
        sd = assemble(unet, {ADAPTER_KEY: FakeTensor(F8, 1048576)})

        self.assertEqual(BF16, sd[ADAPTER_KEY].dtype)

    def test_adapter_follows_an_explicitly_chosen_dit_format(self):
        # DiT converted to FP16, encoder left alone: the adapter goes with the DiT.
        unet = {"model.diffusion_model.blocks.0.mlp.weight": FakeTensor(FakeDtype("F16"), 4325376)}
        sd = assemble(unet, {ADAPTER_KEY: FakeTensor(BF16, 1048576)})

        self.assertEqual(FakeDtype("F16"), sd[ADAPTER_KEY].dtype)

    def test_a_quantized_dit_leaves_the_adapter_in_plain_precision(self):
        # What Anima-2.9B-preview-v1_int8_convrot does: quantized weights, F32
        # scales, and an adapter left in BF16. The adapter must land on BF16 --
        # not on the scales' F32, and not quantized.
        unet = {}
        for i in range(640):
            unet[f"model.diffusion_model.blocks.{i}.mlp.weight"] = FakeTensor(I8, 4325376)
            unet[f"model.diffusion_model.blocks.{i}.mlp.weight_scale"] = FakeTensor(F32, 2864)
        for i in range(288):
            unet[f"model.diffusion_model.blocks.{i}.norm.weight"] = FakeTensor(BF16, 532316)

        sd = assemble(unet, {ADAPTER_KEY: FakeTensor(BF16, 1048576)})

        self.assertEqual(BF16, sd[ADAPTER_KEY].dtype)

    def test_an_already_quantized_adapter_tensor_is_not_reinterpreted(self):
        unet = {"model.diffusion_model.blocks.0.mlp.weight": FakeTensor(BF16, 4325376)}
        quantized = FakeTensor(I8, 1048576)
        sd = assemble(unet, {ADAPTER_KEY: quantized})

        self.assertIs(quantized, sd[ADAPTER_KEY])

    def test_a_dit_with_no_floating_tensors_leaves_the_adapter_alone(self):
        unet = {"model.diffusion_model.blocks.0.mlp.weight": FakeTensor(I8, 4325376)}
        adapter_tensor = FakeTensor(BF16, 1048576)
        sd = assemble(unet, {ADAPTER_KEY: adapter_tensor})

        self.assertIs(adapter_tensor, sd[ADAPTER_KEY])


if __name__ == "__main__":
    unittest.main()
