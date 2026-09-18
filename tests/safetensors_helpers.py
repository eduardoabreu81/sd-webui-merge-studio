"""Tiny deterministic safetensors files for tests.

A real encoder is several gigabytes, but everything the classifier looks at
lives in the header: tensor names, shapes and dtypes. These helpers write a
valid safetensors file whose payload is all zeros, so a fixture that reads like
a 4B parameter model costs a few kilobytes on disk.
"""

from __future__ import annotations

import json
import struct

#: Bytes per element, for the dtype codes safetensors writes in its header.
DTYPE_WIDTHS = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _nbytes(dtype: str, shape) -> int:
    try:
        width = DTYPE_WIDTHS[dtype]
    except KeyError:
        raise ValueError(f"Unknown safetensors dtype code: {dtype!r}") from None
    count = 1
    for dim in shape:
        count *= dim
    return count * width


def build_header(tensors, metadata=None) -> dict:
    """``{name: (dtype, shape)}`` -> a safetensors header dict.

    Offsets are laid out in the order given, which is what a real writer does
    and what any offset-sensitive reader expects.
    """
    header = {}
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}

    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = _nbytes(dtype, shape)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    return header


def write_safetensors_header(path, tensors, metadata=None) -> str:
    """Write a minimal valid safetensors file from ``{name: (dtype, shape)}``.

    The payload is zero bytes of the right length, so the file parses and its
    offsets are internally consistent without carrying real weights.
    """
    header = build_header(tensors, metadata)
    raw = json.dumps(header).encode("utf-8")
    payload = sum(
        _nbytes(dtype, shape) for dtype, shape in tensors.values()
    )

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(b"\0" * payload)
    return str(path)


# --- Component fixtures ----------------------------------------------------
#
# Shapes below were measured from the real files in a Forge Neo install, except
# where a comment says they come from the model's published config instead.


def qwen_encoder(layers, hidden, *, vision_blocks=0, vocab=151936, dtype="BF16"):
    """A Qwen-family causal LM, optionally with a vision tower.

    The vision tower is the only structural difference between Krea2's
    ``qwen3vl_4b`` and Z-Image's ``qwen3_4b``: both are 36 layers at hidden
    2560 with the same vocabulary.
    """
    tensors = {"model.embed_tokens.weight": (dtype, [vocab, hidden])}
    for i in range(layers):
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = (dtype, [hidden, hidden])
        tensors[f"model.layers.{i}.mlp.down_proj.weight"] = (dtype, [hidden, hidden])
        tensors[f"model.layers.{i}.input_layernorm.weight"] = (dtype, [hidden])
    for i in range(vision_blocks):
        tensors[f"model.visual.blocks.{i}.attn.qkv.weight"] = (dtype, [3072, 1024])
        tensors[f"model.visual.blocks.{i}.attn.proj.weight"] = (dtype, [1024, 1024])
    return tensors


def t5_encoder(*, vocab, d_model=4096, blocks=4, dtype="BF16"):
    """T5 and UMT5 share a layout; Forge tells them apart by vocabulary size."""
    tensors = {"shared.weight": (dtype, [vocab, d_model])}
    for i in range(blocks):
        tensors[f"encoder.block.{i}.layer.0.SelfAttention.k.weight"] = (
            dtype,
            [d_model, d_model],
        )
        tensors[f"encoder.block.{i}.layer.0.layer_norm.weight"] = (dtype, [d_model])
    return tensors


def clip_encoder(*, hidden, layers=12, vocab=49408, dtype="BF16"):
    """CLIP-L is hidden 768, CLIP-G is hidden 1280. From published configs."""
    tensors = {
        "text_model.embeddings.token_embedding.weight": (dtype, [vocab, hidden]),
        "text_model.final_layer_norm.weight": (dtype, [hidden]),
    }
    for i in range(layers):
        tensors[f"text_model.encoder.layers.{i}.self_attn.q_proj.weight"] = (
            dtype,
            [hidden, hidden],
        )
    return tensors


def vae_3d(*, latent_channels=16, dtype="F32"):
    """The Qwen-Image / Wan / Anima autoencoder.

    Measured: 194 tensors, 61 of them five-dimensional (3D convolution), four
    ``time_conv`` pairs, and no ``decoder.conv_in.weight`` at all -- its entry
    convolution is ``decoder.conv1.weight``. Any classifier that looks for
    ``decoder.conv_in.weight`` to read latent channels finds nothing here.
    """
    return {
        "conv1.weight": (dtype, [latent_channels * 2, latent_channels * 2, 1, 1, 1]),
        "conv2.weight": (dtype, [latent_channels, latent_channels, 1, 1, 1]),
        "encoder.conv1.weight": (dtype, [96, 3, 3, 3, 3]),
        "decoder.conv1.weight": (dtype, [384, latent_channels, 3, 3, 3]),
        "decoder.upsamples.3.time_conv.weight": (dtype, [768, 384, 3, 1, 1]),
        "encoder.downsamples.5.time_conv.weight": (dtype, [192, 192, 3, 1, 1]),
    }


def vae_2d(*, latent_channels, dtype="F32"):
    """The Flux AE (16 latent channels) and the SD/SDXL VAE (4)."""
    return {
        "encoder.conv_in.weight": (dtype, [128, 3, 3, 3]),
        "encoder.conv_out.weight": (dtype, [latent_channels * 2, 512, 3, 3]),
        "decoder.conv_in.weight": (dtype, [512, latent_channels, 3, 3]),
        "decoder.conv_out.weight": (dtype, [3, 128, 3, 3]),
    }


def as_fp8_scaled(tensors):
    """Rewrite a plain component the way a scaled FP8 release stores it.

    Measured on ``qwen3vl_4b_fp8_scaled``: weights in F8_E4M3, one F32
    ``weight_scale`` and one U8 block per quantised weight.
    """
    out = {}
    for name, (dtype, shape) in tensors.items():
        if name.endswith(".weight") and len(shape) == 2:
            out[name] = ("F8_E4M3", shape)
            out[name + "_scale"] = ("F32", [shape[0], 1])
            out[name.replace(".weight", ".weight_block")] = ("U8", [shape[0]])
        else:
            out[name] = (dtype, shape)
    return out


def as_fp8_mixed(tensors):
    """As ``as_fp8_scaled``, plus the second-level scale of a mixed release.

    Measured on ``qwen_3_4b_fp4_mixed``: both ``weight_scale`` and
    ``weight_scale_2`` are present, with roughly twice as many U8 blocks.
    """
    out = as_fp8_scaled(tensors)
    for name in list(out):
        if name.endswith("_scale"):
            out[name + "_2"] = ("F32", [1])
    return out
