"""Anima ships in three depths, and the badge used to hide which one you had.

28 blocks is circlestone-labs/Anima, 40 is Anima-2.9B, 52 is Anima-3.8B. The
depth decides whether two checkpoints can be merged at all, and which one has to
be Primary Model (A) -- the inspector already read it to guard merge order, but
displayed all three as plain "Anima (DiT)".
"""

import unittest

from checkpoint_inspector import _architecture_label


class ArchitectureLabelTests(unittest.TestCase):
    def test_each_anima_generation_is_named(self):
        # The depth stays on the badge beside the name: it is the number the
        # merge-order warning quotes and the bound a per-block weight rule has
        # to stay inside. 28 is the base release and has no suffix.
        for blocks, expected in ((28, "Anima (DiT), 28-block"),
                                 (40, "Anima 2.9B (DiT), 40-block"),
                                 (52, "Anima 3.8B (DiT), 52-block")):
            with self.subTest(blocks=blocks):
                label = _architecture_label(
                    {"architecture": "Anima (DiT)", "block_count": blocks}
                )
                self.assertEqual(expected, label)

    def test_other_architectures_are_left_alone(self):
        # Block counts mean something else outside the Anima expansion line, so
        # they are not advertised as a generation.
        for arch in ("SDXL", "Flux.1", "SD 1.5", "Wan2.1 (DiT)", "DiT / Diffusion Model"):
            with self.subTest(arch=arch):
                self.assertEqual(
                    arch, _architecture_label({"architecture": arch, "block_count": 40})
                )

    def test_anima_without_a_readable_block_count_falls_back(self):
        self.assertEqual(
            "Anima (DiT)", _architecture_label({"architecture": "Anima (DiT)"})
        )
        self.assertEqual(
            "Anima (DiT)",
            _architecture_label({"architecture": "Anima (DiT)", "block_count": None}),
        )

    def test_an_unknown_depth_keeps_the_count_without_inventing_a_name(self):
        # Anima could ship a fourth depth tomorrow. Printing the number it
        # actually has beats printing a generation it does not.
        self.assertEqual(
            "Anima (DiT), 36-block",
            _architecture_label({"architecture": "Anima (DiT)", "block_count": 36}),
        )

    def test_missing_architecture_does_not_crash_the_badge(self):
        self.assertEqual("Unknown", _architecture_label({}))


if __name__ == "__main__":
    unittest.main()
