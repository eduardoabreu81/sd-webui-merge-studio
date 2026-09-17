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
        for blocks, expected in ((28, "Anima (DiT), 28-block"),
                                 (40, "Anima (DiT), 40-block"),
                                 (52, "Anima (DiT), 52-block")):
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

    def test_missing_architecture_does_not_crash_the_badge(self):
        self.assertEqual("Unknown", _architecture_label({}))


if __name__ == "__main__":
    unittest.main()
