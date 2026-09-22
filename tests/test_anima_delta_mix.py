"""The delta write rule for Anima's inserted blocks.

A cross-generation merge has always had two kinds of block: the ones both
models own, which merge name-for-name once `anima_remap` corrects the indices,
and the ones the newer generation inserted, which have no counterpart at all.
This file is about the second kind.

Until now the only thing on offer for an inserted block was `extend_ratio`: a
lerp towards the older model's block it was deep-copied from at init. The
ComfyUI Anima Delta Mix node pack argues that is the wrong shape of answer --
the donor block sits at a different depth, so averaging the two writes a
foreign layer's absolute weights into the niche. Its rule instead writes the
donor's *displacement* from the floor Model A inherited alongside the insert:

    insert += r * (donor[src] - kept[src])

What is pinned down here is that mapping (which floor is `kept`), the ordering
it depends on, and the arithmetic. The merge loop that calls it lives in
`checkpoint_merge`, which cannot be imported on a machine without torch -- the
same reason `blend_tensors` lives in `merge_modes`, and why `delta_write` was
put there beside it rather than next to its caller.
"""

import unittest

from merge_studio import anima_remap
from merge_studio.merge_modes import delta_write


class Num:
    """A one-element stand-in for a tensor, as in `test_merge_modes`.

    `delta_write` only ever does `+`, `-` and `*`, plus a shape comparison.
    """

    def __init__(self, value: float, shape=(1,)):
        self.value = float(value)
        self.shape = shape

    def __add__(self, other):
        return Num(self.value + _value(other), self.shape)

    def __sub__(self, other):
        return Num(self.value - _value(other), self.shape)

    def __mul__(self, other):
        return Num(self.value * _value(other), self.shape)

    __rmul__ = __mul__

    def __repr__(self):
        return f"Num({self.value:g})"


def _value(x):
    return x.value if isinstance(x, Num) else float(x)


def _maps(src_blocks: int, dst_blocks: int):
    mapping = anima_remap._FALLBACK_TARGET_TO_SOURCE[(src_blocks, dst_blocks)]
    return anima_remap.split_frozen_inserted(mapping)


class KeptFloorTests(unittest.TestCase):
    """Which block the rule subtracts."""

    def test_every_insert_has_exactly_one_floor(self):
        # An inserted block is by definition a repeat of a source some frozen
        # block already claimed, so a niche without a floor would mean the
        # split itself is wrong.
        for pair in ((28, 40), (28, 52), (40, 52)):
            with self.subTest(pair=pair):
                frozen, inserted = _maps(*pair)
                kept = anima_remap.kept_target_for_insert(frozen, inserted)
                self.assertEqual(len(kept), len(inserted))

    def test_the_floor_shares_the_insert_source(self):
        frozen, inserted = _maps(28, 52)
        kept = anima_remap.kept_target_for_insert(frozen, inserted)
        for target, floor in kept.items():
            self.assertEqual(inserted[target], frozen[floor])

    def test_the_floor_is_itself_a_frozen_block(self):
        # Subtracting another niche would measure the donor against something
        # the donor never saw.
        frozen, inserted = _maps(28, 52)
        kept = anima_remap.kept_target_for_insert(frozen, inserted)
        for floor in kept.values():
            self.assertIn(floor, frozen)
            self.assertNotIn(floor, inserted)

    def test_the_floor_comes_first(self):
        # `_merge_module_tree` walks blocks in order and writes them in place,
        # so a delta read from inside block N sees the merged stem only for
        # floors below N. `checkpoint_merge` refuses the merge when this
        # fails; it holds for every published Anima mapping.
        for pair in ((28, 40), (28, 52), (40, 52)):
            with self.subTest(pair=pair):
                frozen, inserted = _maps(*pair)
                kept = anima_remap.kept_target_for_insert(frozen, inserted)
                self.assertTrue(anima_remap.kept_precedes_insert(kept))

    def test_a_reordered_mapping_is_caught(self):
        # The guard has to actually fire, or it documents an assumption
        # without defending it.
        self.assertFalse(anima_remap.kept_precedes_insert({3: 7}))


class KeptTranslatorTests(unittest.TestCase):
    """Turning that mapping into module names."""

    def setUp(self):
        self.translate = anima_remap.make_kept_translator(*_maps(28, 52))

    def test_an_insert_resolves_to_its_floor(self):
        # 28 and 27 are both copies of source 14, whose kept floor is 26.
        self.assertEqual(
            self.translate("blocks.28.attn.qkv.weight"), "blocks.26.attn.qkv.weight"
        )
        self.assertEqual(
            self.translate("blocks.27.mlp.fc1.weight"), "blocks.26.mlp.fc1.weight"
        )

    def test_a_frozen_block_has_no_floor_to_subtract(self):
        # It merges against Model B directly; the delta rule never sees it.
        self.assertIsNone(self.translate("blocks.26.attn.qkv.weight"))

    def test_the_submodule_stacks_are_left_alone(self):
        # Same anchoring `make_name_translator` relies on: the llm_adapter and
        # the v1.1 connector have their own indexed blocks that the 28/40/52
        # expansion never touched.
        self.assertIsNone(self.translate("llm_adapter.blocks.3.attn.qkv.weight"))
        self.assertIsNone(
            self.translate("anima_v2_connector.semantic_resampler.blocks.2.x.weight")
        )

    def test_both_sides_are_model_a(self):
        # The floor is read from the chassis, not from the donor file -- the
        # rule compares the donor against what Model A already has at that
        # address. A name that came back pointing into B's numbering would be
        # a different rule wearing this one's name.
        frozen, inserted = _maps(28, 52)
        kept = anima_remap.kept_target_for_insert(frozen, inserted)
        for target, floor in kept.items():
            name = self.translate(f"blocks.{target}.attn.qkv.weight")
            self.assertEqual(name, f"blocks.{floor}.attn.qkv.weight")


class DeltaWriteTests(unittest.TestCase):
    """The arithmetic."""

    def test_it_writes_the_donor_displacement(self):
        # insert 10, donor 7, floor 5 -> the donor sits 2 above its floor, and
        # at r=1 the insert ends 2 above where it started.
        out = delta_write(Num(10), Num(7), Num(5), 1.0)
        self.assertEqual(out.value, 12)

    def test_the_ratio_scales_the_displacement_not_the_donor(self):
        out = delta_write(Num(10), Num(7), Num(5), 0.35)
        self.assertAlmostEqual(out.value, 10 + 0.35 * 2)

    def test_a_donor_equal_to_its_floor_writes_nothing(self):
        # This is the property the node pack's "unique skip" exploits: a donor
        # that has not moved away from its floor has nothing to contribute,
        # and the rule already returns the insert untouched.
        out = delta_write(Num(10), Num(5), Num(5), 1.0)
        self.assertEqual(out.value, 10)

    def test_it_is_not_a_lerp(self):
        # The distinction the whole rule exists for. A lerp at r=1 would hand
        # back the donor's absolute weights (7); the delta keeps A's own
        # magnitude and moves it by what the donor moved.
        out = delta_write(Num(10), Num(7), Num(5), 1.0)
        self.assertNotEqual(out.value, 7)

    def test_a_negative_displacement_subtracts(self):
        out = delta_write(Num(10), Num(3), Num(5), 1.0)
        self.assertEqual(out.value, 8)

    def test_a_missing_floor_refuses(self):
        # Not "subtract zero": that would write the donor's absolute weights,
        # which is the failure the rule exists to prevent. The caller leaves
        # the block at A's weights instead.
        self.assertIsNone(delta_write(Num(10), Num(7), None, 1.0))

    def test_a_mismatched_floor_refuses(self):
        self.assertIsNone(
            delta_write(Num(10, (4, 4)), Num(7, (4, 4)), Num(5, (8, 8)), 1.0)
        )

    def test_it_is_not_registered_as_a_merge_mode(self):
        # A mode answers how two whole checkpoints blend. This answers what may
        # be written into a block only one of them has, and it is reached only
        # through the cross-generation path -- putting it in the mode dropdown
        # would offer it for merges that have no inserted blocks at all.
        from merge_studio import merge_modes

        self.assertNotIn(
            "delta", [m.key for m in merge_modes.MERGE_MODES]
        )


class ExtendRuleNameTests(unittest.TestCase):
    def test_blend_stays_the_default(self):
        # Every recipe saved before this existed merged with the lerp, and
        # loads with the radio at its default. Reordering these would silently
        # re-interpret those recipes.
        self.assertEqual(anima_remap.EXTEND_RULES[0], anima_remap.EXTEND_RULE_BLEND)

    def test_the_rules_are_distinct(self):
        self.assertEqual(len(set(anima_remap.EXTEND_RULES)), 2)


if __name__ == "__main__":
    unittest.main()
