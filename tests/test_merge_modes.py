"""One table defines every merge mode, and everything else reads from it.

The merge loop asks which models a mode needs, the interface asks which fields
to show, and the inspector asks what to call it. Answering those separately is
how a Sum Twice merge came to display as the name of the tool that wrote it --
so what is pinned down here is the table itself and the arithmetic that hangs
off it.

`merge_modes` is importable on this side; `checkpoint_merge` is not, because it
pulls in torch, Forge's backend and the WebUI's `modules` at module scope. That
is exactly why the modes and the blend live in their own module.
"""

import unittest

import numpy as np

from merge_studio.merge_modes import (
    INTERP_ADD_DIFFERENCE,
    INTERP_DARE,
    INTERP_NO_INTERPOLATION,
    INTERP_SIMILARITY_ADD_DIFFERENCE,
    INTERP_SUM_TWICE,
    INTERP_WEIGHTED_SUM,
    MERGE_MODES,
    MERGE_MODES_BY_KEY,
    MergeError,
    blend_tensors,
    make_seeded_rand,
    merge_mode,
)
from merge_studio.checkpoint_inspector import _describe_merge_math


class Num:
    """A one-element stand-in for a tensor.

    `blend_tensors` only ever does `lerp`, `+`, `-` and `*` on what it is
    handed, which is what lets the formulas be checked without torch.
    """

    def __init__(self, value: float):
        self.value = float(value)

    def lerp(self, other: "Num", weight: float) -> "Num":
        return Num(self.value + weight * (other.value - self.value))

    def __add__(self, other):
        return Num(self.value + _value(other))

    def __sub__(self, other):
        return Num(self.value - _value(other))

    def __mul__(self, other):
        return Num(self.value * _value(other))

    __rmul__ = __mul__

    def __repr__(self):
        return f"Num({self.value:g})"


def _value(x):
    return x.value if isinstance(x, Num) else float(x)


class TheTableTests(unittest.TestCase):
    def test_every_mode_is_fully_described(self):
        # These four strings are what the picker, the panel and the inspector
        # render. A mode added with any of them blank shows up as a gap.
        for mode in MERGE_MODES:
            with self.subTest(mode=mode.key):
                self.assertTrue(mode.key)
                self.assertTrue(mode.label)
                self.assertTrue(mode.formula)
                self.assertTrue(mode.description)

    def test_keys_and_labels_are_both_unique(self):
        self.assertEqual(len(MERGE_MODES), len({m.key for m in MERGE_MODES}))
        self.assertEqual(len(MERGE_MODES), len({m.label for m in MERGE_MODES}))

    def test_a_mode_needing_c_also_needs_b(self):
        # The interface shows Model C's column on needs_c and Model B's on
        # needs_b. A mode needing C but not B would render a gap where B is.
        for mode in MERGE_MODES:
            with self.subTest(mode=mode.key):
                if mode.needs_c:
                    self.assertTrue(mode.needs_b)

    def test_a_mode_with_a_beta_labels_its_slider(self):
        for mode in MERGE_MODES:
            with self.subTest(mode=mode.key):
                if mode.needs_beta:
                    self.assertTrue(mode.beta_label)

    def test_the_lookup_names_what_it_was_asked_for(self):
        with self.assertRaises(MergeError) as caught:
            merge_mode("triple_sum")
        self.assertIn("triple_sum", str(caught.exception))

    def test_sum_twice_needs_three_models_and_two_ratios(self):
        mode = MERGE_MODES_BY_KEY[INTERP_SUM_TWICE]
        self.assertTrue(mode.needs_b)
        self.assertTrue(mode.needs_c)
        self.assertTrue(mode.needs_beta)

    def test_no_interpolation_needs_nothing_but_a(self):
        mode = MERGE_MODES_BY_KEY[INTERP_NO_INTERPOLATION]
        self.assertFalse(mode.needs_b)
        self.assertFalse(mode.needs_c)
        self.assertFalse(mode.needs_beta)


class BlendTests(unittest.TestCase):
    def blend(self, method, alpha, beta, a, b, c=None):
        result = blend_tensors(
            method, alpha, beta, Num(a), Num(b), None if c is None else Num(c)
        )
        return None if result is None else result.value

    def test_weighted_sum_walks_from_a_to_b(self):
        for alpha, expected in ((0.0, 2.0), (0.5, 4.0), (1.0, 6.0)):
            with self.subTest(alpha=alpha):
                self.assertAlmostEqual(
                    expected, self.blend(INTERP_WEIGHTED_SUM, alpha, 0.0, 2.0, 6.0)
                )

    def test_add_difference_adds_what_b_gained_over_c(self):
        # A + alpha(B - C): 10 + 0.5 * (8 - 6)
        self.assertAlmostEqual(
            11.0, self.blend(INTERP_ADD_DIFFERENCE, 0.5, 0.0, 10.0, 8.0, 6.0)
        )

    def test_sum_twice_matches_the_published_formula(self):
        # (1 - beta)((1 - alpha)A + alpha B) + beta C, written out longhand
        # rather than as the two lerps the implementation uses, so the test
        # would catch the two drifting apart.
        a, b, c, alpha, beta = 1.0, 5.0, 9.0, 0.25, 0.4
        expected = (1 - beta) * ((1 - alpha) * a + alpha * b) + beta * c
        self.assertAlmostEqual(
            expected, self.blend(INTERP_SUM_TWICE, alpha, beta, a, b, c)
        )

    def test_sum_twice_at_beta_zero_is_a_weighted_sum(self):
        # The inner blend is exactly Weighted Sum, which is what makes beta
        # readable as "how much of Model C".
        self.assertAlmostEqual(
            self.blend(INTERP_WEIGHTED_SUM, 0.3, 0.0, 1.0, 5.0),
            self.blend(INTERP_SUM_TWICE, 0.3, 0.0, 1.0, 5.0, 9.0),
        )

    def test_sum_twice_at_beta_one_is_model_c(self):
        self.assertAlmostEqual(
            9.0, self.blend(INTERP_SUM_TWICE, 0.3, 1.0, 1.0, 5.0, 9.0)
        )

    def test_a_missing_third_model_skips_rather_than_merging_half(self):
        # A module present in A and B but absent from C. Merging it on two
        # terms when the mode has three would silently apply a different
        # formula to that one tensor.
        for method in (INTERP_ADD_DIFFERENCE, INTERP_SUM_TWICE):
            with self.subTest(method=method):
                self.assertIsNone(self.blend(method, 0.5, 0.5, 1.0, 2.0, None))

    def test_weighted_sum_does_not_care_that_c_is_absent(self):
        self.assertAlmostEqual(
            1.5, self.blend(INTERP_WEIGHTED_SUM, 0.5, 0.0, 1.0, 2.0, None)
        )

    def test_no_interpolation_has_no_arithmetic_and_says_so(self):
        # The merge loop skips these modules before it gets here, so reaching
        # this is a caller bug and not something to paper over with a default.
        with self.assertRaises(MergeError):
            self.blend(INTERP_NO_INTERPOLATION, 0.5, 0.0, 1.0, 2.0)

    def test_an_unknown_method_is_named_in_the_error(self):
        with self.assertRaises(MergeError) as caught:
            self.blend("tensor_sum", 0.5, 0.0, 1.0, 2.0)
        self.assertIn("tensor_sum", str(caught.exception))


class SimilarityAddDifferenceTests(unittest.TestCase):
    """Ported from `s1dlx/meh` (MIT, Copyright (c) 2023 s1dlx).

    Add Difference that holds back where A and B already agree. Real arrays
    here rather than the scalar stand-in, because the formula reads each
    element against its neighbours' magnitudes.
    """

    def blend(self, a, b, c, alpha, beta):
        return blend_tensors(
            INTERP_SIMILARITY_ADD_DIFFERENCE,
            alpha,
            beta,
            np.array(a, dtype=float),
            np.array(b, dtype=float),
            np.array(c, dtype=float),
            xp=np,
        )

    def test_at_beta_zero_it_is_exactly_add_difference(self):
        # beta scales the similarity term to nothing, so the mode collapses
        # onto the one it is built from. A reader who sets beta to 0 and sees
        # a different answer has found a bug.
        a, b, c = [1.0, 2.0, 3.0], [2.0, 0.0, 3.0], [1.0, 1.0, 1.0]
        plain = np.array(a) + 0.5 * (np.array(b) - np.array(c))
        np.testing.assert_allclose(plain, self.blend(a, b, c, 0.5, 0.0))

    def test_where_the_models_agree_it_leans_on_the_average(self):
        # a == b is perfect agreement, so similarity is beta everywhere and
        # the result is beta of the way from Add Difference towards A itself
        # (ab_sum with a == b is just a).
        a = b = [1.0, 2.0, 3.0]
        c = [0.0, 0.0, 0.0]
        alpha, beta = 0.5, 1.0
        np.testing.assert_allclose(
            np.array(a), self.blend(a, b, c, alpha, beta)
        )

    def test_two_zeros_agree_rather_than_producing_nan(self):
        # `threshold` is zero there, so the similarity is 0/0. Upstream sends
        # that to beta, which is the right reading: two zeros agree.
        result = self.blend([0.0, 1.0], [0.0, -1.0], [0.0, 0.0], 0.5, 0.5)
        self.assertFalse(np.isnan(result).any(), result)

    def test_it_needs_a_third_model(self):
        self.assertIsNone(
            blend_tensors(
                INTERP_SIMILARITY_ADD_DIFFERENCE,
                0.5, 0.5, np.array([1.0]), np.array([2.0]), None, xp=np,
            )
        )


class DareTests(unittest.TestCase):
    """Written from arXiv:2311.03099, not from the reference implementation.

    The two disagree. The paper drops with probability p and rescales the
    survivors by 1/(1-p); `martyn/safetensors-merge-supermario` builds its
    mask with `binomial(1, p)` -- which *keeps* with probability p -- and
    rescales by 1/(1-p) anyway, which is only unbiased at p = 0.5. These pin
    down the paper's reading, so beta is the drop rate.
    """

    def blend(self, a, b, alpha, beta):
        return blend_tensors(
            INTERP_DARE,
            alpha,
            beta,
            np.array(a, dtype=float),
            np.array(b, dtype=float),
            xp=np,
        )

    def test_dropping_nothing_is_a_plain_difference(self):
        a, b = [1.0, 2.0, 3.0], [5.0, 0.0, 3.0]
        np.testing.assert_allclose(
            np.array(a) + 0.5 * (np.array(b) - np.array(a)),
            self.blend(a, b, 0.5, 0.0),
        )

    def test_dropping_everything_leaves_model_a_untouched(self):
        # And does not divide by zero on the way.
        a = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(np.array(a), self.blend(a, [9.0, 9.0, 9.0], 1.0, 1.0))

    def test_every_element_is_either_kept_whole_or_dropped(self):
        # The mask is Bernoulli, not a soft weight: an element is A, or it is
        # A plus the whole rescaled difference. Anything between means the
        # mask stopped being 0/1.
        np.random.seed(7)
        a, b, beta = np.zeros(2000), np.ones(2000), 0.75
        result = self.blend(a, b, 1.0, beta)
        allowed = {0.0, round(1 / (1 - beta), 6)}
        self.assertEqual(allowed, {round(v, 6) for v in result})

    def test_the_rescale_keeps_the_expected_difference(self):
        # This is the whole point of the R in DARE: throw away three quarters
        # of the change and the average change is unchanged.
        np.random.seed(11)
        a, b = np.zeros(200000), np.ones(200000)
        result = self.blend(a, b, 1.0, 0.75)
        self.assertAlmostEqual(1.0, float(result.mean()), places=2)

    def test_it_does_not_need_a_third_model(self):
        self.assertIsNotNone(self.blend([1.0], [2.0], 0.5, 0.5))


class SeedTests(unittest.TestCase):
    """A merge that cannot be repeated makes its own recipe a lie.

    Every other mode is deterministic, and the recipe exists to repeat a
    merge. DARE draws a mask per tensor, so it gets a seed and records it.
    """

    def dare(self, rand):
        return blend_tensors(
            INTERP_DARE, 1.0, 0.5, np.zeros(500), np.ones(500), xp=np, rand=rand
        )

    def test_the_same_seed_reproduces_the_merge(self):
        np.testing.assert_array_equal(
            self.dare(make_seeded_rand(7, np)), self.dare(make_seeded_rand(7, np))
        )

    def test_a_different_seed_draws_a_different_subset(self):
        self.assertFalse(
            np.array_equal(
                self.dare(make_seeded_rand(7, np)), self.dare(make_seeded_rand(8, np))
            )
        )

    def test_no_seed_means_the_global_stream(self):
        # `None` is "draw differently every run", which is what a caller gets
        # by not asking for reproducibility.
        self.assertIsNone(make_seeded_rand(None, np))

    def test_one_generator_advances_across_tensors(self):
        # The mask has to differ from tensor to tensor within a merge, or
        # every layer would be dropped in the same places.
        rand = make_seeded_rand(7, np)
        first, second = rand(np.zeros(500)), rand(np.zeros(500))
        self.assertFalse(np.array_equal(first, second))

    def test_only_the_stochastic_modes_declare_a_seed(self):
        seeded = {m.key for m in MERGE_MODES if m.needs_seed}
        self.assertEqual({INTERP_DARE}, seeded)


class TheInspectorKnowsEveryModeWeCanWriteTests(unittest.TestCase):
    """A merge this app produces must be legible to this app's own reader.

    Sum Twice used to display as "merge-models-chattiori" -- the name of the
    tool, not the method -- because the reader's list and the writer's list
    were maintained separately. This fails the moment a mode is added to one
    and not the other.
    """

    def test_each_mode_key_gets_a_readable_label_back(self):
        for mode in MERGE_MODES:
            with self.subTest(mode=mode.key):
                label, _ = _describe_merge_math(mode.key, {"multiplier": 0.5})
                self.assertNotEqual(mode.key, label)
                self.assertEqual(mode.label, label)

    def test_a_mode_with_a_beta_renders_it(self):
        _, formula_html = _describe_merge_math(
            INTERP_SUM_TWICE, {"multiplier": 0.5, "beta": 0.3}
        )
        self.assertIn("0.3", formula_html)


if __name__ == "__main__":
    unittest.main()
