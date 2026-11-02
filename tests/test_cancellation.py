"""The Cancel button stops a merge at its next progress report, never mid-save."""

import os
import unittest

from merge_studio.cancellation import Cancelled, cancellable

UI_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "merge_studio_ui.py",
)


class CancellableTests(unittest.TestCase):
    def setUp(self):
        self.seen = []
        self.flag = False
        self.report = cancellable(self.seen.append, lambda: self.flag)

    def test_passes_messages_through_until_cancelled(self):
        self.report("Loading Primary Model (A)...")
        self.report("Merging UNet (1/10): layers.0")
        self.assertEqual(self.seen, ["Loading Primary Model (A)...", "Merging UNet (1/10): layers.0"])

    def test_raises_at_the_next_report_after_cancel(self):
        self.report("Merging UNet (1/10): layers.0")
        self.flag = True
        with self.assertRaises(Cancelled):
            self.report("Merging UNet (2/10): layers.1")
        self.assertEqual(self.seen, ["Merging UNet (1/10): layers.0"])

    def test_a_cancel_during_the_save_is_ignored(self):
        self.report("Saving file...")
        self.flag = True
        self.report("Verifying the saved checkpoint reopens on its own...")
        self.assertEqual(self.seen, ["Saving file...", "Verifying the saved checkpoint reopens on its own..."])

    def test_a_cancel_already_pending_stops_before_the_save(self):
        self.flag = True
        with self.assertRaises(Cancelled):
            self.report("Quantizing (5/400): layers.1.attn")


class CancelWiringTests(unittest.TestCase):
    def setUp(self):
        with open(UI_PATH, encoding="utf-8") as f:
            self.source = f.read()

    def test_progress_reports_honour_forges_interrupt_flag(self):
        self.assertIn("return cancellable(progress_cb, lambda: shared.state.interrupted)", self.source)

    def test_cancel_button_sets_the_interrupt_flag(self):
        self.assertIn("merge_cancel_btn.click(fn=lambda: shared.state.interrupt()", self.source)

    def test_every_long_handler_reports_a_cancel_as_such(self):
        for what in ("Bake", "Merge", "Conversion"):
            self.assertIn(f'return _cancelled_result("{what}")', self.source)


if __name__ == "__main__":
    unittest.main()
