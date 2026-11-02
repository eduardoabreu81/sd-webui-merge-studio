"""Stopping a running merge from the UI.

A merge, bake or conversion runs as one blocking call inside Forge's GPU
queue, so nothing can kill it from outside without leaving the models half
loaded. What it does do is report progress constantly -- once per layer while
merging or quantizing -- and that report is where it can stop cleanly: the
Cancel button sets Forge's own `shared.state.interrupted` (the flag its
Interrupt button sets), and the next progress report raises `Cancelled`. The
exception unwinds through the same `finally` blocks an error would, so the
models are released exactly as they are after a failure.

Steps that report nothing while they run (loading a checkpoint, materializing
the LoRA patches) finish first; the cancel lands at the next report.

Once the file starts being written the request is ignored: the write itself
cannot be stopped, and reporting "cancelled" for a file that ends up on disk
anyway would be wrong.

No torch and no Forge here, so it can be tested on its own.
"""

from __future__ import annotations

from typing import Callable


class Cancelled(RuntimeError):
    """Raised from a progress report after the user pressed Cancel."""


# A progress message starting with this means the output is being written.
SAVING_PREFIX = "Saving"


def cancellable(progress_cb: Callable[[str], None], is_interrupted: Callable[[], bool]) -> Callable[[str], None]:
    """Wrap a progress callback so it raises `Cancelled` once the user asks.

    `is_interrupted` is read before every report until saving starts, and
    never after.
    """
    saving = False

    def report(msg):
        nonlocal saving
        if not saving:
            if str(msg).startswith(SAVING_PREFIX):
                saving = True
            elif is_interrupted():
                raise Cancelled("Cancelled by the user. Nothing was saved.")
        progress_cb(msg)

    return report
