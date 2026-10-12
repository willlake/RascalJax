"""Small shared helpers."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Callable, Iterable

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def transposed(a: Array) -> Array:
    """Swap the last two (matrix) axes."""
    return np.swapaxes(a, -2, -1)


def symmetrized(a: Array) -> Array:
    """Symmetrize over the last two axes."""
    return 0.5 * (a + transposed(a))


def format_skip_s_bins(skip: int | Iterable[int]) -> tuple[int, int]:
    """Normalize ``skip_s_bins`` to ``(n_start, n_end)``."""
    if isinstance(skip, (int, np.integer)):
        return int(skip), 0
    skip = tuple(int(s) for s in skip)
    if len(skip) == 0:
        return 0, 0
    if len(skip) == 1:
        return skip[0], 0
    if len(skip) == 2:
        return skip
    raise ValueError("skip_s_bins must be an int or a tuple of up to two ints")


class Logger:
    """Print to stdout and optionally append to a log file, with timestamps."""

    def __init__(self, logfile: str | None = None, verbose: bool = True):
        self.logfile = logfile
        self.verbose = verbose
        if logfile is not None:
            os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        if self.verbose:
            print(line, flush=True)
        if self.logfile is not None:
            with open(self.logfile, "a") as f:
                f.write(line + "\n")


def silent(msg: str) -> None:
    """A logger that does nothing."""
    return None
