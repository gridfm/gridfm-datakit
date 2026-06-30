"""Centralized control of Julia/PowerModels/Ipopt solver output.

One :class:`SolverOutputConfig` maps a single verbosity level onto the three
places solver chatter comes from: Ipopt's ``print_level``, PowerModels'
``silence()``, and Memento's root logger. :func:`redirect_c_stdio` is the
companion for the cases the loggers can't reach -- Ipopt and MUMPS write
straight to file descriptors 1/2, so only an OS-level dup2 captures them.
"""

from __future__ import annotations

import ctypes
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator, Optional

try:
    _LIBC: Optional[ctypes.CDLL] = ctypes.CDLL(None)
except Exception:  # pragma: no cover - platform without a loadable libc
    _LIBC = None


class SolverVerbosity(IntEnum):
    """Solver verbosity, from quietest to loudest."""

    SILENT = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4

    @classmethod
    def parse(cls, value: "SolverVerbosity | str | int") -> "SolverVerbosity":
        """Coerce a name (case-insensitive), int, or member into a member."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError:
                names = [m.name.lower() for m in cls]
                raise ValueError(
                    f"unknown verbosity {value!r}; expected one of {names}",
                )
        return cls(int(value))


# Ipopt print levels run 0..12; 5 already prints the full iteration table.
_IPOPT_PRINT_LEVEL = {
    SolverVerbosity.SILENT: 0,
    SolverVerbosity.ERROR: 0,
    SolverVerbosity.WARNING: 0,
    SolverVerbosity.INFO: 3,
    SolverVerbosity.DEBUG: 5,
}

_MEMENTO_LEVEL = {
    SolverVerbosity.SILENT: "not_set",
    SolverVerbosity.ERROR: "error",
    SolverVerbosity.WARNING: "warn",
    SolverVerbosity.INFO: "info",
    SolverVerbosity.DEBUG: "debug",
}


@dataclass(frozen=True)
class SolverOutputConfig:
    """How much the solvers should print, and where it goes.

    With ``log_dir`` set, output is captured to per-process files; otherwise it
    reaches the console (or is dropped when SILENT).
    """

    verbosity: SolverVerbosity = SolverVerbosity.SILENT
    log_dir: Optional[str] = None

    @classmethod
    def from_settings(
        cls,
        verbosity: "SolverVerbosity | str | int | None" = None,
        log_dir: Optional[str] = None,
        enable_solver_logs: bool = False,
    ) -> "SolverOutputConfig":
        """Build a config from user settings.

        When ``verbosity`` is omitted it defaults to DEBUG if logs are enabled
        (matching the old "write everything to files" behaviour) else SILENT.
        """
        if verbosity is None:
            verbosity = (
                SolverVerbosity.DEBUG if enable_solver_logs else SolverVerbosity.SILENT
            )
        return cls(
            SolverVerbosity.parse(verbosity),
            log_dir if enable_solver_logs else None,
        )

    @property
    def ipopt_print_level(self) -> int:
        return _IPOPT_PRINT_LEVEL[self.verbosity]

    @property
    def memento_level(self) -> str:
        return _MEMENTO_LEVEL[self.verbosity]

    @property
    def to_console(self) -> bool:
        return self.log_dir is None and self.verbosity > SolverVerbosity.SILENT

    @property
    def silence_powermodels(self) -> bool:
        # PowerModels logs through its own Memento logger, which ignores the
        # root level, so silence() is the only thing that quiets its Info/Warn.
        # Silence it unless output is explicitly headed for the console -- when
        # logging to files, its chatter still leaks to stdout from the fast
        # (non-redirected) solver paths.
        return not self.to_console

    def log_file(self, name: str) -> Optional[str]:
        """Per-process log path for solver ``name`` (e.g. "opf"), or None."""
        if self.log_dir is None:
            return None
        import multiprocessing

        proc = multiprocessing.current_process().name
        return os.path.join(self.log_dir, f"{name}_{proc}.log").replace("\\", "/")


@contextmanager
def redirect_c_stdio(target: Optional[str]) -> Iterator[None]:
    """Redirect fd 1 and 2 to ``target`` (a file) or /dev/null (when None).

    Works at the file-descriptor level, so unlike ``redirect_stdout`` it also
    captures C/Fortran output from Ipopt and MUMPS. Output is appended.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    dest = os.open(
        os.devnull if target is None else target,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        os.dup2(dest, 1)
        os.dup2(dest, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        # Flush C buffers before restoring, or block-buffered Ipopt/MUMPS output
        # would spill out afterwards.
        if _LIBC is not None:
            try:
                _LIBC.fflush(None)
            except Exception:  # pragma: no cover - defensive
                pass
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(dest)
