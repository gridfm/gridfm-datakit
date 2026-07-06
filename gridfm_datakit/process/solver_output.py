"""Controlled routing of sub-module output (Ipopt, PowerModels, MUMPS, ...).

The problem this solves: several sub-modules print on their own terms. Ipopt
and MUMPS write straight to file descriptors 1/2 from C/Fortran; PowerModels
logs through its own Memento logger; Pkg prints activation banners. There is no
single Python or Julia switch that quiets or captures all of them.

This module gives every noisy sub-system **one well-defined path**:

* Policy -- :class:`SolverOutputConfig` maps a single :class:`SolverVerbosity`
  onto the knobs each sub-system understands (Ipopt ``print_level``, Memento
  level, whether to call ``PowerModels.silence()``).
* Routing -- a :class:`LogChannel` is a *named* destination (console, a file,
  or discard). A :class:`LogRouter` owns one channel per sub-system for the
  current process.
* Capture -- a single fd-level context manager (:func:`redirect_fds`) moves the
  bytes. Because Ipopt/MUMPS bypass Python's ``sys.stdout``, only an OS-level
  ``dup2`` reliably captures them, so this is the *only* capture mechanism --
  it is applied uniformly around every solve rather than compiled into the
  Julia solver definitions.

Typical wiring (see ``process_network.init_julia`` / ``process.solvers``)::

    router = build_router(config)        # once per worker process
    set_active_router(router)            # publish it process-wide
    ...
    with solver_capture("opf"):          # around each jl.run_opf(...) call
        result = jl.run_opf(case_file)
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Iterator, Optional

try:
    _LIBC: Optional[ctypes.CDLL] = ctypes.CDLL(None)
except Exception:  # pragma: no cover - platform without a loadable libc
    _LIBC = None

# dup2 swaps process-global fds 1/2, so it must not run concurrently from two
# threads. An RLock also lets a single thread nest captures (warm-up inside a
# solve) without self-deadlock; save/restore is per level so nesting is sound.
_FD_LOCK = threading.RLock()


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
# Ipopt has no true error/warning granularity, so ERROR/WARNING collapse to 0 --
# to actually see anything from Ipopt you need INFO or louder.
_IPOPT_PRINT_LEVEL = {
    SolverVerbosity.SILENT: 0,
    SolverVerbosity.ERROR: 0,
    SolverVerbosity.WARNING: 0,
    SolverVerbosity.INFO: 3,
    SolverVerbosity.DEBUG: 5,
}

# Memento levels, quietest to loudest: emergency > ... > error > warn > info >
# debug > not_set. "not_set" means "inherit" (i.e. NOT silent), so SILENT maps
# to the highest level that suppresses everything the solvers emit.
_MEMENTO_LEVEL = {
    SolverVerbosity.SILENT: "emergency",
    SolverVerbosity.ERROR: "error",
    SolverVerbosity.WARNING: "warn",
    SolverVerbosity.INFO: "info",
    SolverVerbosity.DEBUG: "debug",
}


@dataclass(frozen=True)
class SolverOutputConfig:
    """Policy: how loud the solvers are, and whether output goes to files.

    This is a pure value object -- it knows nothing about multiprocessing or
    where individual log files live. Use :func:`build_router` to turn it into
    the per-process :class:`LogRouter` that owns the actual destinations.

    With ``log_dir`` set, solver output is captured to per-process files;
    otherwise it reaches the console (or is discarded when SILENT).
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
        """True when output should stream to the live console (not files)."""
        return self.log_dir is None and self.verbosity > SolverVerbosity.SILENT

    @property
    def silence_powermodels(self) -> bool:
        """Whether to call ``PowerModels.silence()``.

        Driven purely by the level: PowerModels' own Info/Warn is wanted only
        at INFO or louder. This is independent of console-vs-file, so file logs
        at DEBUG genuinely contain PowerModels output (the old code silenced it
        whenever logging to files, defeating the logs).
        """
        return self.verbosity < SolverVerbosity.INFO


@dataclass(frozen=True)
class LogChannel:
    """A named output destination for one sub-system.

    Exactly one of the three modes applies:

    * ``to_console`` -- pass through; the sub-system writes to the real
      stdout/stderr (used when the user asked for live output).
    * ``path`` set -- capture fd 1/2 into this file (append).
    * neither -- discard (route fd 1/2 to /dev/null).
    """

    name: str
    path: Optional[str] = None
    to_console: bool = False

    def capture(self):
        """Context manager routing fd 1/2 for the duration of a call."""
        if self.to_console:
            return nullcontext()
        return redirect_fds(self.path)

    def write_header(self, text: str) -> None:
        """Emit a human-readable marker to wherever this channel points."""
        if self.to_console:
            print(text, end="" if text.endswith("\n") else "\n", flush=True)
        elif self.path is not None:
            with open(self.path, "a") as f:
                f.write(text if text.endswith("\n") else text + "\n")
        # discard -> nothing


# Sub-systems that get their own channel. Extend this list to bring any new
# noisy object under the same controlled routing.
SOLVER_CHANNELS = ("opf", "dcopf", "pf", "dcpf", "warmup")


class LogRouter:
    """Per-process registry mapping sub-system name -> :class:`LogChannel`."""

    def __init__(self, channels: Dict[str, LogChannel]):
        self._channels = dict(channels)

    def channel(self, name: str) -> LogChannel:
        # Unknown names discard by default rather than leaking to the console.
        return self._channels.get(name, LogChannel(name))

    def capture(self, name: str):
        return self.channel(name).capture()


def _process_log_path(log_dir: str, name: str) -> str:
    """Per-process log file for sub-system ``name`` under ``log_dir``."""
    import multiprocessing

    proc = multiprocessing.current_process().name
    return os.path.join(log_dir, f"{name}_{proc}.log").replace("\\", "/")


def build_router(
    config: SolverOutputConfig,
    names: "tuple[str, ...]" = SOLVER_CHANNELS,
) -> LogRouter:
    """Resolve a :class:`SolverOutputConfig` into concrete per-process channels.

    This is where the (impure) multiprocessing-aware file naming lives, keeping
    :class:`SolverOutputConfig` a pure policy object.
    """
    channels: Dict[str, LogChannel] = {}
    for name in names:
        if config.log_dir is not None:
            channels[name] = LogChannel(
                name,
                path=_process_log_path(config.log_dir, name),
            )
        else:
            channels[name] = LogChannel(name, to_console=config.to_console)
    return LogRouter(channels)


# --- Process-wide active router -------------------------------------------
# Set once per worker in init_julia; read at each solve site in process.solvers
# so the capture policy travels with the process, not the call signature.
_active_router: Optional[LogRouter] = None


def set_active_router(router: Optional[LogRouter]) -> None:
    global _active_router
    _active_router = router


def get_active_router() -> Optional[LogRouter]:
    return _active_router


def solver_capture(name: str):
    """Capture fd 1/2 for sub-system ``name`` via the active router.

    A no-op when no router is installed, so solver functions stay usable
    outside the generation pipeline (e.g. ad-hoc scripts, tests).
    """
    router = _active_router
    return router.capture(name) if router is not None else nullcontext()


@contextmanager
def redirect_fds(target: Optional[str]) -> Iterator[None]:
    """Redirect fd 1 and 2 to ``target`` (a file) or /dev/null (when None).

    Works at the file-descriptor level, so unlike ``redirect_stdout`` it also
    captures C/Fortran output from Ipopt and MUMPS. Output is appended.
    Serialised (and nestable) via ``_FD_LOCK``.
    """
    with _FD_LOCK:
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
            # Flush C buffers before restoring, or block-buffered Ipopt/MUMPS
            # output would spill out afterwards.
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


# Backwards-compatible alias for the previous public name.
redirect_c_stdio = redirect_fds
