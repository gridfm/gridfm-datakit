"""
Availability guards for the Dynawo backend.

Two independent things must be present to run a dynamic simulation, and each
fails in a different way:

1. The ``pypowsybl.dynamic`` Python module — missing when pypowsybl is not
   installed. Guarded by the lazy-import pattern below, mirroring
   gridfm_datakit/powsybl/api.py, so that importing gridfm_datakit.dynamic
   never raises ImportError.

2. A local **Dynawo installation** — the native solver that powsybl-dynawo
   shells out to. This is *not* bundled with pypowsybl: ``import
   pypowsybl.dynamic`` succeeds without it, and the run only fails much later,
   deep inside ``Simulation.run()``, with an opaque "DynawoSimulationProvider
   could not be instantiated". ``check_dynawo_available()`` turns that into an
   actionable up-front error.

All functions in simulate.py and dynawo/__init__.py that call into
pypowsybl.dynamic must go through _get_pypowsybl_dynamic() rather than
importing at module level.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

_pypowsybl_dynamic = None

# Where powsybl looks for its configuration, and how the Dynawo installation is
# declared in it. See https://powsybl.readthedocs.io/ (itools configuration).
_CONFIG_DIR_ENV = "POWSYBL_CONFIG_DIR"
_CONFIG_NAME_ENV = "POWSYBL_CONFIG_NAME"
_DEFAULT_CONFIG_DIR = Path.home() / ".itools"
_DEFAULT_CONFIG_NAME = "config"
_CONFIG_EXTENSIONS = (".yml", ".yaml", ".xml")

_INSTALL_HINT = (
    "The Dynawo backend needs a local Dynawo installation (it is NOT bundled "
    "with pypowsybl).\n"
    "  1. Download and extract Dynawo (https://dynawo.github.io/install/), e.g. to ~/dynawo\n"
    "  2. Declare it for powsybl in {config_dir}/config.yml:\n"
    "         dynawo:\n"
    "           homeDir: /path/to/dynawo\n"
    "           debug: false"
)


def _get_pypowsybl_dynamic():
    """Lazily import pypowsybl.dynamic.

    Returns the pypowsybl.dynamic module on success.

    Raises
    ------
    ImportError
        If pypowsybl[dynamic] is not installed, with install instructions.
    """
    global _pypowsybl_dynamic
    if _pypowsybl_dynamic is None:
        try:
            import pypowsybl.dynamic as _dyn

            _pypowsybl_dynamic = _dyn
        except ImportError as exc:
            raise ImportError(
                "pypowsybl dynamic simulation support is required for the Dynawo backend. "
                "Install it with: pip install 'gridfm-datakit[dynamic]' "
                "(the plain 'powsybl' extra is not enough — the backend lives behind "
                "pypowsybl's own 'dynamic' extra).",
            ) from exc
    return _pypowsybl_dynamic


def is_pypowsybl_dynamic_available() -> bool:
    """Return True if pypowsybl.dynamic can be imported, False otherwise."""
    try:
        _get_pypowsybl_dynamic()
        return True
    except ImportError:
        return False


def check_pypowsybl_dynamic_available() -> None:
    """Raise ImportError with install instructions if pypowsybl.dynamic is absent."""
    _get_pypowsybl_dynamic()


# ---------------------------------------------------------------------------
# Local Dynawo installation
# ---------------------------------------------------------------------------


def _powsybl_config_dir() -> Path:
    """Return the directory powsybl reads its configuration from."""
    env = os.environ.get(_CONFIG_DIR_ENV)
    return Path(env) if env else _DEFAULT_CONFIG_DIR


def _find_powsybl_config_file() -> Optional[Path]:
    """Return the powsybl config file, or None if there is none."""
    config_dir = _powsybl_config_dir()
    name = os.environ.get(_CONFIG_NAME_ENV, _DEFAULT_CONFIG_NAME)
    for ext in _CONFIG_EXTENSIONS:
        candidate = config_dir / f"{name}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _read_dynawo_home_dir(config_file: Path) -> Optional[str]:
    """Extract ``dynawo.homeDir`` from a powsybl YAML or XML config file."""
    try:
        text = config_file.read_text()
    except OSError:
        return None

    if config_file.suffix == ".xml":
        # <dynawo><homeDir>/path/to/dynawo</homeDir></dynawo>
        match = re.search(r"<homeDir>\s*(.+?)\s*</homeDir>", text, re.DOTALL)
        return match.group(1) if match else None

    # YAML: a "dynawo:" block with a "homeDir:" entry under it. Parsed with a
    # regex rather than yaml.safe_load so a malformed *unrelated* section of the
    # config cannot make this pre-flight check raise.
    match = re.search(
        r"^dynawo:\s*$(?:\n[ \t]+.*$)*?\n[ \t]+homeDir:[ \t]*(.+?)[ \t]*$",
        text,
        re.MULTILINE,
    )
    return match.group(1).strip("\"'") if match else None


def _dynawo_unavailable_reason() -> Optional[str]:
    """Return why Dynawo cannot run, or None if the installation looks usable."""
    config_file = _find_powsybl_config_file()
    if config_file is None:
        return (
            f"no powsybl configuration file found in {_powsybl_config_dir()} "
            f"(looked for {_DEFAULT_CONFIG_NAME}{{{','.join(_CONFIG_EXTENSIONS)}}})"
        )

    home_dir = _read_dynawo_home_dir(config_file)
    if not home_dir:
        return f"{config_file} declares no 'dynawo.homeDir' entry"

    home = Path(home_dir).expanduser()
    if not home.is_dir():
        return f"dynawo.homeDir '{home}' (from {config_file}) is not a directory"

    # The launcher powsybl-dynawo invokes. Its absence means homeDir points at
    # something that is not a Dynawo installation.
    if not (home / "dynawo.sh").is_file() and not (home / "bin" / "dynawo").is_file():
        return (
            f"dynawo.homeDir '{home}' (from {config_file}) contains no Dynawo "
            f"launcher (expected 'dynawo.sh' or 'bin/dynawo')"
        )

    return None


def is_dynawo_available() -> bool:
    """Return True if pypowsybl.dynamic AND a local Dynawo installation are usable."""
    if not is_pypowsybl_dynamic_available():
        return False
    return _dynawo_unavailable_reason() is None


def check_dynawo_available() -> None:
    """Raise if the Dynawo backend cannot run, explaining exactly what is missing.

    Called before launching simulations so a missing installation surfaces as an
    actionable message instead of an opaque "DynawoSimulationProvider could not
    be instantiated" from deep inside the solver.

    Raises
    ------
    ImportError
        If pypowsybl.dynamic is not installed.
    RuntimeError
        If no usable local Dynawo installation is declared to powsybl.
    """
    check_pypowsybl_dynamic_available()
    reason = _dynawo_unavailable_reason()
    if reason is not None:
        raise RuntimeError(
            f"Dynawo backend unavailable: {reason}.\n\n"
            + _INSTALL_HINT.format(config_dir=_powsybl_config_dir()),
        )
