"""
Optional import guard for pypowsybl.dynamic.

Mirrors the lazy-import pattern used in gridfm_datakit/powsybl/api.py so that
importing gridfm_datakit.dynamic (or gridfm_datakit.dynamic.dynawo) never
raises ImportError when pypowsybl[dynamic] is not installed.

All functions in simulate.py and dynawo/__init__.py that call into
pypowsybl.dynamic must go through _get_pypowsybl_dynamic() rather than
importing at module level.
"""
# TODO: check local Dynawo installation ? or put a warning for that
# TODO: discuss this is necessary since pypowsybl.dynamic always comes with pypowsybl so maybe just recycle pypowsybl's checks

from __future__ import annotations

_pypowsybl_dynamic = None


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
                "Install it with: pip install 'pypowsybl[dynamic]'",
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