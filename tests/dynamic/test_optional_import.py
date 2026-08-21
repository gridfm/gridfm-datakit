"""
Verify that importing gridfm_datakit.dynamic never raises ImportError even
when pypowsybl[dynamic] is not installed.

This is the most fundamental contract of the optional-dependency guard.
"""

import importlib


def test_import_gridfm_datakit_does_not_raise():
    """Top-level package import must always succeed."""
    importlib.import_module("gridfm_datakit")


def test_import_dynamic_module_does_not_raise():
    """gridfm_datakit.dynamic import must not raise even without pypowsybl[dynamic]."""
    importlib.import_module("gridfm_datakit.dynamic")


def test_import_dynawo_submodule_does_not_raise():
    """gridfm_datakit.dynamic.dynawo import must not raise at module level."""
    importlib.import_module("gridfm_datakit.dynamic.dynawo")


def test_import_dynawo_api_does_not_raise():
    """gridfm_datakit.dynamic.dynawo.api import must not raise."""
    importlib.import_module("gridfm_datakit.dynamic.dynawo.api")


def test_import_dynamic_simulate_does_not_raise():
    """gridfm_datakit.dynamic.dynawo.simulate import must not raise."""
    importlib.import_module("gridfm_datakit.dynamic.dynawo.simulate")


def test_import_process_dynamic_does_not_raise():
    """gridfm_datakit.dynamic.process_dynamic import must not raise."""
    importlib.import_module("gridfm_datakit.dynamic.process_dynamic")


def test_import_generate_dynamic_does_not_raise():
    """gridfm_datakit.dynamic.generate_dynamic import must not raise."""
    importlib.import_module("gridfm_datakit.dynamic.generate_dynamic")


def test_is_pypowsybl_dynamic_available_returns_bool():
    """is_pypowsybl_dynamic_available() must return a bool, never raise."""
    from gridfm_datakit.dynamic.dynawo.api import is_pypowsybl_dynamic_available

    result = is_pypowsybl_dynamic_available()
    assert isinstance(result, bool)
