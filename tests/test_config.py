"""Tests for static generation configuration validation."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from gridfm_datakit.config import validate_static_config
from gridfm_datakit.generate import _setup_environment
from gridfm_datakit.utils.param_handler import NestedNamespace


_CONFIG_PATHS = sorted(
    [*Path("scripts/config").glob("*.yaml"), *Path("tests/config").glob("*.yaml")],
)
_MAX_NUMPY_SEED = 2**32 - 1
_DISTRIBUTED_SEED_STRIDE = 20_000


def _default_config() -> dict[str, Any]:
    """Return an independent copy of the default static configuration."""
    with Path("scripts/config/default.yaml").open() as stream:
        return yaml.safe_load(stream)


def _create_existing_output(config: dict[str, Any]) -> Path:
    """Create an output marker that must survive configuration failures."""
    base_path = Path(config["settings"]["data_dir"]) / config["network"]["name"] / "raw"
    base_path.mkdir(parents=True)
    marker = base_path / "marker.txt"
    marker.write_text("must survive validation failure")
    return marker


@pytest.mark.parametrize(
    "config_path",
    _CONFIG_PATHS,
    ids=lambda path: str(path),
)
def test_validate_static_config_accepts_shipped_configs(config_path: Path) -> None:
    """Every static configuration shipped by the repository must remain valid."""
    with config_path.open() as stream:
        config = yaml.safe_load(stream)

    validated = validate_static_config(config)

    assert validated["settings"]["opf_formulation"] == "polar"
    assert validated["settings"]["pf_solver"] == "powermodel"
    assert validated["network"]["reader"] == "native"


@pytest.mark.parametrize("input_kind", ["yaml", "dict", "namespace"])
def test_setup_environment_validates_before_overwrite(
    tmp_path: Path,
    input_kind: str,
) -> None:
    """All static input forms must fail before an existing output is removed."""
    config = _default_config()
    config["settings"]["data_dir"] = str(tmp_path / "data")
    del config["topology_perturbation"]["k"]

    marker = _create_existing_output(config)

    if input_kind == "yaml":
        config_input = tmp_path / "invalid.yaml"
        config_input.write_text(yaml.safe_dump(config))
        config_input = str(config_input)
    elif input_kind == "namespace":
        config_input = NestedNamespace(**deepcopy(config))
    else:
        config_input = deepcopy(config)

    with pytest.raises(
        ValueError,
        match=r"topology_perturbation\.k: Field required",
    ):
        _setup_environment(config_input)

    assert marker.read_text() == "must survive validation failure"


def test_validate_static_config_rejects_wrong_type() -> None:
    """Strict validation must reject values that only look like the right type."""
    config = _default_config()
    config["settings"]["num_processes"] = "16"

    with pytest.raises(
        ValueError,
        match=r"settings\.num_processes: Input should be a valid integer",
    ):
        validate_static_config(config)


@pytest.mark.parametrize(
    ("seed", "match"),
    [
        (-1, r"settings\.seed: Input should be greater than or equal to 0"),
        (
            2**32,
            r"settings\.seed: Input should be less than or equal to 4294967295",
        ),
        (
            (_MAX_NUMPY_SEED - 10_000) // _DISTRIBUTED_SEED_STRIDE + 1,
            r"configuration: settings\.seed and load\.scenarios produce a derived random seed",
        ),
    ],
    ids=["negative", "above-numpy-limit", "derived-seed-overflow"],
)
def test_setup_environment_rejects_invalid_seed_before_overwrite(
    tmp_path: Path,
    seed: int,
    match: str,
) -> None:
    """Base and derived seeds must fail before an existing output is removed."""
    config = _default_config()
    config["settings"]["data_dir"] = str(tmp_path / "data")
    config["settings"]["seed"] = seed
    marker = _create_existing_output(config)

    with pytest.raises(ValueError, match=match):
        _setup_environment(config)

    assert marker.read_text() == "must survive validation failure"


def test_validate_static_config_accepts_largest_safe_derived_seed() -> None:
    """The derived-seed guard must include NumPy's upper boundary."""
    config = _default_config()
    scenarios = config["load"]["scenarios"]
    config["settings"]["seed"] = (
        _MAX_NUMPY_SEED - scenarios
    ) // _DISTRIBUTED_SEED_STRIDE

    validated = validate_static_config(config)

    assert validated["settings"]["seed"] == config["settings"]["seed"]


@pytest.mark.parametrize(
    "value",
    [float("inf"), float("-inf"), float("nan")],
    ids=["positive-infinity", "negative-infinity", "nan"],
)
def test_setup_environment_rejects_non_finite_float_before_overwrite(
    tmp_path: Path,
    value: float,
) -> None:
    """Non-finite numeric values must not reach NumPy or remove prior output."""
    config = _default_config()
    config["settings"]["data_dir"] = str(tmp_path / "data")
    config["load"]["sigma"] = value
    marker = _create_existing_output(config)
    config_path = tmp_path / "non-finite.yaml"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(
        ValueError,
        match=r"load\.sigma: Input should be a finite number",
    ):
        _setup_environment(str(config_path))

    assert marker.read_text() == "must survive validation failure"


def test_setup_environment_rejects_incompatible_powsybl_reader_before_overwrite(
    tmp_path: Path,
) -> None:
    """PowSyBl PF must request the reader that builds its network and mappings."""
    config = _default_config()
    config["settings"]["data_dir"] = str(tmp_path / "data")
    config["settings"]["mode"] = "pf"
    config["settings"]["pf_solver"] = "powsybl"
    config["network"]["reader"] = "native"
    marker = _create_existing_output(config)

    with pytest.raises(
        ValueError,
        match=(
            r"configuration: settings\.pf_solver='powsybl' is incompatible "
            r"with network\.reader='native' in PF mode; missing capabilities:"
        ),
    ):
        _setup_environment(config)

    assert marker.read_text() == "must survive validation failure"


def test_validate_static_config_does_not_require_powsybl_reader_for_opf() -> None:
    """The PF-only solver setting must not constrain OPF network loading."""
    config = _default_config()
    config["settings"]["mode"] = "opf"
    config["settings"]["pf_solver"] = "powsybl"
    config["network"]["reader"] = "native"

    validated = validate_static_config(config)

    assert validated["network"]["reader"] == "native"


def test_validate_static_config_accepts_powsybl_reader_for_powsybl_pf() -> None:
    """A PF backend must be accepted when its reader provides all capabilities."""
    config = _default_config()
    config["settings"]["mode"] = "pf"
    config["settings"]["pf_solver"] = "powsybl"
    config["network"]["reader"] = "powsybl"

    validated = validate_static_config(config)

    assert validated["settings"]["pf_solver"] == "powsybl"


def test_validate_static_config_supplies_execution_defaults() -> None:
    """Sequential callers need not provide parallel execution tuning fields."""
    config = _default_config()
    del config["settings"]["num_processes"]
    del config["settings"]["large_chunk_size"]

    validated = validate_static_config(config)

    assert validated["settings"]["num_processes"] == 1
    assert validated["settings"]["large_chunk_size"] == 1_000


def test_validate_static_config_rejects_unknown_field() -> None:
    """Unknown fields must name their full path so configuration typos are clear."""
    config = _default_config()
    config["load"]["scenerios"] = 10

    with pytest.raises(
        ValueError,
        match=r"load\.scenerios: Extra inputs are not permitted",
    ):
        validate_static_config(config)


def test_setup_environment_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    """A syntactically valid YAML sequence must not enter generation."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- network\n- settings\n")

    with pytest.raises(
        ValueError,
        match="Configuration must contain a YAML mapping, got list",
    ):
        _setup_environment(str(config_path))


def test_setup_environment_rejects_unsupported_input_type() -> None:
    """The Python API must name the accepted configuration input forms."""
    with pytest.raises(
        TypeError,
        match="YAML path, a dictionary, or a NestedNamespace",
    ):
        _setup_environment(["network"])


@pytest.mark.parametrize(
    ("section", "config_type", "field"),
    [
        ("topology_perturbation", "random", "n_topology_variants"),
        ("topology_perturbation", "n_minus_k", "k"),
        ("generation_perturbation", "cost_perturbation", "sigma"),
        ("admittance_perturbation", "random_perturbation", "sigma"),
    ],
)
def test_validate_static_config_requires_conditional_fields(
    section: str,
    config_type: str,
    field: str,
) -> None:
    """Fields required by a selected generator type must fail during parsing."""
    config = _default_config()
    config[section]["type"] = config_type
    config[section].pop(field, None)

    with pytest.raises(ValueError, match=rf"{section}\.{field}: Field required"):
        validate_static_config(config)


def test_validate_static_config_reports_all_errors() -> None:
    """A single validation pass should report independent configuration errors."""
    config = _default_config()
    del config["network"]["name"]
    config["load"]["scenarios"] = 0
    config["settings"]["mode"] = "invalid"

    with pytest.raises(ValueError) as exc_info:
        validate_static_config(config)

    message = str(exc_info.value)
    assert "network.name: Field required" in message
    assert "load.scenarios: Input should be greater than 0" in message
    assert "settings.mode: Input should be 'pf' or 'opf'" in message
