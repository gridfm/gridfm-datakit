"""Validation models for gridfm-datakit configuration."""

from math import isclose
from typing import Annotated, Any, Dict, Literal, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


_PositiveInt = Annotated[int, Field(gt=0)]
_PositiveFloat = Annotated[float, Field(gt=0)]
_NonNegativeFloat = Annotated[float, Field(ge=0)]
_Probability = Annotated[float, Field(ge=0, le=1)]
_NonEmptyString = Annotated[str, Field(min_length=1)]
_TopologyElement = Literal["branch", "gen"]
_MAX_NUMPY_SEED = 2**32 - 1
_DISTRIBUTED_SEED_STRIDE = 20_000
_AUTO_SEED_UPPER_BOUND = 50_000
_Seed = Annotated[int, Field(ge=0, le=_MAX_NUMPY_SEED)]


class _ConfigModel(BaseModel):
    """Base model shared by strict configuration schemas."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        strict=True,
    )


class _NetworkConfig(_ConfigModel):
    """Static network input configuration."""

    name: _NonEmptyString
    source: Literal["pglib", "file"]
    network_dir: Optional[_NonEmptyString] = None
    file: Optional[_NonEmptyString] = None
    reader: Literal["native", "powsybl"] = "native"

    @model_validator(mode="after")
    def validate_file_location(self) -> "_NetworkConfig":
        """Require a usable location when loading a network from a file."""
        if self.source != "file":
            return self
        if self.reader == "native" and self.network_dir is None:
            raise ValueError(
                "network_dir is required when source='file' and reader='native'",
            )
        if self.reader == "powsybl" and self.file is None and self.network_dir is None:
            raise ValueError(
                "file or network_dir is required when source='file' and "
                "reader='powsybl'",
            )
        return self


class _LoadConfig(_ConfigModel):
    """Fields shared by static load scenario generators."""

    agg_profile: _NonEmptyString
    scenarios: _PositiveInt


class _AggregatedProfileLoadConfig(_LoadConfig):
    """Configuration for aggregated-profile load scenarios."""

    generator: Literal["agg_load_profile"]
    sigma: _NonNegativeFloat
    change_reactive_power: bool
    global_range: Annotated[float, Field(ge=0, le=1)]
    max_scaling_factor: _PositiveFloat
    step_size: _PositiveFloat
    start_scaling_factor: _PositiveFloat

    @model_validator(mode="after")
    def validate_scaling_bounds(self) -> "_AggregatedProfileLoadConfig":
        """Require the scaling search to start within its configured range."""
        if self.start_scaling_factor > self.max_scaling_factor:
            raise ValueError(
                "start_scaling_factor must be less than or equal to max_scaling_factor",
            )
        return self


class _PowergraphLoadConfig(_LoadConfig):
    """Configuration for PowerGraph load scenarios."""

    generator: Literal["powergraph"]
    sigma: Optional[_NonNegativeFloat] = None
    change_reactive_power: Optional[bool] = None
    global_range: Optional[Annotated[float, Field(ge=0, le=1)]] = None
    max_scaling_factor: Optional[_PositiveFloat] = None
    step_size: Optional[_PositiveFloat] = None
    start_scaling_factor: Optional[_PositiveFloat] = None


_StaticLoadConfig = Annotated[
    Union[_AggregatedProfileLoadConfig, _PowergraphLoadConfig],
    Field(discriminator="generator"),
]


class _TopologyConfig(_ConfigModel):
    """Optional fields shared by static topology perturbations."""

    k: Optional[_PositiveInt] = None
    n_topology_variants: Optional[_PositiveInt] = None
    elements: Optional[Annotated[list[_TopologyElement], Field(min_length=1)]] = None
    outage_count_probabilities: Optional[list[_Probability]] = None


class _RandomTopologyConfig(_TopologyConfig):
    """Configuration for random topology perturbations."""

    type: Literal["random"]
    k: _PositiveInt
    n_topology_variants: _PositiveInt

    @model_validator(mode="after")
    def validate_outage_count_probabilities(self) -> "_RandomTopologyConfig":
        """Validate the optional probability assigned to each outage count."""
        probabilities = self.outage_count_probabilities
        if probabilities is None:
            return self
        if len(probabilities) != self.k + 1:
            raise ValueError(
                "outage_count_probabilities must have length k + 1",
            )
        if not isclose(sum(probabilities), 1.0, abs_tol=1e-8):
            raise ValueError(
                "outage_count_probabilities must sum to 1.0",
            )
        return self


class _NMinusKTopologyConfig(_TopologyConfig):
    """Configuration for exhaustive N-k topology perturbations."""

    type: Literal["n_minus_k"]
    k: _PositiveInt


class _NoTopologyPerturbationConfig(_TopologyConfig):
    """Configuration that disables topology perturbations."""

    type: Literal["none"]


_StaticTopologyConfig = Annotated[
    Union[
        _RandomTopologyConfig,
        _NMinusKTopologyConfig,
        _NoTopologyPerturbationConfig,
    ],
    Field(discriminator="type"),
]


class _GenerationPerturbationConfig(_ConfigModel):
    """Fields shared by generator cost perturbations."""

    sigma: Optional[_NonNegativeFloat] = None


class _CostPermutationConfig(_GenerationPerturbationConfig):
    """Configuration for generator cost permutation."""

    type: Literal["cost_permutation"]


class _CostPerturbationConfig(_GenerationPerturbationConfig):
    """Configuration for generator cost scaling."""

    type: Literal["cost_perturbation"]
    sigma: _NonNegativeFloat


class _NoGenerationPerturbationConfig(_GenerationPerturbationConfig):
    """Configuration that disables generator perturbations."""

    type: Literal["none"]


_StaticGenerationPerturbationConfig = Annotated[
    Union[
        _CostPermutationConfig,
        _CostPerturbationConfig,
        _NoGenerationPerturbationConfig,
    ],
    Field(discriminator="type"),
]


class _AdmittancePerturbationConfig(_ConfigModel):
    """Fields shared by branch admittance perturbations."""

    sigma: Optional[_NonNegativeFloat] = None


class _RandomAdmittancePerturbationConfig(_AdmittancePerturbationConfig):
    """Configuration for random branch admittance perturbations."""

    type: Literal["random_perturbation"]
    sigma: _NonNegativeFloat


class _NoAdmittancePerturbationConfig(_AdmittancePerturbationConfig):
    """Configuration that disables admittance perturbations."""

    type: Literal["none"]


_StaticAdmittancePerturbationConfig = Annotated[
    Union[
        _RandomAdmittancePerturbationConfig,
        _NoAdmittancePerturbationConfig,
    ],
    Field(discriminator="type"),
]


class _StaticSettingsConfig(_ConfigModel):
    """Settings required by static PF and OPF generation."""

    num_processes: _PositiveInt = 1
    data_dir: _NonEmptyString
    large_chunk_size: _PositiveInt = 1_000
    overwrite: bool
    mode: Literal["pf", "opf"]
    include_dc_res: bool
    enable_solver_logs: bool
    pf_fast: bool
    dcpf_fast: bool
    max_iter: _PositiveInt
    seed: Optional[_Seed] = None
    pf_solver: Literal["powermodel", "powsybl"] = "powermodel"
    opf_formulation: Literal["polar", "rectangular"] = "polar"


class _StaticGenerationConfig(_ConfigModel):
    """Complete configuration for static PF or OPF generation."""

    network: _NetworkConfig
    load: _StaticLoadConfig
    topology_perturbation: _StaticTopologyConfig
    generation_perturbation: _StaticGenerationPerturbationConfig
    admittance_perturbation: _StaticAdmittancePerturbationConfig
    settings: _StaticSettingsConfig

    @model_validator(mode="after")
    def validate_pf_solver_network_reader(self) -> "_StaticGenerationConfig":
        """Require the network representation needed by the PowSyBl PF solver."""
        if (
            self.settings.mode == "pf"
            and self.settings.pf_solver == "powsybl"
            and self.network.reader != "powsybl"
        ):
            raise ValueError(
                "settings.pf_solver='powsybl' requires "
                "network.reader='powsybl' in PF mode because the native reader "
                "does not initialize a PowSyBl network or its index mappings",
            )
        return self

    @model_validator(mode="after")
    def validate_derived_seed_range(self) -> "_StaticGenerationConfig":
        """Keep every seed derived by the distributed path in NumPy's range."""
        seed = self.settings.seed
        if seed is None:
            # _setup_environment draws auto seeds from [0, 50_000). Validate
            # against the largest possible draw so every generated seed is safe.
            seed = _AUTO_SEED_UPPER_BOUND - 1
        max_derived_seed = seed * _DISTRIBUTED_SEED_STRIDE + self.load.scenarios
        if max_derived_seed > _MAX_NUMPY_SEED:
            raise ValueError(
                "settings.seed and load.scenarios produce a derived random seed "
                f"of {max_derived_seed}, exceeding NumPy's maximum seed "
                f"{_MAX_NUMPY_SEED}",
            )
        return self


_DISCRIMINATOR_VALUES = {
    "agg_load_profile",
    "powergraph",
    "random",
    "n_minus_k",
    "none",
    "cost_permutation",
    "cost_perturbation",
    "random_perturbation",
}


def _format_error_location(location: tuple[Union[str, int], ...]) -> str:
    """Convert a Pydantic error location to a user-facing config path."""
    parts = [str(part) for part in location if part not in _DISCRIMINATOR_VALUES]
    return ".".join(parts) if parts else "configuration"


def validate_static_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a static PF or OPF configuration.

    Args:
        config: Parsed configuration mapping.

    Returns:
        Validated configuration with established defaults applied.

    Raises:
        ValueError: If required fields are missing, values have invalid types or
            ranges, or unknown fields are present.
    """
    try:
        validated = _StaticGenerationConfig.model_validate(config)
    except ValidationError as exc:
        messages = []
        for error in exc.errors(include_url=False):
            location = _format_error_location(error["loc"])
            message = error["msg"].removeprefix("Value error, ")
            messages.append(f"- {location}: {message}")
        raise ValueError(
            "Invalid static configuration:\n" + "\n".join(messages),
        ) from exc
    return validated.model_dump(exclude_none=True)
