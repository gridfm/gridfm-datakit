"""Validation models for gridfm-datakit configuration."""

from dataclasses import dataclass
from enum import Enum
from math import isclose
from typing import Annotated, Any, Dict, Literal, Mapping, Optional, Protocol, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from gridfm_datakit.utils.random_seed import _DEFAULT_SEED_POLICY, _SeedPolicy


_PositiveInt = Annotated[int, Field(gt=0)]
_PositiveFloat = Annotated[float, Field(gt=0)]
_NonNegativeFloat = Annotated[float, Field(ge=0)]
_Probability = Annotated[float, Field(ge=0, le=1)]
_NonEmptyString = Annotated[str, Field(min_length=1)]
_TopologyElement = Literal["branch", "gen"]
_Seed = Annotated[
    int,
    Field(ge=_DEFAULT_SEED_POLICY.min_seed, le=_DEFAULT_SEED_POLICY.max_seed),
]


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


@dataclass(frozen=True)
class _ConfigIssue:
    """Describe one semantic configuration error."""

    location: str
    message: str


class _StaticConfigRule(Protocol):
    """Interface implemented by cross-field configuration rules."""

    def check(self, config: _StaticGenerationConfig) -> tuple[_ConfigIssue, ...]:
        """Return every issue found by this rule."""
        ...


class _NetworkCapability(Enum):
    """Network representations that readers can provide to solver backends."""

    GRIDFM_NETWORK = "GridFM network"
    POWSYBL_NETWORK = "PowSyBl network"
    POWSYBL_INDEX_MAPPING = "PowSyBl-to-GridFM index mapping"


_READER_CAPABILITIES = {
    "native": frozenset({_NetworkCapability.GRIDFM_NETWORK}),
    "powsybl": frozenset(
        {
            _NetworkCapability.GRIDFM_NETWORK,
            _NetworkCapability.POWSYBL_NETWORK,
            _NetworkCapability.POWSYBL_INDEX_MAPPING,
        },
    ),
}

_PF_SOLVER_REQUIREMENTS = {
    "powermodel": frozenset({_NetworkCapability.GRIDFM_NETWORK}),
    "powsybl": frozenset(
        {
            _NetworkCapability.POWSYBL_NETWORK,
            _NetworkCapability.POWSYBL_INDEX_MAPPING,
        },
    ),
}


class _PfBackendCapabilityRule:
    """Require the selected reader to provide the PF backend's inputs."""

    def check(self, config: _StaticGenerationConfig) -> tuple[_ConfigIssue, ...]:
        """Report capabilities missing from a PF reader/solver combination."""
        if config.settings.mode != "pf":
            return ()

        provided = _READER_CAPABILITIES[config.network.reader]
        required = _PF_SOLVER_REQUIREMENTS[config.settings.pf_solver]
        missing = required - provided
        if not missing:
            return ()

        missing_names = ", ".join(sorted(capability.value for capability in missing))
        return (
            _ConfigIssue(
                location="configuration",
                message=(
                    f"settings.pf_solver={config.settings.pf_solver!r} is "
                    f"incompatible with network.reader={config.network.reader!r} "
                    f"in PF mode; missing capabilities: {missing_names}"
                ),
            ),
        )


@dataclass(frozen=True)
class _DerivedSeedRule:
    """Keep every possible distributed seed in NumPy's supported range."""

    policy: _SeedPolicy = _DEFAULT_SEED_POLICY

    def check(self, config: _StaticGenerationConfig) -> tuple[_ConfigIssue, ...]:
        """Report an overflow in the largest derived chunk seed."""
        max_derived_seed = self.policy.maximum_distributed_seed(
            config.settings.seed,
            config.load.scenarios,
        )
        if max_derived_seed <= self.policy.max_seed:
            return ()
        return (
            _ConfigIssue(
                location="configuration",
                message=(
                    "settings.seed and load.scenarios produce a derived random "
                    f"seed of {max_derived_seed}, exceeding NumPy's maximum seed "
                    f"{self.policy.max_seed}"
                ),
            ),
        )


_STATIC_CONFIG_RULES: tuple[_StaticConfigRule, ...] = (
    _PfBackendCapabilityRule(),
    _DerivedSeedRule(),
)


def _check_static_config_rules(
    config: _StaticGenerationConfig,
) -> tuple[_ConfigIssue, ...]:
    """Run all registered semantic rules for a parsed static configuration."""
    return tuple(issue for rule in _STATIC_CONFIG_RULES for issue in rule.check(config))


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

    issues = _check_static_config_rules(validated)
    if issues:
        messages = [f"- {issue.location}: {issue.message}" for issue in issues]
        raise ValueError(
            "Invalid static configuration:\n" + "\n".join(messages),
        )
    return validated.model_dump(exclude_none=True)
