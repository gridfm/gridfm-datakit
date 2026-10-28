"""Drawn values for config entries.

Forms:
    number                                              fixed
    {distribution: normal, mean, std, min?, max?}       truncated normal
    {distribution: uniform, low, high}                  on [low, high)
    {distribution: choice, values, weights?}            one listed value

Out of range values are redrawn, never clipped: a normal is sampled from
``scipy.stats.truncnorm`` on its ``support()``, ``[min, max]`` intersected with
the floor, an unbounded side ending 6 std beyond the mean, or beyond the other
side's bound when that is further out. A truncation interval with no overlap with
``[mean - 6 std, mean + 6 std]`` is rejected at parse. Draws never leave the
support, so a caller validates against it.

A floor is checked at parse: fixed, uniform and choice specs are rejected when
their support fails it, a normal takes it as a lower truncation bound and redraws a
value equal to a strict one, at most ``_MAX_REDRAWS`` times: ``truncnorm`` returns
the bound itself on an interval too narrow to resolve.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import truncnorm

_NORMAL_SPAN = 6.0
_MAX_REDRAWS = 1000

_KEYS = {
    "normal": ({"mean", "std"}, {"min", "max"}),
    "uniform": ({"low", "high"}, set()),
    "choice": ({"values"}, {"weights"}),
}


@dataclass(frozen=True)
class Floor:
    """Lower bound every drawn value must admit.

    Args:
        bound: The bound.
        strict: Whether the bound itself is excluded.
    """

    bound: float
    strict: bool

    def admits(self, x: float) -> bool:
        """Check a value against the floor.

        Args:
            x: The value.

        Returns:
            bool: Whether ``x`` lies above the bound, or on it when not strict.
        """
        return x > self.bound if self.strict else x >= self.bound


POSITIVE = Floor(0.0, strict=True)
NON_NEGATIVE = Floor(0.0, strict=False)


class ValueSpec:
    """A value drawn once per use."""

    def sample(self, rng: np.random.Generator) -> Any:
        """Draw one value.

        Args:
            rng: The generator every draw comes from.

        Returns:
            Any: A float for numeric specs, the list element for a choice.
        """
        raise NotImplementedError

    def support(self) -> Tuple[Any, Any]:
        """Return the smallest and largest value a draw can take.

        Returns:
            Tuple[Any, Any]: The worst case bounds.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Fixed(ValueSpec):
    """A constant.

    Args:
        value: The value every draw returns.
    """

    value: float

    def sample(self, rng: np.random.Generator) -> float:
        """Return the constant.

        Args:
            rng: Unused.

        Returns:
            float: The value.
        """
        return self.value

    def support(self) -> Tuple[float, float]:
        """Return the value twice.

        Returns:
            Tuple[float, float]: ``(value, value)``.
        """
        return self.value, self.value


@dataclass(frozen=True)
class Normal(ValueSpec):
    """A normal distribution truncated to its support.

    Args:
        mean: The mean before truncation.
        std: The standard deviation before truncation.
        lower: The lower truncation bound, ``-inf`` when unbounded.
        upper: The upper truncation bound, ``inf`` when unbounded.
        strict: Whether ``lower`` is excluded.
    """

    mean: float
    std: float
    lower: float = -math.inf
    upper: float = math.inf
    strict: bool = False

    def sample(self, rng: np.random.Generator) -> float:
        """Draw from the truncated normal.

        Args:
            rng: The generator the draw comes from.

        Returns:
            float: A value inside the support, above ``lower`` when strict.

        Raises:
            ValueError: If no draw lands above a strict ``lower``.
        """
        lower, upper = self.support()
        a = (lower - self.mean) / self.std
        b = (upper - self.mean) / self.std
        for _ in range(_MAX_REDRAWS):
            x = float(
                truncnorm.rvs(a, b, loc=self.mean, scale=self.std, random_state=rng),
            )
            if not (self.strict and x <= self.lower):
                return x
        raise ValueError(
            f"no draw above {self.lower} in [{lower}, {upper}] after "
            f"{_MAX_REDRAWS} attempts",
        )

    def support(self) -> Tuple[float, float]:
        """Return the worst case bounds of a draw.

        An unbounded side lies 6 std beyond the mean or beyond the bound on the
        other side, whichever is further out.

        Returns:
            Tuple[float, float]: The worst case bounds.
        """
        span = _NORMAL_SPAN * self.std
        lower = self.lower
        if math.isinf(lower):
            lower = min(self.mean, self.upper) - span
        upper = self.upper
        if math.isinf(upper):
            upper = max(self.mean, self.lower) + span
        return lower, upper


@dataclass(frozen=True)
class Uniform(ValueSpec):
    """A uniform distribution on ``[low, high)``.

    Args:
        low: The inclusive lower bound.
        high: The exclusive upper bound.
    """

    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        """Draw from the uniform distribution.

        Args:
            rng: The generator the draw comes from.

        Returns:
            float: A value inside ``[low, high)``.
        """
        return float(rng.uniform(self.low, self.high))

    def support(self) -> Tuple[float, float]:
        """Return the bounds.

        Returns:
            Tuple[float, float]: ``(low, high)``.
        """
        return self.low, self.high


@dataclass(frozen=True)
class Choice(ValueSpec):
    """One of a list of values.

    Args:
        values: The values to choose from.
        probabilities: The probability of each value.
    """

    values: Tuple[Any, ...]
    probabilities: Tuple[float, ...]

    def sample(self, rng: np.random.Generator) -> Any:
        """Draw one listed value.

        Args:
            rng: The generator the draw comes from.

        Returns:
            Any: The drawn element of ``values``.
        """
        return self.values[rng.choice(len(self.values), p=self.probabilities)]

    def support(self) -> Tuple[Any, Any]:
        """Return the smallest and largest listed value.

        Returns:
            Tuple[Any, Any]: ``(min(values), max(values))``.
        """
        return min(self.values), max(self.values)


def parse_value_spec(
    spec: Any,
    path: str,
    floor: Optional[Floor] = None,
) -> ValueSpec:
    """Parse a config entry into a value spec.

    Args:
        spec: A number or a mapping with a ``distribution`` key.
        path: The config path named in every error.
        floor: A lower bound every value must admit.

    Returns:
        ValueSpec: The parsed spec.

    Raises:
        ValueError: If the entry is malformed, a parameter is invalid, or a value
            can fail the floor. The message starts with ``f"{path}: "``.
    """
    if not isinstance(spec, Mapping):
        value = _number(spec, path, "value")
        _check_floor(value, path, floor)
        return Fixed(value)

    distribution = spec.get("distribution")
    if not isinstance(distribution, str) or distribution not in _KEYS:
        raise ValueError(
            f"{path}: distribution must be one of {sorted(_KEYS)}, "
            f"got {distribution!r}",
        )
    required, optional = _KEYS[distribution]
    unknown = sorted(set(spec) - required - optional - {"distribution"})
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {unknown} for {distribution}")
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"{path}: missing key(s) {missing} for {distribution}")

    if distribution == "normal":
        return _parse_normal(spec, path, floor)
    if distribution == "uniform":
        return _parse_uniform(spec, path, floor)
    return _parse_choice(spec, path, floor)


def _parse_normal(
    spec: Mapping[str, Any],
    path: str,
    floor: Optional[Floor],
) -> Normal:
    mean = _number(spec["mean"], path, "mean")
    std = _number(spec["std"], path, "std")
    if std <= 0:
        raise ValueError(f"{path}: std must be positive, got {std}")
    lower = _number(spec["min"], path, "min") if "min" in spec else -math.inf
    upper = _number(spec["max"], path, "max") if "max" in spec else math.inf
    if lower >= upper:
        raise ValueError(f"{path}: min must be below max, got {lower} and {upper}")
    if floor is not None:
        lower = max(lower, floor.bound)
    span = _NORMAL_SPAN * std
    if lower >= min(upper, mean + span) or upper <= max(lower, mean - span):
        raise ValueError(
            f"{path}: truncation interval [{lower}, {upper}] misses "
            f"[{mean - span}, {mean + span}], mean +/- {_NORMAL_SPAN:g} std",
        )
    strict = floor is not None and floor.strict and lower == floor.bound
    return Normal(mean, std, lower, upper, strict)


def _parse_uniform(
    spec: Mapping[str, Any],
    path: str,
    floor: Optional[Floor],
) -> Uniform:
    low = _number(spec["low"], path, "low")
    high = _number(spec["high"], path, "high")
    if low >= high:
        raise ValueError(f"{path}: low must be below high, got {low} and {high}")
    _check_floor(low, path, floor)
    return Uniform(low, high)


def _parse_choice(
    spec: Mapping[str, Any],
    path: str,
    floor: Optional[Floor],
) -> Choice:
    values = spec["values"]
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{path}: values must be a list, got {values!r}")
    if not values:
        raise ValueError(f"{path}: values must not be empty")
    if floor is not None:
        for value in values:
            _check_floor(_number(value, path, "values"), path, floor)

    weights = spec.get("weights")
    if weights is None:
        probabilities = tuple(1.0 / len(values) for _ in values)
    else:
        if isinstance(weights, (str, bytes)) or not isinstance(weights, Sequence):
            raise ValueError(f"{path}: weights must be a list, got {weights!r}")
        if len(weights) != len(values):
            raise ValueError(
                f"{path}: weights has {len(weights)} entries, values has {len(values)}",
            )
        weights = [_number(w, path, "weights") for w in weights]
        if any(w < 0 for w in weights):
            raise ValueError(f"{path}: weights must be non-negative, got {weights}")
        total = sum(weights)
        if total <= 0:
            raise ValueError(f"{path}: weights must have a positive sum")
        probabilities = tuple(w / total for w in weights)
    return Choice(tuple(values), probabilities)


def _number(value: Any, path: str, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{path}: {key} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{path}: {key} must be a finite number, got {value!r}")
    return float(value)


def _check_floor(value: float, path: str, floor: Optional[Floor]) -> None:
    if floor is not None and not floor.admits(value):
        relation = ">" if floor.strict else ">="
        raise ValueError(
            f"{path}: every value must be {relation} {floor.bound}, got {value}",
        )


__all__ = [
    "ValueSpec",
    "Fixed",
    "Normal",
    "Uniform",
    "Choice",
    "Floor",
    "POSITIVE",
    "NON_NEGATIVE",
    "parse_value_spec",
]
