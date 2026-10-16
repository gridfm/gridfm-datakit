"""Random dynamic events: the ``dynamic.event_perturbation`` block.

Form:
    type: none | file | random          required when the block is present
    n_event_variants: M                 positive int, default 1, 1 unless random
    scenarios:                          random only, one or more
      - name: str                       unique, recorded in events.parquet
        anchor: random | <bus ID>       default random, places distance targets
        weight: float                   > 0, default 1
        start_time: <value spec>        inside the solver window
        events:                         one or more, targets distinct
          - type: <event type>
            target: {element: <element>, distance: int | [min, max]}
                  | {static_id: <network file ID>}
            params: {<param>: <value spec>}   every param, none for Disconnect
            delay: <value spec>         >= 0, default 0, time start_time + delay

Event types, allowed elements, params and floors (``_EVENTS``):
    Disconnect                 every ELEMENT_TYPES
    NodeFault                  bus                 fault_time > 0, r_pu >= 0, x_pu >= 0
    ActivePowerVariation       generator, load     delta_p any sign
    ReactivePowerVariation     generator, load     delta_q any sign
    ReferenceVoltageVariation  generator           delta_u any sign

``_EVENTS`` copies the keys of ``dynawo.utils.EVENT_PARAMS_MAPPING``: importing
it here is circular. A test keeps them in sync.

Fixed target: checked on the base network at config load, an in-service element
of a type the event accepts. Out of service in a topology variant, it fails that
variant. Distance targets of its scenario never pick it. It draws nothing, and a
scenario of fixed targets only draws no anchor.

Window, per event: ``start_time + delay``, plus ``fault_time`` for a NodeFault,
must not exceed the stop time, checked on the largest value each spec can take.
The other types are instantaneous.

Without the block, the events are the rows of ``events_file``, as with
``type: file``.

Seeding: one ``default_rng([seed, scenario_index, perturbation_index,
event_index])`` per event variant, so a draw does not depend on chunking or
process count, unless the topology perturbation is random, which draws per chunk
and so moves the targets. Draw order: the scenario by weight, only when there
are several; the placement of its distance targets; its start time; then per
event in list order its delay and its params in table order. A placement failure
raises: another scenario would bias the weights. A fixed spec draws nothing, so
no delay and Disconnect cost no draw.

The M event variants of a topology variant share its balanced state, computed
once. A Dynawo run writes its final state into its variant, so every event
variant but the last runs on a clone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from gridfm_datakit.dynamic.placement import (
    ELEMENT_TYPES,
    EventGraph,
    PlacementError,
    place_scenario,
)
from gridfm_datakit.utils.param_handler import NestedNamespace
from gridfm_datakit.utils.value_spec import (
    NON_NEGATIVE,
    POSITIVE,
    Fixed,
    Floor,
    ValueSpec,
    parse_value_spec,
)

EVENT_COLUMNS = ["event_name", "static_id", "start_time", "params"]
EVENT_RECORD_COLUMNS = ["scenario"] + EVENT_COLUMNS

_PATH = "dynamic.event_perturbation"
_TYPES = ("none", "file", "random")
_ANY_NUMBER = Floor(-math.inf, strict=False)
_DISCONNECT_PARAMS = "disconnect_only=;"
_NO_DELAY = Fixed(0.0)
_EVENTS = {
    "Disconnect": (ELEMENT_TYPES, ()),
    "NodeFault": (
        ("bus",),
        (("fault_time", POSITIVE), ("r_pu", NON_NEGATIVE), ("x_pu", NON_NEGATIVE)),
    ),
    "ActivePowerVariation": (("generator", "load"), (("delta_p", _ANY_NUMBER),)),
    "ReactivePowerVariation": (("generator", "load"), (("delta_q", _ANY_NUMBER),)),
    "ReferenceVoltageVariation": (("generator",), (("delta_u", _ANY_NUMBER),)),
}


@dataclass(frozen=True)
class EventTarget:
    """The element an event applies to, by hop distance from the anchor.

    Args:
        element: One of ``ELEMENT_TYPES``.
        low: The smallest allowed distance.
        high: The largest allowed distance.
    """

    element: str
    low: int
    high: int


@dataclass(frozen=True)
class FixedTarget:
    """A named element an event applies to.

    Args:
        static_id: The element ID in the network file.
    """

    static_id: str


@dataclass(frozen=True)
class EventSpec:
    """One event of a scenario.

    Args:
        type: The Dynawo event name.
        target: Where the event applies: an ``EventTarget`` drawn by distance
            from the anchor, or a ``FixedTarget``.
        params: The ``(param, spec)`` pairs of the type, in ``_EVENTS`` order.
        delay: The time after the scenario's start time.
    """

    type: str
    target: Union[EventTarget, FixedTarget]
    params: Tuple[Tuple[str, ValueSpec], ...] = ()
    delay: ValueSpec = _NO_DELAY


@dataclass(frozen=True)
class EventScenario:
    """A set of events drawn around one anchor bus.

    Args:
        name: The scenario name, used in error messages and recorded with every
            drawn event.
        anchor: A bus ID, or ``None`` to draw one.
        start_time: The event time spec.
        events: The events of the scenario.
        weight: The relative chance of drawing this scenario.
    """

    name: str
    anchor: Optional[str]
    start_time: ValueSpec
    events: Tuple[EventSpec, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class EventPerturbation:
    """The parsed ``dynamic.event_perturbation`` block.

    Args:
        type: ``none``, ``file`` or ``random``.
        n_event_variants: The event variants per topology variant.
        scenarios: The scenarios of ``random`` mode, one drawn per event
            variant.
    """

    type: str = "file"
    n_event_variants: int = 1
    scenarios: Tuple[EventScenario, ...] = ()


def parse_event_perturbation(
    block: Any,
    start_time: Optional[float],
    stop_time: Optional[float],
) -> EventPerturbation:
    """Parse the ``dynamic.event_perturbation`` block.

    Args:
        block: The block as a mapping or a ``NestedNamespace``, or ``None``.
        start_time: The simulation start time, ``None`` when not configured.
        stop_time: The simulation stop time, ``None`` when not configured.

    Returns:
        EventPerturbation: The parsed block, ``EventPerturbation()`` when absent.

    Raises:
        ValueError: If the block is malformed. The message starts with the config
            path of the offending entry.
    """
    if block is None:
        return EventPerturbation()
    if isinstance(block, NestedNamespace):
        block = block.to_dict()
    _check_keys(block, _PATH, {"type"}, {"n_event_variants", "scenarios"})

    kind = block["type"]
    if kind not in _TYPES:
        raise ValueError(f"{_PATH}.type: must be one of {list(_TYPES)}, got {kind!r}")

    n_event_variants = block.get("n_event_variants", 1)
    if (
        isinstance(n_event_variants, bool)
        or not isinstance(n_event_variants, int)
        or n_event_variants < 1
    ):
        raise ValueError(
            f"{_PATH}.n_event_variants: must be a positive integer, "
            f"got {n_event_variants!r}",
        )

    if kind != "random":
        if n_event_variants != 1:
            raise ValueError(
                f"{_PATH}.n_event_variants: must be 1 for type {kind!r}, "
                f"got {n_event_variants}",
            )
        if "scenarios" in block:
            raise ValueError(
                f"{_PATH}.scenarios: only accepted with type 'random', "
                f"got type {kind!r}",
            )
        return EventPerturbation(kind, n_event_variants)

    if start_time is None or stop_time is None:
        raise ValueError(
            f"{_PATH}: type 'random' needs dynamic.solver_parameters.start_time "
            "and stop_time to check event times",
        )
    scenarios = block.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(
            f"{_PATH}.scenarios: must be a non-empty list of scenarios, "
            f"got {scenarios!r}",
        )
    parsed = []
    for index, entry in enumerate(scenarios):
        path = f"{_PATH}.scenarios[{index}]"
        scenario = _parse_scenario(entry, path, float(start_time), float(stop_time))
        if any(scenario.name == other.name for other in parsed):
            raise ValueError(
                f"{path}.name: {scenario.name!r} is already the name of an "
                "earlier scenario",
            )
        parsed.append(scenario)
    return EventPerturbation(kind, n_event_variants, tuple(parsed))


def check_event_perturbation(
    perturbation: EventPerturbation,
    network_path: str,
) -> None:
    """Check fixed anchors and fixed targets against the base network.

    Args:
        perturbation: The parsed block.
        network_path: The network file.

    Raises:
        ValueError: If a fixed target is not an in-service element of a type its
            event accepts, or a fixed anchor is not a bus of the network or
            cannot place its distance targets. The message starts with the
            config path of the target or the anchor.
    """
    anchors = [
        (index, scenario)
        for index, scenario in enumerate(perturbation.scenarios)
        if scenario.anchor is not None
    ]
    targets = [
        (index, position, event)
        for index, scenario in enumerate(perturbation.scenarios)
        for position, event in enumerate(scenario.events)
        if isinstance(event.target, FixedTarget)
    ]
    if not anchors and not targets:
        return
    from gridfm_datakit.powsybl import load_net

    graph = EventGraph.from_network(load_net(network_path).pp_net)
    for index, position, event in targets:
        path = f"{_PATH}.scenarios[{index}].events[{position}].target.static_id"
        static_id = event.target.static_id
        element = graph.element_type(static_id)
        if element is None:
            raise ValueError(
                f"{path}: {static_id!r} is not an in-service bus, generator, load, "
                "line or transformer of the network",
            )
        elements = _EVENTS[event.type][0]
        if element not in elements:
            raise ValueError(
                f"{path}: {event.type} accepts {list(elements)}, "
                f"{static_id!r} is a {element}",
            )
    for index, scenario in anchors:
        try:
            place_scenario(
                graph,
                scenario.anchor,
                _targets(scenario),
                np.random.default_rng(0),
                exclude=_fixed_ids(scenario),
            )
        except (ValueError, PlacementError) as error:
            raise ValueError(
                f"{_PATH}.scenarios[{index}].anchor: {error}",
            ) from error


def draw_events(
    perturbation: EventPerturbation,
    graph: EventGraph,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw the events of one event variant.

    Args:
        perturbation: The parsed block.
        graph: The graph of the topology variant.
        rng: The generator of the event variant.

    Returns:
        pd.DataFrame: One row per event, columns ``EVENT_RECORD_COLUMNS``.

    Raises:
        PlacementError: If the drawn scenario cannot be placed, or one of its
            fixed targets is out of service in this variant.
    """
    rows = []
    scenarios = perturbation.scenarios
    if len(scenarios) > 1:
        weights = np.array([scenario.weight for scenario in scenarios])
        scenarios = (scenarios[rng.choice(len(scenarios), p=weights / weights.sum())],)
    for scenario in scenarios:
        static_ids = _place_events(scenario, graph, rng)
        start = float(scenario.start_time.sample(rng))
        for event, static_id in zip(scenario.events, static_ids):
            delay = float(event.delay.sample(rng))
            rows.append(
                (
                    scenario.name,
                    event.type,
                    static_id,
                    start + delay,
                    _draw_params(event, rng),
                ),
            )
    return pd.DataFrame(rows, columns=EVENT_RECORD_COLUMNS).astype(
        {"start_time": float},
    )


def _draw_params(event: EventSpec, rng: np.random.Generator) -> str:
    if not event.params:
        return _DISCONNECT_PARAMS
    return ";".join(f"{key}={float(spec.sample(rng))!r}" for key, spec in event.params)


def _place_events(
    scenario: EventScenario,
    graph: EventGraph,
    rng: np.random.Generator,
) -> List[str]:
    fixed = _fixed_ids(scenario)
    for static_id in fixed:
        if graph.element_type(static_id) is None:
            raise PlacementError(
                f"fixed target {static_id!r} is out of service in this variant",
            )
    targets = _targets(scenario)
    placed = iter(
        place_scenario(graph, scenario.anchor, targets, rng, exclude=fixed)[1]
        if targets
        else [],
    )
    return [
        (
            event.target.static_id
            if isinstance(event.target, FixedTarget)
            else next(placed)
        )
        for event in scenario.events
    ]


def _targets(scenario: EventScenario) -> list:
    return [
        (event.target.element, event.target.low, event.target.high)
        for event in scenario.events
        if isinstance(event.target, EventTarget)
    ]


def _fixed_ids(scenario: EventScenario) -> List[str]:
    return [
        event.target.static_id
        for event in scenario.events
        if isinstance(event.target, FixedTarget)
    ]


def _parse_scenario(
    block: Any,
    path: str,
    start_time: float,
    stop_time: float,
) -> EventScenario:
    _check_keys(block, path, {"name", "start_time", "events"}, {"anchor", "weight"})

    name = block["name"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"{path}.name: must be a non-empty string, got {name!r}")

    anchor = block.get("anchor", "random")
    if not isinstance(anchor, str) or not anchor:
        raise ValueError(
            f"{path}.anchor: must be 'random' or a bus ID, got {anchor!r}",
        )

    weight = block.get("weight", 1)
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight <= 0
    ):
        raise ValueError(
            f"{path}.weight: must be a positive finite number, got {weight!r}",
        )

    spec = parse_value_spec(block["start_time"], f"{path}.start_time", POSITIVE)
    low, high = spec.support()
    if low < start_time or high > stop_time:
        raise ValueError(
            f"{path}.start_time: values in [{low}, {high}] fall outside the "
            f"simulation window [{start_time}, {stop_time}]",
        )

    events = block["events"]
    if not isinstance(events, list) or not events:
        raise ValueError(
            f"{path}.events: must be a non-empty list of events, got {events!r}",
        )
    parsed = []
    for index, entry in enumerate(events):
        event_path = f"{path}.events[{index}]"
        event = _parse_event(entry, event_path)
        target = event.target
        if isinstance(target, FixedTarget) and any(
            target == other.target for other in parsed
        ):
            raise ValueError(
                f"{event_path}.target.static_id: {event.target.static_id!r} is "
                "already the target of an earlier event",
            )
        end = high + event.delay.support()[1]
        if end > stop_time:
            raise ValueError(
                f"{event_path}.delay: an event at up to {end} falls after the "
                f"stop time {stop_time}",
            )
        fault_time = dict(event.params).get("fault_time")
        if fault_time is not None and end + fault_time.support()[1] > stop_time:
            raise ValueError(
                f"{event_path}.params.fault_time: a fault starting by {end} and "
                f"lasting up to {fault_time.support()[1]} ends after the stop "
                f"time {stop_time}",
            )
        parsed.append(event)
    return EventScenario(
        name=name,
        anchor=None if anchor == "random" else anchor,
        start_time=spec,
        events=tuple(parsed),
        weight=float(weight),
    )


def _parse_event(block: Any, path: str) -> EventSpec:
    _check_keys(block, path, {"type", "target"}, {"params", "delay"})
    kind = block["type"]
    if not isinstance(kind, str) or kind not in _EVENTS:
        raise ValueError(f"{path}.type: must be one of {list(_EVENTS)}, got {kind!r}")
    target = _parse_target(block["target"], f"{path}.target")
    elements, floors = _EVENTS[kind]
    if isinstance(target, EventTarget) and target.element not in elements:
        raise ValueError(
            f"{path}.target.element: {kind} accepts {list(elements)}, "
            f"got {target.element!r}",
        )
    delay = (
        parse_value_spec(block["delay"], f"{path}.delay", NON_NEGATIVE)
        if "delay" in block
        else _NO_DELAY
    )

    if not floors:
        if "params" in block:
            raise ValueError(f"{path}.params: {kind} takes no params")
        return EventSpec(kind, target, delay=delay)
    if "params" not in block:
        raise ValueError(f"{path}.params: required key is missing")
    params = block["params"]
    _check_keys(params, f"{path}.params", {key for key, _ in floors}, set())
    return EventSpec(
        kind,
        target,
        tuple(
            (key, parse_value_spec(params[key], f"{path}.params.{key}", floor))
            for key, floor in floors
        ),
        delay,
    )


def _parse_target(block: Any, path: str) -> Union[EventTarget, FixedTarget]:
    if isinstance(block, Mapping) and "static_id" in block:
        _check_keys(block, path, {"static_id"}, set())
        static_id = block["static_id"]
        if not isinstance(static_id, str) or not static_id:
            raise ValueError(
                f"{path}.static_id: must be a non-empty string, got {static_id!r}",
            )
        return FixedTarget(static_id)
    _check_keys(block, path, {"element", "distance"}, set())
    element = block["element"]
    if element not in ELEMENT_TYPES:
        raise ValueError(
            f"{path}.element: must be one of {list(ELEMENT_TYPES)}, got {element!r}",
        )

    distance = block["distance"]
    bounds = distance if isinstance(distance, list) else [distance, distance]
    if (
        len(bounds) != 2
        or any(isinstance(b, bool) or not isinstance(b, int) for b in bounds)
        or bounds[0] < 0
        or bounds[0] > bounds[1]
    ):
        raise ValueError(
            f"{path}.distance: must be a non-negative integer or an inclusive "
            f"[min, max] of them, got {distance!r}",
        )
    return EventTarget(element, bounds[0], bounds[1])


def _check_keys(block: Any, path: str, required: set, optional: set) -> None:
    if not isinstance(block, Mapping):
        raise ValueError(f"{path}: must be a mapping, got {block!r}")
    unknown = sorted(set(block) - required - optional)
    if unknown:
        raise ValueError(
            f"{path}.{unknown[0]}: unknown key, accepted {sorted(required | optional)}",
        )
    missing = sorted(required - set(block))
    if missing:
        raise ValueError(f"{path}.{missing[0]}: required key is missing")


__all__ = [
    "EVENT_COLUMNS",
    "EVENT_RECORD_COLUMNS",
    "EventTarget",
    "FixedTarget",
    "EventSpec",
    "EventScenario",
    "EventPerturbation",
    "parse_event_perturbation",
    "check_event_perturbation",
    "draw_events",
]
