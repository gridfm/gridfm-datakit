"""Place dynamic event targets by graph distance from an anchor bus.

Graph: the working variant of the network. Nodes are the buses of the
bus-breaker view, edges the lines and two-winding transformers connected at both
ends, each pair of connected ends of a three-winding transformer, and the closed
switches of the view, so a graph built after the topology perturbation sees what
was cut. A three-winding transformer is an edge only, never a target.

Distance: hop count from the anchor. A closed switch adds no hop, so the buses it
joins share a distance. A generator or load sits at its bus's distance, a branch
at its nearer end's. Unreachable elements are never candidates.

Targets are drawn in list order, each among the candidates that leave the later
targets a distinct element each (bipartite matching), so a placement fails only
when no assignment exists. A random anchor is redrawn when none does, at most
``MAX_ANCHOR_ATTEMPTS`` times, so a scenario unplaceable from most buses fails
the variant instead of spinning.

``exclude``: IDs no target may take, the scenario's fixed targets.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ELEMENT_TYPES = ("bus", "generator", "load", "line", "transformer")
MAX_ANCHOR_ATTEMPTS = 100


class PlacementError(RuntimeError):
    """No placement satisfies the targets."""


@dataclass(frozen=True)
class EventGraph:
    """Bus graph and event candidates of one network variant.

    Args:
        adjacency: The neighbours of every bus.
        candidates: Per element type, the element IDs sorted, each with its buses.
        couplings: The buses each bus is joined to by a closed switch.
    """

    adjacency: Mapping[str, Tuple[str, ...]]
    candidates: Mapping[str, Tuple[Tuple[str, Tuple[str, ...]], ...]]
    couplings: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_network(cls, pp_net: Any) -> EventGraph:
        """Build the graph from the working variant of a pypowsybl network.

        Args:
            pp_net: The pypowsybl network.

        Returns:
            EventGraph: The graph and its candidates.
        """
        buses = sorted(pp_net.get_bus_breaker_view_buses().index)
        branch_attributes = [
            "bus_breaker_bus1_id",
            "bus_breaker_bus2_id",
            "connected1",
            "connected2",
        ]
        branches = {
            "line": pp_net.get_lines(attributes=branch_attributes),
            "transformer": pp_net.get_2_windings_transformers(
                attributes=branch_attributes,
            ),
        }
        injections = {
            "generator": pp_net.get_generators(
                attributes=["bus_breaker_bus_id", "connected"],
            ),
            "load": pp_net.get_loads(attributes=["bus_breaker_bus_id", "connected"]),
        }

        neighbours: Dict[str, set] = {bus: set() for bus in buses}
        candidates: Dict[str, List[Tuple[str, Tuple[str, ...]]]] = {
            "bus": [(bus, (bus,)) for bus in buses],
        }
        for element, frame in branches.items():
            connected = frame[frame["connected1"] & frame["connected2"]]
            candidates[element] = []
            for branch_id, row in connected.iterrows():
                bus1, bus2 = row["bus_breaker_bus1_id"], row["bus_breaker_bus2_id"]
                neighbours[bus1].add(bus2)
                neighbours[bus2].add(bus1)
                candidates[element].append((branch_id, (bus1, bus2)))
        three_windings = pp_net.get_3_windings_transformers(
            attributes=[
                "bus_breaker_bus1_id",
                "bus_breaker_bus2_id",
                "bus_breaker_bus3_id",
                "connected1",
                "connected2",
                "connected3",
            ],
        )
        for _, row in three_windings.iterrows():
            ends = [
                row[f"bus_breaker_bus{i}_id"] for i in (1, 2, 3) if row[f"connected{i}"]
            ]
            for bus1, bus2 in combinations(ends, 2):
                neighbours[bus1].add(bus2)
                neighbours[bus2].add(bus1)
        couplings: Dict[str, set] = {bus: set() for bus in buses}
        switches = pp_net.get_switches(
            attributes=["bus_breaker_bus1_id", "bus_breaker_bus2_id", "open"],
        )
        closed = switches[
            ~switches["open"]
            & (switches["bus_breaker_bus1_id"] != "")
            & (switches["bus_breaker_bus2_id"] != "")
        ]
        for bus1, bus2 in zip(
            closed["bus_breaker_bus1_id"],
            closed["bus_breaker_bus2_id"],
        ):
            couplings[bus1].add(bus2)
            couplings[bus2].add(bus1)
        for element, frame in injections.items():
            connected = frame[frame["connected"]]
            candidates[element] = [
                (element_id, (bus,))
                for element_id, bus in connected["bus_breaker_bus_id"].items()
            ]

        return cls(
            adjacency={bus: tuple(sorted(n)) for bus, n in neighbours.items()},
            candidates={
                element: tuple(sorted(candidates[element])) for element in ELEMENT_TYPES
            },
            couplings={bus: tuple(sorted(c)) for bus, c in couplings.items() if c},
        )

    def distances(self, anchor: str) -> Dict[str, int]:
        """Return the hop count from the anchor to every reachable bus.

        Args:
            anchor: The bus ID to start from.

        Returns:
            Dict[str, int]: The distance of every bus reachable from ``anchor``.

        Raises:
            ValueError: If ``anchor`` is not a bus of the graph.
        """
        if anchor not in self.adjacency:
            raise ValueError(f"unknown anchor bus {anchor!r}")
        distance = {anchor: 0}
        queue = deque([anchor])
        while queue:
            bus = queue.popleft()
            for neighbour in self.couplings.get(bus, ()):
                if distance.get(neighbour, math.inf) > distance[bus]:
                    distance[neighbour] = distance[bus]
                    queue.appendleft(neighbour)
            for neighbour in self.adjacency[bus]:
                if distance.get(neighbour, math.inf) > distance[bus] + 1:
                    distance[neighbour] = distance[bus] + 1
                    queue.append(neighbour)
        return distance

    def element_type(self, element_id: str) -> Optional[str]:
        """Return the type of a candidate element.

        Args:
            element_id: The element ID.

        Returns:
            Optional[str]: The ``ELEMENT_TYPES`` entry whose candidates hold the
            ID, ``None`` if none does.
        """
        for element in ELEMENT_TYPES:
            if any(
                candidate == element_id for candidate, _ in self.candidates[element]
            ):
                return element
        return None


def place_scenario(
    graph: EventGraph,
    anchor: Optional[str],
    targets: Sequence[Tuple[str, int, int]],
    rng: np.random.Generator,
    exclude: Sequence[str] = (),
) -> Tuple[str, List[str]]:
    """Draw an anchor bus and one element per target.

    Args:
        graph: The graph of the network variant.
        anchor: A bus ID, or ``None`` to draw one.
        targets: ``(element, low, high)`` per target, distances inclusive.
        rng: The generator every draw comes from.
        exclude: Element IDs no target may take.

    Returns:
        Tuple[str, List[str]]: The anchor and the drawn element IDs, in target
        order.

    Raises:
        ValueError: If the anchor is unknown, an element type is not in
            ``ELEMENT_TYPES``, or a distance range is invalid.
        PlacementError: If a fixed anchor cannot place every target, or no
            random anchor can in ``MAX_ANCHOR_ATTEMPTS`` draws.
    """
    for element, low, high in targets:
        if element not in ELEMENT_TYPES:
            raise ValueError(
                f"unknown element type {element!r}, accepted: {list(ELEMENT_TYPES)}",
            )
        if low < 0 or low > high:
            raise ValueError(
                f"invalid distance range [{low}, {high}] for {element}",
            )

    if anchor is not None:
        placed = _place_from(graph, graph.distances(anchor), targets, rng, exclude)
        if placed is None:
            raise PlacementError(
                f"cannot place {list(targets)} from anchor {anchor!r}",
            )
        return anchor, placed

    buses = sorted(graph.adjacency)
    for _ in range(MAX_ANCHOR_ATTEMPTS):
        drawn = buses[rng.integers(len(buses))]
        placed = _place_from(graph, graph.distances(drawn), targets, rng, exclude)
        if placed is not None:
            return drawn, placed
    raise PlacementError(
        f"cannot place {list(targets)} from a random anchor in "
        f"{MAX_ANCHOR_ATTEMPTS} attempts",
    )


def _place_from(
    graph: EventGraph,
    distance: Dict[str, int],
    targets: Sequence[Tuple[str, int, int]],
    rng: np.random.Generator,
    exclude: Sequence[str],
) -> Optional[List[str]]:
    pools = [
        [
            element_id
            for element_id, buses in graph.candidates[element]
            if element_id not in exclude
            and _element_distance(distance, buses) in range(low, high + 1)
        ]
        for element, low, high in targets
    ]
    placed: List[str] = []
    for index, pool in enumerate(pools):
        rest = pools[index + 1 :]
        shared = set().union(*rest)
        rest_placeable = _matchable(rest, placed)
        allowed = [
            element_id
            for element_id in pool
            if element_id not in placed
            and (
                _matchable(rest, placed + [element_id])
                if element_id in shared
                else rest_placeable
            )
        ]
        if not allowed:
            return None
        placed.append(allowed[rng.integers(len(allowed))])
    return placed


def _matchable(pools: Sequence[List[str]], taken: Sequence[str]) -> bool:
    owner: Dict[str, int] = {}

    def augment(index: int, seen: set) -> bool:
        for element_id in pools[index]:
            if element_id in taken or element_id in seen:
                continue
            seen.add(element_id)
            if element_id not in owner or augment(owner[element_id], seen):
                owner[element_id] = index
                return True
        return False

    return all(augment(index, set()) for index in range(len(pools)))


def _element_distance(
    distance: Dict[str, int],
    buses: Tuple[str, ...],
) -> Optional[int]:
    reachable = [distance[bus] for bus in buses if bus in distance]
    return min(reachable) if reachable else None


__all__ = [
    "ELEMENT_TYPES",
    "MAX_ANCHOR_ATTEMPTS",
    "EventGraph",
    "PlacementError",
    "place_scenario",
]
