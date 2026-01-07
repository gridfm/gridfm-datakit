# Topology Perturbations

## Overview

Topology perturbations generate variations of the original network by altering its topology. These variations simulate contingencies and component failures, and are useful for robustness testing, contingency analysis, and training ML models on diverse grid conditions.

The module provides three topology perturbation strategies:

- `NoPerturbationGenerator` yields the original topology without changes. It is useful for baseline comparisons and debugging.

- `NMinusKGenerator` generates all possible combinations of up to *k* components (lines and transformers) being out of service. Only feasible topologies with no unsupplied buses are returned.

- `RandomComponentDropGenerator` randomly generates a specified number of feasible topologies by disabling up to *k* randomly selected components, including lines, transformers, generators, and static generators.

## Comparison of Perturbation Strategies

| Feature                            | `NoPerturbationGenerator` | `NMinusKGenerator`        | `RandomComponentDropGenerator` |
|-----------------------------------|----------------------------|---------------------------|--------------------------------|
| **Number of topologies**          | 1 (original)               | All feasible (up to k elements lost)    | User-defined         |
| **Component types supported**     | –                          | Lines, Transformers       | Lines, Transformers, Gens, Sgens |
| **Randomized generation**         | ❌ No                      | ❌ No                     | ✅ Yes                         |
