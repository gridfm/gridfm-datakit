# Topology Perturbations

## Overview

Topology perturbations generate variations of the original network by altering its topology. These variations simulate contingencies and component failures, and are useful for robustness testing, contingency analysis, and training ML models on diverse grid conditions.

The module provides three topology perturbation strategies:

- `NoPerturbationGenerator` yields the original topology without changes. It is useful for baseline comparisons and debugging.
- `NMinusKGenerator` generates all possible combinations of up to *k* components (lines and transformers) being out of service. Only feasible topologies with no unsupplied buses are returned.
- `RandomComponentDropGenerator` randomly generates a specified number of feasible topologies by disabling up to *k* randomly selected components, including lines, transformers, generators, and static generators.

For `type: random`, you can optionally control the outage-count mix directly with a probability vector whose index `i` represents the probability of sampling `i` outages.

Example:

```yaml
topology_perturbation:
  type: "random"
  k: 2
  n_topology_variants: 20
  elements: [branch, gen]
  outage_count_probabilities: [0.1, 0.8, 0.1]
```

This requests approximately 10% N-0, 80% N-1, and 10% N-2 attempted perturbations. Realized proportions can differ slightly because infeasible sampled topologies are rejected.

## Comparison of Perturbation Strategies

| Feature                             | `NoPerturbationGenerator` | `NMinusKGenerator`                 | `RandomComponentDropGenerator` |
| ----------------------------------- | --------------------------- | ------------------------------------ | -------------------------------- |
| **Number of topologies**      | 1 (original)                | All feasible (up to k elements lost) | User-defined                     |
| **Component types supported** | –                          | Lines, Transformers                  | Lines, Transformers, Gens, Sgens |
| **Randomized generation**     | ❌ No                       | ❌ No                                | ✅ Yes                           |
