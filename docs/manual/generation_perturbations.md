# Generation Perturbations

## Overview
Generation perturbations introduce random changes to the cost functions of generators. The effect of this is that the cost of operating generators in the grid changes across examples, which allows them to be utilized differently when executing optimal power flow. As a result, examples produced will have more diverse generator setpoints which is beneficial for training ML models to improve generalization. Generation perturbation is applied to the existing topology perturbations.

The module provides three options for generation perturbation strategies:

- `NoGenPerturbationGenerator` yields the original example without any additional changes in generation cost.

- `PermuteGenCostGenerator` randomly permutes the generator cost coefficients across and among generator elements.

- `PerturbGenCostGenerator` applies a scaling factor to all generator cost coefficients. The scaling factor is sampled from a uniform distribution with a range given by `[max(0, 1-sigma), 1+sigma)`, where `sigma` is a user-defined adjustable parameter.



### Constant-Cost Generators

Generators with **constant-only costs** (where `c1 = 0` and `c2 = 0`, but `c0 ≠ 0`) are **excluded from perturbation and permutation** operations. These generators maintain identical cost coefficients across all scenarios.
