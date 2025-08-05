# Generation Perturbations

## Overview
Generation perturbations introduce random changes to the cost functions of generators and static generators (`gens` and `sgens`) in the Panda Power `poly_cost` table.  The effect of this is that the cost of operating generators in the grid changes across examples, which allows them to be utilised differently when executing optimal power flow.  As a result, examples produced will have more diverse generator setpoints which is beneficial for training ML models to improve generalisation.  Generation perturbation is applied to the existing topology perturbations.

The module provides three options for generation perturbation strategies:

- `NoGenPerturbationGenerator` yields the original example produced by the topology perturbation generator without any additional changes in generation cost.

- `PermuteGenCostGenerator` randomly permutes the rows of generator cost coefficients in the `poly_cost` table across and among generator elements.

- `PerturbGenCostGenerator` applies a scaling factor to all generator cost coefficients in the `poly_cost` table.  The scaling factor is sampled from a uniform distribution with a range given by `[max(0, 1-sigma), 1+sigma)`, where `sigma` is a user-defined adjustable parameter.
