# Admittance Perturbations

## Overview
Admittance perturbations introduce changes to branch admittance values by applying random scaling factors to the resistance ($R$) and reactance ($X$) parameters of grid branches. This results in more variance and diversity in power flow solutions which is beneficial for training ML models to improve generalization.

The module provides two options for admittance perturbation strategies:

- `NoAdmittancePerturbationGenerator` yields the original example without any additional changes in branch admittances.

- `PerturbAdmittanceGenerator` applies a scaling factor to all resistance and reactance values of network branches. The scaling factor is sampled from a uniform distribution with a range given by `[max(0, 1-sigma), 1+sigma)`, where `sigma` is a user-defined adjustable parameter.
