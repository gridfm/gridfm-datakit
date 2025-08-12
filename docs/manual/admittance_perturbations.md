# Generation Perturbations

## Overview
Admittance perturbations introduce changes to line admittance values by applpying random scaling factors to the resistance ($R$) and reactance ($X$) parameters of grid lines.  Admittance ($Y$) is related to impedance ($Z$) through $Y=1/Z$, and the impedance, in turn, is related to resistance and reactance through $Z=R+jX$. This results in more variance and diversity in power flow solutions which is beneficial for training ML models to improve generalisation.  Admittance perturbations are applied to the existing topology and generation perturbations.

The module provides two options for admittance perturbation strategies:

- `NoAdmittancePerturbationGenerator` yields the original example produced by the generation perturbation generator without any additional changes in line admittances.

- `PerturbAdmittanceGenerator` applies a scaling factor to all resistance and reactance values of network lines.  The scaling factor is sampled from a uniform distribution with a range given by `[max(0, 1-sigma), 1+sigma)`, where `sigma` is a user-defined adjustable parameter.
