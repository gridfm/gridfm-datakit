<p align="center">
  <img src="https://raw.githubusercontent.com/gridfm/gridfm-datakit/refs/heads/main/docs/figs/KIT_logo.png" alt="GridFM logo" style="width: 40%; height: auto;"/>
  <br/>
</p>

<p align="center" style="font-size: 25px;">
</p>


# GridFM datakit

**GridFM datakit** (`gridfm-datakit`) is a Python library for generating realistic, diverse, and scalable synthetic datasets for power flow (PF) and optimal power flow (OPF) machine learning solvers. It unifies state-of-the-art methods for perturbing loads, generator dispatches, network topologies, and branch parameters, addressing limitations of existing data generation libraries.

## Key Features

* **Scalable**: Supports grids with up to 30,000 buses for PF and 10,000 buses for OPF. Compatible with MATPOWER (`.m`) files and the PGLib dataset.
* **Realistic load scenarios**: Combines global scaling from real-world aggregated profiles with localized per-bus noise, preserving temporal and spatial correlations.
* **Flexible topology perturbations**: Handles arbitrary (N-k) outages for lines, transformers, and generators, ensuring feasible network states.
* **Generator cost diversity**: Permutes or randomly scales generator cost functions when solving OPF to produce diverse dispatches and improve generalization across different cost conditions.
* **Out-of-operating-limits scenarios for PF**: PF datasets include realistic violations of operating limits (e.g., voltage or branch overloads) resulting from topology and load perturbations without re-optimizing generator dispatch.
* **Admittance perturbations**: Randomly scales branch resistances and reactances to enhance diversity.
* **Structured outputs for ML**: Per-bus, per-branch, and per-generator data ready for training neural PF/OPF solvers, with pre-computed DC-PF and DC-OPF baselines and runtime.
* **Dynamic (time-domain) simulation**: Optional [Dynaωo](https://dynawo.github.io/) backend that runs an RMS simulation from each scenario's balanced operating point, producing trajectories alongside the static snapshot. See [Dynamic Simulation](manual/dynamic_simulation.md).
* **Data validation and benchmarking**: Includes CLI tools for consistency checks, statistics, and constraint validation.


<p align="center">
  <img src="https://raw.githubusercontent.com/gridfm/gridfm-datakit/refs/heads/main/docs/figs/comparison_table.png" alt="Comparison table" style="width: 80%; height: auto;"/>
  <br/>
</p>

## Citation

If you use `gridfm-datakit` in your research, please cite both:

```bibtex
@article{puech2025gridfmdatakitv1pythonlibraryscalable,
  title={gridfm-datakit-v1: A Python Library for Scalable and Realistic Power Flow and Optimal Power Flow Data Generation},
  author={Alban Puech and Matteo Mazzonelli and Celia Cintas and Tamara R. Govindasamy and Mangaliso Mngomezulu and Jonas Weiss and Matteo Baù and Anna Varbella and François Mirallès and Kibaek Kim and Le Xie and Hendrik F. Hamann and Etienne Vos and Thomas Brunschwiler},
  journal={arXiv preprint arXiv:2512.14658},
  year={2025},
  url={https://arxiv.org/abs/2512.14658}
}


@article{puech2026gencounifiedneural,
  title={GENCO - A Unified Neural Solver Embedded in a Development Framework for Steady-State Grid Analysis},
  author={Alban Puech and Matteo Mazzonelli and Tamara R. Govindasamy and Mangaliso Mngomezulu and Héctor Maeso-García and Thomas Tolhurst and Javad Bayazi and Ali Moeini and Naomi Simumba and Celia Cintas and David Nelischer and Romeo Kienzler and Jonas Weiss and Anna Varbella and Florian Dörfler and Gabriela Hug and Martin Mevissen and Juan Bernabé-Moreno and François Mirallès and Hendrik F. Hamann and Etienne Vos and Thomas Brunschwiler},
  journal={arXiv preprint arXiv:2608.09921},
  year={2026},
  url={https://arxiv.org/abs/2608.09921}
}
```
