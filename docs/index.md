<p align="center">
  <img src="https://raw.githubusercontent.com/gridfm/gridfm-datakit/refs/heads/main/docs/figs/KIT_logo.png" alt="GridFM logo" style="width: 40%; height: auto;"/>
  <br/>
</p>

<p align="center" style="font-size: 25px;">
</p>


# GridFM DataKit

**GridFM DataKit** (`gridfm-datakit`) is a Python library for generating realistic, diverse, and scalable synthetic datasets for power flow (PF) and optimal power flow (OPF) machine learning solvers. It unifies state-of-the-art methods for perturbing loads, generator dispatches, network topologies, and branch parameters, addressing limitations of existing data generation libraries.

## Key Features

* **Scalable**: Supports grids with up to 30,000 buses for PF and 10,000 buses for OPF. Compatible with MATPOWER (`.m`) files and the PGLib dataset.
* **Realistic load scenarios**: Combines global scaling from real-world aggregated profiles with localized per-bus noise, preserving temporal and spatial correlations.
* **Flexible topology perturbations**: Handles arbitrary (N-k) outages for lines, transformers, and generators, ensuring feasible network states.
* **Generator cost diversity**: Permutes or randomly scales generator cost functions when solving OPF to produce diverse dispatches and improve generalization across different cost conditions.
* **Out-of-operating-limits scenarios for PF**: PF datasets include realistic violations of operating limits (e.g., voltage or branch overloads) resulting from topology and load perturbations without re-optimizing generator dispatch.
* **Admittance perturbations**: Randomly scales branch resistances and reactances to enhance diversity.
* **Structured outputs for ML**: Per-bus, per-branch, and per-generator data ready for training neural PF/OPF solvers, with pre-computed DC-PF and DC-OPF baselines and runtime.
* **Data validation and benchmarking**: Includes CLI tools for consistency checks, statistics, and constraint validation.


<p align="center">
  <img src="https://raw.githubusercontent.com/gridfm/gridfm-datakit/refs/heads/main/docs/figs/comparison_table.png" alt="Comparison table" style="width: 80%; height: auto;"/>
  <br/>
</p>

## Citation

Please cite the library when using it in your work:

```bibtex
@misc{puech2025gridfmdatakitv1pythonlibraryscalable,
      title={gridfm-datakit-v1: A Python Library for Scalable and Realistic Power Flow and Optimal Power Flow Data Generation},
      author={Alban Puech and Matteo Mazzonelli and Celia Cintas and Tamara R. Govindasamy and Mangaliso Mngomezulu and Jonas Weiss and Matteo Baù and Anna Varbella and François Mirallès and Kibaek Kim and Le Xie and Hendrik F. Hamann and Etienne Vos and Thomas Brunschwiler},
      year={2025},
      eprint={2512.14658},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.14658},
}
```
