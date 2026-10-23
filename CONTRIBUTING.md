# Contribution Checklist ✅

Before opening a PR, make sure you complete all steps:

### 1. Development setup
- [ ] Install dev and test dependencies:
  ```bash
  pip install -e ".[dev,test]"
  ````

* [ ] Create a new branch for your feature from main



### 2. Code quality

* [ ] Write clear, readable code with self-explanatory variable names.
* [ ] Comment code where needed.
* [ ] Prefer the clearest implementation if multiple options have similar complexity.

### 3. Code style & documentation

* [ ] Add Google-style Python docstrings for all functions/classes.
* [ ] Use type hints for all function arguments and return values.
* [ ] **Double-check that no debug prints or temporary files are left in the code.**
* [ ] Update documentation:

  * [ ] `docs/components/` → add markers to your functions.
  * [ ] `docs/manual/` → update relevant sections.

### 4. Configuration & dependencies

* [ ] Add any new dependencies to `pyproject.toml`.
* [ ] If introducing new parameters, update all relevant YAML files:

  * `scripts/config/`
  * `tests/config`

### 5. Testing
* [ ] Ask your favorite code assistant to identify bugs and edge cases.
* [ ] Add unit tests covering:

  * Core functionality of your changes.
  * Potential edge cases.
* [ ] Run all tests:

  ```bash
  pytest tests/ -n <number_of_cores> -v
  ```
* [ ] Run the end-to-end integration tests, see [Running the integration tests](#running-the-integration-tests).

### 6. Pre-commit & linting

* [ ] Run pre-commit hooks:

  ```bash
  pre-commit run --all
  ```
* [ ] Fix any issues:

  ```bash
  pip install ruff
  ruff check . --fix
  ```
* [ ] Manually fix remaining issues (if any).
* [ ] Re-run pre-commit hooks after any code changes.

### 7. Documentation build

* [ ] Build docs and check rendering:

  ```bash
  mkdocs build
  mkdocs serve
  ```

### 8. Pull request

* [ ] **Rebase your branch onto the latest `main`/`dev` before opening a PR.**
* [ ] Open a PR with a short description of your changes.
* [ ] Ensure code, tests, and documentation are clear and complete.

## Running the integration tests

`integrationtests/` holds end-to-end tests that exercise the whole pipeline the way a
user does. For `case14_ieee` (200 load scenarios x 2 topology variants, fixed seed) in
both `pf` and `opf` mode, each test:

1. builds a config from `scripts/config/default.yaml`,
2. runs `gridfm_datakit generate` and then `gridfm_datakit validate --n-partitions 0`,
3. computes summary metrics from the generated parquet files (dataset size, total
   load and dispatch, generation cost, voltages, branch flows and loading, DC-vs-AC
   angle differences), and
4. checks them against calibrated bounds read from
   `integrationtests/calibration_baseline.json`.

`validate` checks that every scenario is physically valid; the calibrated bounds
catch changes in *what* is generated, which `validate` cannot see.

They are not part of `pytest tests/` and take a few minutes. Run them with:

```bash
pytest integrationtests -s
```

The committed baseline was calibrated on unchanged code. Generation runs on the CPU
and is seeded, so it is bit-reproducible on one machine and the default bounds are
`mean ± 0.1%`. If you run on a different machine the test warns that the environment
fingerprint differs; if a bound then fails **before** you have changed any code,
calibrate a local baseline first:

```bash
# On an unchanged checkout:
export GRIDFM_CALIBRATION_BASELINE=~/gridfm-datakit-baseline.json
pytest integrationtests --calibrate -s
# Make your changes, then assert against your baseline:
pytest integrationtests -s
```

If you calibrate *after* changing the code, the baseline just encodes your changed
behaviour and can no longer catch regressions.

If your change is *meant* to alter the generated data (e.g. a new perturbation
default), recalibrate the committed baseline in the same PR and explain the metric
shifts in the PR description, so reviewers can see what moved and by how much.

### Calibration options

Passed to `pytest`:

| Flag | When omitted | Meaning |
|------|--------------|---------|
| `--calibrate [N]` | assert mode (generate once, check bounds) | `--calibrate` alone runs 3 calibration passes; `--calibrate N` runs `N`. Both write bounds and skip the baseline assertions. |
| `--ci C` | `0.995` | Confidence level for the Student-t interval. |
| `--pad P` | `0.001` | Relative floor on each bound's half-width (`P * \|mean\|`). Seeded runs are identical, so this is usually the whole interval; metrics whose mean is exactly `0` stay exactly `(0, 0)`. |

Generated data is written under pytest's temporary directory (see `--basetemp`).
