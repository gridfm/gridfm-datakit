# gridfm-datakit — Technical Review

**Subject:** Code, dependency, and CI pipeline assessment
**Repository:** gridfm/gridfm-datakit, branch `main` (v1.0.4)
**Date:** June 2026

---

## 1. Scope and method

This review covers the full Python package (`gridfm_datakit/`), the test suite
(`tests/`), the GitHub Actions workflows, `pyproject.toml`, the pre-commit
configuration, and the coverage configuration. The objectives, as agreed, were to
identify robustness issues and to determine why CI runs take several hours.

The assessment was done by direct reading of the source, supplemented by static
analysis (ruff with the `E,F,W,B,UP,SIM` rule sets) to confirm and quantify certain
findings. All findings were re-verified against the current tree before delivery;
behavioral claims that could be tested cheaply (notably 2.7 and 4.2) were confirmed
empirically, and the Julia package UUIDs cited in 2.3 were checked against the
JuliaRegistries General registry. Line references are against the current `main`.

The CI duration has three root causes, all addressed below: (a) both dependency caches
are keyed on files that do not exist in the repository, so they never invalidate
correctly and the Julia toolchain is repeatedly rebuilt from cold; (b) the Julia runtime
boot cost is paid far more often than necessary, both in tests and in the production
code path; (c) two failure modes cause indefinite hangs, which then run until GitHub's
6-hour job limit terminates them.

Thirty findings follow, grouped by area. Each is tagged:

- **P0** — defect, or a large CI improvement at trivial cost. Recommended immediately.
- **P1** — significant robustness or performance improvement, low risk.
- **P2** — maintainability and hygiene.

Findings are deliberately scoped so that each can be implemented and reviewed as an
independent change. Implementing the P0 set alone is expected to reduce a typical CI
run from several hours to the 15–25 minute range. A suggested sequencing is given in
section 7.

---

## 2. CI pipeline (`.github/workflows/ci-build.yaml`)

### 2.1 Pip cache key references non-existent files — P0

All three jobs define:

```yaml
key: pip-${{ runner.os }}-${{ hashFiles('**/requirements*.txt') }}
```

The repository contains no `requirements*.txt`; dependencies are declared in
`pyproject.toml`. `hashFiles` therefore evaluates to an empty string and the key is the
constant `pip-Linux-`. The cache never invalidates on dependency changes and accumulates
stale wheels indefinitely. Correct key:

```yaml
key: pip-${{ runner.os }}-${{ hashFiles('pyproject.toml') }}
```

Alternatively, remove the manual cache blocks entirely in favor of `setup-python`'s
built-in caching (see 2.4).

### 2.2 Julia cache key has the same defect — P0

The `pytests` job caches `~/.julia` keyed on `hashFiles('**/Project.toml')`. No
`Project.toml` exists in the repository, so this key is also constant. In addition,
caching the entire `~/.julia` directory with `actions/cache` is unreliable for Julia
workloads (registries, compiled caches, and artifacts require differentiated handling).
The supported mechanism is:

```yaml
- uses: julia-actions/cache@v2
```

This finding is significant for run time: precompiling PowerModels and Ipopt from cold
takes several minutes, and under the current configuration this occurs whenever the
stale cache is evicted.

### 2.3 Julia dependencies are unpinned — P0

`gridfm_datakit setup_pm` executes `Pkg.add("Ipopt")`, `Pkg.add("PowerModels")`,
`Pkg.add("Memento")` without version constraints. Consequences: CI is not reproducible
(an upstream PowerModels release can break `main` with no code change), and there is no
file whose hash can serve as a deterministic cache key for the Julia environment.

Since the project already depends on juliacall, the standard remedy is a `juliapkg.json`
shipped with the package, which juliacall resolves automatically:

```json
{
  "julia": "1.12",
  "packages": {
    "PowerModels": {"uuid": "c36e90e8-916a-50a6-bd94-075b64ef4655", "version": "0.21"},
    "Ipopt":       {"uuid": "b6b21f68-93f8-5de0-b562-5493be1d77c9", "version": "1"},
    "Memento":     {"uuid": "f28f55f0-a522-5efc-85c2-fe41dfb9b2d9", "version": "1"}
  }
}
```

This also renders most of `setup_pm` redundant, as juliacall installs the pinned project
on first import.

### 2.4 Outdated action versions — P0

- `actions/cache@v3` belongs to a deprecated line: GitHub retired cache v1–v2 and shut
  down the legacy cache backend in April 2025; only the final patched v3 releases remain
  functional and v4 is the maintained version. The docs workflow already uses v4; the CI
  workflow does not.
- `actions/setup-python@v4` should be moved to `@v5`, which also provides pip caching
  keyed correctly out of the box:

```yaml
- uses: actions/setup-python@v5
  with:
    python-version: '3.12'
    cache: pip
    cache-dependency-path: pyproject.toml
```

- `julia-actions/setup-julia@v1` should be moved to `@v2`.

### 2.5 No concurrency control, no job timeouts — P0

Successive pushes to the same PR currently queue full pipeline runs in parallel, and a
hung job (a concrete hang mechanism is documented in 4.2) runs until the 6-hour default
limit. Recommended additions:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

with `timeout-minutes: 45` on `pytests` and `timeout-minutes: 10` on the lint and
security jobs. This bounds the worst case independently of any other fix.

### 2.6 Tests run serially despite xdist being installed and the suite being prepared for it — P0

`pytest-xdist` is declared in the `test` extra, and `tests/test_generate.py` already
isolates output directories per worker via `PYTEST_XDIST_WORKER`. The CI invocation
nevertheless runs serially:

```
pytest --cov=. tests/ -v -s
```

Recommended invocation:

```
pytest -n auto --dist loadgroup --cov=gridfm_datakit --durations=25 tests/
```

Rationale: `-n auto` parallelizes across runner cores and, because each test class boots
its own Julia session, amortizes that boot cost across cores instead of serializing it.
`--cov=.` currently measures the entire checkout including `scripts/` and `tests/`;
`--cov=gridfm_datakit` reflects the intended target and is cheaper. The `-s` flag
disables output capture and floods the log with prints and tqdm output, which has a
measurable cost on chatty suites. `--durations=25` surfaces the slowest tests so future
regressions in CI time are attributable.

### 2.7 Lint and security jobs install the full package unnecessarily — P0

The `pre-commit-run` job executes `pip install -e ".[dev]"`, building and installing
juliacall, numba, pandas, scipy, plotly, and the Jupyter stack in order to run lint
hooks. pre-commit manages its own isolated hook environments and requires none of these:

```yaml
- uses: actions/setup-python@v5
  with: {python-version: '3.12'}
- uses: pre-commit/action@v3.0.1
```

The action additionally caches `~/.cache/pre-commit` keyed on the config file, which the
current setup does not do.

The `security-test` job has the same install pattern, but a more serious problem was
confirmed during verification: the scan itself is a no-op. The job runs

```
bandit --severity-level high .
```

without the `-r` flag, and bandit does not recurse into directory targets unless `-r`
is given. It emits `Skipping directory (.), use -r flag to scan contents`, reports
**"Total lines of code: 0"**, and exits 0. This was reproduced locally against the
current tree: the security gate has never scanned any code. The corrected invocation:

```
bandit -r gridfm_datakit --severity-level high
```

(verified to scan the package: 0 high-severity findings at present, so the job
continues to pass once fixed). Combined with the `pip install bandit`-only setup, both
jobs also drop from roughly 5–10 minutes to under 2.

### 2.8 PGLib cases are re-downloaded and re-corrected on every run — P0

`load_net_from_pglib` downloads `.m` files from `raw.githubusercontent.com` at test
time and then performs a Julia round-trip (`correct_network`) to produce the
`_corrected.m` variant. Under CI this incurs, per run and per case: network latency,
exposure to GitHub raw rate limiting (a plausible source of intermittent failures), and
one Julia process launch per correction.

Remediation in two steps. First, relocate the download cache out of the installed
package directory (a defect in its own right, see 3.1) to a user cache directory such as
`~/.cache/gridfm-datakit/grids`, overridable by environment variable. Second, cache that
directory in CI:

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/gridfm-datakit
    key: pglib-${{ hashFiles('tests/**/*.py') }}
    restore-keys: pglib-
```

After one warm run, the test suite no longer touches the network.

### 2.9 The suite requires tiering; the exhaustive PGLib sweep is not a per-commit test — P0

The current controls are an environment variable (`SKIP_LARGE_GRIDS`) and an opt-in
(`RUN_ALL_PGLIB`). These should be replaced with registered pytest markers (`julia`,
`slow`, `integration`) and the pipeline split accordingly:

- **Pull requests:** `pytest -n auto -m "not slow and not integration"`. The
  pure-Python tests (perturbations, parsing, parameter handling, seeding) complete in
  minutes and do not require the Julia setup steps at all.
- **Merge to main:** full suite excluding the exhaustive sweep.
- **Nightly schedule:** large grids and the all-PGLib comparison (63 cases × 2 solver
  types, up to `case78484_epigrids`).

This is the structural change that keeps CI duration bounded as the suite grows, rather
than a one-time reduction.

### 2.10 Redundant virtualenv steps — P1

The `pytests` job creates a `.venv` and re-sources it in each step. `setup-python`
already provides an isolated interpreter, and the activation performed in the dedicated
"Create virtualenv" step does not persist into subsequent steps in any case (each `run:`
is a fresh shell). The steps can be removed without functional change.

### 2.11 CodeQL autobuild and missing path filters — P1

`github/codeql-action/autobuild` is a no-op for Python and adds time. Additionally, the
workflow lacks `paths-ignore` filters (`docs/**`, `**.md`), so documentation-only
changes trigger full pipeline runs.

### 2.12 Test configurations sized for a workstation rather than a runner — P1

- `tests/config/consistency_test.yaml` sets `num_processes: 5`. Standard GitHub runners
  provide 2–4 cores, and under the spawn start method each worker boots its own Julia
  runtime — for a total of 10 scenarios. A value of 1–2 is appropriate.
- The same file sets `enable_solver_logs: true`, which raises the Ipopt print level to 5
  and writes per-process log files; this is pure overhead in CI.
- The `data_dir: "./testdelll"` value in that file is dead (the fixture overrides it)
  and appears to be residual debugging; it should be cleaned up.

### 2.13 No per-test timeout — P1

A solver stall, a juliacall deadlock, or the hang described in 4.2 currently blocks the
job until the 6-hour limit. Adding `pytest-timeout` to the test extra with:

```toml
[tool.pytest.ini_options]
timeout = 600
timeout_method = "thread"
```

converts an unattributed multi-hour stall into a named failure within minutes. Together
with 2.5, this makes multi-hour CI runs structurally impossible.

---

## 3. Packaging and dependencies (`pyproject.toml`)

### 3.1 Downloaded grids are written into the installed package directory — P0

`get_pglib_file_path` (`network.py`, ~line 650) writes downloaded grid files, and their
Julia-corrected copies, into `resources.files("gridfm_datakit.grids")` — that is, into
site-packages. This fails on read-only installations (system Python, containers, Nix),
leaves files behind after `pip uninstall`, would break under zipped distribution, and
permits a race between concurrent processes sharing one installation — for example
parallel pytest-xdist workers (directly relevant once 2.6 is adopted) or two generation
runs started side by side — in which one process reads another's partially written
download.

Recommended: `platformdirs.user_cache_dir("gridfm-datakit")` with an environment-variable
override; download to a temporary file in the target directory and `os.replace` into
place for atomicity. This change is also the prerequisite for the CI caching in 2.8.

### 3.2 `pathlib` listed as a dependency — P0

`pathlib` has been part of the standard library since Python 3.4. The PyPI distribution
of the same name is an abandoned Python-2 backport which, when installed, can shadow the
standard-library module and break unrelated packages. Given
`requires-python = ">=3.10"`, the entry is at best inert and at worst hazardous. Remove
it.

### 3.3 Unused and mis-tiered dependencies — P1

- `numba` is declared but not imported anywhere in the package (verified by search). It
  pulls in llvmlite and constrains the permissible numpy range, imposing a cost on every
  install for no benefit. Remove.
- `ipykernel`, `nbformat`, `ipywidgets`, and `ipyfilechooser` are used only by
  `interactive.py`; `plotly` and `matplotlib` only by plotting and statistics code. For
  the primary use case — headless data generation — these are dead weight in every
  installation, including every CI job. Recommended extras:

```toml
[project.optional-dependencies]
interactive = ["ipykernel", "ipywidgets", "ipyfilechooser", "nbformat"]
viz = ["plotly", "matplotlib"]
```

with lazy imports and explicit error messages in `interactive.py` and `utils/stats.py`.

- The code imports `yaml`; the declared dependency should therefore be `pyyaml` rather
  than `pyaml` (a wrapper that transitively provides pyyaml). Depend on what is
  imported.

### 3.4 Release workflow installs the Julia bindings to build a wheel — P1

`release.yaml` runs `pip install -e .[dev,test]` — juliacall, numba, pytest, mkdocs —
before executing `python -m build`. The job requires only `pip install build`. While in
that file, the `dev` extra should be split into `docs` and `lint` groups so each job
installs only its own tooling, and `twine check dist/*` should be added before
publication.

---

## 4. Defects

### 4.1 The exhaustive PGLib test passes `solver_type` into the `fast` parameter — P0

Function signature (`process/solvers.py:192`):

```python
def compare_pf_results(net, jl, case_name, fast, solver_type="pf") -> bool
```

Call site (`tests/test_compare_pf_opf_results_all_pglib.py`):

```python
compare_pf_results(net, self.jl, case_name, solver_type)
```

The fourth positional argument binds to `fast`, which receives the string `"pf"` or
`"opf"` (both truthy), while `solver_type` silently takes its default of `"pf"`. The
effect is that the exhaustive sweep never tests OPF and always exercises the fast-PF
path. The small-grid variant of this test passes the arguments correctly, which
explains why the defect went unnoticed. Beyond fixing the call site, `solver_type`
should be made keyword-only so that this class of error raises a `TypeError`.

### 4.2 The distributed progress loop can block indefinitely — P0

`generate.py`, ~line 510:

```python
while completed < chunk_size:
    progress_queue.get()
    pbar.update(1)
    completed += 1
```

`Queue.get()` is called without a timeout. The worker's exception handler does backfill
progress ticks for caught Python exceptions (`process_network.py`, ~line 1171), so those
do not hang the loop. The hang occurs on hard worker death — an OOM kill by the kernel
or a native crash inside Julia/Ipopt, both realistic for large grids — where no handler
runs, no ticks arrive, and `multiprocessing.Pool` leaves the in-flight `AsyncResult`
pending indefinitely. The parent then blocks on `get()` forever: locally a frozen run,
in CI a job consuming the full 6-hour limit. The loop should use `get(timeout=...)` and
concurrently poll the `AsyncResult` objects so that worker death raises promptly.

A secondary defect in the same handler: on a caught mid-chunk exception it backfills
`end_idx - start_idx` ticks — the full chunk — on top of the ticks already emitted for
completed scenarios. The queue is over-filled, the current chunk's loop exits early, and
the surplus ticks are consumed by the next chunk's loop, corrupting progress accounting
from that point on (synchronization is rescued only by the blocking `result.get()`
calls that follow).

Related, at line 528: on worker error the code calls `sys.exit(e)`. Library code must
not terminate the host interpreter; any caller embedding
`generate_power_flow_data_distributed` in a larger pipeline — including pytest — is
killed along with it. The captured exception and traceback should be re-raised instead.

### 4.3 `from numpy import any` shadows the builtin — P0

`network.py:23`:

```python
from numpy import any, conj, exp, hstack, int64, nonzero, ones, pi, real
```

Throughout this ~800-line module, `any(...)` resolves to `numpy.any`, whose semantics
differ from the builtin (a generator argument is truthy regardless of its contents).
Even if current call sites are coincidentally safe, the import is a latent defect for
future edits. Use `np.any` explicitly.

### 4.4 Network download without timeout or retry — P1

`network.py:656`: `requests.get(url)`. A stalled connection blocks a worker
indefinitely, and a single transient 503 from GitHub raw fails the entire generation
run. Minimum remediation: `requests.get(url, timeout=(5, 60))`. Preferably, a `Session`
with urllib3 `Retry`, combined with the atomic write described in 3.1.

### 4.5 Mutable default argument — P1

`perturbations/topology_perturbation.py:139`:

```python
elements: List[str] = ["branch", "gen"]
```

A shared mutable default: mutation by any caller propagates to all subsequent instances.
Flagged by ruff B006. Default to `None` and assign within the function body.

---

## 5. Test suite structure

### 5.1 One Julia session per run, not per test class — P1

Four test classes (`test_compare_pf_opf_results`, `test_compare_pf_opf_results_all_pglib`
— normally skipped — `test_solve`, `test_verify_network`) each invoke `init_julia` in
`setup_class`, and each invocation re-evaluates the full PowerModels/Ipopt setup — tens
of seconds per boot even with warm precompilation. Further sessions are booted inside
the spawned workers of the generation-pipeline tests. A session-scoped fixture performs
the test-process boot once (once per xdist worker, which remains a substantial
improvement):

```python
@pytest.fixture(scope="session")
def jl():
    from gridfm_datakit.process.process_network import init_julia
    return init_julia(max_iter=150)
```

Under xdist, Julia-heavy tests should additionally be grouped with
`@pytest.mark.xdist_group("julia")` and `--dist loadgroup` so they co-locate on workers.

### 5.2 Production path boots Julia once per worker per chunk — P1

`generate_power_flow_data_distributed` constructs a new `Pool` for every large chunk,
and `process_scenario_chunk` calls `init_julia` on entry. Total Julia boots therefore
scale as `num_processes × num_chunks`; a 100,000-scenario run with
`large_chunk_size: 1000` incurs 100 pool teardowns and several hundred Julia startups.
Restructure: construct the pool once outside the chunk loop, pass a pool `initializer=`
that boots Julia and stores the handle in worker-local state, and have the chunk
function consume that handle. Seeding is applied per chunk via `custom_seed`, so outputs
are unchanged; the gain is wall-clock only, benefiting both end users and the
integration tests.

### 5.3 Test artifacts written into the working tree — P2

`tests/test_generate.py` writes to `./tests/test_data_{mode}_{worker}` and
`tests/test_data`, with cleanup distributed across the tests themselves (lines 182–190,
367). A mid-test failure leaves directories behind, which subsequently dirty `git
status` and are swept up by `--cov=.`. pytest's `tmp_path` / `tmp_path_factory` fixtures
provide cleanup on all exit paths. Separately, a `.coverage` binary is committed at the
repository root; it should be removed and added to `.gitignore`.

### 5.4 Hardcoded coverage badge — P2

The README badge displays a static 76%. Once the test job emits `coverage.xml`
(`--cov-report=xml`), integrating Codecov or an equivalent badge action makes the figure
verifiable and turns coverage regressions into review signal.

---

## 6. Maintainability

### 6.1 Three overlapping lint tools — P2

`.pre-commit-config.yaml` runs ruff (check and format), flake8, and add-trailing-comma.
Ruff subsumes flake8's rule set (the `--ignore=E501,W503,E203` arguments map directly to
ruff configuration) and `ruff format` handles trailing commas; the additional hooks add
environments to provision in CI and introduce formatter-conflict risk. Retain ruff
alone, with configuration in `pyproject.toml`. It is also worth widening the rule
selection: a run with `--select E,F,W,B,UP,SIM` currently reports approximately 700
findings, including the defects documented in section 4 (B006, B023, B904 ×27).

### 6.2 Print-based diagnostics — P2

The package contains approximately 180 `print()` calls and no use of the `logging`
module. Consumers
cannot silence the library, CI logs are inflated, and diagnostic context (failing
scenario, worker identity) is unstructured. The migration is mechanical —
`logging.getLogger(__name__)` per module, `--verbose/--quiet` on the CLI — and can
proceed module by module. The 42 broad `except Exception` blocks should be tightened
opportunistically during the same passes.

### 6.3 Claimed Python support is untested — P2

`requires-python = ">=3.10,<3.13"` and the classifiers declare 3.10–3.12, but CI
exercises only 3.12 (and the release job builds on 3.10). With the fast tier from 2.9 in
place, a 3.10/3.11/3.12 matrix over that tier alone costs a few minutes and catches
version drift before users encounter it.

### 6.4 `validate` reconstructs configuration by line offset — P2

`cli.py`, ~line 74: the run configuration is recovered by skipping the first two lines
of `args.log` and YAML-parsing the remainder. Any change to the log format breaks the
command. The generation step already serializes the arguments; it should additionally
write a clean `config.yaml` alongside the data, which `validate` reads, with `--mode` as
fallback.

---

## 7. Recommended sequencing

**Phase 1 — immediate (all small, independently reviewable):** 2.1, 2.2, 2.4, 2.5, 2.6,
2.7, 3.2, 4.1, 4.3. These nine changes address the two cheapest defects and are expected
to bring CI from hours to well under one hour.

**Phase 2:** 2.3, 2.8, 2.9, 2.13 — pinned Julia dependencies, grid caching, test
tiering, and timeouts. This phase makes the reduction durable as the suite grows.

**Phase 3:** the remaining robustness items (3.1, 3.3, 4.2, 4.4, 4.5, 5.1, 5.2), with
4.2 and 5.2 prioritized since they affect production users, not only CI. The P2 items
can be absorbed alongside routine work.
