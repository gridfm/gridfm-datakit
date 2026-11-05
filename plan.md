# Porting unmerged_pr onto main

## Scope

Carry the developments from `unmerged_pr` onto current `main` one feature at a time, adapting them to the newer implementation.

This comparison uses local refs inspected on November 4, 2026. No remote fetch was performed. At inspection, `main` and the locally recorded `origin/main` both pointed to `d1aaa48`.

## Branch divergence

```text
b423d46 — common ancestor, October 10, 2026
├── main:        18 commits → d1aaa48
└── unmerged_pr: 10 commits → ac52a02
```

The common ancestor is `b423d46d532cd762c93ee094f32c65fe8a3023ec`: “Drop CodeQL autobuild and move CodeQL to v4 (#100).”

## Developments to carry over

The final branch diff contains 9 changed files, with 794 insertions and 28 deletions.

| Development | Location |
| --- | --- |
| Alternative multiprocessing pipeline and CLI selection | `gridfm_datakit/generate.py`, `gridfm_datakit/cli.py` |
| Per-scenario timeout, pool restart, and skipping timed-out scenarios | `gridfm_datakit/generate.py`, `gridfm_datakit/process/process_network.py` |
| Resume after completed chunks and persist the random seed | `gridfm_datakit/generate.py` |
| CoinHSL/Ipopt solver settings | `gridfm_datakit/process/process_network.py` |
| Configuration examples | Six YAML files under `scripts/config/` and `tests/config/` |

Most implementation landed in `e532632`. Later commits remove timing/profiling code, add CoinHSL settings, and refine error handling and documentation. Port features from the final branch state rather than cherry-picking every intermediate commit.

### Branch commits, oldest first

| Commit | Description |
| --- | --- |
| `e532632` | transferred multiprocessing, checkpointing, restart |
| `0debc88` | removed timer |
| `8dd45a7` | removed profiler |
| `2f865f0` | applied precommit |
| `9fd3a6e` | imported coinhsl settings |
| `418aa3f` | removed ipopt robustness options |
| `90f495c` | added new options to test and script configs |
| `902d447` | addressed pr comments |
| `285f9cf` | addressed pr comments |
| `ac52a02` | fixed docstring |

## Relevant changes on main

- Strict configuration validation, introduced by `81386a1`.
- LightSim2Grid performance improvements in `a48ad50`.
- Centralized random-seed handling.
- Bus ordering and PowSyBl mapping fixes.
- UTF-8 handling, Windows fixes, and integration tests.

## Conflicts and compatibility

Only three files changed on both branches:

- `gridfm_datakit/cli.py`
- `gridfm_datakit/generate.py`
- `gridfm_datakit/process/process_network.py`

A three-way text-merge simulation using temporary files found one conflict region in `generate.py`, around configuration setup and seed handling. The other two shared files merged textually without conflicts. This was a text-level inspection, not a full merge or runtime verification.

| Area | Adaptation needed |
| --- | --- |
| Configuration | Main rejects the new settings until they are added to its schema. |
| Checkpoint/resume | Seed restoration must fit main’s validation and seed policy. |
| New worker | It uses the old seed calculation and does not forward `opf_formulation`. |
| CoinHSL | Settings currently flow through the new multiprocessing path; other generation paths need consideration. |
| Platform support | Timeout termination uses `SIGKILL`, which needs adaptation for Windows support. |

## Implementation order

1. **CoinHSL support — completed**
   - Add settings to the configuration schema and validate them.
   - Adapt solver initialization and pass settings through the applicable generation paths.
   - Update relevant examples and documentation.
   - Verify option forwarding and default behavior.

   Implemented configuration validation, library-path validation before overwrite,
   and option forwarding through sequential/distributed generation, load-scaling
   OPF, and dynamic generation's initial OPF. Added an example and documentation.
   Validation: 129 tests passed, 22 skipped, and 8 deselected in the selected
   regression suite; formatting, core lint, and whitespace checks passed.
   Real MA57 solves with the installed CoinHSL library returned `LOCALLY_SOLVED`
   for AC/DC OPF and AC/DC PF on the bundled dummy case. Other CoinHSL solvers
   and Windows execution were not exercised with a real library.

2. **Checkpoint/resume**
   - Persist and restore the run seed while preserving main’s seed policy.
   - Resume after successfully saved large chunks.
   - Define compatibility checks for resuming with changed configuration.
   - Verify interrupted-run recovery and scenario/output consistency.

3. **Multiprocessing backend**
   - Port the alternative backend while preserving current worker behavior and solver options.
   - Use main’s centralized seed policy and forward `opf_formulation`.
   - Add schema support, CLI selection, examples, and documentation for the backend.
   - Verify supported solver paths and compare expected behavior with the existing backend.

4. **Timeout and restart handling**
   - Add per-scenario timeout tracking, pool restart, and skipping of timed-out scenarios.
   - Adapt process termination for supported platforms.
   - Verify normal worker exceptions, timeout recovery, resource cleanup, and checkpoint behavior.

5. **Final integration review**
   - Finish configuration examples and documentation alongside the applicable features.
   - Run relevant existing tests and focused failure/recovery tests.
   - Run the applicable integration checks after the features are combined.

## Inspection status

The branch inspection did not change repository files or Git history. Existing untracked files were left untouched. No application tests were run during the comparison.
