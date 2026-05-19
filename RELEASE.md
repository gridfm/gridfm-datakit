# GridFM Release Process

This document describes how GridFM releases are planned, coordinated, and delivered.
Its goal is to help contributors from multiple organizations align on *what goes into a release* and *when*.

This process applies to all repositories under https://github.com/gridfm unless stated otherwise.
``

## Release cadence

- **Patch releases**: as needed for bug fixes
- **Minor releases**: as needed for new features
- **Major releases**: when backward‑incompatible changes are introduced
``

## Versioning

GridFM follows semantic versioning:

- **MAJOR**: backward‑incompatible changes
- **MINOR**: new features, backward‑compatible
- **PATCH**: bug fixes and documentation updates
``

## Feature proposal process

To be considered for an upcoming release, contributors must announce their intent
*before* or *early during* the release cycle.

### Step 1 — Open a Feature Proposal

Contributors open a GitHub Issue using the **Feature Proposal** template, including:

- Short description of the feature
- Target repository/repositories
- Expected impact (API change, new model, tooling, docs, etc.)
- Maturity level (experimental / production‑ready)
- Dependencies on other features or data
- Target release (e.g., v0.4)

### Step 2 — Release planning

All feature proposals targeting the release will be collected and the development team will:

- Label them (e.g., `release:v0.4`, `experimental`)
- Identify conflicts or dependencies
- Facilitate discussion (async) if trade‑offs are needed

Acceptance into the release plan does NOT guarantee inclusion.
Features must still meet quality and integration criteria as well as stay in coherence with the GridFM roadmap.
``

## Development and pull requests

All code contributions follow the standard CONTRIBUTING.md process.

Pull requests targeting a specific release should:

- Reference an accepted Feature Proposal issue
- Be labeled with the target release (e.g., `release:v0.4`)
- Include tests, documentation, and examples where applicable


## Release freeze

At the **Release Freeze Date**:

- No new features are accepted
- Only bug fixes, documentation, and release‑blocking changes are allowed

Features not ready by the freeze will be deferred to the next release.
``

## Release candidates

One or more release candidates (RCs) may be published:

- RCs are used for integration testing and validation
- Contributors are expected to test RCs against their use cases
- Blocking issues must be reported with the `release-blocker` label

## Final release

Once all blocking issues are resolved:

- The release is tagged
- Release notes are published
- Artifacts and documentation are updated

The release notes summarize:
- New features
- Breaking changes
- Experimental features
- Known limitations

## Post‑release

After the release:

- A retrospective issue may be opened
- Deferred features are re‑labeled for the next release
- Patch releases may be scheduled if needed

## Common labels:
- `feature-proposal`
- `release:vX.Y`
- `release-blocker`
- `experimental`
- `breaking-change`
