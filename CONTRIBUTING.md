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
* [ ] Open a PR with a short description of your changes and add Alban Puech as a reviewer.
* [ ] Ensure code, tests, and documentation are clear and complete.
