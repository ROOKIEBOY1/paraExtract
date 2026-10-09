# Text Extractor Package Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the `pp_uie` Python package to `text_extractor` and isolate generative PP-UIE and span-based UIE implementations behind separate backend subpackages without changing user-facing behavior.

**Architecture:** Shared schema, registry, validation, runtime, evaluation, and web services move into responsibility-based `text_extractor` subpackages. Model-specific loading, downloading, and inference move into `text_extractor.backends.pp_uie` and `text_extractor.backends.uie`; executable scripts compose those pieces and preserve the existing CLI and HTTP contracts.

**Tech Stack:** Python 3.9, PaddlePaddle/PaddleNLP, pytest, standard-library HTTP server.

**Spec:** `docs/superpowers/specs/2026-10-08-text-extractor-package-refactor-design.md`

## Global Constraints

- Preserve model IDs `0.5b`, `1.5b`, `uie-mini`, and `uie-base`.
- Preserve model directories under `models/`; do not move or redownload weights.
- Preserve CLI flags, HTTP routes, JSON output contracts, strict validation behavior, and single-resident-model switching.
- Do not keep an import compatibility shim or duplicate implementation under `pp_uie`.
- Do not commit weights, generated static models, caches, virtual environments, or `.DS_Store`.
- Use `git mv` for tracked source moves so history remains traceable.
- Each migration task follows RED → GREEN and ends with its focused test group passing.

## Review Focus

- Relative imports after deep package moves must resolve when scripts run directly from `scripts/`; Task 6 adds subprocess help/import smoke tests.
- UIE alias expansion must still import the canonical alias registry without creating a backend/validation cycle; Task 3 runs the alias and span-preservation tests.
- PP-UIE nested schemas must still resolve `core.schema` after the move; Task 2 runs backend and schema tests together.
- Model switching must continue accepting heterogeneous config/backend types while retaining only one active model; Task 5 runs all switching-service tests.
- Offline subprocess tests must reference `text_extractor.runtime.offline`, and the removed `pp_uie` package must not remain importable; Tasks 4 and 6 pin both conditions.

---

### Task 1: Create the shared package foundation

**Files:**
- Create: `tests/test_package_layout.py`
- Create: `text_extractor/__init__.py`
- Create: `text_extractor/core/__init__.py`
- Move: `pp_uie/model_registry.py` → `text_extractor/core/registry.py`
- Move: `pp_uie/normalize.py` → `text_extractor/core/normalize.py`
- Move: `pp_uie/operations.py` → `text_extractor/core/operations.py`
- Move: `pp_uie/schema.py` → `text_extractor/core/schema.py`
- Create: `text_extractor/validation/__init__.py`
- Move: `pp_uie/strict_validation.py` → `text_extractor/validation/strict.py`
- Modify: `tests/test_normalize.py`
- Modify: `tests/test_operations.py`
- Modify: `tests/test_schema.py`
- Modify: `tests/test_strict_validation.py`

**Interfaces:**
- Produces: `text_extractor.core.registry.ModelDefinition`, `model_options`, and `index_definitions`.
- Produces: existing normalization, operation logging, schema parsing, and strict validation APIs at their mapped module paths.
- Produces: `text_extractor.validation.strict.EN_TO_CN` for the UIE backend.

- [ ] **Step 1: Write the failing package-layout test**

Add `test_shared_modules_import_from_text_extractor()` to import the public objects above and assert their defining modules start with `text_extractor.`. Update the four focused test files to use the new imports.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_package_layout.py tests/test_normalize.py tests/test_operations.py tests/test_schema.py tests/test_strict_validation.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'text_extractor'`.

- [ ] **Step 3: Move the shared modules and add package initializers**

Use `git mv` for the mapped files. Keep public function/class signatures unchanged, and expose no backend-specific objects from the top-level package.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add text_extractor tests/test_package_layout.py tests/test_normalize.py tests/test_operations.py tests/test_schema.py tests/test_strict_validation.py pp_uie
git commit -m "refactor: create text extractor shared package"
```

### Task 2: Isolate the PP-UIE backend family

**Files:**
- Create: `text_extractor/backends/__init__.py`
- Create: `text_extractor/backends/pp_uie/__init__.py`
- Move: `pp_uie/backend.py` → `text_extractor/backends/pp_uie/backend.py`
- Move: `pp_uie/models.py` → `text_extractor/backends/pp_uie/models.py`
- Modify: `tests/test_backend_contract.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: `text_extractor.core.schema.SchemaNode` and `parse_schema` from Task 1.
- Produces from `text_extractor.backends.pp_uie`: `GenerationEngine`, `PaddleGenerationEngine`, `PPUIEBackend`, `RuntimeConfig`, `MODEL_FILES`, `MODEL_SPECS`, `download_model`, `manifest_as_dict`, and `validate_model_dir`.

- [ ] **Step 1: Change the PP-UIE tests to the intended imports**

Import implementation-private `_parse_answer` from `text_extractor.backends.pp_uie.backend`; import public backend and model APIs through `text_extractor.backends.pp_uie`.

- [ ] **Step 2: Run the PP-UIE tests and verify RED**

```bash
.venv/bin/python -m pytest tests/test_backend_contract.py tests/test_models.py -q
```

Expected: collection fails because `text_extractor.backends.pp_uie` does not exist.

- [ ] **Step 3: Move the backend files and repair imports**

Change the backend's relative schema import to `...core.schema`. Define the exact public exports listed in the Interfaces block in `backends/pp_uie/__init__.py`.

- [ ] **Step 4: Run PP-UIE and shared schema tests**

```bash
.venv/bin/python -m pytest tests/test_backend_contract.py tests/test_models.py tests/test_schema.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add text_extractor/backends tests/test_backend_contract.py tests/test_models.py pp_uie
git commit -m "refactor: isolate PP-UIE backend"
```

### Task 3: Isolate the UIE backend family

**Files:**
- Create: `text_extractor/backends/uie/__init__.py`
- Move: `pp_uie/uie_backend.py` → `text_extractor/backends/uie/backend.py`
- Move: `pp_uie/uie_models.py` → `text_extractor/backends/uie/models.py`
- Modify: `tests/test_uie_backend.py`
- Modify: `tests/test_uie_models.py`
- Modify: `tests/test_uie_evaluation.py`

**Interfaces:**
- Consumes: `text_extractor.validation.strict.EN_TO_CN` from Task 1.
- Produces from `text_extractor.backends.uie`: `UIETaskflowBackend`, `UIETaskflowConfig`, `UIE_MODEL_SPECS`, `download_uie_model`, `download_uie_mini`, and `validate_uie_model_dir`.

- [ ] **Step 1: Change UIE tests to the intended imports**

Use `text_extractor.backends.uie.backend` where tests need implementation detail and `text_extractor.backends.uie` for public APIs.

- [ ] **Step 2: Run the UIE tests and verify RED**

```bash
.venv/bin/python -m pytest tests/test_uie_backend.py tests/test_uie_models.py tests/test_uie_evaluation.py -q
```

Expected: collection fails because `text_extractor.backends.uie` does not exist.

- [ ] **Step 3: Move UIE files and repair the alias import**

Import `EN_TO_CN` from `...validation.strict`. Define only the public exports listed above in `backends/uie/__init__.py`; keep static-cache repair and checksum helpers internal.

- [ ] **Step 4: Run UIE and strict-validation tests**

```bash
.venv/bin/python -m pytest tests/test_uie_backend.py tests/test_uie_models.py tests/test_uie_evaluation.py tests/test_strict_validation.py -q
```

Expected: all selected tests pass, including dynamic aliases, span identity, and deduplication.

- [ ] **Step 5: Commit**

```bash
git add text_extractor/backends/uie tests/test_uie_backend.py tests/test_uie_models.py tests/test_uie_evaluation.py pp_uie
git commit -m "refactor: isolate UIE backend"
```

### Task 4: Separate runtime and evaluation utilities

**Files:**
- Create: `text_extractor/runtime/__init__.py`
- Move: `pp_uie/cli.py` → `text_extractor/runtime/cli.py`
- Move: `pp_uie/environment.py` → `text_extractor/runtime/environment.py`
- Move: `pp_uie/offline.py` → `text_extractor/runtime/offline.py`
- Create: `text_extractor/evaluation/__init__.py`
- Move: `pp_uie/benchmarking.py` → `text_extractor/evaluation/benchmarking.py`
- Move: `pp_uie/experiments.py` → `text_extractor/evaluation/experiments.py`
- Move: `pp_uie/metrics.py` → `text_extractor/evaluation/metrics.py`
- Move: `pp_uie/monitoring.py` → `text_extractor/evaluation/monitoring.py`
- Modify: `tests/test_benchmarking.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_environment.py`
- Modify: `tests/test_experiments.py`
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_monitoring.py`
- Modify: `tests/test_offline.py`

**Interfaces:**
- Produces the existing utility functions and classes at their mapped modules; these subpackage `__init__.py` files do not need broad re-exports.

- [ ] **Step 1: Update focused tests to the new runtime/evaluation imports**

Also update the inline subprocess source strings in `test_offline.py` to import `text_extractor.runtime.offline`.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
.venv/bin/python -m pytest tests/test_benchmarking.py tests/test_cli.py tests/test_environment.py tests/test_experiments.py tests/test_metrics.py tests/test_monitoring.py tests/test_offline.py -q
```

Expected: collection or subprocess imports fail because the new subpackages do not exist.

- [ ] **Step 3: Move the modules into their responsibility packages**

Keep APIs unchanged; add only the package initializers required for imports.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all selected tests pass, including both blocked-network subprocess cases.

- [ ] **Step 5: Commit**

```bash
git add text_extractor/runtime text_extractor/evaluation tests pp_uie
git commit -m "refactor: separate runtime and evaluation utilities"
```

### Task 5: Move the shared web application and server

**Files:**
- Create: `text_extractor/web/__init__.py`
- Move: `pp_uie/webui.py` → `text_extractor/web/application.py`
- Move: `pp_uie/webserver.py` → `text_extractor/web/server.py`
- Modify: `tests/test_webui.py`
- Modify: `tests/test_web_server.py`

**Interfaces:**
- Consumes: `text_extractor.core.registry` and `text_extractor.validation.strict`.
- Produces: `ExtractionService`, `SwitchingExtractionService`, `WebApplication`, `load_text_samples`, and `load_web_samples` from `text_extractor.web.application`; `make_handler` from `text_extractor.web.server`.

- [ ] **Step 1: Update web tests to the intended imports**

Update concrete backend imports in `test_webui.py` to the family subpackages and web service imports to `text_extractor.web.application`/`server`.

- [ ] **Step 2: Run web tests and verify RED**

```bash
.venv/bin/python -m pytest tests/test_webui.py tests/test_web_server.py -q
```

Expected: collection fails because `text_extractor.web` does not exist.

- [ ] **Step 3: Move web modules and repair shared imports**

In `application.py`, import model definitions from `..core.registry` and strict validation from `..validation.strict`. Preserve locking, timing, model replacement, request validation, and response dictionaries exactly.

- [ ] **Step 4: Run web and backend integration tests**

```bash
.venv/bin/python -m pytest tests/test_webui.py tests/test_web_server.py tests/test_uie_backend.py tests/test_backend_contract.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add text_extractor/web tests/test_webui.py tests/test_web_server.py pp_uie
git commit -m "refactor: move shared web services"
```

### Task 6: Migrate executable scripts, remove the old package, and document the architecture

**Files:**
- Create: `tests/test_script_imports.py`
- Modify: `tests/test_package_layout.py`
- Modify: `scripts/benchmark.py`
- Modify: `scripts/check_env.py`
- Modify: `scripts/download_models.py`
- Modify: `scripts/evaluate_uie_mini.py`
- Modify: `scripts/infer.py`
- Modify: `scripts/run_experiments.py`
- Modify: `scripts/web_ui.py`
- Modify: `README.md`
- Delete: `pp_uie/__init__.py`

**Interfaces:**
- Consumes all new public paths from Tasks 1–5.
- Preserves every existing script command, flag, model ID, model path, HTTP route, and output contract.

- [ ] **Step 1: Add failing script and removal-contract tests**

In `test_script_imports.py`, invoke these commands with `subprocess.run`, `sys.executable`, and the project root as `cwd`, then assert return code zero:

```text
python scripts/web_ui.py --help
python scripts/download_models.py --help
python scripts/infer.py --help
python scripts/benchmark.py --help
python scripts/run_experiments.py --help
```

Extend `test_package_layout.py` to assert `Path("pp_uie").exists()` is false and `importlib.util.find_spec("pp_uie")` returns `None` in a fresh subprocess.

- [ ] **Step 2: Run the migration-contract tests and verify RED**

```bash
.venv/bin/python -m pytest tests/test_script_imports.py tests/test_package_layout.py -q
```

Expected: script imports still reference `pp_uie`, and the old directory still exists.

- [ ] **Step 3: Update every script import and remove the empty old package**

Use the backend-family public imports for model construction/download and the responsibility packages for shared utilities. Remove the final `pp_uie/__init__.py` and directory after confirming no source modules remain.

- [ ] **Step 4: Update README architecture and developer references**

Document `text_extractor/`, the two backend families, the unchanged user commands, and the absence of an old-package compatibility path. Replace package-import examples without changing model IDs or filesystem model locations.

- [ ] **Step 5: Run stale-reference and source hygiene checks**

```bash
rg -n '^\s*(from|import) pp_uie\b' --glob '*.py' .
rg -n 'pp_uie/' README.md scripts tests text_extractor
git status --short
git diff --check
```

Expected: both `rg` commands return no stale source references; status contains only intended source/document changes and no model/cache/local metadata. Historical design and plan documents may describe the removed name and are intentionally outside the second search.

- [ ] **Step 6: Run the complete automated suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass, with only the existing environment-dependent skip allowed.

- [ ] **Step 7: Run real local-model web smoke tests**

Start `scripts/web_ui.py` with PP-UIE-0.5B, call `/api/extract` once, switch to UIE-base through `/api/model`, and call `/api/extract` again. Confirm both responses retain the existing JSON shape and model IDs and that existing `models/PP-UIE-0.5B` and `models/UIE-base` paths were used without downloads.

- [ ] **Step 8: Commit**

```bash
git add README.md scripts tests/test_package_layout.py tests/test_script_imports.py pp_uie
git commit -m "refactor: adopt text extractor package"
```

### Task 7: Final review and GitHub publishing verification

**Files:**
- Review only: all files changed by Tasks 1–6

**Interfaces:**
- Produces a reviewed branch ready for the existing flattened GitHub publishing flow.

- [ ] **Step 1: Inspect the complete branch diff**

Confirm moves are represented as renames where practical, public imports match the spec, and no behavioral code was changed incidentally.

- [ ] **Step 2: Re-run final verification from a clean process**

```bash
.venv/bin/python -m pytest -q
git diff --check
git status --short
```

Expected: suite passes; diff check is clean; only the unrelated root `.DS_Store`, if still present, remains untracked.

- [ ] **Step 3: Verify publish-tree contents before pushing**

In the temporary flattened publishing worktree, confirm `text_extractor/` exists, `pp_uie/` does not, and no paths under `models/`, `.venv/`, cache directories, or `.DS_Store` are tracked.

- [ ] **Step 4: Publish only after explicit user request**

Push the reviewed flattened commit to `origin/main` using the existing non-force publishing workflow.
