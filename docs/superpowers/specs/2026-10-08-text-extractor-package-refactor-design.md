# Text Extractor Package Refactor Design

## Purpose

Rename the overly specific `pp_uie` Python package to `text_extractor` and
separate the two model families so that their different inference and model
management implementations are obvious. Shared schema, validation, runtime,
evaluation, and web behavior must remain reusable across all models.

## Goals

- Replace the `pp_uie` package name with `text_extractor` everywhere.
- Put generative PP-UIE code and span-based UIE code in separate backend
  subpackages.
- Preserve all user-facing model IDs, commands, HTTP endpoints, model paths,
  extraction results, and model-switching behavior.
- Keep shared logic independent of either model family.
- Leave one canonical implementation; do not retain a duplicate compatibility
  package under the old name.
- Keep model weights and generated caches out of Git.

## Non-goals

- Changing inference behavior, prompts, validation rules, or output formats.
- Moving or redownloading existing model weights.
- Adding a new model family.
- Renaming command-line flags, API routes, or browser controls.
- Reworking the HTML interface.

## Package Structure

```text
text_extractor/
├── __init__.py
├── backends/
│   ├── __init__.py
│   ├── pp_uie/
│   │   ├── __init__.py
│   │   ├── backend.py
│   │   └── models.py
│   └── uie/
│       ├── __init__.py
│       ├── backend.py
│       └── models.py
├── core/
│   ├── __init__.py
│   ├── normalize.py
│   ├── operations.py
│   ├── registry.py
│   └── schema.py
├── evaluation/
│   ├── __init__.py
│   ├── benchmarking.py
│   ├── experiments.py
│   ├── metrics.py
│   └── monitoring.py
├── runtime/
│   ├── __init__.py
│   ├── cli.py
│   ├── environment.py
│   └── offline.py
├── validation/
│   ├── __init__.py
│   └── strict.py
└── web/
    ├── __init__.py
    ├── application.py
    └── server.py
```

## Module Mapping

| Current module | New module | Responsibility |
| --- | --- | --- |
| `pp_uie.backend` | `text_extractor.backends.pp_uie.backend` | PP-UIE generation and prompt handling |
| `pp_uie.models` | `text_extractor.backends.pp_uie.models` | PP-UIE manifests, download, and validation |
| `pp_uie.uie_backend` | `text_extractor.backends.uie.backend` | Taskflow UIE inference and dynamic aliases |
| `pp_uie.uie_models` | `text_extractor.backends.uie.models` | UIE manifests, download, and validation |
| `pp_uie.model_registry` | `text_extractor.core.registry` | Model definitions and indexing |
| `pp_uie.schema` | `text_extractor.core.schema` | Dynamic and nested schema parsing |
| `pp_uie.normalize` | `text_extractor.core.normalize` | Output normalization |
| `pp_uie.operations` | `text_extractor.core.operations` | Operation logging |
| `pp_uie.strict_validation` | `text_extractor.validation.strict` | Canonical fields and evidence validation |
| `pp_uie.benchmarking` | `text_extractor.evaluation.benchmarking` | Length and latency matrices |
| `pp_uie.experiments` | `text_extractor.evaluation.experiments` | Schema switching experiments |
| `pp_uie.metrics` | `text_extractor.evaluation.metrics` | Extraction evaluation metrics |
| `pp_uie.monitoring` | `text_extractor.evaluation.monitoring` | Process memory and latency summaries |
| `pp_uie.cli` | `text_extractor.runtime.cli` | Shared command-line parsing |
| `pp_uie.environment` | `text_extractor.runtime.environment` | Environment inspection |
| `pp_uie.offline` | `text_extractor.runtime.offline` | Offline network blocking |
| `pp_uie.webui` | `text_extractor.web.application` | Persistent services and web application |
| `pp_uie.webserver` | `text_extractor.web.server` | Local HTTP server adapter |

## Dependency Rules

- `backends.pp_uie` may depend on `core.schema` but not on UIE modules.
- `backends.uie` may depend on `validation.strict` for the shared alias registry
  but not on PP-UIE modules.
- `core`, `runtime`, `evaluation`, and `validation` must not import concrete
  backend modules.
- `web.application` may depend on `core.registry` and `validation.strict`; the
  executable script is responsible for constructing concrete model definitions.
- Scripts may import any public package module needed to compose an executable.
- Backend subpackages expose their main config, backend, model specs, and model
  validation functions through their own `__init__.py` files.

These rules keep model-family dependencies one-way and allow another backend to
be added later without changing shared services.

## Data and Control Flow

`scripts/web_ui.py` constructs the four existing model definitions from the two
backend subpackages. `text_extractor.web.application` owns exactly one active
backend, switches it when requested, applies shared validation, and returns the
existing response contract. The browser and HTTP server remain unaware of which
backend family produced a result.

Download and evaluation scripts import the appropriate backend family directly.
Shared metrics, monitoring, environment collection, and offline controls remain
model-agnostic.

## Compatibility and Migration

This is a clean internal package rename. All repository scripts, tests, and
documentation move to `text_extractor.*` imports in the same commit sequence.
The obsolete `pp_uie/` directory is removed after imports have migrated. No
`pp_uie` shim is retained because it would preserve the naming ambiguity and
create two supported public paths.

The following external behavior remains unchanged:

- CLI entry commands under `scripts/` and their flags.
- Model IDs: `0.5b`, `1.5b`, `uie-mini`, and `uie-base`.
- Model directories under `models/`.
- HTTP routes and request/response JSON.
- Web model selection and single-resident-model behavior.
- Raw output, strict validation, timing, and rejected-candidate semantics.

## Testing Strategy

The refactor follows test-driven migration:

1. Add an import-contract test for the complete new package layout and confirm
   that it fails before the package exists.
2. Move one responsibility group at a time, update its unit tests, and keep the
   focused test group green.
3. Update scripts and add script import/help smoke tests so stale imports cannot
   survive unnoticed.
4. Assert that the old `pp_uie` package no longer exists and cannot be imported.
5. Run the complete test suite.
6. Start the local web application and perform real extraction smoke tests for
   one PP-UIE model and one UIE model without moving their weight directories.

## Documentation and Publishing

README architecture descriptions and all reproducible commands will use the new
package paths. Source-only commits will continue to exclude `models/`, caches,
virtual environments, and local metadata. The flattened GitHub publishing flow
will publish `text_extractor/` at the repository root.

## Acceptance Criteria

- No tracked Python or Markdown source refers to `pp_uie` as a package import.
- `text_extractor.backends.pp_uie` and `text_extractor.backends.uie` expose the
  correct, separate implementations.
- All existing automated tests pass under the new imports.
- Script help/import smoke tests pass.
- Both a PP-UIE and UIE model load from their existing local directories and
  complete extraction through the unchanged web API.
- The old `pp_uie/` directory is absent.
- No weights, caches, `.DS_Store`, or virtual-environment files are committed.
