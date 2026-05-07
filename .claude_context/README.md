# Claude Code Context

Read this file first.

## Mandatory operating rules

1. Do not read the whole repository.
2. Do not use broad glob searches such as `**/*`.
3. Use these context files as the navigation map.
4. Efficient context use is the goal, not blind minimalism.
5. Reading a few extra relevant functions is acceptable.
6. Reading half of the repo to answer one question is not acceptable.
7. Before opening source files, state:
   - which files you want to open,
   - why each file is needed,
   - what specific information you expect to find.
8. Open at most 3 source files in the first step.
9. Ask for the next specific file only if needed.
10. Modify at most 1 file per implementation step.
11. After each modification, show the diff and one concrete test command.

## Context files

- `context_entrypoints.md` — likely app starts, manifests and main files.
- `context_routes.md` — API routes, endpoints and client API calls.
- `context_backend.md` — handlers, services, models and backend logic.
- `context_frontend.md` — UI, components, pages and frontend assets.
- `context_tests.md` — tests and test helpers.
- `context_ml_data.md` — ML, datasets, training and data-processing files.
- `context_config_tooling.md` — config, scripts, manifests and tooling.
- `context_keywords.md` — feature keyword to file mapping.
- `context_dependencies.json` — resolved local import edges.
- `context_files.json` — full machine-readable file index.

## Recommended workflow

If task mentions an endpoint or API:

1. Read `context_routes.md`.
2. Pick likely route/handler/service/test files.
3. Open only 1-3 source files.

If task mentions a feature but no endpoint:

1. Read `context_keywords.md`.
2. Then read the matching context file.
3. Open only the most likely source files.

If task is unclear:

1. Read `context_entrypoints.md`.
2. Read one targeted context file.
3. Propose a small file-opening plan.

## Project summary

- indexed files: 33
- indexed source lines: 6489
- large files: 2

## Detected framework hints

- django
- express
- fastapi
- firebase
- flask
- jest/vitest
- nextjs
- pandas
- pytest
- pytorch
- react
- sqlalchemy
- svelte
- tensorflow
- vue

## Likely entrypoints

- `controller.py`
  - python main guard
- `main.py`
  - entrypoint filename
  - python main guard
- `scripts/generate_report.py`
  - python main guard
- `src/entrypoint.py`
  - python main guard

## Large files warning

- `src/report.py` — 842 lines
- `tools/build_project_context.py` — 787 lines

