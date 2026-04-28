# Config and Tooling Context

Files indexed here: 11

## `.claude/settings.json`

- kind: `data_model`
- roles: config, data_model
- lines: 32
- size_bytes: 722
- framework hints:
  - jest/vitest
  - pytest
- keywords:
  - permission
  - schema
  - test
  - settings
  - env
  - form

## `.claude/settings.local.json`

- kind: `data_model`
- roles: config, data_model
- lines: 26
- size_bytes: 767
- framework hints:
  - pytest
- keywords:
  - permission
  - schema
  - test
  - settings
  - env
  - file

## `CLAUDE.md`

- kind: `handler_controller`
- roles: data_model, docs, handler_controller, ml_data
- lines: 90
- size_bytes: 4050
- framework hints:
  - jest/vitest
  - pytest
  - pytorch
- keywords:
  - role
  - controller
  - model
  - test
  - mock
  - config
  - file
  - map
  - train
  - predict
  - dataset
  - state

## `README.md`

- kind: `data_model`
- roles: data_model, docs, ml_data
- lines: 26
- size_bytes: 701
- keywords:
  - model
  - test
  - config
  - map
  - predict
  - state

## `configs/config.yaml`

- kind: `data_model`
- roles: config, data_model, ml_data
- lines: 52
- size_bytes: 1446
- keywords:
  - model
  - test
  - config
  - worker
  - map
  - train

## `progress.md`

- kind: `data_model`
- roles: data_model, docs, ml_data
- lines: 30
- size_bytes: 1744
- data access hints:
  - sql
- keywords:
  - model
  - test
  - config
  - task
  - file
  - map
  - train
  - dataset
  - loss
  - state
  - form

## `scripts/generate_report.py`

- kind: `ml_data`
- roles: ml_data, tooling
- lines: 118
- size_bytes: 4122
- entrypoint hints:
  - python main guard
- keywords:
  - file
  - map
  - train
  - predict
- symbols:
  - _opt
  - _resolve
  - _status
  - main
  - parse_args

## `scripts/prepare_labels.py`

- kind: `ml_data`
- roles: ml_data, tooling
- lines: 336
- size_bytes: 13027
- framework hints:
  - jest/vitest
  - pandas
- keywords:
  - test
  - file
  - map
  - train
  - predict
- symbols:
  - _label
  - assign_split_by_fish
  - build_labels
  - extract_fish_id
  - scan_image_dir

## `scripts/smoke_test.py`

- kind: `test`
- roles: data_model, ml_data, test, tooling
- lines: 283
- size_bytes: 9550
- framework hints:
  - jest/vitest
  - pandas
  - pytest
  - pytorch
- data access hints:
  - file_io
- keywords:
  - model
  - test
  - mock
  - config
  - worker
  - file
  - map
  - train
  - predict
  - dataset
  - form
- symbols:
  - _MockDinoBackbone
  - __init__
  - _make_cfg
  - _make_data
  - build_datasets
  - candidates
  - decorator
  - forward
  - forward_features
  - inference
  - interpretation
  - load_ckpt
  - run_smoke_test
  - step
  - train
  - wrapper

## `src/config.py`

- kind: `data_model`
- roles: config, data_model, ml_data
- lines: 110
- size_bytes: 3733
- data access hints:
  - file_io
- keywords:
  - model
  - test
  - config
  - worker
  - file
  - map
  - train
- symbols:
  - CandidatesConfig
  - DataConfig
  - InferenceConfig
  - InterpretationConfig
  - ModelConfig
  - OtolithConfig
  - ProjectConfig
  - TrainingConfig
  - get_default_config
  - image_size_divisible
  - load_config
  - splits_sum_to_one

## `tools/build_project_context.py`

- kind: `test`
- roles: data_model, frontend, handler_controller, ml_data, route_or_api, test, tooling
- lines: 787
- size_bytes: 24866
- warnings:
  - large file: 787 lines
- framework hints:
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
- data access hints:
  - file_io
  - firestore
- keywords:
  - role
  - api
  - route
  - model
  - test
  - config
  - env
  - cache
  - file
  - component
  - state
- symbols:
  - class
  - detect_entrypoints
  - detect_frameworks
  - extract_exports
  - extract_imports
  - extract_routes
  - extract_symbols
  - is_local_import
  - is_probably_generated
  - read_text
  - rel_path
  - should_exclude
  - trim
  - unique_sorted
- routes/api hints:
  - GET ,
  - GET ],

