# Entrypoints and Manifests

Files indexed here: 8

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

## `configs/config.yaml`

- kind: `data_model`
- roles: config, data_model, ml_data
- lines: 52
- size_bytes: 1390
- keywords:
  - model
  - test
  - config
  - worker
  - map
  - train

## `controller.py`

- kind: `handler_controller`
- roles: data_model, handler_controller
- lines: 114
- size_bytes: 3142
- framework hints:
  - jest/vitest
  - pytest
- entrypoint hints:
  - python main guard
- data access hints:
  - file_io
  - sql
- keywords:
  - permission
  - controller
  - test
  - task
  - file
  - state
- symbols:
  - load_state
  - log
  - main
  - read_next_step
  - run_claude_new
  - run_claude_resume
  - save_state
  - smoke_test

## `main.py`

- kind: `ml_data`
- roles: ml_data
- lines: 12
- size_bytes: 316
- entrypoint hints:
  - entrypoint filename
  - python main guard
- keywords:
  - train

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

## `src/entrypoint.py`

- kind: `data_model`
- roles: data_model, ml_data
- lines: 191
- size_bytes: 6985
- framework hints:
  - jest/vitest
  - pytorch
- entrypoint hints:
  - python main guard
- keywords:
  - api
  - model
  - test
  - config
  - worker
  - file
  - map
  - train
  - predict
  - dataset
- symbols:
  - _build_loaders
  - main
  - parse_args
  - print_config_summary
  - run

