# ML and Data Context

Files indexed here: 28

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

## `src/__init__.py`

- kind: `ml_data`
- roles: ml_data
- lines: 1
- size_bytes: 85
- keywords:
  - predict

## `src/candidates.py`

- kind: `data_model`
- roles: data_model, ml_data
- lines: 247
- size_bytes: 8676
- framework hints:
  - pytorch
- data access hints:
  - file_io
- keywords:
  - model
  - config
  - file
  - map
  - train
- symbols:
  - extract_radial_profile
  - find_candidate_peaks
  - peaks_to_pixel_positions
  - run_candidates
  - save_candidates_json
  - save_candidates_overlay

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

## `src/dataset.py`

- kind: `data_model`
- roles: data_model, ml_data
- lines: 225
- size_bytes: 7964
- framework hints:
  - pandas
  - pytorch
- data access hints:
  - file_io
- keywords:
  - api
  - model
  - config
  - file
  - map
  - train
  - predict
  - dataset
  - form
- symbols:
  - OtolithDataset
  - __getitem__
  - __init__
  - __len__
  - _encode_metadata
  - _encode_sex
  - _load_image
  - _try_float
  - _validate_columns
  - build_transforms
  - decode_age_ordinal
  - encode_age_ordinal
  - metadata_dim

## `src/entrypoint.py`

- kind: `data_model`
- roles: data_model, ml_data
- lines: 103
- size_bytes: 3287
- framework hints:
  - jest/vitest
- entrypoint hints:
  - python main guard
- keywords:
  - model
  - config
  - file
  - map
  - train
  - predict
- symbols:
  - main
  - parse_args
  - print_config_summary
  - run

## `src/inference.py`

- kind: `data_model`
- roles: data_model, ml_data
- lines: 124
- size_bytes: 3998
- framework hints:
  - pandas
  - pytorch
- data access hints:
  - file_io
- keywords:
  - model
  - test
  - mock
  - config
  - map
  - train
  - predict
  - dataset
  - state
- symbols:
  - load_model_from_checkpoint
  - run_inference

## `src/interpretation.py`

- kind: `data_model`
- roles: data_model, ml_data
- lines: 179
- size_bytes: 6051
- framework hints:
  - pytorch
- keywords:
  - model
  - config
  - file
  - map
  - train
  - form
- symbols:
  - compute_patch_importance
  - importance_to_heatmap
  - run_interpretation
  - save_heatmap
  - save_overlay

## `src/model.py`

- kind: `data_model`
- roles: data_model, ml_data
- lines: 128
- size_bytes: 4398
- framework hints:
  - pytorch
- keywords:
  - model
  - test
  - config
  - map
  - train
  - loss
- symbols:
  - OtolithModel
  - __init__
  - backbone_is_frozen
  - forward
  - freeze_backbone
  - get_cls_and_patches
  - get_patch_tokens
  - load_dinov2
  - ordinal_loss
  - unfreeze_backbone

## `src/report.py`

- kind: `data_model`
- roles: data_model, ml_data
- lines: 842
- size_bytes: 35248
- warnings:
  - large file: 842 lines
- framework hints:
  - pandas
- data access hints:
  - file_io
- keywords:
  - model
  - test
  - file
  - map
  - train
  - loss
  - form
- symbols:
  - _fig_to_b64
  - _grid
  - _img_tag
  - _notice
  - _pil_to_b64
  - _section
  - _table
  - _two_col
  - build_data_section

## `src/trainer.py`

- kind: `data_model`
- roles: data_model, ml_data
- lines: 214
- size_bytes: 7567
- framework hints:
  - jest/vitest
  - pytorch
- data access hints:
  - file_io
- keywords:
  - model
  - test
  - config
  - file
  - map
  - train
  - predict
  - dataset
  - loss
  - state
- symbols:
  - Trainer
  - __init__
  - _build_scheduler
  - _log
  - _log_epoch
  - _resolve_dir
  - fit
  - load_checkpoint
  - save_checkpoint
  - train_one_epoch
  - validate

## `src/utils.py`

- kind: `ml_data`
- roles: ml_data
- lines: 29
- size_bytes: 1054
- framework hints:
  - pytorch
- symbols:
  - resolve_device
  - tensor_to_uint8_rgb

## `state.json`

- kind: `ml_data`
- roles: ml_data
- lines: 11
- size_bytes: 657
- keywords:
  - test
  - train
  - state

## `tests/test_stage1_scaffold.py`

- kind: `test`
- roles: data_model, ml_data, test
- lines: 69
- size_bytes: 2114
- framework hints:
  - pytest
- keywords:
  - model
  - test
  - config
  - file
  - train
- symbols:
  - test_config_data_splits_sum
  - test_config_image_size_divisible
  - test_config_loads_without_error
  - test_config_model_fields
  - test_config_yaml_exists
  - test_default_config_is_valid
  - test_entrypoint_info_mode
  - test_entrypoint_missing_config_raises
  - test_invalid_splits_raise_error

## `tests/test_stage2_dataset.py`

- kind: `test`
- roles: data_model, ml_data, test
- lines: 246
- size_bytes: 8784
- framework hints:
  - pandas
  - pytest
  - pytorch
- data access hints:
  - file_io
- keywords:
  - model
  - test
  - fixture
  - config
  - file
  - train
  - dataset
  - form
- symbols:
  - _make_cfg
  - dummy_data
  - test_dataset_age_dtype
  - test_dataset_age_ordinal_shape
  - test_dataset_image_id_is_string
  - test_dataset_image_shape
  - test_dataset_split_sizes
  - test_decode_age_all_positive
  - test_decode_age_ordinal
  - test_encode_age_max
  - test_encode_age_middle
  - test_encode_age_overflow_clamped
  - test_encode_age_zero
  - test_encode_decode_roundtrip
  - test_metadata_dim_property
  - test_metadata_not_present_when_disabled
  - test_metadata_tensor_when_enabled
  - test_missing_required_column_raises
  - test_negative_age_raises
  - test_sex_encoding_values

## `tests/test_stage3_model.py`

- kind: `test`
- roles: data_model, ml_data, test
- lines: 297
- size_bytes: 9534
- framework hints:
  - pytest
  - pytorch
- keywords:
  - model
  - test
  - mock
  - config
  - train
  - predict
  - dataset
  - loss
- symbols:
  - _MockDinoBackbone
  - __init__
  - _dummy_batch
  - _make_model
  - forward
  - forward_features
  - test_backbone_is_frozen_flag
  - test_backward_frozen_backbone_head_gets_grad
  - test_backward_frozen_backbone_no_grad_on_backbone
  - test_backward_unfrozen_backbone_all_grads
  - test_forward_batch_size_one
  - test_forward_output_changes_with_different_inputs
  - test_forward_output_dtype
  - test_forward_output_shape
  - test_freeze_does_not_affect_head
  - test_freeze_makes_backbone_params_no_grad
  - test_model_embed_dim_from_backbone
  - test_model_instantiates
  - test_ordinal_loss_has_gradient
  - test_ordinal_loss_is_positive
  - test_ordinal_loss_is_scalar
  - test_ordinal_loss_low_for_perfect_prediction
  - test_train_step_completes
  - test_unfreeze_restores_backbone_params_grad

## `tests/test_stage4_trainer.py`

- kind: `test`
- roles: data_model, ml_data, test
- lines: 287
- size_bytes: 9648
- framework hints:
  - jest/vitest
  - pytest
  - pytorch
- keywords:
  - model
  - test
  - mock
  - config
  - file
  - train
  - dataset
  - loss
- symbols:
  - _MockDinoBackbone
  - _SyntheticDataset
  - __getitem__
  - __init__
  - __len__
  - _make_cfg
  - _make_loader
  - _make_model
  - _make_trainer
  - forward
  - forward_features
  - test_fit_creates_checkpoint_per_epoch
  - test_fit_writes_log_file
  - test_load_checkpoint_returns_epoch
  - test_load_checkpoint_round_trip
  - test_resolve_device_auto_returns_device
  - test_resolve_device_cpu
  - test_save_checkpoint_creates_file
  - test_save_checkpoint_filename_contains_epoch_and_loss
  - test_train_one_epoch_returns_float
  - test_train_one_epoch_updates_weights
  - test_validate_does_not_update_weights
  - test_validate_loss_positive
  - test_validate_mae_nonneg
  - test_validate_no_val_loader_returns_nan
  - test_validate_returns_two_floats

## `tests/test_stage5_inference.py`

- kind: `test`
- roles: data_model, ml_data, test
- lines: 315
- size_bytes: 10808
- framework hints:
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
  - file
  - train
  - predict
  - dataset
  - loss
  - state
- symbols:
  - _MockDinoBackbone
  - _SyntheticDataset
  - _SyntheticDatasetNoAge
  - __getitem__
  - __init__
  - __len__
  - _make_cfg
  - _make_loader
  - _make_model
  - _save_checkpoint
  - forward
  - forward_features
  - test_abs_error_equals_abs_diff
  - test_inference_creates_csv
  - test_inference_creates_json
  - test_predicted_age_in_valid_range
  - test_predicted_age_is_integer
  - test_predictions_csv_required_columns
  - test_predictions_csv_row_count
  - test_predictions_json_has_all_fields
  - test_predictions_json_parseable
  - test_summary_has_required_keys
  - test_summary_n_samples_correct

## `tests/test_stage6_interpretation.py`

- kind: `test`
- roles: data_model, ml_data, test
- lines: 369
- size_bytes: 12790
- framework hints:
  - pytest
  - pytorch
- data access hints:
  - file_io
- keywords:
  - model
  - test
  - mock
  - config
  - file
  - map
  - train
  - dataset
  - form
- symbols:
  - _MockDinoBackbone
  - _SyntheticDataset
  - __getitem__
  - __init__
  - __len__
  - _make_cfg
  - _make_loader
  - _make_model
  - _make_single_image_tensor
  - forward
  - forward_features
  - test_heatmap_different_output_sizes
  - test_heatmap_dtype_is_float32
  - test_heatmap_nonuniform_input_has_nonzero_range
  - test_heatmap_output_shape
  - test_heatmap_uniform_input_is_zero
  - test_heatmap_values_in_0_1
  - test_patch_importance_accepts_3d_input
  - test_patch_importance_no_grad
  - test_patch_importance_nonneg
  - test_patch_importance_nonzero_for_nonzero_patches
  - test_patch_importance_shape
  - test_save_heatmap_correct_size
  - test_save_heatmap_creates_file
  - test_save_heatmap_is_grayscale

## `tests/test_stage7_candidates.py`

- kind: `test`
- roles: data_model, ml_data, test
- lines: 437
- size_bytes: 15983
- framework hints:
  - pytest
  - pytorch
- keywords:
  - model
  - test
  - mock
  - config
  - file
  - map
  - train
  - dataset
  - form
- symbols:
  - _MockDinoBackbone
  - _SyntheticDataset
  - __getitem__
  - __init__
  - __len__
  - _make_cfg
  - _make_loader
  - _make_model
  - forward
  - forward_features
  - test_peaks_flat_profile_no_peaks
  - test_peaks_indices_in_valid_range
  - test_peaks_known_single_peak
  - test_peaks_multiple_well_separated
  - test_peaks_respects_min_distance
  - test_peaks_returns_ndarray
  - test_pixel_positions_all_in_range
  - test_pixel_positions_empty_input
  - test_pixel_positions_first_patch_center
  - test_pixel_positions_last_patch_center
  - test_pixel_positions_monotone_increasing
  - test_radial_profile_accepts_tensor
  - test_radial_profile_dtype
  - test_radial_profile_shape
  - test_radial_profile_shape_non_square
  - test_radial_profile_values_are_row_means

## `tests/test_stage8_end_to_end.py`

- kind: `test`
- roles: data_model, ml_data, test
- lines: 212
- size_bytes: 8310
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
- symbols:
  - _MockDinoBackbone
  - __init__
  - _make_cfg
  - _make_synthetic_data
  - forward
  - forward_features
  - test_full_pipeline

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

