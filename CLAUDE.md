# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OtolithDinoStandalone — weakly supervised fish age prediction from otolith (ear bone) images using DINOv2 (self-supervised ViT). The pipeline covers data preparation → training → inference → heatmap interpretation → increment-marker candidate detection → HTML report generation.

## Commands

### Run tests
```bash
python -m pytest                          # all 152 tests
python -m pytest tests/test_stage3_model.py  # single test file
python scripts/smoke_test.py              # standalone smoke test (no pytest)
```

### Main entry point
```bash
python -m src.entrypoint --config configs/config.yaml --mode info
python -m src.entrypoint --config configs/config.yaml --mode train
python -m src.entrypoint --config configs/config.yaml --mode inference
python -m src.entrypoint --config configs/config.yaml --mode report
```

### Data preparation
```bash
python scripts/prepare_labels.py   # reads data/analysisWithOtolithPhoto.xlsx → data/labels.csv
python scripts/generate_report.py  # standalone report generation
```

## Architecture

### Data flow
```
data/labels.csv + image_dir
  → OtolithDataset (ordinal age encoding)
  → DataLoader (train 70% / val 15% / test 15%, fish-level split to prevent leakage)
  → Trainer.fit()  →  checkpoints/
  → run_inference()  →  outputs/predictions.csv + predictions.json
  → run_interpretation()  →  outputs/heatmaps/ + outputs/overlays/
  → run_candidates()  →  outputs/candidates/ (JSON peak files)
  → build_html_report()  →  outputs/report.html
```

### Key modules (`src/`)

| Module | Role |
|---|---|
| `config.py` | Pydantic config hierarchy loaded from `configs/config.yaml`; `ProjectConfig` is the root |
| `dataset.py` | `OtolithDataset` — ordinal encoding `vec[i] = 1 iff age > i` (CORAL method); ImageNet normalization |
| `model.py` | `OtolithModel` — DINOv2 backbone + dropout + linear ordinal head; `get_patch_tokens()` for interpretation |
| `trainer.py` | `Trainer` — AdamW + cosine/step scheduler; supports backbone freezing for first N epochs |
| `inference.py` | `run_inference()` — batch prediction → CSV/JSON; `load_model_from_checkpoint()` |
| `interpretation.py` | Patch token L2-norm importance → JET heatmaps + alpha-blended overlays |
| `candidates.py` | Radial (row-mean) profile of importance grid → `scipy` peak detection → JSON pixel positions |
| `report.py` | `build_html_report()` — self-contained HTML with all plots embedded as base64 PNG |
| `entrypoint.py` | CLI dispatcher for all modes |

### Configuration (`configs/config.yaml`)

Key fields to know:
- `model.backbone`: `dinov2_vits14` (patch_size=14)
- `model.num_age_classes`: 17 — ordinal head outputs K-1 logits
- `data.image_dir`: `Z:/Photo/Otolithes/HER/Processed` (real data path)
- `training.freeze_backbone_epochs`: backbone frozen for this many initial epochs
- `training.device`: `auto` resolves to CUDA → MPS → CPU

### Testing conventions

- One test file per development stage: `test_stage1_scaffold.py` … `test_stage8_end_to_end.py`
- Stage 8 runs the full pipeline on synthetic data using `MockDinoBackbone` — no internet or GPU needed
- `test_bootstrap.py` validates the project scaffold

### State tracking files (do not delete)

- `state.json` — current stage, last action, next action
- `progress.md` — stage completion checklist
- `next_step.txt` — plain-English instructions for the next real-data run
- `controller.py` — orchestration script that reads state.json and invokes Claude sessions

## Real-data pipeline (next steps)

1. Mount `Z:` drive (network share with raw otolith images)
2. `python scripts/prepare_labels.py` — produces `data/labels.csv` with leak-free fish-level splits
3. `python -m src.entrypoint --mode train`
4. Monitor `logs/train.log`
5. `python -m src.entrypoint --mode inference`
6. `python -m src.entrypoint --mode report` → `outputs/report.html`
