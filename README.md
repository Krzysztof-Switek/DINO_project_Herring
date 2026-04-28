# OtolithDinoStandalone

Standalone project for weakly supervised otolith analysis with DINOv2.

## Main goal
- predict fish age from otolith images
- generate interpretable overlays / heatmaps
- propose candidate increment markers
- optionally use biological metadata

## Working rules
- Claude works only inside this repository
- small steps only
- every step must leave a testable artifact
- after each step update:
  - `state.json`
  - `progress.md`
  - `next_step.txt`

## Expected project areas
- `src/` main code
- `tests/` tests
- `configs/` YAML/JSON configs
- `outputs/` inference outputs, heatmaps, overlays
- `logs/` runtime logs
- `checkpoints/` model checkpoints