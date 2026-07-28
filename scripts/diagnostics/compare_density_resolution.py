"""22.07: compare localization_quality.json before/after candidates.density_image_size /
density_crop_to_otolith (Zmiana A+B), on a real checkpoint + its predictions.csv (no
re-inference — CORAL predictions are unaffected by density-only report/localisation code).

Loads the checkpoint into a model whose density_head is rebuilt to the checkpoint's
ORIGINAL (pre-LayerNorm) shape, so this isolates JUST the resolution/crop change — not
the separate, training-time LayerNorm/density_lr_mult/TV changes from the same session,
which need an actual new training run to evaluate (one variable at a time).

NOTE: since ``src.inference.load_model_from_checkpoint`` was fixed (22.07) to gracefully
drop shape-mismatched keys instead of crashing, this custom loader is no longer needed to
avoid an error — it's kept because it's the only way to test with the checkpoint's
ACTUAL TRAINED density_head weights instead of a freshly random-initialised one, which is
the point of this specific comparison. Once a real fine-tuned checkpoint exists (see
``scripts/finetune_density_head.py``), compare it directly with the default loader instead.

Usage: python scripts/diagnostics/compare_density_resolution.py
  (edit CKPT/RUN_DIR below to point at a different run)
"""
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path("C:/Users/kswitek/Documents/DINO_project_Herring")
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.utils import configure_attention
from scripts.run_pipeline import load_merged_config, _step_cards

CKPT = PROJECT_ROOT / "outputs" / "20.07_reg" / "checkpoints" / "embedded" / "best.pt"
RUN_DIR = PROJECT_ROOT / "outputs" / "20.07_reg"
TOP_K = 15   # best + worst -> 30 cards total (established sample size, 21.07)


def _load_legacy_density_model(cfg, ckpt_path, backbone=None):
    """Load into a density_head rebuilt to the PRE-LayerNorm shape (this checkpoint's
    actual training-time architecture) instead of today's default (LayerNorm-first)."""
    from src.model import OtolithModel
    model = OtolithModel(cfg, backbone=backbone)
    if hasattr(model, "density_head"):
        embed_dim = model.density_head[1].in_features   # Linear right after LayerNorm
        model.density_head = nn.Sequential(
            nn.Linear(embed_dim, cfg.model.mil_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=cfg.model.dropout),
            nn.Linear(cfg.model.mil_hidden_dim, 1),
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"  [load] missing={result.missing_keys} unexpected={result.unexpected_keys}")
    model.eval()
    return model


def run_variant(cfg, name: str, density_image_size, crop: bool) -> dict:
    print(f"=== {name}: density_image_size={density_image_size} crop={crop} ===", flush=True)
    cfg_v = cfg.model_copy(deep=True)
    cfg_v.candidates.density_image_size = density_image_size
    cfg_v.candidates.density_crop_to_otolith = crop
    out_dir = PROJECT_ROOT / "outputs" / f"22.07_{name}_local"
    src_pred = RUN_DIR / "emb_on_emb" / "predictions.csv"
    (out_dir / "emb_on_emb").mkdir(parents=True, exist_ok=True)
    dst_pred = out_dir / "emb_on_emb" / "predictions.csv"
    if not dst_pred.exists():
        shutil.copy(src_pred, dst_pred)
    pred_csvs = {"emb_on_emb": dst_pred}
    cond_models = {"emb_on_emb": (cfg_v, CKPT)}
    _step_cards(pred_csvs, cfg_v, Path(cfg_v.data.image_dir), out_dir, cond_models)
    return json.loads((out_dir / "localization_quality.json").read_text(encoding="utf-8"))["emb_on_emb"]


def main() -> None:
    import src.inference as _inf
    _inf.load_model_from_checkpoint = _load_legacy_density_model

    cfg = load_merged_config(PROJECT_ROOT / "configs" / "config.yaml",
                             PROJECT_ROOT / "configs" / "config_embedded.yaml")
    cfg.data.image_dir = "Z:/Photo/Otolithes/HER/Processed"
    cfg.inference.increment_samples.top_k_best = TOP_K
    cfg.inference.increment_samples.top_k_worst = TOP_K
    configure_attention(cfg.interpretation.disable_fused_attention)

    baseline = run_variant(cfg, "baseline", None, False)
    hires = run_variant(cfg, "hires", 728, True)

    print("\n=== BASELINE per_method ===")
    print(json.dumps(baseline["per_method"], indent=2, ensure_ascii=False))
    print("mean_attn_bg_frac:", baseline.get("mean_attn_bg_frac"))
    print("\n=== HIRES+CROP per_method ===")
    print(json.dumps(hires["per_method"], indent=2, ensure_ascii=False))
    print("mean_attn_bg_frac:", hires.get("mean_attn_bg_frac"))
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
