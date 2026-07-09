"""Post-training validation for F5 (CORAL rank-consistency + quality) and
F11 (MIL sparsity / localisation).

Runs the trained model on a split and reports, per the checklist in
``plans and summaries/07.07_walidacja_po_treningu.md``:

  F5  — ordinal probabilities are non-increasing (rank consistency), and
        age-prediction quality (MAE / RMSE / Acc±1yr).
  F11 — number of "active" patches (patch_prob > threshold) vs. true age:
        we want #active ≈ age and a positive correlation with age.

Usage:
    python scripts/validate_f5_f11.py                        # config.yaml, best.pt
    python scripts/validate_f5_f11.py --config configs/config_embedded.yaml
    python scripts/validate_f5_f11.py --checkpoint checkpoints/embedded/best.pt --split test
    python scripts/validate_f5_f11.py --output outputs/f5_f11_validation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Pure metric helpers (unit-tested — no torch needed)
# ---------------------------------------------------------------------------

def monotonicity_stats(probs: np.ndarray) -> dict:
    """F5: how well ordinal probs are non-increasing along the rank axis.

    probs: (N, K-1) array of sigmoid(coral_logits). Expected P(age>0) ≥ P(age>1) ≥ …
    Returns fraction of adjacent pairs that INCREASE (a rank violation) and the
    largest increase seen. For the shared-weight/increasing-threshold head both
    should be ~0.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] < 2:
        return {"pairs": 0, "violations": 0, "violation_frac": 0.0, "max_increase": 0.0}
    diffs = probs[:, 1:] - probs[:, :-1]          # > 0 ⇒ increase ⇒ violation
    viol = diffs > 1e-6
    return {
        "pairs": int(diffs.size),
        "violations": int(viol.sum()),
        "violation_frac": float(viol.mean()),
        "max_increase": float(diffs.max()),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def active_patch_stats(patch_probs: np.ndarray, ages: np.ndarray,
                       threshold: float = 0.5) -> dict:
    """F11: relate #active patches (and soft count) to the true age.

    patch_probs: (N, P) in [0, 1]; ages: (N,). We want #active ≈ age.
    """
    patch_probs = np.asarray(patch_probs, dtype=float)
    ages = np.asarray(ages, dtype=float)
    n_active = (patch_probs > threshold).sum(axis=1).astype(float)
    soft_count = patch_probs.sum(axis=1).astype(float)

    # mean #active per integer age (localisation should grow with age)
    per_age: dict[str, float] = {}
    for a in sorted(set(int(x) for x in ages)):
        sel = ages == a
        per_age[str(a)] = float(n_active[sel].mean())

    return {
        "n_samples": int(len(ages)),
        "threshold": float(threshold),
        "corr_nactive_age": _corr(n_active, ages),
        "mae_nactive_vs_age": float(np.mean(np.abs(n_active - ages))),
        "corr_softcount_age": _corr(soft_count, ages),
        "mae_softcount_vs_age": float(np.mean(np.abs(soft_count - ages))),
        "mean_nactive": float(n_active.mean()),
        "mean_soft_count": float(soft_count.mean()),
        "mean_nactive_per_age": per_age,
    }


# ---------------------------------------------------------------------------
# Model run
# ---------------------------------------------------------------------------

def resolve_checkpoint(cfg, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = PROJECT_ROOT / ckpt_dir
    best = ckpt_dir / "best.pt"
    if best.exists():
        return best
    ckpts = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))
    return ckpts[-1] if ckpts else None


def collect_outputs(cfg, ckpt_path: Path, split: str):
    """Run the model over the split; return (coral_probs|None, patch_probs|None, ages, pred_ages)."""
    import torch
    from torch.utils.data import DataLoader

    from src.dataset import OtolithDataset, decode_age_ordinal
    from src.inference import load_model_from_checkpoint
    from src.utils import resolve_device

    device = resolve_device(cfg.training.device)
    model = load_model_from_checkpoint(cfg, ckpt_path)
    model.to(device)
    model.eval()

    loader = DataLoader(OtolithDataset(cfg, split=split),
                        batch_size=cfg.training.batch_size, shuffle=False,
                        num_workers=cfg.data.num_workers)

    coral_chunks, patch_chunks, age_chunks, pred_chunks = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            ages = batch["age"].numpy()
            metadata = batch.get("metadata")
            if metadata is not None:
                metadata = metadata.to(device)
            out = model(images, metadata=metadata)

            if "coral_logits" in out:
                probs = torch.sigmoid(out["coral_logits"]).cpu().numpy()
                coral_chunks.append(probs)
                pred_chunks.append(decode_age_ordinal(out["coral_logits"]).cpu().numpy())
            elif "patch_probs" in out:
                # top-k MIL loss: age = #active patches (prob>0.5), not the sum
                pred_chunks.append((out["patch_probs"] > 0.5).sum(dim=1).long().cpu().numpy())

            if "patch_probs" in out:
                patch_chunks.append(out["patch_probs"].cpu().numpy())

            age_chunks.append(ages)

    coral = np.concatenate(coral_chunks) if coral_chunks else None
    patch = np.concatenate(patch_chunks) if patch_chunks else None
    ages = np.concatenate(age_chunks) if age_chunks else np.array([])
    preds = np.concatenate(pred_chunks) if pred_chunks else np.array([])
    return coral, patch, ages, preds


def run_validation(cfg, ckpt_path: Path, split: str, threshold: float) -> dict:
    from src.report_common import compute_metrics

    coral, patch, ages, preds = collect_outputs(cfg, ckpt_path, split)
    report: dict = {
        "checkpoint": str(ckpt_path),
        "split": split,
        "head_type": cfg.model.head_type,
        "n_samples": int(len(ages)),
    }

    # --- F5 ---
    f5: dict = {"applicable": coral is not None}
    if coral is not None:
        f5["monotonicity"] = monotonicity_stats(coral)
        f5["monotonicity"]["pass"] = f5["monotonicity"]["max_increase"] < 1e-3
    if len(preds) == len(ages) and len(ages) > 0:
        m = compute_metrics(ages, preds)
        f5["quality"] = {k: round(v, 4) for k, v in m.items()}
    report["F5"] = f5

    # --- F11 ---
    f11: dict = {"applicable": patch is not None}
    if patch is not None and len(ages) > 0:
        stats = active_patch_stats(patch, ages, threshold)
        corr = stats["corr_nactive_age"]
        mae = stats["mae_nactive_vs_age"]
        stats["pass"] = (not np.isnan(corr)) and corr >= 0.5 and mae <= 2.0
        f11["stats"] = stats
    report["F11"] = f11
    return report


# ---------------------------------------------------------------------------
# CLI / printing
# ---------------------------------------------------------------------------

def _print_report(r: dict) -> None:
    print("=" * 64)
    print(f"Walidacja F5/F11  ·  checkpoint: {r['checkpoint']}")
    print(f"split={r['split']}  head_type={r['head_type']}  n={r['n_samples']}")
    print("=" * 64)

    f5 = r["F5"]
    print("\n[F5] CORAL rank-consistency + jakość")
    if not f5["applicable"]:
        print("  (brak głowicy CORAL — pominięto monotoniczność)")
    else:
        mono = f5["monotonicity"]
        tag = "PASS" if mono["pass"] else "FAIL"
        print(f"  [{tag}] monotoniczność: max_increase={mono['max_increase']:.2e}, "
              f"naruszenia={mono['violations']}/{mono['pairs']} "
              f"({mono['violation_frac']:.3%})")
    if "quality" in f5:
        q = f5["quality"]
        print(f"  [INFO] jakość: MAE={q['MAE']}  RMSE={q['RMSE']}  "
              f"Acc±1yr={q['Acc1yr']:.1%}  Acc±2yr={q['Acc2yr']:.1%}  Bias={q['Bias']:+.3f}")
        print("         (PASS jakości = MAE nie gorsze niż baseline — porównaj ręcznie)")

    f11 = r["F11"]
    print("\n[F11] MIL — liczba aktywnych patchy vs wiek")
    if not f11["applicable"]:
        print("  (brak głowicy MIL — ustaw head_type='mil'/'both'; pominięto)")
    else:
        s = f11["stats"]
        tag = "PASS" if s["pass"] else "WARN"
        print(f"  [{tag}] corr(#aktywnych, wiek)={s['corr_nactive_age']:.3f}  "
              f"MAE(#aktywnych vs wiek)={s['mae_nactive_vs_age']:.3f}  "
              f"(próg prob>{s['threshold']})")
        print(f"         soft-count: corr={s['corr_softcount_age']:.3f}  "
              f"MAE={s['mae_softcount_vs_age']:.3f}  śr.#aktywnych={s['mean_nactive']:.2f}")
        print("         śr. #aktywnych per wiek:",
              {a: round(v, 1) for a, v in s["mean_nactive_per_age"].items()})
        if tag == "WARN":
            print("         → rozważ strojenie mil_sparsity_weight (F11) po treningu.")
    print()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walidacja F5/F11 po treningu")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    p.add_argument("--checkpoint", default=None,
                   help="Ścieżka do .pt; domyślnie best.pt/najnowszy z checkpoint_dir")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Próg prawdopodobieństwa patcha dla '#aktywnych'")
    p.add_argument("--output", default=None, help="Zapis raportu do JSON")
    return p.parse_args(argv)


def main(argv=None) -> int:
    from src.config import load_config

    args = parse_args(argv)
    cfg = load_config(args.config)
    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    if ckpt is None:
        print("[błąd] Nie znaleziono checkpointu — najpierw wytrenuj model "
              "(albo podaj --checkpoint).")
        return 1

    report = run_validation(cfg, ckpt, args.split, args.threshold)
    _print_report(report)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Zapisano raport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
