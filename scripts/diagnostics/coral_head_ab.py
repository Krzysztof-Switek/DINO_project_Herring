"""24.09 — clean A/B of the quarter age adjustment, N seeds, on cached CLS features.

THE QUESTION. `06.08_attention_first` (quarter adjustment ON) scores 53.85 % exact against
`09.08_isolate_a` (OFF) at 43.71 % — but those two runs also differ in `density_concentricity_weight`
and in their whole optimisation trajectory, so +10.1 pp is not attributable to the flag. This
isolates the flag and nothing else.

HOW IT IS CLEAN. The quarter adjustment is purely a RELABELLING of the target
(`src/dataset.py::_effective_age`): same images, same architecture, same features. Running it on
cached CLS features (`scripts/diagnostics/cache_cls_features.py`) therefore holds every other
variable exactly fixed — bit-identical inputs, and the same initialisation per seed in both arms,
so the comparison is paired.

HOW BOTH ARMS ARE MADE COMPARABLE. They predict different things: OFF predicts recorded age, ON
predicts complete rings visible. Scoring them as-is would compare two different tasks. So the ON
arm's predictions are rebased to the recorded scale (`src.report_common.rebase_to_recorded`, add
the season offset back) and BOTH arms are then scored against recorded age. On this dataset the
rebase is exact: there are zero age-0 fish in the Q1 campaigns, so the `max(age-1, 0)` clamp in
`_effective_age` never fires and the relabelling is a bijection.

WHAT THIS DOES NOT ANSWER. The backbone is frozen. Production unfreezes it at epoch 6, and a
backbone free to adapt to a changed target can gain more than the head alone. This is a clean
lower bound with proper seed statistics, not a substitute for the fine-tuned run.

    python scripts/diagnostics/coral_head_ab.py --tag reg_masked_518 --seeds 10

Writes experiments/quarter_ab/<tag>/{metrics.json, per_seed.csv, per_age.csv, notes.md}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.config import OtolithConfig
from src.dataset import decode_age_ordinal, encode_age_ordinal
from src.model import OtolithModel, ordinal_loss
from src.report_common import (
    QUARTER_CAMPAIGNS,
    campaign_of,
    compute_metrics,
    compute_per_age_metrics,
    macro_exact,
    rebase_to_recorded,
)
from src.utils import seed_everything

CACHE_ROOT = PROJECT_ROOT / "data" / "cls_cache"
OUT_ROOT = PROJECT_ROOT / "experiments" / "quarter_ab"
LABELS = PROJECT_ROOT / "data" / "labels_embedded.csv"


class _CachedBackbone(torch.nn.Module):
    """Stands in for DINOv2 and returns the cached CLS token unchanged.

    Using the real `OtolithModel` with this stub means the head, the monotone threshold
    construction (`_coral_logits`), the loss and the decode are all the PRODUCTION code paths,
    not a reimplementation that could drift from them.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.embed_dim = dim

    def forward_features(self, x: torch.Tensor) -> dict:
        return {"x_norm_clstoken": x,
                "x_norm_patchtokens": x.new_zeros((x.shape[0], 1, x.shape[1]))}


def load_cache(tag: str):
    d = CACHE_ROOT / tag
    if not (d / "features.npy").exists():
        raise SystemExit(f"no cache at {d} — run cache_cls_features.py --tag {tag}")
    X = np.load(d / "features.npy")
    index = pd.read_csv(d / "index.csv")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    if len(index) != len(X):
        raise SystemExit(f"cache is inconsistent: {len(X)} features vs {len(index)} rows")
    return X, index, meta


def targets_for_arm(index: pd.DataFrame, quarter: bool) -> np.ndarray:
    """The training target for one arm. ``quarter=True`` reproduces `_effective_age`."""
    age = index["age"].to_numpy(dtype=int)
    if not quarter:
        return age
    is_q1 = np.isin(campaign_of(index["image_id"].values), QUARTER_CAMPAIGNS)
    return np.maximum(age - is_q1.astype(int), 0)


def build_head(cfg: OtolithConfig, dim: int, device) -> OtolithModel:
    model = OtolithModel(cfg, backbone=_CachedBackbone(dim)).to(device)
    return model


def train_one(X: np.ndarray, y: np.ndarray, splits: np.ndarray, cfg: OtolithConfig,
              seed: int, epochs: int, device, recorded: np.ndarray,
              rebase_ids: np.ndarray | None, lr: float | None = None) -> dict:
    """Train the head on the train split; keep the epoch with the best VAL exact accuracy.

    Selecting on exact accuracy (not MAE) because that is the metric under study, and the same
    rule is applied to both arms so the comparison stays fair. Val exact is always measured on
    the RECORDED scale, so the two arms are selected against the same yardstick too.
    """
    seed_everything(seed)
    model = build_head(cfg, X.shape[1], device)
    K = cfg.model.num_age_classes

    tr, va = splits == "train", splits == "val"
    Xtr = torch.from_numpy(X[tr]).to(device)
    Xva = torch.from_numpy(X[va]).to(device)
    Ytr = torch.stack([encode_age_ordinal(int(a), K) for a in y[tr]]).to(device)

    # Production's lr is tuned for a head training ALONGSIDE the backbone. On frozen
    # features the head has to move much further on its own, so the rate is a nuisance
    # parameter here — picked once by `pilot_lr` on val and then shared by both arms.
    opt = torch.optim.AdamW(model.parameters(), lr=lr or cfg.training.lr,
                            weight_decay=cfg.training.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    bs = cfg.training.batch_size
    n = Xtr.shape[0]
    g = torch.Generator().manual_seed(seed)

    best = {"val_exact": -1.0, "state": None, "epoch": -1}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            loss = ordinal_loss(model(Xtr[idx])["coral_logits"], Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pred_va = decode_age_ordinal(model(Xva)["coral_logits"]).cpu().numpy()
        pred_va = _to_recorded(pred_va, rebase_ids[va] if rebase_ids is not None else None)
        val_exact = float(np.mean(pred_va == recorded[va]))
        if val_exact > best["val_exact"]:
            best = {"val_exact": val_exact, "epoch": ep,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    model.eval()
    out = {"seed": seed, "best_epoch": best["epoch"], "val_exact": best["val_exact"],
           "lr": float(lr or cfg.training.lr), "epochs": epochs,
           # A best epoch at the very end means val was still improving when the budget
           # ran out: the arm is under-trained and the comparison would be measuring the
           # budget, not the target.
           "still_improving": bool(best["epoch"] >= epochs - 1)}
    for name, mask in (("val", va), ("test", splits == "test")):
        with torch.no_grad():
            pred = decode_age_ordinal(
                model(torch.from_numpy(X[mask]).to(device))["coral_logits"]).cpu().numpy()
        pred = _to_recorded(pred, rebase_ids[mask] if rebase_ids is not None else None)
        m = compute_metrics(recorded[mask], pred)
        pa = compute_per_age_metrics(recorded[mask], pred)
        out[name] = {"Exact": m["Exact"], "Acc1yr": m["Acc1yr"], "MAE": m["MAE"],
                     "MedAE": m["MedAE"], "Bias": m["Bias"], "MacroExact": macro_exact(pa),
                     "n": m["N"]}
        if name == "test":
            out["test_pred"] = pred
    return out


def _to_recorded(pred: np.ndarray, ids: np.ndarray | None) -> np.ndarray:
    """Put an arm's predictions on the recorded-age scale (no-op for the OFF arm)."""
    if ids is None:
        return pred.astype(float)
    return rebase_to_recorded(ids, pred)


def pilot_lr(X, index, splits, cfg, device, recorded, ids, grid, epochs, seed=42) -> dict:
    """Pick one learning rate on val, shared by both arms.

    Chosen by the MEAN val exact across the two arms rather than on one of them, so the choice
    cannot quietly favour either. One seed only — this is a nuisance parameter, not the thing
    under study, and spending the seed budget on it would defeat the point.
    """
    scores = {}
    for lr in grid:
        per_arm = []
        for quarter in (False, True):
            y = targets_for_arm(index, quarter)
            r = train_one(X, y, splits, cfg, seed, epochs, device, recorded,
                          ids if quarter else None, lr=lr)
            per_arm.append(r["val_exact"])
        scores[lr] = {"mean_val_exact": float(np.mean(per_arm)),
                      "per_arm": [float(v) for v in per_arm]}
        print(f"  pilot lr={lr:<8} val exact OFF {100 * per_arm[0]:.2f}% / "
              f"ON {100 * per_arm[1]:.2f}%  -> srednia {100 * scores[lr]['mean_val_exact']:.2f}%",
              flush=True)
    best = max(scores, key=lambda k: scores[k]["mean_val_exact"])
    return {"grid": [float(g) for g in grid], "scores": {str(k): v for k, v in scores.items()},
            "chosen_lr": float(best)}


def summarise(runs: list[dict], key: str = "test") -> dict:
    fields = ("Exact", "Acc1yr", "MAE", "MedAE", "Bias", "MacroExact")
    return {f: {"mean": float(np.mean([r[key][f] for r in runs])),
                "sd": float(np.std([r[key][f] for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
                "values": [float(r[key][f]) for r in runs]} for f in fields}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="reg_masked_518")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr-grid", default="1e-4,5e-4,2e-3,1e-2",
                    help="pilot grid for the head learning rate")
    ap.add_argument("--lr", type=float, default=None,
                    help="skip the pilot and use this lr for both arms")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    X, index, meta = load_cache(args.tag)
    out_dir = Path(args.out) if args.out else OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OtolithConfig()
    cfg.model.head_type = "coral"
    cfg.model.use_density_head = False
    cfg.training.device = "cpu"
    device = torch.device("cpu")

    splits = index["split"].to_numpy()
    recorded = index["age"].to_numpy(dtype=float)
    ids = index["image_id"].to_numpy()
    seeds = [42, 7, 13, 21, 99, 123, 256, 512, 1024, 2048][:args.seeds]

    print(f"A/B korekty kwartalnej — cache {args.tag}, {len(seeds)} ziaren, "
          f"{args.epochs} epok, backbone ZAMROZONY")
    print(f"  n train/val/test = {(splits == 'train').sum()}/{(splits == 'val').sum()}"
          f"/{(splits == 'test').sum()}  dim={X.shape[1]}")
    print(f"  cache identity: {meta}")

    if args.lr is not None:
        pilot = {"chosen_lr": float(args.lr), "note": "podane z CLI, bez pilota"}
    else:
        print("\nPilot learning rate (jedno ziarno, wybor po SREDNIEJ z obu ramion):")
        pilot = pilot_lr(X, index, splits, cfg, device, recorded, ids,
                         [float(x) for x in args.lr_grid.split(",")], args.epochs)
    lr = pilot["chosen_lr"]
    print(f"  -> lr = {lr} dla OBU ramion\n")

    arms: dict[str, list[dict]] = {}
    t0 = time.time()
    for arm, quarter in (("quarter_OFF", False), ("quarter_ON", True)):
        y = targets_for_arm(index, quarter)
        rebase_ids = ids if quarter else None
        runs = []
        for seed in seeds:
            r = train_one(X, y, splits, cfg, seed, args.epochs, device, recorded,
                          rebase_ids, lr=lr)
            runs.append(r)
            print(f"  [{arm}] seed {seed:>4}  e{r['best_epoch']:>2}  "
                  f"val {100 * r['val_exact']:.2f}%  test {100 * r['test']['Exact']:.2f}%  "
                  f"MAE {r['test']['MAE']:.4f}", flush=True)
        arms[arm] = runs
    print(f"  ({(time.time() - t0) / 60:.1f} min)")

    # Paired across seeds: same seed means the same initialisation in both arms, so the
    # per-seed difference removes init variance and the test is paired.
    d_exact = np.array([on["test"]["Exact"] - off["test"]["Exact"]
                        for on, off in zip(arms["quarter_ON"], arms["quarter_OFF"])])
    d_mae = np.array([on["test"]["MAE"] - off["test"]["MAE"]
                      for on, off in zip(arms["quarter_ON"], arms["quarter_OFF"])])
    paired = {
        "delta_exact_mean": float(d_exact.mean()),
        "delta_exact_sd": float(d_exact.std(ddof=1)) if len(d_exact) > 1 else 0.0,
        "delta_exact_per_seed": [float(x) for x in d_exact],
        "delta_mae_mean": float(d_mae.mean()),
        "wins_for_ON": int((d_exact > 0).sum()), "n_seeds": len(d_exact),
        "p_two_sided": (float(stats.ttest_rel(
            [r["test"]["Exact"] for r in arms["quarter_ON"]],
            [r["test"]["Exact"] for r in arms["quarter_OFF"]]).pvalue)
            if len(d_exact) > 1 else None),
    }

    rows = [{"arm": a, **{k: v for k, v in r.items() if k not in ("val", "test", "test_pred")},
             **{f"test_{k}": v for k, v in r["test"].items()},
             **{f"val_{k}": v for k, v in r["val"].items()}}
            for a, runs in arms.items() for r in runs]
    pd.DataFrame(rows).to_csv(out_dir / "per_seed.csv", index=False)

    # Per-age, on the seed-majority prediction of each arm (a stable per-arm view).
    per_age_rows = []
    test_mask = splits == "test"
    for arm, runs in arms.items():
        stacked = np.stack([r["test_pred"] for r in runs])
        majority = stats.mode(stacked, axis=0, keepdims=False).mode
        for age, m in sorted(compute_per_age_metrics(recorded[test_mask], majority).items()):
            per_age_rows.append({"arm": arm, "age": age, **m})
    pd.DataFrame(per_age_rows).to_csv(out_dir / "per_age.csv", index=False)

    payload = {
        "question": "does data.quarter_age_adjustment_enabled help, holding all else fixed",
        "regime": "frozen backbone, head-only, cached CLS features",
        "cache": meta, "seeds": seeds, "epochs": args.epochs,
        "scoring": "both arms scored against RECORDED age; ON arm rebased by +1 for Q1",
        "lr_pilot": pilot,
        "still_improving_runs": {a: [r["seed"] for r in runs if r["still_improving"]]
                                 for a, runs in arms.items()},
        "arms": {a: summarise(runs) for a, runs in arms.items()},
        "arms_val": {a: summarise(runs, "val") for a, runs in arms.items()},
        "paired": paired,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    write_notes(out_dir, payload)
    stuck = {a: v for a, v in payload["still_improving_runs"].items() if v}
    if stuck:
        print(f"\nUWAGA: val nadal sie poprawial na koncu budzetu w: {stuck} — "
              f"podnies --epochs, inaczej porownanie mierzy budzet, nie cel.")
    print(f"\ndelta exact = {100 * paired['delta_exact_mean']:+.2f} pp "
          f"+- {100 * paired['delta_exact_sd']:.2f} (sd), "
          f"ON wins {paired['wins_for_ON']}/{paired['n_seeds']} seeds, "
          f"p={paired['p_two_sided']}")
    print(f"wrote {out_dir}")


def write_notes(out_dir: Path, payload: dict) -> None:
    def pct(x): return f"{100 * x:.2f} %"
    L = ["# A/B korekty kwartalnej — glowica na zamrozonych cechach", "",
         f"Rezim: {payload['regime']}.  Ziarna: {payload['seeds']}.  "
         f"Epok: {payload['epochs']}.  lr: {payload['lr_pilot']['chosen_lr']} "
         f"(wybrany na val, ten sam w obu ramionach).", "",
         f"Punktacja: {payload['scoring']}.", "",
         "Korekta kwartalna jest **wylacznie przeetykietowaniem celu**, wiec na cache'u cech",
         "wszystkie inne zmienne sa trzymane dokladnie stale: te same cechy bit w bit, ta sama",
         "architektura, ta sama inicjalizacja przy tym samym ziarnie. Porownanie jest **parowane**.",
         "", "## Wynik (zbior testowy, wiek zapisany)", "",
         "| ramie | exact | +-1 rok | MAE | macro exact | bias |",
         "|---|---:|---:|---:|---:|---:|"]
    for arm in ("quarter_OFF", "quarter_ON"):
        a = payload["arms"][arm]
        L.append(f"| {arm} | **{pct(a['Exact']['mean'])}** ± {100 * a['Exact']['sd']:.2f} "
                 f"| {pct(a['Acc1yr']['mean'])} | {a['MAE']['mean']:.4f} ± {a['MAE']['sd']:.4f} "
                 f"| {pct(a['MacroExact']['mean'])} | {a['Bias']['mean']:+.3f} |")
    p = payload["paired"]
    L += ["", "## Test parowany po ziarnach", "",
          f"- delta exact = **{100 * p['delta_exact_mean']:+.2f} pkt proc.** "
          f"(sd {100 * p['delta_exact_sd']:.2f})",
          f"- delta MAE = **{p['delta_mae_mean']:+.4f}**",
          f"- ON wygrywa w **{p['wins_for_ON']}/{p['n_seeds']}** ziarnach",
          f"- p (dwustronne, parowane) = {p['p_two_sided']}", "",
          "Per ziarno: " + ", ".join(f"{100 * d:+.2f}" for d in p["delta_exact_per_seed"]), "",
          "## Kontrola zbieznosci", "",
          f"Biegi, w ktorych val nadal sie poprawial na koncu budzetu: "
          f"`{payload['still_improving_runs']}`. Pusto = budzet epok wystarczyl; jesli nie,",
          "porownanie mierzy budzet, nie cel, i trzeba podniesc `--epochs`.", "",
          "## Czego to NIE rozstrzyga", "",
          "Backbone jest zamrozony. Produkcja odmraza go w epoce 6, a backbone majacy swobode",
          "dostosowania sie do zmienionego celu moze zyskac wiecej niz sama glowica. To jest",
          "czysta dolna granica z porzadna statystyka po ziarnach, nie zamiennik pelnego biegu.", ""]
    (out_dir / "notes.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
