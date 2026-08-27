"""24.08 -- Frozen-backbone + small localization head, trained ONLY on the 42 ZEGAR images with
real (x, y) ring positions ("Podejście B" / Część B option 2, plans and summaries/
12.08_ZEGAR_ANOTACJE_TO_DO.md §4.2, precedent: arXiv:2604.16758, frozen ViT + lightweight
CenterNet head, 48 annotated images, 3-fold CV).

Context (full write-up: plans and summaries/24.08_ROI_oraz_małe_głowice.md): four independent
ablations of the weakly-supervised `radial_attention` density head (window_deg, Zmiana B/E9, loss
weights, radial_bins/Fourier harmonics -- outputs/DINO_proces.md section 7) all failed to move
localization past ~22-28px (post 24.08 GT fix) on real ground truth. This is a DELIBERATE,
CONSCIOUS departure from the project's standing "zero position annotations in training" principle
(see memory `weak_supervision_attention_first`) -- not a replacement for it, an additional,
separate measurement of what's achievable WITH real position supervision on this same backbone,
at the one scale of data (42 images) where we actually have it.

Design, in one sentence: freeze the EXISTING, already-trained DINOv2 backbone (Run N's checkpoint
-- backbone quality is provably independent of which density head was ever attached to it, see
outputs/DINO_proces.md's repeated stop-gradient-isolation findings), extract its patch tokens for
the 42 ZEGAR images ONCE, and train nothing but a tiny (~25K-param) per-patch MLP head on top,
directly regressing a CenterNet-style Gaussian heatmap target built from the REAL (x, y) ring
positions (after the 24.08 GT fix: core/edge/last-ring excluded -- see
scripts/diagnostics/expert_annotation_eval.py::load_expert_annotations).

Why frozen + tiny head, not fine-tuning the backbone itself: 42 images is far too small to
fine-tune a ViT without destroying its general features (this project's own resolution-ladder
run, Etap 12/23.07_hires966, showed real regressions from touching the backbone's input regime
even with ~13,000 training images) -- freezing keeps 99.99%+ of parameters fixed, only the tiny
head can overfit, and the literature precedent (arXiv:2604.16758) validates this EXACT regime
(frozen large ViT + lightweight CenterNet head, ~48 images, cross-validated).

Reuses (imports, does not duplicate) scripts/diagnostics/expert_annotation_eval.py's own
annotation loading / crop-disambiguation / axis-projection machinery, for two reasons: (1) it is
already correct and tested against these exact 42 images, (2) it makes this experiment's held-out
`model_vs_gt_axis_px` number DIRECTLY comparable to every weakly-supervised checkpoint's ZEGAR
score (Run N 27.88px etc., outputs/24.08_zegar_lastring_excluded/) -- same metric, same GT, same
matching algorithm, only the localization method differs.

Runs entirely on CPU, no GPU/server required: the backbone forward pass happens ONCE per (image,
augmentation) pair up front (168 forward passes total: 42 images x 4 fixed flip augmentations),
cached in memory; every training epoch after that touches only the tiny head on cached tensors,
so even hundreds of epochs across 3 folds finish in well under a minute of head-only compute.
`data/ZEGAR/Zdjęcia/` is local repo data (not on Z:), so this needs no network-share access either.

Usage:
    python scripts/diagnostics/train_zegar_localization_head.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DIAG_DIR = PROJECT_ROOT / "scripts" / "diagnostics"
if str(DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(DIAG_DIR))

import json
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image as PILImage
from scipy import ndimage as ndi
from torchvision import transforms

import expert_annotation_eval as ev
from src.otolith_axis import detect_axis, apply_background_mask
from src.inference import load_model_from_checkpoint
from src.model import EMBED_DIMS
from scripts.run_pipeline import load_merged_config

# ============================================================
# KONFIGURACJA -- zmień tylko tutaj
# ============================================================

# Frozen backbone source -- Run N (standing reference checkpoint for the whole radial_attention
# series, outputs/DINO_proces.md section 7/8). Only the backbone is used; whatever density head
# this checkpoint carries is loaded but never called -- backbone quality has been repeatedly shown
# (stop-gradient isolation) to be independent of which density head was ever attached.
REFERENCE_CKPT = PROJECT_ROOT / "outputs" / "11.08_radial_attention" / "checkpoints" / "embedded" / "best.pt"
REFERENCE_CONFIG = PROJECT_ROOT / "configs" / "config_radial_attention.yaml"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "24.08_zegar_localization_head"

IMAGE_SIZE = 518          # matches production OtolithDataset resize target
PATCH_SIZE = 14
HP = WP = IMAGE_SIZE // PATCH_SIZE   # 37x37 patch grid, dinov2_vits14(_reg)

N_FOLDS = 3               # matches arXiv:2604.16758's own validation protocol
SEED = 42                 # matches this project's standing seed convention
GAUSSIAN_SIGMA_PATCHES = 1.2   # CenterNet-style soft target radius, in PATCH units
HIDDEN_DIM = 64
DROPOUT = 0.1
EPOCHS = 300
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-4
FOCAL_ALPHA = 2.0          # CornerNet/CenterNet penalty-reduced focal loss, standard values
FOCAL_BETA = 4.0
DEVICE = "cpu"              # frozen backbone + ~25K-param head -- CPU is plenty, no GPU needed

# ============================================================

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_NORMALIZE = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)])

# 4 fixed augmentations: (flip_h, flip_v). Precomputed once per image -- see module docstring for
# why this is safe (frozen, deterministic backbone) and why it's NOT done per-epoch (cost).
_AUGMENTATIONS = [(False, False), (True, False), (False, True), (True, True)]


# ---------------------------------------------------------------------------
# Step 1 -- tiny model: per-patch MLP -> sigmoid heatmap (the ONLY trainable part)
# ---------------------------------------------------------------------------

class ZegarLocalizationHead(nn.Module):
    """Per-patch MLP -> sigmoid heatmap, CenterNet-style. Operates on FROZEN backbone patch
    tokens only. ~25K params for embed_dim=384/hidden_dim=64 -- deliberately tiny, matching the
    "3.8M trainable params on 308M frozen" ratio of the arXiv:2604.16758 precedent (this project's
    backbone is smaller, vits14, so the head is scaled down proportionally, not copied verbatim).
    """

    def __init__(self, embed_dim: int, hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """(B, Hp, Wp, D) -> (B, Hp, Wp) sigmoid heatmap."""
        logits = self.net(patch_tokens).squeeze(-1)
        return torch.sigmoid(logits)


def centernet_focal_loss(
    pred: torch.Tensor, target: torch.Tensor, alpha: float = FOCAL_ALPHA, beta: float = FOCAL_BETA,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Standard CornerNet/CenterNet penalty-reduced pixel-wise focal loss. ``target`` is the soft
    Gaussian heatmap (peak=1.0); pixels near a peak (target close to but not exactly 1) get a
    reduced negative penalty via ``(1-target)**beta``, so the loss doesn't punish the model for
    predicting high values immediately adjacent to a true peak -- necessary on our coarse 37x37
    grid where a Gaussian's near-peak neighbourhood is a real, intended part of the target, not
    background."""
    pred = pred.clamp(eps, 1.0 - eps)
    pos_mask = (target >= 0.999).float()
    neg_mask = 1.0 - pos_mask
    neg_weight = (1.0 - target).pow(beta)
    pos_loss = -(1 - pred).pow(alpha) * torch.log(pred) * pos_mask
    neg_loss = -neg_weight * pred.pow(alpha) * torch.log(1 - pred) * neg_mask
    num_pos = pos_mask.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def gaussian_heatmap(peaks_rc: list[tuple[float, float]], hp: int, wp: int, sigma: float) -> np.ndarray:
    """Max-of-Gaussians target heatmap (CenterNet convention: overlapping peaks take the max, not
    the sum, so two close rings don't create an artificially inflated combined peak)."""
    heat = np.zeros((hp, wp), dtype=np.float32)
    if not peaks_rc:
        return heat
    rr, cc = np.meshgrid(np.arange(hp), np.arange(wp), indexing="ij")
    for r0, c0 in peaks_rc:
        g = np.exp(-((rr - r0) ** 2 + (cc - c0) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        heat = np.maximum(heat, g)
    return heat


def decode_topk_peaks(heat: np.ndarray, k: int, min_dist_patches: float = 1.5) -> list[tuple[int, int]]:
    """Greedy top-k local maxima with minimum-separation NMS, count fixed to the sample's own
    known GT ring count (k). Threshold-free by construction.

    Replaces an earlier fixed-threshold decode (PEAK_THRESHOLD=0.3), removed 26.08. Diagnosed:
    this head's actual held-out sigmoid outputs never exceed ~0.01-0.15, so the fixed threshold
    returned ZERO peaks for 41/42 held-out evaluations in the original 24.08 run --
    evaluate_sample() silently treated each as "no match" (empty t_model, not a crash), and the
    run's own aggregate (outputs/24.08_zegar_localization_head/metrics.json) ended up scoring
    only 1/42 samples without anyone noticing at the time."""
    if k <= 0:
        return []
    local_max = ndi.maximum_filter(heat, size=3) == heat
    cand_r, cand_c = np.nonzero(local_max)
    cand_v = heat[cand_r, cand_c]
    order = np.argsort(-cand_v)
    chosen: list[tuple[int, int]] = []
    for idx in order:
        r, c = int(cand_r[idx]), int(cand_c[idx])
        if all((r - rr) ** 2 + (c - cc) ** 2 >= min_dist_patches ** 2 for rr, cc in chosen):
            chosen.append((r, c))
        if len(chosen) >= k:
            break
    return chosen


# ---------------------------------------------------------------------------
# Step 2 -- per-sample feature/target extraction (backbone runs ONCE here, never again)
# ---------------------------------------------------------------------------

def build_dataset(model, ann: pd.DataFrame, seg_params: dict) -> dict:
    """Returns {sample: {"axis_info", "crop_wh", "tokens": {aug: (Hp,Wp,D) tensor},
    "heatmaps": {aug: (Hp,Wp) np.ndarray}}} for every sample where crop+segmentation succeed.
    """
    samples = sorted(ann["Sample"].unique())
    data: dict = {}
    n_failed = 0
    for i, sample in enumerate(samples, 1):
        image_id = f"{sample}.jpg"
        sub = ann[ann.Sample == sample]
        mean_xy = (float(sub.x.mean()), float(sub.y.mean()))
        raw_rgb = np.array(PILImage.open(ev.IMAGE_DIR / image_id).convert("RGB"), dtype=np.uint8)
        cropped, x0, y0, _used_second = ev.resolve_and_crop_target_otolith(raw_rgb, mean_xy, seg_params)
        if cropped is None:
            print(f"  [{i}/{len(samples)}] {sample}: crop failed, skipping")
            n_failed += 1
            continue
        axis_info = detect_axis(cropped, seg_params)
        if axis_info is None:
            print(f"  [{i}/{len(samples)}] {sample}: axis detection failed, skipping")
            n_failed += 1
            continue
        masked = apply_background_mask(cropped, axis_info["mask"])
        crop_h, crop_w = masked.shape[:2]
        resized = PILImage.fromarray(masked).resize((IMAGE_SIZE, IMAGE_SIZE), PILImage.BILINEAR)
        scale_x = IMAGE_SIZE / crop_w
        scale_y = IMAGE_SIZE / crop_h

        # Rebase this sample's annotations into crop-local, then resized-pixel space.
        pts_resized = []
        for _, row in sub.iterrows():
            xl, yl = row.x - x0, row.y - y0
            pts_resized.append((xl * scale_x, yl * scale_y))

        tokens_by_aug: dict[tuple[bool, bool], torch.Tensor] = {}
        heat_by_aug: dict[tuple[bool, bool], np.ndarray] = {}
        base_tensor = _NORMALIZE(resized)   # (3, IMAGE_SIZE, IMAGE_SIZE)
        for flip_h, flip_v in _AUGMENTATIONS:
            t = base_tensor
            pts = pts_resized
            if flip_h:
                t = torch.flip(t, dims=[-1])
                pts = [((IMAGE_SIZE - 1) - x, y) for x, y in pts]
            if flip_v:
                t = torch.flip(t, dims=[-2])
                pts = [(x, (IMAGE_SIZE - 1) - y) for x, y in pts]
            with torch.no_grad():
                tokens = model.get_patch_tokens(t.unsqueeze(0).to(DEVICE))[0].cpu()   # (Hp,Wp,D)
            peaks_rc = [(y / PATCH_SIZE, x / PATCH_SIZE) for x, y in pts]
            heat = gaussian_heatmap(peaks_rc, HP, WP, GAUSSIAN_SIGMA_PATCHES)
            tokens_by_aug[(flip_h, flip_v)] = tokens
            heat_by_aug[(flip_h, flip_v)] = heat

        data[sample] = {
            "axis_info": axis_info, "x0": x0, "y0": y0,
            "scale_x": scale_x, "scale_y": scale_y,
            "tokens": tokens_by_aug, "heatmaps": heat_by_aug,
        }
        print(f"  [{i}/{len(samples)}] {sample}: OK ({len(pts_resized)} points)")
    print(f"Built {len(data)}/{len(samples)} samples ({n_failed} failed)")
    return data


# ---------------------------------------------------------------------------
# Step 3 -- held-out evaluation, DIRECTLY comparable to expert_annotation_eval.py's own metric
# ---------------------------------------------------------------------------

def evaluate_sample(head: ZegarLocalizationHead, sample_data: dict, ann: pd.DataFrame, sample: str,
                     max_gap_t: float) -> Optional[dict]:
    axis_info = sample_data["axis_info"]

    # GT first -- decode_topk_peaks needs k (the sample's own true ring count) as its argument;
    # see decode_peaks's docstring for why a fixed absolute threshold doesn't work for this head.
    sub = ann[ann.Sample == sample]
    t_kk = sub[sub.annotator == "KK"].apply(lambda r: ev.point_to_axis_t(r.x, r.y, axis_info), axis=1).tolist()
    t_ss = sub[sub.annotator == "SS"].apply(lambda r: ev.point_to_axis_t(r.x, r.y, axis_info), axis=1).tolist()
    pairs, _u_kk, _u_ss = ev.match_t_values(t_kk, t_ss, max_gap_t)
    gt_t = [(t_kk[i] + t_ss[j]) / 2.0 for i, j in pairs]
    if not gt_t:
        return None

    tokens = sample_data["tokens"][(False, False)].unsqueeze(0)   # identity view only, for eval
    with torch.no_grad():
        heat = head(tokens)[0].numpy()
    peaks_rc = decode_topk_peaks(heat, len(gt_t))

    scale_x, scale_y = sample_data["scale_x"], sample_data["scale_y"]
    t_model = []
    for r, c in peaks_rc:
        x_resized = (c + 0.5) * PATCH_SIZE
        y_resized = (r + 0.5) * PATCH_SIZE
        x_crop = x_resized / scale_x
        y_crop = y_resized / scale_y
        t_model.append(ev.point_to_axis_t(x_crop, y_crop, axis_info))
    if not t_model:
        return None
    length = axis_info["length_px"]
    d = np.abs(np.asarray(t_model)[:, None] - np.asarray(gt_t)[None, :]) * length
    return {
        "sample": sample, "n_model_peaks": len(t_model), "n_gt": len(gt_t),
        "model_vs_gt_axis_px": float(d.min(axis=1).mean()),
    }


# ---------------------------------------------------------------------------
# Step 4 -- k-fold training + evaluation
# ---------------------------------------------------------------------------

def make_folds(samples: list[str], n_folds: int, seed: int) -> list[list[str]]:
    rng = np.random.RandomState(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    return [shuffled[i::n_folds] for i in range(n_folds)]


def random_control_baseline(dataset: dict, ann: pd.DataFrame, samples: list[str], max_gap_t: float,
                             seed: int = 0) -> Optional[float]:
    """'No signal' floor: k uniformly-random axis-t guesses per sample (k = that sample's own true
    ring count), scored with the identical metric. Added 26.08 after this head's trained
    predictions scored WORSE than this floor at every (decode/regularization/architecture)
    configuration tried -- see plans and summaries/26.08_zegar_localization_head_negative_result.md.
    Without this control, a bad-but-nonzero px number is easy to misread as "some signal"."""
    rng = np.random.RandomState(seed)
    dists = []
    for s in samples:
        axis_info = dataset[s]["axis_info"]
        sub = ann[ann.Sample == s]
        t_kk = sub[sub.annotator == "KK"].apply(lambda r: ev.point_to_axis_t(r.x, r.y, axis_info), axis=1).tolist()
        t_ss = sub[sub.annotator == "SS"].apply(lambda r: ev.point_to_axis_t(r.x, r.y, axis_info), axis=1).tolist()
        pairs, _u_kk, _u_ss = ev.match_t_values(t_kk, t_ss, max_gap_t)
        gt_t = [(t_kk[i] + t_ss[j]) / 2.0 for i, j in pairs]
        if not gt_t:
            continue
        t_model = rng.uniform(0.0, 1.0, size=len(gt_t)).tolist()
        length = axis_info["length_px"]
        d = np.abs(np.asarray(t_model)[:, None] - np.asarray(gt_t)[None, :]) * length
        dists.append(float(d.min(axis=1).mean()))
    return float(np.mean(dists)) if dists else None


def train_one_fold(dataset: dict, train_samples: list[str], embed_dim: int) -> ZegarLocalizationHead:
    head = ZegarLocalizationHead(embed_dim).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    examples = []
    for s in train_samples:
        for aug, tokens in dataset[s]["tokens"].items():
            examples.append((tokens, torch.from_numpy(dataset[s]["heatmaps"][aug])))

    head.train()
    for epoch in range(EPOCHS):
        perm = np.random.permutation(len(examples))
        total_loss = 0.0
        for idx in perm:
            tokens, target = examples[idx]
            pred = head(tokens.unsqueeze(0).to(DEVICE))[0]
            loss = centernet_focal_loss(pred, target.to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch + 1}/{EPOCHS}  mean_loss={total_loss / len(examples):.4f}")
    head.eval()
    return head


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print("ZEGAR localization head -- frozen backbone, real-position supervision, 42 images only")
    print("=" * 70)

    print("\n[1/5] Wczytywanie adnotacji (bez core/edge/ostatniego przyrostu, 24.08 fix)...")
    ann = ev.load_expert_annotations()
    ev.validate_annotation_bounds(ann, ev.IMAGE_DIR)

    print("\n[2/5] Wczytanie zamrożonego backbone'u (Run N)...")
    cfg = load_merged_config(REFERENCE_CONFIG, None)
    if not REFERENCE_CKPT.exists():
        sys.exit(f"Brak checkpointu: {REFERENCE_CKPT}")
    model = load_model_from_checkpoint(cfg, REFERENCE_CKPT)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    embed_dim = EMBED_DIMS.get(cfg.model.backbone, 384)
    seg_params = cfg.segmentation.as_params()
    max_gap_t = cfg.candidates.min_peak_distance / ev.N_SAMPLES_AXIS

    print("\n[3/5] Ekstrakcja cech (JEDNORAZOWA, 42 obrazy x 4 augmentacje = 168 forward passes)...")
    dataset = build_dataset(model, ann, seg_params)
    del model   # backbone not needed again -- everything below trains/evaluates on cached tokens

    samples = sorted(dataset.keys())
    folds = make_folds(samples, N_FOLDS, SEED)

    print(f"\n[4/5] Trening {N_FOLDS}-fold cross-walidacji ({EPOCHS} epok/fold, tylko głowica)...")
    all_results = []
    for fold_idx, held_out in enumerate(folds):
        train_samples = [s for s in samples if s not in held_out]
        print(f"\n  --- Fold {fold_idx + 1}/{N_FOLDS}: {len(train_samples)} trening, "
              f"{len(held_out)} held-out ({held_out}) ---")
        head = train_one_fold(dataset, train_samples, embed_dim)
        torch.save(head.state_dict(), OUTPUT_DIR / f"head_fold{fold_idx}.pt")
        for s in held_out:
            r = evaluate_sample(head, dataset[s], ann, s, max_gap_t)
            if r is not None:
                r["fold"] = fold_idx
                all_results.append(r)
                print(f"    {s}: model_vs_gt_axis_px={r['model_vs_gt_axis_px']:.2f}px "
                      f"(n_pred={r['n_model_peaks']}, n_gt={r['n_gt']})")

    print("\n[5/5] Wyniki zbiorcze (held-out, k-fold, PORÓWNYWALNE z checkpointami słabego nadzoru)...")
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "per_sample_results.csv", index=False)
    random_px = random_control_baseline(dataset, ann, samples, max_gap_t, seed=SEED)
    aggregate = {
        "n_samples_scored": int(len(df)),
        "n_folds": N_FOLDS,
        "model_vs_gt_axis_px_mean": float(df["model_vs_gt_axis_px"].mean()),
        "model_vs_gt_axis_px_median": float(df["model_vs_gt_axis_px"].median()),
        "model_vs_gt_axis_px_std": float(df["model_vs_gt_axis_px"].std()),
        "random_control_axis_px": random_px,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    rand_str = f"{random_px:.1f}px" if random_px is not None else "n/a"
    print(f"\nPorównanie: Run N (słaby nadzór, poprawiony GT) = 27.88px, klasyka = 122.92px, "
          f"podłoga szumu ludzkiego = 4.28px, kontrola losowa (ten skrypt) = {rand_str}.")
    print(f"Zapisano: {OUTPUT_DIR / 'metrics.json'}, {OUTPUT_DIR / 'per_sample_results.csv'}")


if __name__ == "__main__":
    main()
