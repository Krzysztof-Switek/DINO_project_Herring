/home/kswitek/Documents/DINO_project_Herring/.venv/bin/python /home/kswitek/Documents/DINO_project_Herring/main_wedge_b.py 
[main_wedge_b] LOCATION=server  IMAGE_DIR=/home/kswitek/Documents/Photo/Otolithes/HER/Processed  RESCAN=False
[main_wedge_b] BASE_CONFIG=/home/kswitek/Documents/DINO_project_Herring/configs/config_wedge_b.yaml
[main_wedge_b] OUTPUT_DIR=/home/kswitek/Documents/DINO_project_Herring/outputs/data/22.09_wedge_b
[main_wedge_b] Karty raportu SA swiadome galezi wycinka/pasm od 22.09 (sekcja G2: sektor na zdjeciu, kanwy pasm, density, zdekodowane piki). Koszt: 5852 patchy/probke na galezi density, 4,27x wiecej niz Run N. Niezalezna ocena liczbowa: scripts/diagnostics/expert_annotation_eval_wedge.py.
============================================================
OtolithDino — pipeline Embedded vs NotEmbedded
============================================================

[1/9] SCAN — pominięty (używam istniejących data/labels_*.csv; --rescan wymusza skan)

[2/9] TRAIN — Embedded
Using cache found in /home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main
/home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:45: UserWarning: xFormers is disabled (SwiGLU)
  warnings.warn("xFormers is disabled (SwiGLU)")
/home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
  warnings.warn("xFormers is not available (SwiGLU)")
/home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:29: UserWarning: xFormers is disabled (Attention)
  warnings.warn("xFormers is disabled (Attention)")
/home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:33: UserWarning: xFormers is not available (Attention)
  warnings.warn("xFormers is not available (Attention)")
/home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:35: UserWarning: xFormers is disabled (Block)
  warnings.warn("xFormers is disabled (Block)")
/home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:40: UserWarning: xFormers is not available (Block)
  warnings.warn("xFormers is not available (Block)")
[2026-09-22 12:04:18] RUN IDENTITY  model.backbone=dinov2_vits14_reg  model.use_density_head=True  model.density_head_type=radial_attention  data.mask_background=True  data.dual_branch_density=False  data.dual_branch_wedge=True  data.wedge_band_edges_t=[0.0, 0.6, 0.8, 0.9, 1.0]  data.multi_wycinek_k=1  data.strip_mask_background_loss=False  data.quarter_age_adjustment_enabled=False
[2026-09-22 12:04:18] Backbone frozen for first 5 epochs

[2026-09-22 17:33:11] epoch=  1  train_loss=42.5955  val_loss=4.0634  val_mae=1.612  lr=1.00e-04  coral_loss=0.1204  mil_loss=0.0176  mil_active=31.3151  density_loss=3.9255  density_active=0.0000  mean_age=3.8102
[2026-09-22 22:21:29] epoch=  2  train_loss=2.9206  val_loss=2.5719  val_mae=1.320  lr=9.99e-05  coral_loss=0.1065  mil_loss=0.0120  mil_active=21.2211  density_loss=2.4534  density_active=0.0000  mean_age=3.8102
[2026-09-23 03:08:19] epoch=  3  train_loss=2.2696  val_loss=2.3751  val_mae=1.208  lr=9.96e-05  coral_loss=0.1007  mil_loss=0.0098  mil_active=17.2444  density_loss=2.2646  density_active=0.0000  mean_age=3.8102
[2026-09-23 07:54:57] epoch=  4  train_loss=2.0412  val_loss=2.2454  val_mae=1.191  lr=9.91e-05  coral_loss=0.0978  mil_loss=0.0085  mil_active=15.6338  density_loss=2.1391  density_active=0.0000  mean_age=3.8102
[2026-09-23 12:42:03] epoch=  5  train_loss=1.9276  val_loss=2.3910  val_mae=1.161  lr=9.84e-05  coral_loss=0.0958  mil_loss=0.0077  mil_active=14.6222  density_loss=2.2876  density_active=0.0000  mean_age=3.8102
[2026-09-23 12:42:03] Backbone unfrozen at epoch 6
[2026-09-23 17:43:45] epoch=  6  train_loss=2.0317  val_loss=2.0255  val_mae=0.902  lr=9.76e-05  coral_loss=0.0834  mil_loss=0.0054  mil_active=9.7601  density_loss=1.9366  density_active=0.0000  mean_age=3.8102
[2026-09-23 22:42:25] epoch=  7  train_loss=1.8998  val_loss=2.0780  val_mae=0.885  lr=9.65e-05  coral_loss=0.0827  mil_loss=0.0055  mil_active=8.7278  density_loss=1.9897  density_active=0.0000  mean_age=3.8102
[2026-09-24 03:42:32] epoch=  8  train_loss=1.8713  val_loss=1.9264  val_mae=0.827  lr=9.52e-05  coral_loss=0.0792  mil_loss=0.0045  mil_active=8.7798  density_loss=1.8428  density_active=0.0000  mean_age=3.8102
[2026-09-24 08:44:19] epoch=  9  train_loss=1.8717  val_loss=2.4676  val_mae=0.852  lr=9.38e-05  coral_loss=0.0783  mil_loss=0.0041  mil_active=8.7144  density_loss=2.3852  density_active=0.0000  mean_age=3.8102