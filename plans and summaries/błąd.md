(.venv) kswitek@labworks:~/Documents/DINO_project_Herring$ /home/kswitek/Documents/DINO_project_Herring/.venv/bin/python /home/kswitek/Documents/DINO_project_Herring/main_wedge_b.py 
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
[2026-09-22 12:04:18] Backbone frozen for first 5 epochs.multi_wycinek_k=1  data.strip_mask_background_loss=False  data.quarter_age_adjustment_enabled=Falseround=True  data.dual_branch_density=False  data.dual_branch_wedge=True 
[main_wedge_b] LOCATION=server  IMAGE_DIR=/home/kswitek/Documents/Photo/Otolithes/HER/Processed  RESCAN=False
[main_wedge_b] BASE_CONFIG=/home/kswitek/Documents/DINO_project_Herring/configs/config_wedge_b.yaml
[main_wedge_b] OUTPUT_DIR=/home/kswitek/Documents/DINO_project_Herring/outputs/data/22.09_wedge_b
[main_wedge_b] Karty raportu SA swiadome galezi wycinka/pasm od 22.09 (sekcja G2: sektor na zdjeciu, kanwy pasm, density, zdekodowane piki). Koszt: 5852 patchy/probke na galezi density, 4,27x wiecej niz Run N. Niezalezna ocena liczbowa: scripts/diagnostics/expert_annotation_eval_wedge.py.
[fresh] Czyszczę katalog runu /home/kswitek/Documents/DINO_project_Herring/outputs/data/22.09_wedge_b — pełny bieg od zera
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
[2026-09-22 15:04:12] RUN IDENTITY  model.backbone=dinov2_vits14_reg  model.use_density_head=True  model.density_head_type=radial_attention  data.mask_background=True  data.dual_branch_density=False  data.dual_branch_wedge=True  data.wedge_band_edges_t=[0.0, 0.6, 0.8, 0.9, 1.0]  data.multi_wycinek_k=1  data.strip_mask_background_loss=False  data.quarter_age_adjustment_enabled=False
[2026-09-22 15:04:12] Backbone frozen for first 5 epochs