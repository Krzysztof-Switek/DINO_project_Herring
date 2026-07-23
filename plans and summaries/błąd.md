/home/kswitek/Documents/DINO_project_Herring/.venv/bin/python /home/kswitek/Documents/DINO_project_Herring/main_hires966.py 
[main_hires966] MODE=demo  LOCATION=server  IMAGE_DIR=/home/kswitek/Documents/Photo/Otolithes/HER/Processed
[main_hires966] BASE_CONFIG=/home/kswitek/Documents/DINO_project_Herring/configs/config_demo_hires966.yaml
[main_hires966] OUTPUT_DIR=/home/kswitek/Documents/DINO_project_Herring/outputs/data/demo_hires966
============================================================
OtolithDino — pipeline Embedded vs NotEmbedded
============================================================

[1/9] SCAN — pominięty (używam istniejących data/labels_*.csv; --rescan wymusza skan)

[2/9] TRAIN — Embedded
Using cache found in /home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main
/home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
  warnings.warn("xFormers is not available (SwiGLU)")
/home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:33: UserWarning: xFormers is not available (Attention)
  warnings.warn("xFormers is not available (Attention)")
/home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:40: UserWarning: xFormers is not available (Block)
  warnings.warn("xFormers is not available (Block)")
[2026-07-23 10:23:46] Backbone frozen for first 5 epochs
[2026-07-23 10:40:14] epoch=  1  train_loss=0.1723  val_loss=0.1213  val_mae=1.438  lr=1.00e-04  coral_loss=0.1138  mil_loss=0.0076  mil_active=13.6617  mean_age=3.7974
[2026-07-23 10:56:43] epoch=  2  train_loss=0.1142  val_loss=0.1077  val_mae=1.185  lr=9.99e-05  coral_loss=0.1004  mil_loss=0.0074  mil_active=12.7896  mean_age=3.7974