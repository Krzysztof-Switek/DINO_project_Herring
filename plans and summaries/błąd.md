/home/kswitek/Documents/DINO_project_Herring/.venv/bin/python /home/kswitek/Documents/DINO_project_Herring/main_hires966.py 
[main_hires966] MODE=full  LOCATION=server  IMAGE_DIR=/home/kswitek/Documents/Photo/Otolithes/HER/Processed
[main_hires966] BASE_CONFIG=/home/kswitek/Documents/DINO_project_Herring/configs/config_hires966.yaml
[main_hires966] OUTPUT_DIR=/home/kswitek/Documents/DINO_project_Herring/outputs/data/23.07_hires966
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
[2026-07-23 11:22:02] Backbone frozen for first 5 epochs
[2026-07-23 12:42:35] epoch=  1  train_loss=180.9875  val_loss=3.2483  val_mae=1.578  lr=1.00e-04  coral_loss=0.1199  mil_loss=0.0091  mil_active=43.8913  density_loss=3.1193  density_active=0.0000  mean_age=3.7974
[2026-07-23 14:03:44] epoch=  2  train_loss=2.8685  val_loss=2.8837  val_mae=1.265  lr=9.99e-05  coral_loss=0.1057  mil_loss=0.0057  mil_active=25.0061  density_loss=2.7723  density_active=0.0000  mean_age=3.7974
[2026-07-23 15:26:01] epoch=  3  train_loss=2.6510  val_loss=2.7052  val_mae=1.180  lr=9.96e-05  coral_loss=0.0999  mil_loss=0.0046  mil_active=25.2426  density_loss=2.6006  density_active=0.0000  mean_age=3.7974
[2026-07-23 16:47:10] epoch=  4  train_loss=2.4684  val_loss=2.4764  val_mae=1.153  lr=9.91e-05  coral_loss=0.0974  mil_loss=0.0043  mil_active=20.3861  density_loss=2.3748  density_active=0.0000  mean_age=3.7974
[2026-07-23 18:08:57] epoch=  5  train_loss=2.2841  val_loss=2.2920  val_mae=1.150  lr=9.84e-05  coral_loss=0.0959  mil_loss=0.0039  mil_active=20.4322  density_loss=2.1922  density_active=0.0000  mean_age=3.7974
[2026-07-23 18:08:58] Backbone unfrozen at epoch 6
[2026-07-23 21:41:30] epoch=  6  train_loss=2.2851  val_loss=2.5032  val_mae=0.913  lr=9.76e-05  coral_loss=0.0824  mil_loss=0.0019  mil_active=13.1409  density_loss=2.4189  density_active=0.0000  mean_age=3.7974
[2026-07-24 01:15:51] epoch=  7  train_loss=1.8927  val_loss=1.8120  val_mae=0.926  lr=9.65e-05  coral_loss=0.0809  mil_loss=0.0019  mil_active=12.9774  density_loss=1.7292  density_active=0.0000  mean_age=3.7974
[2026-07-24 04:49:54] epoch=  8  train_loss=1.8332  val_loss=1.7316  val_mae=0.948  lr=9.52e-05  coral_loss=0.0817  mil_loss=0.0017  mil_active=10.1070  density_loss=1.6482  density_active=0.0000  mean_age=3.7974
[2026-07-24 08:24:31] epoch=  9  train_loss=1.7637  val_loss=1.7006  val_mae=0.883  lr=9.38e-05  coral_loss=0.0792  mil_loss=0.0020  mil_active=14.5330  density_loss=1.6193  density_active=0.0000  mean_age=3.7974
[2026-07-24 11:55:31] epoch= 10  train_loss=1.7355  val_loss=1.7158  val_mae=0.829  lr=9.22e-05  coral_loss=0.0789  mil_loss=0.0025  mil_active=16.4696  density_loss=1.6345  density_active=0.0000  mean_age=3.7974