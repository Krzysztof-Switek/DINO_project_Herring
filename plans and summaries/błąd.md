/home/kswitek/Documents/DINO_project_Herring/.venv/bin/python /home/kswitek/Documents/DINO_project_Herring/main_hires966.py 
[main_hires966] MODE=demo  LOCATION=server  IMAGE_DIR=/home/kswitek/Documents/Photo/Otolithes/HER/Processed
[main_hires966] BASE_CONFIG=/home/kswitek/Documents/DINO_project_Herring/configs/config_demo_hires966.yaml
[main_hires966] OUTPUT_DIR=/home/kswitek/Documents/DINO_project_Herring/outputs/data/demo_hires966
[fresh] Czyszczę katalog runu /home/kswitek/Documents/DINO_project_Herring/outputs/data/demo_hires966 — pełny bieg od zera
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
[2026-07-23 11:14:43] epoch=  1  train_loss=0.4999  val_loss=0.4650  val_mae=1.750  lr=1.00e-04  coral_loss=0.1116  mil_loss=0.3534  mil_active=1337.0000  mean_age=4.5000
[2026-07-23 11:14:44] Training complete
  Best checkpoint: /home/kswitek/Documents/DINO_project_Herring/outputs/data/demo_hires966/checkpoints/embedded/best.pt

[3/9] TRAIN NotEmbedded — pominięty (--embedded-only)

[4/9] INFER — emb_on_emb
Using cache found in /home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main
  Inferencja (wszystkie próbki)...
Inference complete: 4 samples
  MAE  mean=2.250  median=2.500
  Interpretacja dla 4 próbek (3 najlepszych + 3 najgorszych)
  Heatmapy i nakładki (oryginalna rozdzielczość)...
  Predictions: /home/kswitek/Documents/DINO_project_Herring/outputs/data/demo_hires966/emb_on_emb/predictions.csv

[8/9] CARDS — karty rozumowania
Using cache found in /home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main
    [cards] walkthrough zbudowany dla 2023_BITS1q_HER_UsteckoLebskie_Embedded_Sharpest_FishIndex28_Single2_Left.jpg (wiek 4)
    [cards] emb_on_emb: gridy 4/6 (brak obrazu: 0, segmentacja nieudana: 0)
  Localization quality: /home/kswitek/Documents/DINO_project_Herring/outputs/data/demo_hires966/localization_quality.json

[9/9] REPORT — raport porównawczy
  Report: /home/kswitek/Documents/DINO_project_Herring/outputs/data/demo_hires966/comparison_report.html
  Pipeline summary: /home/kswitek/Documents/DINO_project_Herring/outputs/data/demo_hires966/pipeline_summary.json

=== Pipeline zakończony ===
Raport:          /home/kswitek/Documents/DINO_project_Herring/outputs/data/demo_hires966/comparison_report.html
Pipeline summary: /home/kswitek/Documents/DINO_project_Herring/outputs/data/demo_hires966/pipeline_summary.json

Process finished with exit code 0