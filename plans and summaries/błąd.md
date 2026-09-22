/home/kswitek/Documents/DINO_project_Herring/.venv/bin/python /home/kswitek/Documents/DINO_project_Herring/main_wedge_b.py 
[main_wedge_b] LOCATION=server  IMAGE_DIR=/home/kswitek/Documents/Photo/Otolithes/HER/Processed  RESCAN=False
[main_wedge_b] BASE_CONFIG=/home/kswitek/Documents/DINO_project_Herring/configs/config_wedge_b.yaml
[main_wedge_b] OUTPUT_DIR=/home/kswitek/Documents/DINO_project_Herring/outputs/data/09.09_wedge_b
[main_wedge_b] UWAGA: karty report.html (density/kandydaci) NIE są świadome gałęzi wycinka/pasm — ufać tylko predictions.csv/pipeline_summary.json (wiek); lokalizacja wymaga scripts/diagnostics/expert_annotation_eval_wedge.py (wykrywa pasma automatycznie z configu). Koszt: 5852 patchy/próbkę na gałęzi density, 4,27x więcej niż Run N.
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
[2026-09-09 14:50:45] Backbone frozen for first 5 epochs
[2026-09-09 15:06:25] epoch=  1  train_loss=0.1729  val_loss=0.1191  val_mae=1.400  lr=1.00e-04  coral_loss=0.1116  mil_loss=0.0076  mil_active=13.5792  mean_age=3.8102
[2026-09-09 15:18:54] epoch=  2  train_loss=0.1137  val_loss=0.1042  val_mae=1.160  lr=9.99e-05  coral_loss=0.0971  mil_loss=0.0071  mil_active=13.3330  mean_age=3.8102
[2026-09-09 15:31:09] epoch=  3  train_loss=0.1063  val_loss=0.0998  val_mae=1.107  lr=9.96e-05  coral_loss=0.0929  mil_loss=0.0070  mil_active=12.8442  mean_age=3.8102
[2026-09-09 15:43:27] epoch=  4  train_loss=0.1027  val_loss=0.0975  val_mae=1.067  lr=9.91e-05  coral_loss=0.0908  mil_loss=0.0067  mil_active=12.6893  mean_age=3.8102
[2026-09-09 15:56:17] epoch=  5  train_loss=0.1013  val_loss=0.0958  val_mae=1.051  lr=9.84e-05  coral_loss=0.0893  mil_loss=0.0065  mil_active=12.2381  mean_age=3.8102
[2026-09-09 15:56:18] Backbone unfrozen at epoch 6
[2026-09-09 16:32:28] epoch=  6  train_loss=0.1104  val_loss=0.0870  val_mae=0.883  lr=9.76e-05  coral_loss=0.0819  mil_loss=0.0051  mil_active=9.4861  mean_age=3.8102
[2026-09-09 17:04:43] epoch=  7  train_loss=0.0897  val_loss=0.0940  val_mae=1.038  lr=9.65e-05  coral_loss=0.0876  mil_loss=0.0064  mil_active=13.7610  mean_age=3.8102
[2026-09-09 17:36:31] epoch=  8  train_loss=0.0888  val_loss=0.0913  val_mae=1.230  lr=9.52e-05  coral_loss=0.0869  mil_loss=0.0044  mil_active=9.8299  mean_age=3.8102
[2026-09-09 18:08:31] epoch=  9  train_loss=0.0870  val_loss=0.0816  val_mae=0.809  lr=9.38e-05  coral_loss=0.0778  mil_loss=0.0038  mil_active=8.5318  mean_age=3.8102
[2026-09-09 18:40:20] epoch= 10  train_loss=0.0843  val_loss=0.0808  val_mae=0.814  lr=9.22e-05  coral_loss=0.0766  mil_loss=0.0042  mil_active=9.8988  mean_age=3.8102
[2026-09-09 19:11:58] epoch= 11  train_loss=0.0841  val_loss=0.0855  val_mae=0.885  lr=9.05e-05  coral_loss=0.0807  mil_loss=0.0047  mil_active=8.1325  mean_age=3.8102
[2026-09-09 19:43:48] epoch= 12  train_loss=0.0830  val_loss=0.0907  val_mae=1.040  lr=8.85e-05  coral_loss=0.0854  mil_loss=0.0053  mil_active=11.8039  mean_age=3.8102
[2026-09-09 20:15:48] epoch= 13  train_loss=0.0807  val_loss=0.0812  val_mae=0.818  lr=8.64e-05  coral_loss=0.0767  mil_loss=0.0045  mil_active=10.2265  mean_age=3.8102
[2026-09-09 20:47:35] epoch= 14  train_loss=0.0833  val_loss=0.0899  val_mae=1.017  lr=8.42e-05  coral_loss=0.0861  mil_loss=0.0038  mil_active=8.9329  mean_age=3.8102
[2026-09-09 21:18:55] epoch= 15  train_loss=0.0793  val_loss=0.0806  val_mae=0.801  lr=8.19e-05  coral_loss=0.0767  mil_loss=0.0039  mil_active=9.6831  mean_age=3.8102
[2026-09-09 21:50:00] epoch= 16  train_loss=0.0782  val_loss=0.0829  val_mae=0.893  lr=7.94e-05  coral_loss=0.0794  mil_loss=0.0036  mil_active=8.2874  mean_age=3.8102
[2026-09-09 22:21:54] epoch= 17  train_loss=0.0774  val_loss=0.0808  val_mae=0.850  lr=7.68e-05  coral_loss=0.0770  mil_loss=0.0038  mil_active=8.9132  mean_age=3.8102
[2026-09-09 22:54:14] epoch= 18  train_loss=0.0771  val_loss=0.0923  val_mae=1.013  lr=7.41e-05  coral_loss=0.0872  mil_loss=0.0050  mil_active=7.8899  mean_age=3.8102
[2026-09-09 23:25:51] epoch= 19  train_loss=0.0786  val_loss=0.0837  val_mae=0.909  lr=7.13e-05  coral_loss=0.0797  mil_loss=0.0039  mil_active=8.0009  mean_age=3.8102
[2026-09-09 23:56:59] epoch= 20  train_loss=0.0748  val_loss=0.0797  val_mae=0.801  lr=6.84e-05  coral_loss=0.0754  mil_loss=0.0043  mil_active=7.2999  mean_age=3.8102
[2026-09-10 00:28:39] epoch= 21  train_loss=0.0744  val_loss=0.0783  val_mae=0.830  lr=6.55e-05  coral_loss=0.0749  mil_loss=0.0033  mil_active=8.3751  mean_age=3.8102
[2026-09-10 01:00:24] epoch= 22  train_loss=0.0734  val_loss=0.0799  val_mae=0.854  lr=6.24e-05  coral_loss=0.0760  mil_loss=0.0038  mil_active=9.1799  mean_age=3.8102
[2026-09-10 01:32:22] epoch= 23  train_loss=0.0746  val_loss=0.0818  val_mae=0.820  lr=5.94e-05  coral_loss=0.0779  mil_loss=0.0039  mil_active=7.8720  mean_age=3.8102
[2026-09-10 02:04:28] epoch= 24  train_loss=0.0730  val_loss=0.0798  val_mae=0.798  lr=5.63e-05  coral_loss=0.0759  mil_loss=0.0038  mil_active=8.9776  mean_age=3.8102
[2026-09-10 02:36:01] epoch= 25  train_loss=0.0729  val_loss=0.0795  val_mae=0.825  lr=5.31e-05  coral_loss=0.0746  mil_loss=0.0049  mil_active=7.3957  mean_age=3.8102
[2026-09-10 03:07:56] epoch= 26  train_loss=0.0712  val_loss=0.0800  val_mae=0.862  lr=5.00e-05  coral_loss=0.0762  mil_loss=0.0038  mil_active=9.2525  mean_age=3.8102
[2026-09-10 03:39:14] epoch= 27  train_loss=0.0699  val_loss=0.0796  val_mae=0.836  lr=4.69e-05  coral_loss=0.0749  mil_loss=0.0048  mil_active=6.9615  mean_age=3.8102
[2026-09-10 04:10:53] epoch= 28  train_loss=0.0695  val_loss=0.0780  val_mae=0.829  lr=4.37e-05  coral_loss=0.0741  mil_loss=0.0039  mil_active=7.5461  mean_age=3.8102
[2026-09-10 04:42:35] epoch= 29  train_loss=0.0679  val_loss=0.0798  val_mae=0.825  lr=4.06e-05  coral_loss=0.0746  mil_loss=0.0053  mil_active=6.6526  mean_age=3.8102
[2026-09-10 05:13:59] epoch= 30  train_loss=0.0670  val_loss=0.0766  val_mae=0.792  lr=3.76e-05  coral_loss=0.0728  mil_loss=0.0038  mil_active=7.3205  mean_age=3.8102
[2026-09-10 05:45:35] epoch= 31  train_loss=0.0654  val_loss=0.0783  val_mae=0.855  lr=3.45e-05  coral_loss=0.0740  mil_loss=0.0043  mil_active=7.1173  mean_age=3.8102
[2026-09-10 06:17:44] epoch= 32  train_loss=0.0644  val_loss=0.0776  val_mae=0.824  lr=3.16e-05  coral_loss=0.0734  mil_loss=0.0042  mil_active=7.1728  mean_age=3.8102
[2026-09-10 06:49:53] epoch= 33  train_loss=0.0636  val_loss=0.0776  val_mae=0.804  lr=2.87e-05  coral_loss=0.0731  mil_loss=0.0045  mil_active=6.8254  mean_age=3.8102
[2026-09-10 07:21:44] epoch= 34  train_loss=0.0630  val_loss=0.0793  val_mae=0.815  lr=2.59e-05  coral_loss=0.0747  mil_loss=0.0046  mil_active=6.7735  mean_age=3.8102
[2026-09-10 07:53:25] epoch= 35  train_loss=0.0626  val_loss=0.0779  val_mae=0.805  lr=2.32e-05  coral_loss=0.0735  mil_loss=0.0044  mil_active=7.7305  mean_age=3.8102
[2026-09-10 08:24:54] epoch= 36  train_loss=0.0609  val_loss=0.0793  val_mae=0.817  lr=2.06e-05  coral_loss=0.0736  mil_loss=0.0057  mil_active=6.4575  mean_age=3.8102
[2026-09-10 08:57:02] epoch= 37  train_loss=0.0603  val_loss=0.0822  val_mae=0.898  lr=1.81e-05  coral_loss=0.0766  mil_loss=0.0056  mil_active=6.3993  mean_age=3.8102
[2026-09-10 09:28:58] epoch= 38  train_loss=0.0594  val_loss=0.0815  val_mae=0.849  lr=1.58e-05  coral_loss=0.0750  mil_loss=0.0064  mil_active=6.3053  mean_age=3.8102
[2026-09-10 10:00:29] epoch= 39  train_loss=0.0587  val_loss=0.0788  val_mae=0.825  lr=1.36e-05  coral_loss=0.0742  mil_loss=0.0046  mil_active=6.8165  mean_age=3.8102
[2026-09-10 10:32:08] epoch= 40  train_loss=0.0584  val_loss=0.0789  val_mae=0.847  lr=1.15e-05  coral_loss=0.0741  mil_loss=0.0048  mil_active=6.8854  mean_age=3.8102
[2026-09-10 10:32:08] Early stopping — brak poprawy val_mae przez 10 epok (best=0.7923)
[2026-09-10 10:32:08] Training complete
  Best checkpoint: /home/kswitek/Documents/DINO_project_Herring/outputs/data/09.09_wedge_b/checkpoints/embedded/best.pt

[3/9] TRAIN NotEmbedded — pominięty (--embedded-only)

[4/9] INFER — emb_on_emb
Using cache found in /home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main
  Inferencja (wszystkie próbki)...
Inference complete: 1105 samples
  MAE  mean=0.792  median=1.000
  Interpretacja dla 20 próbek (10 najlepszych + 10 najgorszych)
  Heatmapy i nakładki (oryginalna rozdzielczość)...
  Predictions: /home/kswitek/Documents/DINO_project_Herring/outputs/data/09.09_wedge_b/emb_on_emb/predictions.csv

[8/9] CARDS — karty rozumowania
Using cache found in /home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main
    [cards] walkthrough zbudowany dla 2023_BITS1q_HER_ZatokaGdanska_Embedded_Sharpest_FishIndex30_Single1_Right.jpg (wiek 4)
    [cards] emb_on_emb: gridy 20/20 (brak obrazu: 0, segmentacja nieudana: 0)
  Localization quality: /home/kswitek/Documents/DINO_project_Herring/outputs/data/09.09_wedge_b/localization_quality.json

[9/9] REPORT — raport porównawczy
  Report: /home/kswitek/Documents/DINO_project_Herring/outputs/data/09.09_wedge_b/comparison_report.html
  Pipeline summary: /home/kswitek/Documents/DINO_project_Herring/outputs/data/09.09_wedge_b/pipeline_summary.json

=== Pipeline zakończony ===
Raport:          /home/kswitek/Documents/DINO_project_Herring/outputs/data/09.09_wedge_b/comparison_report.html
Pipeline summary: /home/kswitek/Documents/DINO_project_Herring/outputs/data/09.09_wedge_b/pipeline_summary.json

Process finished with exit code 0
