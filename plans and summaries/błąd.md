/home/kswitek/Documents/DINO_project_Herring/.venv/bin/python /home/kswitek/Documents/DINO_project_Herring/main_density_lab.py 
[main_density_lab 08:28:08] LOCATION=server  IMAGE_DIR=/home/kswitek/Documents/Photo/Otolithes/HER/Processed  CKPT_B=/home/kswitek/Documents/DINO_project_Herring/outputs/data/22.09_wedge_b/checkpoints/embedded/best_age.pt
[main_density_lab 08:28:08] plan: cache ['raw']; ramiona raw ['A3', 'A4'] × ziarna [0, 1, 2]; kontrola [] na wedge_b_best_age; 20 epok, batch 16, cel quarter
[main_density_lab 08:28:13] CPU: 128 rdzeni, torch 2.12.0+cu130, wątki 64 (serwer nie ma GPU — wszystko na CPU)
[main_density_lab 08:28:13] $ /home/kswitek/Documents/DINO_project_Herring/.venv/bin/python -m pytest tests/test_density_head_lab.py tests/test_cache_wedge_band_tokens.py tests/test_stage4_trainer.py -q
........................................................................ [ 80%]
..................                                                       [100%]
90 passed in 32.73s
[main_density_lab 08:28:47] cache raw: kompletny — pomijam
[main_density_lab 08:28:47] $ /home/kswitek/Documents/DINO_project_Herring/.venv/bin/python scripts/diagnostics/density_head_lab.py --cache raw --arms A3 --seeds 0,1,2 --epochs 20 --stop-after-mature 3 --device cpu --batch-size 16 --label quarter --skip-done
INFO: ramiona A0–A3 (maska produkcyjna liczona blokami binów) to ~1,35 s na próbkę na 16 rdzeniach — ~2 h/epokę; serwer (128 rdzeni, bez GPU) szybciej.
LAB  cache=raw  label=quarter  device=cpu  arms=['A3']  seeds=[0, 1, 2]  epochs=20  N=5852  train=5150  val=1117
  A3 seed0: gotowe wcześniej — pomijam
  A3 seed1: gotowe wcześniej — pomijam
  A3 seed2: gotowe wcześniej — pomijam

     matured  seeds  median_epoch  final_zero_ratio
arm                                                
A3         3      3           1.0          0.317688
A5         5      5           1.0          0.269867
[main_density_lab 08:28:49] $ /home/kswitek/Documents/DINO_project_Herring/.venv/bin/python scripts/diagnostics/density_head_lab.py --cache raw --arms A4 --seeds 0,1,2 --epochs 20 --stop-after-mature 3 --device cpu --batch-size 16 --label quarter --skip-done
LAB  cache=raw  label=quarter  device=cpu  arms=['A4']  seeds=[0, 1, 2]  epochs=20  N=5852  train=5150  val=1117
  A4 seed0 e0  zero_ratio=0.677  active=0.00  max_logit=-7.29  prior_bias=-7.45
  A4 seed0 e1  train=3.3178  zero_ratio=0.632  active=1.60  max_logit=-0.18  median_logit=-15.03  Σp/age=2.00  (1948 s/ep)
  A4 seed0 e2  train=2.2160  zero_ratio=0.450  active=0.86  max_logit=-0.55  median_logit=-19.06  Σp/age=0.89  (1997 s/ep)
  A4 seed0 e3  train=2.0631  zero_ratio=0.443  active=2.69  max_logit=0.19  median_logit=-22.66  Σp/age=1.55  (1977 s/ep)
  A4 seed0 e4  train=2.0267  zero_ratio=0.394  active=2.67  max_logit=0.14  median_logit=-25.70  Σp/age=1.31  (1965 s/ep)