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
  A4 seed0 e5  train=1.8867  zero_ratio=0.375  active=2.20  max_logit=-0.07  median_logit=-27.19  Σp/age=1.10  (2012 s/ep)
  A4 seed0 e6  train=1.7984  zero_ratio=0.376  active=2.21  max_logit=-0.01  median_logit=-27.38  Σp/age=1.29  (1997 s/ep)
  A4 seed0: dojrzało w e3, stop po 3 epokach
  A4 seed1 e0  zero_ratio=0.635  active=0.00  max_logit=-6.87  prior_bias=-7.45
  A4 seed1 e1  train=2.2408  zero_ratio=0.365  active=5.02  max_logit=0.56  median_logit=-13.42  Σp/age=1.23  (1923 s/ep)
  A4 seed1 e2  train=1.6666  zero_ratio=0.326  active=5.09  max_logit=1.04  median_logit=-14.62  Σp/age=1.13  (1951 s/ep)
  A4 seed1 e3  train=1.4867  zero_ratio=0.309  active=5.21  max_logit=1.47  median_logit=-14.98  Σp/age=1.14  (1956 s/ep)
  A4 seed1 e4  train=1.3811  zero_ratio=0.302  active=5.52  max_logit=1.63  median_logit=-15.52  Σp/age=1.14  (1981 s/ep)
  A4 seed1: dojrzało w e1, stop po 3 epokach
  A4 seed2 e0  zero_ratio=0.764  active=0.00  max_logit=-6.48  prior_bias=-7.45
  A4 seed2 e1  train=2.7909  zero_ratio=0.474  active=2.49  max_logit=0.14  median_logit=-12.38  Σp/age=1.64  (2112 s/ep)
  A4 seed2 e2  train=2.1345  zero_ratio=0.389  active=2.16  max_logit=-0.11  median_logit=-14.46  Σp/age=1.09  (2018 s/ep)
  A4 seed2 e3  train=1.9381  zero_ratio=0.372  active=2.73  max_logit=0.18  median_logit=-15.93  Σp/age=1.17  (2006 s/ep)
  A4 seed2 e4  train=1.8643  zero_ratio=0.401  active=4.11  max_logit=0.76  median_logit=-19.92  Σp/age=1.59  (1985 s/ep)
  A4 seed2: dojrzało w e1, stop po 3 epokach

     matured  seeds  median_epoch  final_zero_ratio
arm                                                
A3         3      3           1.0          0.317688
A4         3      3           1.0          0.375501
A5         5      5           1.0          0.269867
[main_density_lab 16:16:59] raport: /home/kswitek/Documents/DINO_project_Herring/experiments/density_head_lab/WYNIKI.md  (11 ziaren)
[main_density_lab 16:16:59] raport: /home/kswitek/Documents/DINO_project_Herring/experiments/density_head_lab/WYNIKI.md  (11 ziaren)
[main_density_lab 16:16:59] KONIEC — wyniki w experiments/density_head_lab/WYNIKI.md