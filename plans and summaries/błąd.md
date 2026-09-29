/home/kswitek/Documents/DINO_project_Herring/.venv/bin/python /home/kswitek/Documents/DINO_project_Herring/main_density_lab.py 
[main_density_lab 12:50:11] LOCATION=server  IMAGE_DIR=/home/kswitek/Documents/Photo/Otolithes/HER/Processed  CKPT_B=/home/kswitek/Documents/DINO_project_Herring/outputs/data/22.09_wedge_b/checkpoints/embedded/best_age.pt
[main_density_lab 12:50:11] plan: cache ['raw']; ramiona raw ['A3'] × ziarna [0, 1, 2]; kontrola [] na wedge_b_best_age; 20 epok, batch 16, cel quarter
[main_density_lab 12:50:12] CPU: 128 rdzeni, torch 2.12.0+cu130, wątki 64 (serwer nie ma GPU — wszystko na CPU)
[main_density_lab 12:50:12] $ /home/kswitek/Documents/DINO_project_Herring/.venv/bin/python -m pytest tests/test_density_head_lab.py tests/test_cache_wedge_band_tokens.py tests/test_stage4_trainer.py -q
........................................................................ [ 80%]
..................                                                       [100%]
90 passed in 26.52s
[main_density_lab 12:50:41] $ /home/kswitek/Documents/DINO_project_Herring/.venv/bin/python scripts/diagnostics/cache_wedge_band_tokens.py --tag raw --workers 8
Using cache found in /home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main
CACHE IDENTITY  tag=raw  device=cpu  n=6267  patches=5852 [1067, 1548, 1305, 1932]  dim=384
  model.backbone = dinov2_vits14_reg
  data.patch_size = 14
  data.mask_background = True
  data.wedge_delta_theta_deg = 98.7
  data.wedge_band_edges_t = [0.0, 0.6, 0.8, 0.9, 1.0]
  data.wedge_band_n_angle_patches = [97, 129, 145, 161]
  data.wedge_band_n_radius_patches = [11, 12, 9, 12]
  backbone_from = raw
  checkpoint_sha = None
  checkpoint_epoch = None
  canvases = production _build_wedge_bands_tensors_and_polar, no flip
  dtype = float16
  do zrobienia: 6267 / 6267

[train] 5150 obrazów
  64/5150  2.01 img/s
  128/5150  2.03 img/s
  192/5150  2.03 img/s
  256/5150  2.02 img/s
  320/5150  2.05 img/s
  384/5150  2.06 img/s
  448/5150  2.07 img/s
  512/5150  2.06 img/s
  576/5150  2.06 img/s
  640/5150  2.05 img/s
  704/5150  2.05 img/s
  768/5150  2.05 img/s
  832/5150  2.04 img/s
  896/5150  2.05 img/s
  960/5150  2.05 img/s
  1024/5150  2.05 img/s
  1088/5150  2.06 img/s
  1152/5150  2.06 img/s


4288/5150  2.10 img/s
  4352/5150  2.10 img/s
  4416/5150  2.10 img/s
  4480/5150  2.11 img/s
  4544/5150  2.11 img/s
  4608/5150  2.11 img/s
  4672/5150  2.11 img/s
  4736/5150  2.11 img/s
  4800/5150  2.11 img/s
  4864/5150  2.11 img/s
  4928/5150  2.11 img/s
  4992/5150  2.11 img/s
  5056/5150  2.11 img/s
  5120/5150  2.11 img/s

[val] 1117 obrazów
  64/1117  2.12 img/s
  128/1117  2.11 img/s
  192/1117  2.13 img/s
  256/1117  2.15 img/s
  320/1117  2.15 img/s
  384/1117  2.17 img/s
  448/1117  2.15 img/s
  512/1117  2.15 img/s
  576/1117  2.15 img/s
  640/1117  2.16 img/s
  704/1117  2.18 img/s
  768/1117  2.19 img/s
  832/1117  2.21 img/s
  896/1117  2.23 img/s
  960/1117  2.24 img/s
  1024/1117  2.26 img/s
  1088/1117  2.27 img/s

zapisano /home/kswitek/Documents/DINO_project_Herring/data/wedge_band_tokens/raw  gotowe 6267/6267  w 48.9 min
[main_density_lab 13:39:40] $ /home/kswitek/Documents/DINO_project_Herring/.venv/bin/python scripts/diagnostics/density_head_lab.py --cache raw --arms A3 --seeds 0,1,2 --epochs 20 --stop-after-mature 3 --device cpu --batch-size 16 --label quarter --skip-done
UWAGA: ramiona z pełną maską 5852² na CPU to ~3 s na próbkę (forward+backward) — przeznaczone na serwer GPU.
LAB  cache=raw  label=quarter  device=cpu  arms=['A3']  seeds=[0, 1, 2]  epochs=20  N=5852  train=5150  val=1117
  A3 seed0 e0  zero_ratio=0.677  active=0.00  max_logit=-7.29  prior_bias=-7.45
