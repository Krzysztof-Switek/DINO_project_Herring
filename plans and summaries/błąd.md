(.venv) kswitek@labworks:~/Documents/DINO_project_Herring$ D=/home/kswitek/Documents/DINO_project_Herring                                                                                                                                                                                   
  echo "--- postep epoki 1 (przetworzonych probek) ---"                                                                                                                                                                            
  ls -1 $D/data/wedge_bands_cache/band3_168x2254_98.7deg/ | grep -c '\.png$'                                                                                                                                                       
  echo "--- proces treningu (czas dzialania, %CPU) ---"                                                                                                                                                                            
  ps -eo pid,etime,%cpu,cmd | grep main_wedge_b | grep -v grep                                                                                                                                                                     
  echo "--- GPU ---"                                                                                                                                                                                                               
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv                                                                                                                                                                  
  echo "--- ostatnie linie logu ---"                                                                                                                                                                                               
  tail -n 3 $D/outputs/data/22.09_wedge_b/logs/embedded/train.log                                                                                                                                                                  
  echo "--- miejsce na dysku ---"                                                                                                                                                                                                  
  df -h $D | tail -1  
--- postep epoki 1 (przetworzonych probek) ---
3392
--- proces treningu (czas dzialania, %CPU) ---
2002677    03:11:53 1932 /home/kswitek/Documents/DINO_project_Herring/.venv/bin/python /home/kswitek/Documents/DINO_project_Herring/main_wedge_b.py
--- GPU ---
Command 'nvidia-smi' not found, but can be installed with:
apt install nvidia-utils-390         # version 390.157-0ubuntu0.22.04.2, or
apt install nvidia-utils-418-server  # version 418.226.00-0ubuntu5~0.22.04.1
apt install nvidia-utils-450-server  # version 450.248.02-0ubuntu0.22.04.1
apt install nvidia-utils-470-server  # version 470.256.02-0ubuntu0.22.04.1
apt install nvidia-utils-535         # version 535.309.01-0ubuntu0.22.04.1
apt install nvidia-utils-535-server  # version 535.309.01-0ubuntu0.22.04.1
apt install nvidia-utils-545         # version 545.29.06-0ubuntu0.22.04.2
apt install nvidia-utils-565-server  # version 565.57.01-0ubuntu0.22.04.4
apt install nvidia-utils-580         # version 580.159.03-0ubuntu0.22.04.1
apt install nvidia-utils-580-server  # version 580.159.03-0ubuntu0.22.04.1
apt install nvidia-utils-510         # version 510.60.02-0ubuntu1
apt install nvidia-utils-510-server  # version 510.47.03-0ubuntu3
apt install nvidia-utils-470         # version 470.256.02-0ubuntu0.22.04.1
apt install nvidia-utils-550-server  # version 550.163.01-0ubuntu0.22.04.1
apt install nvidia-utils-595         # version 595.71.05-0ubuntu0.22.04.1
apt install nvidia-utils-595-server  # version 595.71.05-0ubuntu0.22.04.1
Ask your administrator to install one of them.
--- ostatnie linie logu ---
[2026-09-22 15:04:12] RUN IDENTITY  model.backbone=dinov2_vits14_reg  model.use_density_head=True  model.density_head_type=radial_attention  data.mask_background=True  data.dual_branch_density=False  data.dual_branch_wedge=True  data.wedge_band_edges_t=[0.0, 0.6, 0.8, 0.9, 1.0]  data.multi_wycinek_k=1  data.strip_mask_background_loss=False  data.quarter_age_adjustment_enabled=False
[2026-09-22 15:04:12] Backbone frozen for first 5 epochs
--- miejsce na dysku ---
/dev/sda2        29T   22T  5,6T  80% /