(.venv) kswitek@labworks:~/Documents/DINO_project_Herring$ D=/home/kswitek/Documents/DINO_project_Herring                                                                                                                                                                                   
  C=$D/data/wedge_bands_cache/band3_168x2254_98.7deg                                                                                                                                                                               
  echo "=== godzina teraz ==="; date +%H:%M:%S                                                                                                                                                                                     
  echo "=== 3 NAJNOWSZE pliki cache (kluczowe: godzina zapisu) ==="                                                                                                                                                                
  ls -lt --time-style=+%H:%M:%S $C | head -4                                                                                                                                                                                       
  echo "=== log: rozmiar i godzina modyfikacji ==="                                                                                                                                                                                
  ls -l --time-style=+%H:%M:%S $D/outputs/data/22.09_wedge_b/logs/embedded/train.log                                                                                                                                               
  echo "=== proces: czas, CPU, PAMIEC, stan ==="                                                                                                                                                                                   
  ps -eo pid,etime,%cpu,%mem,rss,stat,cmd | grep main_wedge_b | grep -v grep                                                                                                                                                       
  echo "=== RAM i SWAP ==="                                                                                                                                                                                                        
  free -h  
=== godzina teraz ===
17:16:15
=== 3 NAJNOWSZE pliki cache (kluczowe: godzina zapisu) ===
total 1073512
-rw-rw-r-- 1 kswitek kswitek   4462 17:16:11 2023_BITS4q_HER_GlebiaGdanska_Embedded_Sharpest_FishIndex45_Single2_Left_wedge_band3.geom.npz
-rw-rw-r-- 1 kswitek kswitek 175448 17:16:11 2023_BITS4q_HER_GlebiaGdanska_Embedded_Sharpest_FishIndex45_Single2_Left_wedge_band3.png
-rw-rw-r-- 1 kswitek kswitek   4462 17:16:11 2023_BITS4q_HER_GlebiaGdanska_Embedded_Sharpest_FishIndex45_Single1_Right_wedge_band3.geom.npz
=== log: rozmiar i godzina modyfikacji ===
-rw-rw-r-- 1 kswitek kswitek 442 15:04:12 /home/kswitek/Documents/DINO_project_Herring/outputs/data/22.09_wedge_b/logs/embedded/train.log
=== proces: czas, CPU, PAMIEC, stan ===
2002677    05:12:53 1959  0.2 2763856 Rl /home/kswitek/Documents/DINO_project_Herring/.venv/bin/python /home/kswitek/Documents/DINO_project_Herring/main_wedge_b.py
=== RAM i SWAP ===
               total        used        free      shared  buff/cache   available
Mem:           1,0Ti        36Gi       147Gi       196Mi       823Gi       964Gi
Swap:          2,0Gi       2,0Gi        13Mi
(.venv) kswitek@labworks:~/Documents/DINO_project_Herring$ 