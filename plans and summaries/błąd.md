/home/kswitek/Documents/DINO_project_Herring/.venv/bin/python /home/kswitek/Documents/DINO_project_Herring/main_density_lab.py 
[main_density_lab 12:36:44] LOCATION=server  IMAGE_DIR=/home/kswitek/Documents/Photo/Otolithes/HER/Processed  CKPT_B=/home/kswitek/Documents/DINO_project_Herring/outputs/data/22.09_wedge_b/checkpoints/embedded/best_age.pt
[main_density_lab 12:36:44] plan: cache ['raw', 'wedge_b_best_age']; ramiona raw ['A3', 'A0', 'A1', 'A2', 'A4'] × ziarna [0, 1, 2, 3, 4]; kontrola ['A0'] na wedge_b_best_age; 30 epok, batch 16, cel quarter
[main_density_lab 12:36:45] BŁĄD: torch 2.12.0+cu130 nie widzi GPU (torch.cuda.is_available() = False). Ramiona A0–A4 na CPU to dziesiątki godzin na ziarno — przerywam. Sprawdź: nvidia-smi; python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"; zmienną CUDA_VISIBLE_DEVICES. Świadomie na CPU: REQUIRE_CUDA = False.

Process finished with exit code 1