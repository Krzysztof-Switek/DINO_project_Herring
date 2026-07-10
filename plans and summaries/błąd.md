/home/kswitek/Documents/DINO_project_Herring/.venv/bin/python /home/kswitek/Documents/DINO_project_Herring/main.py 
[main] LOCATION=server  IMAGE_DIR=/home/kswitek/Documents/Photo/Otolithes/HER/Processed
[fresh] Usunięto /home/kswitek/Documents/DINO_project_Herring/outputs/pipeline_state.json — wymuszam pełny re-run
State: /home/kswitek/Documents/DINO_project_Herring/outputs/pipeline_state.json
============================================================
OtolithDino — pipeline Embedded vs NotEmbedded
============================================================

[1/9] SCAN — budowanie labels CSVs
  Zdjecia na dysku: 18727
  Unikalne tokeny [4] (typ preparacji): ['Embedded', 'NotEmbedded']
  Unikalne tokeny [5] (postproc):       ['Sharpest', 'WithoutPostproc']
  Sparsowane pliki: 18387
  Embedded:    9310
  NotEmbedded: 9077

  Wczytywanie metadanych z Excel ...
  Sieroty (brak metadanych w Excel): 4453
  Rozkład wieku per split (mediana wieku ryby):
    train: n_ryb= 2720  mean=3.75  median=4.0  min=0  max=16
    val  : n_ryb=  588  mean=3.78  median=4.0  min=0  max=15
    test : n_ryb=  578  mean=3.73  median=4.0  min=0  max=15

  Rozkład wiekowy (labeled, n=12496):
    Embedded: wiek 0–16, mediana=4.0, n=7588
    NotEmbedded: wiek 0–15, mediana=4.0, n=4908
  Ryby w >1 zbiorze: 0 (brak wycieku)
  Saved: /home/kswitek/Documents/DINO_project_Herring/outputs/data/labels_embedded.csv, /home/kswitek/Documents/DINO_project_Herring/outputs/data/labels_not_embedded.csv

[2/9] TRAIN — Embedded
Using cache found in /home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main
/home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
  warnings.warn("xFormers is not available (SwiGLU)")
/home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:33: UserWarning: xFormers is not available (Attention)
  warnings.warn("xFormers is not available (Attention)")
/home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:40: UserWarning: xFormers is not available (Block)
  warnings.warn("xFormers is not available (Block)")
[2026-07-10 09:02:48] Backbone frozen for first 5 epochs
Traceback (most recent call last):
  File "/home/kswitek/Documents/DINO_project_Herring/main.py", line 80, in <module>
    main(ARGV)
  File "/home/kswitek/Documents/DINO_project_Herring/scripts/run_pipeline.py", line 787, in main
    ckpt_emb, logs_emb = _step_train(cfg_emb, emb_labels)
  File "/home/kswitek/Documents/DINO_project_Herring/scripts/run_pipeline.py", line 194, in _step_train
    trainer.fit()
  File "/home/kswitek/Documents/DINO_project_Herring/src/trainer.py", line 246, in fit
    train_loss = self.train_one_epoch()
  File "/home/kswitek/Documents/DINO_project_Herring/src/trainer.py", line 144, in train_one_epoch
    for batch in self.train_loader:
  File "/home/kswitek/Documents/DINO_project_Herring/.venv/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 718, in __next__
    data = self._next_data()
  File "/home/kswitek/Documents/DINO_project_Herring/.venv/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 778, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/home/kswitek/Documents/DINO_project_Herring/.venv/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 54, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/kswitek/Documents/DINO_project_Herring/.venv/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 54, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/kswitek/Documents/DINO_project_Herring/src/dataset.py", line 196, in __getitem__
    image_tensor = self._load_image(image_id)
  File "/home/kswitek/Documents/DINO_project_Herring/src/dataset.py", line 223, in _load_image
    image = Image.open(path).convert("RGB")
  File "/home/kswitek/Documents/DINO_project_Herring/.venv/lib/python3.10/site-packages/PIL/Image.py", line 3635, in open
    fp = builtins.open(filename, "rb")
FileNotFoundError: [Errno 2] No such file or directory: '/home/kswitek/Documents/DINO_project_Herring/Z:/Photo/Otolithes/HER/Processed/2023_BITS4q_HER_Wladyslawowskie_Embedded_Sharpest_FishIndex128_Single1_Right.jpg'