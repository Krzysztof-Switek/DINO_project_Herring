/home/kswitek/Documents/DINO_project_Herring/.venv/bin/python /home/kswitek/Documents/DINO_project_Herring/main.py 
State: /home/kswitek/Documents/DINO_project_Herring/outputs/pipeline_state.json
============================================================
OtolithDino — pipeline Embedded vs NotEmbedded
============================================================

[1/9] SCAN — pominięty
  Embedded labels:    /home/kswitek/Documents/DINO_project_Herring/outputs/data/labels_embedded.csv
  NotEmbedded labels: /home/kswitek/Documents/DINO_project_Herring/outputs/data/labels_not_embedded.csv

[2/9] TRAIN — Embedded
Using cache found in /home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main
/home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
  warnings.warn("xFormers is not available (SwiGLU)")
/home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:33: UserWarning: xFormers is not available (Attention)
  warnings.warn("xFormers is not available (Attention)")
/home/kswitek/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:40: UserWarning: xFormers is not available (Block)
  warnings.warn("xFormers is not available (Block)")
[2026-07-09 21:55:52] Backbone frozen for first 5 epochs
Traceback (most recent call last):
  File "/home/kswitek/Documents/DINO_project_Herring/main.py", line 65, in <module>
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
FileNotFoundError: [Errno 2] No such file or directory: '/home/kswitek/Documents/DINO_project_Herring/Z:/Photo/Otolithes/HER/Processed/2023_BITS4q_HER_LawicaSrodkowa_Embedded_Sharpest_FishIndex4_Single1_Left.jpg'

Process finished with exit code 1
