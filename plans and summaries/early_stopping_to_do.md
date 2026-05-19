# Early stopping — plan wdrożenia

Data: 2026-05-07  
Status: **do zaimplementowania** (trening przebiega na sztywno 50 epok)

---

## 1. Dlaczego early stopping jest potrzebny

Aktualny trener (`src/trainer.py`) wykonuje dokładnie `cfg.training.epochs` epok bez możliwości wcześniejszego zatrzymania. Oznacza to dwa ryzyka:

**Przeuczenie (overfitting):** Model może osiągnąć optimum np. po 30 epokach, a przez kolejne 20 epok stopniowo "zapamiętywać" szum treningowy — val_MAE rośnie, choć train_loss spada. Bez early stopping zapisujemy checkpoint z ostatniej epoki, nie z najlepszej.

**Marnowanie zasobów:** Trening DINOv2 ViT-S na ~8 000 zdjęć na GPU trwa kilka godzin. Jeśli model zbiegł już po 25 epokach, kolejne 25 epok to stracony czas.

---

## 2. Wybór metryki — uzasadnienie literaturowe

### Konsensus literaturowy

**Estymacja wieku ryb z otolytów:**
- Thanassas et al. (2024, *ICES J. Marine Science*) — modele głębokiego uczenia do odczytu otolytów stosują walidację krzyżową z monitorowaniem błędu predykcji w jednostkach naturalnych (lata)
- Moen et al. (2024, *PLOS ONE*) — automatyczna estymacja wieku z obrazów otolytów halibuta grenlandzkiego: kryterium stopu oparte na walidacyjnym MAE
- Chen et al. (2021, *Fisheries Research*) — multitask learning do estymacji wieku: wczesne zatrzymanie na podstawie błędu walidacyjnego MAE

**Regresja ordinalna (CORAL):**
- Cao et al. (2020, *Pattern Recognition Letters*, "Rank consistent ordinal regression for neural networks") — autorzy CORAL stosują early stopping z **patience = 10 epok** przy max 100 epokach; metryka: walidacyjny MAE
- Raschka (2021, coral-pytorch) — implementacja referencyjna CORAL: `monitor="val_mae"` jako domyślna metryka wczesnego zatrzymania

**Modele ViT i self-supervised learning:**
- Dosovitskiy et al. (2021, ViT) oraz Oquab et al. (2023, DINOv2): fine-tuning pretrenowanych ViT jest wrażliwy na przeuczenie w późnych epokach — autorzy zalecają monitorowanie metryki zadaniowej (MAE/accuracy), nie straty

### Dlaczego `val_mae`, nie `val_loss`

| Kryterium | `val_mae` | `val_loss` (ordinalna) |
|-----------|-----------|------------------------|
| Interpretowalność | Błąd w latach — bezpośrednio znaczący | Bezwymiarowa strata — trudna do interpretacji |
| Cel biologiczny | Tak — minimalizujemy błąd predykcji wieku | Pośrednio |
| Odporność na kalibrację | Nie zależy od kalibracji logitów | Może spadać przy samej kalibracji bez poprawy MAE |
| Konsensus | **Zalecany** przez CORAL i studia otolytowe | Drugorzędny |

**Wniosek: `val_mae` jako metryka primary, patience = 10 epok, min_delta = 0.001 roku.**

---

## 3. Parametry early stopping

| Parametr | Wartość | Uzasadnienie |
|----------|---------|--------------|
| `early_stopping_patience` | **10** | Standard z CORAL paper; przy 50 epokach daje ok. 20% tolerancji plateau |
| `early_stopping_metric` | **`val_mae`** | Bezpośredni błąd predykcji w latach, zalecany w literaturze |
| `early_stopping_min_delta` | **0.001** | Poprawa < 0.001 roku to szum numeryczny (poniżej rozdzielczości biologicznej) |
| `patience=0` | wyłączone | Wartość 0 wyłącza early stopping (np. w eksperymentach) |

---

## 4. Stan aktualny kodu

```
src/trainer.py
  fit():
    for epoch in range(1, cfg.epochs + 1):   ← brak break
        train_loss = self.train_one_epoch()
        val_loss, val_mae = self.validate()
        self._log_epoch(...)
        self.save_checkpoint(epoch, val_loss)  ← każda epoka, val_loss w nazwie
    self._log("Training complete")             ← zawsze po wszystkich epokach

scripts/run_pipeline.py
  _step_train():
    trainer.fit()
    ckpt_files = glob("checkpoint_epoch*.pt")
    best = min(ckpt_files, key=val_loss_from_filename)  ← post-hoc wybór
    copy(best, "best.pt")
```

**Brakuje:**
- pól `early_stopping_*` w `TrainingConfig`
- logiki early stopping w `fit()`
- zapisu `best.pt` w trakcie treningu (nie post-hoc)

---

## 5. Kroki wdrożenia

### Krok 1 — `src/config.py`

Dodaj do klasy `TrainingConfig` (po `freeze_backbone_epochs`):

```python
early_stopping_patience: int = Field(10, ge=0)
early_stopping_metric: Literal["val_mae", "val_loss"] = "val_mae"
early_stopping_min_delta: float = Field(0.001, ge=0.0)
```

> `patience=0` wyłącza early stopping (warunek `patience > 0` nigdy nie spełniony).

---

### Krok 2 — `src/trainer.py`

**a) Nowa metoda `_save_best_checkpoint(epoch, val_loss)`** — wstaw po `save_checkpoint()`:

```python
def _save_best_checkpoint(self, epoch: int, val_loss: float) -> None:
    import shutil as _shutil
    src = self.checkpoint_dir / f"checkpoint_epoch{epoch:03d}_loss{val_loss:.4f}.pt"
    if src.exists():
        _shutil.copy2(src, self.checkpoint_dir / "best.pt")
```

**b) Zmodyfikuj `fit()` — dodaj early stopping do pętli treningowej:**

```python
patience     = self.cfg.training.early_stopping_patience
min_delta    = self.cfg.training.early_stopping_min_delta
metric_name  = self.cfg.training.early_stopping_metric
best_metric  = float("inf")
patience_counter = 0

for epoch in range(1, self.cfg.training.epochs + 1):
    # ... istniejący kod freeze/unfreeze ...

    train_loss = self.train_one_epoch()
    val_loss, val_mae = self.validate()

    if self.scheduler is not None:
        self.scheduler.step()

    self._log_epoch(epoch, train_loss, val_loss, val_mae)
    self.save_checkpoint(epoch, val_loss)

    # --- Early stopping ---
    current = val_mae if metric_name == "val_mae" else val_loss
    if current < best_metric - min_delta:
        best_metric = current
        patience_counter = 0
        self._save_best_checkpoint(epoch, val_loss)   # zapisz best.pt
    else:
        patience_counter += 1
        if patience > 0 and patience_counter >= patience:
            self._log(
                f"Early stopping — brak poprawy {metric_name} przez {patience} epok "
                f"(best={best_metric:.4f})"
            )
            break

self._log("Training complete")
```

---

### Krok 3 — `scripts/run_pipeline.py`

Trener teraz tworzy `best.pt` sam. Uprość `_step_train()`:

```python
best_ckpt = trainer.checkpoint_dir / "best.pt"
if not best_ckpt.exists():
    # Fallback dla starych checkpointów (wznowienie po poprzedniej wersji kodu)
    ckpt_files = sorted(trainer.checkpoint_dir.glob("checkpoint_epoch*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint saved in {trainer.checkpoint_dir}")
    best_ckpt_src = min(ckpt_files, key=lambda p: float(p.stem.split("_loss")[-1]))
    import shutil as _shutil
    _shutil.copy2(best_ckpt_src, best_ckpt)
print(f"  Best checkpoint: {best_ckpt}")
return best_ckpt, training_log_data
```

---

### Krok 4 — `configs/config.yaml`

Dodaj po `freeze_backbone_epochs`:

```yaml
  early_stopping_patience: 10      # 0 = wyłączone; zalecane: 10
  early_stopping_metric: val_mae   # val_mae (zalecane) | val_loss
  early_stopping_min_delta: 0.001  # minimalna poprawa żeby zresetować licznik
```

---

### Krok 5 — testy w `tests/test_stage4_trainer.py`

Dodaj 3 testy używając podklasy `_ConstantValTrainer` która zawsze zwraca `val_mae=5.0`:

```python
class _ConstantValTrainer(Trainer):
    def validate(self): return 1.0, 5.0   # brak poprawy po epoce 1
```

| Test | Co sprawdza |
|------|-------------|
| `test_early_stopping_triggers` | `epochs=10, patience=2` → tylko 3 checkpointy (stop po epoce 3) |
| `test_early_stopping_saves_best_pt` | `best.pt` istnieje po `fit()` |
| `test_early_stopping_disabled` | `patience=0, epochs=3` → wszystkie 3 checkpointy |

---

## 6. Pliki do modyfikacji

| Plik | Zmiana |
|------|--------|
| `src/config.py` | 3 nowe pola w `TrainingConfig` |
| `src/trainer.py` | `_save_best_checkpoint()` + early stopping w `fit()` |
| `scripts/run_pipeline.py` | uproszczenie wyboru `best.pt` w `_step_train()` |
| `configs/config.yaml` | 3 nowe pola YAML |
| `tests/test_stage4_trainer.py` | 3 nowe testy |

`configs/config_demo.yaml` — **brak zmian** (1 epoka w demo nie wyzwoli early stopping).

---

## 7. Źródła

- Cao, W. et al. (2020). "Rank consistent ordinal regression for neural networks with application to age estimation." *Pattern Recognition Letters*, 140, 325–331. https://arxiv.org/abs/1901.07884
- Raschka, S. (2021). coral-pytorch — reference implementation. https://github.com/Raschka-research-group/coral-pytorch
- Moen, E. et al. (2024). "Age prediction by deep learning for Greenland halibut otolith images." *PLOS ONE*. https://doi.org/10.1371/journal.pone.0277244
- Chen, Z. et al. (2021). "Automating fish age estimation combining otolith images and deep learning." *Fisheries Research*, 234. https://doi.org/10.1016/j.fishres.2021.105776
- Dosovitskiy, A. et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*. https://arxiv.org/abs/2010.11929