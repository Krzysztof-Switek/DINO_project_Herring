# Wdrożenie 1 — Podsumowanie

Data: 2026-05-07  
Gałąź: `master`  
Stan końcowy: pipeline uruchomiony na danych rzeczywistych (Z: drive)

---

## 1. Co zostało zbudowane i dlaczego

### 1.1 `src/config.py` — rozszerzenie konfiguracji

**Dodano:**
- klasa `IncrementSamplesConfig` — parametry kart przyrostów (`top_k_best`, `top_k_worst`, `annotate_all`)
- pole `increment_samples` w `InferenceConfig`
- pole `profile_axis: Literal["vertical", "horizontal"]` w `CandidatesConfig`

**Dlaczego:** Pipeline potrzebował konfigurowalnych parametrów dla nowych modułów (wizualizacja, detekcja kandydatów). Wszystkie parametry są w jednym miejscu (YAML) — nie ma magic numbers w kodzie.

**Ważna pułapka:** `IncrementSamplesConfig` musiała być zdefiniowana **przed** `InferenceConfig` w pliku — Pydantic v2 ewaluuje `Field(default_factory=...)` w czasie definicji klasy, więc kolejność klas w pliku ma znaczenie.

---

### 1.2 `src/scan_labels.py` — skanowanie obrazów i budowanie CSV etykiet

**Cel:** Zeskanować katalog Z:/Photo/Otolithes/HER/Processed, dopasować zdjęcia do metadanych z Excela i podzielić na train/val/test bez wycieku danych między rybami.

**Kluczowe decyzje projektowe:**

**`neutral_fish_key`** — identyfikator ryby niezależny od typu preparacji:
```
neutral_fish_key = "{ROK}_{KAMPANIA}_{GATUNEK}_{LOKALIZACJA}_{FishIndex{N}}"
```
Bez niego ta sama ryba miałaby różne klucze dla Embedded i NotEmbedded, co uniemożliwiłoby przypisanie obu zdjęć do tego samego splitu. Z tym kluczem: 2 516 ryb ma zdjęcia w obu typach preparacji — wszystkie trafiają do tego samego splitu (brak wycieku).

**Podział na poziomie ryby** — `assign_split_by_fish()` stratyfikuje po wieku i dzieli po unikalnych `neutral_fish_key`, nie po zdjęciach. Dzięki temu ta sama ryba nie może być jednocześnie w treningu i teście.

**Parametry testowe** (`_image_filenames`, `_excel_df`) — pozwalają wstrzyknąć syntetyczne dane w testach bez dostępu do dysku Z:.

**Wyniki skanu realnych danych:**
- 18 727 zdjęć na dysku
- 12 761 etykietowanych (wiek ≥ 0): 7 837 Embedded + 4 924 NotEmbedded
- 4 482 sieroty (brak metadanych w Excelu — głównie nowsze kampanie 2024+)
- 1 484 wiersze z wiek=-9 (nieznany wiek) — wykluczone z treningu

---

### 1.3 `src/candidates.py` — naprawa osi profilu

**Problem:** Otolyty są fotografowane pionowo — profil ważności powinien być liczony po osi pionowej (mean po kolumnach → wektor wysokości H_p). Stary kod używał domyślnej osi poziomej.

**Rozwiązanie:** Dodano parametr `axis: str = "vertical"` do `extract_radial_profile()`. Przy `axis="vertical"` funkcja robi `mean(axis=1)` (kolapsuje kolumny → zostaje wektor wierszy). Wartość domyślna zmieniona na `"vertical"` — odpowiada orientacji otolytów.

---

### 1.4 `src/visualization.py` — karty przyrostów

**Cel:** Generować wizualne karty dla najlepszych i najgorszych predykcji, z naniesionymi kandydatami na przyrosty roczne (żółte kropki na pionowej osi heatmapy).

**Moduł zawiera:**
- `load_original_image()` — wczytuje oryginalne zdjęcie
- `select_top_k_samples()` — wybiera K najlepszych/najgorszych predykcji z CSV
- `compute_dot_positions()` — przelicza pikselowe pozycje kandydatów na siatkę wyświetlania
- `draw_increment_card()` — rysuje kartę (matplotlib): oryginał + overlay + heatmapa + profil z kropkami
- `save_increment_cards()` — zapisuje karty na dysk

**Uwaga techniczna:** Matplotlib ≥ 3.8 usunął `fig.canvas.tostring_rgb()`. Użyto `fig.canvas.buffer_rgba()` jako zamiennika.

---

### 1.5 `src/comparison_report.py` — raport porównawczy HTML

**Cel:** Jeden samowystarczalny plik HTML porównujący modele Embedded i NotEmbedded we wszystkich 4 warunkach ewaluacji.

**Sekcje raportu:**

| Sekcja | Zawartość |
|--------|-----------|
| A | Statystyki zbioru danych (rozkład wiekowy, counts, sieroty) |
| B | Krzywe treningowe (loss, val_MAE, LR per epoka) |
| C | Metryki ewaluacyjne dla 4 warunków (MAE, RMSE, R², Acc±1yr, Acc±2yr, Bias) |
| D | Macierz cross-ewaluacji 2×2 z komentarzem generalizacji |
| E | Karty przyrostów (best/worst) |
| F | Informacje o modelu i konfiguracji |

**`compute_metrics(y_true, y_pred)`** zwraca: MAE, RMSE, R² (inline, bez sklearn), Acc1yr, Acc2yr, Bias.

**`cross_comment(own_mae, cross_mae)`** — automatyczna interpretacja: jeśli cross_mae < 1.5 × own_mae → "generalizuje dobrze", inaczej → "słaba generalizacja".

---

### 1.6 `scripts/run_pipeline.py` — orkiestrator całego pipeline'u

**9 kroków z możliwością wznowienia:**
```
scan → train_e → train_n → infer_ee → infer_nn → infer_en → infer_ne → cards → report
```

**Resume mechanism** — `pipeline_state.json` przechowuje ukończone kroki. Jeśli pipeline przerwie się w kroku 5, następne uruchomienie zaczyna od kroku 6.

**Config merging** — `load_merged_config(base, override)` z `_deep_update()`: `config.yaml` (lub `config_demo.yaml`) jako baza, `config_embedded.yaml` / `config_not_embedded.yaml` nadpisują pola specyficzne dla modelu.

**`_parse_train_log(log_path)`** — parsuje plain-text `train.log` trainera do listy słowników `{epoch, train_loss, val_loss, val_mae}`. Logi trafiają do sekcji B raportu HTML.

**`_write_pipeline_summary()`** — na końcu każdego uruchomienia zapisuje `pipeline_summary.json`:
```json
{
  "generated_at": "2026-05-07T14:23:45",
  "steps_completed": ["train_e", "train_n", ...],
  "training": {
    "embedded":     {"epochs_completed": 1, "best_val_mae": 3.2, "final_train_loss": 0.61},
    "not_embedded": {"epochs_completed": 1, "best_val_mae": 3.5, "final_train_loss": 0.59}
  },
  "inference": {
    "emb_on_emb":          {"n_samples": 1161, "MAE": 2.8, "RMSE": 3.5, "Acc1yr": 0.65, "Bias": 0.1},
    "notemb_on_notemb":    {...},
    "cross_emb_on_notemb": {...},
    "cross_notemb_on_emb": {...}
  }
}
```

---

### 1.7 `main.py` — punkt wejścia dla PyCharm

**Cel:** Uruchomienie całego pipeline'u jednym kliknięciem ▶ bez pamiętania parametrów CLI.

**Użycie:** Zmień dwie zmienne na górze pliku:

```python
MODE = "demo"   # "demo" → 1 epoka, outputs/demo/
                # "full" → 50 epok, outputs/

SKIP_SCAN  = True   # True = używaj istniejących data/labels_*.csv
SKIP_TRAIN = False  # True = pomija trening, używa istniejących checkpointów
```

Mechanizm: `main.py` buduje `sys.argv` i wywołuje `scripts/run_pipeline.py:main()`.

---

### 1.8 Naprawione błędy

| Błąd | Przyczyna | Naprawa |
|------|-----------|---------|
| `ValueError: Column 'age' contains negative values` | `_validate_columns` sprawdzała cały CSV łącznie z wierszami `age=-9, split=None` | Walidacja tylko dla wierszy z przypisanym splitem (`split.notna()`) |
| `AttributeError: 'TrainingConfig' has no attribute 'num_workers'` | `run_pipeline.py` używał `cfg.training.num_workers` zamiast `cfg.data.num_workers` | Poprawka ścieżki atrybutu |
| Sekcja B raportu zawsze pusta | `build_comparison_report(training_logs={})` — logi nigdy nie były przekazywane | `_step_train()` teraz zwraca `(checkpoint, logi)`, logi trafiają do raportu |
| `fig.canvas.tostring_rgb()` — AttributeError | Usunięte w matplotlib ≥ 3.8 | Zastąpiono `fig.canvas.buffer_rgba()` |
| Kolejność klas w config.py | `IncrementSamplesConfig` użyta w `default_factory` przed definicją | Przeniesiono definicję klasy przed `InferenceConfig` |

---

## 2. Tryb demo — jak działa

**Uruchomienie:** W `main.py` ustaw `MODE = "demo"` i kliknij ▶.

**Co się dzieje:**
1. `config_demo.yaml` ładowany jako baza konfiguracji (zamiast `config.yaml`)
2. Na wierzch nakładane są `config_embedded.yaml` i `config_not_embedded.yaml`
3. Pipeline wykonuje wszystkie 9 kroków, ale z uproszczonymi ustawieniami:

| Parametr | Demo | Pełny trening |
|----------|------|---------------|
| Epoki | 1 | 50 |
| Batch size | 8 | 16 |
| Scheduler LR | none | cosine |
| freeze_backbone | 0 epok | 5 epok |
| num_workers | 0 (bezpieczne Windows) | 0 |
| top_k cards | 3 | 10 |
| Output dir | `outputs/demo/` | `outputs/` |

**Cel demo:** Weryfikacja że **cały pipeline działa end-to-end** — oba modele zostają wytrenowane (po 1 epoce każdy), uruchamiane są 4 warunki inferencji (Emb→Emb, NotEmb→NotEmb, cross Emb→NotEmb, cross NotEmb→Emb), generowane karty przyrostów i raport HTML.

**Jak sprawdzić wyniki demo:**
1. `outputs/demo/pipeline_summary.json` — natychmiastowy podgląd metryk (bez otwierania HTML)
2. `outputs/demo/comparison_report.html` — pełny raport z wykresami
3. `checkpoints/embedded/best.pt` i `checkpoints/not_embedded/best.pt` — zapisane checkpointy

---

## 3. Model — jak działa

### Architektura

```
Zdjęcie otolytu (518×518 px)
    ↓
DINOv2 ViT-S/14 (backbone, pretrenowany self-supervised)
    ├── patch_size = 14 → 37×37 = 1369 patchy
    ├── embed_dim = 384
    └── [CLS] token (globalna reprezentacja) + patch tokens (lokalne)
    ↓
[CLS] token (384-wymiarowy wektor)
    ↓
Dropout(0.1) + Linear(384 → K-1=16)
    ↓
Głowica ordinalna (CORAL method): 16 logitów
```

### Kodowanie ordinalne (CORAL)

Wiek jest zmienną porządkową (0 < 1 < 2 < ... < 16). Zamiast klasyfikacji wieloklasowej używamy kodowania:

```
target[i] = 1  jeśli  wiek > i,  dla i = 0..K-2
target[i] = 0  jeśli  wiek ≤ i
```

Przykład: wiek=3, K=17 → target = [1,1,1,0,0,...,0] (pierwsze 3 jedynki)

Predykcja wieku: `predicted_age = sum(sigmoid(logits) > 0.5)`

**Dlaczego ordinalne zamiast klasyfikacji?** Klasyfikacja traktuje klasy jako niezależne — błąd "3 zamiast 16" jest równie poważny co "3 zamiast 4". Kodowanie ordinalne naturalnie karze większe błędy i wykorzystuje strukturę porządkową danych biologicznych.

### Strategie treningu

**Zamrożenie backbone'u** (`freeze_backbone_epochs: 5`): przez pierwsze 5 epok DINOv2 jest zamrożony, trenuje się tylko głowica. Chroni to pretrenowane reprezentacje przed zniszczeniem zbyt dużym gradientem na początku.

**Odmrożenie:** od epoki 6 cały model jest trenowany ze współczynnikiem uczenia `lr=1e-4`.

**Scheduler cosine:** LR spada kosinusowo od `lr` do 0 przez wszystkie epoki — dobra praktyka dla modeli ViT.

### Cztery warunki ewaluacji (cross-evaluation)

| Warunek | Model | Dane testowe | Pytanie badawcze |
|---------|-------|--------------|------------------|
| `emb_on_emb` | Embedded | Embedded | Baseline — jak dobry jest model na własnych danych? |
| `notemb_on_notemb` | NotEmbedded | NotEmbedded | Baseline — jak dobry jest model na własnych danych? |
| `cross_emb_on_notemb` | Embedded | NotEmbedded | Czy model Embedded generalizuje na inny typ preparacji? |
| `cross_notemb_on_emb` | NotEmbedded | Embedded | Czy model NotEmbedded generalizuje na inny typ preparacji? |

Jeśli cross-MAE ≈ own-MAE → cechy wieku są niezależne od typu preparacji (dobry wynik biologiczny).  
Jeśli cross-MAE >> own-MAE → modele nauczyły się artefaktów preparacji zamiast cech biologicznych.

---

## 4. Pliki konfiguracyjne

| Plik | Rola |
|------|------|
| `configs/config.yaml` | Baza pełnego treningu (50 epok, cosine scheduler, batch=16) |
| `configs/config_demo.yaml` | Baza demo — nadpisuje config.yaml (1 epoka, no scheduler) |
| `configs/config_embedded.yaml` | Nadpisuje: ścieżki CSV/checkpoints/logów dla modelu Embedded |
| `configs/config_not_embedded.yaml` | Nadpisuje: ścieżki CSV/checkpoints/logów dla modelu NotEmbedded |

Hierarchia ładowania: `base_config` + `config_embedded` (lub `config_not_embedded`) → `_deep_update()` → `OtolithConfig`.

---

## 5. Dane wejściowe — podsumowanie

Szczegółowy opis w `plans and summaries/dane_wejściowe_info.md`.

| Element | Wartość |
|---------|---------|
| Katalog zdjęć | `Z:/Photo/Otolithes/HER/Processed` |
| Excel | `data/analysisWithOtolithPhoto.xlsx` |
| Zdjęcia ogółem | 18 727 |
| Labeled (wiek ≥ 0) | 12 761 |
| Sieroty | 4 482 (brak w Excelu) |
| Wiek=-9 (nieznany) | 1 484 (wykluczone) |
| Klasy wiekowe | 0–16 (17 klas) |
| Podział | train 70% / val 15% / test 15% |
| Seed | 42 |

---

## 6. Kolejne kroki po demo

1. **Weryfikacja demo:** otwórz `outputs/demo/pipeline_summary.json` — sprawdź czy `steps_completed` zawiera wszystkie 9 kroków i czy MAE jest w rozsądnym zakresie (oczekiwane 3–6 dla 1 epoki)
2. **Pełny trening:** zmień `MODE = "full"` w `main.py`, kliknij ▶ — trening ~50 epok × 2 modele (kilka godzin na GPU)
3. **Opcjonalnie:** włączyć `use_metadata: true` i dodać `metadata_cols` aby sprawdzić czy dane biologiczne (długość, waga) poprawiają predykcję
4. **Opcjonalnie:** przetestować większy backbone `dinov2_vitb14` (wyższy `embed_dim`, wolniejszy)
