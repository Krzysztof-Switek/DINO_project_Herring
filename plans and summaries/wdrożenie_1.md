# Wdrożenie 1 — Pipeline Embedded vs NotEmbedded

Data: 2026-05-07
Odniesienie: `embedded_notembedded_plan.md`

---

## Zasady ogólne

- Każde zadanie kończy się działającym testem — nie przechodzimy dalej bez zielonego testu
- Zadania oznaczone 🔌 wymagają dostępu do dysku `Z:` (uruchamiamy je na końcu)
- Zadania bez 🔌 można wykonać i przetestować bez sieci / GPU
- Modyfikujemy co najwyżej 1 plik na krok
- Po każdej modyfikacji: `python -m pytest <plik_testu> -v`

---

## Mapa zależności

```
Faza 0: config.py + YAML
    ↓
Faza 1: src/scan_labels.py
    ↓
Faza 2: src/candidates.py (naprawa osi)
    ↓
Faza 3: src/visualization.py
    ↓
Faza 4: src/comparison_report.py
    ↓
Faza 5: scripts/run_pipeline.py
    ↓
Faza 6: uruchomienie na prawdziwych danych 🔌
```

---

## Faza 0 — Konfiguracja (prereqs)

### Zadanie 0.1 — Rozszerzyć `src/config.py`

**Co dodać:**

```python
class CandidatesConfig(BaseModel):
    min_peak_distance: int = Field(5, ge=1)
    prominence_threshold: float = Field(0.1, ge=0.0)
    profile_axis: Literal["vertical", "horizontal"] = "vertical"  # NOWE

class IncrementSamplesConfig(BaseModel):                           # NOWA klasa
    top_k_best: int = Field(10, ge=1)
    top_k_worst: int = Field(10, ge=1)
    annotate_all: bool = False

class InferenceConfig(BaseModel):
    output_dir: str = "outputs"
    save_heatmaps: bool = True
    save_overlays: bool = True
    save_candidates: bool = True
    increment_samples: IncrementSamplesConfig = Field(             # NOWE
        default_factory=IncrementSamplesConfig)
```

Uwaga: `use_metadata` już ma default `False` w `ModelConfig:19` — bez zmian.

**Pliki:** `src/config.py`

**Test:**
```bash
python -c "from src.config import get_default_config; c = get_default_config(); \
  assert c.candidates.profile_axis == 'vertical'; \
  assert c.inference.increment_samples.top_k_best == 10; \
  print('OK')"
```

**Gotowe gdy:** powyższy one-liner drukuje `OK` bez błędu.

---

### Zadanie 0.2 — Zaktualizować `configs/config.yaml`

**Co dodać** (do istniejących sekcji):

```yaml
model:
  use_metadata: false      # upewnić się że false (domyślnie już false w config.py)

candidates:
  profile_axis: vertical   # pionowe zdjęcia — profil wzdłuż osi Y

inference:
  increment_samples:
    top_k_best: 10
    top_k_worst: 10
    annotate_all: false
```

**Pliki:** `configs/config.yaml`

**Test:**
```bash
python -c "from src.config import load_config; c = load_config('configs/config.yaml'); \
  assert c.candidates.profile_axis == 'vertical'; \
  assert c.model.use_metadata == False; print('OK')"
```

**Gotowe gdy:** `python -m pytest tests/ -v` — wszystkie istniejące testy nadal zielone.

---

### Zadanie 0.3 — Stworzyć `configs/config_embedded.yaml` i `configs/config_not_embedded.yaml`

Każdy plik nadpisuje tylko te pola które różnią się od `config.yaml`:

**`config_embedded.yaml`:**
```yaml
data:
  labels_csv: data/labels_embedded.csv
  image_dir: "Z:/Photo/Otolithes/HER/Processed"

training:
  checkpoint_dir: checkpoints/embedded
  log_dir: logs/embedded

inference:
  output_dir: outputs/emb_on_emb
```

**`config_not_embedded.yaml`:**
```yaml
data:
  labels_csv: data/labels_not_embedded.csv
  image_dir: "Z:/Photo/Otolithes/HER/Processed"

training:
  checkpoint_dir: checkpoints/not_embedded
  log_dir: logs/not_embedded

inference:
  output_dir: outputs/notemb_on_notemb
```

**Pliki:** `configs/config_embedded.yaml`, `configs/config_not_embedded.yaml` (nowe)

**Test:**
```bash
python -c "
from src.config import load_config
e = load_config('configs/config_embedded.yaml')
n = load_config('configs/config_not_embedded.yaml')
assert 'embedded' in e.data.labels_csv
assert 'not_embedded' in n.data.labels_csv
assert e.model.use_metadata == False
assert n.model.use_metadata == False
print('OK')
"
```

**Gotowe gdy:** oba pliki ładują się bez błędu walidacji Pydantic.

---

## Faza 1 — Przygotowanie danych

### Zadanie 1.1 — Stworzyć `src/scan_labels.py`

Szczegółowy projekt: sekcja 7 w `embedded_notembedded_plan.md`.

**Funkcje do implementacji:**

```python
def parse_filename(name: str) -> dict | None:
    """parts[4] ∈ {Embedded, NotEmbedded}, neutral_fish_key = parts[0..3]+parts[6]"""

def load_excel_metadata(excel_path: Path) -> pd.DataFrame:
    """Mapuje te same kolumny co prepare_labels.py:156. Filtruje GOOD_TYPES.
    Zwraca lookup keyed by neutral_fish_key."""

def build_combined_labels(
    image_dir: Path, excel_path: Path,
    train: float = 0.70, val: float = 0.15, seed: int = 42,
) -> pd.DataFrame:
    """Scan → parse → match → assign_split_by_fish → return DataFrame."""
```

Reużywa z `scripts/prepare_labels.py`: `scan_image_dir()` (linia 44),
`assign_split_by_fish()` (linia 71) — import bezpośredni.

Drukuje na stdout raport (unikalne tokeny parts[4] i parts[5], liczby, rozkład wiekowy,
liczba sierot not-embedded).

**Pliki:** `src/scan_labels.py` (nowy)

**Test:**
```bash
python -m pytest tests/test_stage9_scan_labels.py -v
```

**Zawartość testu** (`tests/test_stage9_scan_labels.py`):

| Test | Co sprawdza |
|------|-------------|
| `test_parse_embedded` | Poprawny parse embedded filename |
| `test_parse_not_embedded` | Poprawny parse not-embedded filename |
| `test_parse_short_filename` | Zbyt krótka nazwa → None |
| `test_parse_unknown_type` | Nieznany token [4] → None |
| `test_neutral_key_same_fish` | Embedded i NotEmbedded tej samej ryby → identyczny klucz |
| `test_build_combined_no_split_leak` | Żadna ryba nie w >1 splicie |
| `test_build_combined_shared_split` | Ta sama ryba: embedded i not-embedded w tym samym splicie |
| `test_build_combined_orphan_flag` | Not-embedded bez odpowiednika w Excel → orphan=True |

Wszystkie testy używają syntetycznych danych (bez Z:, bez Excela — mock DataFrame).

**Gotowe gdy:** `pytest tests/test_stage9_scan_labels.py -v` — 8/8 zielonych.

---

### Zadanie 1.2 — Skrypt CLI dla scan_labels

Dodać blok `if __name__ == "__main__"` do `src/scan_labels.py` z argparse:

```bash
python src/scan_labels.py \
    --image-dir "Z:/Photo/Otolithes/HER/Processed" \
    --excel data/analysisWithOtolithPhoto.xlsx \
    --output data/labels_combined.csv
```

Po uruchomieniu: dzieli `labels_combined.csv` na `labels_embedded.csv`
i `labels_not_embedded.csv` automatycznie.

**Test (bez Z:):**
```bash
python src/scan_labels.py --help   # musi wydrukować usage bez błędu
```

**Gotowe gdy:** `--help` działa; testy z Zadania 1.1 nadal zielone.

---

## Faza 2 — Naprawa `src/candidates.py`

### Zadanie 2.1 — Parametr `profile_axis` w `extract_radial_profile`

**Zmiana:** Obecna funkcja używa `mean(axis=0)` → profil poziomy.
Dla pionowych zdjęć otolitów potrzebny `mean(axis=1)` → profil pionowy.

```python
def extract_radial_profile(
    importance_grid: Union[Tensor, np.ndarray],
    axis: str = "vertical",   # "vertical" | "horizontal"
) -> np.ndarray:
    arr = importance_grid.cpu().numpy() if hasattr(importance_grid, "cpu") else importance_grid
    collapse_axis = 1 if axis == "vertical" else 0
    return arr.mean(axis=collapse_axis).astype(np.float32)
```

Zaktualizować wywołanie w `run_candidates()` aby czytało `cfg.candidates.profile_axis`.

Zaktualizować `peaks_to_pixel_positions()` — dla osi pionowej konwertuje na y-piksele
(zamiast x-pikseli), używając `image_height` zamiast `image_width`.

**Pliki:** `src/candidates.py`

**Test:**
```bash
python -m pytest tests/test_stage7_candidates.py -v
```

Dodać do testu:
- `test_vertical_profile_shape` — grid (4, 8) + axis='vertical' → profil (4,)
- `test_horizontal_profile_shape` — grid (4, 8) + axis='horizontal' → profil (8,)
- `test_pixel_positions_vertical` — y-piksele mieszczą się w (0, image_height)

**Gotowe gdy:** wszystkie testy `test_stage7_candidates.py` zielone (stare + nowe).

---

## Faza 3 — Wizualizacja przyrostów

### Zadanie 3.1 — Stworzyć `src/visualization.py`

Szczegółowy projekt: sekcja 10 w `embedded_notembedded_plan.md`.

**Funkcje:**

```python
def load_original_image(image_id: str, image_dir: Path) -> np.ndarray:
    """Ładuje oryginał z dysku (bez żadnego przetwarzania modelu). RGB uint8."""

def select_top_k_samples(
    predictions_csv: Path, k_best: int = 10, k_worst: int = 10,
) -> tuple[list[dict], list[dict]]:
    """Sortuje po |predicted_age - age|. Zwraca (best_k, worst_k)."""

def draw_increment_card(
    original_rgb: np.ndarray,
    dot_y_positions: list[int],
    importance_grid,
    predicted_age: int,
    true_age: int,
    last_sigmoid: float,
) -> np.ndarray:
    """
    Panel A — oryginał:
      - Pionowa linia (220, 30, 30) przez x = W//2, grubość 1px
      - Żółte kółka (255, 230, 0) r=8px z czarną obwódką, numer w środku
      - Jeśli last_sigmoid > 0.3: ostatnie kółko puste + etykieta 'N+'
      - Wiek w rogu: biały tekst, czarny cień; kolor ramki: zielony/czerwony
    Panel B — heatmap overlay (istniejąca logika z candidates.py)
    Panel C — pionowy profil ważności (matplotlib, poziome przerywane linie na pikach)
    Zwraca skomponowany obraz (A|B|C) jako np.ndarray.
    """

def save_increment_cards(
    samples: list[dict],
    image_dir: Path,
    importance_grids: dict,    # image_id → importance_grid
    last_sigmoids: dict,       # image_id → float
    output_dir: Path,
    label: str,                # "best" lub "worst"
) -> list[Path]:
    """Generuje i zapisuje karty PNG dla listy próbek."""
```

**Kolory (uzasadnione analizą zdjęć — czarne tło, jasny otolit):**
- Oś: `(220, 30, 30)` ciemnoczerwona
- Kropki wypełnione: `(255, 230, 0)` żółte, obwódka czarna 1px, numer czarny
- Niepełny przyrost: żółty kontur bez wypełnienia + "N+"
- Tekst wieku: biały z czarnym cieniem (`cv2.putText` z dwiema warstwami)

**Pliki:** `src/visualization.py` (nowy)

**Test:**
```bash
python -m pytest tests/test_stage10_visualization.py -v
```

| Test | Co sprawdza |
|------|-------------|
| `test_load_original_image` | Ładuje syntetyczny plik PNG, zwraca (H,W,3) uint8 |
| `test_select_top_k` | Poprawna selekcja best/worst z testowego CSV |
| `test_draw_card_shape` | Karta ma oczekiwane wymiary (3 panele obok siebie) |
| `test_draw_card_last_sigmoid_hollow` | last_sigmoid > 0.3 → ostatnie kółko puste |
| `test_draw_card_saves_file` | `save_increment_cards()` tworzy pliki PNG na dysku |

**Gotowe gdy:** 5/5 testów zielonych; wizualna inspekcja jednej karty z syntetycznym obrazem.

---

## Faza 4 — Raport porównawczy

### Zadanie 4.1 — Stworzyć `src/comparison_report.py`

Szczegółowy projekt: sekcja 9 w `embedded_notembedded_plan.md`.

**Główna funkcja:**
```python
def build_comparison_report(
    results: dict,           # klucze: emb_on_emb, notemb_on_notemb, emb_on_notemb, notemb_on_emb
    training_logs: dict,     # klucze: embedded, not_embedded → lista dicts per epoka
    increment_cards: dict,   # klucze: best, worst → lista ścieżek PNG
    dataset_stats: dict,
    output_path: Path,
    image_dir: Path,
) -> None
```

**Sekcje raportu HTML** (pełna lista — sekcja 9 planu):

| Sekcja | Zawartość |
|--------|-----------|
| A. Statystyki zbioru | Tabela N per typ per split, histogram rozkładu wiekowego, liczba sierot |
| B. Dane treningowe | Krzywe loss (train+val), LR schedule, val_MAE per epoka, tabela best epoch |
| C. Metryki ewaluacyjne | MAE, RMSE, R², Acc±1yr, Acc±2yr, Bias — dla 4 warunków |
| C2. Wykresy | Scatter predicted vs actual, histogram błędów, MAE per klasa, macierz pomyłek, box plot |
| D. Cross-ewaluacja | Tabela 2×2 z automatycznym komentarzem (cross MAE < 1.5×own MAE → generalizuje) |
| E. Karty przyrostów | 10 najlepszych + 10 najgorszych predykcji z kropkami, per typ i per cross |
| F. Info o modelu | Backbone, N parametrów, ścieżki checkpointów, czas treningu |

**Pliki:** `src/comparison_report.py` (nowy)

**Test:**
```bash
python -m pytest tests/test_stage11_comparison_report.py -v
```

| Test | Co sprawdza |
|------|-------------|
| `test_compute_metrics` | MAE/RMSE/R²/Acc na syntetycznych predykcjach |
| `test_cross_comment_good` | cross MAE < 1.5×own → komentarz "generalizuje" |
| `test_cross_comment_bad` | cross MAE > 1.5×own → komentarz "słaba generalizacja" |
| `test_build_report_creates_file` | `build_comparison_report()` tworzy plik HTML |
| `test_report_has_all_sections` | HTML zawiera sekcje A–F (sprawdzenie po tytułach) |

**Gotowe gdy:** 5/5 testów zielonych; plik HTML otwiera się w przeglądarce bez błędów JS.

---

## Faza 5 — Orkiestrator pipeline'u

### Zadanie 5.1 — Stworzyć `scripts/run_pipeline.py`

Szczegółowy projekt: sekcja 8 w `embedded_notembedded_plan.md`.

**Argumenty CLI:**
```bash
python scripts/run_pipeline.py \
    --image-dir  "Z:/Photo/Otolithes/HER/Processed" \
    --excel      data/analysisWithOtolithPhoto.xlsx \
    --output-dir outputs/ \
    --config-embedded     configs/config_embedded.yaml \
    --config-not-embedded configs/config_not_embedded.yaml \
    [--skip-scan]    # jeśli labels_combined.csv już istnieje
    [--skip-train]   # jeśli checkpointy już istnieją
```

**Kroki wewnętrzne (w kolejności):**
```
1. scan  → build_combined_labels() → labels_embedded.csv + labels_not_embedded.csv
2. train embedded   → checkpoints/embedded/best.pt
3. train not_emb    → checkpoints/not_embedded/best.pt
4. infer emb→emb    → outputs/emb_on_emb/predictions.csv
5. infer notemb→notemb → outputs/notemb_on_notemb/predictions.csv
6. infer emb→notemb (CROSS) → outputs/cross_emb_on_notemb/predictions.csv
7. infer notemb→emb (CROSS) → outputs/cross_notemb_on_emb/predictions.csv
8. generate increment cards (top 10 best + worst per warunek)
9. build_comparison_report() → outputs/comparison_report.html
```

Po każdym ukończonym kroku: zapis `outputs/pipeline_state.json`
(umożliwia wznowienie od miejsca przerwania).

**Pliki:** `scripts/run_pipeline.py` (nowy)

**Test bez Z: (smoke test z MockDinoBackbone):**
```bash
python scripts/run_pipeline.py \
    --image-dir data/test_images \
    --excel     data/labels_sample.csv \
    --output-dir outputs/smoke_test/ \
    --skip-train \
    --dry-run    # drukuje plan kroków bez wykonania
```

```bash
python -m pytest tests/test_stage12_pipeline.py -v
```

| Test | Co sprawdza |
|------|-------------|
| `test_pipeline_state_file` | `pipeline_state.json` tworzony po każdym kroku |
| `test_skip_scan_flag` | `--skip-scan` pomija krok 1, używa istniejącego CSV |
| `test_skip_train_flag` | `--skip-train` pomija kroki 2-3 |
| `test_dry_run` | `--dry-run` drukuje plan bez uruchamiania |
| `test_full_smoke` | Pełny pipeline na syntetycznych danych z MockDinoBackbone |

**Gotowe gdy:** `test_full_smoke` przechodzi; generuje `comparison_report.html`.

---

## Faza 6 — Uruchomienie na prawdziwych danych 🔌

### Zadanie 6.1 — Skan i weryfikacja danych

**Wymaga:** zamontowany dysk `Z:`

```bash
python src/scan_labels.py \
    --image-dir "Z:/Photo/Otolithes/HER/Processed" \
    --excel     data/analysisWithOtolithPhoto.xlsx \
    --output    data/labels_combined.csv
```

**Sprawdzić w raporcie skanu:**
- [ ] Unikalne tokeny parts[4]: powinno być tylko `{'Embedded', 'NotEmbedded'}`
- [ ] Unikalne tokeny parts[5]: zapisz co się pojawia (np. `Sharpest`, `WithoutPostproc`)
- [ ] Liczba Embedded ≈ 7 372 (zgodna z bieżącym `labels.csv`)
- [ ] Liczba sierot not-embedded (not-embedded bez odpowiednika w Excel)
- [ ] Rozkład wiekowy embedded ≈ rozkład not-embedded

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/labels_combined.csv')
leak = df.groupby('neutral_fish_key')['split'].nunique()
assert (leak > 1).sum() == 0, 'PRZECIEKI!'
pivot = df.groupby(['neutral_fish_key','preprocessing_type'])['split'].first().unstack()
mismatch = pivot.dropna().apply(lambda r: r['Embedded'] != r['NotEmbedded'], axis=1).sum()
assert mismatch == 0, 'NIEZGODNE SPLITY!'
print(df.groupby(['preprocessing_type','split']).size().unstack())
"
```

**Gotowe gdy:** oba asserty przechodzą; raport skanu wygląda sensownie.

---

### Zadanie 6.2 — Pełny pipeline treningowy 🔌

```bash
python scripts/run_pipeline.py \
    --image-dir  "Z:/Photo/Otolithes/HER/Processed" \
    --excel      data/analysisWithOtolithPhoto.xlsx \
    --output-dir outputs/ \
    --config-embedded     configs/config_embedded.yaml \
    --config-not-embedded configs/config_not_embedded.yaml \
    --skip-scan   # jeśli 6.1 już wykonane
```

**Monitorowanie treningu:**
```bash
# W osobnym terminalu:
tail -f logs/embedded/train.log
tail -f logs/not_embedded/train.log
```

**Gotowe gdy:** `outputs/comparison_report.html` istnieje i zawiera:
- [ ] Krzywe loss dla obu modeli
- [ ] Tabelę cross-ewaluacji (4 warunki)
- [ ] Karty z kropkami przyrostów (10 best + 10 worst)
- [ ] Brak błędów JS w przeglądarce

---

## Checklist — kolejność zadań

```
Faza 0
  [ ] 0.1  src/config.py — profile_axis + IncrementSamplesConfig
  [ ] 0.2  configs/config.yaml — nowe pola
  [ ] 0.3  configs/config_embedded.yaml + config_not_embedded.yaml

Faza 1
  [ ] 1.1  src/scan_labels.py — parse_filename + build_combined_labels
  [ ] 1.2  src/scan_labels.py — blok CLI + argparse

Faza 2
  [ ] 2.1  src/candidates.py — profile_axis + pionowy profil

Faza 3
  [ ] 3.1  src/visualization.py — load_original + draw_increment_card + select_top_k

Faza 4
  [ ] 4.1  src/comparison_report.py — build_comparison_report + compute_metrics

Faza 5
  [ ] 5.1  scripts/run_pipeline.py — orkiestrator + pipeline_state.json

Faza 6 🔌
  [ ] 6.1  Skan na prawdziwych danych — weryfikacja raportu
  [ ] 6.2  Pełny pipeline treningowy
```

---

## Nowe pliki testowe do stworzenia

| Plik | Faza | Testy |
|------|------|-------|
| `tests/test_stage9_scan_labels.py` | 1 | 8 testów |
| `tests/test_stage10_visualization.py` | 3 | 5 testów |
| `tests/test_stage11_comparison_report.py` | 4 | 5 testów |
| `tests/test_stage12_pipeline.py` | 5 | 5 testów |

Łącznie: 23 nowe testy. Po zakończeniu wszystkich faz:
```bash
python -m pytest   # wszystkie 152 + 23 = ~175 testów musi być zielonych
```