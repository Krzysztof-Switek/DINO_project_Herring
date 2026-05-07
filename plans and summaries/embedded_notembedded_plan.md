# Plan: Zautomatyzowany pipeline Embedded vs NotEmbedded

Data: 2026-05-07

---

## 1. Kontekst i cel

Na udziale sieciowym (`Z:/Photo/Otolithes/HER/Processed`) znajdują się zdjęcia otolit w dwóch
wariantach preparacji tych samych ryb:

- **Embedded** — otolit zalany żywicą (szlifowany przekrój)
- **NotEmbedded** — otolit bez zalewania, zdjęcie powierzchniowe

Cel: jeden komend uruchamia cały pipeline:
1. Przygotowanie etykiet (scan + split)
2. Dwa treningi (embedded, not-embedded)
3. Cztery warianty inferencji (2 normalne + 2 cross)
4. Zbiorczy raport porównawczy z pełnymi danymi treningowymi

Metodologicznie kluczowe: te same ryby lądują w tym samym splicie w obu eksperymentach
→ rzetelne porównanie na identycznym zestawie testowym.

---

## 2. Stan obecny bazy danych

### labels.csv (bieżący)

- **7 372 zdjęć**, wyłącznie Embedded
- Kolumny: `image_id, age, length_mm, weight_g, sex, population, subdivision, otolith_type, year, split`
- Podział: ~5 160 train / ~1 106 val / ~1 106 test (poziom ryby, stratyfikowany wiekiem)

### analysisWithOtolithPhoto.xlsx

- **14 498 wierszy** — wszystkie Embedded
- `Typ otolitu` ∈ {Left, Right, **Raw Pair**, LowQuality, Wrong, Broken}
- **Raw Pair** = zdjęcie pary otolit razem — tych plików **NIE MA** w katalogu ze zdjęciami
- Filtrujemy: `GOOD_TYPES = {"Left", "Right"}` (jak w `prepare_labels.py:31`)
- Mapowanie kolumn: `Wiek→age, Plec→sex, Klasa_dlugosci_mm→length_mm, Masa_g→weight_g,
  Populacja→population, Subdivision→subdivision, Rok→year, FilePath→image_id, Typ otolitu→otolith_type`

---

## 3. Schemat nazwy pliku — zweryfikowany

```
{YEAR}_{CAMPAIGN}_{SPECIES}_{LOCATION}_{TYPE}_{POSTPROC}_FishIndex{N}_Single{N}_{SIDE}.jpg
 [0]    [1]        [2]       [3]         [4]    [5]        [6]          [7]       [8]
```

| Poz. | Nazwa       | Przykłady                    | Uwagi                              |
|------|-------------|------------------------------|------------------------------------|
| [0]  | ROK         | 2022, 2023, 2024             | rok połowu                         |
| [1]  | KAMPANIA    | BIAS, BITS1q                 | kampania badawcza                  |
| [2]  | GATUNEK     | HER                          | zawsze HER (śledź)                 |
| [3]  | LOKALIZACJA | ZatokaGdanska, GlebiaGdanska | CamelCase, bez podkreślnika wewn.  |
| [4]  | TYP         | `Embedded` / `NotEmbedded`   | **klucz klasyfikacji**             |
| [5]  | POST-PROC   | Sharpest, withoutPostproc    | **ignorowany** w parsowaniu        |
| [6]  | FISH_INDEX  | FishIndex1, FishIndex43      | numer ryby                         |
| [7]  | SINGLE      | Single1, Single2             | numer zdjęcia otolitu              |
| [8]  | STRONA      | Left, Right                  | strona otolitu                     |

Przykłady rzeczywistych nazw (embedded, z `data/labels.csv`):
```
2022_BIAS_HER_ZatokaGdanska_Embedded_Sharpest_FishIndex2_Single1_Left.jpg
2023_BITS1q_HER_GlebiaGdanska_Embedded_Sharpest_FishIndex1_Single2_Right.jpg
```

---

## 4. neutral_fish_key — klucz unifikujący

`extract_fish_id()` (`prepare_labels.py:52`) obcina tylko `_Single{N}_{Side}`, zostawiając token
`Embedded`/`NotEmbedded` w kluczu → te same ryby mają różne fish_id. Potrzebny nowy klucz:

```python
neutral_fish_key = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}_{parts[6]}"
# → "2022_BIAS_HER_ZatokaGdanska_FishIndex2"
# identyczny dla Embedded i NotEmbedded tej samej ryby
```

---

## 5. Architektura pełnego pipeline'u

```
┌─────────────────────────────────────────────────────────────────┐
│  python scripts/run_pipeline.py [--skip-scan] [--skip-train]    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │  KROK 1: SCAN & LABEL           │
          │  scan_labels.py (importowalny)  │
          │  → data/labels_combined.csv     │
          │  → data/labels_embedded.csv     │
          │  → data/labels_not_embedded.csv │
          └────────────┬───────────┬────────┘
                       │           │
          ┌────────────▼──┐   ┌────▼──────────────┐
          │  KROK 2A      │   │  KROK 2B           │
          │  TRAIN        │   │  TRAIN             │
          │  Embedded     │   │  NotEmbedded       │
          │  → ckpt/emb/  │   │  → ckpt/not_emb/  │
          └────────┬──────┘   └──────┬─────────────┘
                   │                  │
          ┌────────▼──────────────────▼─────────────┐
          │  KROK 3: INFERENCJA (4 warianty)         │
          │  A→A : Model_Emb    na teście Emb        │
          │  B→B : Model_NotEmb na teście NotEmb     │
          │  A→B : Model_Emb    na teście NotEmb  ★  │
          │  B→A : Model_NotEmb na teście Emb     ★  │
          │  (★ = cross-ewaluacja)                   │
          └─────────────────┬───────────────────────┘
                            │
          ┌─────────────────▼───────────────────────┐
          │  KROK 4: ZBIORCZY RAPORT HTML            │
          │  outputs/comparison_report.html          │
          └──────────────────────────────────────────┘
```

### Uruchomienie

```bash
python scripts/run_pipeline.py \
    --image-dir "Z:/Photo/Otolithes/HER/Processed" \
    --excel     data/analysisWithOtolithPhoto.xlsx \
    --output-dir outputs/ \
    --train 0.70 --val 0.15 --seed 42

# Flagi pomijające ukończone kroki (restart po awarii):
#   --skip-scan    jeśli labels_combined.csv już istnieje
#   --skip-train   jeśli checkpointy już istnieją
```

---

## 6. Pliki do stworzenia / zmodyfikowania

| Plik | Akcja | Rola |
|------|-------|------|
| `src/scan_labels.py` | NOWY | Importowalny moduł: scan + parse + match + split → CSV |
| `src/visualization.py` | NOWY | Oryginalne obrazy z dysku + karty z kropkami przyrostów |
| `scripts/run_pipeline.py` | NOWY | Główny orkiestrator pipeline'u |
| `src/comparison_report.py` | NOWY | Zbiorczy raport HTML z metrykami obu modeli + kartami |
| `src/candidates.py` | MODYFIKACJA | Zmiana osi profilu na pionową (`mean(axis=1)`), parametr `profile_axis` |
| `configs/config_embedded.yaml` | NOWY | Konfiguracja treningu Embedded |
| `configs/config_not_embedded.yaml` | NOWY | Konfiguracja treningu NotEmbedded |
| `configs/config.yaml` | MODYFIKACJA | Dodanie `candidates.profile_axis`, `report.increment_samples.*` |

Bez zmian: `src/trainer.py`, `src/inference.py`, `src/model.py`, `src/dataset.py`,
`src/interpretation.py`, `scripts/prepare_labels.py`.

---

## 7. Szczegóły: `src/scan_labels.py`

### Kluczowe funkcje

**`parse_filename(name: str) -> dict | None`**
```python
def parse_filename(name):
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 9:
        return None
    type_token = parts[4].lower()
    if type_token == "embedded":
        preprocessing_type = "Embedded"
    elif type_token == "notembedded":
        preprocessing_type = "NotEmbedded"
    else:
        return None
    neutral_fish_key = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}_{parts[6]}"
    side = parts[8] if parts[8].lower() in {"left", "right"} else None
    return {"image_id": name, "preprocessing_type": preprocessing_type,
            "neutral_fish_key": neutral_fish_key, "side": side}
```

**`build_combined_labels(image_dir, excel_path, train, val, seed) -> pd.DataFrame`**
- Skanuje katalog (`scan_image_dir` z `prepare_labels.py:44`)
- Parsuje każdą nazwę przez `parse_filename()`
- Ładuje Excel → filtruje GOOD_TYPES → buduje lookup `neutral_fish_key → metadane`
- Dopasowuje not-embedded do metadanych; brak → `orphan=True`
- Wywołuje `assign_split_by_fish()` (`prepare_labels.py:71`) na połączonym zbiorze
- Zwraca DataFrame, drukuje raport (unikalne tokeny [4] i [5], liczby, rozkład wiekowy)

**Schemat `labels_combined.csv`:**

| Kolumna | Źródło |
|---------|--------|
| image_id | nazwa pliku |
| neutral_fish_key | parsowanie |
| preprocessing_type | parsowanie (`Embedded` / `NotEmbedded`) |
| age, length_mm, weight_g, sex, population, subdivision, otolith_type, year | Excel |
| split | `assign_split_by_fish()` — wspólny |
| orphan | True jeśli not-embedded bez metadanych |

---

## 8. Szczegóły: `scripts/run_pipeline.py`

```python
def main():
    # Krok 1: scan & label
    if not args.skip_scan:
        combined = build_combined_labels(image_dir, excel, train, val, seed)
        combined.to_csv("data/labels_combined.csv")
        combined[combined.preprocessing_type=="Embedded"]   .to_csv("data/labels_embedded.csv")
        combined[combined.preprocessing_type=="NotEmbedded"].to_csv("data/labels_not_embedded.csv")

    # Krok 2: treningi
    if not args.skip_train:
        run_training(config_embedded)      # → checkpoints/embedded/best.pt
        run_training(config_not_embedded)  # → checkpoints/not_embedded/best.pt

    # Krok 3: 4 × inferencja
    results = {}
    results["emb_on_emb"]         = run_inference(ckpt_emb,     labels_emb_test,     "outputs/emb_on_emb/")
    results["notemb_on_notemb"]   = run_inference(ckpt_notemb,  labels_notemb_test,  "outputs/notemb_on_notemb/")
    results["emb_on_notemb"]      = run_inference(ckpt_emb,     labels_notemb_test,  "outputs/cross_emb_on_notemb/")   # CROSS
    results["notemb_on_emb"]      = run_inference(ckpt_notemb,  labels_emb_test,     "outputs/cross_notemb_on_emb/")   # CROSS

    # Krok 4: raport
    build_comparison_report(results, training_logs, output="outputs/comparison_report.html")
```

---

## 9. Szczegóły: `src/comparison_report.py`

### Sekcje raportu HTML

**A. Statystyki zbioru danych**
- Liczba zdjęć per typ per split (tabela)
- Histogram rozkładu wiekowego: Embedded vs NotEmbedded (overlaid)
- Rozkład płci, lat, subdivisions
- Liczba sierot (not-embedded bez dopasowania w Excelu)

**B. Dane treningowe — oba modele**

Per model (Embedded, NotEmbedded):
- Krzywa straty: `train_loss` i `val_loss` per epoka (wykres liniowy)
- Harmonogram learning rate per epoka
- `val_MAE` per epoka (wykres liniowy) — zaznaczony best epoch
- Tabela podsumowująca: best epoch, best val_MAE, czas treningu, N parametrów modelu

**C. Metryki ewaluacyjne — 4 warunki**

Dla każdego z 4 wariantów (A→A, B→B, A→B cross, B→A cross):

| Metryka | Opis |
|---------|------|
| MAE | Mean Absolute Error (lata) — główna metryka |
| RMSE | Root Mean Square Error |
| Accuracy ±1 rok | % predykcji w odległości ≤1 roku od prawdy |
| Accuracy ±2 lata | % predykcji w odległości ≤2 lat od prawdy |
| R² | Współczynnik determinacji |
| Bias | Średni błąd (ujemny = model zaniża, dodatni = zawyża) |

Wykresy:
- **Scatter**: predicted vs actual age (4 panele — po jednym per warunek)
- **Histogram błędów**: (predicted − actual) — osobno dla każdego warunku
- **MAE per klasa wiekowa**: wykres słupkowy, klasy 0–16, per warunek
- **Macierz pomyłek**: 17×17 heatmapa predicted vs actual (per warunek)
- **Box plot**: rozkład błędów bezwzględnych per warunek (4 boxy obok siebie)

**D. Tabela cross-ewaluacji (podsumowanie)**

```
                      Test: Embedded    Test: NotEmbedded
Model: Embedded       MAE = X.XX yr     MAE = X.XX yr  ← CROSS
Model: NotEmbedded    MAE = X.XX yr  ←  MAE = X.XX yr
                      CROSS
```

Komentarz automatyczny: jeśli MAE_cross < 1.5 × MAE_own → modele generalizują dobrze.

**E. Informacje o modelu i konfiguracji**

- Backbone (np. dinov2_vits14), patch_size
- Łączna liczba parametrów, liczba trenowalnych
- Liczba epok freeze_backbone
- Optimizer, lr, scheduler
- Ścieżki checkpointów (best epoch)
- Data i czas uruchomienia pipeline'u

---

## 10. Wizualizacja przyrostów — projekt

### Cel i filozofia

Odwzorować sposób pracy techników: pionowa oś czytania od centrum otolitu ku najdalszemu brzegowi,
numerowane kropki na każdym przyroście, oryginalne zdjęcie z dysku (nie przetworzone przez model).

### Kluczowa poprawka (błąd w bieżącym kodzie)

`extract_radial_profile()` w `src/candidates.py:47` używa `mean(axis=0)` → profil **poziomy**.
Wszystkie zdjęcia są pionowe (dłuższa oś otolitu ≈ pionowa). Należy zmienić na `mean(axis=1)` →
profil pionowy (H_p,). Kontrolowane przez nowy parametr konfiguracyjny `candidates.profile_axis`.

### Kluczowa obserwacja o obrazach (z analizy przykładów)

| Cecha | Embedded | NotEmbedded |
|-------|----------|-------------|
| Technika | Przekrój poprzeczny otolitu w żywicy | Widok powierzchniowy otolitu |
| Tło | Czarne | Czarne |
| Otolit | Niebiesko-biały, półprzezroczysty | Biało-kremowy, nieprzezroczysty |
| Widoczne przyrosty | Wewnętrzne pierścienie (jak słoje drzewa) | Powierzchniowe grzbiety (jak poziomice) |
| Jądro (nucleus) | Widoczne jako jasna gwiazdka w centrum | Primordium na powierzchni |

Obie metody są standardowe w ichtiologii. Model uczy się różnych cech wizualnych — stąd konieczność osobnych treningów.

### Nowy moduł: `src/visualization.py`

**`load_original_image(image_id, image_dir) -> np.ndarray`**
- Ładuje oryginalne zdjęcie z dysku (`image_dir / image_id`) bez żadnego przetwarzania
- Używane zamiast odwrotnie znormalizowanego tensora modelu

**`extract_vertical_profile(importance_grid) -> np.ndarray`**
- `importance_grid.mean(axis=1)` → (H_p,) — profil pionowy
- (Zastępuje obecne `mean(axis=0)` dla pionowych zdjęć)

**`compute_dot_positions(profile, predicted_age, image_height, num_patches_h) -> list[int]`**
- Wykrywa N pików (N = predicted_age) w profilu pionowym
- Konwertuje indeksy patchy → piksele oryginału:
  `y_pixel = (peak_idx + 0.5) * H_original / H_p`
- Zwraca listę y-współrzędnych (od góry obrazu = najdalszego brzegu)

**`save_increment_card(image_id, image_dir, dot_positions, predicted_age, true_age,
                       importance_grid, output_path)`**
- Panel A — oryginał + kropki:
  - **Oś czytania**: cienka pionowa linia `(220, 30, 30)` — ciemnoczerwona, 1 px,
    przez x = W_orig/2. Czerwień widoczna zarówno na czarnym tle jak i na białym/niebieskim
    otolicie (oba typy mają ciemne tło i jasny otolit).
  - **Kropki przyrostów**: filled circle `(255, 230, 0)` — żółty, r ≈ 8 px, z czarną
    obwódką 1 px. Żółty daje maksymalny kontrast na obu typach obrazów (embedded: niebiesko-biały
    otolit na czarnym tle; not-embedded: biało-kremowy otolit na czarnym tle). Numer kropki
    wewnątrz w kolorze czarnym.
  - **Niepełny przyrost**: pusty okrąg (żółty kontur bez wypełnienia) + etykieta "N+" obok,
    jeśli ostatni sigmoid modelu > 0.3
  - Predykcja i prawdziwy wiek w rogu obrazu — biały tekst z czarnym cieniem
- Panel B — overlay heatmapy + te same kropki (kontekst gdzie model patrzy)
- Panel C — pionowy profil ważności (wykres: oś Y = pozycja w obrazie, oś X = ważność)
  - Poziome przerywane linie na każdym piku + numer
  - Intuicyjne: pozycja na wykresie odpowiada pozycji na obrazie obok

### Selekcja zdjęć do raportu

Kryterium: błąd = `|predicted_age − true_age|` na zbiorze testowym, po jednym zdjęciu per ryba
(jeśli ryba ma Left i Right — wybieramy to z mniejszym błędem lub oba jeśli błąd równy).

- **10 najlepszych predykcji** (błąd najmniejszy, ideał = 0) → sekcja "Trafione"
- **10 najgorszych predykcji** (błąd największy) → sekcja "Błędne — analiza"
- Dla cross-ewaluacji: analogicznie 10 + 10 z każdego wariantu (A→B i B→A)

Parametry w konfiguracji:
```yaml
report:
  increment_samples:
    top_k_best: 10           # ile najlepszych predykcji z kropkami
    top_k_worst: 10          # ile najgorszych predykcji z kropkami
    annotate_all: false      # true = generuj dla wszystkich (wolno, ~godziny)
```

### Schemat karty HTML

```
┌──────────────────────────────────────────────────────────────────┐
│ FishIndex42 · Embedded · Predykcja: 7 lat · Prawda: 7 lat ✓      │
├────────────────────┬────────────────────┬────────────────────────┤
│  ORYGINAŁ + KROPKI │  HEATMAP + KROPKI  │  PROFIL PIONOWY        │
│                    │                    │  góra (brzeg)          │
│  ●1  (top → dół)  │  (JET overlay)     │  ───── ●1              │
│  │                 │  same dots         │  ─── ●2                │
│  ●2                │                    │  ──── ●3               │
│  │                 │                    │  ── ●4                 │
│  ●3...●7           │                    │  ─── ●5...             │
│                    │                    │  dół (centrum/jądro)   │
└────────────────────┴────────────────────┴────────────────────────┘
```

### Plan rozwojowy (opcja "interactive")

W przyszłości: karta HTML z możliwością zaznaczenia przez technika `✓ OK` / `✗ Błąd` / `+ Brakuje`
per kropka → zapis do JSON → dane do dalszego fine-tuningu lub ewaluacji jakości modelu.
Wymaga dodania lekkiego JS do raportu HTML — odłożone na późniejszy etap.

---

## 10. Struktura katalogów wyjściowych

```
outputs/
  emb_on_emb/
    predictions.csv
    predictions.json
    heatmaps/
    overlays/
    candidates/
  notemb_on_notemb/
    predictions.csv
    predictions.json
    heatmaps/
    overlays/
    candidates/
  cross_emb_on_notemb/
    predictions.csv        ← tylko predykcje, bez heatmap (cross)
  cross_notemb_on_emb/
    predictions.csv
  comparison_report.html   ← główny raport
```

---

## 11. Weryfikacja spójności splitów

```python
import pandas as pd
df = pd.read_csv("data/labels_combined.csv")

# Brak przecieków
leak = df.groupby("neutral_fish_key")["split"].nunique()
assert (leak > 1).sum() == 0, "PRZECIEKI: ryba w >1 splicie!"

# Identyczny split dla obu typów tej samej ryby
pivot = df.groupby(["neutral_fish_key","preprocessing_type"])["split"].first().unstack()
mismatch = pivot.dropna().apply(lambda r: r["Embedded"] != r["NotEmbedded"], axis=1).sum()
assert mismatch == 0, "NIEZGODNOŚĆ: różne splity dla tej samej ryby!"

print(df.groupby(["preprocessing_type","split"]).size().unstack())
```

---

## 12. Decyzja: tylko wiek jako sygnał treningowy (`use_metadata=False`)

### Kontekst pytania

Czy do treningu używać wyłącznie wieku, czy też dodatkowych parametrów biologicznych
(klasa długości, masa) dostępnych w Excelu? Bieżący default konfiguracji to `use_metadata=True`,
co oznacza że model otrzymuje długość, masę, płeć i populację jako cechy wejściowe.

### Trzy możliwe podejścia

| Podejście | Opis | Ocena |
|-----------|------|-------|
| **Metadata jako INPUT** (obecny default) | Model dostaje długość/masę jako gotową informację wejściową obok obrazu | Metodologicznie błędne — shortcut learning |
| **Metadata jako pomocniczy OUTPUT** (multitask) | Model musi *przewidzieć* długość z obrazu otolitu, uczy się cech wzrostu | Poprawne, lecz złożone |
| **Tylko wiek** ← wybrane | Model uczy się wyłącznie z obrazu, jedyna etykieta to wiek | Najczystsze metodologicznie |

### Dlaczego metadata jako INPUT jest błędna metodologicznie

Gdy długość/masa trafia jako cecha wejściowa, model może nauczyć się:
*"duży otolit → stara ryba"* — bez analizy żadnego pierścienia przyrostu.
To shortcut learning: model omija cel zadania (liczenie pierścieni) i uczy się
prostej korelacji morfometrycznej. Jest to dokładnie to, czego chcemy uniknąć.
Analogia: technik czytający otolit nie zna długości ryby — liczy pierścienie.
Model powinien robić to samo.

### Dlaczego nie multitask z długością jako OUTPUT

Artykuł *DeepOtolith* (Politikos et al., 2021, Fisheries Research) pokazuje, że
użycie długości jako **pomocniczego zadania wyjściowego** (model przewiduje wiek I długość
z obrazu) poprawia trafność z 64,4% → 69,2% — szczególnie dla starszych klas wiekowych.
To metodologicznie poprawne podejście. Jednak w naszym przypadku:

- DINOv2 z bogatymi cechami pretrenowanymi + CORAL + ~7 000 obrazów →
  overfitting nie jest głównym ryzykiem; dodatkowe zadanie nie jest konieczne
- Porównanie embedded vs not-embedded jest czystsze przy identycznej architekturze głowicy
- Opis projektu sprowadza się do jednego zdania: *"model przewiduje wiek wyłącznie
  z obrazu, bez żadnych dodatkowych pomiarów biologicznych"* — mocny argument
  dla biologów i recenzentów
- Multitask z długością jako OUTPUT pozostaje jako **eksperyment wariantowy** jeśli
  model będzie miał duże błędy na starszych klasach wiekowych (rzadkie dane)

### Decyzja i działanie

**`use_metadata: false`** we wszystkich konfiguracjach treningowych.

Jeśli zajdzie potrzeba regularyzacji (np. słabe wyniki na klasach wiekowych > 10 lat):
→ dodać długość jako pomocniczy OUTPUT (dodatkowa głowica regresyjna), nie jako INPUT.

### Literatura źródłowa

- Politikos, D.V. et al. (2021). *Automating fish age estimation combining otolith images
  and deep learning: The role of multitask learning.*
  Fisheries Research, 242, 105986.
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0165783621001612) |
  [GitHub DeepOtolith](https://github.com/dimpolitik/DeepOtolith)

- Shi, K. et al. (2021). *Tackling Ordinal Regression Problem for Heterogeneous Data:
  Sparse and Deep Multi-Task Learning Approaches.*
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8153254/) —
  MTL działa gdy zadania są skorelowane; brak korelacji szkodzi wynikom

- Cao, W. et al. (2020). *Rank consistent ordinal regression for neural networks
  with application to age estimation (CORAL).*
  Pattern Recognition Letters.
  [arXiv](https://arxiv.org/abs/1901.07884) — podstawa obecnej głowicy ordinalnej

- Greenland halibut otolith age prediction (PLOS One, 2022):
  [link](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0277244) —
  przykład prostego podejścia (wiek only) z dobrymi wynikami

---

## 13. Znane ryzyka i otwarte kwestie

| Kwestia | Status |
|---------|--------|
| Czy token not-embedded to dokładnie `NotEmbedded` (bez podkreślnika)? | Weryfikacja po 1. uruchomieniu — skrypt drukuje unikalne tokeny [4] |
| Jaki jest dokładny token post-proc dla not-embedded (parts[5])? | Skrypt drukuje unikalne tokeny [5] — informacyjnie |
| Liczba sierot not-embedded (brak w Excelu) | Raport skryptu |
| Czy rozkład wiekowy embedded ≈ not-embedded? | Sekcja A raportu HTML |
| Bug przecieków w obecnym labels.csv (FishIndex2 w train i val) | Istniejący, nie blokuje — nowy pipeline generuje labels od nowa |