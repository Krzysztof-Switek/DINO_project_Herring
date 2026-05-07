# Dane wejściowe — podsumowanie

Data weryfikacji: 2026-05-07  
Skrypt: `src/scan_labels.py` + `python -c "..."` (weryfikacja asertywna)

---

## 1. Źródło danych

| Element | Wartość |
|---------|---------|
| Katalog zdjęć | `Z:/Photo/Otolithes/HER/Processed` |
| Plik metadanych | `data/analysisWithOtolithPhoto.xlsx` |
| Wygenerowane CSV | `data/labels_combined.csv` (18 727 wierszy) |
| | `data/labels_embedded.csv` (9 628 wierszy) |
| | `data/labels_not_embedded.csv` (9 099 wierszy) |

---

## 2. Format nazwy pliku

```
{ROK}_{KAMPANIA}_{GATUNEK}_{LOKALIZACJA}_{TYP}_{POSTPROC}_FishIndex{N}_Single{N}_{STRONA}.jpg
  [0]    [1]       [2]        [3]           [4]    [5]          [6]          [7]      [8]
```

Unikalne wartości zaobserwowane po skanowaniu:

| Pozycja | Warianty |
|---------|---------|
| `[4]` typ preparacji | `Embedded`, `NotEmbedded` |
| `[5]` postprocessing | `Sharpest`, `WithoutPostproc` |

---

## 3. Liczba zdjęć na dysku

| Typ preparacji | Zdjęcia na dysku | Z metadanymi (labeled) | Sieroty (brak metadanych) |
|----------------|-----------------|------------------------|--------------------------|
| **Embedded**     | 9 628 | 7 837 | ~307 + 1 484 wiek=-9 |
| **NotEmbedded**  | 9 099 | 4 924 | **4 175** |
| **Razem**        | **18 727** | **12 761** | **4 482** |

> **Uwaga:** `~307` Embedded orphans to zdjęcia z nowszych kampanii jeszcze nie wprowadzonych do bazy Excel.  
> **1 484** Embedded zdjęć ma metadane w Excel, ale `Wiek = -9` (nieznany wiek) — wykluczone z treningu.

---

## 4. Przyczyna sierot NotEmbedded (4 175 zdjęć)

Excel (`analysisWithOtolithPhoto.xlsx`) zawiera **wyłącznie** zdjęcia Embedded (14 498 wierszy).  
Dopasowanie odbywa się przez `neutral_fish_key`:

```
neutral_fish_key = "{ROK}_{KAMPANIA}_{GATUNEK}_{LOKALIZACJA}_{FishIndex{N}}"
```

Zdjęcia NotEmbedded ryb, których kampanie **nie są jeszcze w Excelu**, trafiają jako sieroty.  
~46% zdjęć NotEmbedded to sieroty — prawdopodobnie nowsze kampanie (2024+) bez wprowadzonych danych biologicznych.

---

## 5. Podział na splity (tylko labeled, wiek ≥ 0)

| Typ | train | val | test | Razem labeled |
|-----|------:|----:|-----:|-------------:|
| Embedded | 5 488 | 1 188 | 1 161 | **7 837** |
| NotEmbedded | 3 498 | 737 | 689 | **4 924** |
| **Razem** | **8 986** | **1 925** | **1 850** | **12 761** |

Podział na poziomie ryby (`neutral_fish_key`), stratyfikowany wiekiem, seed=42.

---

## 6. Rozkład wiekowy (labeled)

| Typ | Min wiek | Max wiek | Mediana | N zdjęć |
|-----|----------|----------|---------|---------|
| Embedded | 0 | 16 | 4.0 | 7 837 |
| NotEmbedded | 0 | 15 | 4.0 | 4 924 |

Klasy wiekowe: 0–16 (17 klas), głowica ordynalna daje K-1 = 16 logitów.  
Zdjęcia w wieku 0: **1 596** (wiek=0 to realny wiek, nie brakująca wartość).

---

## 7. Weryfikacja spójności splitów

```
Przecieki (ryba w >1 splicie):         0  ✅
Ryby z obu typów w różnych splitach:   0  ✅
Ryby z obu typami (Emb + NotEmb):   2 516
```

Te 2 516 ryb jest kluczowe dla **cross-ewaluacji** — ten sam zestaw testowy dla obu modeli gwarantuje rzetelne porównanie.

---

## 8. Co jest wykluczone z treningu

| Powód wykluczenia | Liczba zdjęć |
|-------------------|-------------|
| Sieroty (brak metadanych w Excel) | 4 482 |
| Wiek = -9 (nieznany) | 1 484 |
| Typ otolitu `Raw Pair`, `LowQuality`, `Wrong`, `Broken` | usunięte na etapie Excela |
| **Razem wykluczone** | **~5 966** |
| **Użyte do treningu/ewaluacji** | **12 761** |

---

## 9. Konfiguracja treningu

| Parametr | Wartość |
|----------|---------|
| `use_metadata` | `false` — model uczy się wyłącznie z obrazu |
| `num_age_classes` | 17 (klasy 0–16) |
| `profile_axis` | `vertical` — pionowa oś profilu ważności |
| Embedded checkpoint | `checkpoints/embedded/best.pt` |
| NotEmbedded checkpoint | `checkpoints/not_embedded/best.pt` |
| Epochs | 50 |
| Backbone | `dinov2_vits14` (patch_size=14, image_size=518) |
