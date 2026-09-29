# Notatki — laboratorium dekodowania CORAL

Dump: `C:\Users\kswitek\Documents\DINO_project_Herring\experiments\logits\09.08_isolate_a__best`  ·  K = 17  ·  n val = 1117, n test = 1105

Wszystkie reguly wybrane **wylacznie na val**; test policzony raz, na koncu.

**Skala celu:** val = `recorded`, test = `recorded`.
Przy skali `rings(-1 for Q1)` model przewiduje liczbe widocznych pierscieni, a nie
zapisany wiek. Exact / MAE / +-1 sa **niezmiennicze** wzgledem tej zamiany, ale
tabela per wiek nie jest: ryba z Q1 lezy o klase nizej. Do porownan z zapisanym
wiekiem trzeba dodac offset sezonowy (`src.report_common.rebase_to_recorded`).

## Dekodery

| dekoder | split | exact | macro exact | +-1 rok | MAE | bias |
|---|---|---:|---:|---:|---:|---:|
| production  count@0.5 | val | **46.37 %** | 32.18 % | 83.62 % | 0.8263 | -0.135 |
| tuned global count@0.505 | val | **46.55 %** | 31.98 % | 83.35 % | 0.8272 | -0.159 |
| per-boundary thresholds | val | **51.75 %** | 39.06 % | 82.99 % | 0.7932 | -0.159 |
| argmax P(y=k) | val | **37.69 %** | 23.42 % | 79.23 % | 0.9687 | -0.270 |
| CEILING optimal partition of g | val | **52.46 %** | 44.00 % | 82.63 % | 0.7932 | -0.088 |
| optimal partition (MAE objective) | val | **51.75 %** | 38.99 % | 84.06 % | 0.7637 | -0.074 |
| per-campaign optimal partition | val | **58.64 %** | 55.62 % | 82.63 % | 0.7153 | -0.040 |
| shared partition + per-campaign shift of g | val | **54.70 %** | 46.83 % | 81.83 % | 0.7690 | -0.062 |
| production  count@0.5 | test | **43.71 %** | 27.19 % | 83.17 % | 0.8362 | -0.121 |
| tuned global count@0.505 | test | **43.89 %** | 27.64 % | 82.99 % | 0.8371 | -0.138 |
| per-boundary thresholds | test | **47.15 %** | 27.83 % | 82.35 % | 0.8172 | -0.133 |
| argmax P(y=k) | test | **38.10 %** | 25.08 % | 80.09 % | 0.9267 | -0.235 |
| val-optimal partition applied to test | test | **46.97 %** | 28.37 % | 82.53 % | 0.8235 | -0.038 |
| val-optimal partition (MAE objective) | test | **47.24 %** | 30.67 % | 83.44 % | 0.7946 | -0.043 |
| per-campaign partition (fitted on val) | test | **49.86 %** | 30.77 % | 82.81 % | 0.8018 | +0.016 |
| shared partition + per-campaign shift (fitted on val) | test | **49.59 %** | 31.28 % | 83.53 % | 0.7837 | -0.056 |

## Sufit — i czego on NIE ogranicza

Optymalny podzial osi `g` na val daje **52.46 %**.
Zadna zmiana progu, kalibracji ani progow per granica nie moze tego przebic —
wszystkie sa szczegolnym przypadkiem monotonicznego podzialu tej samej osi.
Liczba dopasowana na val jest optymistyczna; wiersz *val-optimal partition
applied to test* pokazuje, ile z tego zostaje.

**Dekoder warunkowany kampania lezy POZA ta rodzina** i moze byc wyzszy — widzi
informacje, ktorej jedna os nigdy nie miala (kampania polowu, obecna w nazwie
pliku). To nie sprzecznosc z sufitem, tylko jego zakres.

## Kandydaci i niepewnosc

| split | top-1 | top-2 | top-3 |
|---|---:|---:|---:|
| val | 37.69 % | 68.76 % | 83.53 % |
| test | 38.10 % | 69.68 % | 82.44 % |

**Zmierzony** udzial przypadkow, w ktorych drugi kandydat jest sasiadem
pierwszego: **94.57 %** (test). To NIE jest wlasnosc strukturalna — przy
nierownych odstepach progow `P(y=k)` nie musi byc unimodalne. Im blizej 100 %,
tym mocniej top-2 zbliza sie do Acc+-1, i tym scislej ograniczone jest to, co
jakikolwiek reranking kandydatow moze odzyskac: wylacznie bledy o 1 rok.

| split | 25 % najpewniejszych | 50 % | 75 % | 100 % |
|---|---:|---:|---:|---:|
| val | 46.95 % | 41.76 % | 40.38 % | 37.69 % |
| test | 42.03 % | 39.67 % | 38.16 % | 38.10 % |

## Poziom ryby (srednia `g` po zdjeciach jednej ryby)

| regula | n ryb | exact | +-1 rok | MAE |
|---|---:|---:|---:|---:|
| val_production | 588 | 47.96 % | 85.88 % | 0.7823 |
| val_tuned_global | 588 | 47.45 % | 85.88 % | 0.7874 |
| val_per_boundary | 588 | 52.72 % | 84.35 % | 0.7585 |
| val_argmax | 588 | 39.80 % | 81.29 % | 0.9184 |
| val_optimal_partition | 588 | 52.89 % | 84.52 % | 0.7551 |
| val_optimal_partition_mae | 588 | 52.55 % | 85.37 % | 0.7398 |
| val_per_campaign | 588 | 55.27 % | 85.37 % | 0.7092 |
| val_campaign_offset | 588 | 53.40 % | 84.86 % | 0.7347 |
| test_production | 578 | 43.94 % | 85.47 % | 0.7976 |
| test_tuned_global | 578 | 43.94 % | 85.64 % | 0.7941 |
| test_per_boundary | 578 | 49.65 % | 85.12 % | 0.7526 |
| test_argmax | 578 | 37.72 % | 80.28 % | 0.9256 |
| test_optimal_partition | 578 | 49.65 % | 84.60 % | 0.7526 |
| test_optimal_partition_mae | 578 | 48.96 % | 85.47 % | 0.7526 |
| test_per_campaign | 578 | 51.21 % | 84.26 % | 0.7526 |
| test_campaign_offset | 578 | 51.21 % | 84.43 % | 0.7422 |

## Wybor reguly — 2-krotna CV wewnatrz val (podzial po rybie)

Reguly roznia sie pojemnoscia o rzad wielkosci, wiec porownanie na danych, na
ktorych byly dopasowane, rankinguje je po liczbie parametrow. Skurcz val->test
odpowiedzialby na to pytanie, ale wydalby zbior testowy na decyzje selekcyjna.
Podzial po **rybie**, zeby dwa zdjecia tej samej ryby nie trafily po obu stronach.

| regula | parametry | exact out-of-fold |
|---|---|---:|
| production | 0 | 46.38 % |
| tuned_global | 1 | 46.29 % |
| per_boundary | K-1 = 16 | 47.18 % |
| argmax | 0 | 37.69 % |
| optimal_partition | K-1 = 16 | 46.73 % |
| optimal_partition_mae | K-1 = 16 (cel MAE) | 47.01 % |
| per_campaign | (K-1) x kampanie = 48 | 48.16 % |
| campaign_offset | 16 + 1 na kampanie | 47.26 % |


## Gorna granica dla rerankera kandydatow

Ile dalby reranker, ktory ZAWSZE wybiera poprawny wiek z okna +-r wokol
zdekodowanego. Liczone wokol realnego dekodera, nie wokol `argmax` — okno
wycentrowane na slabszym estymatorze zanizalaby te granice.

| regula | +-0 (exact) | +-1 | +-2 |
|---|---:|---:|---:|
| val_production | 46.37 % | 83.62 % | 93.46 % |
| val_tuned_global | 46.55 % | 83.35 % | 93.46 % |
| val_per_boundary | 51.75 % | 82.99 % | 92.66 % |
| val_argmax | 37.69 % | 79.23 % | 92.93 % |
| val_optimal_partition | 52.46 % | 82.63 % | 92.39 % |
| val_optimal_partition_mae | 51.75 % | 84.06 % | 93.46 % |
| val_per_campaign | 58.64 % | 82.63 % | 93.55 % |
| val_campaign_offset | 54.70 % | 81.83 % | 93.02 % |
| test_production | 43.71 % | 83.17 % | 93.57 % |
| test_tuned_global | 43.89 % | 82.99 % | 93.57 % |
| test_per_boundary | 47.15 % | 82.35 % | 93.12 % |
| test_argmax | 38.10 % | 80.09 % | 93.30 % |
| test_optimal_partition | 46.97 % | 82.53 % | 92.85 % |
| test_optimal_partition_mae | 47.24 % | 83.44 % | 93.57 % |
| test_per_campaign | 49.86 % | 82.81 % | 92.76 % |
| test_campaign_offset | 49.59 % | 83.53 % | 93.30 % |

## Dyscyplina selekcji — ktora liczba obowiazuje

Regula wybrana: per-campaign optimal partition (`per_campaign`).
Podstawa wyboru: out-of-fold exact accuracy, 2-fold by fish, inside val.

**Wynik tego etapu, na tescie, jeden raz:**
- per obraz: exact **49.86 %**, MAE 0.8018, bias +0.016
- per ryba: exact **51.21 %**, MAE 0.7526, n = 578

To jej wynik na tescie jest wynikiem tego etapu. Jesli inna regula wypadla na
tescie lepiej, ta informacja pochodzi z testu i **nie wolno** jej uzyc do wyboru —
inaczej stroimy na zbiorze testowym. Pozostale wiersze testowe sa raportowane
wylacznie dla pelnego obrazu.

Wybrany prog globalny: **0.505** (produkcja: 0,5).
