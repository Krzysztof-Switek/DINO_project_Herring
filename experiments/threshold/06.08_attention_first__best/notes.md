# Notatki — laboratorium dekodowania CORAL

Dump: `C:\Users\kswitek\Documents\DINO_project_Herring\experiments\logits\06.08_attention_first__best`  ·  K = 17  ·  n val = 1117, n test = 1105

Wszystkie reguly wybrane **wylacznie na val**; test policzony raz, na koncu.

**Skala celu:** val = `rings(-1 for Q1)`, test = `rings(-1 for Q1)`.
Przy skali `rings(-1 for Q1)` model przewiduje liczbe widocznych pierscieni, a nie
zapisany wiek. Exact / MAE / +-1 sa **niezmiennicze** wzgledem tej zamiany, ale
tabela per wiek nie jest: ryba z Q1 lezy o klase nizej. Do porownan z zapisanym
wiekiem trzeba dodac offset sezonowy (`src.report_common.rebase_to_recorded`).

## Dekodery

| dekoder | split | exact | macro exact | +-1 rok | MAE | bias |
|---|---|---:|---:|---:|---:|---:|
| production  count@0.5 | val | **53.63 %** | 31.62 % | 84.69 % | 0.7422 | -0.076 |
| tuned global count@0.485 | val | **54.16 %** | 35.98 % | 84.51 % | 0.7341 | -0.018 |
| per-boundary thresholds | val | **59.80 %** | 40.62 % | 84.60 % | 0.6858 | -0.056 |
| argmax P(y=k) | val | **45.75 %** | 27.31 % | 81.29 % | 0.8684 | -0.184 |
| CEILING optimal partition of g | val | **60.43 %** | 41.06 % | 84.87 % | 0.6732 | -0.039 |
| optimal partition (MAE objective) | val | **60.25 %** | 40.04 % | 85.23 % | 0.6688 | -0.049 |
| per-campaign optimal partition | val | **62.49 %** | 51.16 % | 85.23 % | 0.6562 | -0.024 |
| shared partition + per-campaign shift of g | val | **60.43 %** | 41.06 % | 84.87 % | 0.6732 | -0.039 |
| production  count@0.5 | test | **53.85 %** | 29.14 % | 86.06 % | 0.6968 | -0.090 |
| tuned global count@0.485 | test | **52.94 %** | 28.88 % | 85.97 % | 0.7077 | -0.029 |
| per-boundary thresholds | test | **55.75 %** | 30.77 % | 86.06 % | 0.6824 | -0.076 |
| argmax P(y=k) | test | **46.24 %** | 23.54 % | 82.90 % | 0.8190 | -0.214 |
| val-optimal partition applied to test | test | **55.38 %** | 30.55 % | 86.15 % | 0.6860 | -0.058 |
| val-optimal partition (MAE objective) | test | **55.29 %** | 30.35 % | 86.33 % | 0.6824 | -0.069 |
| per-campaign partition (fitted on val) | test | **56.11 %** | 37.94 % | 86.43 % | 0.6959 | -0.006 |
| shared partition + per-campaign shift (fitted on val) | test | **55.38 %** | 30.55 % | 86.15 % | 0.6860 | -0.058 |

## Sufit — i czego on NIE ogranicza

Optymalny podzial osi `g` na val daje **60.43 %**.
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
| val | 45.75 % | 73.50 % | 84.42 % |
| test | 46.24 % | 74.39 % | 85.16 % |

**Zmierzony** udzial przypadkow, w ktorych drugi kandydat jest sasiadem
pierwszego: **93.30 %** (test). To NIE jest wlasnosc strukturalna — przy
nierownych odstepach progow `P(y=k)` nie musi byc unimodalne. Im blizej 100 %,
tym mocniej top-2 zbliza sie do Acc+-1, i tym scislej ograniczone jest to, co
jakikolwiek reranking kandydatow moze odzyskac: wylacznie bledy o 1 rok.

| split | 25 % najpewniejszych | 50 % | 75 % | 100 % |
|---|---:|---:|---:|---:|
| val | 67.74 % | 53.58 % | 49.22 % | 45.75 % |
| test | 73.19 % | 56.88 % | 50.72 % | 46.24 % |

## Poziom ryby (srednia `g` po zdjeciach jednej ryby)

| regula | n ryb | exact | +-1 rok | MAE |
|---|---:|---:|---:|---:|
| val_production | 588 | 54.08 % | 85.03 % | 0.7228 |
| val_tuned_global | 588 | 55.44 % | 85.88 % | 0.7041 |
| val_per_boundary | 588 | 58.50 % | 85.71 % | 0.6837 |
| val_argmax | 588 | 45.92 % | 81.80 % | 0.8537 |
| val_optimal_partition | 588 | 58.16 % | 85.71 % | 0.6871 |
| val_optimal_partition_mae | 588 | 58.16 % | 85.71 % | 0.6786 |
| val_per_campaign | 588 | 60.54 % | 85.54 % | 0.6633 |
| val_campaign_offset | 588 | 58.16 % | 85.71 % | 0.6871 |
| test_production | 578 | 57.79 % | 88.41 % | 0.6246 |
| test_tuned_global | 578 | 57.61 % | 88.58 % | 0.6280 |
| test_per_boundary | 578 | 61.59 % | 88.24 % | 0.5986 |
| test_argmax | 578 | 48.96 % | 84.95 % | 0.7578 |
| test_optimal_partition | 578 | 61.42 % | 87.72 % | 0.6055 |
| test_optimal_partition_mae | 578 | 61.07 % | 87.37 % | 0.6125 |
| test_per_campaign | 578 | 59.17 % | 86.85 % | 0.6522 |
| test_campaign_offset | 578 | 61.42 % | 87.72 % | 0.6055 |

## Wybor reguly — 2-krotna CV wewnatrz val (podzial po rybie)

Reguly roznia sie pojemnoscia o rzad wielkosci, wiec porownanie na danych, na
ktorych byly dopasowane, rankinguje je po liczbie parametrow. Skurcz val->test
odpowiedzialby na to pytanie, ale wydalby zbior testowy na decyzje selekcyjna.
Podzial po **rybie**, zeby dwa zdjecia tej samej ryby nie trafily po obu stronach.

| regula | parametry | exact out-of-fold |
|---|---|---:|
| production | 0 | 53.62 % |
| tuned_global | 1 | 53.98 % |
| per_boundary | K-1 = 16 | 54.25 % |
| argmax | 0 | 45.74 % |
| optimal_partition | K-1 = 16 | 53.62 % |
| optimal_partition_mae | K-1 = 16 (cel MAE) | 54.61 % |
| per_campaign | (K-1) x kampanie = 48 | 53.53 % |
| campaign_offset | 16 + 1 na kampanie | 53.62 % |


## Gorna granica dla rerankera kandydatow

Ile dalby reranker, ktory ZAWSZE wybiera poprawny wiek z okna +-r wokol
zdekodowanego. Liczone wokol realnego dekodera, nie wokol `argmax` — okno
wycentrowane na slabszym estymatorze zanizalaby te granice.

| regula | +-0 (exact) | +-1 | +-2 |
|---|---:|---:|---:|
| val_production | 53.63 % | 84.69 % | 93.91 % |
| val_tuned_global | 54.16 % | 84.51 % | 94.09 % |
| val_per_boundary | 59.80 % | 84.60 % | 93.73 % |
| val_argmax | 45.75 % | 81.29 % | 93.20 % |
| val_optimal_partition | 60.43 % | 84.87 % | 93.91 % |
| val_optimal_partition_mae | 60.25 % | 85.23 % | 93.91 % |
| val_per_campaign | 62.49 % | 85.23 % | 93.64 % |
| val_campaign_offset | 60.43 % | 84.87 % | 93.91 % |
| test_production | 53.85 % | 86.06 % | 95.11 % |
| test_tuned_global | 52.94 % | 85.97 % | 95.11 % |
| test_per_boundary | 55.75 % | 86.06 % | 94.66 % |
| test_argmax | 46.24 % | 82.90 % | 94.57 % |
| test_optimal_partition | 55.38 % | 86.15 % | 94.66 % |
| test_optimal_partition_mae | 55.29 % | 86.33 % | 94.84 % |
| test_per_campaign | 56.11 % | 86.43 % | 93.94 % |
| test_campaign_offset | 55.38 % | 86.15 % | 94.66 % |

## Dyscyplina selekcji — ktora liczba obowiazuje

Regula wybrana: optimal partition (MAE objective) (`optimal_partition_mae`).
Podstawa wyboru: out-of-fold exact accuracy, 2-fold by fish, inside val.

**Wynik tego etapu, na tescie, jeden raz:**
- per obraz: exact **55.29 %**, MAE 0.6824, bias -0.069
- per ryba: exact **61.07 %**, MAE 0.6125, n = 578

To jej wynik na tescie jest wynikiem tego etapu. Jesli inna regula wypadla na
tescie lepiej, ta informacja pochodzi z testu i **nie wolno** jej uzyc do wyboru —
inaczej stroimy na zbiorze testowym. Pozostale wiersze testowe sa raportowane
wylacznie dla pelnego obrazu.

Wybrany prog globalny: **0.485** (produkcja: 0,5).
