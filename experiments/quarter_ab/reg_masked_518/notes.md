# A/B korekty kwartalnej — glowica na zamrozonych cechach

Rezim: frozen backbone, head-only, cached CLS features.  Ziarna: [42, 7, 13, 21, 99, 123, 256, 512, 1024, 2048].  Epok: 60.  lr: 0.01 (wybrany na val, ten sam w obu ramionach).

Punktacja: both arms scored against RECORDED age; ON arm rebased by +1 for Q1.

Korekta kwartalna jest **wylacznie przeetykietowaniem celu**, wiec na cache'u cech
wszystkie inne zmienne sa trzymane dokladnie stale: te same cechy bit w bit, ta sama
architektura, ta sama inicjalizacja przy tym samym ziarnie. Porownanie jest **parowane**.

## Wynik (zbior testowy, wiek zapisany)

| ramie | exact | +-1 rok | MAE | macro exact | bias |
|---|---:|---:|---:|---:|---:|
| quarter_OFF | **37.25 %** ± 0.73 | 79.44 % | 1.0133 ± 0.0226 | 20.85 % | -0.395 |
| quarter_ON | **44.39 %** ± 0.40 | 80.89 % | 0.9163 ± 0.0073 | 25.06 % | -0.347 |

## Test parowany po ziarnach

- delta exact = **+7.14 pkt proc.** (sd 0.98)
- delta MAE = **-0.0970**
- ON wygrywa w **10/10** ziarnach
- p (dwustronne, parowane) = 2.6361106973574273e-09

Per ziarno: +7.96, +7.42, +7.24, +8.24, +7.78, +7.24, +7.87, +6.24, +5.07, +6.33

## Kontrola zbieznosci

Biegi, w ktorych val nadal sie poprawial na koncu budzetu: `{'quarter_OFF': [], 'quarter_ON': []}`. Pusto = budzet epok wystarczyl; jesli nie,
porownanie mierzy budzet, nie cel, i trzeba podniesc `--epochs`.

## Czego to NIE rozstrzyga

Backbone jest zamrozony. Produkcja odmraza go w epoce 6, a backbone majacy swobode
dostosowania sie do zmienionego celu moze zyskac wiecej niz sama glowica. To jest
czysta dolna granica z porzadna statystyka po ziarnach, nie zamiennik pelnego biegu.
