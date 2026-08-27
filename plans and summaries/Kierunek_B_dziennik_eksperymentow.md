# Kierunek B — lokalizacja przyrostów: dziennik eksperymentów

Żywa mapa prób: **cel → próba → wynik/efekt/dlaczego → następna metoda (z literatury)**. Aktualizować przy
każdej próbie, żeby zawsze było wiadomo, co zrobiliśmy, co zadziałało i dlaczego próbujemy czegoś innego.

## Cel
Pokazać PER-PRZYROST, gdzie model „widzi" roczne przyrosty wzdłuż osi odczytu, tak by ich liczba zgadzała się
z wiekiem — mając (na razie) **tylko etykietę wieku** (słaby nadzór), i **NIE degradując** dokładnego werdyktu
wieku (CORAL). Docelowo w karcie: czerwone punkty = ostateczne przyrosty (liczba = wiek), żółte = kandydaci.

## Próba 1 — prior wariancji promieniowej (`mil_radial_spread_loss`) — 2026-07-12 — ❌ PORAŻKA
- **Idea:** dołożyć do straty MIL człon rozpychający aktywne patche wzdłuż promienia (żeby separowały się w
  pierścienie); cel wariancji promieniowej rosnący z wiekiem (`0.10 + 0.03·wiek`).
- **Wynik:** test MAE 0.789 → **0.838**, Bias −0.039 → −0.14, early-stop 37 → **17 epok**, mapy uwagi rozmyte/
  bez sensu. Szczegóły i dowód: `12.07_CORAL_diagnoza_problemu.md`.
- **Dlaczego zawiodło:**
  1. Cel wariancji jest **sztuczny** — niezwiązany z realnym położeniem przyrostów → rozmazuje mapę.
  2. Gradient wpływa na **współdzielony backbone** (po odmrożeniu) i psuje cechy dla CORAL.
- **Lekcja:** lokalizacji **nie wolno sprzęgać** z backbonem głowicy wieku; prior musi wynikać z danych,
  nie z arbitralnej heurystyki.
- **Status kodu:** usunięte w całości (bez martwego kodu), 2026-07-12.

## Przegląd literatury (metody z półki — nie zgadujemy)
- **Otolity, nadzorowana lokalizacja (standard domenowy):** detekcja / segmentacja przyrostów wzdłuż osi
  odczytu — Mask R-CNN + U-Net. Ref: *Fish age reading using DL for object-detection and segmentation*, ICES
  JMS 2024 (https://academic.oup.com/icesjms/article/81/4/687/7614790). Najbardziej wiarygodne, ale wymaga
  adnotacji pierścieni/osi.
- **Interpretowalność werdyktu wieku (nie lokalizacja):** *Explaining decisions of DNNs used for fish age
  prediction* (Grad-CAM/saliency), https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7304622/ — wyjaśnia CO widzi
  CORAL; u nas realizują to Grad-CAM + uwaga CLS w karcie 2-głowicowej.
- **Słaby nadzór samą liczbą (crowd counting = najbliższy analog):** mapa gęstości, całka ≈ liczba, piki =
  obiekty. Kluczowe: samo regresowanie całki do liczby zawodzi → potrzeba **strat spójności/kompozycji**:
  - Wan & Chan, *A Generalized Loss Function for Crowd Counting and Localization*, CVPR'21
    (http://visal.cs.cityu.edu.hk/static/pubs/conf/cvpr21-generalizedloss-www.pdf)
  - *Towards using count-level weak supervision for crowd counting*, Pattern Recognition
    (https://www.sciencedirect.com/science/article/abs/pii/S0031320320304192)
  - *Optimal Transport Minimization — Crowd Localization on Density Maps*, CVPR'23
    (https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Optimal_Transport_Minimization_Crowd_Localization_on_Density_Maps_for_Semi-Supervised_CVPR_2023_paper.pdf)
  - *Deeply-Supervised Density Regression for Automatic Cell Counting* (https://arxiv.org/pdf/2011.03683)
- **Klasyka słojów (analog dendro):** profil intensywności + detekcja pików / szablony B-spline (Troadec —
  deformable template; Dendro-AutoCount — pith + peak analysis). U nas dostępne jako interaktywny reference
  OpenCV (sekcja H raportu).

## Kandydaci na Próbę 2 (ranking + uzasadnienie)
1. **Twarda zasada niezależnie od metody: ODSPRZĘGNĄĆ lokalizację od backbone wieku** — stop-gradient z głowicy
   lokalizacyjnej do backbone (albo osobny model/etap). Naprawia bezpośrednio root cause Próby 1.
2. **Density-map counting ze stratą spójności liczby** (crowd-counting): mapa gęstości D ≥ 0, całka ≈ wiek,
   piki wzdłuż osi = przyrosty; dodać count-consistency (nie samą całkę — to zawodzi). Zostaje słaby nadzór.
3. **Fallback (najpewniejszy): nadzorowana detekcja przyrostów** — U-Net/keypoints na osi odczytu; wymaga
   adnotacji kilku–kilkunastu otolitów. Standard domenowy, gdy słaby nadzór nie wystarcza.

## Rekomendacja następnej próby
**Próba 2 = (1) odsprzęgnięcie od backbone + (2) density-map counting z count-consistency.**
Bramka akceptacji: **MAE CORAL nie może się pogorszyć** (≤ ~0.79). Jeśli słaby nadzór nie da sensownych,
policzalnych pików — przejść do (3) z małym zbiorem adnotacji.
