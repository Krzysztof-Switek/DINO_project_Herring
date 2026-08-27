# Głowice modelu — jak działają CORAL i MIL

Wyjaśnienie na dwóch poziomach: **dla laika** (analogia) i **profesjonalnie, ale prostym językiem**
(bez matematycznego żargonu). Oparte na `src/model.py`.

---

## W jednym zdaniu

Model to wspólny „wzrok" (backbone DINOv2), który patrzy na zdjęcie otolitu, a na jego wyjściu siedzą **dwie
głowice** czytające ten sam obraz na dwa różne sposoby:

- **CORAL** odpowiada na pytanie **„ile lat?"** — to jest werdykt (liczba).
- **MIL** odpowiada na pytanie **„gdzie są przyrosty?"** — to jest lokalizacja (mapa punktów).

To dwie NIEZALEŻNE głowice. Dlatego werdykt wieku może być bardzo dobry, nawet gdy lokalizacja jest słaba
(i odwrotnie).

---

## Co to jest „backbone" i „patch" (wspólne dla obu głowic)

Zanim głowice zaczną liczyć, zdjęcie przechodzi przez **DINOv2** — dużą, wstępnie wytrenowaną sieć, która
„rozumie" obrazy. Dzieli ona zdjęcie (518×518 px) na **siatkę 37×37 = 1369 kwadracików („patchy")**, po 14×14 px
każdy, i dla każdego patcha oraz dla całego obrazu tworzy **wektor cech** (zestaw liczb opisujący, co tam widać):

- **token CLS** — jeden wektor opisujący **cały obraz naraz** (streszczenie). Używa go CORAL.
- **tokeny patchy** — po jednym wektorze **na każdy kwadracik**. Używa ich MIL.

---

## 1. CORAL — głowica wieku (werdykt)

### Dla laika
Wyobraź sobie doświadczoną osobę, która patrzy na przekrój pnia drzewa i mówi „to ma jakieś 6 lat" — **nie
licząc słojów po kolei**, tylko z ogólnego wyglądu (rozmiar, gęstość, dojrzałość). CORAL działa podobnie:
patrzy na **całość** otolitu i podaje wiek. Nie wskazuje palcem każdego rocznego przyrostu — po prostu zna liczbę.

Sprytna sztuczka: zamiast zgadywać wiek „od zera", CORAL odpowiada na **ciąg pytań tak/nie**:
„starszy niż 0?", „starszy niż 1?", „starszy niż 2?" … i **liczy, ile razy padło TAK**. Jeśli TAK padło 6 razy →
wiek = 6. Pytania są **uporządkowane**, więc nie może odpowiedzieć bez sensu („starszy niż 5, ale nie starszy
niż 3") — to gwarantuje spójne odpowiedzi.

### Profesjonalnie, ale prosto
- Bierze **token CLS** (streszczenie całego obrazu) → Dropout → warstwa liniowa → **jedna liczba `g`**
  („surowy wynik dojrzałości").
- Ma **16 progów** θ₀ < θ₁ < … < θ₁₅ (bo mamy 17 klas wieku, 0–16 → 16 granic między nimi). Progi są tak
  skonstruowane, że są **zawsze rosnące** (θ = baza + skumulowane dodatnie przyrosty).
- Dla każdego progu liczy: `logit_k = g − θ_k`. Ponieważ progi rosną, prawdopodobieństwa maleją:
  P(wiek>0) ≥ P(wiek>1) ≥ … To jest **rank-consistency** (Cao i in. 2020) — odpowiedzi „tak/nie" nigdy się nie
  przeczą.
- **Dekodowanie wieku**: policz, ile z tych prawdopodobieństw jest > 0.5. Tyle wynosi przewidziany wiek.
- **Uczenie**: dla każdego z 16 pytań osobny błąd „tak/nie" (binary cross-entropy). Cel: dla ryby w wieku k
  pierwsze k pytań ma odpowiedź TAK, reszta NIE.

**Dlaczego ordinal, a nie zwykła klasyfikacja albo regresja?**
- Zwykła klasyfikacja traktuje wiek 5 i 6 jako *zupełnie różne etykiety* — nie wie, że 6 jest „bliżej" 5 niż 15.
- Zwykła regresja (jedna liczba) gubi niepewność i słabo radzi sobie z dyskretnymi klasami.
- Ordinal (CORAL) **wie, że wiek jest uporządkowany** — pomyłka o 1 rok jest „mało zła", o 5 lat „bardzo zła",
  i to naturalnie wchodzi w uczenie.

### Na jakiej podstawie CORAL rozpoznaje wiek?

Mechanicznie: wiek = monotoniczna funkcja iloczynu `wagi · CLS`, gdzie CLS to wektor (384 liczby) streszczający
**cały obraz**. Czyli jedna nauczona warstwa zamienia „globalny odcisk" otolitu na jedną liczbę. **Nie ma tu
liczenia pierścieni.**

Kluczowe: **nie mówimy modelowi, czego ma szukać.** W treningu wsteczna propagacja stroi wagi głowicy i sam
backbone tak, by ta liczba **korelowała z prawdziwym wiekiem**. Model **sam odkrywa**, które cechy wizualne idą
w parze z wiekiem — najprawdopodobniej:
- **rozmiar otolitu** (starsza ryba → większy otolit — bardzo silna korelacja),
- **liczba, rozstaw i kontrast koncentrycznych pasm**,
- **tekstura, przezierność, szerokość zewnętrznego marginesu**.

To są **korelacje statystyczne znalezione przez sieć**, a nie reguła, którą jej wpisaliśmy. Bardzo prawdopodobne,
że CORAL mocno opiera się na **globalnym rozmiarze/teksturze**, a nie na policzeniu każdego przyrostu — dlatego
potrafi być trafny, **nie lokalizując pierścieni**. Ta liczba jest „zwinięta" w 384 wymiarach nauczonych cech;
nie odczytasz jej z wag jak z instrukcji — to **natura czarnej skrzynki**. Stąd wizualizację przyrostów liczymy
OSOBNO: z samego CORAL nie da się odczytać „policzył 6".

### Ryzyko „skrótu" (dlaczego trzeba to kontrolować)

Skoro model sam wybiera cechy, może nauczyć się **skrótu** — szacować wiek głównie z rozmiaru otolitu, albo
(gorzej) z artefaktu zdjęcia — zamiast z biologicznej struktury przyrostów. To znany problem w literaturze
o automatycznym odczycie otolitów. Dlatego po treningu patrzymy nie tylko na MAE, ale też na:
- **macierz pomyłek i bias per wiek** — czy nie zawala starych/rzadkich roczników (objaw regresji do rozmiaru),
- **heatmapy** — czy patrzy na ciało otolitu i pasma, a nie na tło / krawędź preparatu.

### Czy CORAL ma własną mapę uwagi?

**Natywnie — nie.** CORAL czyta tylko token CLS i zwraca **jedną liczbę**, więc sam z siebie nie produkuje
mapy przestrzennej (w przeciwieństwie do MIL, który daje wartość na każdy patch).

**Można ją wyprowadzić** dwoma sposobami — i **oba są już wdrożone** (11.07 Punkt 7, redesign karty):
1. **Uwaga CLS z backbone'u** (`compute_cls_attention`) — DINOv2 (ViT) w swoich warstwach liczy uwagę między
   tokenem CLS a patchami; wagi tej uwagi (ostatnia warstwa, uśrednione po głowach) pokazują, z których patchy
   CLS „poskładał" streszczenie. To najbliżej „skąd wziął się wejściowy sygnał CORAL". (Sięga wnętrza DINOv2 →
   z fallbackiem: gdy niedostępna, panel pokazuje „uwaga CLS niedostępna".)
2. **Grad-CAM / gradient** (`compute_coral_gradcam`) — gradient wyniku wieku `g` względem tokenów patchy →
   mapa „które patche najbardziej wpływają na wiek". Mapa **specyficzna dla CORAL**, solidna i testowalna.

**Jak to teraz wygląda w raporcie:** karta rozumowania (sekcja E) jest przebudowana na **dwa rzędy = dwie
głowice**:
- **Rząd 1 — GŁOWICA WIEKU (CORAL):** [Grad-CAM werdyktu] · [uwaga CLS] · [werdykt: wiek = X].
- **Rząd 2 — GŁOWICA LOKALIZACJI (MIL):** [mapa MIL] · [kandydaci (żółte)] · [finalne przyrosty (czerwone, N=wiek)].

Każdy panel ma pasek tytułu w kolorze głowicy (granatowy = CORAL, pomarańcz = MIL), więc od razu widać, która
plansza to wynik której głowicy. Dawna „mapa uwagi" (z MIL) jest teraz jasno przypisana do rzędu lokalizacji,
a werdykt wieku ma własne, osobne mapy (Grad-CAM + uwaga CLS) w rzędzie CORAL.

---

## 2. MIL — głowica lokalizacji („gdzie są przyrosty")

MIL = *Multiple Instance Learning*, czyli „uczenie z wielu instancji". „Instancje" = kwadraciki (patche).

### Dla laika
Wyobraź sobie **zakreślacz**, który ma zaznaczyć na zdjęciu miejsca, gdzie są roczne przyrosty. Kłopot: **nikt
nie powiedział mu, GDZIE one są** — dostał tylko informację „na tym zdjęciu jest ich 6 sztuk". Musi sam
zgadnąć rozmieszczenie tak, żeby zgadzała się liczba. To jest **słaby nadzór** (weakly supervised): uczymy
lokalizacji, mając tylko liczbę, bez zaznaczonych pozycji.

Efekt: MIL zapala **około `wiek` kwadracików** jako „tu chyba jest przyrost". Reszta obrazu ma być wygaszona.

### Profesjonalnie, ale prosto
- Bierze **tokeny patchy** (1369 kwadracików) → **wspólny mały klasyfikator** ocenia KAŻDY patch osobno →
  `patch_probs` ∈ [0,1]: „jakie jest prawdopodobieństwo, że TEN kwadracik leży na przyroście".
- **Uczenie (top-k concentration)**: bierzemy `⌈wiek⌉` patchy o najwyższym prawdopodobieństwie i ciągniemy je
  do 1 (to „przyrosty"), a wszystkie pozostałe do 0 (to „tło"). Dzięki temu **liczba aktywnych patchy
  (prob>0.5) ≈ wiek**.
  - *Dlaczego nie „suma = wiek"?* Bo wtedy sieć rozmazuje po ~wiek/1369 na każdym patchu (nic nie widać).
    Top-k wymusza kilka wyraźnych zapaleń zamiast mgły.
- **Nasz dodatek (radial-spread, 11.07)**: sama „liczba aktywnych" tworzy JEDNĄ plamę, a roczne przyrosty są
  **koncentryczne** (jeden za drugim, od jądra do brzegu). Dodaliśmy więc karę, która **rozprowadza aktywne
  patche wzdłuż promienia** od jądra otolitu — im starsza ryba, tym szerszy rozrzut. Cel: żeby przyrosty
  **separowały się w pierścienie**, a nie zlepiały w kleks. (Skuteczność na realnych danych ocenimy po treningu.)

---

## 3. Jak głowice współpracują (`head_type = both`)

Trenujemy **obie naraz**, łącząc ich błędy z wagami (z `configs/config.yaml`):

```
strata = coral_loss_weight · (błąd CORAL)          # wiek — werdykt
       + mil_count_weight  · (błąd MIL top-k)       # ile przyrostów
       + mil_radial_weight · (błąd rozrzutu)        # gdzie — rozłóż w pierścienie
```

Podział ról:
- **CORAL = werdykt.** Wiek pokazywany użytkownikowi bierze się z CORAL (jest stabilny i dokładny).
- **MIL = interpretacja.** Mapa MIL napędza wizualizacje: heatmapę uwagi oraz kropki kandydatów/przyrostów.

**Dlaczego dwie głowice, a nie jedna?** Bo „podaj wiek" i „wskaż każdy przyrost" to dwa różne zadania.
CORAL świetnie liczy globalnie, ale nie umie wskazywać. MIL próbuje wskazywać, ale jako licznik jest mniej
dokładny. Razem: dokładny wiek + próba wyjaśnienia „dlaczego".

---

## 4. Dlaczego wiek potrafi być trafny, a lokalizacja słaba

To NORMALNE w uczeniu słabo-nadzorowanym. CORAL czyta wiek z **całości** (rozmiar, tekstura, dojrzałość) —
jak zgadywanie wieku człowieka z ogólnego wyglądu, bez liczenia zmarszczek. MIL to **osobny, słabszy sygnał**
„gdzie ogólnie ważne", a nie „gdzie każdy pierścień". Dlatego model może dać dobry wiek 6, a jego mapa uwagi
być jedną plamą (stąd wcześniejsze „wiek 6, a jedna kropka"). Naprawą lokalizacji zajmuje się dodatek
radial-spread + składanie przyrostów z wielu osi przy dekodowaniu (patrz `11.07_pipeline_TO.DO.md`, Punkt 7).

---

## 5. Mały słowniczek
- **Backbone (DINOv2)** — sieć, która „widzi" obraz i zamienia go na liczby; wspólna dla obu głowic.
- **Patch** — kwadracik obrazu (14×14 px); obraz = siatka 37×37 = 1369 patchy.
- **Token CLS** — jeden wektor streszczający cały obraz (dla CORAL). **Tokeny patchy** — po jednym na kwadracik (dla MIL).
- **CORAL** — głowica wieku metodą „uporządkowanych pytań tak/nie" (ordinal regression, rank-consistent).
- **MIL** — głowica lokalizacji ucząca się z samej liczby (słaby nadzór); zapala ~`wiek` patchy.
- **Logit** — surowy wynik przed zamianą na prawdopodobieństwo (przez sigmoid).
- **Weakly supervised (słaby nadzór)** — uczymy „gdzie", mając tylko „ile", bez zaznaczonych pozycji.
- **Rank-consistency** — gwarancja, że odpowiedzi „starszy niż k" nie przeczą sobie nawzajem.
