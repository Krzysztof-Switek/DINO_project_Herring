# Jak działa nasz model — wyjaśnienie dla laika

Dokument tłumaczy prostymi słowami, **jak nasz model rozpoznaje wiek ryby ze zdjęcia otolitu
i jak się tego uczy**. Dwa poziomy: (1) zwykła analogia, (2) trochę bardziej fachowo, ale wciąż
bez matematyki.

---

## 1. Prostymi słowami — analogia

Znasz porównanie do **CNN jak układanka z puzzli**: sieć patrzy na pojedyncze puzzle, rozpoznaje
na nich cechy (krawędzie, kształty) i z tego zgaduje, co jest na całym obrazku. Nasz model działa
podobnie na poziomie „patrzenia", ale **najciekawsze jest to, JAK się uczy**. Najlepsza analogia:

> **Wyobraź sobie nowego asystenta w laboratorium, który ma wyjątkowo wyćwiczone oko.**

Zanim do nas trafił, obejrzał **miliony różnych zdjęć** i nauczył się dostrzegać kształty,
krawędzie, pasma i kontrasty. To są jego „oczy" — u nas nazywają się **DINOv2**. Nie musimy uczyć
go patrzenia od zera; on już umie patrzeć.

Dajemy mu **ogromny stos zdjęć otolitów**. I teraz sedno: **przy każdym zdjęciu mówimy mu tylko
jedno — ile ryba miała lat.** Nigdy nie pokazujemy, **gdzie** są pierścienie ani jak je liczyć.
Musi dojść do tego sam.

Na początku **zgaduje losowo**. Ale za każdym razem, gdy powie „5 lat", a my odpowiemy „nie, 3" —
odrobinę koryguje swoją intuicję. Po tysiącach zdjęć **sam odkrywa**: *„te słabe, koncentryczne
pasma to jest to, co zmienia się z wiekiem — jeśli zaznaczę mniej więcej tyle pasm, ile wynosi
wiek, moje odpowiedzi zaczynają się zgadzać".*

W ten sposób uczy się **dwóch rzeczy naraz**, choć uczyliśmy go tylko wieku:
- **liczyć wiek** ryby,
- **wskazywać, gdzie są pierścienie** — mimo że nikt nigdy nie pokazał mu ani jednego pierścienia.

**Jak patrzy na zdjęcie (tu wraca układanka):** dzieli otolit na **siatkę małych kwadracików**
(jak puzzle) i dla każdego ocenia: *„na ile to wygląda na roczny przyrost?"*. Kwadraciki, które
„zapala", to nasze **kropki i krzywe pierścieni** na kartach. A liczba zapalonych kwadracików
powinna zgadzać się z policzonym wiekiem — to jego wewnętrzny sposób sprawdzania samego siebie.

W środku pracuje **dwóch specjalistów**:
- **Rachmistrz** — patrzy na cały otolit i mówi „to jakieś 4 lata" (to głowica **wieku**).
- **Wskazywacz** — chodzi kwadracik po kwadraciku i zaznacza te, które wyglądają na pierścień
  (to głowica **lokalizacji**).

Obaj się nawzajem pilnują: **liczba zaznaczeń ≈ policzony wiek**.

**Jak się uczy (w skrócie):** to gra „**zgadnij i popraw**". Model zgaduje → porównujemy z prawdą →
różnica (błąd) delikatnie przestawia jego wewnętrzne „pokrętła" w dobrą stronę → i tak tysiące
razy. Z czasem i wiek się zgadza, i mapa „gdzie pierścienie" robi się ostrzejsza — bo **jedyny
sposób, żeby stale trafiać w wiek, to naprawdę znaleźć pierścienie.**

---

## 2. CNN vs ViT — ta sama układanka, dwa sposoby patrzenia

Ważne: nasz model jest z rodziny **Vision Transformer (ViT)** — **nie** CNN. I jeszcze jedno
rozróżnienie, które łatwo pomylić: **„ViT" to architektura** (czym model *jest*), a **„DINO/DINOv2"
to sposób uczenia** (jak go *wytrenowano* — samodzielnie, bez etykiet). Pod maską mamy więc **ViT
wytrenowany metodą DINO**.

Weźmy tę samą **układankę z puzzli** i zobaczmy, jak patrzą na nią oba podejścia:

**CNN — lokalna lupa przesuwająca się po obrazie.**
CNN bierze małe „okienko" (lupę) i **przesuwa je kawałek po kawałku** po całym obrazie, patrząc
zawsze tylko na **mały wycinek i jego najbliższe otoczenie**. Z tych lokalnych obserwacji składa
zrozumienie **od dołu do góry**: najpierw krawędzie, potem kształty, potem całość. Ma wbudowane
założenie: **liczy się sąsiedztwo** (piksele tuż obok siebie). Żeby połączyć dwa **odległe**
fragmenty obrazu, potrzebuje wielu warstw — dopiero po drodze „lupa" obejmuje coraz większy obszar.
*(Twoja intuicja jest trafna: CNN to lokalna lupa jeżdżąca po obrazie.)*

**ViT — wszystkie puzzle rozłożone naraz i „rozmawiające" ze sobą.**
ViT **tnie obraz na kwadraciki (patche)** — dosłownie na puzzle — i **kładzie je wszystkie na stole
naraz**. Potem pozwala, żeby **każdy kwadracik od razu porównał się z każdym innym**: „jak ja się
mam do ciebie?". Ten mechanizm nazywa się **uwagą (attention)**. Dzięki temu ViT **od pierwszej
chwili widzi zależności między odległymi fragmentami** — nie musi ich mozolnie „sklejać" warstwa po
warstwie.

**Różnica w jednym zdaniu:**
- **CNN:** *„oglądam po kawałku lupą i stopniowo składam całość"* — mocny w lokalnych szczegółach, uczy się z mniejszej ilości danych.
- **ViT:** *„widzę wszystkie kawałki naraz i od razu wiem, jak się łączą"* — mocny w strukturze globalnej, ale potrzebuje więcej danych (stąd samonadzorowany trening DINO na milionach zdjęć).

**Dlaczego to ma znaczenie dla pierścieni:**
Roczny pierścień to **jedna, koncentryczna linia obejmująca cały otolit**. Kwadracik po lewej i po
prawej **należą do tego samego pierścienia**. ViT łączy je **bezpośrednio** (attention), więc od
razu „rozumie", że to jedna struktura; CNN musiałby to budować przez wiele warstw. Dlatego **nasza
analogia z puzzlami pasuje lepiej do ViT** — bo tam te puzzle naprawdę leżą razem i gadają ze sobą,
a w CNN są oglądane pojedynczo, przesuwaną lupą.

---

## 3. Trochę bardziej fachowo (ale wciąż po ludzku)

Nasz model to połączenie **trzech elementów**:

**a) Wytrenowane „oczy" — DINOv2 (uczenie samonadzorowane).**
DINOv2 to sieć, która nauczyła się „widzieć" na milionach zdjęć **bez żadnych etykiet** — sama, przez
porównywanie fragmentów obrazów. Efekt: uniwersalny **„ekstraktor cech"**, który każdy fragment
obrazu opisuje zestawem liczb. Nie uczymy widzenia od zera — **przenosimy** tę wiedzę na otolity
(*transfer learning*), a potem lekko ją **doszlifowujemy** na naszych zdjęciach (*fine-tuning*).

**b) Słaby nadzór + liczenie (MIL).**
„**Słaby nadzór**" = uczymy tylko etykietą całego obrazu (wiek), **bez zaznaczania pierścieni**.
Technika **MIL** wymusza, żeby **suma „prawdopodobieństw przyrostu" ze wszystkich kwadracików
odpowiadała wiekowi**. Dzięki temu ograniczeniu model — chcąc dobrze policzyć — **musi** nauczyć
się, które kwadraciki to pierścienie. Czyli **lokalizuje bez ani jednej ręcznej anotacji**.

**c) Liczenie wieku z zachowaniem porządku (CORAL).**
Wiek jest **uporządkowany** (4 lata są bliżej 5 niż 1). Zamiast traktować lata jak niezależne
„szufladki", używamy głowicy **ordinalnej** (CORAL), która ten porządek rozumie — dzięki temu
pomyłka „4 zamiast 5" liczy się jako **mały** błąd, a nie kompletne pudło.

**Jak przebiega trening (pętla uczenia):**
1. Bierzemy paczkę zdjęć → model przewiduje **wiek** i **mapę pierścieni**.
2. Liczymy **błąd** (o ile pomylił wiek + jak spójna jest mapa).
3. Algorytm „cofa" ten błąd i **minimalnie poprawia miliony wewnętrznych parametrów** (*backpropagation*).
4. Powtarzamy przez wiele przejść przez dane (**epok**).

Po każdej epoce sprawdzamy model na **odłożonym zbiorze** (walidacja) i patrzymy m.in. na:
- **val_MAE** — średni błąd wieku w latach (mniej = lepiej; u nas już poniżej 1 roku),
- **#aktywnych vs średni wiek** — czy liczba zapalonych kwadracików zbliża się do średniego wieku
  (czyli czy model **naprawdę lokalizuje**, a nie zgaduje).

**Dwie fazy treningu:** najpierw **zamrażamy „oczy"** (używamy DINOv2 tak jak jest) i uczymy tylko
dwóch specjalistów — szybko i stabilnie; potem **odmrażamy** i pozwalamy oczom lekko dostroić się
do otolitów.

**Dlaczego zaczęliśmy od najmniejszego „silnika" (ViT-S):** żeby cały proces (skan → trening →
raport) najpierw **zadziałał** szybko i tanio. Większy, mocniejszy backbone podłączymy, gdy
fundament jest pewny — i porównamy, ile realnie zyskujemy.

---

## Mały słowniczek
- **Backbone / „oczy" (DINOv2)** — gotowa sieć rozpoznająca wzorce na obrazie; opisuje każdy fragment liczbami.
- **Patch / kwadracik** — mały fragment zdjęcia; model ocenia każdy osobno.
- **Słaby nadzór** — uczenie tylko etykietą całości (wiek), bez wskazywania pierścieni.
- **MIL** — mechanizm, który każe sumie „ocen pierścienia" po kwadracikach odpowiadać wiekowi → zmusza do lokalizacji.
- **CORAL** — sposób przewidywania wieku, który rozumie, że lata są po kolei (4 blisko 5).
- **Epoka** — jedno pełne przejście modelu przez wszystkie zdjęcia treningowe.
- **val_MAE** — średni błąd wieku na danych, których model nie widział w treningu.

> Pełny plan modeli i kolejne kroki: `10.07_MODELE_PLAN_TO_DO.md`.
