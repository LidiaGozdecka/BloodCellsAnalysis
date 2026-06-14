# BloodCellsAnalysis 🔬

Projekt zaliczeniowy z przedmiotu **Automatyczna Analiza Obrazu**.  
Temat: klasyfikacja białych krwinek (leukocytów) z obrazów mikroskopowych przy użyciu klasycznych metod CV oraz konwolucyjnych sieci neuronowych.

---

## Dataset

**Źródło:** [Blood Cell Images – Kaggle (Paul Mooney)](https://www.kaggle.com/datasets/paultimothymooney/blood-cells/data)

Zbiór zawiera obrazy mikroskopowe białych krwinek podzielone na klasy:
`NEUTROPHIL`, `EOSINOPHIL`, `MONOCYTE`, `LYMPHOCYTE`, `BASOPHIL`

> ⚠️ Klasa `BASOPHIL` została wykluczona z modelowania ze względu na krytycznie niską liczebność (3 próbki), uniemożliwiającą stratyfikowany podział na zbiory.

Po pobraniu umieść dane w katalogu `data/raw/` zgodnie ze strukturą:
```
data/
└── raw/
    ├── JPEGImages/          ← obrazy .jpg
    └── labels.csv           ← etykiety klas
```

---

## Struktura projektu

```
BloodCellsAnalysis/
├── data/
│   └── raw/                        ← dane wejściowe (nie w repozytorium – patrz Dataset)
├── src/
│   ├── L1_akwizycja.py             ← L1: wczytanie i konwersja przestrzeni barw
│   ├── L1_dyskretyzacja.py         ← L1: próbkowanie, kwantyzacja, histogramy
│   ├── L2_przetwarzanie.py         ← L2: filtry, transformacje, progowanie
│   ├── L3_pomiary.py               ← L3: segmentacja i klasyfikacja regułowa
│   ├── L5_cnn_weighted.py          ← L5: CNN z wagami klas (model 4-klasowy)
│   ├── L5_cnn_unweighted.py        ← L5: CNN bez ważenia (model bazowy / naiwny)
│   ├── L5_cnn_3_class.py           ← L5: CNN 3-klasowy z oversamplingiem
│   └── L5_cnn_3_class_better.py    ← L5: Transfer learning MobileNetV2 (3 klasy)
├── raport_bazowy_unweighted.html   ← raport modelu naiwnego
├── raport_koncowy.html             ← raport modelu ważonego
├── raport_koncowy_3class.html      ← raport modelu 3-klasowego (CNN własna)
├── raport_koncowy_3class_better.html ← raport modelu z transfer learningiem
├── sprawozdanie.html               ← pełne sprawozdanie projektu
├── requirements.txt
└── README.md
```

---

## Wymagania i instalacja

### Python
Projekt wymaga **Python 3.10+**.

```bash
# Utwórz i aktywuj wirtualne środowisko (zalecane)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Zainstaluj zależności
pip install -r requirements.txt
```

### Zawartość requirements.txt
```
numpy==1.26.4
matplotlib==3.8.3
opencv-python==4.9.0.80
scikit-image==0.22.0
pandas==2.2.1
tensorflow>=2.13
scikit-learn
```

> 💡 Projekt był rozwijany w środowisku **PyCharm** na Windows. TensorFlow GPU nie jest obsługiwany natywnie na Windows dla wersji >= 2.11 – użyj WSL2 lub CPU.

---

## Kolejność uruchamiania

Skrypty należy uruchamiać z głównego katalogu projektu lub z poziomu `src/` (ścieżki są względne do lokalizacji skryptu):

```bash
# 1. Akwizycja i konwersja przestrzeni barw
python src/L1_akwizycja.py

# 2. Dyskretyzacja (próbkowanie + kwantyzacja + histogramy)
python src/L1_dyskretyzacja.py

# 3. Filtry, transformacje geometryczne, progowanie
python src/L2_przetwarzanie.py

# 4. Segmentacja i klasyfikacja regułowa (L3)
python src/L3_pomiary.py

# 5. CNN – model naiwny (bez ważenia)
python src/L5_cnn_unweighted.py

# 6. CNN – model ważony (wagi klas)
python src/L5_cnn_weighted.py

# 7. CNN – model 3-klasowy (oversampling)
python src/L5_cnn_3_class.py

# 8. Transfer learning – MobileNetV2 (3 klasy)
python src/L5_cnn_3_class_better.py
```

---

## Reprodukowalność

Wszystkie skrypty korzystają z ustalonego seeda losowości:

| Biblioteka | Seed |
|-----------|------|
| Python `random` | 2026 |
| NumPy | 2026 |
| TensorFlow | 2026 |
| scikit-learn (`train_test_split`) | `random_state=42` |

Modele `.keras` nie są przechowywane w repozytorium ze względu na rozmiar. Aby uzyskać zapisany model, uruchom odpowiedni skrypt — zostanie wygenerowany lokalnie.

---

## Wyniki (skrót)

| Model | Podejście | Accuracy (test) | Macro F1 |
|-------|-----------|----------------|----------|
| L3 klasyczne CV | Reguły geometryczne | 56% | 0.33 |
| L5 CNN unweighted | Naiwny (4 klasy) | ~60%* | ~0.17 |
| L5 CNN weighted | Wagi klas (4 klasy) | – | – |
| L5 CNN 3-class | Oversampling (3 klasy) | 15% | 0.09 |
| L5 CNN 3-class better | MobileNetV2 TL (3 klasy) | **55%** | **0.46** |

*Model naiwny klasyfikuje prawie wszystko jako NEUTROPHIL – accuracy jest złudna.

Szczegółowe wyniki, macierze pomyłek i wnioski znajdują się w pliku `sprawozdanie.html`.

---

## Autor

Lidia Gozdecka – projekt indywidualny, semestr 6.
