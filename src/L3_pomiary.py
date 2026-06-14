import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import os

# 1. Przygotowanie obrazu binarnego (powtórzenie kroków z L2)
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "JPEGImages", "BloodImage_00000.jpg"))

img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
_, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 2. OPERACJA MORFOLOGICZNA: Domknięcie (Closing)
# Tworzymy element strukturalny (kółko o promieniu 3 pikseli) do "łatania" dziur
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
img_closed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

# 3. ETYKIETOWANIE (Connected Component Labeling)
# Zwraca macierz, gdzie tło ma wartość 0, pierwsza komórka 1, druga 2 itd.
labeled_mask = label(img_closed)

# 4. POMIARY PARAMETRÓW (skimage.measure.regionprops)
properties = regionprops(labeled_mask)

data_list = []
min_area = 100  # Ignorujemy paprochy mniejsze niż 100 pikseli

for prop in properties:
    # Filtrowanie szumów
    if prop.area < min_area:
        continue

    # Wyznaczenie współczynnika okrągłości (Form Factor / Circularity)
    area = prop.area
    perimeter = prop.perimeter
    if perimeter > 0:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    else:
        circularity = 0

    # Pobranie środka ciężkości (y, x)
    cy, cx = prop.centroid

    # Zapisujemy cechy do słownika
    data_list.append({
        "ID": prop.label,
        "Pole powierzchni [px]": area,
        "Obwód [px]": round(perimeter, 2),
        "Centroid X": round(cx, 1),
        "Centroid Y": round(cy, 1),
        "Okrągłość": round(circularity, 3)
    })

# 5. WRZUCENIE WYNIKÓW DO PANDAS DATAFRAME (Wymóg formalny)
df_cells = pd.DataFrame(data_list)

# Wyświetlamy tabelę w konsoli PyCharma
print("\n--- TABELA POMIAROWA KOMÓREK KRWI ---")
print(df_cells.to_string(index=False))

# Zapisujemy opcjonalnie do pliku CSV w folderze processed
output_csv = os.path.join(script_dir, "..", "data", "processed", "pomiary_L3.csv")
df_cells.to_csv(output_csv, index=False)
print(f"\nZapisano wyniki do: {output_csv}")

# 6. WIZUALIZACJA: Ponumerowane obiekty
plt.figure(figsize=(10, 8))
plt.imshow(img_gray, cmap='gray')

# Nakładamy numery ID na oryginalny obrazek, żeby widzieć, który wiersz z tabeli to która krwinka
for cell in data_list:
    plt.text(cell["Centroid X"], cell["Centroid Y"], str(cell["ID"]),
             color='red', fontsize=12, weight='bold',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

plt.title(f"Zidentyfikowane i zmierzone obiekty (Liczba: {len(df_cells)})")
plt.axis('off')
plt.tight_layout()
plt.show()