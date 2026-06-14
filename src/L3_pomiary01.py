import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import os

# 1. Konfiguracja ścieżek
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "JPEGImages"))

# Rygorystyczny zakres fioletu
lower_purple = np.array([115, 70, 60])
upper_purple = np.array([160, 255, 255])

all_features = []
images_processed = 0

for file_name in sorted(os.listdir(images_dir)):
    if not file_name.lower().endswith(('.jpg', '.jpeg')):
        continue

    img_path = os.path.join(images_dir, file_name)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_thresh = cv2.inRange(img_hsv, lower_purple, upper_purple)

    # Główne czyszczenie - zachowujemy spójność jądra jako całości
    kernel_main = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_cleaned = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_main)

    labeled_mask = label(img_cleaned)
    props = regionprops(labeled_mask)

    valid_props = [p for p in props if p.area > 250]

    if len(valid_props) > 0:
        images_processed += 1

        for prop in valid_props:
            area = prop.area
            perimeter = prop.perimeter
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            cy, cx = prop.centroid
            minr, minc, maxr, maxc = prop.bbox

            # --- ALGORYTM LICZENIA PŁATÓW (LOBES) ---
            # Wycinamy fragment maski zawierający tylko to jedno jądro
            cell_crop = img_cleaned[minr:maxr, minc:maxc].copy()

            # Używamy transformaty dystansowej (Distance Transform)
            # Mierzy ona odległość każdego białego piksela od najbliższego czarnego (brzegu)
            dist_transform = cv2.distanceTransform(cell_crop, cv2.DIST_L2, 5)

            # Progowanie transformaty dystansowej: zostawiamy tylko "środki" płatów,
            # odcinając wąskie mostki łączące je.
            _, peaks = cv2.threshold(dist_transform, 0.35 * dist_transform.max(), 255, 0)
            peaks = np.uint8(peaks)

            # Liczymy ile osobnych maksimów (płatów) powstało
            num_labels, _ = cv2.connectedComponents(peaks)
            lobes_count = num_labels - 1  # Odejmujemy 1, bo tło też jest liczone jako komponent

            # Zabezpieczenie: jeśli algorytm nic nie wykrył, przyjmujemy domyślnie przynajmniej 1 płat
            if lobes_count <= 0:
                lobes_count = 1
            # ----------------------------------------

            all_features.append({
                "Global_ID": len(all_features) + 1,
                "Plik": file_name,
                "Pole [px]": area,
                "Obwod [px]": round(perimeter, 2),
                "Okraglosc": round(circularity, 3),
                "Liczba płatów": lobes_count,
                "Centroid_X": round(cx, 1),
                "Centroid_Y": round(cy, 1),
                "BBox": f"[{minc}, {minr}, {maxc}, {maxr}]"
            })

    if len(all_features) >= 12:
        break

# 2. Pandas DataFrame
df_cells = pd.DataFrame(all_features)

# Obliczenie odległości i kątów do punktu referencyjnego (pierwsza komórka)
ref_x = df_cells.loc[0, "Centroid_X"]
ref_y = df_cells.loc[0, "Centroid_Y"]

distances = []
angles = []
for idx, row in df_cells.iterrows():
    tx, ty = row["Centroid_X"], row["Centroid_Y"]
    dist = np.sqrt((tx - ref_x) ** 2 + (ty - ref_y) ** 2)
    distances.append(round(dist, 1))

    angle = np.degrees(np.arctan2(ty - ref_y, tx - ref_x))
    if angle < 0: angle += 360
    angles.append(round(angle, 1))

df_cells["Odległość do ref [px]"] = distances
df_cells["Kąt do ref [°]"] = angles

# --- PREZENTACJA WYNIKÓW ---
print("\n--- OSTATECZNA REPRODUKOWALNA TABELA DANYCH PANDAS Z LICZBĄ PŁATÓW (L3) ---")
print(df_cells.to_string(index=False))

# Zapis do CSV
output_csv = os.path.normpath(os.path.join(script_dir, "..", "data", "processed", "final_pomiary_L3.csv"))
df_cells.to_csv(output_csv, index=False)

# 3. WIZUALIZACJA OSTATNIEGO OBRAZU
plt.figure(figsize=(10, 8))
img_rgb_show = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb_show)

for idx, row in df_cells[df_cells["Plik"] == file_name].iterrows():
    cx, cy = row["Centroid_X"], row["Centroid_Y"]
    bbox_coords = [int(x) for x in row["BBox"].strip('[]').split(',')]
    minc, minr, maxc, maxr = bbox_coords

    rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='lime', linewidth=2)
    plt.gca().add_patch(rect)
    plt.plot(cx, cy, 'ro', markersize=6)

    # Wyświetlamy ID oraz wykrytą liczbę płatów w ramce nad komórką
    plt.text(minc, minr - 10, f"ID {row['Global_ID']} | Płaty: {row['Liczba płatów']}",
             color='black', fontsize=11, weight='bold',
             bbox=dict(facecolor='lime', alpha=0.8, edgecolor='none', pad=2))

plt.title(f"Weryfikacja L3: Wykrywanie obiektów i segmentacja płatów jądra\nPlik: {file_name}")
plt.axis('off')
plt.tight_layout()
plt.show()

# =====================================================================
# INTERAKTYWNA WIZUALIZACJA: PRZEWIJANIE ZDJĘĆ STRZAŁKAMI (DOKLEJ NA KONIEC)
# =====================================================================

# Pobieramy listę wszystkich plików ze zdjęciami w folderze
all_image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
current_idx = [0]  # Używamy listy, aby móc modyfikować indeks wewnątrz funkcji sterującej

# Tworzymy nowe, dedykowane okno do przewijania
fig_scroll, ax_scroll = plt.subplots(figsize=(10, 8))


def wyswietl_zdjecie(idx):
    ax_scroll.clear()
    fn = all_image_files[idx]
    ip = os.path.join(images_dir, fn)
    img = cv2.imread(ip)
    if img is None:
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax_scroll.imshow(img_rgb)

    # Filtrujemy wiersze z tabeli df_cells odpowiadające bieżącemu zdjęciu
    df_sub = df_cells[df_cells["Plik"] == fn]

    if not df_sub.empty:
        for _, row in df_sub.iterrows():
            cx, cy = row["Centroid_X"], row["Centroid_Y"]
            bbox_coords = [int(x) for x in row["BBox"].strip('[]').split(',')]
            minc, minr, maxc, maxr = bbox_coords

            # Rysowanie ramki
            rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='lime', linewidth=2)
            ax_scroll.add_patch(rect)
            # Rysowanie centroidu
            ax_scroll.plot(cx, cy, 'ro', markersize=6)
            # Podpis nad ramką
            ax_scroll.text(minc, minr - 10, f"ID {row['Global_ID']} | Płaty: {row['Liczba płatów']}",
                           color='black', fontsize=10, weight='bold',
                           bbox=dict(facecolor='lime', alpha=0.8, edgecolor='none', pad=2))
        ax_scroll.set_title(
            f"Przeglądarka L3 (Zdjęcie {idx + 1}/{len(all_image_files)}): {fn}\n[Użyj strzałek <- -> do przewijania]")
    else:
        ax_scroll.set_title(
            f"Przeglądarka L3 (Zdjęcie {idx + 1}/{len(all_image_files)}): {fn}\n(Brak wykrytych leukocytów na tym zdjęciu)\n[Użyj strzałek <- -> do przewijania]")

    ax_scroll.axis('off')
    fig_scroll.canvas.draw()


def on_key(event):
    if event.key == 'right':
        current_idx[0] = (current_idx[0] + 1) % len(all_image_files)
        wyswietl_zdjecie(current_idx[0])
    elif event.key == 'left':
        current_idx[0] = (current_idx[0] - 1) % len(all_image_files)
        wyswietl_zdjecie(current_idx[0])


# Rejestracja zdarzenia klawiatury i pierwsze wyświetlenie
fig_scroll.canvas.mpl_connect('key_press_event', on_key)
wyswietl_zdjecie(current_idx[0])
plt.show()