import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import os

# 1. Ścieżki do plików
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "JPEGImages"))
labels_csv_path = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "labels.csv"))

# --- PROGI HSV DLA 4 KLAS ---
lower_purple = np.array([110, 50, 40])
upper_purple = np.array([165, 255, 255])

lower_red1 = np.array([0, 60, 50])
upper_red1 = np.array([15, 255, 255])
lower_red2 = np.array([165, 60, 50])
upper_red2 = np.array([180, 255, 255])

# 4 docelowe listy na struktury danych
neutrophil_list = []
eosinophil_list = []
monocyte_list = []
lymphocyte_list = []

classical_predictions = {}
all_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg'))])

print("Przetwarzanie obrazów dla 4 oficjalnych klas (bez bazofili)...")

for file_name in all_files:
    img_path = os.path.join(images_dir, file_name)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask_purple = cv2.inRange(img_hsv, lower_purple, upper_purple)
    mask_r1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel)
    clean_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

    props_purple = [p for p in regionprops(label(clean_purple)) if p.area > 200]
    props_red = [p for p in regionprops(label(clean_red)) if p.area > 200]

    detected_type = "UNKNOWN"

    if len(props_purple) > 0:
        props_purple_sorted = sorted(props_purple, key=lambda x: x.area, reverse=True)
        main_obj = props_purple_sorted[0]

        area = main_obj.area
        perimeter = main_obj.perimeter
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        minr, minc, maxr, maxc = main_obj.bbox

        crop = clean_purple[minr:maxr, minc:maxc]
        dist = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
        _, peaks = cv2.threshold(dist, 0.45 * dist.max() if dist.max() > 0 else 0, 255, 0)
        num_labels, _ = cv2.connectedComponents(np.uint8(peaks))
        lobes = max(1, num_labels - 1)

        # --- REGUŁY DECYZYJNE DLA OFICJALNYCH 4 KLAS ---
        if len(props_red) > 0:
            detected_type = "EOSINOPHIL"
            eosinophil_list.append({"Plik": file_name, "Pole [px]": area, "Obwod [px]": round(perimeter, 1),
                                    "Okraglosc": round(circularity, 3), "BBox": f"[{minc},{minr},{maxc},{maxr}]"})
        elif lobes > 1 or circularity < 0.55:
            detected_type = "NEUTROPHIL"
            neutrophil_list.append({"Plik": file_name, "Pole [px]": area, "Obwod [px]": round(perimeter, 1),
                                    "Okraglosc": round(circularity, 3), "Liczba_platow": lobes,
                                    "BBox": f"[{minc},{minr},{maxc},{maxr}]"})
        else:
            if circularity >= 0.75 and area < 7500:
                detected_type = "LYMPHOCYTE"
                lymphocyte_list.append({"Plik": file_name, "Pole [px]": area, "Obwod [px]": round(perimeter, 1),
                                        "Okraglosc": round(circularity, 3), "BBox": f"[{minc},{minr},{maxc},{maxr}]"})
            else:
                detected_type = "MONOCYTE"
                monocyte_list.append({"Plik": file_name, "Pole [px]": area, "Obwod [px]": round(perimeter, 1),
                                      "Okraglosc": round(circularity, 3), "BBox": f"[{minc},{minr},{maxc},{maxr}]"})

    classical_predictions[file_name] = detected_type

# Tworzenie 4 poprawnych DataFrame'ów
df_neutrophils = pd.DataFrame(neutrophil_list)
df_eosinophils = pd.DataFrame(eosinophil_list)
df_monocytes = pd.DataFrame(monocyte_list)
df_lymphocytes = pd.DataFrame(lymphocyte_list)

print(f"\n--- RESTRUKTURYZACJA DANYCH ZAKOŃCZONA (OSTATECZNE 4 KLASY) ---")
print(
    f"Neutrofile: {len(df_neutrophils)} | Eozynofile: {len(df_eosinophils)} | Monocyty: {len(df_monocytes)} | Limfocyty: {len(df_lymphocytes)}")

if os.path.exists(labels_csv_path):
    df_labels = pd.read_csv(labels_csv_path)
    img_col = df_labels.columns[1]
    cat_col = df_labels.columns[2]

    df_labels['Clean_File'] = df_labels[img_col].apply(
        lambda x: f"BloodImage_{int(x):05d}.jpg" if str(x).isdigit() else f"{str(x).strip()}.jpg" if not str(
            x).lower().endswith('.jpg') else str(x).strip())

    detected_count = 0
    correct_type_count = 0
    total_rows = 0

    for _, row in df_labels.iterrows():
        true_cat = str(row[cat_col]).upper().strip()
        # Ignorujemy bazofile w ewaluacji końcowej
        if true_cat == "BASOPHIL":
            continue

        total_rows += 1
        f_name = row['Clean_File']
        pred_cat = classical_predictions.get(f_name, "UNKNOWN")

        if pred_cat != "UNKNOWN": detected_count += 1
        if pred_cat == true_cat: correct_type_count += 1

    print(f"\n=========== RAPORT ANALITYCZNY CV (4 KLASY) ==============")
    print(f"1) WYKRYWANIE OGÓŁEM: {detected_count}/{total_rows} komórek ({(detected_count / total_rows) * 100 :.2f}%)")
    print(f"2) KLASYFIKACJA TYPÓW (Accuracy): {(correct_type_count / total_rows) * 100 :.2f}%")
    print(f"=================================================")

    # --- GRAFICZNA MACIERZ POMYŁEK (L3) ---
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report

    y_true = []
    y_pred = []
    l3_classes = ['NEUTROPHIL', 'EOSINOPHIL', 'MONOCYTE', 'LYMPHOCYTE']

    for f_name in classical_predictions.keys():
        row_match = df_labels[df_labels['Clean_File'] == f_name]
        if not row_match.empty:
            t_cat = str(row_match.iloc[0][cat_col]).upper().strip()
            if t_cat in l3_classes:
                y_true.append(t_cat)
                y_pred.append(classical_predictions[f_name])

    cm_labels = sorted(l3_classes)
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)

    # Tworzenie wykresu kaflowego (Seaborn Heatmap)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=cm_labels, yticklabels=cm_labels)
    plt.title('Macierz Pomyłek - Metoda Regułowa (L3)')
    plt.ylabel('Rzeczywista klasa')
    plt.xlabel('Predykcja')

    # Zapisujemy na dysk jako osobny plik (Ważne dla pipeline!)
    plt.savefig(os.path.join(script_dir, "macierz_pomylek_l3.png"), bbox_inches='tight', dpi=150)
    plt.close()

    print("\n================ RAPORT KLASYFIKACJI (L3) ================")
    print(classification_report(y_true, y_pred, labels=cm_labels, zero_division=0))