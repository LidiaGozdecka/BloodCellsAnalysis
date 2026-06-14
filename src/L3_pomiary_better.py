import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import os

# ─────────────────────────────────────────────────────────────────────────────
# 1. ŚCIEŻKI
# ─────────────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "JPEGImages"))
labels_csv_path = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "labels.csv"))

# ─────────────────────────────────────────────────────────────────────────────
# 2. PROGI HSV
#
# Maska fioletowa (jądra komórkowe – standardowa)
lower_purple = np.array([110, 50, 40])
upper_purple = np.array([165, 255, 255])

# Maska ciemnoniebieska/granatowa – specjalnie dla limfocytów
# Jądra limfocytów są bardzo ciemne (niskie V) i bardziej niebieskie niż
# fioletowe. Zakres V=[20,130] wyklucza jasne struktury tła.
lower_lymph = np.array([100, 30, 20])
upper_lymph = np.array([145, 255, 130])

# Maska czerwona (ziarnistości eozynofilów – dwa zakresy, bo czerwień
# w HSV "zawija się" przez 0°)
lower_red1 = np.array([0,  60, 50])
upper_red1 = np.array([15, 255, 255])
lower_red2 = np.array([165, 60, 50])
upper_red2 = np.array([180, 255, 255])
# ─────────────────────────────────────────────────────────────────────────────


def segment_nucleus(img_bgr):
    """
    Dwuetapowa segmentacja jądra komórkowego.

    Etap 1 – maska fioletowa (Hue 110-165): działa dla neutrofili,
              eozynofilów i monocytów.
    Etap 2 – jeśli maska fioletowa daje mało pikseli (< 300), dokładamy
              maskę granatową (Hue 100-145, niskie V) przeznaczoną dla
              limfocytów z bardzo ciemnym, zwartym jądrem.
    Fallback – jeśli suma nadal < 100 px, używamy Otsu na kanale szarości.

    Zwraca: (mask_final, mask_red, used_lymph_mask)
      - mask_final  – maska binarna jądra
      - mask_red    – maska ziarnistości czerwonych (dla eozynofilów)
      - used_lymph  – True jeśli skorzystano z maski limfocytowej
    """
    img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # ── Maska fioletowa ──
    mask_purple = cv2.inRange(img_hsv, lower_purple, upper_purple)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN,  kernel5)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel7)

    # ── Maska czerwona (eozynofile) ──
    mask_r1  = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask_r2  = cv2.inRange(img_hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel5)

    used_lymph = False

    if cv2.countNonZero(mask_purple) < 300:
        # ── Maska limfocytowa ──
        mask_lymph = cv2.inRange(img_hsv, lower_lymph, upper_lymph)
        mask_lymph = cv2.morphologyEx(mask_lymph, cv2.MORPH_OPEN,  kernel5)
        mask_lymph = cv2.morphologyEx(mask_lymph, cv2.MORPH_CLOSE, kernel7)

        mask_combined = cv2.bitwise_or(mask_purple, mask_lymph)

        if cv2.countNonZero(mask_combined) >= 100:
            used_lymph = True
            return mask_combined, mask_red, used_lymph

        # ── Fallback: Otsu ──
        _, mask_otsu = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask_otsu, mask_red, used_lymph

    return mask_purple, mask_red, used_lymph


# ─────────────────────────────────────────────────────────────────────────────
# 3. PĘTLA GŁÓWNA – segmentacja i klasyfikacja regułowa
# ─────────────────────────────────────────────────────────────────────────────
neutrophil_list  = []
eosinophil_list  = []
monocyte_list    = []
lymphocyte_list  = []

classical_predictions = {}
# Słownik do wizualizacji: file_name → dane geometryczne + typ
vis_data = {}

all_files = sorted([f for f in os.listdir(images_dir)
                    if f.lower().endswith(('.jpg', '.jpeg'))])

print("Przetwarzanie obrazów dla 4 oficjalnych klas (bez bazofili)...")

for file_name in all_files:
    img_path = os.path.join(images_dir, file_name)
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        continue

    mask_nucleus, mask_red, used_lymph = segment_nucleus(img_bgr)

    props_purple = [p for p in regionprops(label(mask_nucleus)) if p.area > 200]
    props_red    = [p for p in regionprops(label(mask_red))     if p.area > 200]

    detected_type = "UNKNOWN"

    if len(props_purple) > 0:
        main_obj   = sorted(props_purple, key=lambda x: x.area, reverse=True)[0]
        area       = main_obj.area
        perimeter  = main_obj.perimeter
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        minr, minc, maxr, maxc = main_obj.bbox
        centroid_r, centroid_c = main_obj.centroid

        # Liczba płatów jądra (transformata dystansowa → connected components)
        crop = mask_nucleus[minr:maxr, minc:maxc]
        dist = cv2.distanceTransform(np.uint8(crop > 0), cv2.DIST_L2, 5)
        thresh_val = 0.45 * dist.max() if dist.max() > 0 else 0
        _, peaks    = cv2.threshold(dist, thresh_val, 255, 0)
        num_labels, _ = cv2.connectedComponents(np.uint8(peaks))
        lobes = max(1, num_labels - 1)

        # ── REGUŁY DECYZYJNE ──────────────────────────────────────────────
        # Kolejność ma znaczenie:
        # 1) Eozynofil: jądro fioletowe + obecność czerwonych ziarnistości
        # 2) Neutrofil: jądro wielopłatowe (lobes>1) lub nieokrągłe (<0.55)
        # 3) Limfocyt:  mała, bardzo okrągła komórka (poluzowane progi
        #               vs oryginał, żeby wyłapać więcej prawdziwych limf.)
        # 4) Monocyt:   wszystko inne (duże, nerkowate jądro jednopłatowe)
        # ─────────────────────────────────────────────────────────────────
        if len(props_red) > 0:
            detected_type = "EOSINOPHIL"
            eosinophil_list.append({
                "Plik": file_name, "Pole [px]": area,
                "Obwód [px]": round(perimeter, 1),
                "Okrągłość": round(circularity, 3),
                "BBox": f"[{minc},{minr},{maxc},{maxr}]",
                "Centroid": f"({centroid_c:.0f},{centroid_r:.0f})"})

        elif lobes > 1 or circularity < 0.55:
            detected_type = "NEUTROPHIL"
            neutrophil_list.append({
                "Plik": file_name, "Pole [px]": area,
                "Obwód [px]": round(perimeter, 1),
                "Okrągłość": round(circularity, 3),
                "Liczba płatów": lobes,
                "BBox": f"[{minc},{minr},{maxc},{maxr}]",
                "Centroid": f"({centroid_c:.0f},{centroid_r:.0f})"})

        elif circularity >= 0.68 and area < 9000:
            # Poluzowane progi: było >=0.75 i <7500
            # Uzasadnienie: limfocyty mają okrągłe jądra (0.68–1.0)
            # i małą powierzchnię; zaostrzony próg 0.75 wykluczał
            # limfocyty z lekko nieregularną membraną
            detected_type = "LYMPHOCYTE"
            lymphocyte_list.append({
                "Plik": file_name, "Pole [px]": area,
                "Obwód [px]": round(perimeter, 1),
                "Okrągłość": round(circularity, 3),
                "BBox": f"[{minc},{minr},{maxc},{maxr}]",
                "Centroid": f"({centroid_c:.0f},{centroid_r:.0f})"})

        else:
            detected_type = "MONOCYTE"
            monocyte_list.append({
                "Plik": file_name, "Pole [px]": area,
                "Obwód [px]": round(perimeter, 1),
                "Okrągłość": round(circularity, 3),
                "BBox": f"[{minc},{minr},{maxc},{maxr}]",
                "Centroid": f"({centroid_c:.0f},{centroid_r:.0f})"})

        # Zachowaj dane do wizualizacji
        vis_data[file_name] = {
            "type": detected_type,
            "bbox": (minr, minc, maxr, maxc),
            "centroid": (centroid_r, centroid_c),
            "circularity": round(circularity, 3),
            "area": area,
            "lobes": lobes
        }

    classical_predictions[file_name] = detected_type

# ─────────────────────────────────────────────────────────────────────────────
# 4. TABELA WYNIKÓW (wymóg: co najmniej 10 obiektów)
# ─────────────────────────────────────────────────────────────────────────────
df_neutrophils  = pd.DataFrame(neutrophil_list)
df_eosinophils  = pd.DataFrame(eosinophil_list)
df_monocytes    = pd.DataFrame(monocyte_list)
df_lymphocytes  = pd.DataFrame(lymphocyte_list)

print(f"\n--- RESTRUKTURYZACJA DANYCH ZAKOŃCZONA (OSTATECZNE 4 KLASY) ---")
print(f"Neutrofile: {len(df_neutrophils)} | Eozynofile: {len(df_eosinophils)} "
      f"| Monocyty: {len(df_monocytes)} | Limfocyty: {len(df_lymphocytes)}")

# Połącz wszystkie klasy w jedną zbiorczą tabelę (min. 10 wierszy)
all_records = []
for rec in neutrophil_list:
    all_records.append({**rec, "Klasa": "NEUTROPHIL"})
for rec in eosinophil_list:
    all_records.append({**rec, "Klasa": "EOSINOPHIL"})
for rec in monocyte_list:
    all_records.append({**rec, "Klasa": "MONOCYTE"})
for rec in lymphocyte_list:
    all_records.append({**rec, "Klasa": "LYMPHOCYTE"})

df_all = pd.DataFrame(all_records)

# Wybierz po max 3 z każdej klasy → ~10-12 wierszy w tabeli wynikowej
df_sample = (df_all.groupby("Klasa", group_keys=False)
             .apply(lambda g: g.head(3))
             .reset_index(drop=True))

print("\n========= TABELA PARAMETRÓW GEOMETRYCZNYCH (próbka po 3 z klasy) =========")
print(df_sample[["Klasa", "Plik", "Pole [px]", "Obwód [px]", "Okrągłość",
                  "BBox", "Centroid"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 5. WIZUALIZACJA POMIARÓW NA OBRAZACH
#    Rysuje bounding box, centroid i etykietę na 8 przykładowych obrazach
# ─────────────────────────────────────────────────────────────────────────────
COLOR_MAP = {
    "NEUTROPHIL":  (0,   200, 255),   # cyjan
    "EOSINOPHIL":  (0,   255, 0),     # zielony
    "MONOCYTE":    (255, 140, 0),     # pomarańczowy
    "LYMPHOCYTE":  (255, 0,   180),   # różowy
    "UNKNOWN":     (128, 128, 128),   # szary
}

# Wybierz po 2 przykłady z każdej klasy (jeśli dostępne)
examples = {}
for fname, vd in vis_data.items():
    t = vd["type"]
    if t not in examples:
        examples[t] = []
    if len(examples[t]) < 2:
        examples[t].append(fname)

example_list = []
for t in ["NEUTROPHIL", "EOSINOPHIL", "MONOCYTE", "LYMPHOCYTE"]:
    example_list.extend(examples.get(t, []))

n_vis = min(8, len(example_list))
cols  = 4
rows  = (n_vis + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
axes = np.array(axes).flatten()

for i in range(len(axes)):
    axes[i].axis("off")

for i, fname in enumerate(example_list[:n_vis]):
    img_bgr = cv2.imread(os.path.join(images_dir, fname))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    vd    = vis_data[fname]
    dtype = vd["type"]
    color_rgb = tuple(c / 255 for c in COLOR_MAP.get(dtype, (128, 128, 128)))
    minr, minc, maxr, maxc = vd["bbox"]
    cy, cx = vd["centroid"]

    ax = axes[i]
    ax.imshow(img_rgb)

    # Bounding box
    rect = mpatches.Rectangle(
        (minc, minr), maxc - minc, maxr - minr,
        linewidth=2, edgecolor=color_rgb, facecolor="none")
    ax.add_patch(rect)

    # Centroid
    ax.plot(cx, cy, "x", color=color_rgb, markersize=10, markeredgewidth=2)

    # Etykieta z parametrami
    label_txt = (f"{dtype}\n"
                 f"Area={vd['area']} px²\n"
                 f"Circ={vd['circularity']}\n"
                 f"Lobes={vd['lobes']}")
    ax.set_title(label_txt, fontsize=7.5, color="black",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.axis("off")

# Legenda klas
legend_handles = [
    mpatches.Patch(color=tuple(c / 255 for c in v), label=k)
    for k, v in COLOR_MAP.items() if k != "UNKNOWN"
]
fig.legend(handles=legend_handles, loc="lower center",
           ncol=4, fontsize=9, frameon=True,
           title="Klasy komórek", title_fontsize=9)

plt.suptitle("Wizualizacja pomiarów geometrycznych (L3)\n"
             "Bounding box ■ | Centroid ✕ | Etykieta z parametrami",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "segmentacja_klasyczna_l3.png"),
            bbox_inches="tight", dpi=150)
plt.show()
print("✅ Zapisano: segmentacja_klasyczna_l3.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. EWALUACJA I MACIERZ POMYŁEK
# ─────────────────────────────────────────────────────────────────────────────
if os.path.exists(labels_csv_path):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report

    df_labels = pd.read_csv(labels_csv_path)
    img_col = df_labels.columns[1]
    cat_col = df_labels.columns[2]

    df_labels["Clean_File"] = df_labels[img_col].apply(
        lambda x: (f"BloodImage_{int(x):05d}.jpg" if str(x).isdigit()
                   else f"{str(x).strip()}.jpg" if not str(x).lower().endswith(".jpg")
                   else str(x).strip()))

    detected_count     = 0
    correct_type_count = 0
    total_rows         = 0
    y_true, y_pred     = [], []
    l3_classes = ["NEUTROPHIL", "EOSINOPHIL", "MONOCYTE", "LYMPHOCYTE"]

    for _, row in df_labels.iterrows():
        true_cat = str(row[cat_col]).upper().strip()
        if true_cat == "BASOPHIL":
            continue
        total_rows += 1
        f_name   = row["Clean_File"]
        pred_cat = classical_predictions.get(f_name, "UNKNOWN")

        if pred_cat != "UNKNOWN":
            detected_count += 1
        if pred_cat == true_cat:
            correct_type_count += 1

        if true_cat in l3_classes:
            y_true.append(true_cat)
            y_pred.append(pred_cat if pred_cat in l3_classes else "UNKNOWN")

    print(f"\n=========== RAPORT ANALITYCZNY CV (4 KLASY) ==============")
    print(f"1) WYKRYWANIE OGÓŁEM: {detected_count}/{total_rows} komórek "
          f"({(detected_count / total_rows) * 100:.2f}%)")
    print(f"2) KLASYFIKACJA TYPÓW (Accuracy): "
          f"{(correct_type_count / total_rows) * 100:.2f}%")
    print(f"=================================================")

    print("\n================ RAPORT KLASYFIKACJI (L3) ================")
    print(classification_report(y_true, y_pred,
                                labels=l3_classes, zero_division=0))

    # Macierz pomyłek
    cm_labels = sorted(l3_classes)
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cm_labels, yticklabels=cm_labels)
    plt.title("Macierz Pomyłek – Metoda Regułowa (L3)")
    plt.ylabel("Rzeczywista klasa")
    plt.xlabel("Predykcja")
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "macierz_pomylek_l3.png"),
                bbox_inches="tight", dpi=150)
    plt.show()
    print("✅ Zapisano: macierz_pomylek_l3.png")
