import cv2
import matplotlib.pyplot as plt
import os

#scikza absolutna
script_dir = os.path.dirname(os.path.abspath(__file__))
#sklejenie sciezki
image_path = os.path.join(script_dir, "..", "data", "raw", "JPEGImages", "BloodImage_00000.jpg")

# Opcjonalnie: Normalizacja ścieżki (ładnie czyści uśniki i ukośniki na zgodne z systemem Windows/Mac)
image_path = os.path.normpath(image_path)


if not os.path.exists(image_path):
    print(f"BŁĄD: nie znaleziono pliku pod ścieżką: {image_path}")
else:
    #wczytuje obraz za pomoca opencv domyslnie w formacie BGR blue geeen red
    img_bgr=cv2.imread(image_path)

    #robie kwoenrsje przestrzeni bawr

    img_rgb=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


    #wizyalizacje
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.title(f"Obraz oryginalny RGB\nRozdzielczosc: {img_rgb.shape[:2]}")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(img_gray, cmap="gray")
    plt.title("Obraz w skali szarości")
    plt.axis("off")

    plt.tight_layout()
    plt.show()