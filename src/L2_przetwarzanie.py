import cv2
import matplotlib.pyplot as plt
import os

# 1. Wczytanie obrazu w skali szarości
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "JPEGImages", "BloodImage_00000.jpg"))

img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 2. FILTROWANIE: Rozmycie Gaussa (usuwanie szumów)
# (5, 5) to rozmiar maski filtru (musi być nieparzysty), a 0 to odchylenie standardowe obliczane automatycznie
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

# 3. DETEKCJA KRAWĘDZI: Filtr Sobela
# Wyznaczamy pochodne w kierunku X i Y, a potem wyciągamy z nich wartość bezwzględną
sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
img_sobel = cv2.magnitude(sobel_x, sobel_y) # Wyznaczenie siły krawędzi

# 4. SEGMENTACJA: Progowanie metodą Otsu
# cv2.THRESH_BINARY_INV odwraca kolory: tło będzie czarne, a krwinki białe (idealne do pomiarów w L3!)
_, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


# --- WIZUALIZACJA ---
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title("1. Oryginał w skali szarości")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_blur, cmap='gray')
plt.title("2. Filtr Gaussa (Wygładzenie)")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(img_sobel, cmap='gray')
plt.title("3. Filtr Sobela (Krawędzie)")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_thresh, cmap='gray')
plt.title("4. Progowanie Otsu (Obraz binarny)")
plt.axis('off')

plt.tight_layout()
plt.show()