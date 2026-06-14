import cv2
import matplotlib.pyplot as plt
import numpy as np  # ← DODAJ ten import (potrzebny do np.uint8)
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "JPEGImages", "BloodImage_00000.jpg"))

img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Filtr Gaussa
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

# Filtr Sobela
sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
img_sobel = cv2.magnitude(sobel_x, sobel_y)

# Progowanie Otsu
_, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ── NOWE: Filtr Laplacjana ─────────────────────────────────────────────
# Laplacjan wykrywa krawędzie we WSZYSTKICH kierunkach naraz (w odróżnieniu
# od Sobela, który liczy pochodne osobno dla X i Y).
# CV_64F = wynik jako float64, bo Laplacjan może dawać wartości ujemne
# np.uint8(np.absolute(...)) → bierzemy wartość bezwzględną i konwertujemy
# z powrotem do 8-bit, żeby móc wyświetlić jako obraz szary
img_laplacian_raw = cv2.Laplacian(img_blur, cv2.CV_64F, ksize=3)
img_laplacian = np.uint8(np.absolute(img_laplacian_raw))
# ──────────────────────────────────────────────────────────────────────

# Filtr medianowy (opcjonalny czwarty filtr – usuwa szum "sól i pieprz")
img_median = cv2.medianBlur(img_gray, 5)

# ── WIZUALIZACJA: zmień plt.figure i wszystkie numery subplotów ─────────
plt.figure(figsize=(16, 8))  # ← było (12,10), teraz szerzej na 6 paneli

plt.subplot(2, 3, 1)          # ← było (2,2,1)
plt.imshow(img_gray, cmap='gray')
plt.title("1. Oryginał w skali szarości")
plt.axis('off')

plt.subplot(2, 3, 2)          # ← było (2,2,2)
plt.imshow(img_blur, cmap='gray')
plt.title("2. Filtr Gaussa\n(wygładzenie, σ=auto, maska 5×5)")
plt.axis('off')

plt.subplot(2, 3, 3)          # ← było (2,2,3)
plt.imshow(img_sobel, cmap='gray')
plt.title("3. Filtr Sobela\n(krawędzie kierunkowe X+Y)")
plt.axis('off')

plt.subplot(2, 3, 4)          # ← było (2,2,4)
plt.imshow(img_thresh, cmap='gray')
plt.title("4. Progowanie Otsu\n(obraz binarny, THRESH_BINARY_INV)")
plt.axis('off')

# ── NOWE subploty 5 i 6 ───────────────────────────────────────────────
plt.subplot(2, 3, 5)
plt.imshow(img_laplacian, cmap='gray')
plt.title("5. Filtr Laplacjana\n(krawędzie wielokierunkowe, ksize=3)")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(img_median, cmap='gray')
plt.title("6. Filtr medianowy\n(redukcja szumu sól-i-pieprz, ksize=5)")
plt.axis('off')
# ──────────────────────────────────────────────────────────────────────

plt.tight_layout()
plt.show()