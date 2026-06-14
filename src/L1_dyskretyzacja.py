import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

#wczytanie obrazka ze sciezki

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path=os.path.join(script_dir, "..", "data", "raw", "JPEGImages", "BloodImage_00000.jpg")
image_path=os.path.normpath(image_path)

img_bgr = cv2.imread(image_path)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_hsv=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

#dyskretyzacja

#glebia bitowa oryginalneog obraka

bit_depth=img_gray.dtype

#kwanytzacja glebia tonalna od 256 odcieni do 4
poziomy=4
mnoznik=256 // poziomy

img_quantized=(img_gray//mnoznik)*mnoznik


#probkowanie czyli zmniejszenie rozdzielczosci 10krotnie
height, width=img_gray.shape
img_sampled=cv2.resize(img_gray, (width//10, height//10), interpolation=cv2.INTER_NEAREST)

#wiz i histogramy

plt.figure(figsize=(14,10))

#szary orginał
plt.subplot(2,3,1)
plt.imshow(img_gray, cmap="gray")
plt.title(f"Oryginał ({bit_depth})/nRozdzielczość: {img_gray.shape}")
plt.axis("off")


#po kwantyzacji
plt.subplot(2,3,2)
plt.imshow(img_quantized, cmap="gray")
plt.title(f"Kwantyzacja\n  (tylko {poziomy})odcienie szarości)")
plt.axis("off")

#po probkowaniu
plt.subplot(2,3,3)
plt.imshow(img_sampled, cmap="gray")
plt.title(f"Próbkowanie\nRozdzielczość: {img_sampled.shape}")
plt.axis("off")


#histogram oryginału
plt.subplot(2,3,4)
#ravel spłaszcza maceirz2d do listy 1d --> moge policzyc pixele
plt.hist(img_gray.ravel(),bins=256,range=[0,256], color="gray", alpha=0.7)
plt.title("Histogram: Oryginał")
plt.xlabel("Wartośc piksela (0-255)")
plt.ylabel("Liczba pikseli")


#histogram kwantyzacji
plt.subplot(2,3,5)
plt.hist(img_quantized.ravel(), bins=256, range=[0,256], color="black", alpha=0.7)
plt.title("Histogram: kwantyzacja")
plt.xlabel("Wartosć piksela (0-255)")


#wizualizacja kanału hue z przestrzeni HSV
plt.subplot(2,3,6)
#wyciagam tylko kanał hsv czyli same barwy, bez ich jasnosci i
plt.imshow(img_hsv[:,:,0], cmap="hsv")
plt.title("Przestrzeń HSV (tylko kanał Barwy/Hue")
plt.axis("off")

plt.tight_layout()
plt.show()





