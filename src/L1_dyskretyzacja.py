import cv2
import matplpotlib.pyplot as plt
import numpy as np
import os

#wczytanie obrazka ze sciezki

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path=os.path.join(script_dir, "..", "data", "raw", "JPEGImages", "BloodImage_00000.jpg")
image_path=os.path.normpath(image_path)

img_bgr = cv2.imread(image_path)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img._hsv=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)