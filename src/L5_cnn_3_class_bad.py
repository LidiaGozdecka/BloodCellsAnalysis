import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import random
SEED = 2026
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "JPEGImages"))
labels_csv_path = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "labels.csv"))

# --- PARAMETRY ---
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 64
EPOCHS = 50

print("--- L5 (3-KLASY): PRZYGOTOWANIE I BIOLOGICZNA FUZJA KLAS ---")
df_labels = pd.read_csv(labels_csv_path)
img_col = df_labels.columns[1]
cat_col = df_labels.columns[2]

df_labels['Clean_File'] = df_labels[img_col].apply(lambda x: f"BloodImage_{int(x):05d}.jpg" if str(x).isdigit() else f"{str(x).strip()}.jpg" if not str(x).lower().endswith('.jpg') else str(x).strip())
df_labels['Raw_Class'] = df_labels[cat_col].astype(str).str.upper().str.strip()

# --- BIOLOGICZNA FUZJA KLAS (Agranulocyty) ---
def merge_to_3_classes(class_name):
    if class_name in ['LYMPHOCYTE', 'MONOCYTE']:
        return 'AGRANULOCYTE'  # Łączymy komórki jednojądrzaste
    elif class_name in ['NEUTROPHIL', 'EOSINOPHIL']:
        return class_name
    return 'IGNORE'

df_labels['Class_String'] = df_labels['Raw_Class'].apply(merge_to_3_classes)

allowed_classes = ['NEUTROPHIL', 'EOSINOPHIL', 'AGRANULOCYTE']
df_valid = df_labels[df_labels['Class_String'].isin(allowed_classes)].copy()

# Weryfikacja fizycznego istnienia plików na dysku
df_valid['Exists'] = df_valid['Clean_File'].apply(lambda x: os.path.exists(os.path.join(images_dir, x)))
df_valid = df_valid[df_valid['Exists'] == True].copy()

# Stratyfikacja na 3 klasy gwarantuje idealny podział zbioru
df_train, df_temp = train_test_split(df_valid, test_size=0.30, stratify=df_valid['Class_String'], random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.50, stratify=df_temp['Class_String'], random_state=42)

print(f"Zbiory po fuzji i podziale: Trening: {len(df_train)} | Walidacja: {len(df_val)} | Test: {len(df_test)}")

# Konfiguracja augmentacji
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.15,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Iteratory zsynchronizowane z nową listą 3 klas
train_gen = train_datagen.flow_from_dataframe(dataframe=df_train, directory=images_dir, x_col='Clean_File', y_col='Class_String', classes=allowed_classes, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', seed=42)
val_gen = val_test_datagen.flow_from_dataframe(dataframe=df_val, directory=images_dir, x_col='Clean_File', y_col='Class_String', classes=allowed_classes, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', seed=42)
test_gen = val_test_datagen.flow_from_dataframe(dataframe=df_test, directory=images_dir, x_col='Clean_File', y_col='Class_String', classes=allowed_classes, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False, seed=42)

print("\n[INFO] Obliczanie wag dla 3 skonsolidowanych klas...")
train_labels = train_gen.classes
unique_classes = np.unique(train_labels)
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=unique_classes,
    y=train_labels
)
class_weight_dict = dict(zip(unique_classes, class_weights_array))

for cls_idx, weight in class_weight_dict.items():
    class_name = list(train_gen.class_indices.keys())[list(train_gen.class_indices.values()).index(cls_idx)]
    print(f" - Klasa {class_name} (Indeks {cls_idx}): waga {weight:.4f}")

num_classes = len(train_gen.class_indices)

print("\n--- L5 (3-KLASY): BUDOWA MODELU CNN ---")
model = Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- L5 (3-KLASY): TRENING SIECI NEURONOWEJ ---")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weight_dict,
    verbose=2
)

# ========================================================================
# --- ZAPIS WYKRESÓW UCZENIA (WERSJA 3-KLASOWA) --------------------------
# ========================================================================
print("\n[INFO] Zapisywanie krzywych uczenia...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Walidacja')
plt.title('Krzywa dokładności (Accuracy) - 3 Klasy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trening')
plt.plot(history.history['val_loss'], label='Walidacja')
plt.title('Krzywa funkcji straty (Loss) - 3 Klasy')
plt.legend()

# Bezpieczny, niezależny zapis pliku
plt.savefig(os.path.join(script_dir, "krzywe_uczenia_cnn_l5_3class.png"), bbox_inches='tight', dpi=150)
plt.close()
print("[INFO] Wykres krzywych uczenia CNN (3 klasy) został zapisany jako 'krzywe_uczenia_cnn_l5_3class.png'.")

# ========================================================================
# --- EWALUACJA KOŃCOWEJ PREDYKCJI I MACIERZY POMYŁEK (3 KLASY) ---------
# ========================================================================
print("\n--- L5 (3-KLASY): URUCHAMIANIE PREDYKCJI NA ZBIORZE TESTOWYM ---")
test_gen.reset()

predictions = model.predict(test_gen, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

print("\n================ MACIERZ POMYŁEK - SIEĆ CNN (L5 - 3 KLASY) ================")
cm = confusion_matrix(true_classes, predicted_classes)
df_cm_cnn = pd.DataFrame(
    cm,
    index=[f"Prawdziwy {l}" for l in class_labels],
    columns=[f"Predykcja {l}" for l in class_labels]
)
print(df_cm_cnn)

# --- GRAFICZNA MACIERZ POMYŁEK DLA CNN (3 KLASY) ---
print("\n[INFO] Zapisywanie graficznej macierzy pomyłek (Teal)...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Macierz Pomyłek CNN - Po Fuzji Biologicznej (3 Klasy)')
plt.ylabel('Rzeczywista klasa')
plt.xlabel('Predykcja')

# Bezpieczny, niezależny zapis pliku graficznego
plt.savefig(os.path.join(script_dir, "macierz_pomylek_cnn_l5_3class.png"), bbox_inches='tight', dpi=150)
plt.close()
print("[INFO] Graficzna macierz pomyłek została zapisana jako 'macierz_pomylek_cnn_l5_3class.png'.")

print("\n================ RAPORT KLASYFIKACJI / METRYKI (L5 - 3 KLASY) ================")
print(classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0))
print("========================================================================")