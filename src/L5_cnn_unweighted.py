import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import os

import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "JPEGImages"))
labels_csv_path = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "labels.csv"))


# --- PARAMETRY ZOPTYMALIZOWANE POD 16GB RAM I BOGATSZĄ AUGMENTACJĘ ---
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 64  # Zwiększone z 32
EPOCHS = 50      # Zwiększone z 15 -> 12 100 obrazów w locie!

print("--- L5: PRZYGOTOWANIE I PODZIAŁ DANYCH DLA 4 OFICJALNYCH KLAS ---")

print("--- L5: PRZYGOTOWANIE I PODZIAŁ DANYCH DLA 4 OFICJALNYCH KLAS ---")
df_labels = pd.read_csv(labels_csv_path)
img_col = df_labels.columns[1]
cat_col = df_labels.columns[2]

df_labels['Clean_File'] = df_labels[img_col].apply(lambda x: f"BloodImage_{int(x):05d}.jpg" if str(x).isdigit() else f"{str(x).strip()}.jpg" if not str(x).lower().endswith('.jpg') else str(x).strip())
df_labels['Class_String'] = df_labels[cat_col].astype(str).str.upper().str.strip()

allowed_classes = ['NEUTROPHIL', 'EOSINOPHIL', 'MONOCYTE', 'LYMPHOCYTE']
df_valid = df_labels[df_labels['Class_String'].isin(allowed_classes)].copy()

# Filtrowanie rekordów w DataFrame - zostawiamy tylko te, które fizycznie istnieją na dysku
# (To naprawi błąd ostrzeżenia o 'invalid image filename')
df_valid['Exists'] = df_valid['Clean_File'].apply(lambda x: os.path.exists(os.path.join(images_dir, x)))
df_valid = df_valid[df_valid['Exists'] == True].copy()

df_train, df_temp = train_test_split(df_valid, test_size=0.30, stratify=df_valid['Class_String'], random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.50, stratify=df_temp['Class_String'], random_state=42)

print(f"Zbiory po odrzuceniu bazofili i braków: Trening: {len(df_train)} | Walidacja: {len(df_val)} | Test: {len(df_test)}")




# JESTEŚMY ODWAŻNIEJSI Z TRANSLACJĄ I ZOOMEEM
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,          # Obrót aż do 45 stopni
    width_shift_range=0.2,      # Mocniejsze przesunięcia (do 20%)
    height_shift_range=0.2,
    zoom_range=0.15,            # NOWOŚĆ: Losowe zbliżenie/oddalenie o 15%
    brightness_range=[0.7, 1.3],# Jeszcze większa rozpiętość jasności
    horizontal_flip=True,
    vertical_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Przekazujemy jawnie listew klas w 'classes', by iteratory idealnie się zsynchronizowały
train_gen = train_datagen.flow_from_dataframe(dataframe=df_train, directory=images_dir, x_col='Clean_File', y_col='Class_String', classes=allowed_classes, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', seed=42)
val_gen = val_test_datagen.flow_from_dataframe(dataframe=df_val, directory=images_dir, x_col='Clean_File', y_col='Class_String', classes=allowed_classes, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', seed=42)
test_gen = val_test_datagen.flow_from_dataframe(dataframe=df_test, directory=images_dir, x_col='Clean_File', y_col='Class_String', classes=allowed_classes, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False, seed=42)



# POPRAWKA BŁĘDU: Pobieramy liczbę klas ze struktury indeksów generatora
num_classes = len(train_gen.class_indices)
print(f"\nKlasy zmapowane w sieci CNN (4 klasy): {list(train_gen.class_indices.keys())}")

print("\n--- L5: PROJEKTOWANIE ARCHITEKTURY MODELU CNN ---")
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
model.summary()

print("\n--- L5: TRENING SIECI NEURONOWEJ ---")
# --- KROK 2: PODMIANKA MODEL.FIT Z UWZGLĘDNIENIEM WAG KLAS ---
print("[INFO] Rozpoczynanie treningu ze zbalansowanymi wagami klas. Logi będą wyświetlane na żywo w pipeline.")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    verbose=2
)


# Wizualizacja wyników uczenia
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Walidacja')
plt.title('Krzywa dokładności (Accuracy)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trening')
plt.plot(history.history['val_loss'], label='Walidacja')
plt.title('Krzywa funkcji straty (Loss)')
plt.legend()


# --- Sekcja zapisu wykresu krzywych uczenia ---
plt.savefig(os.path.join(script_dir, "krzywe_uczenia_cnn_l5_unweighted.png"), bbox_inches='tight', dpi=150)
plt.close()
print("[INFO] Wykres krzywych uczenia został zapisany jako 'krzywe_uczenia_cnn_l5_unweighted.png'.")





# ========================================================================
# --- BLOK EWALUACJI KOŃCOWEJ (MACIERZ POMYŁEK I METRYKI DLA CNN) --------
# ========================================================================
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

print("\n--- L5: URUCHAMIANIE PREDYKCJI NA ZBIORZE TESTOWYM ---")
# Resetujemy generator testowy, aby upewnić się, że czyta próbki od początku
test_gen.reset()

# Pobieramy predykcje z modelu (prawdopodobieństwa dla każdej klasy)
predictions = model.predict(test_gen, verbose=1)
# Zamieniamy prawdopodobieństwa na konkretne indeksy klas (arg max)
predicted_classes = np.argmax(predictions, axis=1)

# Pobieramy prawdziwe indeksy z generatora testowego
true_classes = test_gen.classes
# Pobieramy tekstowe nazwy klas zsynchronizowane z modelami
class_labels = list(test_gen.class_indices.keys())

print("\n================ MACIERZ POMYŁEK - SIEĆ CNN (L5) ================")
cm = confusion_matrix(true_classes, predicted_classes)
df_cm_cnn = pd.DataFrame(
    cm,
    index=[f"Prawdziwy {l}" for l in class_labels],
    columns=[f"Predykcja {l}" for l in class_labels]
)
print(df_cm_cnn)

print("\n================ RAPORT KLASYFIKACJI / METRYKI (L5) ================")
report = classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0)
print(report)
print("========================================================================")

# --- GRAFICZNA MACIERZ POMYŁEK DLA CNN (L5) ---
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

print("\n--- L5: URUCHAMIANIE PREDYKCJI NA ZBIORZE TESTOWYM ---")
test_gen.reset()
predictions = model.predict(test_gen, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

cm = confusion_matrix(true_classes, predicted_classes)

# Rysowanie kolorowej macierzy
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Macierz Pomyłek - Sieć CNN (L5)')
plt.ylabel('Rzeczywista klasa')
plt.xlabel('Predykcja')

# Zapisujemy na dysk
plt.savefig(os.path.join(script_dir, "macierz_pomylek_cnn_l5.png"), bbox_inches='tight', dpi=150)
plt.close()

print("\n================ RAPORT KLASYFIKACJI / METRYKI (L5) ================")
print(classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0))


# --- Sekcja zapisu graficznej macierzy pomyłek (na samym dole skryptu) ---
plt.savefig(os.path.join(script_dir, "macierz_pomylek_cnn_l5_unweighted.png"), bbox_inches='tight', dpi=150)
plt.close()