import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# 2. Ścieżki do danych
script_dir = os.path.dirname(os.path.abspath(__file__))
# UWAGA: Sieci CNN trenuje się na CAŁYM zbiorze danych. Zakładamy strukturę,
# gdzie w data/raw/ masz foldery z podziałem na klasy lub używamy generatora.
# Na potrzeby tego skryptu zakładamy, że obrazy są odpowiednio skategoryzowane.
data_dir = os.path.normpath(os.path.join(script_dir, "..", "data", "raw"))

# Parametry obrazu i uczenia
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 10

print("--- L5: PRZYGOTOWANIE DANYCH (AUGMENTACJA I NORMALIZACJA) ---")
# Generatory danych z augmentacją (obroty, przesunięcia, odbicia) i normalizacją (1./255)
# Wykorzystujemy wbudowany validation_split do podziału zbioru (Wymóg z PDF)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.30  # Rezerwujemy 30% na walidację i testy
)

# Generator dla zbioru treningowego (70%)
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=42
)

# Generator dla zbioru walidacyjnego (połowa z pozostałych 30% = 15%)
val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=42
)

# Pobieramy liczbę klas z folderów (np. Neutrophil, Lymphocyte, Monocyte, Eosinophil)
num_classes = train_generator.num_classes
print(f"Wykryte klasy komórek: {list(train_generator.class_indices.keys())}")

print("\n--- L5: PROJEKTOWANIE ARCHITEKTURY SIECI CNN ---")
# Budowa modelu sekwencyjnego
model = Sequential([
    # 1. Blok Konwolucyjny
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),

    # 2. Blok Konwolucyjny
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # 3. Blok Konwolucyjny
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Spłaszczenie macierzy cech do wektora 1D
    Flatten(),

    # Warstwa gęsta (Dense) z regularyzacją Dropout przeciwdziałającą overfittingowi
    Dense(128, activation='relu'),
    Dropout(0.5),

    # Warstwa wyjściowa - softmax zwraca prawdopodobieństwa dla każdej z klas
    Dense(num_classes, activation='softmax')
])

# Kompilacja modelu z optymalizatorem Adam i funkcją straty dla wielu klas
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Wyświetlenie struktury sieci w konsoli
model.summary()

print("\n--- L5: URUCHOMIENIE PROCESU UCZENIA SIECI ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# Zapisanie wytrenowanego modelu (Wymóg reprodukowalności z PDF)
model_output_path = os.path.join(script_dir, "saved_models")
os.makedirs(model_output_path, exist_ok=True)
model.save(os.path.join(model_output_path, "blood_cell_cnn_v1.h5"))
print(f"\nModel został zapisany w folderze: {model_output_path}")

print("\n--- L5: WIZUALIZACJA KRZYWYCH UCZENIA ---")
plt.figure(figsize=(12, 5))

# Wykres dokładności (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Walidacja')
plt.title('Dokładność modelu (Accuracy)')
plt.xlabel('Epoka')
plt.ylabel('Accuracy')
plt.legend()

# Wykres funkcji straty (Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trening')
plt.plot(history.history['val_loss'], label='Walidacja')
plt.title('Funkcja straty (Loss)')
plt.xlabel('Epoka')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()