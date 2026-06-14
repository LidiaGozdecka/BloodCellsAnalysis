import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import confusion_matrix, classification_report

import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "JPEGImages"))
labels_csv_path = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "labels.csv"))

IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32
EPOCHS = 25  # Modele pre-trenowane uczą się znacznie szybciej, 25 epok w zupełności wystarczy

print("--- L5 (3-KLASY): BIOLOGICZNA FUZJA KLAS ---")
df_labels = pd.read_csv(labels_csv_path)
img_col = df_labels.columns[1]
cat_col = df_labels.columns[2]

df_labels['Clean_File'] = df_labels[img_col].apply(lambda x: f"BloodImage_{int(x):05d}.jpg" if str(x).isdigit() else f"{str(x).strip()}.jpg" if not str(x).lower().endswith('.jpg') else str(x).strip())
df_labels['Raw_Class'] = df_labels[cat_col].astype(str).str.upper().str.strip()

def merge_to_3_classes(class_name):
    if class_name in ['LYMPHOCYTE', 'MONOCYTE']:
        return 'AGRANULOCYTE'
    elif class_name in ['NEUTROPHIL', 'EOSINOPHIL']:
        return class_name
    return 'IGNORE'

df_labels['Class_String'] = df_labels['Raw_Class'].apply(merge_to_3_classes)

allowed_classes = ['NEUTROPHIL', 'EOSINOPHIL', 'AGRANULOCYTE']
df_valid = df_labels[df_labels['Class_String'].isin(allowed_classes)].copy()

df_valid['Exists'] = df_valid['Clean_File'].apply(lambda x: os.path.exists(os.path.join(images_dir, x)))
df_valid = df_valid[df_valid['Exists'] == True].copy()

# Podział pierwotny
df_train_raw, df_temp = train_test_split(df_valid, test_size=0.30, stratify=df_valid['Class_String'], random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.50, stratify=df_temp['Class_String'], random_state=42)

# --- MECHANIZM OVERSAMPLINGU ---
print("\n[INFO] Równoważenie zbioru treningowego przez Oversampling...")
max_size = df_train_raw['Class_String'].value_counts().max()
lst = []
for class_name in allowed_classes:
    df_class = df_train_raw[df_train_raw['Class_String'] == class_name]
    df_class_over = df_class.sample(max_size, replace=True, random_state=42)
    lst.append(df_class_over)
#df_train = pd.concat(lst, axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
df_train = pd.concat(lst, axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Zbiory po OVERSAMPLINGU: Trening: {len(df_train)} | Walidacja: {len(df_val)} | Test: {len(df_test)}")

# Augmentacja dostosowana do Transfer Learning
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(dataframe=df_train, directory=images_dir, x_col='Clean_File', y_col='Class_String', classes=allowed_classes, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', seed=42)
val_gen = val_test_datagen.flow_from_dataframe(dataframe=df_val, directory=images_dir, x_col='Clean_File', y_col='Class_String', classes=allowed_classes, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', seed=42)
test_gen = val_test_datagen.flow_from_dataframe(dataframe=df_test, directory=images_dir, x_col='Clean_File', y_col='Class_String', classes=allowed_classes, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False, seed=42)

num_classes = len(train_gen.class_indices)

print("\n--- L5 (3-KLASY): INICJALIZACJA TRANSFER LEARNING (MobileNetV2) ---")
# Importujemy gotową, potężną sieć przeszkoloną na ImageNet
base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')

# Zamrażamy warstwy bazowe - nie pozwalamy im na zmianę wag, bo już potrafią świetnie widzieć
base_model.trainable = False

# Budujemy nowy klasyfikator dedykowany dla Twoich krwinek
inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

# Używamy małego learning rate, by precyzyjnie dostroić nasz klasyfikator
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- L5 (3-KLASY): TRENING SIECI NEURONOWEJ ---")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    verbose=2
)

# --- ZAPIS WYKRESÓW UCZENIA ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Walidacja')
plt.title('Krzywa dokładności (Accuracy) - Transfer Learning')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trening')
plt.plot(history.history['val_loss'], label='Walidacja')
plt.title('Krzywa funkcji straty (Loss) - Transfer Learning')
plt.legend()

plt.savefig(os.path.join(script_dir, "krzywe_uczenia_cnn_l5_3class.png"), bbox_inches='tight', dpi=150)
plt.close()

# --- GENEROWANIE PREDYKCJI I METRYK FINALNYCH ---
print("\n--- L5 (3-KLASY): GENEROWANIE PREDYKCJI I METRYK FINALNYCH ---")
test_gen.reset()
predictions = model.predict(test_gen, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Macierz Pomyłek CNN - Transfer Learning (3 Klasy)')
plt.ylabel('Rzeczywista klasa')
plt.xlabel('Predykcja')

plt.savefig(os.path.join(script_dir, "macierz_pomylek_cnn_l5_3class.png"), bbox_inches='tight', dpi=150)
plt.close()

print("\n================ RAPORT KLASYFIKACJI / METRYKI (L5 - 3 KLASY) ================")
print(classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0))
print("========================================================================")