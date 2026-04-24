import tensorflow as tf
from tensorflow.keras import layers
import json
import os

img_size = 224
batch_size = 16

# ---------- DATASET ----------
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset-resized",
    image_size=(img_size, img_size),
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset-resized",
    image_size=(img_size, img_size),
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=123
)

class_names = train_data.class_names

with open("model/class_names.json", "w") as f:
    json.dump(class_names, f)

# ---------- PREFETCH ----------
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(AUTOTUNE)
val_data = val_data.prefetch(AUTOTUNE)

# ---------- DATA AUGMENTATION ----------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ---------- BASE MODEL (TRANSFER LEARNING) ----------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # freeze base model

# ---------- MODEL ----------
model = tf.keras.Sequential([
    layers.Rescaling(1./255),
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# ---------- COMPILE ----------
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ---------- TRAIN ----------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)

# ---------- SAVE ----------
model.save("model/model.h5")
print("Model saved successfully 🚀")