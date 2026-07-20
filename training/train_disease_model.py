"""
Crop Disease Detection — Model Training Script
================================================
DESIGNED TO RUN ON GOOGLE COLAB (free GPU tier). Do not run this locally
unless you have a GPU — it will be painfully slow on CPU.

How to use:
1. Go to https://colab.research.google.com/ -> New Notebook
2. Runtime -> Change runtime type -> select "T4 GPU" (free tier)
3. Paste this entire file into a cell (or upload it and !python train_disease_model.py)
4. Run it. Training ~38-class EfficientNetB0 on PlantVillage takes ~30-45 min
   on a free Colab T4 GPU.
5. Download the resulting `crop_disease_model.keras` and `class_names.json`
   at the end — you'll copy both into app/models/ for the inference step.

Dataset: PlantVillage, pulled from Kaggle (abdallahalidev/plantvillage-dataset)
— public, free, requires only a free Kaggle account + API token (kaggle.json).
~54,000 labeled leaf images, 38 classes covering 14 crop species with
healthy/diseased variants. (Note: we use Kaggle rather than TensorFlow
Datasets' built-in "plant_village" config because that config downloads
from Mendeley's servers, which frequently reject automated requests with
a 403 error — Kaggle is a more reliable free host for the same data.)
"""

import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_STAGE1 = 8   # frozen base — train just the new classification head
EPOCHS_STAGE2 = 6   # fine-tune the top layers of EfficientNet
LEARNING_RATE_STAGE1 = 1e-3
LEARNING_RATE_STAGE2 = 1e-5

# ---------------------------------------------------------------------------
# 2. Load dataset — via Kaggle (TFDS's plant_village config depends on
#    Mendeley's servers, which frequently 403 automated downloads; Kaggle
#    is a far more reliable free host for the same data).
#
#    BEFORE running this cell: upload your kaggle.json API token via
#      from google.colab import files
#      files.upload()
#    (get kaggle.json free from kaggle.com -> Settings -> API -> Create New Token)
# ---------------------------------------------------------------------------
import os
import pathlib

print("Setting up Kaggle credentials...")
os.makedirs("/root/.kaggle", exist_ok=True)
os.system("cp kaggle.json /root/.kaggle/kaggle.json")
os.system("chmod 600 /root/.kaggle/kaggle.json")

print("Downloading PlantVillage dataset from Kaggle (this takes a few minutes)...")
os.system("pip install -q kaggle")
os.system("kaggle datasets download -d abdallahalidev/plantvillage-dataset -p /content/data --unzip")

DATA_DIR = pathlib.Path("/content/data")
# This Kaggle dataset nests the actual class folders one level in —
# find the directory that actually contains the class subfolders.
candidate_dirs = [p for p in DATA_DIR.rglob("*") if p.is_dir() and
                   any(sub.is_dir() for sub in p.iterdir())]
# Heuristic: the real image root has many (30+) subdirectories (the classes)
image_root = max(candidate_dirs, key=lambda p: len(list(p.iterdir())))
print(f"Using image root: {image_root}")
print(f"Detected {len(list(image_root.iterdir()))} class folders")

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    image_root,
    validation_split=0.15,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)
val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    image_root,
    validation_split=0.15,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

class_names = train_ds_raw.class_names
num_classes = len(class_names)
print(f"Loaded {num_classes} classes.")

# ---------------------------------------------------------------------------
# 3. Preprocessing + augmentation
#    Augmentation matters a lot here — real farmer photos are rarely clean,
#    well-lit, top-down leaf shots like the training set, so we simulate
#    that variability during training.
#
#    Strengthened vs. the original version: wider zoom range (including
#    zooming OUT, since real photos often have the leaf as a smaller part
#    of a busier frame — the original only zoomed in), added translation
#    (real photos are rarely perfectly centered), and cutout (simulates
#    partial occlusion — a shadow, another leaf, a hand — covering part of
#    the leaf, which is common in real-world photos but never happens in
#    PlantVillage's clean lab shots).
# ---------------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(height_factor=(-0.25, 0.15)),  # both zoom in AND out
    layers.RandomTranslation(height_factor=0.15, width_factor=0.15),
    layers.RandomContrast(0.25),
    layers.RandomBrightness(0.25),
])


def _random_cutout(image, label, cutout_size=40, prob=0.3):
    """Randomly blacks out a square patch — simulates a shadow, another
    leaf, or a hand partially covering the subject, which real-world
    photos have often but PlantVillage's lab photos never do."""
    if tf.random.uniform([]) > prob:
        return image, label
    img_shape = tf.shape(image)
    h, w = img_shape[0], img_shape[1]
    y = tf.random.uniform([], 0, h - cutout_size, dtype=tf.int32)
    x = tf.random.uniform([], 0, w - cutout_size, dtype=tf.int32)
    mask = tf.ones((cutout_size, cutout_size, 3), dtype=image.dtype)
    padded_mask = tf.pad(
        mask,
        [[y, h - y - cutout_size], [x, w - x - cutout_size], [0, 0]],
    )
    image = image * (1 - padded_mask)
    return image, label


def prepare(ds, training=False):
    if training:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(_random_cutout, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = prepare(train_ds_raw, training=True)
val_ds = prepare(val_ds_raw, training=False)

# ---------------------------------------------------------------------------
# 4. Build model — transfer learning on EfficientNetB0
# ---------------------------------------------------------------------------
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base_model.trainable = False  # frozen for stage 1

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_STAGE1),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ---------------------------------------------------------------------------
# 5. Stage 1 — train the classification head only
# ---------------------------------------------------------------------------
print("\n=== Stage 1: training classification head (base frozen) ===")
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
]
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE1, callbacks=callbacks)

# ---------------------------------------------------------------------------
# 6. Stage 2 — unfreeze top layers of EfficientNet, fine-tune at low LR
#    This is what actually pushes accuracy from "decent" to "good" —
#    skipping this step is the most common reason transfer-learning
#    projects underperform.
# ---------------------------------------------------------------------------
print("\n=== Stage 2: fine-tuning top layers ===")
base_model.trainable = True
# Freeze all but the last ~20 layers — keeps early generic features intact,
# only adapts the more specialized late layers to leaf-disease patterns.
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_STAGE2),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE2, callbacks=callbacks)

# ---------------------------------------------------------------------------
# 7. Evaluate + save
# ---------------------------------------------------------------------------
loss, acc = model.evaluate(val_ds)
print(f"\nFinal validation accuracy: {acc:.4f}")

model.save("crop_disease_model.keras")
with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)

print("\nSaved crop_disease_model.keras and class_names.json")
print("Download both from the Colab file browser (left sidebar) and copy")
print("them into your project's app/models/ directory.")

# ---------------------------------------------------------------------------
# 8. (Optional but recommended) Convert to TFLite for faster/free CPU
#    inference in production — full Keras model works fine too, this is
#    an optimization if you want lower latency / smaller deployment size.
# ---------------------------------------------------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("crop_disease_model.tflite", "wb") as f:
    f.write(tflite_model)
print("Also saved crop_disease_model.tflite (smaller, faster CPU inference).")
