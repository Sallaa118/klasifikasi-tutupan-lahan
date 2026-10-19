
from pathlib import Path
from io import BytesIO
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

import tensorflow as tf
from tensorflow.keras import layers, models

# Agar hasil eksperimen lebih mudah direproduksi
np.random.seed(42)
tf.random.set_seed(42)

DATA_DIR = Path("data")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

# Urutan label EuroSAT standar
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

IMAGE_SIZE = (32, 32)
TRAIN_PER_CLASS = 300
VAL_PER_CLASS = 100
TEST_PER_CLASS = 100


def load_subset(filename, samples_per_class):
    """Membaca subset gambar secara seimbang dari Parquet."""
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    df = pd.read_parquet(path)
    selected = []

    for label in sorted(df["label"].unique()):
        group = df[df["label"] == label]

        n = min(samples_per_class, len(group))
        selected.append(
            group.sample(n=n, random_state=42)
        )

    df = pd.concat(selected).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    images = []
    labels = []

    for _, row in df.iterrows():
        image_data = row["image"]

        # Data gambar tersimpan sebagai byte JPEG
        with Image.open(BytesIO(image_data["bytes"])) as img:
            img = img.convert("RGB")
            img = img.resize(IMAGE_SIZE)
            images.append(np.asarray(img, dtype=np.uint8))

        labels.append(int(row["label"]))

    return np.array(images), np.array(labels)


print("Memuat subset dataset EuroSAT...")

X_train, y_train = load_subset(
    "train-00000-of-00001.parquet",
    TRAIN_PER_CLASS
)

X_val, y_val = load_subset(
    "validation-00000-of-00001.parquet",
    VAL_PER_CLASS
)

X_test, y_test = load_subset(
    "test-00000-of-00001.parquet",
    TEST_PER_CLASS
)

print("Jumlah train:", len(X_train))
print("Jumlah validation:", len(X_val))
print("Jumlah test:", len(X_test))
print("Ukuran gambar:", IMAGE_SIZE)

# Normalisasi piksel untuk CNN
X_train_norm = X_train.astype("float32") / 255.0
X_val_norm = X_val.astype("float32") / 255.0
X_test_norm = X_test.astype("float32") / 255.0

results = []
predictions = {}

# ==========================================
# MODEL 1: RANDOM FOREST
# ==========================================
print("\nMelatih Random Forest...")

# Mengubah gambar menjadi vektor fitur piksel
X_train_rf = X_train.reshape(len(X_train), -1)
X_test_rf = X_test.reshape(len(X_test), -1)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

start = time.time()
rf.fit(X_train_rf, y_train)
rf_time = time.time() - start

rf_pred = rf.predict(X_test_rf)
predictions["Random Forest"] = rf_pred

results.append({
    "Model": "Random Forest",
    "Accuracy": accuracy_score(y_test, rf_pred),
    "Precision_Macro": precision_score(
        y_test, rf_pred, average="macro", zero_division=0
    ),
    "Recall_Macro": recall_score(
        y_test, rf_pred, average="macro", zero_division=0
    ),
    "F1_Macro": f1_score(
        y_test, rf_pred, average="macro", zero_division=0
    ),
    "Training_Time_Seconds": rf_time
})

print("Random Forest selesai.")
print("Waktu pelatihan:", round(rf_time, 2), "detik")

# ==========================================
# MODEL 2: CONVOLUTIONAL NEURAL NETWORK
# ==========================================
print("\nMembangun CNN...")

cnn = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(CLASS_NAMES), activation="softmax")
])

cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Melatih CNN...")

start = time.time()

history = cnn.fit(
    X_train_norm,
    y_train,
    validation_data=(X_val_norm, y_val),
    epochs=5,
    batch_size=32,
    verbose=1
)

cnn_time = time.time() - start

cnn_pred = np.argmax(
    cnn.predict(X_test_norm, verbose=0),
    axis=1
)

predictions["CNN"] = cnn_pred

results.append({
    "Model": "CNN",
    "Accuracy": accuracy_score(y_test, cnn_pred),
    "Precision_Macro": precision_score(
        y_test, cnn_pred, average="macro", zero_division=0
    ),
    "Recall_Macro": recall_score(
        y_test, cnn_pred, average="macro", zero_division=0
    ),
    "F1_Macro": f1_score(
        y_test, cnn_pred, average="macro", zero_division=0
    ),
    "Training_Time_Seconds": cnn_time
})

# ==========================================
# EVALUASI DAN PENYIMPANAN HASIL
# ==========================================
results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_DIR / "perbandingan_model.csv",
    index=False
)

print("\n=== HASIL PERBANDINGAN MODEL ===")
print(results_df.round(4).to_string(index=False))

for model_name, pred in predictions.items():
    print(f"\n=== Classification Report: {model_name} ===")
    print(classification_report(
        y_test,
        pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0
    ))

# Confusion matrix kedua model
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, (model_name, pred) in zip(
    axes, predictions.items()
):
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred,
        labels=list(range(len(CLASS_NAMES))),
        display_labels=CLASS_NAMES,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.set_title(model_name)

plt.tight_layout()
plt.savefig(
    RESULT_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()

# Grafik akurasi dan F1-score
fig, ax = plt.subplots(figsize=(8, 5))
results_df.set_index("Model")[
    ["Accuracy", "F1_Macro"]
].plot(kind="bar", ax=ax)

ax.set_ylim(0, 1)
ax.set_ylabel("Skor")
ax.set_title("Perbandingan Accuracy dan Macro F1-score")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(
    RESULT_DIR / "perbandingan_grafik.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()

print("\nEksperimen selesai!")
print("Hasil tersimpan di folder results/")
