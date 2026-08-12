
import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ============================================================
# CONFIGURATION
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

DATASET_PATH = "dataset"
MODEL_PATH = "medicine_defect_model.keras"
CLASS_FILE = "class_names.json"

# ============================================================
# DEFECT DESCRIPTIONS
# ============================================================
DEFECT_INFO = {
    "normal": "No visible defect detected.",
    "pill": "Normal pill detected.",
    "capsule": "Normal capsule detected.",
    "cracked": "Surface crack detected.",
    "broken": "Broken medicine detected.",
    "chipped": "Edge chipping detected.",
    "discolored": "Discoloration defect detected.",
    "contaminated": "Contamination detected.",
    "coating_damage": "Tablet coating damage detected.",
    "capped": "Tablet capping defect detected.",
    "black_spot": "Black spot contamination detected.",
    "rough_surface": "Rough surface defect detected.",
    "dented": "Dented capsule detected."
}

# ============================================================
# CHECK DATASET
# ============================================================
def check_dataset():
    required = ["train", "validation", "test"]

    for folder in required:
        path = os.path.join(DATASET_PATH, folder)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing folder: {path}")

# ============================================================
# LOAD DATASETS
# ============================================================
def load_datasets():
    train_ds = image_dataset_from_directory(
        os.path.join(DATASET_PATH, "train"),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    val_ds = image_dataset_from_directory(
        os.path.join(DATASET_PATH, "validation"),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    test_ds = image_dataset_from_directory(
        os.path.join(DATASET_PATH, "test"),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # `.class_names` only exists on the raw dataset objects returned by
    # image_dataset_from_directory(). Once `.prefetch()` is applied, the
    # resulting `_PrefetchDataset` no longer exposes that attribute, so we
    # must capture the class names here and hand them back explicitly
    # instead of letting callers read `<dataset>.class_names` later.
    train_classes = train_ds.class_names
    val_classes = val_ds.class_names
    test_classes = test_ds.class_names

    if not (train_classes == val_classes == test_classes):
        raise ValueError(
            "Class folders differ between splits:\n"
            f"  train:      {train_classes}\n"
            f"  validation: {val_classes}\n"
            f"  test:       {test_classes}\n"
            "Make sure dataset/train, dataset/validation, and dataset/test "
            "all contain the exact same set of class subfolders."
        )

    return (
        train_ds.prefetch(tf.data.AUTOTUNE),
        val_ds.prefetch(tf.data.AUTOTUNE),
        test_ds.prefetch(tf.data.AUTOTUNE),
        train_classes
    )

# ============================================================
# BUILD CNN MODEL
# ============================================================
def build_model(num_classes):

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = models.Sequential([
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ============================================================
# TRAIN MODEL
# ============================================================
def train_model():

    check_dataset()

    train_ds, val_ds, _, class_names = load_datasets()

    with open(CLASS_FILE, "w") as f:
        json.dump(class_names, f)

    model = build_model(len(class_names))

    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )

    model.save(MODEL_PATH)

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Class names saved: {CLASS_FILE}")

    plot_training(history)

# ============================================================
# PLOT TRAINING GRAPH
# ============================================================
def plot_training(history):

    plt.figure(figsize=(10, 5))

    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Performance")
    plt.legend()

    plt.savefig("training_history.png")

    print("Training graph saved: training_history.png")

# ============================================================
# TEST MODEL
# ============================================================
def test_model():

    check_dataset()

    _, _, test_ds, class_names = load_datasets()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Train the model first.")

    model = tf.keras.models.load_model(MODEL_PATH)

    predictions = model.predict(test_ds)

    y_pred = np.argmax(predictions, axis=1)
    y_true = np.concatenate([y for x, y in test_ds], axis=0)

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(cmap="Blues")

    plt.xticks(rotation=45)

    plt.savefig("confusion_matrix.png")

    print("Confusion matrix saved: confusion_matrix.png")

# ============================================================
# PREDICT SINGLE IMAGE
# ============================================================
def predict_image(image_path):

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Train the model first.")

    if not os.path.exists(CLASS_FILE):
        raise FileNotFoundError("class_names.json missing.")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_FILE, "r") as f:
        class_names = json.load(f)

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Invalid image path.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # NOTE: do not divide by 255 here. The model's first layer is
    # layers.Rescaling(1./255), which already normalizes raw 0-255 pixel
    # values during both training and inference. Normalizing here as well
    # would scale pixels down twice (by 1/65025 total), feeding the model
    # input far outside the range it was trained on and producing
    # unreliable predictions.
    input_img = np.expand_dims(img, axis=0).astype("float32")

    predictions = model.predict(input_img)

    class_index = int(np.argmax(predictions))

    confidence = float(np.max(predictions))

    predicted_class = class_names[class_index]

    print("\n=================================")
    print("PREDICTION RESULT")
    print("=================================")

    print(f"Class       : {predicted_class}")
    print(f"Confidence  : {confidence:.2%}")
    print(f"Description : {DEFECT_INFO.get(predicted_class, 'Unknown defect')}")

# ============================================================
# MAIN MENU
# ============================================================
if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")
        print("python app.py train")
        print("python app.py test")
        print("python app.py predict image.jpg")

        sys.exit()

    command = sys.argv[1].lower()

    if command == "train":
        train_model()

    elif command == "test":
        test_model()

    elif command == "predict":

        if len(sys.argv) < 3:
            print("Please provide image path.")
        else:
            predict_image(sys.argv[2])

    else:
        print("Invalid command.")
