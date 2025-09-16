import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# ========== CONFIG ==========
IMG_SIZE = (128, 128)
EPOCHS = 30
BATCH_SIZE = 8
N_SPLITS = 20
RANDOM_STATE = 42
UNET_OUTPUT_DIR = "Unet"

os.makedirs(UNET_OUTPUT_DIR, exist_ok=True)

# ========== U-Net Model ==========
def build_unet(input_shape=(128, 128, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    b = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    b = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(b)

    # Decoder
    u1 = layers.UpSampling2D((2, 2))(b)
    u1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)

    u2 = layers.UpSampling2D((2, 2))(c3)
    u2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c4)

    # Regression output (1 value)
    x = layers.GlobalAveragePooling2D()(c4)
    outputs = layers.Dense(1, activation='linear')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ========== Data Loader ==========
def load_image(path, target_size):
    img = load_img(path, target_size=target_size)
    return img_to_array(img) / 255.0

# ========== Plotting ==========
def plot_scatter_and_error(y_true, y_pred, name_prefix):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.8)
    plt.xlabel("True ng")
    plt.ylabel("Predicted ng")
    plt.title("U-Net Prediction vs True")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.grid(True)
    plt.savefig(os.path.join(UNET_OUTPUT_DIR, f"{name_prefix}_scatter.png"))
    plt.close()

    # Error bar plot
    abs_error = np.abs(np.array(y_true) - np.array(y_pred))
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, abs_error, alpha=0.7)
    plt.xlabel("True ng")
    plt.ylabel("Absolute Error")
    plt.title("Error Distribution")
    plt.grid(True)
    plt.savefig(os.path.join(UNET_OUTPUT_DIR, f"{name_prefix}_error_bar.png"))
    plt.close()

# ========== Main Training Loop ==========
def train_unet_with_kfold(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df[df["type"] == "test"].copy()

    image_paths = df["file_path"].values
    targets = df["ng"].values

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_preds = []
    fold_truths = []

    fold = 1
    for train_idx, val_idx in kf.split(image_paths):
        print(f"\n📦 Fold {fold}/{N_SPLITS}")
        
        X_train_paths = image_paths[train_idx]
        y_train = targets[train_idx]
        X_val_paths = image_paths[val_idx]
        y_val = targets[val_idx]

        # Load images
        X_train = np.array([load_image(p, IMG_SIZE) for p in tqdm(X_train_paths, desc="Loading train")])
        X_val = np.array([load_image(p, IMG_SIZE) for p in tqdm(X_val_paths, desc="Loading val")])

        model = build_unet(input_shape=IMG_SIZE + (3,))
        model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        y_pred = model.predict(X_val).flatten()
        fold_preds.extend(y_pred)
        fold_truths.extend(y_val)

        print(f"  MAE: {mean_absolute_error(y_val, y_pred):.4f}")
        print(f"  MSE: {mean_squared_error(y_val, y_pred):.4f}")
        print(f"  R² : {r2_score(y_val, y_pred):.4f}")

        fold += 1

    # Final report
    print("\n📊 K-Fold CV Summary")
    print(f"✅ Overall MAE: {mean_absolute_error(fold_truths, fold_preds):.4f}")
    print(f"✅ Overall MSE: {mean_squared_error(fold_truths, fold_preds):.4f}")
    print(f"✅ Overall R² : {r2_score(fold_truths, fold_preds):.4f}")

    # Save output predictions
    result_df = pd.DataFrame({"True": fold_truths, "Pred": fold_preds})
    result_df.to_csv(os.path.join(UNET_OUTPUT_DIR, "unet_cv_predictions.csv"), index=False)

    # Save final model trained on all data
    all_images = np.array([load_image(p, IMG_SIZE) for p in tqdm(image_paths, desc="Final training")])
    model = build_unet(input_shape=IMG_SIZE + (3,))
    model.fit(all_images, targets, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    model.save(os.path.join(UNET_OUTPUT_DIR, "final_unet_model.h5"))
    print("\n💾 Final U-Net model saved.")

    # Plots
    plot_scatter_and_error(fold_truths, fold_preds, name_prefix="unet")


if __name__ == "__main__":
    csv_path = "label.csv"  
    train_unet_with_kfold(csv_path)
