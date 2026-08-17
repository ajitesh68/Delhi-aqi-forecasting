import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

POLLUTANTS = ["pm2_5", "pm10", "co", "no2"]
FORECAST_HOURS = 72


def build_model(input_shape, output_size):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(output_size, activation="linear"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def train_location(loc_tag):
    X_train = np.load(os.path.join(PROCESSED_DIR, f"{loc_tag}_X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, f"{loc_tag}_X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, f"{loc_tag}_y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, f"{loc_tag}_y_test.npy"))

    y_train_flat = y_train.reshape(y_train.shape[0], -1)
    y_test_flat = y_test.reshape(y_test.shape[0], -1)

    input_shape = (X_train.shape[1], X_train.shape[2])
    output_size = y_train_flat.shape[1]

    print(f"Input shape: {input_shape}")
    print(f"Output size: {output_size}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    model = build_model(input_shape, output_size)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train_flat,
        validation_data=(X_test, y_test_flat),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    model_path = os.path.join(MODELS_DIR, f"{loc_tag}_lstm.h5")
    model.save(model_path)
    print(f"Model saved: {model_path}")

    loss, mae = model.evaluate(X_test, y_test_flat, verbose=0)
    print(f"Test Loss (MSE): {loss:.6f}")
    print(f"Test MAE: {mae:.6f}")

    return model, history


def train_all():
    os.makedirs(MODELS_DIR, exist_ok=True)

    locations = set()
    for f in os.listdir(PROCESSED_DIR):
        if f.endswith("_X_train.npy"):
            loc_tag = f.replace("_X_train.npy", "")
            locations.add(loc_tag)

    for loc_tag in sorted(locations):
        model_path = os.path.join(MODELS_DIR, f"{loc_tag}_lstm.h5")
        if os.path.exists(model_path):
            print(f"Skipping {loc_tag}, model already exists.")
            continue

        print(f"\n{'='*60}")
        print(f"Training: {loc_tag}")
        print(f"{'='*60}")
        train_location(loc_tag)


if __name__ == "__main__":
    train_all()
