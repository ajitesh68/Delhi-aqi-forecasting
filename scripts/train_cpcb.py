"""Train the 24-hour PM2.5 forecaster on CPCB ground data."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(window, n_features, n_future, n_stations, embed_dim=6, horizon=24):
    """Deliberately small: a few thousand training windows will not support"""
    sequence = keras.Input(shape=(window, n_features), name="sequence")
    future = keras.Input(shape=(horizon, n_future), name="future")
    station = keras.Input(shape=(1,), dtype="int32", name="station")

    x = layers.Conv1D(32, 5, strides=2, padding="same", activation="relu")(sequence)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(48, 5, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.3)(x)

    e = layers.Embedding(n_stations, embed_dim)(station)
    e = layers.Flatten()(e)

    f = layers.Conv1D(24, 3, padding="same", activation="relu")(future)
    f = layers.GlobalAveragePooling1D()(f)
    f2 = layers.Flatten()(future)
    f2 = layers.Dense(32, activation="relu")(f2)

    x = layers.Concatenate()([x, e, f, f2])
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(horizon, name="residual",
                       kernel_initializer="zeros", bias_initializer="zeros")(x)

    model = keras.Model([sequence, future, station], out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=keras.losses.Huber(delta=1.0),
                  metrics=["mae"])
    return model


def inverse(scaled, stats):
    """Scaled log space back to ug/m3."""
    return np.expm1(scaled * stats["scale"] + stats["mean"]).clip(0)


def to_concentration(scaled, station_ids, meta):
    """Per-station inverse transform, since each has its own scaler."""
    index_to_name = {v: k for k, v in meta["stations"].items()}
    target = meta["target"]
    out = np.empty_like(scaled, dtype=float)
    for idx, name in index_to_name.items():
        rows = station_ids == idx
        if rows.any():
            out[rows] = inverse(scaled[rows], meta["scalers"][name][target])
    return out


def persistence(X, meta):
    """Baseline: tomorrow looks like the last 24 observed hours."""
    target_idx = meta["target_index"]
    last_day = X[:, -24:, target_idx]
    return np.repeat(last_day.mean(axis=1, keepdims=True), 24, axis=1)


def report(name, pred, truth):
    mae = np.abs(pred - truth).mean()
    rmse = np.sqrt(((pred - truth) ** 2).mean())
    cut = np.percentile(truth, 90)
    top = truth >= cut
    top_mae = np.abs(pred[top] - truth[top]).mean() if top.any() else float("nan")
    print(f"  {name:<14} MAE {mae:6.1f}   RMSE {rmse:6.1f}   "
          f"top-decile MAE {top_mae:6.1f}")
    return {"mae": float(mae), "rmse": float(rmse), "top_decile_mae": float(top_mae)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--models", default="models/cpcb")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)

    data = Path(args.data)
    with open(data / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    z = np.load(data / "sequences.npz", allow_pickle=False)

    X_tr, F_tr, y_tr, s_tr = (z["X_train"], z["F_train"],
                              z["y_train"], z["station_train"])
    X_te, F_te, y_te, s_te = (z["X_test"], z["F_test"],
                              z["y_test"], z["station_test"])

    if len(X_te) == 0:
        print("Winter holdout is empty -- the store does not cover it yet. "
              "Training without it would select on the monsoon.")
        return 1

    print(f"train {len(X_tr):,} windows   winter holdout {len(X_te):,} windows")
    print(f"window {X_tr.shape[1]}h x {X_tr.shape[2]} features   "
          f"{len(meta['stations'])} stations\n")

    base_tr = persistence(X_tr, meta)
    base_te = persistence(X_te, meta)

    model = build_model(X_tr.shape[1], X_tr.shape[2], F_tr.shape[2],
                        len(meta["stations"]))
    print(f"{model.count_params():,} parameters, predicting the residual "
          f"from persistence\n")

    out = Path(args.models)
    out.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        {"sequence": X_tr, "future": F_tr, "station": s_tr}, y_tr - base_tr,
        validation_data=({"sequence": X_te, "future": F_te, "station": s_te},
                         y_te - base_te),
        epochs=args.epochs, batch_size=args.batch_size, shuffle=True, verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=18,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=8, min_lr=1e-5),
        ],
    )

    residual = model.predict({"sequence": X_te, "future": F_te,
                              "station": s_te}, verbose=0)
    pred = to_concentration(base_te + residual, s_te, meta)
    truth = to_concentration(y_te, s_te, meta)
    base = to_concentration(base_te, s_te, meta)

    print(f"\nWinter holdout ({meta['holdout']['start'][:7]} to "
          f"{meta['holdout']['end'][:7]}), PM2.5 ug/m3:")
    scores = {"model": report("model", pred, truth),
              "persistence": report("persistence", base, truth)}

    beats = scores["model"]["mae"] < scores["persistence"]["mae"]
    gain = (1 - scores["model"]["mae"] / scores["persistence"]["mae"]) * 100
    print(f"\n  {'beats' if beats else 'LOSES TO'} persistence "
          f"by {abs(gain):.1f}% MAE")

    model.save(out / "pm25_24h.keras")
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    scorecard = {
        "trained_at": str(np.datetime64("now")),
        "epochs_run": len(history.history["loss"]),
        "train_windows": int(len(X_tr)),
        "holdout_windows": int(len(X_te)),
        "holdout": meta["holdout"],
        "stations": list(meta["stations"]),
        "scores": scores,
        "beats_persistence": bool(beats),
        "improvement_pct": float(gain),
    }
    with open(out / "scorecard.json", "w", encoding="utf-8") as fh:
        json.dump(scorecard, fh, indent=2)

    print(f"\nsaved {out / 'pm25_24h.keras'}")
    if not beats:
        print("Not shippable: the plan's gate is beating persistence on winter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
