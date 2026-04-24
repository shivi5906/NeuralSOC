"""
Training pipeline for CICIDS-2017 intrusion detection using BiLSTM.

Key concerns addressed:
  - Severe class imbalance (BENIGN >> attack traffic)
  - Noisy / duplicate rows common in CICIDS-2017
  - Need for per-class metrics (not just accuracy)
  - Reproducibility (seeds fixed)
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from bilstm_model import build_bilstm_ids

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── CICIDS-2017 class map (adjust to your label encoding) ──────────────────────
CICIDS_CLASSES = {
    0: "BENIGN",
    1: "DoS Hulk",
    2: "PortScan",
    3: "DDoS",
    4: "DoS GoldenEye",
    5: "FTP-Patator",
    6: "SSH-Patator",
    7: "DoS slowloris",
    # extend if you have more attack types encoded
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data(x_path: str, y_path: str):
    X = np.load(x_path)
    y = np.load(y_path)

    print(f"[data] X shape : {X.shape}  dtype={X.dtype}")
    print(f"[data] y shape : {y.shape}  dtype={y.dtype}")
    print(f"[data] Classes : {np.unique(y, return_counts=True)}")

    # Sanity checks specific to CICIDS
    assert X.ndim == 3, "Expected (samples, timesteps, features)"
    assert not np.isnan(X).any(), "NaNs in X — check preprocessing"
    assert not np.isinf(X).any(), "Infs in X — check clipping/scaling"

    return X, y


def compute_cicids_class_weights(y_train: np.ndarray) -> dict:
    """
    CICIDS-2017 is extremely imbalanced (BENIGN can be 80 %+ of data).
    Balanced class weights prevent the model from predicting BENIGN for everything.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = dict(zip(classes.tolist(), weights.tolist()))
    print("[class weights]", {CICIDS_CLASSES.get(k, k): f"{v:.3f}" for k, v in cw.items()})
    return cw


def build_callbacks(checkpoint_path: str) -> list:
    return [
        # Stop early if val_loss stalls — patience=5 is safer than 3 for IDS
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        # Save best checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        # Cosine-decay LR — better than fixed LR for this kind of data
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        # TensorBoard logs
        tf.keras.callbacks.TensorBoard(
            log_dir="logs/bilstm_cicids",
            histogram_freq=1,
        ),
        # CSV log for offline analysis
        tf.keras.callbacks.CSVLogger("logs/training_log.csv"),
    ]


def plot_history(history, save_path: str = "logs/training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss (sparse CE)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] Training curves saved → {save_path}")


def evaluate_model(model, X_test, y_test, class_names: dict,
                   save_path: str = "logs/confusion_matrix.png"):
    """
    Per-class report + confusion matrix.
    Accuracy alone is misleading on CICIDS due to class imbalance.
    """
    y_pred = np.argmax(model.predict(X_test, batch_size=512, verbose=0), axis=1)

    labels = sorted(class_names.keys())
    names  = [class_names[l] for l in labels]

    print("\n── Classification Report ──────────────────────────────────────")
    print(classification_report(y_test, y_pred, labels=labels,
                                 target_names=names, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=names,
                yticklabels=names, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — CICIDS 2017")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[eval] Confusion matrix saved → {save_path}")

    return y_pred


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train BiLSTM on CICIDS-2017")
    p.add_argument("--x",          default="data/processed/X_bilstm.npy")
    p.add_argument("--y",          default="data/processed/y_bilstm.npy")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch_size", type=int,   default=256)   # larger = faster; CICIDS is big
    p.add_argument("--val_split",  type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.10)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--dropout",    type=float, default=0.4)
    p.add_argument("--model_out",  default="models/bilstm_cicids2017.keras")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    X, y = load_data(args.x, args.y)
    num_classes = len(np.unique(y))

    # ── Stratified split (important — attack classes are rare) ────────────────
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=(args.val_split + args.test_split),
        stratify=y,
        random_state=SEED,
    )
    val_ratio_of_tmp = args.val_split / (args.val_split + args.test_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=(1.0 - val_ratio_of_tmp),
        stratify=y_tmp,
        random_state=SEED,
    )
    print(f"[split] train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = compute_cicids_class_weights(y_train)

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_bilstm_ids(
        input_shape=(X.shape[1], X.shape[2]),
        num_classes=num_classes,
        learning_rate=args.lr,
        dropout_rate=args.dropout,
    )
    model.summary(line_length=90)

    # Save config alongside weights for reproducibility
    config = vars(args)
    config["input_shape"] = list(X.shape[1:])
    config["num_classes"] = num_classes
    with open("models/train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Train ─────────────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,        # ← critical for CICIDS imbalance
        callbacks=build_callbacks(
            checkpoint_path="models/best_bilstm.keras"
        ),
        verbose=1,
    )

    # ── Save final model (.keras format — preferred over .h5) ─────────────────
    model.save(args.model_out)
    print(f"\n✅ Model saved → {args.model_out}")

    # ── Plots & evaluation ────────────────────────────────────────────────────
    plot_history(history)

    # Subset of classes actually present in test set
    present_classes = {k: v for k, v in CICIDS_CLASSES.items()
                       if k in np.unique(y_test)}
    evaluate_model(model, X_test, y_test, class_names=present_classes)

    # Final held-out metrics
    loss, acc, top2 = model.evaluate(X_test, y_test,
                                     batch_size=512, verbose=0)
    print(f"\n── Test Results ───────────────────────────────────────────────")
    print(f"   Loss     : {loss:.4f}")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Top-2 Acc: {top2:.4f}")


if __name__ == "__main__":
    main()