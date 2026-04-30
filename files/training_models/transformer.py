"""
Transformer-based Network Intrusion Classifier — TensorFlow / Keras
Wednesday DoS Dataset (CICIDS2017)
======================================================================
Optimised for Google Colab (T4 / A100 / CPU fallback)

Setup in Colab:
    1. Runtime → Change runtime type → GPU (T4 recommended)
    2. Mount Drive if loading data from there (see QUICK START below)
    3. !pip install -q tensorflow scikit-learn pandas

QUICK START (first cell in your notebook):
    from google.colab import drive
    drive.mount('/content/drive')
    # then set DATA_PATH below to your Drive path
"""

# ── Colab: uncomment to install if needed ──────────────────────────
# !pip install -q tensorflow scikit-learn pandas

import os
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight


# ─────────────────────────────────────────────
#  GPU SETUP  (Colab-safe)
# ─────────────────────────────────────────────
def setup_gpu():
    """Configure GPU memory growth to avoid OOM on Colab's shared GPU."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅  GPU detected: {[g.name for g in gpus]}")
    else:
        print("⚠️  No GPU found — running on CPU (will be slow)")
    print(f"    TensorFlow {tf.__version__}\n")

setup_gpu()


# ─────────────────────────────────────────────
#  COLAB PATH HELPERS
# ─────────────────────────────────────────────
def get_data_path(filename: str = "Wed_DoS.csv") -> str:
    """
    Resolves the data path in order of preference:
      1. /content/drive/MyDrive/  (Google Drive mount)
      2. /content/                (direct Colab upload)
      3. current working directory (local run / custom path)
    """
    candidates = [
        f"/content/drive/MyDrive/{filename}",
        f"/content/{filename}",
        filename,
    ]
    for p in candidates:
        if Path(p).exists():
            print(f"📂  Found data at: {p}")
            return p
    # Default — user must update manually
    print(f"⚠️  Data file not found in common locations.")
    print(f"    Set DATA_PATH manually or upload to /content/")
    return f"/content/{filename}"


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

# ── ✏️  EDIT THESE FOR YOUR COLAB SESSION ─────
DATA_PATH  = get_data_path("Wed_DoS.csv")   # or full path to your file
OUTPUT_DIR = "/content/dos_keras_output"    # saved inside Colab (ephemeral)
# To persist outputs → use "/content/drive/MyDrive/dos_keras_output"
# ──────────────────────────────────────────────

DEFAULT_CONFIG = {
    # paths
    "data_path":       DATA_PATH,
    "output_dir":      OUTPUT_DIR,

    # data
    "test_size":       0.15,
    "val_size":        0.15,
    "random_seed":     42,

    # model
    "d_model":         128,
    "n_heads":         8,
    "n_layers":        3,
    "ffn_dim":         256,
    "dropout":         0.2,
    "n_tokens":        4,

    # training
    # batch_size: 2048 is safer on Colab T4 (15GB VRAM); raise to 4096 on A100
    "epochs":          20,
    "batch_size":      2048,
    "lr":              3e-4,
    "weight_decay":    1e-4,
    "label_smoothing": 0.03,
    "patience":        5,
    "lr_patience":     3,
    "lr_factor":       0.5,
    "min_lr":          1e-6,
}


# ─────────────────────────────────────────────
#  MODEL BLOCKS
# ─────────────────────────────────────────────
def feature_embedding_block(x, d_model, dropout, l2):
    mid = d_model * 2
    h = layers.Dense(mid, kernel_regularizer=regularizers.l2(l2))(x)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("gelu")(h)
    h = layers.Dropout(dropout)(h)
    h = layers.Dense(d_model, kernel_regularizer=regularizers.l2(l2))(h)
    h = layers.BatchNormalization()(h)
    skip = layers.Dense(d_model, use_bias=False)(x)
    return layers.Add()([h, skip])


def transformer_block(x, d_model, n_heads, ffn_dim, dropout, l2):
    attn_input = layers.LayerNormalization(epsilon=1e-6)(x)
    attn_out = layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=d_model // n_heads,
        dropout=dropout
    )(attn_input, attn_input)
    attn_out = layers.Dropout(dropout)(attn_out)
    x = layers.Add()([x, attn_out])

    ff_input = layers.LayerNormalization(epsilon=1e-6)(x)
    ff = layers.Dense(ffn_dim, activation="gelu",
                      kernel_regularizer=regularizers.l2(l2))(ff_input)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(d_model, kernel_regularizer=regularizers.l2(l2))(ff)
    ff = layers.Dropout(dropout)(ff)
    x = layers.Add()([x, ff])
    return x


def build_model(in_features, num_classes, cfg):
    d_model  = cfg["d_model"]
    n_heads  = cfg["n_heads"]
    n_layers = cfg["n_layers"]
    ffn_dim  = cfg["ffn_dim"]
    dropout  = cfg["dropout"]
    l2       = cfg["weight_decay"]
    n_tokens = cfg["n_tokens"]

    inputs = keras.Input(shape=(in_features,), name="features")

    chunk = (in_features + n_tokens - 1) // n_tokens
    pad   = chunk * n_tokens - in_features

    if pad > 0:
        x_pad = layers.ZeroPadding1D(padding=(0, pad))(
            layers.Reshape((in_features, 1))(inputs)
        )
        x_pad = layers.Reshape((in_features + pad,))(x_pad)
    else:
        x_pad = inputs

    token_list = []
    for i in range(n_tokens):
        chunk_i = layers.Lambda(
            lambda z, s=i*chunk, e=(i+1)*chunk: z[:, s:e],
            name=f"chunk_{i}"
        )(x_pad)
        token_i = feature_embedding_block(chunk_i, d_model, dropout, l2)
        token_i = layers.Reshape((1, d_model))(token_i)
        token_list.append(token_i)

    tokens = layers.Concatenate(axis=1)(token_list)

    cls_token = layers.Dense(d_model, name="cls_proj")(
        layers.Lambda(lambda z: tf.reduce_mean(z, axis=1))(tokens)
    )
    cls_token = layers.Reshape((1, d_model))(cls_token)
    seq = layers.Concatenate(axis=1)([cls_token, tokens])

    positions = tf.range(start=0, limit=n_tokens + 1, delta=1)
    pos_emb   = layers.Embedding(input_dim=n_tokens + 1, output_dim=d_model,
                                  name="pos_emb")(positions)
    seq = seq + pos_emb
    seq = layers.Dropout(dropout)(seq)

    for _ in range(n_layers):
        seq = transformer_block(seq, d_model, n_heads, ffn_dim, dropout, l2)

    seq = layers.LayerNormalization(epsilon=1e-6)(seq)
    cls_out = layers.Lambda(lambda z: z[:, 0], name="cls_extract")(seq)
    cls_out = layers.Dropout(dropout)(cls_out)
    cls_out = layers.Dense(d_model // 2, activation="gelu",
                           kernel_regularizer=regularizers.l2(l2))(cls_out)
    cls_out = layers.Dropout(dropout * 0.5)(cls_out)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(cls_out)

    return keras.Model(inputs=inputs, outputs=outputs, name="DoSTransformer")


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
def load_and_preprocess(cfg):
    print("📂  Loading data ...")
    df = pd.read_csv(cfg["data_path"])

    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype("int32")

    print(f"    Loaded {len(df):,} rows × {df.shape[1]} cols  "
          f"({df.memory_usage(deep=True).sum()/1e6:.0f} MB)")

    df.columns = df.columns.str.strip()

    if "label_enc" not in df.columns and "label enc" in df.columns:
        df.rename(columns={"label enc": "label_enc"}, inplace=True)

    if "label_enc" not in df.columns:
        label_col = next((c for c in df.columns if c.lower() == "label"), None)
        if label_col is None:
            raise ValueError("No 'Label' or 'label_enc' column found in CSV.")
        cats = df[label_col].astype("category")
        df["label_enc"] = cats.cat.codes.astype("int32")
        print(f"    Auto-encoded labels: {dict(enumerate(cats.cat.categories))}")

    label_text_col = next((c for c in df.columns
                           if c.lower() == "label" and c != "label_enc"), None)

    drop_cols = {"label_enc"}
    if label_text_col:
        drop_cols.add(label_text_col)
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = np.nan_to_num(
        df[feature_cols].values.astype(np.float32),
        nan=0.0, posinf=0.0, neginf=0.0
    )
    y = df["label_enc"].values.astype(np.int32)

    num_classes = int(y.max()) + 1
    if label_text_col:
        label_map = (df[["label_enc", label_text_col]]
                     .drop_duplicates()
                     .sort_values("label_enc")[label_text_col]
                     .tolist())
    else:
        label_map = [str(i) for i in range(num_classes)]

    print(f"    Features: {X.shape[1]}  |  Classes: {num_classes}")
    print(f"    Class distribution:\n"
          f"{pd.Series(df[label_text_col].values if label_text_col else y).value_counts().to_string()}\n")

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=cfg["test_size"],
        stratify=y, random_state=cfg["random_seed"]
    )
    val_frac = cfg["val_size"] / (1 - cfg["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac,
        stratify=y_tmp, random_state=cfg["random_seed"]
    )
    print(f"    Split → train {len(y_train):,} | val {len(y_val):,} | test {len(y_test):,}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    cw = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train)
    class_weight = {i: round(float(cw[i]), 4) for i in range(num_classes)}
    print(f"    Class weights (balanced): {class_weight}\n")

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            scaler, class_weight, num_classes,
            feature_cols, label_map)


# ─────────────────────────────────────────────
#  EARLY-STOP CALLBACK  (macro-F1)
# ─────────────────────────────────────────────
class MacroF1Callback(keras.callbacks.Callback):
    def __init__(self, X_val, y_val, patience: int, model_path: str):
        super().__init__()
        self.X_val      = X_val
        self.y_val      = y_val
        self.patience   = patience
        self.model_path = model_path
        self.best_f1    = -1.0
        self.wait       = 0

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0).argmax(axis=1)
        f1 = f1_score(self.y_val, y_pred, average="macro", zero_division=0)
        print(f"  val_macro_f1: {f1:.4f}", end="")

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.wait    = 0
            self.model.save(self.model_path)
            print(f"  ✅ best saved ({f1:.4f})")
        else:
            self.wait += 1
            print(f"  (no improve, patience {self.wait}/{self.patience})")
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"\n⏹️  Early stopping at epoch {epoch + 1}")


# ─────────────────────────────────────────────
#  COLAB SESSION SAVER
# ─────────────────────────────────────────────
def maybe_zip_to_drive(output_dir: str):
    """
    Optionally zip output artefacts to Google Drive so they survive
    Colab session resets. Silently skips if Drive isn't mounted.
    """
    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.exists():
        print("ℹ️  Drive not mounted — artefacts saved to /content/ only (ephemeral).")
        print("   To persist: mount Drive and set OUTPUT_DIR to a Drive path.")
        return

    import shutil
    zip_path = str(drive_root / "dos_keras_output")
    shutil.make_archive(zip_path, "zip", output_dir)
    print(f"📦  Artefacts zipped → {zip_path}.zip (in your Google Drive)")


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
def train(cfg):
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    (X_tr, X_val, X_te,
     y_tr, y_val, y_te,
     scaler, class_weight, num_classes,
     feature_cols, label_map) = load_and_preprocess(cfg)

    y_tr_oh  = keras.utils.to_categorical(y_tr,  num_classes).astype(np.float32)
    y_val_oh = keras.utils.to_categorical(y_val, num_classes).astype(np.float32)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_tr, y_tr_oh))
        .shuffle(10_000)
        .batch(cfg["batch_size"])
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val_oh))
        .batch(cfg["batch_size"] * 2)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_model(X_tr.shape[1], num_classes, cfg)
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=cfg["lr"],
            weight_decay=cfg["weight_decay"]
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg["label_smoothing"]
        ),
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")]
    )
    model.summary(line_length=80)

    model_path = str(out / "best_model.keras")
    callbacks = [
        MacroF1Callback(X_val, y_val,
                        patience=cfg["patience"],
                        model_path=model_path),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            patience=cfg["lr_patience"],
            factor=cfg["lr_factor"],
            min_lr=cfg["min_lr"],
            verbose=1
        ),
        keras.callbacks.CSVLogger(str(out / "history.csv")),
        # Colab sessions can disconnect — TensorBoard helps monitor remotely
        keras.callbacks.TensorBoard(
            log_dir=str(out / "tb_logs"),
            histogram_freq=0,       # 0 = faster; set 1 for weight histograms
            update_freq="epoch"
        ),
    ]

    print("─" * 60)
    print(f"Training — up to {cfg['epochs']} epochs, "
          f"early stop patience {cfg['patience']}")
    print(f"Batch size: {cfg['batch_size']}  |  GPU: "
          f"{bool(tf.config.list_physical_devices('GPU'))}")
    print("─" * 60)

    model.fit(
        train_ds,
        epochs=cfg["epochs"],
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    # ── evaluate best checkpoint ──
    print("\n" + "=" * 60)
    print("📊  Loading best model → evaluating on TEST set ...")
    best = keras.models.load_model(model_path)
    y_pred = best.predict(X_te, batch_size=cfg["batch_size"] * 2,
                          verbose=0).argmax(axis=1)

    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)
    print(f"\n   Test Accuracy : {acc*100:.3f}%")
    print(f"   Test Macro-F1 : {f1:.4f}\n")
    print(classification_report(y_te, y_pred, target_names=label_map, digits=4))

    # ── save artefacts ──
    with open(out / "scaler.pkl",        "wb") as f: pickle.dump(scaler, f)
    with open(out / "feature_cols.json", "w")  as f: json.dump(feature_cols, f)
    with open(out / "label_map.json",    "w")  as f: json.dump(label_map, f)
    with open(out / "config.json",       "w")  as f: json.dump(cfg, f, indent=2)
    with open(out / "model_meta.json",   "w")  as f:
        json.dump({"in_features": X_tr.shape[1],
                   "num_classes": num_classes}, f, indent=2)

    print(f"\n✅  Artefacts saved to '{out}/'")

    # ── optionally persist to Drive ──
    maybe_zip_to_drive(str(out))

    return best


# ─────────────────────────────────────────────
#  INFERENCE API
# ─────────────────────────────────────────────
class DoSClassifier:
    """
    Real-world inference wrapper.

    Usage:
        clf   = DoSClassifier("/content/dos_keras_output")
        preds = clf.predict(df)          # list of label strings
        probs = clf.predict_proba(df)    # numpy array (N, num_classes)
    """
    def __init__(self, output_dir: str = "/content/dos_keras_output"):
        out = Path(output_dir)
        self.model  = keras.models.load_model(str(out / "best_model.keras"))
        with open(out / "scaler.pkl",        "rb") as f: self.scaler = pickle.load(f)
        with open(out / "feature_cols.json")       as f: self.fcols  = json.load(f)
        with open(out / "label_map.json")          as f: self.labels = json.load(f)
        with open(out / "config.json")             as f: self.cfg    = json.load(f)
        print(f"[DoSClassifier] Loaded from '{output_dir}'")

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.fcols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.scaler.transform(X).astype(np.float32)

    def predict(self, df: pd.DataFrame) -> list:
        idx = self.model.predict(self._prepare(df), verbose=0).argmax(axis=1)
        return [self.labels[i] for i in idx]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prepare(df), verbose=0)


# ─────────────────────────────────────────────
#  CLI  (for local use; in Colab just call train(DEFAULT_CONFIG))
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DoS Transformer — TensorFlow/Keras")
    p.add_argument("--mode",       choices=["train", "predict"], default="train")
    p.add_argument("--data",       default=DEFAULT_CONFIG["data_path"])
    p.add_argument("--input",      default=None, help="CSV to classify (predict mode)")
    p.add_argument("--output_dir", default=DEFAULT_CONFIG["output_dir"])
    p.add_argument("--epochs",     type=int,   default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--batch_size", type=int,   default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--lr",         type=float, default=DEFAULT_CONFIG["lr"])
    p.add_argument("--d_model",    type=int,   default=DEFAULT_CONFIG["d_model"])
    p.add_argument("--n_layers",   type=int,   default=DEFAULT_CONFIG["n_layers"])
    p.add_argument("--dropout",    type=float, default=DEFAULT_CONFIG["dropout"])
    return p.parse_args()


# ─────────────────────────────────────────────
#  ENTRY POINT
#  In Colab: just run   train(DEFAULT_CONFIG)
#  As script: python dos_transformer_colab.py --mode train
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Detect if running inside a Colab notebook
    try:
        import google.colab  # noqa
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        # In Colab, skip argparse and run directly
        train(DEFAULT_CONFIG)
    else:
        args = parse_args()
        cfg  = DEFAULT_CONFIG.copy()
        cfg.update({
            "data_path":  args.data,
            "output_dir": args.output_dir,
            "epochs":     args.epochs,
            "batch_size": args.batch_size,
            "lr":         args.lr,
            "d_model":    args.d_model,
            "n_layers":   args.n_layers,
            "dropout":    args.dropout,
        })

        if args.mode == "train":
            train(cfg)
        elif args.mode == "predict":
            if args.input is None:
                raise ValueError("--input must be provided in predict mode")
            clf   = DoSClassifier(args.output_dir)
            df    = pd.read_csv(args.input)
            preds = clf.predict(df)
            probs = clf.predict_proba(df)
            df["prediction"] = preds
            df["confidence"] = probs.max(axis=1)
            out_path = Path(args.output_dir) / "predictions.csv"
            df.to_csv(out_path, index=False)
            print(f"\nPredictions saved → {out_path}")
            print(df[["prediction", "confidence"]].value_counts().head(20).to_string())