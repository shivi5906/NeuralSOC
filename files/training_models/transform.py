"""
Transformer-based Network Intrusion Classifier — TensorFlow / Keras
Wednesday DoS Dataset (CICIDS2017)
======================================================================
Single file: model definition + training + evaluation + inference
Anti-overfitting: label smoothing, dropout, weight decay, mixup,
                  early stopping, class-balanced loss, ReduceLROnPlateau
Real-world ready: scaler + label encoder saved, clean predict() API
"""

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
#  CONFIG
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    # paths
    "data_path":        r"X:\dev\neuralSOC\NeuralSOC\data\processed\Wed_DoS.csv",
    "output_dir":       "dos_keras_output",

    # data
    "test_size":        0.15,
    "val_size":         0.15,
    "random_seed":      42,

    # model
    "d_model":          128,
    "n_heads":          8,
    "n_layers":         3,
    "ffn_dim":          256,
    "dropout":          0.3,
    "n_tokens":         4,         # split features into N virtual tokens

    # training
    "epochs":           40,
    "batch_size":       2048,
    "lr":               3e-4,
    "weight_decay":     1e-4,      # L2 on dense layers
    "label_smoothing":  0.05,
    "mixup_alpha":      0.2,       # set 0.0 to disable
    "patience":         7,         # early stopping on val macro-F1
    "lr_patience":      3,
    "lr_factor":        0.5,
    "min_lr":           1e-6,
}


# ─────────────────────────────────────────────
#  MIXUP
# ─────────────────────────────────────────────
def mixup_batch(X, y_onehot, alpha):
    """Mixup augmentation — runs on numpy arrays, called inside tf.data.map."""
    if alpha <= 0:
        return X, y_onehot
    lam = np.random.beta(alpha, alpha)
    idx = np.random.permutation(len(X))
    X_mix = lam * X + (1 - lam) * X[idx]
    y_mix = lam * y_onehot + (1 - lam) * y_onehot[idx]
    return X_mix.astype(np.float32), y_mix.astype(np.float32)


# ─────────────────────────────────────────────
#  MODEL BLOCKS
# ─────────────────────────────────────────────
def feature_embedding_block(x, d_model, dropout, l2):
    """Project raw features → d_model with a skip connection."""
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
    """Pre-norm Transformer block with MHA + FFN."""
    # Multi-head self-attention
    attn_input = layers.LayerNormalization(epsilon=1e-6)(x)
    attn_out = layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=d_model // n_heads,
        dropout=dropout
    )(attn_input, attn_input)
    attn_out = layers.Dropout(dropout)(attn_out)
    x = layers.Add()([x, attn_out])

    # Feed-forward
    ff_input = layers.LayerNormalization(epsilon=1e-6)(x)
    ff = layers.Dense(ffn_dim, activation="gelu",
                      kernel_regularizer=regularizers.l2(l2))(ff_input)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(d_model, kernel_regularizer=regularizers.l2(l2))(ff)
    ff = layers.Dropout(dropout)(ff)
    x = layers.Add()([x, ff])
    return x


def build_model(in_features, num_classes, cfg):
    """
    Build the DoS Transformer model.
    Each input row is split into n_tokens virtual tokens, a CLS token
    is prepended, and the transformer attends over all tokens.
    The CLS representation goes to the classification head.
    """
    d_model  = cfg["d_model"]
    n_heads  = cfg["n_heads"]
    n_layers = cfg["n_layers"]
    ffn_dim  = cfg["ffn_dim"]
    dropout  = cfg["dropout"]
    l2       = cfg["weight_decay"]
    n_tokens = cfg["n_tokens"]

    inputs = keras.Input(shape=(in_features,), name="features")

    # ── split into n_tokens chunks and embed each ──
    chunk = (in_features + n_tokens - 1) // n_tokens
    pad   = chunk * n_tokens - in_features

    # pad if needed so we can cleanly split
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
        token_i = feature_embedding_block(chunk_i, d_model, dropout, l2)   # (B, d_model)
        token_i = layers.Reshape((1, d_model))(token_i)                    # (B, 1, d_model)
        token_list.append(token_i)

    tokens = layers.Concatenate(axis=1)(token_list)   # (B, n_tokens, d_model)

    # ── learned CLS token ──
    cls_token = layers.Dense(d_model, name="cls_proj")(
        layers.Lambda(lambda z: tf.reduce_mean(z, axis=1))(tokens)
    )
    cls_token = layers.Reshape((1, d_model))(cls_token)
    seq = layers.Concatenate(axis=1)([cls_token, tokens])  # (B, n_tokens+1, d_model)

    # ── learned positional embedding ──
    positions = tf.range(start=0, limit=n_tokens + 1, delta=1)
    pos_emb   = layers.Embedding(input_dim=n_tokens + 1, output_dim=d_model,
                                  name="pos_emb")(positions)          # (n_tokens+1, d_model)
    seq = seq + pos_emb                                                # broadcast over batch
    seq = layers.Dropout(dropout)(seq)

    # ── transformer blocks ──
    for _ in range(n_layers):
        seq = transformer_block(seq, d_model, n_heads, ffn_dim, dropout, l2)

    seq = layers.LayerNormalization(epsilon=1e-6)(seq)

    # ── CLS output → classification head ──
    cls_out = layers.Lambda(lambda z: z[:, 0], name="cls_extract")(seq)  # (B, d_model)
    cls_out = layers.Dropout(dropout)(cls_out)
    cls_out = layers.Dense(d_model // 2, activation="gelu",
                           kernel_regularizer=regularizers.l2(l2))(cls_out)
    cls_out = layers.Dropout(dropout * 0.5)(cls_out)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(cls_out)

    model = keras.Model(inputs=inputs, outputs=outputs, name="DoSTransformer")
    return model


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
def load_and_preprocess(cfg):
    print("📂  Loading data ...")
    df = pd.read_csv(cfg["data_path"])
    # downcast to save RAM
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64").columns:
        if col not in ("label_enc",):
            df[col] = df[col].astype("int32")

    print(f"    Loaded {len(df):,} rows × {df.shape[1]} cols  "
          f"({df.memory_usage(deep=True).sum()/1e6:.0f} MB)")

    # auto-create label_enc if missing
    if "label_enc" not in df.columns:
        cats = df["Label"].astype("category")
        df["label_enc"] = cats.cat.codes.astype("int32")
        mapping = dict(enumerate(cats.cat.categories))
        print(f"    Auto-encoded labels: {mapping}")

    drop_cols    = ["Label", "label_enc"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label_enc"].values.astype(np.int32)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    num_classes  = int(y.max()) + 1
    label_map    = (df[["label_enc", "Label"]]
                    .drop_duplicates()
                    .sort_values("label_enc")["Label"]
                    .tolist())

    print(f"    Features: {X.shape[1]}  |  Classes: {num_classes}")
    print(f"    Class distribution:\n{pd.Series(df['Label'].values).value_counts().to_string()}\n")

    # stratified split
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

    # scale (fit on train only)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # class weights
    classes      = np.arange(num_classes)
    cw           = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {i: cw[i] for i in range(num_classes)}
    print(f"    Class weights (balanced): { {k: round(v,2) for k,v in class_weight.items()} }\n")

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            scaler, class_weight, num_classes,
            feature_cols, label_map)


# ─────────────────────────────────────────────
#  CUSTOM F1 CALLBACK (for early stopping on macro-F1)
# ─────────────────────────────────────────────
class MacroF1Callback(keras.callbacks.Callback):
    def __init__(self, val_data, patience, model_path):
        super().__init__()
        self.X_val      = val_data[0]
        self.y_val      = val_data[1]
        self.patience   = patience
        self.model_path = model_path
        self.best_f1    = -1.0
        self.wait        = 0

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
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
def train(cfg):
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load and Preprocess
    (X_tr, X_val, X_te,
     y_tr, y_val, y_te,
     scaler, class_weight, num_classes,
     feature_cols, label_map) = load_and_preprocess(cfg)

    alpha = cfg["mixup_alpha"]
    
    # 2. Data Pipeline Logic (The Fix)
    if alpha > 0:
        print(f"✨ Mixup enabled (alpha={alpha}). Using CategoricalCrossentropy.")
        
        # One-hot encode both sets for CategoricalCrossentropy
        y_tr_oh = tf.keras.utils.to_categorical(y_tr, num_classes).astype(np.float32)
        y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes).astype(np.float32)
        
        # Apply Mixup to training data
        X_tr_mix, y_tr_mix = mixup_batch(X_tr, y_tr_oh, alpha)
        
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((X_tr_mix, y_tr_mix))
            .shuffle(10000)
            .batch(cfg["batch_size"])
            .prefetch(tf.data.AUTOTUNE)
        )
        
        val_dataset = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val_oh))
            .batch(cfg["batch_size"] * 2)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        loss_fn = keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg["label_smoothing"]
        )
        metric = keras.metrics.CategoricalAccuracy(name="acc")
        cw_train = None  # Class weights are mathematically tricky with Mixup
    else:
        print("🚀 Mixup disabled. Using SparseCategoricalCrossentropy.")
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
            .shuffle(10000)
            .batch(cfg["batch_size"])
            .prefetch(tf.data.AUTOTUNE)
        )
        
        val_dataset = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .batch(cfg["batch_size"] * 2)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        metric = keras.metrics.SparseCategoricalAccuracy(name="acc")
        cw_train = class_weight

    # 3. Build and Compile Model
    model = build_model(X_tr.shape[1], num_classes, cfg)
    
    # Use AdamW as defined in your config
    optimizer = keras.optimizers.AdamW(
        learning_rate=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric])

    # 4. Callbacks
    model_path = str(out / "best_model.keras")
    
    callbacks = [
        # Note: y_val remains integers here because f1_score expects class indices
        MacroF1Callback(val_data=(X_val, y_val),
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
    ]

    # 5. Training Loop
    print("─" * 60)
    print(f"Training for up to {cfg['epochs']} epochs ...")
    print("─" * 60)

    model.fit(
        train_dataset,
        epochs=cfg["epochs"],
        validation_data=val_dataset,
        callbacks=callbacks,
        class_weight=cw_train,
        verbose=1,
    )

    # 6. Final Evaluation (Reloading the best weights)
    print("\n" + "=" * 60)
    print("📊 Loading best model → evaluating on TEST set ...")
    best_model = keras.models.load_model(model_path)

    y_pred = best_model.predict(X_te, batch_size=cfg["batch_size"] * 2,
                                verbose=0).argmax(axis=1)
    
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)
    
    print(f"\n   Test Accuracy : {acc*100:.3f}%")
    print(f"   Test F1 Macro : {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_te, y_pred, target_names=label_map, digits=4))

    # 7. Save Artefacts
    with open(out / "scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    with open(out / "feature_cols.json", "w") as f: json.dump(feature_cols, f)
    with open(out / "label_map.json", "w") as f: json.dump(label_map, f)
    with open(out / "config.json", "w") as f: json.dump(cfg, f, indent=2)
    with open(out / "model_meta.json", "w") as f:
        json.dump({"in_features": X_tr.shape[1],
                   "num_classes": num_classes}, f, indent=2)

    print(f"\n✅ All artefacts saved to '{out}/'")
    return best_model

# ─────────────────────────────────────────────
#  INFERENCE API
# ─────────────────────────────────────────────
class DoSClassifier:
    """
    Real-world inference wrapper.

    Usage:
        clf   = DoSClassifier("dos_keras_output")
        preds = clf.predict(df)          # list of label strings
        probs = clf.predict_proba(df)    # numpy array (N, num_classes)
    """
    def __init__(self, output_dir: str = "dos_keras_output"):
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
        X   = self._prepare(df)
        idx = self.model.predict(X, verbose=0).argmax(axis=1)
        return [self.labels[i] for i in idx]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prepare(df), verbose=0)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DoS Transformer — TensorFlow/Keras")
    p.add_argument("--mode",        choices=["train", "predict"], default="train")
    p.add_argument("--data",        default=DEFAULT_CONFIG["data_path"])
    p.add_argument("--input",       default=None, help="CSV to classify (predict mode)")
    p.add_argument("--output_dir",  default=DEFAULT_CONFIG["output_dir"])
    p.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--batch_size",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["lr"])
    p.add_argument("--d_model",     type=int,   default=DEFAULT_CONFIG["d_model"])
    p.add_argument("--n_layers",    type=int,   default=DEFAULT_CONFIG["n_layers"])
    p.add_argument("--dropout",     type=float, default=DEFAULT_CONFIG["dropout"])
    p.add_argument("--mixup_alpha", type=float, default=DEFAULT_CONFIG["mixup_alpha"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg.update({
        "data_path":   args.data,
        "output_dir":  args.output_dir,
        "epochs":      args.epochs,
        "batch_size":  args.batch_size,
        "lr":          args.lr,
        "d_model":     args.d_model,
        "n_layers":    args.n_layers,
        "dropout":     args.dropout,
        "mixup_alpha": args.mixup_alpha,
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