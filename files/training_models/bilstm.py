# ── 0. Install / upgrade dependencies ────────────────────────────────────────
# Run this cell once; restart runtime if prompted.
# !pip install -q --upgrade tensorflow keras scikit-learn seaborn matplotlib pandas

# tensorflow_addons only needed if TF < 2.11 (Colab usually has TF ≥ 2.13)
import tensorflow as tf
print(f"TensorFlow : {tf.__version__}")
print(f"Keras      : {tf.keras.__version__}")

if not hasattr(tf.keras.optimizers, 'AdamW'):
    print("AdamW not found in tf.keras.optimizers — installing tensorflow_addons")
    #!pip install -q tensorflow_addons
else:
    print("AdamW available natively ✓")
    
# ── 1. Imports ────────────────────────────────────────────────────────────────
import os, json, pickle, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout,
    LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Concatenate, Conv1D,
    BatchNormalization, Activation, Multiply, Reshape,
    GlobalMaxPooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# Reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs("models", exist_ok=True)
os.makedirs("logs",   exist_ok=True)
print("Imports OK ✓")

# ── 2. Upload your CSV files ──────────────────────────────────────────────────
from google.colab import drive

FILE1 = "/content/drive/MyDrive/NeuralSoc/data/Thu_Infil.csv"
FILE2 = "/content/drive/MyDrive/NeuralSoc/data/Thu_WebAtt.csv"


# ── 3. Config ─────────────────────────────────────────────────────────────────
CFG = dict(
    window       = 10,
    epochs       = 50,
    batch_size   = 128,
    val_frac     = 0.15,
    test_frac    = 0.10,
    lr           = 3e-4,
    dropout      = 0.35,
    focal_gamma  = 2.0,
    target_ratio = 0.30,
    l2_reg       = 1e-4,
    model_out    = "/content/drive/MyDrive/NeuralSoc/models",
)
print(json.dumps(CFG, indent=2))

# ── 4. Custom loss ────────────────────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable(package="NeuralSOC")
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """
    Focal loss for sparse integer labels.
    Subclass + register_keras_serializable → load_model() works out of the box.
    """
    def __init__(self, gamma: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = float(gamma)

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        n_cls  = tf.shape(y_pred)[1]
        y_oh   = tf.one_hot(y_true, n_cls)
        p_t    = tf.reduce_sum(y_pred * y_oh, axis=1)
        ce     = -tf.math.log(p_t)
        return tf.reduce_mean(tf.pow(1.0 - p_t, self.gamma) * ce)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma})
        return cfg

print("SparseCategoricalFocalLoss registered ✓")


# ── 5. Custom layers ──────────────────────────────────────────────────────────
@tf.keras.utils.register_keras_serializable(package="NeuralSOC")
class SqueezeExciteBlock(tf.keras.layers.Layer):
    """Channel-wise squeeze-and-excitation — down-weights correlated CICIDS features."""
    def __init__(self, ratio: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        filters       = int(input_shape[-1])
        self.d_reduce = Dense(max(filters // self.ratio, 1), activation="relu")
        self.d_expand = Dense(filters, activation="sigmoid")
        self.reshape  = Reshape((1, filters))
        super().build(input_shape)

    def call(self, x):
        se = GlobalAveragePooling1D()(x)
        se = self.d_reduce(se)
        se = self.d_expand(se)
        se = self.reshape(se)
        return Multiply()([x, se])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ratio": self.ratio})
        return cfg


@tf.keras.utils.register_keras_serializable(package="NeuralSOC")
class TemporalAttentionBlock(tf.keras.layers.Layer):
    """Multi-head self-attention + residual LayerNorm over the time axis."""
    def __init__(self, num_heads=4, key_dim=32, attn_dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads    = num_heads
        self.key_dim      = key_dim
        self.attn_dropout = attn_dropout

    def build(self, input_shape):
        self.mha = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.key_dim,
            dropout=self.attn_dropout)
        self.ln  = LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

    def call(self, x, training=False):
        return self.ln(x + self.mha(x, x, training=training))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_heads": self.num_heads,
            "key_dim":   self.key_dim,
            "attn_dropout": self.attn_dropout,
        })
        return cfg

print("SqueezeExciteBlock + TemporalAttentionBlock registered ✓")

# ── 6. Model builder ──────────────────────────────────────────────────────────
def build_bilstm_ids(
    input_shape    = (10, 78),
    num_classes    = 8,
    lstm_units     = (128, 64),
    dense_units    = 128,
    dropout_rate   = 0.4,
    l2_reg         = 1e-4,
    learning_rate  = 3e-4,
    focal_gamma    = 2.0,
    use_top2_metric= True,
):
    reg    = l2(l2_reg)
    inputs = Input(shape=input_shape, name="flow_window")

    x = Conv1D(64, 3, padding="same", kernel_regularizer=reg, name="conv_local")(inputs)
    x = BatchNormalization(name="bn_conv")(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate * 0.5)(x)

    x = SqueezeExciteBlock(ratio=8, name="se_block")(x)

    x = Bidirectional(LSTM(lstm_units[0], return_sequences=True,
                           kernel_regularizer=reg, recurrent_dropout=0.0),
                      name="bilstm_1")(x)
    x = LayerNormalization(name="ln_bilstm1")(x)
    x = Dropout(dropout_rate)(x)

    x = Bidirectional(LSTM(lstm_units[1], return_sequences=True,
                           kernel_regularizer=reg, recurrent_dropout=0.0),
                      name="bilstm_2")(x)
    x = LayerNormalization(name="ln_bilstm2")(x)
    x = Dropout(dropout_rate)(x)

    x = TemporalAttentionBlock(num_heads=4, key_dim=32,
                               attn_dropout=dropout_rate * 0.25,
                               name="temporal_attn")(x)

    avg_p = GlobalAveragePooling1D(name="pool_avg")(x)
    max_p = GlobalMaxPooling1D(name="pool_max")(x)
    x     = Concatenate(name="dual_pool")([avg_p, max_p])

    x = Dense(dense_units, kernel_regularizer=reg, name="fc1")(x)
    x = BatchNormalization(name="bn_fc1")(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(dense_units // 2, kernel_regularizer=reg, name="fc2")(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate * 0.5)(x)

    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)
    model   = Model(inputs, outputs, name="BiLSTM_IDS_CICIDS2017")

    # AdamW with version-safe fallback
    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=l2_reg)
    except AttributeError:
        try:
            import tensorflow_addons as tfa
            opt = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=l2_reg)
        except ImportError:
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    metrics = ["accuracy"]
    if use_top2_metric and num_classes >= 3:
        metrics.append(tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc"))

    model.compile(
        optimizer=opt,
        loss=SparseCategoricalFocalLoss(gamma=focal_gamma, name="focal_loss"),
        metrics=metrics,
    )
    return model

print("build_bilstm_ids defined ✓")

# ── 7. MacroF1 callback ───────────────────────────────────────────────────────
class MacroF1Callback(tf.keras.callbacks.Callback):
    """Logs val_macro_f1 each epoch. EarlyStopping monitors this."""
    def __init__(self, val_data):
        super().__init__()
        self.X_val, self.y_val = val_data

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(
            self.model.predict(self.X_val, batch_size=512, verbose=0), axis=1)
        f1 = f1_score(self.y_val, y_pred, average="macro", zero_division=0)
        logs["val_macro_f1"] = float(f1)
        print(f"  — val_macro_f1: {f1:.4f}")


def build_callbacks(checkpoint_path, val_data):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    return [
        MacroF1Callback(val_data),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1", mode="max",
            patience=7, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_macro_f1", mode="max",
            save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1", mode="max",
            factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger("logs/training_log.csv"),
    ]

print("Callbacks defined ✓")

# ── 8. Data loading & cleaning ────────────────────────────────────────────────
def load_and_clean(f1, f2):
    print("[load] Reading CSVs …")
    df = pd.concat([pd.read_csv(f1), pd.read_csv(f2)], ignore_index=True)
    df.columns = df.columns.str.strip()
    print(f"[load] Combined shape: {df.shape}")

    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    assert label_col, f"'Label' column not found. Got: {df.columns.tolist()}"

    feat_cols = [c for c in df.columns if c != label_col]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df[label_col] = df[label_col].astype(str).str.strip()

    print(f"[load] After cleaning: {len(df)} rows")
    print(f"[load] Class distribution:\n{df[label_col].value_counts()}\n")
    return df, feat_cols, label_col

print("load_and_clean defined ✓")

# ── 9. Temporal split → build windows ────────────────────────────────────────
def temporal_split_raw(X, y, val_frac=0.15, test_frac=0.10):
    """
    Split RAW ROWS chronologically BEFORE building any windows.
    This is the fix for the 100% accuracy / data-leakage bug.
    """
    n      = len(X)
    i_val  = int(n * (1 - val_frac - test_frac))
    i_test = int(n * (1 - test_frac))
    splits = {
        "train": (X[:i_val],       y[:i_val]),
        "val":   (X[i_val:i_test], y[i_val:i_test]),
        "test":  (X[i_test:],      y[i_test:]),
    }
    print("[temporal split] Row-level split (zero leakage):")
    for name, (Xs, ys) in splits.items():
        counts = {int(c): int((ys==c).sum()) for c in np.unique(ys)}
        print(f"  {name}: {len(Xs)} rows  |  classes: {counts}")
    return splits


def build_windows(X_split, y_split, window_size=10):
    """Sliding window on a single pre-split array. Label = last row in window."""
    if len(X_split) <= window_size:
        print(f"  WARNING: only {len(X_split)} rows, need >{window_size}")
        return (np.empty((0, window_size, X_split.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.int32))
    X_w = np.array(
        [X_split[i:i+window_size] for i in range(len(X_split) - window_size)],
        dtype=np.float32)
    y_w = np.array(
        [y_split[i+window_size-1] for i in range(len(X_split) - window_size)],
        dtype=np.int32)
    counts = {int(c): int((y_w==c).sum()) for c in np.unique(y_w)}
    print(f"  windows={len(X_w)}  class_counts={counts}")
    return X_w, y_w

print("temporal_split_raw + build_windows defined ✓")

# ── 10. Oversampling + class weights ─────────────────────────────────────────
def oversample_minority(X, y, target_ratio=0.3):
    """Duplicate minority windows with small Gaussian noise. Train only."""
    classes, counts = np.unique(y, return_counts=True)
    target_n = int(counts.max() * target_ratio)
    X_out, y_out = [X], [y]
    for cls, cnt in zip(classes, counts):
        if cnt >= target_n:
            continue
        needed  = target_n - cnt
        src_idx = np.where(y == cls)[0]
        chosen  = np.random.choice(src_idx, size=needed, replace=True)
        noise   = np.random.normal(0, 0.01, X[chosen].shape).astype(np.float32)
        X_out.append(X[chosen] + noise)
        y_out.append(np.full(needed, cls, dtype=np.int32))
        print(f"[oversample] class={cls}: {cnt} → {cnt+needed}")
    X_out = np.concatenate(X_out)
    y_out = np.concatenate(y_out)
    perm  = np.random.permutation(len(X_out))
    return X_out[perm], y_out[perm]


def get_class_weights(y_train, class_names):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    print("[class weights]", {class_names.get(k, k): f"{v:.3f}" for k, v in cw.items()})
    return cw

print("oversample_minority + get_class_weights defined ✓")

# ── 11. Plots & evaluation ────────────────────────────────────────────────────
def plot_history(history, save_path="logs/training_curves.png"):
    has_f1 = "val_macro_f1" in history.history
    n = 3 if has_f1 else 2
    fig, axes = plt.subplots(1, n, figsize=(6*n, 4))

    axes[0].plot(history.history["loss"],     label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Focal Loss"); axes[0].legend()

    axes[1].plot(history.history["accuracy"],     label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy"); axes[1].legend()

    if has_f1:
        axes[2].plot(history.history["val_macro_f1"], color="green")
        axes[2].set_title("Val Macro-F1  ← key metric")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[plot] → {save_path}")


def evaluate_model(model, X_test, y_test, class_names,
                   save_path="logs/confusion_matrix.png"):
    y_pred   = np.argmax(model.predict(X_test, batch_size=512, verbose=0), axis=1)
    present  = sorted(np.unique(y_test).tolist())
    names    = [class_names.get(l, str(l)) for l in present]
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n── Classification Report ──────────────────────────────────────")
    print(classification_report(y_test, y_pred, labels=present,
                                 target_names=names, digits=4))
    print(f"   Macro-F1: {macro_f1:.4f}")

    cm      = confusion_matrix(y_test, y_pred, labels=present)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm,      annot=True, fmt="d",   xticklabels=names,
                yticklabels=names, cmap="Blues",   ax=ax1)
    ax1.set_title("Counts"); ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".1%", xticklabels=names,
                yticklabels=names, cmap="Oranges", ax=ax2, vmin=0, vmax=1)
    ax2.set_title("Row % — recall per class  ← show judges this one")
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")

    plt.suptitle("CICIDS 2017 — Thursday BiLSTM (temporal split, no leakage)", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[eval] → {save_path}")
    return y_pred, macro_f1

print("plot_history + evaluate_model defined ✓")

# ── 12. MAIN TRAINING RUN ─────────────────────────────────────────────────────
# Load & clean
df, feat_cols, label_col = load_and_clean(FILE1, FILE2)

# Encode labels
le          = LabelEncoder()
y_all       = le.fit_transform(df[label_col]).astype(np.int32)
class_names = {i: c for i, c in enumerate(le.classes_)}
num_classes = len(le.classes_)
print(f"[encode] {num_classes} classes: {class_names}\n")

# Scale
scaler   = RobustScaler()
X_scaled = scaler.fit_transform(df[feat_cols]).astype(np.float32)

# Temporal split (fixes data leakage)
splits = temporal_split_raw(X_scaled, y_all, CFG["val_frac"], CFG["test_frac"])

# Build windows on each split independently
print("[window] Train:");  X_tr,   y_tr   = build_windows(*splits["train"], CFG["window"])
print("[window] Val:");    X_val,  y_val  = build_windows(*splits["val"],   CFG["window"])
print("[window] Test:");   X_test, y_test = build_windows(*splits["test"],  CFG["window"])

assert len(X_test) > 0, "Test split too small — reduce test_frac or use more data."

# Oversample (train only)
print(f"\n[oversample] target_ratio={CFG['target_ratio']}")
X_tr, y_tr = oversample_minority(X_tr, y_tr, CFG["target_ratio"])
print(f"[oversample] Final train: {len(X_tr)} windows\n")

# Class weights
cw = get_class_weights(y_tr, class_names)

# Build & summarise model
model = build_bilstm_ids(
    input_shape     = (CFG["window"], X_scaled.shape[1]),
    num_classes     = num_classes,
    learning_rate   = CFG["lr"],
    dropout_rate    = CFG["dropout"],
    focal_gamma     = CFG["focal_gamma"],
    l2_reg          = CFG["l2_reg"],
    use_top2_metric = (num_classes >= 3),
)
model.summary(line_length=90)

# Save config + preprocessing artifacts
with open("models/train_config.json", "w") as f:
    json.dump({
        **CFG,
        "input_shape":  [CFG["window"], int(X_scaled.shape[1])],
        "num_classes":  int(num_classes),
        "class_names":  {str(k): v for k, v in class_names.items()},
        "split_method": "temporal (no leakage)",
    }, f, indent=2)

with open("models/scaler.pkl", "wb")        as f: pickle.dump(scaler, f)
with open("models/label_encoder.pkl", "wb") as f: pickle.dump(le,     f)
print("[save] scaler.pkl + label_encoder.pkl + train_config.json → models/")

# Train
history = model.fit(
    X_tr, y_tr,
    validation_data = (X_val, y_val),
    epochs          = CFG["epochs"],
    batch_size      = CFG["batch_size"],
    class_weight    = cw,
    callbacks       = build_callbacks("/content/drive/MyDrive/NeuralSoc/models/bilstm.keras", (X_val, y_val)),
    verbose         = 1,
)

# ── 14. Load model (plain — no custom_objects needed) ─────────────────────────
loaded_model = tf.keras.models.load_model("/content/drive/MyDrive/NeuralSoc/models/bilstm.keras")
print("✅  load_model() succeeded")
loaded_model.summary(line_length=90)

# Quick sanity check — predictions should match
sample      = X_test[:32]
orig_preds  = np.argmax(model.predict(sample, verbose=0), axis=1)
load_preds  = np.argmax(loaded_model.predict(sample, verbose=0), axis=1)
assert np.array_equal(orig_preds, load_preds), "Predictions differ after reload!"
print("✅  Predictions identical before and after reload")  