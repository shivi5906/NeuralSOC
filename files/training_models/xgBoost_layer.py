# =============================================================================
# NeuralSOC ⚡ — XGBoost Meta-Learner (Ensemble Fusion Layer)
# =============================================================================
# DESCRIPTION:
#   This script builds the STACKING ENSEMBLE layer of NeuralSOC.
#   The three trained deep learning models (CNN, BiLSTM, Transformer) each
#   output a softmax probability vector (16 classes). These vectors are
#   CONCATENATED (48 features total) and fed as input to an XGBoost
#   meta-learner, which makes the FINAL prediction.
#
#   Additionally, the LSTM Autoencoder's reconstruction error is appended as
#   a 49th feature — acting as a zero-day anomaly signal.
#
# FLOW:
#   Raw Features → [CNN, BiLSTM, Transformer, Autoencoder]
#                          ↓
#       Concatenated probability vectors (48 + 1 = 49 features)
#                          ↓
#               XGBoost Meta-Learner
#                          ↓
#         Final Class Label + Confidence + Zero-Day Flag
#
# HOW TO RUN (Google Colab):
#   1. Upload your trained model files (.h5, .pkl) to Colab or mount Drive.
#   2. Upload the processed test/train arrays (X_single, X_seq, y).
#   3. Run all cells top-to-bottom. GPU runtime recommended.
#
# REQUIREMENTS:
#   tensorflow>=2.13, xgboost>=1.7, scikit-learn, numpy, joblib, matplotlib
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install & Import Dependencies
# ─────────────────────────────────────────────────────────────────────────────


from tensorflow.keras.models import load_model
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tf_keras  as keras

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

print("TensorFlow  :", tf.__version__)
print("XGBoost     :", xgb.__version__)
print("GPU devices :", tf.config.list_physical_devices('GPU'))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Configuration
# ─────────────────────────────────────────────────────────────────────────────

# ── Paths ──────────────────────────────────────────────────────────────────
# Adjust these paths to where your files live (Drive mount or Colab uploads)
MODEL_DIR   = "/content/drive/MyDrive/NeuralSoc/models"          # folder with .h5 / .pkl files
DATA_DIR    = "/content/drive/MyDrive/NeuralSoc/data"  # folder with numpy arrays

CNN_PATH         = os.path.join(MODEL_DIR, "cnn_model.h5")

TRANSFORMER_PATH = os.path.join(MODEL_DIR, "model_transformer.keras")

AUTOENCODER_PATH = os.path.join(MODEL_DIR, "autoencoder_best.h5")

SCALER_PATH      = os.path.join(MODEL_DIR, "scaler.pkl")
META_LEARNER_SAVE = os.path.join(MODEL_DIR, "meta_learner.pkl")

# ── Dataset config ──────────────────────────────────────────────────────────
NUM_CLASSES     = 16        # 1 Benign + 15 Attack categories (CICIDS-2017)
SEQUENCE_LENGTH = 10        # Sliding window length for BiLSTM & Transformer
SINGLE_FEATURES = 40        # Features fed to 1D-CNN (scalar input)
ANOMALY_PERCENTILE = 95     # Reconstruction error threshold for zero-day flag

# ── XGBoost hyperparameters ─────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators"      : 400,
    "max_depth"         : 6,
    "learning_rate"     : 0.05,
    "subsample"         : 0.8,
    "colsample_bytree"  : 0.8,
    "min_child_weight"  : 3,
    "gamma"             : 0.1,
    "reg_alpha"         : 0.1,   # L1 regularisation
    "reg_lambda"        : 1.0,   # L2 regularisation
    "objective"         : "multi:softprob",
    "eval_metric"       : "mlogloss",
    "use_label_encoder" : False,
    "tree_method"       : "gpu_hist",   # change to "hist" if no GPU
    "random_state"      : 42,
    "n_jobs"            : -1,
}

ATTACK_CLASSES = [
    "BENIGN", "DDoS", "PortScan", "BruteForce_FTP", "BruteForce_SSH",
    "DoS_Hulk", "DoS_GoldenEye", "DoS_Slowloris", "DoS_SlowHTTPTest",
    "Heartbleed", "WebAttack_Brute", "WebAttack_XSS", "WebAttack_SQLi",
    "Infiltration", "Bot", "Unknown_ZeroDay"
]

# ── 16. Standalone inference (new session / new file) ────────────────────────
# Copy-paste this block into ANY new notebook or script.
# You need these files downloaded from training:
#   bilstm_cicids2017.weights.h5  ← weights
#   model_cfg.json                ← architecture params
#   scaler.pkl                    ← feature scaler
#   label_encoder.pkl             ← class name mapping
#
# NO version conflicts — load_weights() only reads tensors, no config parsing.

import pickle, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Concatenate, Conv1D,
    BatchNormalization, Activation, Multiply, Reshape, GlobalMaxPooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# ── Must define + register custom objects before loading weights ──────────────
@tf.keras.utils.register_keras_serializable(package="NeuralSOC")
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, **kwargs):
        super().__init__(**kwargs); self.gamma = float(gamma)
    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
        y_oh = tf.one_hot(y_true, tf.shape(y_pred)[1])
        p_t  = tf.reduce_sum(y_pred * y_oh, axis=1)
        return tf.reduce_mean(tf.pow(1-p_t, self.gamma) * (-tf.math.log(p_t)))
    def get_config(self):
        cfg = super().get_config(); cfg.update({"gamma": self.gamma}); return cfg

@tf.keras.utils.register_keras_serializable(package="NeuralSOC")
class SqueezeExciteBlock(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs); self.ratio = ratio
    def build(self, input_shape):
        f = int(input_shape[-1])
        self.d_reduce = Dense(max(f//self.ratio,1), activation="relu")
        self.d_expand = Dense(f, activation="sigmoid")
        self.reshape  = Reshape((1, f))
        super().build(input_shape)
    def call(self, x):
        se = self.reshape(self.d_expand(self.d_reduce(GlobalAveragePooling1D()(x))))
        return Multiply()([x, se])
    def get_config(self):
        cfg = super().get_config(); cfg.update({"ratio": self.ratio}); return cfg

@tf.keras.utils.register_keras_serializable(package="NeuralSOC")
class TemporalAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads=4, key_dim=32, attn_dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads=num_heads; self.key_dim=key_dim; self.attn_dropout=attn_dropout
    def build(self, input_shape):
        self.mha = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, dropout=self.attn_dropout)
        self.ln  = LayerNormalization(epsilon=1e-6)
        super().build(input_shape)
    def call(self, x, training=False):
        return self.ln(x + self.mha(x, x, training=training))
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_heads":self.num_heads,"key_dim":self.key_dim,"attn_dropout":self.attn_dropout})
        return cfg

def build_bilstm_ids(input_shape, num_classes, dropout_rate=0.35, l2_reg=1e-4,
                     learning_rate=3e-4, focal_gamma=2.0, use_top2_metric=True):
    reg = l2(l2_reg)
    inputs = Input(shape=input_shape, name="flow_window")
    x = Conv1D(64, 3, padding="same", kernel_regularizer=reg, name="conv_local")(inputs)
    x = BatchNormalization(name="bn_conv")(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate * 0.5)(x)
    x = SqueezeExciteBlock(ratio=8, name="se_block")(x)
    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=reg, recurrent_dropout=0.0), name="bilstm_1")(x)
    x = LayerNormalization(name="ln_bilstm1")(x)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=reg, recurrent_dropout=0.0), name="bilstm_2")(x)
    x = LayerNormalization(name="ln_bilstm2")(x)
    x = Dropout(dropout_rate)(x)
    x = TemporalAttentionBlock(num_heads=4, key_dim=32, attn_dropout=dropout_rate*0.25, name="temporal_attn")(x)
    x = Concatenate(name="dual_pool")([GlobalAveragePooling1D(name="pool_avg")(x), GlobalMaxPooling1D(name="pool_max")(x)])
    x = Dense(128, kernel_regularizer=reg, name="fc1")(x)
    x = BatchNormalization(name="bn_fc1")(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, kernel_regularizer=reg, name="fc2")(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate * 0.5)(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)
    model = Model(inputs, outputs, name="BiLSTM_IDS_CICIDS2017")
    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=l2_reg)
    except AttributeError:
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=SparseCategoricalFocalLoss(gamma=focal_gamma), metrics=["accuracy"])
    return model

# ── Load ──────────────────────────────────────────────────────────────────────
with open("/content/drive/MyDrive/NeuralSoc/models/model_cfg.json")       as f: mcfg   = json.load(f)
with open("/content/drive/MyDrive/NeuralSoc/models/scaler.pkl",    "rb") as f: scaler = pickle.load(f)
with open("/content/drive/MyDrive/NeuralSoc/models/label_encoder.pkl","rb") as f: le   = pickle.load(f)

bilstm_model = build_bilstm_ids(
    input_shape     = tuple(mcfg["input_shape"]),
    num_classes     = mcfg["num_classes"],
    dropout_rate    = mcfg["dropout_rate"],
    focal_gamma     = mcfg["focal_gamma"],
    l2_reg          = mcfg["l2_reg"],
    learning_rate   = mcfg["learning_rate"],
    use_top2_metric = mcfg["use_top2_metric"],
)
bilstm_model.load_weights("/content/drive/MyDrive/NeuralSoc/models/bilstm_cicids2017.weights.h5")
print("✅  Model loaded — ready for inference")
print("Class mapping:", dict(enumerate(le.classes_)))

# Predict on new data:
# X_new_scaled  = scaler.transform(X_new_raw)         # shape (N, n_features)
# X_windows     = build_windows(X_new_scaled, ..., window_size=mcfg['window'])
# preds_idx     = np.argmax(model.predict(X_windows), axis=1)
# preds_label   = le.inverse_transform(preds_idx)


print("Loading base models …")

cnn_model         = load_model(CNN_PATH)

@tf.keras.utils.register_keras_serializable()
class ChunkSlice(tf.keras.layers.Layer):
    def __init__(self, start, end, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end = end

    def call(self, inputs):
        return inputs[:, self.start:self.end]

    def get_config(self):
        config = super().get_config()
        config.update({
            "start": self.start,
            "end": self.end,
        })
        return config
@tf.keras.utils.register_keras_serializable()
class ReduceMeanAxis1(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class ExtractCLS(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs[:, 0, :]

    def get_config(self):
        return super().get_config()
transformer_model = tf.keras.models.load_model(
    TRANSFORMER_PATH,
    custom_objects={'ChunkSlice': ChunkSlice, 'ReduceMeanAxis1': ReduceMeanAxis1, 'ExtractCLS': ExtractCLS},
    compile=False
)


autoencoder       = load_model(AUTOENCODER_PATH, compile=False)

cnn_model.summary()
transformer_model.summary()
autoencoder.summary()

# scaler            = joblib.load(SCALER_PATH)
print("✅ All models loaded successfully.")




# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Load & Prepare Data
# ─────────────────────────────────────────────────────────────────────────────
# Expected numpy files in DATA_DIR:
#   X_single.npy  — shape (N, 40)   : scaled scalar features for CNN
#   X_seq.npy     — shape (N, 10, 40): sequence windows for BiLSTM & Transformer
#   y.npy         — shape (N,)       : integer class labels 0–15

X_single = np.load(os.path.join(DATA_DIR, "X_single.npy"))
X_seq    = np.load(os.path.join(DATA_DIR, "X_seq.npy"))
y        = np.load(os.path.join(DATA_DIR, "y.npy")).astype(int)

print(f"X_single : {X_single.shape}")
print(f"X_seq    : {X_seq.shape}")
print(f"y        : {y.shape}  |  Classes: {np.unique(y)}")

# Train / Validation / Test split (70 / 15 / 15)
X_s_tmp, X_s_test, X_q_tmp, X_q_test, y_tmp, y_test = train_test_split(
    X_single, X_seq, y, test_size=0.15, stratify=y, random_state=42
)
X_s_train, X_s_val, X_q_train, X_q_val, y_train, y_val = train_test_split(
    X_s_tmp, X_q_tmp, y_tmp, test_size=0.1765, stratify=y_tmp, random_state=42
)

print(f"\nTrain : {y_train.shape[0]:,} | Val : {y_val.shape[0]:,} | Test : {y_test.shape[0]:,}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Generate Meta-Features (Probability Vectors from Base Models)
# ─────────────────────────────────────────────────────────────────────────────

def get_meta_features(X_single, X_seq, batch_size=2048):
    """
    Runs inference on all three classifiers and the autoencoder.
    Returns a (N, 49) meta-feature matrix:
      - cols  0–15  : CNN softmax probs
      - cols 16–31  : BiLSTM softmax probs
      - cols 32–47  : Transformer softmax probs
      - col  48     : Autoencoder reconstruction error (zero-day score)
    """
    print("  → CNN inference …")
    # CNN expects (N, 40, 1)
    cnn_probs = cnn_model.predict(
        X_single[..., np.newaxis], batch_size=batch_size, verbose=0
    )  # (N, 16)

    print("  → BiLSTM inference …")
    bilstm_probs = bilstm_model.predict(
        X_seq, batch_size=batch_size, verbose=0
    )  # (N, 16)

    print("  → Transformer inference …")
    transformer_probs = transformer_model.predict(
        X_seq, batch_size=batch_size, verbose=0
    )  # (N, 16)

    print("  → Autoencoder reconstruction error …")
    X_reconstructed = autoencoder.predict(
        X_single[..., np.newaxis], batch_size=batch_size, verbose=0
    )  # (N, 40, 1)
    recon_error = np.mean(
        np.square(X_single[..., np.newaxis] - X_reconstructed), axis=(1, 2)
    )  # (N,)

    # Concatenate → (N, 49)
    meta = np.concatenate(
        [cnn_probs, bilstm_probs, transformer_probs, recon_error[:, np.newaxis]],
        axis=1
    )
    print(f"  Meta-feature matrix shape: {meta.shape}")
    return meta, recon_error


print("\n📐 Generating meta-features for TRAIN set …")
X_meta_train, recon_train = get_meta_features(X_s_train, X_q_train)

print("\n📐 Generating meta-features for VAL set …")
X_meta_val,   recon_val   = get_meta_features(X_s_val,   X_q_val)

print("\n📐 Generating meta-features for TEST set …")
X_meta_test,  recon_test  = get_meta_features(X_s_test,  X_q_test)



# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Compute Zero-Day Threshold (from training benign traffic only)
# ─────────────────────────────────────────────────────────────────────────────

benign_idx = np.where(y_train == 0)[0]  # class 0 = BENIGN
zero_day_threshold = np.percentile(recon_train[benign_idx], ANOMALY_PERCENTILE)
print(f"\n🔵 Zero-day reconstruction error threshold "
      f"(p{ANOMALY_PERCENTILE}): {zero_day_threshold:.6f}")


def zero_day_flag(recon_errors, threshold):
    """Returns bool array — True = potential zero-day / novel attack."""
    return recon_errors > threshold

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Train XGBoost Meta-Learner
# ─────────────────────────────────────────────────────────────────────────────

print("\n🚀 Training XGBoost meta-learner …")

xgb_meta = XGBClassifier(**XGB_PARAMS)

xgb_meta.fit(
    X_meta_train, y_train,
    eval_set=[(X_meta_val, y_val)],
    verbose=50,             # print every 50 trees
    early_stopping_rounds=30
)

print(f"\n✅ Best iteration: {xgb_meta.best_iteration}")
print(f"✅ Best val log-loss: {xgb_meta.best_score:.4f}")

# Save meta-learner
joblib.dump(xgb_meta, META_LEARNER_SAVE)
print(f"💾 Meta-learner saved → {META_LEARNER_SAVE}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Evaluate Ensemble on Test Set
# ─────────────────────────────────────────────────────────────────────────────

y_pred_proba = xgb_meta.predict_proba(X_meta_test)  # (N, 16)
y_pred       = np.argmax(y_pred_proba, axis=1)
confidence   = np.max(y_pred_proba, axis=1)
zd_flags     = zero_day_flag(recon_test, zero_day_threshold)

acc    = accuracy_score(y_test, y_pred)
f1_mac = f1_score(y_test, y_pred, average="macro")
f1_w   = f1_score(y_test, y_pred, average="weighted")

print("\n" + "="*60)
print("  NeuralSOC ENSEMBLE — TEST SET PERFORMANCE")
print("="*60)
print(f"  Accuracy          : {acc*100:.2f}%")
print(f"  F1-Score (macro)  : {f1_mac:.4f}")
print(f"  F1-Score (weighted): {f1_w:.4f}")
print(f"  Zero-Day Detections: {zd_flags.sum()} / {len(zd_flags)}")
print("="*60)

present_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
target_names   = [ATTACK_CLASSES[i] for i in present_labels]

print("\n📋 Per-Class Classification Report:\n")
print(classification_report(
    y_test, y_pred,
    labels=present_labels,
    target_names=target_names,
    digits=4
))
# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Confusion Matrix (Heatmap)
# ─────────────────────────────────────────────────────────────────────────────

cm = confusion_matrix(y_test, y_pred, labels=present_labels)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(
    cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
    xticklabels=target_names, yticklabels=target_names,
    linewidths=0.4, linecolor="#333",
    cbar_kws={"label": "Recall"},
    ax=ax
)
ax.set_title("NeuralSOC Ensemble — Normalised Confusion Matrix", fontsize=14, pad=16)
ax.set_xlabel("Predicted Class", labelpad=10)
ax.set_ylabel("True Class", labelpad=10)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("/content/confusion_matrix_ensemble.png", dpi=150)
plt.show()
print("📊 Confusion matrix saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — Feature Importance (Which base model contributes most?)
# ─────────────────────────────────────────────────────────────────────────────

feature_names = (
    [f"CNN_cls{i}"         for i in range(NUM_CLASSES)] +
    [f"BiLSTM_cls{i}"      for i in range(NUM_CLASSES)] +
    [f"Transformer_cls{i}" for i in range(NUM_CLASSES)] +
    ["Autoencoder_ReconErr"]
)

importances = xgb_meta.feature_importances_

# Aggregate importance per model block
cnn_imp         = importances[:16].sum()
bilstm_imp      = importances[16:32].sum()
transformer_imp = importances[32:48].sum()
ae_imp          = importances[48]

print("\n📊 Aggregated Feature Importance by Model Block:")
print(f"  CNN              : {cnn_imp:.4f}  ({cnn_imp*100:.1f}%)")
print(f"  BiLSTM           : {bilstm_imp:.4f}  ({bilstm_imp*100:.1f}%)")
print(f"  Transformer      : {transformer_imp:.4f}  ({transformer_imp*100:.1f}%)")
print(f"  Autoencoder (ZD) : {ae_imp:.4f}  ({ae_imp*100:.1f}%)")

fig2, axes = plt.subplots(1, 2, figsize=(16, 5))

# Bar chart — per model block
bars = axes[0].bar(
    ["1D-CNN", "BiLSTM", "Transformer", "AE (ZeroDay)"],
    [cnn_imp, bilstm_imp, transformer_imp, ae_imp],
    color=["#00b4d8", "#0077b6", "#90e0ef", "#ef233c"],
    edgecolor="black", linewidth=0.8
)
axes[0].set_title("Ensemble — Per-Model Block Importance", fontsize=12)
axes[0].set_ylabel("Sum of F-score Importances")
for bar, val in zip(bars, [cnn_imp, bilstm_imp, transformer_imp, ae_imp]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10)

# Top-20 individual features
top20_idx  = np.argsort(importances)[::-1][:20]
top20_feat = [feature_names[i] for i in top20_idx]
top20_imp  = importances[top20_idx]
colors = []
for i in top20_idx:
    if i < 16:       colors.append("#00b4d8")
    elif i < 32:     colors.append("#0077b6")
    elif i < 48:     colors.append("#90e0ef")
    else:            colors.append("#ef233c")

axes[1].barh(top20_feat[::-1], top20_imp[::-1], color=colors[::-1],
             edgecolor="black", linewidth=0.5)
axes[1].set_title("Top-20 Meta-Features by Importance", fontsize=12)
axes[1].set_xlabel("F-score Importance")

plt.tight_layout()
plt.savefig("/content/feature_importance_ensemble.png", dpi=150)
plt.show()
print("📊 Feature importance chart saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — Single-Flow Inference Demo
# ─────────────────────────────────────────────────────────────────────────────

def predict_single_flow(raw_features_40: np.ndarray,
                        raw_sequence_10x40: np.ndarray) -> dict:
    """
    Full ensemble inference for ONE network flow.

    Args:
        raw_features_40   : shape (40,) — pre-scaled scalar features
        raw_sequence_10x40: shape (10, 40) — pre-scaled sequence window

    Returns:
        dict with predicted class, confidence, zero-day flag, per-model probs
    """
    sf = raw_features_40[np.newaxis, ..., np.newaxis]    # (1, 40, 1)
    sq = raw_sequence_10x40[np.newaxis]                   # (1, 10, 40)

    cnn_p   = cnn_model.predict(sf, verbose=0)[0]
    bil_p   = bilstm_model.predict(sq, verbose=0)[0]
    tra_p   = transformer_model.predict(sq, verbose=0)[0]

    recon   = autoencoder.predict(sf, verbose=0)
    re_err  = float(np.mean(np.square(sf - recon)))

    meta_f  = np.concatenate([cnn_p, bil_p, tra_p, [re_err]])[np.newaxis]
    proba   = xgb_meta.predict_proba(meta_f)[0]
    pred_cls = int(np.argmax(proba))

    return {
        "predicted_class"  : ATTACK_CLASSES[pred_cls],
        "class_index"      : pred_cls,
        "confidence"       : round(float(np.max(proba)) * 100, 2),
        "zero_day_flag"    : bool(re_err > zero_day_threshold),
        "recon_error"      : round(re_err, 6),
        "zd_threshold"     : round(zero_day_threshold, 6),
        "all_probabilities": {ATTACK_CLASSES[i]: round(float(p)*100, 2)
                              for i, p in enumerate(proba)},
        "base_model_top1"  : {
            "CNN"        : ATTACK_CLASSES[int(np.argmax(cnn_p))],
            "BiLSTM"     : ATTACK_CLASSES[int(np.argmax(bil_p))],
            "Transformer": ATTACK_CLASSES[int(np.argmax(tra_p))],
        }
    }


# ── Demo with a random test sample ──────────────────────────────────────────
idx = np.random.randint(len(X_s_test))
result = predict_single_flow(X_s_test[idx], X_q_test[idx])

print("\n" + "="*55)
print("  SINGLE FLOW PREDICTION DEMO")
print("="*55)
for k, v in result.items():
    if k != "all_probabilities":
        print(f"  {k:<22}: {v}")
print(f"\n  True label: {ATTACK_CLASSES[y_test[idx]]}")
print("="*55)
print("\n  Full class probability breakdown:")
for cls, prob in sorted(result["all_probabilities"].items(),
                        key=lambda x: -x[1])[:8]:
    bar = "█" * int(prob / 2)
    print(f"    {cls:<25} {prob:>6.2f}%  {bar}")
# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 — Save Threshold & Summary Metadata
# ─────────────────────────────────────────────────────────────────────────────

import json

metadata = {
    "num_classes"         : NUM_CLASSES,
    "attack_classes"      : ATTACK_CLASSES,
    "meta_feature_dim"    : 49,
    "zero_day_threshold"  : float(zero_day_threshold),
    "anomaly_percentile"  : ANOMALY_PERCENTILE,
    "test_accuracy"       : round(acc, 4),
    "test_f1_macro"       : round(f1_mac, 4),
    "test_f1_weighted"    : round(f1_w, 4),
    "xgb_best_iteration"  : int(xgb_meta.best_iteration),
}

with open("/content/ensemble_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n📁 Files saved to /content/:")
print("  ✅ meta_learner.pkl")
print("  ✅ confusion_matrix_ensemble.png")
print("  ✅ feature_importance_ensemble.png")
print("  ✅ ensemble_metadata.json")
print("\n🏁 NeuralSOC Ensemble training complete.")

