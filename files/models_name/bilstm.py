"""
BiLSTM model for CICIDS-2017 network intrusion detection.

Input: (batch, timesteps, features) — sliding-window network flow features
       e.g. shape (N, 10, 78) for 78-feature CICIDS windows of length 10

Output: softmax over traffic classes
        e.g. [BENIGN, DoS, PortScan, BruteForce, ...]
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout,
    LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Concatenate, Conv1D,
    BatchNormalization, Activation, Multiply, Reshape,
    GlobalMaxPooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


# ── Channel-wise squeeze-and-excitation (helps with redundant CICIDS features) ─

def squeeze_excite_block(x, ratio: int = 8):
    """
    Recalibrates feature-channel importance.
    Useful because CICIDS has many correlated / near-zero-variance columns.
    """
    filters = x.shape[-1]
    se = GlobalAveragePooling1D()(x)                        # (B, F)
    se = Dense(max(filters // ratio, 1), activation="relu")(se)
    se = Dense(filters, activation="sigmoid")(se)           # (B, F)
    se = Reshape((1, filters))(se)                          # (B, 1, F)
    return Multiply()([x, se])                              # broadcast over T


# ── Temporal self-attention (multi-head, lighter than original Attention()) ────

def temporal_attention_block(x, num_heads: int = 4, key_dim: int = 32,
                              dropout: float = 0.1):
    """
    Multi-head self-attention over the time axis.
    Lets the model focus on specific flow windows (e.g. burst onset).
    """
    attn_out = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout
    )(x, x)                                                 # self-attention
    x = LayerNormalization(epsilon=1e-6)(x + attn_out)      # residual + LN
    return x


# ── Main model ─────────────────────────────────────────────────────────────────

def build_bilstm_ids(
    input_shape: tuple = (10, 78),   # (window_len, n_features)
    num_classes: int = 8,            # CICIDS-2017 has up to 14 attack types
    lstm_units: tuple = (128, 64),
    dense_units: int = 128,
    dropout_rate: float = 0.4,       # slightly higher — CICIDS can overfit fast
    l2_reg: float = 1e-4,
    learning_rate: float = 3e-4,
    label_smoothing: float = 0.05,   # combats noisy CICIDS labels
    use_top2_metric: bool = True,    # set False when num_classes < 3 (binary)
) -> Model:
    """
    BiLSTM + Temporal Attention for CICIDS-2017 intrusion detection.

    Architecture choices justified for network flow data:
    - Conv1D front-end: extracts local temporal patterns (e.g. burst within window)
    - Two BiLSTM layers: capture bidirectional dependencies across flow timesteps
    - Squeeze-and-Excite: down-weights low-variance / near-constant CICIDS columns
    - Multi-head self-attention: highlights anomalous sub-sequences
    - Dual pooling (avg + max): avg captures background, max captures attack spikes
    - Label smoothing: CICIDS-2017 has mislabelled flows, smoothing helps
    """
    inputs = Input(shape=input_shape, name="flow_window")

    # ── 1. Local feature extraction (intra-window patterns) ───────────────────
    x = Conv1D(64, kernel_size=3, padding="same",
               kernel_regularizer=l2(l2_reg), name="conv_local")(inputs)
    x = BatchNormalization(name="bn_conv")(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate * 0.5)(x)

    # ── 2. Squeeze-and-Excite (feature recalibration) ─────────────────────────
    x = squeeze_excite_block(x, ratio=8)

    # ── 3. BiLSTM stack ───────────────────────────────────────────────────────
    x = Bidirectional(
        LSTM(lstm_units[0], return_sequences=True,
             kernel_regularizer=l2(l2_reg),
             recurrent_dropout=0.0),          # keep recurrent_dropout=0 for CuDNN
        name="bilstm_1"
    )(x)
    x = LayerNormalization(name="ln_bilstm1")(x)
    x = Dropout(dropout_rate)(x)

    x = Bidirectional(
        LSTM(lstm_units[1], return_sequences=True,
             kernel_regularizer=l2(l2_reg),
             recurrent_dropout=0.0),
        name="bilstm_2"
    )(x)
    x = LayerNormalization(name="ln_bilstm2")(x)
    x = Dropout(dropout_rate)(x)

    # ── 4. Temporal self-attention ────────────────────────────────────────────
    x = temporal_attention_block(x, num_heads=4, key_dim=32,
                                 dropout=dropout_rate * 0.25)

    # ── 5. Dual pooling — avg catches background, max catches attack peaks ────
    avg_pool = GlobalAveragePooling1D(name="pool_avg")(x)
    max_pool = GlobalMaxPooling1D(name="pool_max")(x)
    x = Concatenate(name="dual_pool")([avg_pool, max_pool])

    # ── 6. Classifier head ────────────────────────────────────────────────────
    x = Dense(dense_units, kernel_regularizer=l2(l2_reg), name="fc1")(x)
    x = BatchNormalization(name="bn_fc1")(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(dense_units // 2, kernel_regularizer=l2(l2_reg), name="fc2")(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate * 0.5)(x)

    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="BiLSTM_IDS_CICIDS2017")

    # SparseCategoricalCrossentropy with label_smoothing via a wrapper
    # (Keras sparse variant doesn't expose smoothing directly, so we use
    #  CategoricalCrossentropy after one-hot — handled in train script instead)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=l2_reg,
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=(
            ["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc")]
            if use_top2_metric else ["accuracy"]
        ),
    )
    return model


if __name__ == "__main__":
    m = build_bilstm_ids(input_shape=(10, 78), num_classes=8)
    m.summary()