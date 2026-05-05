# doing all the training over the google colab so the files are made accordingly only 

# =============================================================================
# NeuralSOC ⚡ — CICIDS-2017 Preprocessing Pipeline
# =============================================================================
# Aligned exactly to neuralsoc_xgboost_ensemble.py
#
# WHAT THIS PRODUCES (saved to DATA_OUT_DIR):
#   X_single.npy        shape (N, 40)      → CNN input
#   X_seq.npy           shape (N, 10, 40)  → BiLSTM & Transformer input
#   y.npy               shape (N,)         → integer labels 0–14
#
# WHAT THIS PRODUCES (saved to MODEL_DIR):
#   scaler.pkl          StandardScaler  → used at inference in FastAPI
#   label_encoder.pkl   LabelEncoder    → maps integer ↔ class string
#   model_cfg.json      BiLSTM build params → used to rebuild BiLSTM arch
#
# KEY ALIGNMENT POINTS WITH THE ENSEMBLE FILE:
#   • 40 features       — matches SINGLE_FEATURES = 40
#   • window size 10    — matches SEQUENCE_LENGTH = 10
#   • labels 0–14       — class 15 is reserved for zero-day (autoencoder only)
#   • scaler.pkl        — same file loaded by ensemble + FastAPI backend
#   • label_encoder.pkl — same file loaded by BiLSTM standalone block
#   • model_cfg.json    — same file loaded by BiLSTM standalone block
#
# CICIDS-2017 CSV FILES (all 8, place in DATA_RAW_DIR):
#   Monday-WorkingHours.pcap_ISCX.csv
#   Tuesday-WorkingHours.pcap_ISCX.csv
#   Wednesday-workingHours.pcap_ISCX.csv
#   Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
#   Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
#   Friday-WorkingHours-Morning.pcap_ISCX.csv
#   Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
#   Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
#
# HOW TO RUN (Google Colab):
#   Runtime → Change runtime type → T4 GPU (optional but faster)
#   Run all cells top to bottom in order.
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install & Imports
# ─────────────────────────────────────────────────────────────────────────────

# Uncomment the line below if running fresh in Colab:
# !pip install imbalanced-learn --quiet

import os, glob, json, pickle, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
from collections import Counter

print("✅ Libraries loaded")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Mount Google Drive  (skip if uploading CSVs directly)
# ─────────────────────────────────────────────────────────────────────────────

from google.colab import drive
drive.mount("/content/drive")
print("✅ Drive mounted")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Configuration
# (All paths must match exactly what neuralsoc_xgboost_ensemble.py uses)
# ─────────────────────────────────────────────────────────────────────────────

# ── Paths — edit ONLY these two lines ───────────────────────────────────────
DATA_RAW_DIR = "/content/drive/MyDrive/NeuralSoc/data/raw"   # your CICIDS CSVs
DATA_OUT_DIR = "/content/drive/MyDrive/NeuralSoc/data"       # X_single, X_seq, y
MODEL_DIR    = "/content/drive/MyDrive/NeuralSoc/models"     # scaler, le, cfg

os.makedirs(DATA_OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,    exist_ok=True)

# ── Must match ensemble file exactly ────────────────────────────────────────
SEQUENCE_LENGTH = 10      # SEQUENCE_LENGTH in ensemble file
SINGLE_FEATURES = 40      # SINGLE_FEATURES in ensemble file
STEP_SIZE       = 1       # sliding window stride
RANDOM_STATE    = 42

# Cap per class — prevents Colab RAM OOM on the 2M-row BENIGN class
# Set to None to use the full dataset (needs ~25 GB RAM)
MAX_SAMPLES_PER_CLASS = 150_000

# ── BiLSTM model_cfg.json values ────────────────────────────────────────────
# Must match the hyperparams used when you trained your BiLSTM.
# Edit only if you used different values during training.
BILSTM_CFG = {
    "input_shape"     : [SEQUENCE_LENGTH, SINGLE_FEATURES],  # [10, 40]
    "num_classes"     : 15,       # 0–14 supervised; 15 = zero-day (AE only)
    "dropout_rate"    : 0.35,
    "focal_gamma"     : 2.0,
    "l2_reg"          : 1e-4,
    "learning_rate"   : 3e-4,
    "use_top2_metric" : True,
    "window"          : SEQUENCE_LENGTH,
}

# ── The 40 features — exact column names as they appear in CICIDS-2017 CSVs ─
# Leading/trailing spaces are intentional — they match the raw CSV headers.
SELECTED_FEATURES = [
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    "Total Length of Fwd Packets",
    " Total Length of Bwd Packets",
    " Fwd Packet Length Max",
    " Fwd Packet Length Min",
    " Fwd Packet Length Mean",
    " Fwd Packet Length Std",
    "Bwd Packet Length Max",
    " Bwd Packet Length Min",
    " Bwd Packet Length Mean",
    " Bwd Packet Length Std",
    " Flow Bytes/s",
    " Flow Packets/s",
    " Flow IAT Mean",
    " Flow IAT Std",
    " Flow IAT Max",
    " Flow IAT Min",
    "Fwd IAT Total",
    " Fwd IAT Mean",
    " Fwd IAT Std",
    " Fwd IAT Max",
    " Fwd IAT Min",
    "Bwd IAT Total",
    " Bwd IAT Mean",
    " Bwd IAT Std",
    " Bwd IAT Max",
    " Bwd IAT Min",
    "Fwd PSH Flags",
    " Fwd URG Flags",
    " Fwd Header Length",
    " Bwd Header Length",
    "Fwd Packets/s",
    " Bwd Packets/s",
    " Min Packet Length",
    " Max Packet Length",
    " Packet Length Mean",
    " Packet Length Std",
    " Packet Length Variance",
]
assert len(SELECTED_FEATURES) == 40, \
    f"Expected 40 features, got {len(SELECTED_FEATURES)}"

# ── Raw label strings → integer index (0–14) ────────────────────────────────
# Covers all spelling/encoding variants across the 8 CICIDS-2017 CSV files.
LABEL_MAP = {
    "BENIGN"                          : 0,
    "DDoS"                            : 1,
    "PortScan"                        : 2,
    "FTP-Patator"                     : 3,
    "SSH-Patator"                     : 4,
    "DoS Hulk"                        : 5,
    "DoS GoldenEye"                   : 6,
    "DoS slowloris"                   : 7,
    "DoS Slowhttptest"                : 8,
    "Heartbleed"                      : 9,
    "Web Attack \x96 Brute Force"     : 10,
    "Web Attack \u2013 Brute Force"   : 10,
    "Web Attack \u2014 Brute Force"   : 10,
    "Web Attack – Brute Force"        : 10,
    "Web Attack \x96 XSS"             : 11,
    "Web Attack \u2013 XSS"           : 11,
    "Web Attack – XSS"                : 11,
    "Web Attack \x96 Sql Injection"   : 12,
    "Web Attack \u2013 Sql Injection" : 12,
    "Web Attack – Sql Injection"      : 12,
    "Infiltration"                    : 13,
    "Bot"                             : 14,
}

# ── Class names — must EXACTLY match ATTACK_CLASSES[0:15] in ensemble file ──
ATTACK_CLASSES = [
    "BENIGN",           # 0
    "DDoS",             # 1
    "PortScan",         # 2
    "BruteForce_FTP",   # 3
    "BruteForce_SSH",   # 4
    "DoS_Hulk",         # 5
    "DoS_GoldenEye",    # 6
    "DoS_Slowloris",    # 7
    "DoS_SlowHTTPTest", # 8
    "Heartbleed",       # 9
    "WebAttack_Brute",  # 10
    "WebAttack_XSS",    # 11
    "WebAttack_SQLi",   # 12
    "Infiltration",     # 13
    "Bot",              # 14
    # index 15 = "Unknown_ZeroDay" — autoencoder only, NOT in y.npy
]

print("✅ Configuration set")
print(f"   DATA_RAW_DIR : {DATA_RAW_DIR}")
print(f"   DATA_OUT_DIR : {DATA_OUT_DIR}")
print(f"   MODEL_DIR    : {MODEL_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Load & Concatenate All CICIDS-2017 CSVs
# ─────────────────────────────────────────────────────────────────────────────

csv_files = sorted(glob.glob(os.path.join(DATA_RAW_DIR, "*.csv")))
print(f"\nFound {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  {os.path.basename(f)}")

if len(csv_files) == 0:
    raise FileNotFoundError(
        f"No CSV files found in {DATA_RAW_DIR}. "
        "Check your Drive path or re-upload the files."
    )

frames = []
for path in csv_files:
    print(f"\nLoading {os.path.basename(path)} …", end=" ")
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    # Strip ALL column whitespace — critical for CICIDS-2017
    df.columns = df.columns.str.strip()
    print(f"{len(df):,} rows  {df.shape[1]} cols")
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)
print(f"\n✅ Total rows loaded : {len(raw):,}")
print(f"   Total columns     : {raw.shape[1]}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Detect Label Column & Map to Integers
# ─────────────────────────────────────────────────────────────────────────────

label_col_candidates = [c for c in raw.columns if "label" in c.lower()]
print(f"Label column candidates : {label_col_candidates}")

if not label_col_candidates:
    raise ValueError(
        "No label column found.\nAvailable columns:\n" +
        "\n".join(raw.columns.tolist())
    )

LABEL_COL = label_col_candidates[0]
print(f"Using : '{LABEL_COL}'")

print("\nRaw label value counts:")
print(raw[LABEL_COL].value_counts().to_string())

# Strip whitespace from label strings before mapping
raw[LABEL_COL] = raw[LABEL_COL].astype(str).str.strip()
raw["label_int"] = raw[LABEL_COL].map(LABEL_MAP)

unmapped = raw[raw["label_int"].isna()][LABEL_COL].unique()
if len(unmapped) > 0:
    print(f"\n⚠️  Unmapped label strings found (will be dropped):")
    for u in unmapped:
        print(f"     '{u}'")

raw = raw.dropna(subset=["label_int"]).copy()
raw["label_int"] = raw["label_int"].astype(int)

print("\nMapped label distribution:")
for idx, count in sorted(Counter(raw["label_int"]).items()):
    print(f"  {idx:>2}  {ATTACK_CLASSES[idx]:<25} {count:>10,}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Build & Save LabelEncoder
# (needed by the BiLSTM standalone block in the ensemble file)
# le.classes_[i] == ATTACK_CLASSES[i]  for i in 0..14
# ─────────────────────────────────────────────────────────────────────────────

le = LabelEncoder()
le.fit(ATTACK_CLASSES[:15])          # supervised classes only (0–14)

le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
with open(le_path, "wb") as f:
    pickle.dump(le, f)

print(f"✅ label_encoder.pkl saved → {le_path}")
print(f"   le.classes_ : {list(le.classes_)}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Save model_cfg.json
# (needed by the BiLSTM standalone block: build_bilstm_ids(**mcfg))
# ─────────────────────────────────────────────────────────────────────────────

cfg_path = os.path.join(MODEL_DIR, "model_cfg.json")
with open(cfg_path, "w") as f:
    json.dump(BILSTM_CFG, f, indent=2)

print(f"✅ model_cfg.json saved → {cfg_path}")
print(json.dumps(BILSTM_CFG, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Feature Selection & Cleaning
# ─────────────────────────────────────────────────────────────────────────────

available_cols = set(raw.columns)
missing_cols   = [f for f in SELECTED_FEATURES if f not in available_cols]

if missing_cols:
    print(f"\n⚠️  {len(missing_cols)} feature(s) missing — zero-filled:")
    for c in missing_cols:
        print(f"     '{c}'")
        raw[c] = 0.0
else:
    print("✅ All 40 selected features found in dataframe")

X_raw = raw[SELECTED_FEATURES].copy()
y_raw = raw["label_int"].values

# Force all columns to numeric
X_raw = X_raw.apply(pd.to_numeric, errors="coerce")

# Replace inf / -inf  (known issue in CICIDS Flow Bytes/s column)
inf_count = np.isinf(X_raw.values).sum()
X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f"\n   Inf values replaced : {inf_count:,}")

# Fill NaN with column median
nan_count = X_raw.isna().sum().sum()
X_raw.fillna(X_raw.median(), inplace=True)
print(f"   NaN values filled   : {nan_count:,}")

print(f"\n✅ Feature matrix shape : {X_raw.shape}")
print(f"   Label array shape   : {y_raw.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Cap Majority Classes  (safe to remove if you have enough RAM)
# ─────────────────────────────────────────────────────────────────────────────

if MAX_SAMPLES_PER_CLASS is not None:
    print(f"\nCapping each class to {MAX_SAMPLES_PER_CLASS:,} samples …")
    rng      = np.random.RandomState(RANDOM_STATE)
    keep_idx = []
    for cls in np.unique(y_raw):
        idx = np.where(y_raw == cls)[0]
        if len(idx) > MAX_SAMPLES_PER_CLASS:
            idx = rng.choice(idx, MAX_SAMPLES_PER_CLASS, replace=False)
        keep_idx.extend(idx.tolist())
    keep_idx = np.array(keep_idx)
    X_raw    = X_raw.iloc[keep_idx].reset_index(drop=True)
    y_raw    = y_raw[keep_idx]
    print(f"Dataset size after capping : {len(y_raw):,}")
else:
    print("⚠️  No class cap applied — ensure you have sufficient RAM (≥25 GB)")

print("\nFinal class distribution:")
for cls, cnt in sorted(Counter(y_raw).items()):
    print(f"  {cls:>2}  {ATTACK_CLASSES[cls]:<25} {cnt:>8,}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — Shuffle & Fit StandardScaler → Save scaler.pkl
# ─────────────────────────────────────────────────────────────────────────────

X_arr  = X_raw.values.astype(np.float32)
X_arr, y_raw = shuffle(X_arr, y_raw, random_state=RANDOM_STATE)

print("\nFitting StandardScaler …")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_arr).astype(np.float32)

# ── IMPORTANT: This is the SAME scaler loaded by the ensemble file ──────────
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"✅ scaler.pkl saved → {scaler_path}")

print(f"\n   X_scaled shape    : {X_scaled.shape}")
print(f"   col-0 mean        : {X_scaled[:, 0].mean():.4f}  (should be ≈ 0.0)")
print(f"   col-0 std         : {X_scaled[:, 0].std():.4f}   (should be ≈ 1.0)")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — Build Sliding Window Sequences → X_seq  shape (N, 10, 40)
# ─────────────────────────────────────────────────────────────────────────────
# Window logic:
#   X_seq[i]  = X_scaled[ i*step : i*step + 10 ]   (10 consecutive flows)
#   y_seq[i]  = y_raw[ i*step + 9 ]                (label of last flow)
#
# After building, the first 9 rows have no complete window.
# Cell 12 trims X_single and y to align with X_seq.

print(f"\nBuilding sliding windows  "
      f"(window={SEQUENCE_LENGTH}, step={STEP_SIZE}) …")
print("May take 1–3 minutes for large datasets …")

N             = len(X_scaled)
num_sequences = (N - SEQUENCE_LENGTH) // STEP_SIZE + 1

# Pre-allocate — much faster than repeated np.append
X_seq = np.empty((num_sequences, SEQUENCE_LENGTH, SINGLE_FEATURES), dtype=np.float32)
y_seq = np.empty(num_sequences, dtype=np.int32)

for i in range(num_sequences):
    start    = i * STEP_SIZE
    end      = start + SEQUENCE_LENGTH
    X_seq[i] = X_scaled[start:end]
    y_seq[i] = y_raw[end - 1]           # label = last flow in window

    if i > 0 and i % 200_000 == 0:
        pct = i / num_sequences * 100
        print(f"  {i:>7,} / {num_sequences:,}  ({pct:.1f}%)")

print(f"\n✅ X_seq  shape : {X_seq.shape}")
print(f"   y_seq shape : {y_seq.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 — Align X_single & y to X_seq length
# ─────────────────────────────────────────────────────────────────────────────
# The first (SEQUENCE_LENGTH - 1) = 9 rows produced no complete window.
# We trim X_scaled and y_raw so all three arrays share the same N.
# Result: X_single_aligned[i] == X_seq[i][-1]  (same flow, same values).

offset           = SEQUENCE_LENGTH - 1   # = 9
X_single_aligned = X_scaled[offset : offset + num_sequences]
y_aligned        = y_raw[offset    : offset + num_sequences].astype(np.int32)

# Hard assertions — these MUST pass before saving
assert len(X_single_aligned) == len(X_seq) == len(y_aligned), \
    "❌ Length mismatch after alignment"
assert np.allclose(X_seq[0][-1],   X_single_aligned[0]),   \
    "❌ Alignment error at index 0"
assert np.allclose(X_seq[100][-1], X_single_aligned[100]), \
    "❌ Alignment error at index 100"

print(f"✅ All arrays aligned — {len(y_aligned):,} samples")
print(f"   X_seq[0][-1] == X_single[0] : "
      f"{np.allclose(X_seq[0][-1], X_single_aligned[0])}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 13 — Save X_single.npy, X_seq.npy, y.npy
# ─────────────────────────────────────────────────────────────────────────────

save_map = {
    "X_single.npy" : X_single_aligned,
    "X_seq.npy"    : X_seq,
    "y.npy"        : y_aligned,
}

for fname, arr in save_map.items():
    fpath   = os.path.join(DATA_OUT_DIR, fname)
    np.save(fpath, arr)
    size_mb = os.path.getsize(fpath) / 1e6
    print(f"💾 {fname:<18} shape={str(arr.shape):<22} {size_mb:.1f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 14 — Final Sanity Check (reloads from disk)
# ─────────────────────────────────────────────────────────────────────────────

print("\n🔍 Reloading arrays from disk …")
X_s = np.load(os.path.join(DATA_OUT_DIR, "X_single.npy"), mmap_mode="r")
X_q = np.load(os.path.join(DATA_OUT_DIR, "X_seq.npy"),    mmap_mode="r")
y   = np.load(os.path.join(DATA_OUT_DIR, "y.npy"),        mmap_mode="r")

checks = {
    "X_single  second dim == 40"    : X_s.shape[1] == 40,
    "X_seq     second dim == 10"    : X_q.shape[1] == SEQUENCE_LENGTH,
    "X_seq     third  dim == 40"    : X_q.shape[2] == SINGLE_FEATURES,
    "All arrays same length"        : len(X_s) == len(X_q) == len(y),
    "X_seq[-1] row == X_single row" : np.allclose(X_q[0][-1], X_s[0]),
    "y dtype is integer"            : y.dtype in [np.int32, np.int64],
    "y labels in range 0–14"        : int(y.max()) <= 14 and int(y.min()) >= 0,
    "No NaN in X_single (sample)"   : not np.isnan(X_s[:1000]).any(),
    "No NaN in X_seq (sample)"      : not np.isnan(X_q[:100]).any(),
    "scaler.pkl exists"             : os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl")),
    "label_encoder.pkl exists"      : os.path.exists(os.path.join(MODEL_DIR, "label_encoder.pkl")),
    "model_cfg.json exists"         : os.path.exists(os.path.join(MODEL_DIR, "model_cfg.json")),
}

all_passed = True
for description, result in checks.items():
    icon = "✅" if result else "❌"
    if not result:
        all_passed = False
    print(f"  {icon}  {description}")

print()
if all_passed:
    print("🎉 ALL CHECKS PASSED")
    print("   You can now run neuralsoc_xgboost_ensemble.py without changes.")
else:
    print("⚠️  Some checks FAILED — fix the issues above before proceeding.")

# ── Print exact paths to paste into ensemble file ───────────────────────────
print("\n" + "="*62)
print("  COPY THESE INTO neuralsoc_xgboost_ensemble.py  →  CELL 2")
print("="*62)
print(f'  MODEL_DIR = "{MODEL_DIR}"')
print(f'  DATA_DIR  = "{DATA_OUT_DIR}"')
print()
print("  Verify these match:")
print(f'  CNN_PATH         = MODEL_DIR + "/cnn_model.h5"')
print(f'  TRANSFORMER_PATH = MODEL_DIR + "/model_transformer.keras"')
print(f'  AUTOENCODER_PATH = MODEL_DIR + "/autoencoder_best.h5"')
print(f'  SCALER_PATH      = MODEL_DIR + "/scaler.pkl"')
print()
print("  BiLSTM standalone block paths:")
print(f'  model_cfg.json       = MODEL_DIR + "/model_cfg.json"')
print(f'  scaler.pkl           = MODEL_DIR + "/scaler.pkl"')
print(f'  label_encoder.pkl    = MODEL_DIR + "/label_encoder.pkl"')
print(f'  bilstm weights       = MODEL_DIR + "/bilstm_cicids2017.weights.h5"')
print("="*62)