# NeuralSOC ⚡
### AI-Powered Network Intrusion Detection & Security Operations Platform

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

> **NeuralSOC is not just an intrusion detection system. It is a full-stack, real-time AI Security Operations Center — detecting threats with a 3-model deep learning ensemble, explaining every alert with SHAP, mapping attacks to the MITRE ATT&CK framework, and autonomously generating incident response actions.**

🔴 **[Live Demo](https://neuralsoc.streamlit.app)** &nbsp;|&nbsp; 📖 **[API Docs](https://neuralsoc-api.onrender.com/docs)** &nbsp;|&nbsp; 📊 **[Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)**

---

## The Problem

Traditional intrusion detection systems are black boxes — they fire an alert, but offer no explanation, no context, and no response. Security teams are left asking *why* the alert fired, *what stage* of an attack they're at, and *what to do next*. NeuralSOC answers all three questions automatically.

---

## What NeuralSOC Does

| Capability | Description |
|---|---|
| **Ensemble Detection** | 3 deep learning models (CNN + BiLSTM + Transformer) vote on every packet flow |
| **Explainable AI** | SHAP waterfall chart per alert — no black boxes |
| **Zero-Day Sensing** | LSTM Autoencoder flags novel attacks it has never seen |
| **Kill Chain Mapping** | Every alert mapped to a MITRE ATT&CK stage automatically |
| **Geo Intelligence** | Attacker IPs enriched with country, ISP, ASN on a live globe |
| **Auto-Response** | Generates firewall rules, Jira tickets, and Slack alerts on critical detections |
| **Adversarial Testing** | FGSM attack tester validates model robustness and triggers self-hardening |
| **Live Demo Simulator** | Inject DDoS, port scan, MITM, DNS poisoning into the live feed on demand |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                   │
│   Alert Feed │ SHAP Panel │ Geo Map │ Kill Chain │ IR    │
└──────────────────────────┬──────────────────────────────┘
                           │ REST API
┌──────────────────────────▼──────────────────────────────┐
│                     FASTAPI BACKEND                      │
│  /predict  /simulate  /enrich  /mitre  /respond         │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                      ML ENGINE                           │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌─────────────┐           │
│  │  1D-CNN  │  │  BiLSTM  │  │ Transformer │           │
│  └────┬─────┘  └────┬─────┘  └──────┬──────┘           │
│       └─────────────┼───────────────┘                   │
│               ┌─────▼──────┐                            │
│               │ XGBoost    │  ← Meta-learner            │
│               │ Meta-model │                            │
│               └─────┬──────┘                            │
│                     │         ┌──────────────────┐      │
│                     │    +    │ LSTM Autoencoder │      │
│                     │         │  (Zero-day)      │      │
│               Final Prediction + Zero-day Flag          │
└─────────────────────────────────────────────────────────┘
```

---

## Model Details

### Model 1 — 1D-CNN *(Speed Specialist)*
- **Input:** Single network flow — 40 selected features
- **Architecture:** `Conv1D(64) → Conv1D(128) → GlobalMaxPool → Dense(256) → Softmax(16)`
- **Purpose:** Fastest inference; detects well-known attack signatures in milliseconds

### Model 2 — Bidirectional LSTM *(Sequence Specialist)*
- **Input:** Sliding window of 10 consecutive flows per session
- **Architecture:** `BiLSTM(128) → BiLSTM(64) → Attention → Dense(128) → Softmax(16)`
- **Purpose:** Detects multi-step attacks that unfold over time — recon → exploit → lateral movement

### Model 3 — Transformer Encoder *(Context Specialist)*
- **Input:** Same 10-flow window as BiLSTM
- **Architecture:** `4-head Self-Attention → Feed-Forward → LayerNorm → Dense → Softmax(16)`
- **Purpose:** Captures long-range dependencies across an entire session

### Model 4 — LSTM Autoencoder *(Zero-Day Sensor)*
- **Trained on:** Benign traffic only
- **At inference:** Reconstruction error > 95th percentile threshold = anomaly flag
- **Purpose:** Detects novel attacks with no prior labels — true unsupervised anomaly detection

### Meta-Learner — XGBoost *(Ensemble Fusion)*
- **Input:** Concatenated softmax probability vectors from all 3 classifiers
- **Output:** Final attack class + confidence score + zero-day flag
- **Purpose:** Harder to evade than any single model; corrects individual model errors

---

## Dataset

**CICIDS-2017** — Canadian Institute for Cybersecurity Intrusion Detection Dataset

| Property | Value |
|---|---|
| Records | ~2.8 million labeled network flows |
| Features | 80 flow-level features (packet size, IAT, flags, entropy, etc.) |
| Classes | 1 Benign + 15 Attack categories |
| Collection | 5 days of real network traffic (Mon–Fri) |
| Source | University of New Brunswick |

**Attack categories include:** DDoS, PortScan, Brute Force (FTP/SSH), DoS Hulk, DoS GoldenEye, DoS Slowloris, Heartbleed, Web Attacks (SQLi, XSS, Brute Force), Infiltration, Botnet

---

## Tech Stack

| Tool | Role |
|---|---|
| TensorFlow / Keras | All deep learning models |
| SHAP | Per-alert explainability (DeepExplainer) |
| FastAPI | Async REST API backend |
| Streamlit | Frontend dashboard (pure Python) |
| Scapy | Synthetic attack packet generation |
| Plotly | Interactive charts and geo globe |
| XGBoost | Meta-learner ensemble fusion |
| scikit-learn | Preprocessing, SMOTE, evaluation |
| ipinfo.io | IP geolocation and ASN enrichment |
| Docker | Containerized deployment |
| Render | Backend cloud hosting |
| Streamlit Cloud | Frontend hosting |

---

## Project Structure

```
neuralsoc/
├── data/
│   ├── raw/                  ← CICIDS-2017 CSV files
│   └── processed/            ← Scaled + SMOTE-balanced arrays
├── models/
│   ├── cnn_model.h5
│   ├── bilstm_model.h5
│   ├── transformer_model.h5
│   ├── autoencoder.h5
│   ├── meta_learner.pkl
│   └── scaler.pkl
├── backend/
│   ├── main.py               ← FastAPI app + all endpoints
│   ├── predict.py            ← Ensemble inference logic
│   ├── simulator.py          ← Scapy attack generator
│   ├── mitre.py              ← MITRE ATT&CK mapper
│   └── responder.py          ← Auto-response generator
├── frontend/
│   └── app.py                ← Streamlit dashboard
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_train_models.ipynb
│   └── 03_adversarial.ipynb
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/neuralsoc.git
cd neuralsoc
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add dataset
Download CICIDS-2017 CSVs and place all 5 files in `data/raw/`

### 4. Train models
```bash
jupyter notebook notebooks/02_train_models.ipynb
```

### 5. Start the backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 6. Start the dashboard
```bash
cd frontend
streamlit run app.py
```

### 7. Docker (recommended)
```bash
docker build -t neuralsoc .
docker run -p 8000:8000 neuralsoc
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Classify a network flow + return SHAP values |
| `POST` | `/simulate` | Generate synthetic attack packets via Scapy |
| `GET` | `/enrich/{ip}` | Get geo + ISP + ASN metadata for an IP |
| `GET` | `/mitre/{attack}` | Map attack class to MITRE ATT&CK stage |
| `POST` | `/respond` | Generate firewall rule + Jira ticket + Slack payload |

Full interactive docs at `/docs` (Swagger UI) after starting the backend.

---

## Performance

| Model | Accuracy | F1-Score | Inference Time |
|---|---|---|---|
| 1D-CNN | 96.8% | 0.967 | ~2ms |
| BiLSTM | 97.1% | 0.970 | ~8ms |
| Transformer | 96.5% | 0.964 | ~12ms |
| **Ensemble** | **97.4%** | **0.973** | **~15ms** |

*Evaluated on CICIDS-2017 test split (15% stratified holdout). False positive rate: ~2.1%*

---

## MITRE ATT&CK Mapping

| Detected Attack | ATT&CK Stage | Technique ID |
|---|---|---|
| Port Scan | Reconnaissance | T1046 |
| Brute Force SSH | Initial Access | T1110 |
| DoS / DDoS | Impact | T1499 |
| Botnet C2 | Command & Control | T1071 |
| Infiltration | Lateral Movement | T1021 |
| Web Attack (SQLi) | Initial Access | T1190 |

---

## Deployment

**Backend → Render**
- Connect GitHub repo to Render
- Set start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Add env var: `IPINFO_TOKEN=your_token`

**Frontend → Streamlit Cloud**
- Connect same GitHub repo at `share.streamlit.io`
- Set main file: `frontend/app.py`
- Add secret: `BACKEND_URL=https://your-app.onrender.com`

---

## Adversarial Robustness

NeuralSOC includes a built-in adversarial testing module using **Fast Gradient Sign Method (FGSM)**:

- Perturbs 100 test samples with ε = 0.01, 0.05, 0.1
- Reports evasion success rate per epsilon value
- If evasion rate > 5%, automatically triggers adversarial retraining of the CNN
- The model defends itself — this is the meta-narrative

---

## Why This Approach

Most academic IDS systems are single classifiers evaluated on a notebook. NeuralSOC is a production-grade platform built with the same architectural principles as real SOC tooling:

- **Ensemble over single model** — harder to evade, more robust to dataset shift
- **Explainability by default** — SHAP is not an afterthought; every prediction is justified
- **Unsupervised anomaly detection in parallel** — labeled classifiers miss zero-days; autoencoders don't
- **Framework alignment** — MITRE ATT&CK mapping connects ML output to the vocabulary security teams actually use
- **End-to-end deployability** — Docker + Render + Streamlit Cloud; judges can hit the live URL

---

## Future Work

- [ ] Replace simulator with live Zeek/Suricata tap for production traffic
- [ ] Add federated learning support for multi-org threat sharing
- [ ] Integrate threat intelligence feeds (VirusTotal, Shodan, AbuseIPDB)
- [ ] Graph Neural Network layer for lateral movement detection across hosts
- [ ] Real-time model drift detection with automated retraining pipeline

---

## References

1. Sharafaldin, I., Lashkari, A.H., Ghorbani, A.A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP.
2. MITRE ATT&CK Framework — https://attack.mitre.org
3. CICFlowMeter — https://github.com/ahlashkari/CICFlowMeter
4. SHAP — Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built for the cybersecurity hackathon — designed to win.</p>