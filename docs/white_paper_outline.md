# White Paper Outline

## Sensor-Agnostic Machine Learning for Indoor Air Quality Prediction: Architecture, Pipeline Engineering, and Comparative Model Evaluation

---

### Abstract
- Problem: indoor IAQ prediction from low-cost sensors, challenges of real-world deployment
- Contribution: sensor/standard-agnostic ML platform, training pipeline for gapped temporal data, comparative evaluation of 5 architectures on real BME688 data, cross-sensor transfer learning from BME688 to BME680
- Key finding: (pending training results)

---

### 1. Introduction

#### 1.1 The Indoor Air Quality Problem
- Health impact of indoor air pollution
- Low-cost sensor proliferation (BME680/688, SPS30, SGP40)
- Gap between raw sensor output and actionable IAQ scores

#### 1.2 Limitations of Existing Approaches
- BSEC as a black box — proprietary, closed-source, no fine-tuning
- Sensor-specific ML models that don't generalize
- Lab-trained models that fail on real deployment data

#### 1.3 Contributions
- Sensor and standard-agnostic architecture
- Training pipeline engineered for real-world temporal sensor data
- Comparative evaluation: MLP, KAN, CNN, LSTM, BNN on 434k real readings
- Cross-sensor transfer learning: BME688 (labeled) → BME680 (unlabeled, 18M readings)
- Open dataset and MCP server for reproducibility

---

### 2. Background and Related Work

#### 2.1 Indoor Air Quality Standards
- BSEC IAQ index, EPA AQI, WHO guidelines
- Why multiple standards matter for different deployments

#### 2.2 Machine Learning for Air Quality
- Prior work on AQ prediction (mostly outdoor, mostly single-model)
- Gap: few indoor studies, fewer with real long-duration data

#### 2.3 Kolmogorov-Arnold Networks
- KAN as an alternative to MLP — theoretical motivation
- Limited real-world evaluation in IoT/sensor domains

#### 2.4 Bayesian Neural Networks for Sensor Data
- Uncertainty quantification in safety-relevant predictions
- BNN for latent variable inference (occupancy, ventilation)

---

### 3. System Architecture

#### 3.1 Design Principles
- Sensor agnosticism: `SensorProfile` abstraction
- Standard agnosticism: `IAQStandard` abstraction
- Feature engineering as code, not configuration

#### 3.2 Sensor Profile Abstraction
- Raw features, engineered features, valid ranges, quality columns
- Physical quantity registry with unit conversion
- Example: BME680Profile — 4 raw → 10 engineered features (including cyclical temporal)

#### 3.3 Model Architectures
- MLP: baseline feedforward, flattened window input
- KAN: B-spline activation functions, same input shape as MLP
- CNN: 1D convolutions over temporal window, learns local patterns
- LSTM: bidirectional recurrent, captures temporal dynamics
- BNN: variational weight layers, calibrated uncertainty via MC sampling

#### 3.4 Inference Engine
- Sliding window buffering
- Staleness detection (gap > 60s)
- MC dropout / weight sampling for uncertainty
- Bayesian conjugate updates from external prior variables

---

### 4. Training Pipeline

#### 4.1 Pipeline Architecture
- FSM with 8 stages: Source Access → Ingestion → Feature Engineering → Windowing → Splitting → Scaling → Training → Evaluation → Saving
- Structured StageResult and PreprocessingReport per run

#### 4.2 Data Ingestion and Quality Filtering
- Multiple data sources: InfluxDB, CSV, Label Studio, synthetic
- BME688 `iaq_accuracy` quality gate (≥ 2)
- Field mapping for heterogeneous sensor schemas

#### 4.3 Temporal Feature Engineering
- Cyclical encoding of hour-of-day and day-of-week (sin/cos pairs)
- VOC ratio (resistance / baseline)
- Absolute humidity from temperature and relative humidity
- Profile-driven: each sensor defines its own engineering

#### 4.4 The Multi-Segment Windowing Problem
- Real sensor data has frequent gaps (power loss, connectivity, restarts)
- Naive approach: use only the longest contiguous segment — discards 93% of data
- Our approach: window each valid segment independently, concatenate
- Impact: 31k → 433k training windows, R² from -6.48 to +0.20

#### 4.5 Chronological Splitting
- No shuffling — prevents temporal data leakage
- 80/20 chronological split preserves real-world evaluation conditions

#### 4.6 Per-Model Window Sizes
- Flatten-based models (MLP, KAN, BNN): window=10 — temporal order irrelevant
- Temporal models (LSTM: window=60, CNN: window=30) — need meaningful context
- Window size as a model-specific hyperparameter, not a global setting

---

### 5. Data Provenance and Artifact Management

#### 5.1 Merkle Tree Provenance
- 6-level chain: Sensor → RawData → CleansedData → PreprocessedData → SplitData → TrainedModel
- Root hash stored alongside model artifacts
- Verification: `python -m iaq4j verify`

#### 5.2 Semantic Versioning for Model Artifacts
- Format: `{model_type}-{MAJOR}.{MINOR}.{PATCH}`
- MAJOR: schema change (features, window size, sensor type)
- MINOR: retrain with new data or metrics improvement
- PATCH: metadata-only change
- Schema fingerprint: SHA256 of (sensor_type, iaq_standard, window_size, num_features, model_type)

---

### 6. Experimental Setup

#### 6.1 Datasets

##### 6.1.1 Primary Dataset: BME688 + BSEC (Arduino R4)
- Source: BME688 sensor via BSEC library, logged to InfluxDB
- Location: residential indoor environment (study room)
- Duration: ~56 days continuous monitoring at 3-second intervals
- Raw: 1.19M readings; after quality filtering (iaq_accuracy ≥ 2): 434k readings
- 72 contiguous segments (71 gaps), 62 segments long enough for windowing
- Fields: temperature, rel_humidity, pressure, gas_resistance, iaq, iaq_accuracy, static_iaq, gas_percentage

##### 6.1.2 Secondary Dataset: BME680 Raw (Raspberry Pi)
- Source: BME680 sensor via direct I2C on Raspberry Pi, no BSEC
- Location: same residential environment
- Duration: ~3 years (Feb 2023 — Feb 2026)
- 18.1M readings at 5-second intervals
- Fields: temperature, humidity, gas (VOC resistance), pressure
- Particulate data (pm2_5, pm10) available for ~5M rows (co-located SPS30)
- No BSEC IAQ ground truth — raw sensor values only

##### 6.1.3 Tertiary Dataset: ESP8266
- 51k cleaned readings with device and location tags
- Same physical quantities, different hardware platform

#### 6.2 Features
- Raw (4): temperature, relative humidity, pressure, VOC resistance
- Engineered (6): VOC ratio, absolute humidity, hour_sin, hour_cos, dow_sin, dow_cos
- Total: 10 features per timestep

#### 6.3 Target
- BSEC IAQ index (0–500 scale)
- Distribution: mean=100.3, std=43.4, range 46.5–500.0

#### 6.4 Training Configuration
- 200 epochs, batch size 32, Adam optimizer
- ReduceLROnPlateau (patience=10, factor=0.5)
- Gradient clipping (max_norm=1.0)
- Hardware: Apple M4 (MPS backend)

---

### 7. Results

#### 7.1 Model Comparison
- Table: MAE, RMSE, R² for all 5 models
- (Pending completion of training run)

#### 7.2 Training Dynamics
- Convergence speed per architecture
- Validation loss curves
- Training time comparison

#### 7.3 Analysis
- Which architectures benefit from temporal context (CNN, LSTM) vs flattened input (MLP, KAN, BNN)?
- Does BNN uncertainty correlate with prediction error?
- Does KAN's theoretical expressiveness translate to better IAQ prediction?

#### 7.4 The Multi-Segment Effect
- Before: single segment, R²=-6.48 (MLP)
- After: all segments, R²=+0.20 (MLP)
- Lesson: data utilization matters more than model architecture

#### 7.5 Cross-Sensor Transfer Learning

##### 7.5.1 Problem Statement
- Source domain: BME688 + BSEC (434k labeled readings, 2 months)
- Target domain: BME680 raw (18M unlabeled readings, 3 years)
- Same sensor family (Bosch MOX), different hardware generations
- Shared features: temperature, humidity, pressure transfer directly
- Open question: VOC resistance relationship (BME688 multi-step heater vs BME680 single-step)

##### 7.5.2 Sensor Differences

| Property | BME680 (Pi) | BME688 (R4) |
|----------|-------------|-------------|
| Gas sensing mode | Single heater step | 10-step heater scan |
| BSEC integration | None — raw resistance | Full — provides IAQ score |
| Calibration | Uncalibrated | BSEC auto-calibration |
| VOC resistance field | `gas` | `gas_resistance` |
| Data volume | 18.1M (3 years) | 1.19M (2 months) |

##### 7.5.3 Approach 1: Direct Transfer (Baseline)
- Train on BME688 data, predict on BME680
- Evaluate degradation: how much does cross-sensor shift affect MAE/R²?
- Expected: temperature/humidity/pressure features transfer well, VOC resistance breaks

##### 7.5.4 Approach 2: Feature-Aligned Transfer
- Map BME680 `gas` to BME688 `gas_resistance` using overlapping time period (if any)
- Learn a calibration function between the two resistance scales
- Re-predict with aligned features

##### 7.5.5 Approach 3: Semi-Supervised Pretraining
- Pretrain encoder on 18M unlabeled BME680 readings (autoencoder or contrastive learning)
- Fine-tune on 434k labeled BME688 readings
- Evaluate whether 3 years of unlabeled temporal patterns improves IAQ prediction
- Hypothesis: diurnal/seasonal patterns learned from BME680 transfer to IAQ prediction

##### 7.5.6 Evaluation
- No direct ground truth for BME680 IAQ — evaluation strategies:
  - Temporal consistency: do predictions follow expected diurnal/seasonal patterns?
  - Correlation with VOC resistance: does predicted IAQ track gas resistance inversely?
  - Cross-validation on BME688: does pretraining on BME680 improve BME688 R²?

---

### 8. MCP Server: LLM-Driven Sensor Analysis

#### 8.1 Motivation
- Gap between ML model output and actionable insight
- Natural language interface for non-expert users

#### 8.2 Architecture
- Dual-protocol: FastAPI for sensors, MCP for LLM clients
- MCP tools map to core functions (predict, train, query history, evaluate)
- MCP resources expose config, manifest, quantities, profiles

#### 8.3 Use Cases
- "Why did IAQ spike at 3am?" — LLM queries InfluxDB, correlates features
- "Train a model on this CSV" — LLM orchestrates the full pipeline
- "Compare model performance" — LLM reads manifests, presents analysis

---

### 9. Discussion

#### 9.1 Practical Lessons
- Real sensor data is messy — gaps, drift, quality variation
- Pipeline engineering (windowing, splitting, feature engineering) has more impact than model selection
- Sensor-agnostic abstractions pay off when onboarding new hardware

#### 9.2 Cross-Sensor Insights
- Which features transfer across sensor generations?
- Does the VOC resistance relationship require explicit calibration or can the model learn it?
- Value of large unlabeled temporal datasets for pretraining

#### 9.3 Limitations
- Single indoor environment — generalization to other spaces untested
- BSEC IAQ as ground truth is itself an approximation
- No occupancy or ventilation data (privileged information)
- Cross-sensor transfer limited to same manufacturer (Bosch MOX family)

#### 9.4 Future Work
- LLM-driven pipeline design (feature engineering + model selection)
- Multi-sensor fusion (BME688 + SPS30 particulate data)
- Online learning / model drift detection in production
- Cross-manufacturer transfer (Bosch → Sensirion)

---

### 10. Conclusion
- Summary of contributions
- Key takeaway: pipeline engineering and data utilization matter more than model architecture for real-world IoT ML
- Cross-sensor transfer unlocks value from legacy unlabeled data
- MCP as a distribution mechanism for IoT ML platforms

---

### References

---

### Appendix A: Dataset Description and Access
- Schema, column descriptions, statistics, download link / Hugging Face
- BME688 dataset: 434k labeled readings
- BME680 dataset: 18.1M unlabeled readings (3 years)

### Appendix B: Hardware Stack
- Arduino R4 + BME688 + BSEC → Node-RED → InfluxDB Cloud
- Raspberry Pi + BME680 (I2C) → Docker InfluxDB (LAN)
- ESP8266 + BME680 → same pipeline
- Network topology diagram

### Appendix C: Reproducibility
- `python -m iaq4j train --data-source sample --model all`
- MCP server setup for Claude Desktop
- Merkle verification of published model artifacts
