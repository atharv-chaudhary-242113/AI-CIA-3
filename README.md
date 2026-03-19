# Meta-Cognitive AI — Cognitive Tunneling Detection
### v1.1 | Cognitive Psychology & Artificial Intelligence

---

## Overview

This project implements an AI system designed to predict and assess cognitive performance risk based on lifestyle and behavioral factors. The core idea is rooted in **Cognitive Tunneling** — a psychological phenomenon where a person's attention narrows so intensely under stress that they lose situational awareness, leading to performance breakdowns in high-stakes environments such as surgical theaters, air traffic control, or intense study sessions.

The system uses an XGBoost model to:
- **Predict** a continuous Cognitive Score from lifestyle telemetry (regression)
- **Classify** individuals into Low, Medium, or High cognitive risk levels (classification)
- **Explain** which factors drive each prediction using SHAP (Shapley Additive explanations)

This project aligns with **SDG 3** (Good Health & Well-being) by providing a data-driven framework for preventing cognitive burnout, and **SDG 9** (Industry, Innovation & Infrastructure) by demonstrating how AI can improve human-machine interaction safety.

---

## Project Structure
```
cognitive_ai/
├── data/
│   ├── raw/
│   │   └── raw_data.csv                  # Original Kaggle dataset (80,000 samples)
│   └── processed/
│       └── processed_data.csv            # Feature-engineered, labelled dataset
├── models/
│   ├── xgb_regressor.ubj                 # Trained XGBoost Regressor
│   ├── xgb_classifier.ubj                # Trained XGBoost Classifier
│   ├── preprocessor.pkl                  # Fitted StandardScaler + OneHotEncoder
│   └── label_encoder.pkl                 # Fitted LabelEncoder (Low/Medium/High)
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb        # EDA, feature engineering, preprocessing
│   └── 02_modeling_evaluation.ipynb      # HPO, training, evaluation, SHAP
├── reports/                              # All generated plots saved here
├── config.py                             # Central path and constant definitions
└── requirements.txt
```

---

## Dataset

**Source:** [Human Cognitive Performance Analysis — Kaggle](https://www.kaggle.com/datasets/samharison/human-cognitive-performance-analysis)  
**License:** CC0 Public Domain  
**Size:** 80,000 samples × 13 columns

| Feature | Type | Description |
|---|---|---|
| Age | Numerical | Age of individual |
| Sleep_Duration | Numerical | Hours of sleep per night |
| Stress_Level | Numerical | Self-reported stress (1–10) |
| Daily_Screen_Time | Numerical | Hours on screens per day |
| Caffeine_Intake | Numerical | mg of caffeine per day |
| Reaction_Time | Numerical | Response time in milliseconds |
| Memory_Test_Score | Numerical | Score out of 100 |
| Gender | Categorical | Male / Female / Other |
| Diet_Type | Categorical | Vegetarian / Non-Vegetarian / Vegan |
| Exercise_Frequency | Categorical | Low / Medium / High |
| Cognitive_Score | Target | Weighted composite cognitive performance score |

> **Note:** `AI_Predicted_Score` is excluded from features to prevent data leakage.

---

## Engineered Features

Three interaction features are derived from the cognitive psychology rationale of this project:

| Feature | Formula | Rationale |
|---|---|---|
| Stress_Sleep_Interaction | Stress × Sleep | Core antagonistic relationship driving cognitive tunneling |
| Caffeine_Sleep_Ratio | Caffeine ÷ (Sleep + 1) | High values indicate stimulant dependency without recovery |
| Fatigue_Load | Stress + Screen Time − Sleep | Composite risk indicator for attentional depletion |

---

## Risk Label Assignment

Cognitive risk tiers are assigned using **data-driven tertile thresholds** rather than arbitrary fixed cutoffs:

| Risk Level | Threshold |
|---|---|
| Low | Cognitive Score ≤ 33rd percentile |
| Medium | 33rd < Cognitive Score ≤ 67th percentile |
| High | Cognitive Score > 67th percentile |

---

## Methodology

### Preprocessing
- Numerical features scaled with `StandardScaler`
- Categorical features encoded with `OneHotEncoder` (handle_unknown='ignore')
- Pipeline fitted on training set only, applied to test set to prevent leakage

### Hyperparameter Optimisation
- **Library:** Optuna (TPE Sampler — Bayesian optimisation)
- **Trials:** 50
- **CV:** 3-Fold KFold on 40% training subsample (RAM-efficient)
- **Metric:** RMSE (minimised)
- Final models retrained on full training set using best found parameters

### Models
- **XGBoost Regressor** — predicts continuous Cognitive Score
- **XGBoost Classifier** — predicts Low / Medium / High risk level (warm-started from regressor's best params)
- `tree_method='hist'` used for memory efficiency on CPU

### Explainability
- **SHAP TreeExplainer** (exact, not approximated) applied to the regressor
- Outputs: Beeswarm (global), Bar (ranked importance), Waterfall (single prediction)

---

## Results (v1.1)

| Metric | Value |
|---|---|
| RMSE | ~0.63 |
| R² | 0.999 |

> The near-perfect R² is expected — the dataset is synthetically generated with `Cognitive_Score` derived from a weighted formula of the input features. The model has effectively reverse-engineered this formula, which validates its ability to learn the weighted relationship between lifestyle factors and cognition. Real-world performance on organic data would differ.

---

## Setup

**Requirements:** Python ≥ 3.11, uv
```bash
# Clone and enter the project
cd AI-CIA-3

# Install all dependencies
uv add -r requirements.txt

# Launch Jupyter
jupyter notebook
```

Then run the notebooks in order:
1. `01_eda_preprocessing.ipynb`
2. `02_modeling_evaluation.ipynb`

---

## Configuration

All paths and constants are defined in `config.py` at the project root. Edit this file if your directory layout differs — no other files need to be changed.
```python
RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_TRIALS     = 50
CV_FOLDS     = 3
```

---

## SDG Alignment

| Goal | Connection |
|---|---|
| SDG 3 — Good Health & Well-being | Early detection of cognitive fatigue to prevent burnout in high-pressure occupations |
| SDG 9 — Industry, Innovation & Infrastructure | AI-assisted human-machine interaction safety through real-time cognitive load monitoring |