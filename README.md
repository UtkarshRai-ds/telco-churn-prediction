# 🔄 Telco Customer Churn Prediction

> End-to-end machine learning project predicting customer churn for a telecom company.
> Built with a production-grade pipeline: data quality gates, feature engineering,
> model comparison, hyperparameter tuning, and an interactive Streamlit dashboard.

**🚀 Live Demo:** [telco-churn-utkarsh.streamlit.app](https://telco-churn-utkarsh.streamlit.app)
**📁 GitHub:** https://github.com/UtkarshRai-ds/telco-churn-prediction

---

## 📋 Project Overview

| | |
|---|---|
| **Problem** | Identify customers likely to cancel their subscription before they do |
| **End User** | Retention teams who need to prioritize outreach |
| **Dataset** | IBM Telco Customer Churn — 7,043 customers × 21 features |
| **Model Output** | Churn probability (0–1) + risk tier (Low / Medium / High) |
| **Key Design Decision** | Optimized for **Recall (0.80)** over Accuracy — missing a churner costs more than a false alarm |

### Business Impact
Targeting only the **High + Medium risk tiers (51% of customer base) captures 87% of all churners** — the efficiency win that makes retention budgets go further.

---

## 🏗️ Architecture

```
Raw Data (CSV)
     │
     ▼
┌─────────────┐
│  loader.py  │  Load + validate schema
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  quality.py │  5 quality gates (nulls, dupes, schema, types, ranges)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  cleaner.py │  Type coercion, TotalCharges fix, binary encoding
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ engineer.py  │  14 engineered features → 36 total columns
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  baseline.py │     │   train.py  │     │  tuning.py   │
│  (LR untuned)│     │ (3 models)  │     │ (Optuna 30t) │
└──────┬───────┘     └──────┬──────┘     └──────┬───────┘
       └──────────────────┬─┘                    │
                          ▼                       │
                 ┌────────────────┐               │
                 │  MLflow        │◄──────────────┘
                 │  Experiment    │  Track all runs
                 │  Tracking      │
                 └───────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ production_model│  Tuned LR — AUC 0.84, Recall 0.80
                │     .pkl        │
                └───────┬─────────┘
                        │
                        ▼
              ┌──────────────────┐
              │ streamlit_app.py │  4-page interactive dashboard
              └──────────────────┘
```

---

## 📊 Results

### Model Comparison (all models trained on same engineered feature matrix)

| Model | AUC-ROC | Recall | F1 | Precision | Accuracy |
|---|---|---|---|---|---|
| **Tuned Logistic Regression ★** | **0.8382** | **0.7995** | **0.6203** | 0.5060 | 0.7400 |
| Random Forest | 0.8378 | 0.7460 | 0.6193 | 0.5294 | 0.7566 |
| XGBoost | 0.8336 | 0.7647 | 0.6079 | 0.5044 | 0.7381 |
| Logistic Regression (Baseline) | 0.8389 | 0.4920 | 0.5601 | 0.6502 | 0.7949 |

### Key Insight
After equalizing preprocessing across all four models, **AUC converges to ~0.83 for everyone** — the 14 engineered features carry most of the predictive signal. What separates the tuned model is **Recall: 0.80 vs the untuned baseline's 0.49** — a 31 percentage point improvement from class balancing + Optuna tuning. For a retention use case where missing a churner costs more than a false alarm, that gap is what matters.

### Customer Risk Segmentation

| Risk Tier | Customers | Churners Captured | Action |
|---|---|---|---|
| 🔴 High (≥ 0.65) | 416 (30%) | 241/374 | Immediate intervention + senior CSM |
| 🟡 Medium (0.35–0.65) | 305 (22%) | 85/374 | Proactive outreach + retention offer |
| 🟢 Low (< 0.35) | 688 (49%) | 48/374 | Standard service cadence |

**Targeting High + Medium tiers (721 customers, 51% of base) captures 87% of all churners.**

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.12** | Core language |
| **pandas 2.2.2** | Data manipulation |
| **scikit-learn 1.5.1** | ML models, preprocessing, evaluation |
| **XGBoost 3.2.0** | Gradient boosting model |
| **Optuna** | Hyperparameter tuning (30 trials, 5-fold CV) |
| **MLflow** | Experiment tracking |
| **Streamlit 1.37.1** | Interactive dashboard |
| **Plotly 5.24.1** | Interactive charts |
| **joblib 1.4.2** | Model serialization |
| **pytest** | Test suite (34 tests) |
| **Docker** | Containerization |
| **GitHub Actions** | CI/CD pipeline |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.12
- Git
- Docker (optional, for containerized run)

### Clone and install

```bash
git clone https://github.com/UtkarshRai-ds/telco-churn-prediction.git
cd telco-churn-prediction
pip install -r requirements.txt
```

### Run the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

### Run with Docker

```bash
docker build -t my-ml-project .
docker run -p 8501:8501 my-ml-project
```

Open `http://localhost:8501` in your browser.

### Run the full training pipeline

```bash
# Step 1 — Clean the data
python src/data/cleaner.py

# Step 2 — Feature engineering
python src/features/run_features.py

# Step 3 — Train and compare models
python src/models/run_training.py

# Step 4 — Pre-compute evaluation artifacts for dashboard
python src/models/retrain_for_eval.py
python src/models/evaluate_for_app.py
```

### Run the test suite

```bash
pytest tests/ -v
```

34 tests across data quality, feature engineering, and model validation.

---

## 🔬 Feature Engineering

14 features engineered across 3 categories:

| Feature | Category | Rationale |
|---|---|---|
| `customer_lifetime_value` | Domain | tenure × MonthlyCharges — proxy for revenue at risk |
| `contract_flexibility_risk` | Domain | Month-to-month = 1.0, One year = 0.5, Two year = 0.0 |
| `is_long_term_customer` | Domain | tenure > 24 months |
| `total_services` | Domain | Count of active add-on services |
| `high_data_user` | Domain | Streaming TV + Movies subscriber |
| `support_adoption` | Domain | Online Security + Tech Support subscriptions |
| `monthly_spending_trend` | Statistical | MonthlyCharges / (tenure + 1) |
| `relative_monthly_spend` | Statistical | Spend relative to dataset mean |
| `tenure_quartile` | Statistical | Q1–Q4 one-hot encoded |
| `charge_consistency` | Statistical | Ratio of monthly to total charges |
| `senior_internet_risk` | Interaction | Senior citizen with fiber optic |
| `month_to_month_new_customer` | Interaction | Month-to-month contract + tenure < 12 |
| `long_tenure_low_engagement` | Interaction | Long-term customer with few services |
| `contract_spend_risk` | Interaction | Month-to-month × MonthlyCharges |

---

## 💡 Key Decisions & Lessons

- **Chose Recall over Accuracy as the primary metric** — with 26% churn rate, a model that predicts "no churn" for everyone gets 74% accuracy. Recall is the metric that matters for retention teams.

- **Caught a preprocessing inconsistency during model comparison** — original XGBoost and Random Forest were trained on raw categorical columns while the tuned LR used engineered features. After equalizing inputs, all models converged to ~0.83 AUC, showing the feature engineering was doing the heavy lifting.

- **Pre-computed evaluation artifacts instead of loading models in the dashboard** — avoids sklearn version mismatch issues on deployment and keeps cold starts fast. The dashboard reads from `data/eval_artifacts.json` (~122 KB) rather than three separate `.pkl` files.

- **Baseline LR shows a ConvergenceWarning** — deliberately not fixed. The untuned baseline uses no feature scaling and no class balancing; the warning is part of what the baseline represents. Fixing it would obscure the gap that tuning closes.

- **One failure worth documenting** — `run_features.py` has a pandas join issue on fresh environments (columns overlap error with tenure_quartile dummies). Worked around by committing `cleaned.csv` to the repo for CI. This is a known technical debt item.

---

## 📁 File Structure

```
telco-churn-prediction/
├── app/
│   └── streamlit_app.py          # 4-page interactive dashboard (~1,068 lines)
├── data/
│   ├── Telco_Customer_Churn.csv  # Raw dataset (7,043 rows × 21 columns)
│   ├── cleaned.csv               # Post-cleaning dataset
│   ├── features.csv              # 36-column engineered feature matrix
│   ├── model_results.json        # Model metrics for dashboard
│   └── eval_artifacts.json       # Pre-computed ROC/PR curves + probabilities
├── models/
│   └── production_model.pkl      # Tuned Logistic Regression (AUC 0.84)
├── notebooks/
│   └── eda.ipynb                 # Exploratory data analysis (7 sections)
├── src/
│   ├── data/
│   │   ├── loader.py             # CSV loader + schema validation
│   │   ├── quality.py            # 5 data quality gates
│   │   └── cleaner.py            # Data cleaning pipeline
│   ├── features/
│   │   ├── engineer.py           # Feature creation + selection functions
│   │   └── run_features.py       # Feature pipeline orchestrator
│   └── models/
│       ├── baseline.py           # Untuned LR baseline
│       ├── train.py              # 3-model comparison
│       ├── tuning.py             # Optuna hyperparameter search
│       ├── run_training.py       # MLflow tracking pipeline
│       ├── predict.py            # Batch prediction script
│       ├── retrain_for_eval.py   # Retrain models on standardized features
│       └── evaluate_for_app.py   # Pre-compute dashboard evaluation artifacts
├── tests/
│   ├── test_data_quality.py      # 9 tests — quality gate validation
│   ├── test_features.py          # 13 tests — feature engineering validation
│   └── test_model.py             # 12 tests — model load + prediction validation
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions — test + lint on every push
├── Dockerfile                    # python:3.12-slim, exposes 8501
├── docker-compose.yml            # Local containerized run with volume mounts
├── requirements.txt              # Pinned production dependencies
└── README.md                     # This file
```
