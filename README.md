# Telco Customer Churn

Predicting customer churn for a telecommunications provider using the IBM Telco Customer Churn dataset.

## Exploratory Data Analysis

**Dataset:** 7,043 rows, 21 columns. One numeric target (`Churn`), three numeric features (`SeniorCitizen`, `tenure`, `MonthlyCharges`), and 17 categorical features. `TotalCharges` is stored as object and requires coercion to numeric before modeling.

**Key findings:**

- **Churn rate ~26%**: The dataset is moderately imbalanced. Models should use class weighting or oversampling (e.g., SMOTE) to avoid majority-class bias.
- **Tenure is the strongest predictor**: Churned customers have a median tenure of ~10 months vs. ~38 months for retained customers -- early-lifecycle customers are the highest-risk segment.
- **Monthly charges correlate with churn**: Churners pay more per month on average, likely reflecting month-to-month contracts and premium add-ons.
- **TotalCharges and tenure are near-redundant**: Correlation ~0.83 -- drop one before training to reduce multicollinearity.
- **Senior citizens churn at a higher rate** despite being only ~16% of customers -- a high-value retention segment.

**Modeling implications:** Prioritize `tenure`, `MonthlyCharges`, and contract type as features. Address class imbalance before training. See `notebooks/eda.ipynb` for full analysis.
