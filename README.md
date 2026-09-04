# Credit Card Approval Risk Prediction

## Problem

Predicting which credit card applicants are likely to become high-risk (60+ days delinquent) based on application data — a classification problem relevant to credit risk assessment in banking and fintech.

## Why this framing

This model predicts risk **among applicants who were already approved and have a credit history** (an inner join between application and credit record data), not whether an application gets approved in the first place. This is a meaningful distinction: the model answers "given this applicant was approved, will they become high-risk?" rather than "should this applicant be approved?" This framing choice is documented here for transparency.

## Dataset

- Source: [Credit Card Approval Prediction (Kaggle)](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)
- ~36,457 applicants after merging application and credit history data
- Severe class imbalance: **~1.7% of applicants are labeled high-risk**
- Target constructed from credit history: applicants who ever reached 60+ days overdue (STATUS 2-5) are labeled high-risk

## Approach

1. **Data understanding & target construction** — merged application and credit history data, defined the high-risk target, audited features for potential data leakage
2. **Exploratory data analysis** — examined feature distributions and their relationship to risk, identified data quality issues (placeholder values in employment duration, negative-day date encodings)
3. **Feature engineering** — converted raw date encodings to interpretable age/tenure, engineered income-per-person and employment-ratio features, encoded categorical variables
4. **Baseline modeling** — Logistic Regression and Decision Tree with class weighting
5. **Model improvement** — Random Forest and XGBoost with class weighting, SMOTE oversampling as an alternative imbalance strategy, and threshold tuning based on the precision/recall tradeoff
6. **Explainability & error analysis** — SHAP values to understand global and individual predictions, comparison of correctly identified vs. missed high-risk cases

## Key Results

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.560 |
| Decision Tree | 0.562 |
| Random Forest | 0.690 |
| **XGBoost** | **0.725** |
| XGBoost + SMOTE | 0.690 |

**XGBoost with class weighting was the best-performing model**, notably outperforming the SMOTE variant — suggesting that for this dataset, reweighting the loss function generalized better than synthetic oversampling.

At a tuned decision threshold (selected via precision-recall tradeoff analysis rather than the default 0.5), the final model achieves:

| Metric | Class 0 (Low Risk) | Class 1 (High Risk) |
|---|---|---|
| Precision | 0.99 | 0.10 |
| Recall | 0.92 | 0.50 |
| F1-score | 0.95 | 0.16 |

This represents roughly a **5x improvement in minority-class precision** over the untuned baseline, while catching half of all high-risk applicants — a deliberate tradeoff favoring recall, on the reasoning that missing a high-risk applicant is costlier than flagging a safe one for manual review.

## Key Insights (SHAP Analysis)

- **Age, income-per-person, employment ratio, years employed, and total income** are the most influential features in the model
- **Employment stability matters**: more years employed consistently pushes predictions toward lower risk, while low employment ratio pushes toward higher risk
- **Income shows a counter-intuitive pattern**: higher income-per-person and total income lean toward *increasing* predicted risk rather than decreasing it. Case-level analysis suggests this is partly driven by pensioners, who often show high per-person income (small household size) alongside genuine repayment risk — a pattern the model has picked up on
- **Being a pensioner is one of the single strongest risk indicators** in the model, with a sharp, consistent positive SHAP value

### Error Analysis

Comparing a correctly identified high-risk applicant against a missed one showed that the model relies on a plausible but imperfect heuristic: "older, non-employed, but with steady income/assets = safe." This heuristic holds often enough to be useful but fails for individual pensioners who look financially stable on paper yet are genuinely high-risk — a limitation worth accounting for in any real-world deployment.

## What I'd Do Differently / Next Steps

- Explore cost-sensitive threshold selection using estimated business costs of false positives vs. false negatives, rather than a threshold chosen from precision-recall curves alone
- Test additional imbalance techniques (e.g., ADASYN, cost-sensitive learning) given that SMOTE underperformed plain class weighting here
- Investigate the income/risk relationship further — it likely reflects an interaction effect (e.g., income type × income level) rather than a simple linear relationship, and a deeper look could refine feature engineering
- With more data, revisit the inner-join framing to see if approval prediction itself could be modeled with additional data sources

## How to Run

```bash
# Clone the repo
git clone https://github.com/<your-username>/credit-approval-prediction.git
cd credit-approval-prediction

# Set up environment
python -m venv ds_env
source ds_env/bin/activate  # Windows: ds_env\Scripts\activate
pip install -r requirements.txt

# Download the dataset from Kaggle and place application_record.csv 
# and credit_record.csv into data/raw/

# Run notebooks in order (01 through 06)
jupyter notebook
```

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP, matplotlib, seaborn