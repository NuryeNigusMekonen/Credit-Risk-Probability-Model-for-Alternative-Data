###  Credit Scoring Business Understanding

---
## basel II
lets think Basel II as a global safety manual for banks. It says:

1, “If you give out loans, you must have a system to estimate how risky those loans are.”

2, “You must keep enough backup capital in case things go wrong.”

3, “And you must document and disclose what you're doing.”
## why does it matter in this credit score project
Because you are modeling creditworthiness, model must:
1, Be interpretable and documented (to satisfy Pillar 3)
2, Measure risk probabilities reliably (for Pillar 1)
3, Be auditable by supervisors (for Pillar 2)
#### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord requires financial institutions to evaluate and report credit risk using structured, transparent, and validated internal models. For regulatory approval, these models must be **interpretable**, **auditable**, and **based on sound statistical principles**.

This means that credit scoring models cannot be treated as black boxes. Instead, they must:

* Provide **explanations** for their predictions (e.g., why a customer was denied credit).
* Be **well-documented**, with clear logic, variable transformations, and assumptions.
* Allow **auditors and compliance officers** to trace predictions to inputs.

In this context, simpler models like **logistic regression with Weight of Evidence (WoE)** encoding are often preferred over complex models, as they are easier to explain and align with Basel II expectations.

---

#### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In the given dataset, we do not have a direct label for whether a customer **defaulted on a loan**. To train a predictive model, we must define a **proxy target variable** that captures similar behavior—such as identifying disengaged customers who **frequently make purchases but stop repaying over time**.

Creating this proxy is essential, but it introduces potential risks:

* **Label Leakage**: The proxy may unintentionally capture future information or irrelevant patterns.
* **Misalignment**: Disengagement does not always equate to financial default (e.g., a customer may pause activity but still be creditworthy).
* **Bias and Overfitting**: If the proxy is poorly defined, the model may generalize poorly or produce unfair results.

Thus, while proxy variables enable modeling, their design must be guided by **business logic**, **statistical rigor**, and **risk awareness**.

---

# lets go business report for Credit Risk Probability Model for Alternative Data


##  Background

This project is designed for micro-lending institutions and digital banks targeting financially underserved customers. Traditional credit scoring models rely on historical loan data, which is often unavailable for new or informal economy users. As a result, financial institutions need alternative models that leverage non-traditional features, like transaction behavior and telecom-based metadata.

---

##  Problem Statement

Financial institutions face difficulty in evaluating the creditworthiness of new or unbanked customers. Without a risk scoring mechanism based on alternative data, these institutions either over-lend (leading to high defaults) or under-lend (missing growth opportunities). This project aims to create a probability-based credit risk model using customer behavioral data to improve credit decision-making.

---

##  Objective

To build a robust, production-ready credit risk probability scoring model using alternative customer data, and deploy it via a containerized API for real-time integration.

---

##  Approach & Methodology

- **Tools & Libraries:** `pandas`, `scikit-learn`, `xgboost`, `mlflow`, `joblib`, `fastapi`, `docker`, `GitHub Actions`
- **Steps:**
  - Data Cleaning & Feature Engineering (RFM metrics, behavior traits)
  - Model Training (Logistic Regression, Random Forest, XGBoost)
  - Performance Evaluation (Precision, Recall, F1, ROC AUC)
  - MLflow Experiment Tracking & Model Registry
  - FastAPI Deployment with Docker
  - CI/CD with GitHub Actions for linting & testing

---

##  Proposed Solution

### a) Preprocessing

- Cleaned missing values, handled class imbalance.
- Derived aggregate features like `Amount_mean`, `txn_hour_nunique`, etc.
- Used `ColumnTransformer` with:
  - `StandardScaler` for numeric features
  - `OneHotEncoder` for categorical features
- Pipeline-ready transformation for deployment

### b) Experimentation & Modeling

Three classifiers were tested using `GridSearchCV` and 5-fold cross-validation:

| Model              | F1 Score | ROC AUC | Best Hyperparameters                        |
|-------------------|----------|---------|---------------------------------------------|
| LogisticRegression| 0.72     | 0.80    | C = 0.1                                      |
| RandomForest       | 0.84     | 0.93    | n_estimators = 200, max_depth = 10          |
| XGBoost            | **0.88** | **0.95**| learning_rate = 0.1, max_depth = 5, n_estimators = 200 |

- The best model (XGBoost) was logged to MLflow and registered as `xgboost_final`.
- Trained models were saved as `.pkl` in `/models` and also to the MLflow registry for production use.

---

##  Evaluation

- Final selected model (XGBoost) achieved:
  - **Precision:** 0.86
  - **Recall:** 0.91
  - **F1 Score:** 0.88
  - **ROC AUC:** 0.95
- Model was validated with real-world-like customer profiles to assess generalizability.

---

##  Conclusion & Recommendations

- This end-to-end system provides an accurate, explainable, and production-ready credit risk scoring solution based on behavioral data.
- Banks and micro-lenders can integrate this into onboarding workflows to improve credit approvals.
- **Future Work:**
  - Add explainability tools (e.g., SHAP, LIME)
  - Integrate telecom and social data sources
  - Include time-series-based risk trends

---

##  References

- [scikit-learn documentation](https://scikit-learn.org/)
- [XGBoost documentation](https://xgboost.readthedocs.io/)
- [MLflow documentation](https://mlflow.org/)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Docker documentation](https://docs.docker.com/)

---

##  Annex

###  API Test Example

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "Amount_sum": 10000.0,
    "Amount_mean": 10000.0,
    "Amount_std": 6558.96,
    "Amount_count": 20,
    "txn_year_nunique": 1,
    "txn_month_nunique": 2,
    "txn_dayofweek_nunique": 5,
    "txn_hour_nunique": 1,
    "ProviderId": "ProviderId_4",
    "ProductCategory": "airtime",
    "ChannelId": "ChannelId_2",
    "PricingStrategy": "2"
}'
````

Response:

```json
{
  "risk_probability": 0.5090070880498481
}
```

---

###  Docker Deployment

```bash
# Build image
docker build -t credit-risk-api .

# Run container
docker run -d -p 8000:8000 credit-risk-api
```

###  CI/CD Workflow (GitHub Actions)

```yaml
name: CI Pipeline

on: [push]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10.18'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Lint with flake8
        run: |
          flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
      - name: Run tests
        run: |
          pytest
```

---

>  **Maintained by:** Nurye Nigus
