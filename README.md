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

# Credit Risk Probability Model for Alternative Data

A full machine learning pipeline for identifying high-risk customers using transactional behavior. This solution leverages alternative data and is built for micro-lending or financial institutions seeking robust, explainable, and scalable credit risk prediction using MLOps best practices.


##  Background

This project is developed for financial institutions operating in low-documentation or underbanked environments, particularly micro-lenders in emerging markets. The system uses **alternative data sources** (e.g., mobile transactions) to assess credit risk where traditional credit history is unavailable.

---

##❗ Problem Statement

Banks and microfinance institutions often struggle to evaluate creditworthiness due to limited or no access to traditional credit history. This increases default risk, reduces lending efficiency, and locks out deserving customers. A scalable system is needed to identify **"high-risk" borrowers** using **transactional alternative data**.

---

##  Objective

To build a machine learning system that predicts:
- `risk_probability`: likelihood of default
- `is_high_risk`: binary classification flag for intervention

---

## Approach / Methodology

### Tools & Technologies:
- **Python, Pandas, Scikit-Learn, XGBoost**
- **MLflow** for experiment tracking and model registry
- **FastAPI** for API deployment
- **Docker** and **GitHub Actions** for CI/CD
- **Flake8**, **Pytest** for quality checks

### Workflow:
1. Cleaned and engineered features from raw transaction data
2. Engineered proxy target (`is_high_risk`) using RFM-like metrics
3. Trained and evaluated multiple models (Logistic Regression, Random Forest, XGBoost)
4. Logged and tracked experiments using MLflow
5. Containerized API service using Docker + FastAPI
6. Implemented automated CI/CD with GitHub Actions

---

##  Proposed Solution

### a) Data Preprocessing

- **Missing Values**: Imputed using `SimpleImputer` (mean for numeric, mode for categorical)
- **Scaling**: Applied `StandardScaler` to numeric features
- **Encoding**: Used `OneHotEncoder` for categorical fields
- **Features Used**:
    - `Amount_sum`, `Amount_mean`, `Amount_std`, `Amount_count`
    - `txn_year_nunique`, `txn_month_nunique`, `txn_dayofweek_nunique`, `txn_hour_nunique`
    - `ProviderId`, `ProductCategory`, `ChannelId`, `PricingStrategy`

- **Proxy Target Engineering**:
    - Created `is_high_risk` label by clustering RFM metrics
    - High-risk customers identified as disengaged or anomalous

### b) Modeling & Experimentation

| Model               | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 66.9%    | 0.54      | 0.87   | 0.668    | 0.768   |
| Random Forest       | **73.7%**| **0.61**  | **0.85**| **0.712**| **0.821**|
| XGBoost             | 72.5%    | 0.59      | 0.88   | 0.711    | 0.821   |

- **Best Model**:  `Random Forest`  
- **Model Registered**: `random_forest_final` via MLflow Registry  
- **Saved Model**: `random_forest_model.pkl`

---

##  Evaluation

- The **Random Forest model** achieved:
    - **F1 Score**: 0.712
    - **ROC AUC**: 0.821
- Performance was consistent across test sets and robust to class imbalance.
- Deployed model returns both:
    ```json
    {
      "risk_probability": 0.51,
      "is_high_risk": true
    }
    ```

---

##  Deployment & CI/CD (Task 6)

###  FastAPI REST API
- `/predict` endpoint
- Accepts input via Pydantic schema
- Returns prediction from best registered model

###  Dockerized Service
- **Dockerfile** builds the API service
- Port `8000` exposed
- Built with:
    ```bash
    docker build -t credit-risk-api .
    docker run -d -p 8000:8000 credit-risk-api
    ```

###  GitHub Actions CI
- Workflow defined in `.github/workflows/ci.yml`
- Runs on every `push` to `main`
- Steps:
  -  Run `flake8` linter for style
  -  Run `pytest` for test coverage
- Ensures code quality and stability

---

##  Conclusion & Recommendation

This solution allows financial institutions to:
- Reliably assess **credit risk** using **alternative data**
- Automate the entire ML workflow from data to deployment
- Detect risky profiles **before issuing loans**
- Adapt easily to new transactional data

###  Future Work
- Integrate with real-time streaming data
- Add explainability (e.g., SHAP) for model transparency
- Extend to multi-tier risk segmentation

---

##  References

- [MLflow Docs](https://mlflow.org/)
- [Scikit-Learn](https://scikit-learn.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [GitHub Actions](https://docs.github.com/en/actions)

---

##  Annex

###  Example API Request (via `curl`)
```bash
curl -X POST http://localhost:8000/predict \
-H 'Content-Type: application/json' \
-d '{
  "Amount_sum": 10000,
  "Amount_mean": 10000,
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

###  Example Response

```json
{
  "risk_probability": 0.509,
  "is_high_risk": true
}

---

>  **Maintained by:** Nurye Nigus
