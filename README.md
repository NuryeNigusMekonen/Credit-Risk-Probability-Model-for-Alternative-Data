##  Final Output for Task 1

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

#### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

| Model Type                                      | Advantages                                                                         | Disadvantages                                                                    |
| ----------------------------------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Logistic Regression + WoE**                   | High interpretability, compliant with regulations, easy to explain to stakeholders | May underperform with nonlinear or complex patterns                              |
| **Gradient Boosting (e.g., XGBoost, LightGBM)** | High predictive power, captures complex relationships, better accuracy             | Difficult to explain, lower transparency, harder to deploy in regulated settings |

In regulated environments, **interpretability often outweighs raw performance**. A balanced strategy might involve:

* Using **logistic regression** for production deployment.
* Supplementing with **GBM models** during development to benchmark or explore insights.

