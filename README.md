# 📌 Term Deposit Marketing – Modeling & Analysis

## Project Overview

This project builds a machine learning system to predict whether a customer will subscribe to a term deposit product, based on historical marketing campaign data from a European banking institution.

The objectives are:

* Achieve ≥ 81% accuracy using 5-fold cross-validation.
* Identify customer segments most likely to subscribe.
* Determine the key features driving subscription decisions.
* Provide a model suitable for marketing targeting and ranking.

---

# 📊 Dataset Description

The dataset contains 40,000 customer records from direct marketing campaigns.

### Key Feature Groups:

* **Demographics:** age, job, marital, education
* **Financial status:** balance, default, housing loan, personal loan
* **Campaign info:** contact type, day, month, duration, campaign count
* **Target variable:** `y` (subscribed: yes/no)

Class imbalance:

* 92.76% "no"
* 7.24% "yes"

---

# 🧠 Modeling Strategy

We approached the problem in two ways:

## 1️⃣ Post-Call Outcome Prediction (Including Duration)

Includes all features.

This predicts whether a customer will subscribe *after a call has already started*.

## 2️⃣ Pre-Call Customer Targeting (Excluding Duration)

Excludes `duration` (since call duration is unknown before calling).

This is the realistic marketing targeting model.

---

# 📈 Evaluation Approach

Given heavy class imbalance, we evaluated models using:

* Accuracy
* Precision
* Recall
* F1 Score
* Gains Chart
* Lift (Top Decile Capture Rate)

We used **5-fold Stratified Cross-Validation** for robustness.

Marketing-relevant metric:

> Top 10% capture rate (Lift)

---

# 🏆 Model Comparison Summary

## A. WITH Duration (Post-Call Prediction)

| Model                | Test Accuracy | Test F1 | Top 10% Buyers Captured |
| -------------------- | ------------- | ------- | ----------------------- |
| Logistic Regression  | ~86.8%        | ~0.49   | 64.5%                   |
| Decision Tree        | ~87.4%        | ~0.50   | 66.6%                   |
| HistGradientBoosting | ~93.9%        | ~0.51   | **77.3%**               |

🔎 **Key Insight:**
Call duration is the strongest predictor of subscription.

---

## B. WITHOUT Duration (Real Targeting Model)

| Model                | Test Accuracy | Test F1 | Top 10% Buyers Captured |
| -------------------- | ------------- | ------- | ----------------------- |
| Logistic Regression  | ~66.6%        | ~0.21   | 31.6%                   |
| Decision Tree        | ~61.7%        | ~0.19   | 28.8%                   |
| HistGradientBoosting | ~92.7%        | ~0.07*  | **44.1%**               |

(*Low recall due to conservative threshold; ranking performance is strong.)

🔎 **Key Insight:**
Gradient Boosting significantly outperforms linear models in ranking customers.

---

# 📊 Gains Analysis


<table>
<tr>
<td align="center">
<b>All Columns</b><br>
<img src="data/gains01.png" width="400"/>
</td>

<td align="center">
<b>No Duration</b><br>
<img src="data/gains02.png" width="400"/>
</td>

<td align="center">
<b>No Duration & Month</b><br>
<img src="data/gains03.png" width="400"/>
</td>
</tr>
</table>

### With Duration:

* Top 10% captures 77% of buyers.
* Top 20% captures ~97%.

### Without Duration:

* Top 10% captures 44% of buyers.
* Top 20% captures ~60%.

This demonstrates strong targeting capability.

---

# 🔍 Feature Importance Insights

Using permutation importance:

## Most Influential Features (Without Duration)

* Month (seasonality effect)
* Campaign count
* Balance / balance bin
* Contact type
* Marital status
* Loan / housing status

## With Duration

* Duration dominates prediction.
* Followed by month and contact variables.

---

# 🎯 Business Recommendations

## 1️⃣ Targeting Strategy (Pre-Call)

Use Gradient Boosting model (without duration) to:

* Rank customers by predicted probability.
* Target top 10–20% highest ranked customers.
* Expect 3–6× improvement over random targeting.

## 2️⃣ Campaign Optimization

Customers more likely to subscribe:

* Contacted during specific months (seasonality effect).
* With higher account balances.
* With fewer previous campaign contacts.
* Without active housing or personal loans.

---

# 📦 Repository Structure

```
├── data/ # due to company policy, the dataset is not available for public consumption.
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── utils.py
├── README.md
```

EDA and experimentation are documented in notebooks.

Final production training and inference code will be implemented under `src/`.

---