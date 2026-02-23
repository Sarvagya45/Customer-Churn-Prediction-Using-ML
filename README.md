# 📉 Customer Churn Prediction Using Machine Learning

A machine learning project that predicts whether a telecom customer is likely to churn (i.e., cancel their subscription), enabling proactive retention strategies.

---

## 📌 Project Overview

Customer churn is one of the most critical challenges in the telecom industry. Losing customers is costly — acquiring a new customer can be 5–25x more expensive than retaining an existing one. This project uses the **IBM Telco Customer Churn dataset** to build and evaluate machine learning models that identify at-risk customers before they leave.

---

## 📁 Repository Structure

```
Customer-Churn-Prediction-Using-ML/
│
├── Customer_churn_prediction_using_ML_.ipynb   # Main Jupyter Notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv        # Dataset
├── customer_churn_model.pkl                    # Trained ML model (serialized)
├── encoders.pkl                                # Label encoders (serialized)
└── README.md                                   # Project documentation
```

---

## 📊 Dataset

- **Source:** [IBM Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** 7,043 customers
- **Features:** 21 columns including demographics, account info, and service usage
- **Target Variable:** `Churn` (Yes / No)

### Key Features

| Feature | Description |
|---|---|
| `gender` | Customer gender |
| `SeniorCitizen` | Whether the customer is a senior citizen |
| `tenure` | Number of months with the company |
| `MonthlyCharges` | Monthly billing amount |
| `TotalCharges` | Total amount charged |
| `Contract` | Contract type (Month-to-month, One year, Two year) |
| `PaymentMethod` | Payment method used |
| `InternetService` | Type of internet service |
| `Churn` | Whether the customer churned (**target**) |

---

## 🔧 Workflow

1. **Data Loading & Exploration** — Understanding distributions, missing values, and class imbalance
2. **Data Preprocessing** — Handling missing values, encoding categorical variables, feature scaling
3. **Exploratory Data Analysis (EDA)** — Visualizing churn patterns across features
4. **Model Building** — Training and comparing multiple ML classifiers
5. **Model Evaluation** — Using accuracy, precision, recall, F1-score, and ROC-AUC
6. **Model Export** — Saving the best model and encoders using `pickle`

---

## 🤖 Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting / XGBoost *(if applicable)*

---

## 📈 Evaluation Metrics

Given the class imbalance in churn datasets, the following metrics are prioritized:

- **Accuracy**
- **Precision & Recall**
- **F1-Score**
- **ROC-AUC Score**
- **Confusion Matrix**

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook

```bash
git clone https://github.com/Sarvagya45/Customer-Churn-Prediction-Using-ML.git
cd Customer-Churn-Prediction-Using-ML
jupyter notebook Customer_churn_prediction_using_ML_.ipynb
```

### Use the Pre-trained Model

```python
import pickle
import pandas as pd

# Load model and encoders
with open('customer_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Prepare your input data and predict
# prediction = model.predict(your_data)
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | ML modeling & evaluation |
| Jupyter Notebook | Interactive development |
| Pickle | Model serialization |

---

## 📌 Key Insights

- Customers on **month-to-month contracts** are significantly more likely to churn.
- Higher **monthly charges** correlate with increased churn risk.
- Customers with **shorter tenure** are at greater risk.
- **Electronic check** payment method users show higher churn rates.

---

## 👤 Author

**Sarvagya45**  
[GitHub Profile](https://github.com/Sarvagya45)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
