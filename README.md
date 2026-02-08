# 📞 Telecom Customer Churn Prediction

A machine learning project to predict customer churn for a telecom company using historical usage and demographic data. By analyzing customer attributes and behavior, this project builds a predictive model that identifies customers likely to leave (churn), helping businesses improve retention strategies and reduce revenue loss.

---

## 📌 Table of Contents

* 🔍 [Problem Statement](#-problem-statement)
* 🎯 [Objective](#-objective)
* 📦 [Dataset](#-dataset)
* 🛠️ [Tech Stack](#-tech-stack)
* 🧠 [Methodology](#-methodology)
* 📈 [Model Training & Evaluation](#-model-training--evaluation)
* 🚀 [Usage](#-usage)
* 🧾 [Project Structure](#-project-structure)
* 📊 [Results & Insights](#-results--insights)
* 📌 [Future Work](#-future-work)
* 📍 [Contact](#-contact)

---

## 🔍 Problem Statement

Telecom companies face a major challenge: **customer churn** — when existing customers discontinue services for a competitor. As acquiring new customers is significantly more expensive than retaining existing ones, predicting churn to proactively retain high-risk customers is critical for profitability and strategic decision-making.

---

## 🎯 Objective

Build a robust machine learning model that:

* Predicts whether a customer will churn or not.
* Identifies key factors contributing to churn.
* Supports data-driven customer retention strategies.

---

## 📦 Dataset

The project uses the `Telco_Customer_Churn.csv` dataset, containing customer information such as:

* Customer demographics (gender, senior citizen status, dependents)
* Account details (tenure, contract type, billing method)
* Service subscriptions (internet, tech support, online security)
* Financial details (monthly charges, total charges)
* Target variable: `Churn` (Yes/No)

---

## 🛠️ Tech Stack

This project utilizes:

* 🐍 **Python**
* 📊 **pandas**, **NumPy**
* 📉 **matplotlib**, **seaborn**
* 🤖 **scikit-learn** for ML modeling
* 🔄 **joblib / pickle** for model persistence
* 🧠 Jupyter Notebook for experimentation

---

## 🧠 Methodology

1. **Data Cleaning & Preprocessing**

   * Handle missing values
   * Encode categorical features
   * Scale/Normalize numerical features using `MinMaxScaler` (saved as `minmax_scaler.joblib`)

2. **Exploratory Data Analysis (EDA)**

   * Understand customer distribution by churn
   * Analyze patterns across features like contract type and monthly charges

3. **Model Training**

   * Train classification models
   * Start with baseline models like Logistic Regression
   * Save best model (`logistic_regression.pkl`)

4. **Evaluation**

   * Accuracy, Precision, Recall
   * Confusion matrix and other metrics

---

## 📈 Model Training & Evaluation

The Logistic Regression model is trained to classify customers as either:

* **Churn = Yes**
* **Churn = No**

The trained model and preprocessor are stored as:

* `logistic_regression.pkl` — Trained ML model
* `minmax_scaler.joblib` — Preprocessing scaler

You can evaluate performance on a hold-out test set or cross-validation.

---

## 🚀 Usage

### 💻 Run the Prediction Script

1. Clone the repository:

   ```bash
   git clone https://github.com/ceodaniyal/telecom_customer_churn_prediction.git
   cd telecom_customer_churn_prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the prediction script:

   ```bash
   python main.py
   ```

### 📊 Prediction

Provide customer feature values via the script interface or API endpoint (if integrated) to get churn predictions.

---

## 🧾 Project Structure

```
telecom_customer_churn_prediction/
├── Telco_Customer_Churn.csv        # Churn dataset
├── telecom_customer_churn_prediction.ipynb  # Notebook with EDA & modeling
├── main.py                         # Inference script
├── logistic_regression.pkl         # Saved trained model
├── minmax_scaler.joblib            # Preprocessing scaler
├── pyproject.toml                  # Project metadata / dependencies
├── .gitignore
└── README.md
```

---

## 📊 Results & Insights

Typical insights from this kind of churn prediction (can be updated with your actual results):

* 📈 **Monthly charges**, **Contract type**, and **Tenure** often strongly correlate with churn likelihood.
* 🧑‍🤝‍🧑 Customers with **month-to-month contracts churn more** than those on long-term plans. ([GitHub][1])
* 📉 **Paperless billing customers** tend to show higher churn rates. ([GitHub][1])

---

## 📌 Future Work

Future improvements could include:

* Feature engineering (interaction terms, tenure buckets, etc.)
* Hyperparameter tuning (GridSearch / RandomSearch)
* Ensemble methods like Random Forest / Gradient Boosting
* Handling class imbalance (SMOTE)
* Deployment (Flask/Streamlit app)

---

## 📍 Contact

Have questions or feedback? Reach out:

📧 **Email:** [kdaniyal7865@gmail.com](mailto:kdaniyal7865@gmail.com)


[1]: https://github.com/Pradnya1208/Telecom-Customer-Churn-prediction?utm_source=chatgpt.com "GitHub - Pradnya1208/Telecom-Customer-Churn-prediction: Customers in the telecom industry can choose from a variety of service providers and actively switch from one to the next. With the help of ML classification algorithms, we are going to predict the Churn."
