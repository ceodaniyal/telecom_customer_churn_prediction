import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Load model and scaler
# -----------------------------------
model = joblib.load("logistic_regression.pkl")
scaler = joblib.load("minmax_scaler.joblib")

# -----------------------------------
# App Title
# -----------------------------------
st.set_page_config(page_title="Telecom Churn Prediction", layout="centered")
st.title("📞 Telecom Customer Churn Prediction")

st.write("Fill in customer details to predict churn")

# -----------------------------------
# User Inputs
# -----------------------------------

senior_citizen = st.radio(
    "Is customer a Senior Citizen?",
    ["Yes", "No"]
)

tenure = st.slider(
    "Customer's Tenure (in months)",
    min_value=1,
    max_value=72,
    value=32
)

total_charges = st.slider(
    "Customer's Total Charges",
    min_value=18.8,
    max_value=8684.8,
    value=2290.0
)

import streamlit as st

# Using more engaging questions for the labels
partner = st.radio(
    "Does customer have a partner?",
    ["Yes", "No"]
)

dependents = st.radio(
    "Does customer have any dependents (children/seniors)?",
    ["Yes", "No"]
)

internet_service = st.radio(
    "What type of internet service does customer use?",
    ['DSL', 'Fiber optic', 'No']
)

online_security = st.radio(
    "Does customer use our Online Security service?",
    ["No", "No internet service", "Yes"]
)

online_backup = st.radio(
    "DDoes customer subscribe to our Online Backup service?",
    ["No", "No internet service", "Yes"]
)

device_protection = st.radio(
    "Is Device Protection included in customer's plan?",
    ["No", "No internet service", "Yes"]
)

tech_support = st.radio(
    "Does customer have access to Premium Tech Support?",
    ["No", "No internet service", "Yes"]
)

contract = st.radio(
    "What is customer's current contract term?",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.radio(
    "Would customer prefer Paperless Billing?",
    ["Yes", "No"]
)

payment_method = st.radio(
    "Which payment method does customer prefer to use?",
    ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
       'Credit card (automatic)']
)

# -----------------------------------
# Feature Engineering
# -----------------------------------

# Create empty dataframe with training columns
input_data = pd.DataFrame(
    columns=['SeniorCitizen', 'tenure', 'TotalCharges', 'Partner_Yes',
       'Dependents_Yes', 'InternetService_Fiber optic', 'InternetService_No',
       'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
       'OnlineBackup_No internet service', 'OnlineBackup_Yes',
       'DeviceProtection_No internet service', 'DeviceProtection_Yes',
       'TechSupport_No internet service', 'TechSupport_Yes',
       'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']
)

# Initialize all values to 0
input_data.loc[0] = 0

# Senior Citizen
if dependents == "Yes":
    input_data.at[0, 'SeniorCitizen'] = 1

# Fill numerical features
input_data.at[0, 'tenure'] = tenure
input_data.at[0, 'TotalCharges'] = total_charges

# Dependents
if dependents == "Yes":
    input_data.at[0, 'Dependents_Yes'] = 1

# Internet Service
if internet_service == "Fiber optic":
    input_data.at[0, 'InternetService_Fiber optic'] = 1
elif internet_service == "No":
    input_data.at[0, 'InternetService_No'] = 1

# Online Security
if online_security == "No internet service":
    input_data.at[0, 'OnlineSecurity_No internet service'] = 1
elif online_security == "Yes":
    input_data.at[0, 'OnlineSecurity_Yes'] = 1

# Online Backup
if online_backup == "No internet service":
    input_data.at[0, 'OnlineBackup_No internet service'] = 1
elif online_backup == "Yes":
    input_data.at[0, 'OnlineBackup_Yes'] = 1

# Device Protection
if device_protection == "No internet service":
    input_data.at[0, 'DeviceProtection_No internet service'] = 1
elif device_protection == "Yes":
    input_data.at[0, 'DeviceProtection_Yes'] = 1

# Tech Support
if tech_support == "No internet service":
    input_data.at[0, 'TechSupport_No internet service'] = 1
elif tech_support == "Yes":
    input_data.at[0, 'TechSupport_Yes'] = 1

# Contract
if contract == "One year":
    input_data.at[0, 'Contract_One year'] = 1
elif contract == "Two year":
    input_data.at[0, 'Contract_Two year'] = 1

# Paperless Billing
if paperless_billing == "Yes":
    input_data.at[0, 'PaperlessBilling_Yes'] = 1

# Payment Method
if payment_method == 'Electronic check':
    input_data.at[0, 'PaymentMethod_Electronic check'] = 1
elif payment_method == 'Mailed check':
    input_data.at[0, 'PaymentMethod_Mailed check'] = 1
elif payment_method == 'Credit card (automatic)':
    input_data.at[0, 'PaymentMethod_Credit card (automatic)'] = 1

# -----------------------------------
# Scaling
# -----------------------------------
scaled_input = scaler.transform(input_data)

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("🔍 Predict Churn"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer is NOT likely to churn (Probability: {probability:.2f})")

    st.subheader("Model Input (After Encoding)")
    st.dataframe(input_data)
