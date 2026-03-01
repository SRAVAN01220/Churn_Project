import streamlit as st
import pandas as pd
import joblib

st.title("📘 Customer Churn Prediction")

# Load model and features
model = joblib.load("churn_model.pkl")
feature_cols = joblib.load("model_features.pkl")

st.header("Customer Input")

tenure = st.number_input("Tenure", min_value=0)
monthly = st.number_input("Monthly Charges", min_value=0.0)
total = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict Churn Risk"):

    input_df = pd.DataFrame(columns=feature_cols)
    input_df.loc[0] = 0

    if "tenure" in feature_cols:
        input_df["tenure"] = tenure

    if "MonthlyCharges" in feature_cols:
        input_df["MonthlyCharges"] = monthly

    if "TotalCharges" in feature_cols:
        input_df["TotalCharges"] = total

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    st.write(f"Churn Probability: {probability*100:.2f}%")

    if probability > 0.7:
        st.error("🔴 High Risk Customer")
    elif probability > 0.4:
        st.warning("🟠 Medium Risk Customer")
    else:
        st.success("🟢 Low Risk Customer")