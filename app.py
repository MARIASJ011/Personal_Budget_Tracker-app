import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource
def load_model():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

model, features = load_model()

st.set_page_config(page_title="Personal Budget Tracker", layout="centered")
st.title("💰 Personal Budget Tracker")
st.markdown("Predict whether your spending habits indicate a healthy or overspending pattern.")

option = st.sidebar.radio("Choose input method:", ("Single Prediction", "Batch Prediction"))

def make_prediction(input_df):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][prediction]
    label = "Overspending" if prediction == 1 else "Healthy Budget"
    return label, prob

if option == "Single Prediction":
    st.subheader("Single Entry Prediction")
    with st.form("prediction_form"):
        income = st.number_input("Monthly Income", min_value=0, value=50000)
        monthly_expenses = st.number_input("Monthly Expenses", min_value=0, value=25000)
        savings = st.number_input("Savings", min_value=0, value=10000)
        debt = st.number_input("Debt", min_value=0, value=5000)
        age = st.slider("Age", 18, 65, 30)
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([[income, monthly_expenses, savings, debt, age]], columns=features)
        label, prob = make_prediction(input_data)
        st.success(f"Prediction: {label} ({prob*100:.2f}% confidence)")

elif option == "Batch Prediction":
    st.subheader("Batch Prediction via CSV Upload")
    uploaded_file = st.file_uploader("Upload CSV file with appropriate columns", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            missing_cols = [col for col in features if col not in data.columns]
            for col in missing_cols:
                data[col] = 0

            data = data[features]
            predictions = model.predict(data)
            probs = model.predict_proba(data)

            data["Prediction"] = ["Overspending" if p == 1 else "Healthy Budget" for p in predictions]
            data["Confidence"] = probs.max(axis=1)

            st.write("## Predictions")
            st.dataframe(data)

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

            st.write("## Visualization")
            fig, ax = plt.subplots()
            sns.countplot(x="Prediction", data=data, ax=ax)
            st.pyplot(fig)

            st.write("### Feature Distribution")
            fig2 = sns.pairplot(data[features + ["Prediction"]], hue="Prediction")
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"Error processing file: {e}")
