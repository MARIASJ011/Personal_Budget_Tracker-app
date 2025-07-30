import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and resources
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, le, features

model, le, feature_names = load_model()

# App Title
st.title("📊 Personal Finance Stability Predictor")

st.write("""
This app predicts whether your monthly financial situation is **Stable** or **Unstable** 
based on your income, expenses, and savings.
""")

# -------------------------------
# Single Prediction Input Section
# -------------------------------
st.header("🔍 Single Prediction")

with st.form("single_pred_form"):
    total_income = st.number_input("Total Monthly Income", min_value=0.0, step=100.0)
    total_expenses = st.number_input("Total Monthly Expenses", min_value=0.0, step=100.0)
    savings = total_income - total_expenses
    ratio = total_expenses / total_income if total_income > 0 else 0.0

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([[
            total_income, total_expenses, savings, ratio
        ]], columns=feature_names)

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][prediction]

        result = le.inverse_transform([prediction])[0]
        st.success(f"**Prediction: {result}**")
        st.info(f"Confidence: {prob:.2%}")

# -------------------------------
# Batch Prediction Section
# -------------------------------
st.header("📁 Batch Prediction")

uploaded_file = st.file_uploader("Upload a CSV file with 'Date', 'Type', 'Amount'", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df["Date"] = pd.to_datetime(df["Date"])
        monthly_summary = (
            df.groupby([df["Date"].dt.to_period("M"), "Type"])["Amount"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
            .rename(columns={"Income": "Total_Income", "Expense": "Total_Expenses"})
        )
        monthly_summary["Savings"] = monthly_summary["Total_Income"] - monthly_summary["Total_Expenses"]
        monthly_summary["Expense_to_Income_Ratio"] = (
            monthly_summary["Total_Expenses"] / monthly_summary["Total_Income"].replace(0, np.nan)
        ).fillna(0)

        input_data = monthly_summary[feature_names]
        predictions = model.predict(input_data)
        prediction_labels = le.inverse_transform(predictions)
        monthly_summary["Prediction"] = prediction_labels

        st.subheader("📃 Prediction Results")
        st.dataframe(monthly_summary)

        # Download CSV
        csv_download = monthly_summary.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv_download, file_name="predictions.csv")

        # Visualization
        st.subheader("📊 Prediction Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Prediction", data=monthly_summary, palette="Set2", ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
