import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Personal Finance Tracker", layout="wide")

# Load model and resources with caching
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, label_encoder, feature_names

model, le, feature_names = load_model()

# Title
st.title("📊 Personal Finance Prediction App")
st.write("This app predicts whether your personal finance outcome is likely to be 'On Track' or 'Off Track' based on your income, expenses, and more.")

# Sidebar
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Choose Mode", ["Single Prediction", "Batch Prediction"])

# Helper function for preprocessing
def preprocess_input(data, feature_names):
    # Ensure all required columns are present
    missing_cols = [col for col in feature_names if col not in data.columns]
    for col in missing_cols:
        data[col] = 0  # default value if missing

    # Reorder columns
    return data[feature_names]

# Single Prediction Mode
if mode == "Single Prediction":
    st.header("🔍 Single Prediction")

    income = st.number_input("Monthly Income", min_value=0.0, step=100.0)
    expenses = st.number_input("Monthly Expenses", min_value=0.0, step=100.0)
    savings = st.number_input("Savings", min_value=0.0, step=50.0)
    debt = st.selectbox("Has Debt?", ["Yes", "No"])
    owns_asset = st.selectbox("Owns Asset?", ["Yes", "No"])

    if st.button("Predict"):
        input_df = pd.DataFrame({
            "Monthly_Income": [income],
            "Monthly_Expenses": [expenses],
            "Savings": [savings],
            "Has_Debt": [1 if debt == "Yes" else 0],
            "Owns_Asset": [1 if owns_asset == "Yes" else 0]
        })

        input_processed = preprocess_input(input_df, feature_names)
        pred = model.predict(input_processed)
        prob = model.predict_proba(input_processed)[0]

        label = le.inverse_transform(pred)[0]
        st.success(f"💡 Prediction: **{label}**")
        st.info(f"🧮 Probability - On Track: {round(prob[1]*100, 2)}% | Off Track: {round(prob[0]*100, 2)}%")

# Batch Prediction Mode
elif mode == "Batch Prediction":
    st.header("📁 Batch Prediction")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.subheader("📄 Uploaded Data")
            st.dataframe(data.head())

            data_processed = preprocess_input(data.copy(), feature_names)
            preds = model.predict(data_processed)
            probas = model.predict_proba(data_processed)

            data["Prediction"] = le.inverse_transform(preds)
            data["Prob_OnTrack"] = probas[:, 1]
            data["Prob_OffTrack"] = probas[:, 0]

            st.subheader("✅ Predictions")
            st.dataframe(data)

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

            # Optional Visuals
            st.subheader("📊 Prediction Distribution")
            fig1, ax1 = plt.subplots()
            sns.countplot(x="Prediction", data=data, ax=ax1, palette="Set2")
            st.pyplot(fig1)

            st.subheader("📈 Income vs Expenses")
            fig2, ax2 = plt.subplots()
            sns.scatterplot(x="Monthly_Income", y="Monthly_Expenses", hue="Prediction", data=data, palette="Set1", ax=ax2)
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
