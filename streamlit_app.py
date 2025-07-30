import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle

# Load your data (replace with actual path if needed)
df = pd.read_csv("C:\\Users\\lenovo\\OneDrive\\Desktop\\budget_data.csv")

# Process the data
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
monthly_summary["Outcome"] = (monthly_summary["Savings"] / monthly_summary["Total_Income"]) >= 0.10
monthly_summary["Outcome"] = monthly_summary["Outcome"].map({True: "Stable", False: "Unstable"})

# Prepare features
processed_df = monthly_summary[[
    "Total_Income", "Total_Expenses", "Savings", "Expense_to_Income_Ratio", "Outcome"
]]
le = LabelEncoder()
processed_df["Outcome_encoded"] = le.fit_transform(processed_df["Outcome"])

X = processed_df[["Total_Income", "Total_Expenses", "Savings", "Expense_to_Income_Ratio"]]
y = processed_df["Outcome_encoded"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))

# Save everything
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("feature_names.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model, encoder, and feature names saved!")
