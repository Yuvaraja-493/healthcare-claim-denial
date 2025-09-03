import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

st.set_page_config(page_title="Healthcare Claim Denial Prediction App", layout="wide")

st.title("🏥 Healthcare Claim Denial Prediction App")

# File path
claims_path = "data/claims.csv"

# Check if file exists
if not os.path.exists(claims_path):
    st.error("❌ claims.csv not found in `data/` folder.")
else:
    # Load dataset
    df = pd.read_csv(claims_path, encoding="latin1")

    # --- Auto-cleaning step (handles your case with '#' column & blank rows) ---
    df = df.dropna(how="all")  # drop fully empty rows
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]  # drop unnamed cols
    if "#" in df.columns:
        df = df.drop(columns=["#"])  # drop '#' column if exists

    st.subheader("📂 Training Data Preview")
    st.write(df.head())

    # Ensure target column exists
    if "Denial Reason" not in df.columns:
        st.error("❌ Could not find target column ('Denial Reason'). Please check dataset.")
    else:
        # Create target variable: 1 if denial reason exists, else 0
        df["Denial"] = df["Denial Reason"].apply(lambda x: 0 if pd.isna(x) or x == "" else 1)

        # Features & Target
        X = df.drop(columns=["Denial", "Denial Reason"])
        y = df["Denial"]

        # Encode categorical columns
        X = pd.get_dummies(X, drop_first=True)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Results
        st.subheader("📊 Model Performance")
        st.write("✅ Accuracy:", accuracy_score(y_test, y_pred))
        st.text(classification_report(y_test, y_pred))

        # --- User Prediction ---
        st.subheader("🧾 Predict Denial for New Claim")

        with st.form("prediction_form"):
            cpt = st.text_input("CPT Code", "99213")
            insurance = st.text_input("Insurance Company", "Medicare")
            physician = st.text_input("Physician Name", "Dr. Smith")
            payment = st.number_input("Payment Amount", min_value=0.0, value=0.0)
            balance = st.number_input("Balance", min_value=0.0, value=100.0)
            submitted = st.form_submit_button("Predict")

        if submitted:
            new_claim = pd.DataFrame([{
                "CPT Code": cpt,
                "Insurance Company": insurance,
                "Physician Name": physician,
                "Payment Amount": payment,
                "Balance": balance
            }])

            # One-hot encode new claim to match training
            new_claim = pd.get_dummies(new_claim)
            new_claim = new_claim.reindex(columns=X.columns, fill_value=0)

            pred = model.predict(new_claim)[0]
            if pred == 1:
                st.error("❌ Claim likely to be DENIED")
            else:
                st.success("✅ Claim likely to be APPROVED")
