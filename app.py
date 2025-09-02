import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Healthcare Claim Denial Prediction", layout="wide")
st.title("🏥 Healthcare Claim Denial Prediction App")

# ========================
# Load Training Data
# ========================
claims_path = os.path.join("data", "claims.csv")

if os.path.exists(claims_path):
    df = pd.read_csv(claims_path, encoding='latin1')


    st.subheader("📂 Training Data Preview")
    st.dataframe(df.head())

    # Split into features and target
    X = df.drop("Denial", axis=1)
    y = df["Denial"]

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    st.success("✅ Model trained successfully on data/claims.csv")

    # ========================
    # Denial Analysis
    # ========================
    st.subheader("📊 Denial Analysis")
    st.write(
        df.groupby("CPT").Denial.mean().reset_index().rename(columns={"Denial": "Denial Rate"})
    )
    st.write(
        df.groupby("Payer").Denial.mean().reset_index().rename(columns={"Denial": "Denial Rate"})
    )
    st.write(
        df.groupby("Provider").Denial.mean().reset_index().rename(columns={"Denial": "Denial Rate"})
    )

    # ========================
    # Prediction Section
    # ========================
    st.subheader("🔮 Predict Denials on New Data")

    new_file = st.file_uploader("Upload new claims CSV file", type=["csv"])

    if new_file:
        new_df = pd.read_csv(new_file)

        st.subheader("📂 New Claims Data Preview")
        st.dataframe(new_df.head())

        # Predict
        predictions = model.predict(new_df)
        new_df["Predicted Denial"] = predictions

        st.subheader("📊 Prediction Results")
        st.dataframe(new_df)

        # Option to download results
        csv = new_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predictions as CSV",
            csv,
            "predicted_claims.csv",
            "text/csv",
            key="download-csv",
        )

else:
    st.error("❌ Could not find data/claims.csv. Please check the path.")
