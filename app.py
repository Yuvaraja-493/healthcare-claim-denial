import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Healthcare Claim Denial Prediction", layout="wide")
st.title("🏥 Healthcare Claim Denial Prediction App")

# =======================
# Utility to clean data
# =======================
def clean_data(df):
    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "#" in df.columns:
        df = df.drop(columns=["#"])

    # Clean numeric columns
    for col in ["Payment Amount", "Balance"]:
        if col in df.columns:
            df[col] = df[col].replace("[\$,]", "", regex=True).astype(float)
    return df


# =======================
# Load training data
# =======================
claims_path = os.path.join("data", "claims.csv")

if os.path.exists(claims_path):
    df = pd.read_csv(claims_path, header=1, encoding="latin1")  # use 2nd row as header
    df = clean_data(df)

    st.subheader("📂 Training Data Preview")
    st.dataframe(df.head())

    if "Denial" in df.columns:
        # Prepare features and target
        X = df.drop("Denial", axis=1)
        y = df["Denial"]

        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        st.success("✅ Model trained successfully!")

        # =======================
        # Denial Analysis
        # =======================
        st.subheader("📊 Denial Analysis")
        for col in ["CPT Code", "Insurance Company", "Physician Name"]:
            if col in df.columns:
                denial_rates = (
                    df.groupby(col)["Denial"]
                    .mean()
                    .reset_index()
                    .rename(columns={"Denial": "Denial Rate"})
                )
                st.write(f"**Denial Rate by {col}:**")
                st.dataframe(denial_rates)

        # =======================
        # Prediction Section
        # =======================
        st.subheader("🔮 Predict Denials on New Data")
        new_file = st.file_uploader("Upload new claims CSV or Excel file", type=["csv", "xlsx"])

        if new_file:
            if new_file.name.endswith(".csv"):
                new_df = pd.read_csv(new_file, header=1)
            else:
                new_df = pd.read_excel(new_file, header=1)

            new_df = clean_data(new_df)
            st.subheader("📂 New Claims Data Preview")
            st.dataframe(new_df.head())

            # Predict
            predictions = model.predict(new_df)
            new_df["Predicted Denial"] = predictions

            st.subheader("📊 Prediction Results")
            st.dataframe(new_df)

            # Download
            csv = new_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions as CSV",
                csv,
                "predicted_claims.csv",
                "text/csv",
                key="download-csv",
            )

    else:
        st.error("❌ 'Denial' column not found in data. Please include it in training dataset.")
else:
    st.error("❌ Could not find data/claims.csv. Please check the path.")
