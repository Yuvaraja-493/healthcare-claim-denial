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
    # Skip the first junk row, use 2nd row as header
    df = pd.read_csv(claims_path, encoding="latin1", skiprows=1)

    # Drop unnamed or irrelevant columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "#" in df.columns:
        df = df.drop(columns=["#"])

    # Clean up currency columns
    if "Payment Amount" in df.columns:
        df["Payment Amount"] = (
            df["Payment Amount"].replace(r"[\$,]", "", regex=True).astype(float)
        )
    if "Balance" in df.columns:
        df["Balance"] = (
            df["Balance"].replace(r"[\$,]", "", regex=True).astype(float)
        )

    # Create binary target column from Denial Reason
    if "Denial Reason" in df.columns:
        df["Denial"] = df["Denial Reason"].fillna("").apply(
            lambda x: 1 if str(x).strip() != "" else 0
        )

    st.subheader("📂 Training Data Preview")
    st.dataframe(df.head())

    # ========================
    # Split into features & target
    # ========================
    if "Denial" in df.columns:
        X = df.drop("Denial", axis=1)
        y = df["Denial"]

        # Convert categorical columns to numeric
        X = pd.get_dummies(X)

        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        st.success("✅ Model trained successfully on data/claims.csv")

        # ========================
        # Denial Analysis
        # ========================
        st.subheader("📊 Denial Analysis")

        for col in ["CPT Code", "Insurance Company", "Physician Name"]:
            if col in df.columns:
                st.write(
                    df.groupby(col)["Denial"].mean().reset_index().rename(
                        columns={"Denial": "Denial Rate"}
                    )
                )

        # ========================
        # Prediction Section
        # ========================
        st.subheader("🔮 Predict Denials on New Data")

        new_file = st.file_uploader("Upload new claims CSV file", type=["csv"])

        if new_file:
            new_df = pd.read_csv(new_file, skiprows=1)
            new_df = new_df.loc[:, ~new_df.columns.str.contains("^Unnamed")]
            if "#" in new_df.columns:
                new_df = new_df.drop(columns=["#"])

            if "Payment Amount" in new_df.columns:
                new_df["Payment Amount"] = (
                    new_df["Payment Amount"].replace(r"[\$,]", "", regex=True).astype(float)
                )
            if "Balance" in new_df.columns:
                new_df["Balance"] = (
                    new_df["Balance"].replace(r"[\$,]", "", regex=True).astype(float)
                )

            st.subheader("📂 New Claims Data Preview")
            st.dataframe(new_df.head())

            # Ensure same features as training
            new_df_encoded = pd.get_dummies(new_df)
            new_df_encoded = new_df_encoded.reindex(columns=X.columns, fill_value=0)

            # Predict
            predictions = model.predict(new_df_encoded)
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
        st.error("❌ Could not find target column ('Denial'). Please check dataset.")

else:
    st.error("❌ Could not find data/claims.csv. Please check the path.")
