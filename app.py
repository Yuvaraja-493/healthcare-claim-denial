import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Healthcare Claim Denial Prediction App", layout="wide")

st.title("🏥 Healthcare Claim Denial Prediction App")
st.write("Upload a CSV with CPT codes, insurance info, payments, balances, and denial reasons.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # 🔥 Auto-clean CSV
    df = pd.read_csv(uploaded_file, skip_blank_lines=True)
    df = df.dropna(how="all")  # drop fully blank rows
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # drop unnamed index columns

    # Rename columns if they have extra spaces
    df.columns = df.columns.str.strip()

    # Clean currency fields
    for col in ["Payment Amount", "Balance"]:
        if col in df.columns:
            df[col] = df[col].replace(r"[\$,]", "", regex=True).astype(float)

    # Create Denial column
    if "Denial" not in df.columns and "Denial Reason" in df.columns:
        df["Denial"] = df["Denial Reason"].apply(
            lambda x: 1 if pd.notnull(x) and str(x).strip() != "" else 0
        )

    # ✅ Show preview
    st.subheader("📂 Raw Data Preview")
    st.dataframe(df.head())

    if "Denial" not in df.columns:
        st.error("❌ No 'Denial' or 'Denial Reason' column found in the file!")
    else:
        st.success("✅ Data loaded successfully!")

        # 📊 Top CPT Codes
        denial_summary = df.groupby("CPT Code")["Denial"].sum().sort_values(ascending=False)

        st.subheader("📊 Top CPT Codes by Denial Frequency")
        st.dataframe(denial_summary)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=denial_summary.index, y=denial_summary.values, ax=ax, palette="coolwarm")
        plt.xticks(rotation=45)
        plt.title("Top CPT Codes by Denial Count")
        plt.ylabel("Denial Count")
        plt.xlabel("CPT Code")
        st.pyplot(fig)
else:
    st.info("📥 Please upload a CSV file to get started.")
