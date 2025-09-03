# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ------------------------------
# Load dataset
# ------------------------------
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

# ------------------------------
# Clean monetary columns
# ------------------------------
def clean_currency_columns(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(r"[\$,]", "", regex=True).astype(float)
    return df

# ------------------------------
# Convert Denial Reason to binary target
# ------------------------------
def encode_target(df):
    df["Denial"] = df["Denial Reason"].apply(lambda x: 0 if pd.isna(x) or str(x).strip() == "" else 1)
    return df

# ------------------------------
# Main App
# ------------------------------
st.title("Healthcare Claim Denial Prediction App")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)

    if df is not None:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        required_columns = ["Payment Amount", "Balance", "Denial Reason"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
        else:
            df = clean_currency_columns(df, ["Payment Amount", "Balance"])
            df = encode_target(df)

            st.subheader("Processed Data Preview")
            st.dataframe(df.head())

            # Features & target
            X = df[["Payment Amount", "Balance"]]
            y = df["Denial"]

            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            st.success("Model trained!")

            # Predictions
            y_pred = model.predict(X_test)

            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))
else:
    st.info("Please upload a CSV file to get started.")
