import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# ------------------------------
# Load dataset
# ------------------------------
@st.cache_data
def load_data():
    claims_path = "data/claims.csv"

    # Use header=1 because first row is empty, second row has column names
    df = pd.read_csv(claims_path, encoding="latin1", header=1)

    # Clean dataset
    df = df.dropna(how="all")  # remove fully empty rows
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]  # drop unnamed cols

    # Drop "#" column if exists
    if "#" in df.columns:
        df = df.drop(columns=["#"])

    return df


# ------------------------------
# Train model
# ------------------------------
@st.cache_resource
def train_model(df):
    if "Denial Reason" not in df.columns:
        st.error("❌ Could not find target column ('Denial Reason'). Please check dataset.")
        return None

    # Convert target into binary classification: Denied vs Not Denied
    df["Denied"] = df["Denial Reason"].notna().astype(int)

    # Use a simple feature (CPT Code + Insurance + Physician) for demo
    df["features"] = df["CPT Code"].astype(str) + " " + df["Insurance Company"].astype(str) + " " + df["Physician Name"].astype(str)

    X = df["features"]
    y = df["Denied"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=200))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return pipeline, report


# ------------------------------
# Streamlit App
# ------------------------------
def main():
    st.title("🏥 Healthcare Claim Denial Prediction App")

    # Load dataset
    df = load_data()

    st.subheader("📂 Training Data Preview")
    st.write(df.head())

    # Train model
    model_result = train_model(df)
    if model_result is None:
        return
    model, report = model_result

    st.subheader("📊 Model Evaluation")
    st.json(report)

    # Prediction section
    st.subheader("🔮 Predict Claim Denial")
    cpt_code = st.text_input("Enter CPT Code (e.g., 99213)")
    insurance = st.text_input("Enter Insurance Company (e.g., Medicare)")
    physician = st.text_input("Enter Physician Name (e.g., Dr. Smith)")

    if st.button("Predict"):
        if not cpt_code or not insurance or not physician:
            st.warning("⚠️ Please fill all fields before predicting.")
        else:
            feature = f"{cpt_code} {insurance} {physician}"
            prediction = model.predict([feature])[0]
            result = "❌ Claim Denied" if prediction == 1 else "✅ Claim Approved"
            st.success(result)


if __name__ == "__main__":
    main()
