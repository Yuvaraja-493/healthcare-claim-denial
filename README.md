# RCM Denials Analyzer & ML Predictor

This app:
1. Trains a ML model to predict **claim denials**.
2. Lets you **upload new CSV/Excel** files for predictions.
3. Provides **analytics** (top denied CPTs, payers, providers).
4. Suggests **root causes and fixes**.

## How to Run
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
