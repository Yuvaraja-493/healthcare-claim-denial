import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ------------------------- Page setup -------------------------
st.set_page_config(page_title="Healthcare Claim Denial Analysis", layout="wide")
st.title("🏥 Healthcare Claim Denial Analysis & Prediction")

st.write(
    "Upload a **CSV or Excel** file with columns like **CPT Code, Insurance Company, "
    "Physician Name, Payment Amount, Balance, Denial Reason** (if available)."
)

# ------------------------- Helpers -------------------------
CURRENCY_COLS_CANDIDATES = ["Payment Amount", "Payment", "Paid", "Balance", "Patient Balance", "Outstanding Balance"]
CPT_COLS = ["CPT Code", "CPT", "CPT_Code"]
PAYER_COLS = ["Insurance Company", "Payer", "Plan"]
PROVIDER_COLS = ["Physician Name", "Provider", "Rendering Provider", "Billing Provider"]
DENIAL_REASON_COLS = ["Denial Reason", "Denial_Reason", "Remark", "CARC/RARC"]
DENIAL_FLAG_COLS = ["Denial", "Denied", "IsDenied"]

# Map common ANSI CARC codes/text to RCM root cause + fix
ROOT_CAUSE_MAP = {
    "16": ("Missing/Incomplete info", "Correct/complete data & resubmit; verify demographics & DX pointers."),
    "45": ("Charge exceeds fee schedule/NCCI/bundling", "Review NCCI edits, modifiers (e.g., -59, -51), contract rates."),
    "96": ("Non-covered service", "Confirm coverage/LCD/NCD; append medical necessity notes or ABN."),
    "197": ("Precert/Authorization required", "Obtain retro-auth if possible; update auth workflow."),
    "22": ("COB/Other carrier", "Work COB, update primary payer and resubmit secondary."),
    "109": ("Claim not covered by this payer", "Verify payer selection/plan; correct payer ID and resubmit."),
    "151": ("Attachment/Medical records needed", "Attach records, operative notes; set documentation checklist."),
    "18": ("Duplicate claim/service", "Void/adjust original; fix billing workflow to avoid duplicates."),
}

def normalize_money(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[^\d\.\-]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )

def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def build_denial_flag(df, denial_reason_col, denial_flag_col):
    if denial_flag_col and denial_flag_col in df.columns:
        # Coerce to 0/1
        return (pd.to_numeric(df[denial_flag_col], errors="coerce").fillna(0) > 0).astype(int)
    if denial_reason_col and denial_reason_col in df.columns:
        return df[denial_reason_col].apply(lambda x: 1 if pd.notnull(x) and str(x).strip() != "" and str(x).strip().lower() != "none" else 0).astype(int)
    # Fallback heuristic: zero payment & positive balance ⇒ likely denial
    pay_col = pick_first_existing(df, CURRENCY_COLS_CANDIDATES)
    bal_col = pick_first_existing(df, ["Balance", "Patient Balance", "Outstanding Balance"])
    if pay_col and bal_col:
        return ((df[pay_col].fillna(0) <= 0) & (df[bal_col].fillna(0) > 0)).astype(int)
    return pd.Series(0, index=df.index, dtype=int)

def parse_carc_code(reason: str) -> str:
    if pd.isna(reason):
        return ""
    txt = str(reason)
    # try to catch leading numeric code like "16 - Missing info"
    parts = txt.split("-", 1)
    code = parts[0].strip()
    if code.isdigit():
        return code
    return ""

def add_root_cause_columns(df, denial_reason_col):
    df["CARC_Code"] = df[denial_reason_col].apply(parse_carc_code) if denial_reason_col else ""
    df["Root Cause"] = df["CARC_Code"].map(lambda c: ROOT_CAUSE_MAP.get(c, (None, None))[0])
    df["Suggested Fix"] = df["CARC_Code"].map(lambda c: ROOT_CAUSE_MAP.get(c, (None, None))[1])
    return df

def df_to_download_link(df, filename):
    buff = io.BytesIO()
    df.to_excel(buff, index=False)
    return buff, f"sandbox:/mnt/data/{filename}"

# ------------------------- File upload -------------------------
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if not uploaded:
    st.info("📥 Upload a file to begin.")
    st.stop()

# Read file
if uploaded.name.lower().endswith(".csv"):
    df = pd.read_csv(uploaded, skip_blank_lines=True)
else:
    df = pd.read_excel(uploaded)

# Basic clean
df = df.dropna(how="all")
df.columns = df.columns.str.strip()

# Normalize currency-like cols if present
for col in CURRENCY_COLS_CANDIDATES:
    if col in df.columns:
        df[col] = normalize_money(df[col])

# Identify key columns (best guess)
cpt_col = pick_first_existing(df, CPT_COLS)
payer_col = pick_first_existing(df, PAYER_COLS)
prov_col = pick_first_existing(df, PROVIDER_COLS)
denial_reason_col = pick_first_existing(df, DENIAL_REASON_COLS)
denial_flag_col = pick_first_existing(df, DENIAL_FLAG_COLS)

# Build Denial flag
df["Denial"] = build_denial_flag(df, denial_reason_col, denial_flag_col)

# Add root cause / fix columns if we have reasons
if denial_reason_col:
    df = add_root_cause_columns(df, denial_reason_col)

# ------------------------- Raw preview -------------------------
st.subheader("📂 Raw Data Preview")
st.dataframe(df.head(50), use_container_width=True)
st.success("✅ Data loaded & standardized.")

# ------------------------- Summary KPIs -------------------------
total_claims = len(df)
denied_claims = int(df["Denial"].sum())
denial_rate = denied_claims / total_claims * 100 if total_claims else 0

pay_col = pick_first_existing(df, CURRENCY_COLS_CANDIDATES)
bal_col = pick_first_existing(df, ["Balance", "Patient Balance", "Outstanding Balance"])

lost_revenue = float(df.loc[df["Denial"] == 1, bal_col].sum()) if bal_col in df.columns else np.nan

kpi_cols = st.columns(4)
kpi_cols[0].metric("Total Claims", f"{total_claims:,}")
kpi_cols[1].metric("Denied Claims", f"{denied_claims:,}")
kpi_cols[2].metric("Denial Rate", f"{denial_rate:.1f}%")
kpi_cols[3].metric("Est. Lost Revenue", f"${lost_revenue:,.2f}" if not np.isnan(lost_revenue) else "—")

# ------------------------- Analysis tabs -------------------------
tabs = st.tabs([
    "Top CPTs",
    "By Payer",
    "By Physician",
    "Denial Reasons & Root Causes",
    "Prediction (Random Forest)"
])

# ---- Top CPTs
with tabs[0]:
    st.subheader("📊 Top CPT Codes by Denial")
    if not cpt_col:
        st.warning("No CPT column detected.")
    else:
        g = df.groupby(cpt_col)["Denial"].agg(["count", "sum"])
        g["Denial Rate %"] = (g["sum"] / g["count"] * 100).round(1)
        g = g.rename(columns={"count": "Total", "sum": "Denied"}).sort_values("Denied", ascending=False)
        st.dataframe(g.head(50), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        top = g.head(15).iloc[::-1]  # plot as horizontal
        ax.barh(top.index.astype(str), top["Denied"])
        ax.set_title("Top CPT Codes by Denial Count")
        ax.set_xlabel("Denied Claims")
        st.pyplot(fig, use_container_width=True)

        # download
        buff, fname = df_to_download_link(g.reset_index(), "top_cpts.xlsx")
        st.download_button("⬇️ Download CPT Summary", data=buff.getvalue(), file_name="top_cpts.xlsx")

# ---- By Payer
with tabs[1]:
    st.subheader("🏦 Denials by Payer")
    if not payer_col:
        st.warning("No Payer column detected.")
    else:
        gp = df.groupby(payer_col)["Denial"].agg(["count", "sum"])
        gp["Denial Rate %"] = (gp["sum"] / gp["count"] * 100).round(1)
        gp = gp.rename(columns={"count": "Total", "sum": "Denied"}).sort_values("Denied", ascending=False)
        st.dataframe(gp, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        top = gp.head(15).iloc[::-1]
        ax.barh(top.index.astype(str), top["Denied"])
        ax.set_title("Top Payers by Denial Count")
        ax.set_xlabel("Denied Claims")
        st.pyplot(fig, use_container_width=True)

        buff, _ = df_to_download_link(gp.reset_index(), "denials_by_payer.xlsx")
        st.download_button("⬇️ Download Payer Summary", data=buff.getvalue(), file_name="denials_by_payer.xlsx")

# ---- By Physician
with tabs[2]:
    st.subheader("👩‍⚕️ Denials by Physician")
    if not prov_col:
        st.warning("No Physician/Provider column detected.")
    else:
        gv = df.groupby(prov_col)["Denial"].agg(["count", "sum"])
        gv["Denial Rate %"] = (gv["sum"] / gv["count"] * 100).round(1)
        gv = gv.rename(columns={"count": "Total", "sum": "Denied"}).sort_values("Denied", ascending=False)
        st.dataframe(gv, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        top = gv.head(15).iloc[::-1]
        ax.barh(top.index.astype(str), top["Denied"])
        ax.set_title("Top Physicians by Denial Count")
        ax.set_xlabel("Denied Claims")
        st.pyplot(fig, use_container_width=True)

        buff, _ = df_to_download_link(gv.reset_index(), "denials_by_physician.xlsx")
        st.download_button("⬇️ Download Physician Summary", data=buff.getvalue(), file_name="denials_by_physician.xlsx")

# ---- Denial Reasons & Root Causes
with tabs[3]:
    st.subheader("🧭 Denial Reasons → Root Causes → Fixes")
    if not denial_reason_col:
        st.info("No Denial Reason column detected. Showing heuristic root-cause is not possible without reasons.")
    else:
        rr = (
            df[df["Denial"] == 1]
            .groupby([denial_reason_col, "CARC_Code", "Root Cause", "Suggested Fix"])
            .size()
            .reset_index(name="Denied Count")
            .sort_values("Denied Count", ascending=False)
        )
        st.dataframe(rr.head(100), use_container_width=True)

        # quick pivot: Root cause summary
        rc = (
            df[df["Denial"] == 1]
            .groupby(["Root Cause", "Suggested Fix"])
            .size()
            .reset_index(name="Denied Count")
            .sort_values("Denied Count", ascending=False)
        )
        st.markdown("**Top Root Causes**")
        st.dataframe(rc.head(50), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        top = rc.head(12).iloc[::-1]
        ax.barh(top["Root Cause"].fillna("Other").astype(str), top["Denied Count"])
        ax.set_title("Top Root Causes by Denied Count")
        ax.set_xlabel("Denied Claims")
        st.pyplot(fig, use_container_width=True)

        buff, _ = df_to_download_link(rr, "denial_reasons_root_causes.xlsx")
        st.download_button("⬇️ Download Root Cause Detail", data=buff.getvalue(), file_name="denial_reasons_root_causes.xlsx")

    st.markdown("---")
    st.markdown("**Guided Fixes (Rules of Thumb):**")
    fixes = pd.DataFrame(
        [
            {"CARC": k, "Root Cause": v[0], "Suggested Fix": v[1]}
            for k, v in ROOT_CAUSE_MAP.items()
        ]
    )
    st.dataframe(fixes, use_container_width=True)

# ---- Prediction tab (Random Forest)
with tabs[4]:
    st.subheader("🤖 Predict Denials (Random Forest)")
    st.caption("Select feature columns below. Categorical features will be one-hot encoded automatically.")
    # Candidate features: anything except the target-like columns
    drop_cols = set(["Denial"])  # target
    candidate_cols = [c for c in df.columns if c not in drop_cols]

    features = st.multiselect("Choose features for the model", candidate_cols,
                              default=[c for c in [cpt_col, payer_col, prov_col, pay_col, bal_col, denial_reason_col] if c in df.columns])

    if len(features) == 0:
        st.info("Select at least one feature to train the model.")
    else:
        X = df[features].copy()
        y = df["Denial"].astype(int)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
        )

        # Identify types
        cat_cols = [c for c in X.columns if X[c].dtype == "object"]
        num_cols = [c for c in X.columns if c not in cat_cols]

        # Pipelines
        num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        cat_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                   ("onehot", OneHotEncoder(handle_unknown="ignore"))])

        pre = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ]
        )

        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced"
        )

        model = Pipeline(steps=[("pre", pre), ("rf", rf)])
        model.fit(X_train, y_train)

        # Metrics
        y_pred = model.predict(X_test)
        # Prob AUC only if both classes present
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1]) if y_test.nunique() > 1 else np.nan

        st.markdown("**Evaluation**")
        st.text(classification_report(y_test, y_pred, zero_division=0))
        st.write(f"ROC-AUC: {auc:.3f}" if not np.isnan(auc) else "ROC-AUC: — (single-class split)")

        # Feature importances (map back through one-hot)
        st.markdown("**Top Feature Importances**")
        # Extract names from ColumnTransformer
        ohe = model.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"] if cat_cols else None
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols)) if ohe is not None else []
        feature_names = num_cols + cat_feature_names
        importances = model.named_steps["rf"].feature_importances_

        fi = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)
        st.dataframe(fi.head(25), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        top = fi.head(15).iloc[::-1]
        ax.barh(top["Feature"], top["Importance"])
        ax.set_title("Top Model Features")
        ax.set_xlabel("Importance")
        st.pyplot(fig, use_container_width=True)

        # Score the whole dataset (optional)
        df["Predicted_Denial_Prob"] = model.predict_proba(X)[:, 1]
        st.markdown("**Scored Dataset (head)**")
        st.dataframe(df[[*features, "Denial", "Predicted_Denial_Prob"]].head(50), use_container_width=True)

        # Download scored data
        buff, _ = df_to_download_link(df[[*features, "Denial", "Predicted_Denial_Prob"]], "scored_claims.xlsx")
        st.download_button("⬇️ Download Scored Claims", data=buff.getvalue(), file_name="scored_claims.xlsx")

# ------------------------- Footer note -------------------------
st.caption(
    "Tip: For the best root-cause mapping, include a clean **Denial Reason** column with CARC codes like "
    "`16`, `45`, `96`, `197` etc., e.g., `16 - Missing information`."
)
