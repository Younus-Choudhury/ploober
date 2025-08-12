# =================================================================================
# Insurance Premium Prediction App using Streatlit and python

# =================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =================================================================================
# Streamlit App Configuration and Layout
# =================================================================================

st.set_page_config(
    page_title="Insurance Premium Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

plt.style.use('ggplot')

st.markdown(
    """
    <style>
    .big-font {
        font-size: 30px !important;
        font-weight: bold;
    }
    .verdana-text {
        font-family: 'Verdana', sans-serif;
        margin-bottom: 1em;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =================================================================================
# Data Loading and Model Training
# =================================================================================

@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv("insurance.csv")
    except FileNotFoundError:
        st.error("Error: 'insurance.csv' not found.")
        return None

    df_clean = df.copy().drop_duplicates().reset_index(drop=True)

    for col in ["sex", "smoker", "region"]:
        df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()

    df_clean["age"] = pd.to_numeric(df_clean["age"], errors="coerce").astype(int)
    df_clean["bmi"] = pd.to_numeric(df_clean["bmi"], errors="coerce")
    df_clean["children"] = pd.to_numeric(df_clean["children"], errors="coerce").astype(int)
    df_clean["charges"] = pd.to_numeric(df_clean["charges"], errors="coerce")

    return df_clean


@st.cache_resource
def train_model(df_clean):
    if df_clean is None:
        return None, None, None

    X = df_clean.drop(columns=["charges"])
    y = df_clean["charges"]

    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_pipeline = Pipeline([
        ("pre", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    rf_pipeline.fit(X_train, y_train)

    y_pred = rf_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return rf_pipeline, y_test, y_pred, preprocessor, numeric_features, categorical_features, rmse, r2


df_clean = load_and_preprocess_data()
if df_clean is not None:
    rf_pipeline, y_test, y_pred, preprocessor, numeric_features, categorical_features, rmse, r2 = train_model(df_clean)
else:
    st.stop()

# =================================================================================
# Interactive Premium Estimator
# =================================================================================

st.title("👨‍⚕️ Insurance Premium Predictor")
st.markdown("### Estimate your insurance premium with a data-driven model!")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("👶 Age", min_value=18, max_value=100, value=30, step=1)
    sex = st.selectbox("🚻 Sex", options=["male", "female"])
with col2:
    bmi = st.number_input("⚖️ BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
    children = st.number_input("👪 Children", min_value=0, max_value=10, value=0, step=1)
with col3:
    smoker = st.selectbox("🚬 Smoker?", options=["yes", "no"])
    region = st.selectbox("🗺️ Region", options=["northeast", "northwest", "southeast", "southwest"])

if st.button("Estimate Premium"):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }])

    prediction = rf_pipeline.predict(input_df)[0]
    st.markdown(f"<div class='big-font' style='color: #28a745;'>${prediction:,.2f}</div>", unsafe_allow_html=True)
    st.write(f"RMSE: ${rmse:,.2f}, R²: {r2:.3f}")

# =================================================================================
# EDA
# =================================================================================

st.markdown("---")
st.header("Exploratory Data Analysis")

palette = "viridis"
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Correlation Heatmap**")
    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
    sns.heatmap(df_clean.corr(numeric_only=True), annot=True, fmt=".2f", cmap=palette, ax=ax_corr)
    st.pyplot(fig_corr)

with col2:
    st.markdown("**Distribution of Charges**")
    fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
    sns.histplot(df_clean["charges"], kde=True, stat="density", color="skyblue", ax=ax_hist)
    st.pyplot(fig_hist)

st.subheader("Interactive Plot")
fig_px = px.scatter(df_clean, x="bmi", y="charges", color="smoker",
                    color_discrete_map={"yes": "red", "no": "green"})
st.plotly_chart(fig_px, use_container_width=True)

# Feature Importances
st.subheader("Feature Importances")
rf = rf_pipeline.named_steps["model"]
cat_transformer = rf_pipeline.named_steps["pre"].named_transformers_["cat"]
cat_names = list(cat_transformer.get_feature_names_out(categorical_features))
feat_names = numeric_features + cat_names
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

fig_feat, ax_feat = plt.subplots(figsize=(7, 4))
sns.barplot(x=importances[sorted_idx], y=[feat_names[i] for i in sorted_idx], palette="plasma", ax=ax_feat)
st.pyplot(fig_feat)
























