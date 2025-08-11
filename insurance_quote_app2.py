# =================================================================================
# Insurance Premium Prediction App
# Refined and improved design with a more sophisticated visual theme
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

# Use a clean, modern style for Matplotlib plots
plt.style.use('ggplot')

# Use a container for the main app title and description
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
# Data Loading and Model Training (Cached for Performance)
# =================================================================================

@st.cache_data
def load_and_preprocess_data():
    """
    Loads, cleans, and returns the preprocessed DataFrame.
    """
    try:
        df = pd.read_csv("insurance.csv")
    except FileNotFoundError:
        st.error("Error: 'insurance.csv' not found. Please make sure the file "
                 "is in the same directory as this script.")
        return None

    # Data cleaning
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # Standardize categorical columns
    for col in ["sex", "smoker", "region"]:
        df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()

    # Ensure numeric types
    df_clean["age"] = (
        pd.to_numeric(df_clean["age"], errors="coerce").astype(int)
    )
    df_clean["bmi"] = pd.to_numeric(df_clean["bmi"], errors="coerce")
    df_clean["children"] = (
        pd.to_numeric(df_clean["children"], errors="coerce").astype(int)
    )
    df_clean["charges"] = pd.to_numeric(df_clean["charges"], errors="coerce")

    return df_clean


@st.cache_resource
def train_model(df_clean):
    """
    Trains and returns the Random Forest model pipeline, along with
    the performance metrics.
    """
    if df_clean is None:
        return None, None, None

    # Prepare features
    X = df_clean.drop(columns=["charges"])
    y = df_clean["charges"]

    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train Random Forest model
    rf_pipeline = Pipeline(
        [
            ("pre", preprocessor),
            (
                "model",
                RandomForestRegressor(random_state=42, n_jobs=-1)
            ),
        ]
    )

    rf_pipeline.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = rf_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return (
        rf_pipeline,
        y_test,
        y_pred,
        preprocessor,
        numeric_features,
        categorical_features,
        rmse,
        r2,
    )


# Load data and train model
df_clean = load_and_preprocess_data()
if df_clean is not None:
    (
        rf_pipeline,
        y_test,
        y_pred,
        preprocessor,
        numeric_features,
        categorical_features,
        rmse,
        r2,
    ) = train_model(df_clean)
else:
    st.stop()


# =================================================================================
# Interactive Premium Estimator & Prediction Display
# =================================================================================

st.title("👨‍⚕️ Insurance Premium Predictor")
st.markdown("### Estimate your insurance premium with a data-driven model!")
st.markdown(
    "This application predicts insurance premiums based on key personal factors. "
    "The model is a **Random Forest Regressor** trained on a public health "
    "insurance dataset."
)
st.info(
    "💡 **Instructions:** Use the interactive controls below to input your "
    "details and get an instant premium estimate."
)

st.header("🧮 Premium Estimator")
st.markdown(
    "Adjust the values below to see how they impact your estimated premium."
)

# Use columns for a more compact layout
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("👶 Age", min_value=18, max_value=100, value=30, step=1)
    sex = st.selectbox("🚻 Sex", options=["male", "female"])
with col2:
    bmi = st.number_input(
        "⚖️ BMI (Body Mass Index)", min_value=15.0, max_value=50.0, value=25.0,
        step=0.1
    )
    children = st.number_input(
        "👪 Number of Children", min_value=0, max_value=10, value=0, step=1
    )
with col3:
    smoker = st.selectbox("🚬 Smoker?", options=["yes", "no"])
    region = st.selectbox(
        "🗺️ Region",
        options=["northeast", "northwest", "southeast", "southwest"]
    )

# Create a button to trigger the prediction
if st.button(
    "Estimate Premium", help="Click to generate an insurance premium estimate"
):
    # Generate prediction from user inputs
    input_df = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region,
            }
        ]
    )

    prediction = rf_pipeline.predict(input_df)[0]

    # Display the result
    st.markdown("---")
    st.subheader("Your Estimated Premium is...")
    st.markdown(
        f"<div class='big-font' style='color: #28a745;'>${prediction:,.2f}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "*This is a detailed quote, but remember that other factors, such as "
        "pre-existing health conditions, as well as promotions, discounts, "
        "and other market commercial decisions, could influence the price "
        "both up and down.*"
    )

    st.markdown("---")
    st.subheader("Model Performance")
    st.markdown(
        f"<div class='verdana-text'>The **Root Mean Squared Error (RMSE)** "
        "is **${rmse:,.2f}**. This means that, on average, the model's "
        "prediction for an insurance premium is off by about "
        "**${rmse:,.2f}**.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='verdana-text'>The **R-squared (R²)** value is "
        f"**{r2:.3f}**. This means that the factors used in the model can "
        f"explain **{r2:.1%}** of the variation in insurance premiums.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**Profile Summary:**")
    st.markdown(
        f"""
    - **Age:** {age} years
    - **Gender:** {sex.title()}
    - **BMI:** {bmi:.2f}
    - **Children:** {children}
    - **Smoker:** {smoker.title()}
    - **Region:** {region.title()}
    """
    )


# =================================================================================
# Main Content - EDA and Diagnostics
# =================================================================================
st.markdown("---")
st.header("Exploratory Data Analysis & Model Diagnostics")

# Use a clean, vibrant color palette for all seaborn plots
palette = "viridis"

# Use columns for a more organized layout
st.subheader("Data Insights")
col1, col2 = st.columns(2)

with col1:
    # Correlation heatmap
    st.markdown("**Correlation Heatmap**")
    fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
    corr = df_clean.select_dtypes(include=[np.number]).corr()
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap=palette, cbar=True, ax=ax_corr
    )
    st.pyplot(fig_corr)

    # Charges by smoker
    st.markdown("**Charges by Smoker Status**")
    fig_box, ax_box = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df_clean, x="smoker", y="charges", hue="smoker",
                palette="rocket", legend=False, ax=ax_box)
    ax_box.set_title("")  # Title removed to avoid redundancy with markdown
    st.pyplot(fig_box)

with col2:
    # Charges distribution
    st.markdown("**Distribution of Charges**")
    fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
    sns.histplot(
        df_clean["charges"], kde=True, stat="density", color="skyblue",
        ax=ax_hist
    )
    ax_hist.set_title("")
    ax_hist.set_xlabel("Charges ($)")
    st.pyplot(fig_hist)

    # Charges by region
    st.markdown("**Median Charges by Region**")
    region_med = df_clean.groupby("region")["charges"].median().sort_values(
        ascending=False
    )
    fig_bar, ax_bar = plt.subplots(figsize=(7, 4))
    sns.barplot(x=region_med.index, y=region_med.values, hue=region_med.index,
                palette="crest", legend=False, ax=ax_bar)
    ax_bar.set_title("")
    ax_bar.set_xlabel("Region")
    ax_bar.set_ylabel("Median Charges ($)")
    ax_bar.tick_params(axis='x', rotation=45)
    st.pyplot(fig_bar)

# BMI vs Charges interactive plot
st.subheader("Interactive Plot")
st.markdown("**BMI vs Charges by Smoker Status**")
fig_px = px.scatter(
    df_clean, x="bmi", y="charges", color="smoker",
    title="Hover over points for more details",
    color_discrete_map={"yes": "red", "no": "green"},
)
st.plotly_chart(fig_px, use_container_width=True)


st.subheader("Model Diagnostics")
col3, col4 = st.columns(2)

with col3:
    st.markdown("**Actual vs Predicted Charges**")
    fig_diag, ax_diag = plt.subplots(figsize=(7, 4))
    ax_diag.scatter(y_test, y_pred, alpha=0.6, s=40, color="#1f77b4")
    minv = min(y_test.min(), y_pred.min())
    maxv = max(y_test.max(), y_pred.max())
    ax_diag.plot(
        [minv, maxv], [minv, maxv],
        linestyle="--", color="red", linewidth=2
    )
    ax_diag.set_title("")
    ax_diag.set_xlabel("Actual Charges ($)")
    ax_diag.set_ylabel("Predicted Charges ($)")
    st.pyplot(fig_diag)

with col4:
    st.markdown("**Feature Importances**")
    rf = rf_pipeline.named_steps["model"]
    num_names = numeric_features
    cat_transformer = rf_pipeline.named_steps["pre"].named_transformers_["cat"]
    cat_names = list(cat_transformer.get_feature_names_out(
        categorical_features)
    )
    feat_names = num_names + cat_names

    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    fig_feat, ax_feat = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x=importances[sorted_idx],
        y=[feat_names[i] for i in sorted_idx],
        hue=[feat_names[i] for i in sorted_idx],
        palette="plasma",
        legend=False,
        ax=ax_feat,
    )
    ax_feat.set_title("")
    ax_feat.set_xlabel("Importance")
    ax_feat.set_ylabel("Feature")
    st.pyplot(fig_feat)

