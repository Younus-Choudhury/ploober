# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Function to load data and models
def load_assets():
    """
    Loads the trained model, cleaned data, and metrics from the file system.
    This function is cached to prevent re-loading on every Streamlit rerun.
    """
    try:
        # Load the pre-trained model and metrics from the products directory
        rf_pipeline = joblib.load('products/rf_pipeline.joblib')
        
        with open('products/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        # Load the cleaned data to be used for insights
        df_clean = pd.read_csv('products/clean_insurance_data.csv')
        
        return rf_pipeline, metrics, df_clean

    except FileNotFoundError:
        st.error("Assets not found. Please run 'ploomber build' first to train the model and generate the data.")
        st.stop()

# Load all required assets at the start of the app
rf_pipeline, metrics, df_clean = load_assets()

# --- Everything below this line is your original Streamlit app code,
# --- now using the loaded variables instead of `upstream`.

r2 = metrics['r2_score']
rmse = metrics['rmse']

st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border-radius: 25px; border: none; padding: 0.75rem 2rem;
        font-weight: 600; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .metric-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem; border-radius: 15px; text-align: center;
        color: #2d6a4f; margin: 1rem 0;
    }
    .insight-card {
        background: linear-gradient(135deg, #ff6b6b, #feca57);
        color: white; padding: 1.5rem; border-radius: 15px;
        margin: 1rem 0; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')

st.title("🏥 Insurance Premium Predictor")
st.markdown(f"### 🎯 Get an accurate estimate of your insurance premium using advanced machine learning")
st.markdown(f"""This application uses a **Random Forest Regressor** trained on real health insurance data to predict your annual premium based on key personal factors. Our model achieves high accuracy with an R² score of **{r2:.3f}** and RMSE of **${rmse:,.0f}**.""")

st.markdown("---")
st.header("🧮 Premium Calculator")
st.markdown("**Adjust the values below to see how different factors impact your estimated premium.**")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### 👤 Personal Information")
    age = st.slider("👶 Age", min_value=18, max_value=100, value=30, step=1)
    sex = st.selectbox("🚻 Gender", options=["male", "female"])
    
with col2:
    st.markdown("#### 📊 Health Metrics")
    bmi = st.number_input(
        "⚖️ BMI (Body Mass Index)",
        min_value=15.0, max_value=50.0, value=25.0, step=0.1,
        help="BMI = weight(kg) / height(m)²"
    )
    children = st.number_input(
        "👪 Number of Children",
        min_value=0, max_value=10, value=0, step=1
    )
    
with col3:
    st.markdown("#### 🌍 Lifestyle & Location")
    smoker = st.selectbox("🚬 Smoking Status", options=["no", "yes"])
    region = st.selectbox(
        "🗺️ Region",
        options=["northeast", "northwest", "southeast", "southwest"]
    )

if bmi < 18.5:
    bmi_category = "Underweight"
    bmi_color = "blue"
elif bmi < 25:
    bmi_category = "Normal weight"
    bmi_color = "green"
elif bmi < 30:
    bmi_category = "Overweight"
    bmi_color = "orange"
else:
    bmi_category = "Obese"
    bmi_color = "red"

st.markdown(f"**BMI Category:** :{bmi_color}[{bmi_category}]")

st.markdown("---")
st.header("💰 Premium Prediction")

if st.button("🔮 Calculate My Premium", type="primary"):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }])

    with st.spinner("🔄 Calculating your premium..."):
        prediction = rf_pipeline.predict(input_df)[0]
        time.sleep(1)

    st.markdown(f"""
    <div class="metric-card">
        <h2>Your Estimated Annual Premium</h2>
        <h1 style="font-size: 3rem; margin: 1rem 0;">${prediction:,.0f}</h1>
        <p>This estimate is based on statistical analysis of real insurance data. Actual premiums may vary based on additional factors and insurer policies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    avg_premium = df_clean['charges'].mean()
    if prediction > avg_premium:
        st.warning(f"💡 Your estimated premium is **${prediction-avg_premium:,.0f}** above the average (${avg_premium:,.0f})")
    else:
        st.success(f"💡 Your estimated premium is **${avg_premium-prediction:,.0f}** below the average (${avg_premium:,.0f})")
    
    risk_factors = []
    if smoker == "yes":
        risk_factors.append("🚬 Smoking status significantly increases premium")
    if bmi >= 30:
        risk_factors.append("⚖️ High BMI may contribute to increased costs")
    if age >= 50:
        risk_factors.append("👴 Age is a contributing factor to higher premiums")
    
    if risk_factors:
        st.markdown("**Factors affecting your premium:**")
        for factor in risk_factors:
            st.markdown(f"- {factor}")

st.markdown("---")
st.header("📊 Data Insights & Analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    smoker_diff = df_clean[df_clean['smoker']=='yes']['charges'].mean() / df_clean[df_clean['smoker']=='no']['charges'].mean()
    st.markdown(f"""
    <div class="insight-card">
        <h3>🚬 Smoking Impact</h3>
        <h2>{smoker_diff:.1f}x</h2>
        <p>Higher cost for smokers</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    age_corr = df_clean['age'].corr(df_clean['charges'])
    st.markdown(f"""
    <div class="insight-card">
        <h3>📈 Age Factor</h3>
        <h2>{age_corr:.2f}</h2>
        <p>Correlation with cost</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    bmi_corr = df_clean['bmi'].corr(df_clean['charges'])
    st.markdown(f"""
    <div class="insight-card">
        <h3>⚖️ BMI Impact</h3>
        <h2>{bmi_corr:.2f}</h2>
        <p>Correlation with cost</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    model_accuracy = r2
    st.markdown(f"""
    <div class="insight-card">
        <h3>🎯 Model Accuracy</h3>
        <h2>{model_accuracy:.1%}</h2>
        <p>R² Score</p>
    </div>
    """, unsafe_allow_html=True)

# ... The rest of your interactive visualization code ...

with st.expander("🔍 Detailed Data Analysis", expanded=False):
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔗 Correlations", "🎯 Model Performance", "🌟 Feature Importance"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Premium Distribution by Smoking Status")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=df_clean, x="smoker", y="charges", palette=["lightgreen", "lightcoral"], ax=ax)
            ax.set_title("Insurance Charges by Smoking Status")
            ax.set_xlabel("Smoking Status")
            ax.set_ylabel("Annual Charges ($)")
            st.pyplot(fig)
        with col2:
            st.subheader("Charges Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df_clean["charges"], kde=True, bins=30, ax=ax)
            ax.set_title("Distribution of Insurance Charges")
            ax.set_xlabel("Annual Charges ($)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
    
    with tab2:
        st.subheader("Correlation Matrix")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df_clean.select_dtypes(include=[np.number]).corr()
            sns.heatmap(corr, annot=True, cmap="RdYlBu_r", center=0, square=True, ax=ax, cbar_kws={"shrink": .8})
            ax.set_title("Feature Correlation Heatmap")
            st.pyplot(fig)
        with col2:
            st.markdown("**Key Correlations:**")
            st.write(f"• Age ↔ Charges: {age_corr:.3f}")
            st.write(f"• BMI ↔ Charges: {bmi_corr:.3f}")
            st.write(f"• Children ↔ Charges: {df_clean['children'].corr(df_clean['charges']):.3f}")
    
    with tab3:
        st.subheader("Model Performance Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            y_test = pd.read_csv('products/clean_insurance_data.csv')['charges'].iloc[-int(len(df_clean)*0.2):]
            y_pred = rf_pipeline.predict(pd.read_csv('products/clean_insurance_data.csv').drop('charges', axis=1).iloc[-int(len(df_clean)*0.2):])
            ax.scatter(y_test, y_pred, alpha=0.6, color="steelblue")
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            ax.set_xlabel("Actual Charges ($)")
            ax.set_ylabel("Predicted Charges ($)")
            ax.set_title("Actual vs Predicted Values")
            st.pyplot(fig)
        with col2:
            st.markdown("**Model Metrics:**")
            st.metric("R² Score", f"{r2:.3f}", help="Coefficient of determination")
            st.metric("RMSE", f"${rmse:,.0f}", help="Root Mean Square Error")
            st.metric("Mean Absolute Error", f"${np.mean(np.abs(y_test - y_pred)):,.0f}")
    
    with tab4:
        st.subheader("Feature Importance")
        rf = rf_pipeline.named_steps["model"]
        preprocessor = rf_pipeline.named_steps["pre"]
        
        numeric_features = ["age", "bmi", "children"]
        categorical_features = ["sex", "smoker", "region"]
        
        num_names = numeric_features
        cat_transformer = preprocessor.named_transformers_["cat"]
        cat_names = list(cat_transformer.get_feature_names_out(categorical_features))
        feat_names = num_names + cat_names
        
        importances = rf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(
                x=importances[sorted_idx],
                y=[feat_names[i] for i in sorted_idx],
                palette="viridis", ax=ax
            )
            ax.set_xlabel("Feature Importance")
            ax.set_title("Random Forest Feature Importance")
            st.pyplot(fig)
