# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import time

# This is a Ploomber-injected variable. Ploomber will
# pass the `upstream` and `product` objects to the script.
# We don't need 'product' here, but it's good practice to accept it.
def run_streamlit_app(upstream, product):
    # Load the pre-trained model and metrics from the upstream paths
    rf_pipeline = joblib.load(str(upstream['model_training']['model']))
    with open(str(upstream['model_training']['metrics']), 'r') as f:
        metrics = json.load(f)
    r2 = metrics['r2_score']
    rmse = metrics['rmse']

    # Ploomber will run this entire script when it's the active task,
    # so we don't need the @st.cache_data decorator or the `if __name__ == "__main__"` block.
    # The application code starts here.

    st.set_page_config(
        page_title="Insurance Premium Predictor",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
    <style>
        .main {
            padding-top: 2rem;
        }
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 25px;
            border: none;
            padding: 0.75rem 2rem;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        .metric-card {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            color: #2d6a4f;
            margin: 1rem 0;
        }
        .insight-card {
            background: linear-gradient(135deg, #ff6b6b, #feca57);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            text-align: center;
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

    # Note: Ploomber's shell executor won't allow the app to be interactive
    # directly within the Ploomber command. This is primarily for demonstrating
    # the flow. To run this app interactively, you would run `streamlit run app.py`
    # after Ploomber has completed the training tasks.

# The `run_streamlit_app` function is called at the end
# to initiate the Streamlit application.
run_streamlit_app(upstream, product)
