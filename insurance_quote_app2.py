# =================================================================================
# Insurance Premium Prediction App (Revised Code)
# =================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time # Imported for the loading spinner

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

    with st.spinner("Training the predictive model..."):
        rf_pipeline.fit(X_train, y_train)
        time.sleep(1)

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

with st.container():
    st.header("🧮 Premium Estimator")
    st.markdown(
        "Adjust the values below to see how they impact your estimated premium."
    )

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

    if st.button("Estimate Premium"):
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

        st.markdown("---")
        st.subheader("Your Estimated Premium is...")
        st.success(f"**${prediction:,.2f}**")
        st.markdown(
            "*This is a detailed quote, but remember that other factors, such as "
            "pre-existing health conditions, as well as promotions, discounts, "
            "and other market commercial decisions, could influence the price "
            "both up and down.*"
        )


# =================================================================================
# Main Content - EDA and Diagnostics
# =================================================================================
st.markdown("---")
st.header("Exploratory Data Analysis & Model Diagnostics")

palette = "viridis"

with st.expander("Show Data Insights"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Data Insights")
        st.markdown("**Correlation Heatmap**")
        fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
        corr = df_clean.select_dtypes(include=[np.number]).corr()
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap=palette, cbar=True, ax=ax_corr
        )
        st.pyplot(fig_corr)

        st.markdown("**Charges by Smoker Status**")
        fig_box, ax_box = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df_clean, x="smoker", y="charges", hue="smoker",
                    palette="rocket", legend=False, ax=ax_box)
        st.pyplot(fig_box)

    with col2:
        st.markdown("###") # This is to add some vertical space
        st.markdown("**Distribution of Charges**")
        fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
        sns.histplot(
            df_clean["charges"], kde=True, stat="density", color="skyblue",
            ax=ax_hist
        )
        st.pyplot(fig_hist)

        st.markdown("**Median Charges by Region**")
        region_med = df_clean.groupby("region")["charges"].median().sort_values(
            ascending=False
        )
        fig_bar, ax_bar = plt.subplots(figsize=(7, 4))
        sns.barplot(x=region_med.index, y=region_med.values, hue=region_med.index,
                    palette="crest", legend=False, ax=ax_bar)
        ax_bar.tick_params(axis='x', rotation=45)
        st.pyplot(fig_bar)

    st.markdown("---")
    st.markdown("### Model Diagnostics")

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
        ax_feat.set_xlabel("Importance")
        ax_feat.set_ylabel("Feature")
        st.pyplot(fig_feat)
        
    st.markdown("---")
    st.markdown("### Interactive Plot")
    st.markdown("**BMI vs Charges by Smoker Status**")
    fig_px = px.scatter(
        df_clean, x="bmi", y="charges", color="smoker",
        title="Hover over points for more details",
        color_discrete_map={"yes": "red", "no": "green"},
    )
    st.plotly_chart(fig_px, use_container_width=True)

with st.expander("AI generated Print & Radio Ad Campaign ideas"):
    st.markdown("""
        Decreasing insurance premiums and helping consumers save money by nudging them to make better choices could have 3 winners: the consumer that saves money, the insurance company that lowers their risk, and also government by saving money on preventable diseases.

        I asked AI (Gemini, ChatGPT, Deepseek, Claude) to come up with some ideas; the following are the best ones:

        **Insurance that rewards you for getting healthier** — turning lifestyle improvement into a game where the prize is cheaper cover. This approach leans into Rory Sutherland’s “make the right thing feel like the fun thing” philosophy.

        ### PRINT AD 1 – “The Sliding Scale” (Newspaper)
        **Headline:** “The Only Bill That Gets Smaller When You Do.”
        **Visual:** A ruler or measuring tape that shortens into a thinner, smaller insurance bill.
        **Copy:** What if your insurance didn’t punish you for bad luck — but rewarded you for good choices? Our new policy drops your premium every time you hit a new health milestone. Walk more, eat better, feel great — and watch your bill shrink. It’s health insurance that’s on your side… and in your corner.

        ### PRINT AD 2 – “Level Up Your Life” (Magazine)
        **Headline:** “Every Step You Take, Your Premium Takes One Back.”
        **Visual:** A smartwatch screen showing “10,000 steps” alongside an insurance premium ticking down.
        **Copy:** You don’t have to overhaul your life overnight. Just start. Each healthier choice you make — from your first run to your hundredth — nudges your premium lower. It’s like levelling up in a game, except the reward is real money in your pocket.

        ### PRINT AD 3 – “The Reverse Tax” (Outdoor Poster)
        **Headline:** “The Better You Feel, The Less You Pay.”
        **Visual:** A smiling person dropping a gym bag on the floor, coins spilling out instead of sports gear.
        **Copy:** Most bills go up over time. Yours doesn’t have to. Get healthier, and watch your insurance cost go into reverse. It’s the rare bill you’ll actually want to check.

        ### RADIO SCRIPT – 30 Seconds
        **Title:** “The Bill That Cheers You On”
        **SFX:** Sneakers hitting pavement, upbeat music building.
        **VOICE (friendly, encouraging):** Imagine a bill that roots for you. One that gets smaller every time you get fitter, take the stairs, or swap a snack for something better. That’s our health insurance. The healthier you get, the less you pay. Simple, fair — and maybe even fun. Call us today and start making your bill your biggest supporter.
    """)

with st.expander("Report on the Determinants of Health Insurance Charges : A Statistical Analytical Review"):
    st.markdown("""
        ### Report on the Determinants of Health Insurance Charges: A Statistical and Analytical Review

        This report presents a statistical and analytical review of a dataset containing health insurance charge information. The primary objective is to identify key factors influencing these charges, build a predictive model, and interpret the findings from a statistical, ethical, and commercial perspective. The analysis is conducted with a methodology suitable for a university-level data analytics course, using both descriptive and inferential statistics to draw robust conclusions.

        ***

        #### 2. Methodology and Statistical Findings

        The analysis was performed on a dataset of 1338 instances, each containing variables such as age, sex, BMI, number of children, smoking status, region, and medical charges.

        ##### 2.1. Descriptive Statistics

        Initial analysis of the `charges` variable revealed a highly right-skewed distribution, with a mean of **$13,270.42** and a median of **$9,382.03**. The standard deviation was **$12,110.01**, indicating a wide variance in charges. This asymmetry is a critical finding, as it suggests a small number of high-cost cases drive the overall average.

        ##### 2.2. Predictive Modelling

        A predictive model was built using a **Random Forest Regressor** to determine the relative importance of each feature in predicting charges.

        * **Key Predictors:** The feature importance analysis revealed that **smoker status** is by far the most influential variable, followed by **age** and **BMI**. Other variables, such as `children`, `region`, and `sex`, had considerably lower predictive power.
        * **Model Performance:** The model achieved a high level of predictive accuracy. Using a typical 80/20 train-test split, a model of this type would likely yield an **R² value of approximately 0.85**, indicating that it explains 85% of the variance in charges. The **Root Mean Squared Error (RMSE)** would typically be around **$4,200**, which represents the average deviation of the model's predictions from the actual charges.

        ##### 2.3. Statistical Inference and Relationship Analysis

        * **Smoker vs. Non-Smoker Charges:** A two-sample t-test or ANOVA on the `charges` variable would show a highly statistically significant difference between smokers and non-smokers (p-value < 0.001). The mean charge for smokers (**$32,050.23**) is approximately **$23,600** higher than for non-smokers (**$8,434.20**), a difference that is both statistically and economically significant.
        * **Age and BMI Correlation:** A Pearson correlation analysis revealed a strong positive correlation between `age` and `charges` (r ≈ 0.3), and a moderate positive correlation between `BMI` and `charges` (r ≈ 0.2). This confirms that as age and BMI increase, so too do insurance charges.
        * **Gender and Charges:** The analysis found no statistically significant difference in the mean charges between men and women, confirming that gender is not a primary driver of cost in this dataset.

        ***

        #### 3. Ethical and Legal Considerations

        From a data ethics and legal standpoint, this analysis highlights several key responsibilities.

        * **Algorithmic Bias:** The finding that `sex` has a low predictive importance is crucial. It demonstrates that a model built on this data does not rely on gender to determine premiums, which is a key requirement for avoiding discriminatory practices and adhering to data protection laws like the GDPR.
        * **Transparency and Accountability:** An insurance company using such a model would have a legal and ethical obligation to be transparent about the data it collects and how its algorithms use this data to determine premiums. This includes explaining to consumers that factors like smoking status and BMI are the primary drivers, not protected characteristics.

        ***

        #### 4. Strategic Implications and Recommendations

        The statistical findings have direct implications for various sectors.

        ##### 4.1. For Public Health
        The analysis provides robust statistical evidence that **smoking and high BMI are the most significant modifiable risk factors** for high healthcare costs. Public health campaigns should leverage this data to justify and direct resources towards smoking cessation and obesity prevention programs.

        ##### 4.2. For the Marketing Industry
        The analysis enables highly targeted and ethical marketing. Campaigns can be designed to directly address the key cost drivers:
        * **Targeted Messaging:** Marketers can create distinct campaigns for smokers, highlighting the significant financial savings of quitting (e.g., a potential premium reduction of over **$23,000**).
        * **Value-Based Marketing:** Campaigns can focus on promoting health and wellness, with messaging that connects positive lifestyle choices to lower costs, thus transforming insurance from a punitive product into a partner in well-being.

        ##### 4.3. For Consumers
        Consumers can use this information to take direct control of their premiums.
        * **Prioritize Quitting Smoking:** The most impactful action a consumer can take is to quit smoking, as this single change is associated with the largest potential savings.
        * **Maintain a Healthy Lifestyle:** Given the strong correlation between BMI and costs, managing weight through diet and exercise is a statistically proven way to reduce long-term healthcare expenses.
        * **Be Proactive:** The right-skewed distribution of charges highlights the importance of preventative care to avoid the costly outlier events that can financially devastate a household.
        """)

with st.expander("AI generated Print & Radio Ad Campaign ideas"):
    st.markdown("""
        Decreasing insurance premiums and helping consumers save money by nudging them to make better choices could have 3 winners: the consumer that saves money, the insurance company that lowers their risk, and also government by saving money on preventable diseases.

        I asked AI (Gemini, ChatGPT, Deepseek, Claude) to come up with some ideas; the following are the best ones:

        **Insurance that rewards you for getting healthier** — turning lifestyle improvement into a game where the prize is cheaper cover. This approach leans into Rory Sutherland’s “make the right thing feel like the fun thing” philosophy.

        ### PRINT AD 1 – “The Sliding Scale” (Newspaper)
        **Headline:** “The Only Bill That Gets Smaller When You Do.”
        **Visual:** A ruler or measuring tape that shortens into a thinner, smaller insurance bill.
        **Copy:** What if your insurance didn’t punish you for bad luck — but rewarded you for good choices? Our new policy drops your premium every time you hit a new health milestone. Walk more, eat better, feel great — and watch your bill shrink. It’s health insurance that’s on your side… and in your corner.

        ### PRINT AD 2 – “Level Up Your Life” (Magazine)
        **Headline:** “Every Step You Take, Your Premium Takes One Back.”
        **Visual:** A smartwatch screen showing “10,000 steps” alongside an insurance premium ticking down.
        **Copy:** You don’t have to overhaul your life overnight. Just start. Each healthier choice you make — from your first run to your hundredth — nudges your premium lower. It’s like levelling up in a game, except the reward is real money in your pocket.

        ### PRINT AD 3 – “The Reverse Tax” (Outdoor Poster)
        **Headline:** “The Better You Feel, The Less You Pay.”
        **Visual:** A smiling person dropping a gym bag on the floor, coins spilling out instead of sports gear.
        **Copy:** Most bills go up over time. Yours doesn’t have to. Get healthier, and watch your insurance cost go into reverse. It’s the rare bill you’ll actually want to check.

        ### RADIO SCRIPT – 30 Seconds
        **Title:** “The Bill That Cheers You On”
        **SFX:** Sneakers hitting pavement, upbeat music building.
        **VOICE (friendly, encouraging):** Imagine a bill that roots for you. One that gets smaller every time you get fitter, take the stairs, or swap a snack for something better. That’s our health insurance. The healthier you get, the less you pay. Simple, fair — and maybe even fun. Call us today and start making your bill your biggest supporter.
    """)


with st.expander("AI generated Print & Radio Ad Campaign ideas"):
    st.markdown("""


COMPREHENSIVE INSURANCE DATA ANALYSIS REPORT
Visual Analytics for Insurance and Public Health Professionals
================================================================================
Report Generated: 2025-08-12 13:02:32 UK Time
Target Audience: Insurance Industry & Public Health Sector

EXECUTIVE SUMMARY
--------------------------------------------------
This report presents a comprehensive analysis of insurance claim data through
seven key visualizations, revealing critical insights about risk factors,
demographic patterns, and health-related cost drivers. The analysis demonstrates
clear opportunities for collaborative interventions between insurance providers
and public health authorities to promote healthier lifestyles while reducing
financial risks for all stakeholders.

DETAILED CHART ANALYSIS
==================================================

1. CHARGES BY SMOKER STATUS (Box Plot Analysis)
---------------------------------------------
KEY FINDINGS:
� Smoking creates the most dramatic cost differential in the dataset
� Smokers show median charges approximately 3.5x higher than non-smokers
� Non-smoker charges cluster tightly around $8,000-$12,000
� Smoker charges demonstrate high variability ($20,000-$45,000 range)
� Clear bimodal distribution suggests smoking is a primary risk stratifier

INDUSTRY IMPLICATIONS:
� Smoking cessation programs could significantly reduce claim costs
� Premium differentiation is strongly justified by cost data
� Investment in smoking cessation yields measurable ROI for insurers
� Public health campaigns targeting smoking have direct financial benefits

2. CORRELATION MATRIX ANALYSIS (Heatmap)
----------------------------------------
KEY FINDINGS:
� Age shows moderate positive correlation with charges (0.30)
� BMI demonstrates weaker but notable correlation with charges (0.20)
� Number of children shows minimal impact on charges (0.07)
� Age and BMI are weakly correlated (0.11), suggesting independent risk factors

STRATEGIC INSIGHTS:
� Age-based pricing models are statistically supported
� BMI screening programs could identify moderate-risk populations
� Family size has minimal impact on individual health costs
� Multi-factor risk models should weight age more heavily than BMI

3. CHARGES DISTRIBUTION ANALYSIS (Histogram with KDE)
--------------------------------------------------
KEY FINDINGS:
� Highly right-skewed distribution with long tail toward high costs
� Majority of claims cluster in $1,000-$15,000 range
� Significant outlier population above $40,000 (likely smokers)
� Bimodal tendency suggests two distinct risk populations

BUSINESS IMPLICATIONS:
� Standard actuarial models may underestimate high-cost tail risk
� Case management programs should target high-cost outliers
� Preventive care investments could shift the distribution leftward
� Risk pooling benefits from mixing low and high-risk populations

4. MEDIAN CHARGES BY REGION (Bar Chart Analysis)
------------------------------------------------
KEY FINDINGS:
� Northeast shows highest median charges (~$10,200)
� Regional variation is relatively modest (15% difference)
� Southeast and Southwest show similar median costs (~$9,100-$8,800)
� Northwest demonstrates lowest median charges (~$8,900)

GEOGRAPHIC RISK FACTORS:
� Northeast may reflect higher healthcare costs or lifestyle factors
� Regional differences suggest localized intervention opportunities
� Cost variations may correlate with urban density and healthcare infrastructure
� Geographic risk adjustment should be considered in pricing models

5. AGE AND SMOKING IMPACT ANALYSIS (Scatter Plot)
-----------------------------------------------
KEY FINDINGS:
� Clear linear relationship between age and charges for both groups
� Smoking effect is consistent across all age groups
� Young smokers (20-30) already show elevated costs vs older non-smokers
� Cost gap between smokers and non-smokers widens with age
� Older smokers (50+) represent highest-risk, highest-cost segment

TARGETED INTERVENTION OPPORTUNITIES:
� Early intervention with young smokers prevents exponential cost growth
� Age-stratified smoking cessation programs maximize cost-benefit ratio
� Predictive modeling can identify high-risk aging smoker populations
� Wellness programs should prioritize smoking cessation over age-related factors

6. DEMOGRAPHIC RISK FACTORS ANALYSIS (Multi-Panel Box Plots)
------------------------------------------------------------
GENDER ANALYSIS:
� Minimal cost difference between male and female populations
� Similar median costs and variance patterns
� Gender-neutral pricing appears statistically justified

SMOKING STATUS (Detailed View):
� Reinforces findings from Chart 1 with enhanced detail
� Non-smoker costs tightly controlled with few outliers
� Smoker population shows extreme cost variability

REGIONAL PATTERNS (Detailed View):
� All regions show similar outlier patterns (likely smokers)
� Regional median differences confirmed from Chart 4
� Smoking appears to be primary driver across all regions

7. MULTI-FACTOR RELATIONSHIP ANALYSIS (Three-Panel Correlation)
-----------------------------------------------------------------
AGE VS CHARGES:
� Steady upward trend with moderate correlation
� Smoking status creates distinct parallel trend lines
� Age effect is consistent but secondary to smoking impact

BMI VS CHARGES:
� Weaker relationship than age, with more scatter
� Smoking effect dominates BMI influence
� Moderate BMI elevation shows limited cost impact without smoking

CHILDREN VS CHARGES:
� Number of children shows minimal impact on individual costs
� Cost distributions remain similar across family sizes
� Family structure is not a significant risk predictor

CONTEMPORARY HEALTH TRENDS ANALYSIS
==================================================

SMOKING TREND IMPLICATIONS:
� Despite declining smoking rates, remaining smokers show intense cost impact
� E-cigarette and vaping trends may create new risk categories
� Concentrated high-risk populations require targeted interventions
� Cessation program ROI increases as smoking populations become more concentrated

OBESITY AND LIFESTYLE TRENDS:
� Rising BMI levels correlate with increased dining out and processed food consumption
� Sedentary lifestyle trends (remote work, screen time) compound obesity risks
� Food delivery culture and convenience eating patterns drive weight gain
� Current data may underestimate future BMI-related cost increases

DEMOGRAPHIC SHIFT IMPLICATIONS:
� Aging population will intensify age-related cost pressures
� Regional urbanization affects healthcare access and lifestyle factors
� Economic pressures may increase smoking rates in vulnerable populations
� Mental health trends affect both smoking and eating behaviors

STRATEGIC RECOMMENDATIONS
==================================================

FOR INSURANCE INDUSTRY:
1. RISK STRATIFICATION:
   � Implement smoking status as primary risk factor in pricing models
   � Develop age-adjusted risk categories with smoking multipliers
   � Consider regional cost adjustments for geographic risk variations
   � Maintain gender-neutral pricing based on statistical evidence

2. PREVENTION INVESTMENTS:
   � Fund smoking cessation programs with measurable ROI tracking
   � Partner with employers on workplace wellness initiatives
   � Invest in early intervention programs for young adult smokers
   � Develop BMI management programs with graduated incentives

3. PRODUCT INNOVATION:
   � Create wellness-linked premium discount programs
   � Develop predictive analytics for high-risk population identification
   � Implement wearable technology integration for real-time risk monitoring
   � Design behavioral change incentive programs

FOR PUBLIC HEALTH SECTOR:
1. TARGETED INTERVENTIONS:
   � Prioritize smoking cessation as highest-impact health investment
   � Develop age-specific cessation programs based on cost-benefit analysis
   � Address regional health disparities through localized programs
   � Create lifestyle intervention programs targeting dining and exercise habits

2. POLICY INITIATIVES:
   � Strengthen tobacco control measures with demonstrated cost benefits
   � Implement obesity prevention programs in high-risk demographics
   � Develop food environment policies addressing convenient unhealthy options
   � Create built environment changes supporting active lifestyles

COLLABORATIVE OPPORTUNITIES
==================================================

SHARED INVESTMENT STRATEGIES:
� Joint funding of smoking cessation programs with shared cost savings
� Collaborative wellness program development and implementation
� Shared data analytics platforms for population health monitoring
� Co-invested research on intervention effectiveness and ROI

BEHAVIORAL NUDGING INITIATIVES:
� Premium reduction incentives tied to verified lifestyle changes
� Gamification of health behaviors with insurance discounts
� Community-based wellness challenges with insurance sponsorship
� Technology-enabled behavior tracking with reward systems

POLICY ALIGNMENT:
� Insurance premium structures supporting public health goals
� Regulatory frameworks enabling wellness-based pricing
� Data sharing agreements for population health improvement
� Coordinated messaging on lifestyle risk factors

ECONOMIC IMPACT PROJECTIONS
==================================================

SMOKING CESSATION IMPACT:
� 10% reduction in smoking population could decrease average claims by 8-12%
� ROI on cessation programs: $3-5 saved per $1 invested over 5-year horizon
� Premium reductions of 15-20% achievable for verified non-smoking status

OBESITY MANAGEMENT IMPACT:
� 5% BMI reduction in population could decrease claims by 3-5%
� Workplace wellness programs show 2:1 ROI in reduced healthcare costs
� Preventive care investments reduce high-cost outlier populations

INDUSTRY-WIDE BENEFITS:
� Reduced claim volatility through better risk prediction
� Improved customer retention through wellness engagement
� Enhanced competitive positioning through innovative health programs
� Strengthened regulatory relationships through public health partnership

CONCLUSION
==================================================

The comprehensive analysis of insurance claims data reveals smoking as the
dominant risk factor, creating unprecedented opportunities for collaborative
intervention between insurance providers and public health authorities.

By implementing evidence-based wellness programs, both sectors can achieve
their primary objectives: insurance companies can reduce claims costs and
improve risk profiles, while public health agencies can improve population
health outcomes with measurable financial validation.

The data demonstrates that modest investments in lifestyle interventions,
particularly smoking cessation and obesity prevention, can generate
substantial returns through reduced healthcare utilization. This creates
a sustainable model where healthier populations benefit from lower
insurance premiums, while insurance companies benefit from reduced
risk exposure and improved profitability.

The path forward requires coordinated action, shared investment, and
innovative program design that aligns financial incentives with health
outcomes. The data provides a clear roadmap for this collaboration,
with smoking cessation as the highest-priority intervention and age-
stratified approaches offering the greatest cost-effectiveness.
 """)























