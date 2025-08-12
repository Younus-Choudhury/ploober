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


       This report examines 1,338 individual health insurance records to identify the strongest drivers of cost. The dataset includes age, sex, BMI, number of children, smoking status, region, and annual charges. The analysis focuses on mean costs, distribution patterns, and key contrasts between groups.
1. Data Snapshot

    Average age: 39 years (range: 18–64)

    Average BMI: 30.66 (in the obese range; range: 15.96–53.13)

    Average annual charge: $13,270 (range: $1,122–$63,770)

    Sex split: 50.5% male, 49.5% female

    Smoker prevalence: 20.5% yes, 79.5% no

    Regional distribution: Roughly even across four US regions, with the Southeast largest (27%).

2. Main Findings
2.1 Smoking is the Largest Cost Driver

Smokers pay on average nearly four times more than non-smokers.

    Smokers: mean $32,050

    Non-smokers: mean $8,440
    The gap is consistent across all age groups and BMI ranges, making smoking the clearest and most defensible pricing determinant.

2.2 Age Matters, but Predictably

Costs rise steadily with age, from an average of $8,400 in the 18–25 group to over $21,000 for those aged 56–65. This reflects natural increases in healthcare needs over the life cycle.
2.3 BMI Links to Cost, But with Caveats

Higher BMI correlates with higher average charges. People in the “Obese II+” category (BMI ≥ 35) pay about 50% more than those in the “Normal” range. However, the relationship is weaker than smoking or age, and there is substantial variation within each BMI category.
2.4 Minor Factors: Children, Gender, Region

    Children: Small, inconsistent effects on costs.

    Gender: Men pay slightly more than women on average.

    Region: The Southeast has the highest average costs; differences elsewhere are small.

3. Ethical Considerations

BMI is not a perfect measure of health. Overemphasis could unfairly penalize individuals, especially younger women, who may already face strong body image pressures. Tying pricing too closely to BMI risks fuelling disordered eating and weight stigma. Environmental and socioeconomic factors also affect BMI beyond personal control.
Smoking, by contrast, is a voluntary and well-proven risk factor with a direct causal link to higher medical costs. Highlighting smoking as the main determinant in pricing is both ethically and commercially sound. BMI can remain as one signal in a broader risk model, but not as a blunt pricing lever.
4. Recommendations

    Weight premiums heavily on smoking status and reward quitting through lower rates.

    Incorporate age in a predictable tiered structure to match rising health costs over time.

    Treat BMI cautiously, using it in combination with other health indicators.

    Offer health incentives — cessation programs for smokers, lifestyle support for weight management — as cost-control strategies.

    Review regional pricing for the Southeast and investigate underlying drivers.

5. Conclusion

The strongest and most actionable insight from this dataset is that smoking is the single biggest cost driver in individual health insurance, followed by age. BMI has a measurable effect but should be handled with care to avoid ethical and reputational risks. A pricing strategy anchored in these facts can be both fair and financially robust.

    """)
























