# data_preprocessing.py
import pandas as pd

# The 'upstream' and 'product' objects are injected by Ploomber
# 'product' is a dictionary containing the path to the output file
def load_and_preprocess_data(product):
    """Loads, cleans, and saves the preprocessed DataFrame."""
    try:
        df = pd.read_csv("insurance.csv")
    except FileNotFoundError:
        raise FileNotFoundError("Error: 'insurance.csv' not found. "
                                "Please make sure the file is in the same directory.")

    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    for col in ["sex", "smoker", "region"]:
        df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()

    df_clean["age"] = pd.to_numeric(df_clean["age"], errors="coerce").astype(int)
    df_clean["bmi"] = pd.to_numeric(df_clean["bmi"], errors="coerce")
    df_clean["children"] = pd.to_numeric(df_clean["children"], errors="coerce").astype(int)
    df_clean["charges"] = pd.to_numeric(df_clean["charges"], errors="coerce")

    # Save the cleaned data to the path specified in the pipeline.yaml
    df_clean.to_csv(str(product['data']), index=False)

# Call the function with the Ploomber-injected 'product' object
load_and_preprocess_data(product)
