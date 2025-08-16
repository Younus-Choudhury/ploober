# model_training.py
import pandas as pd
import numpy as np
import joblib # The missing import
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model(upstream, product):
    """
    Trains and returns the Random Forest model pipeline, along with
    the performance metrics.
    """
    df_clean = pd.read_csv(str(upstream['data_preprocessing']['data']))

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

    joblib.dump(rf_pipeline, str(product['model']))

    metrics = {
        'r2_score': r2,
        'rmse': rmse,
    }
    with open(str(product['metrics']), 'w') as f:
        json.dump(metrics, f)

train_model(upstream, product)
