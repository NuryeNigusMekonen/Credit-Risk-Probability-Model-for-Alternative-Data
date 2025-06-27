from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

def log_transform(x):
    # Clip negative and missing values
    x = np.nan_to_num(x, nan=0.0)
    return np.log1p(np.clip(x, 0, None))

def get_preprocessing_pipeline(numeric_features, categorical_features):
    # Numeric pipeline
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('log', FunctionTransformer(log_transform, validate=False)),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    return preprocessor

def preprocess_raw_data(df):
    # Drop columns based on business and EDA insights
    drop_cols = [
        'TransactionId', 'BatchId', 'SubscriptionId',
        'Value', 'CurrencyCode', 'CountryCode'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')
    # Convert transaction start time to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    return df
