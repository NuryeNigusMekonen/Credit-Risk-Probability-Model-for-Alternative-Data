import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
# Custom transformer for customer-level aggregation
class CustomerAggregationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Aggregate features per CustomerId
        # create aggregate features 
        agg_df = X.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'std', 'count'],
            'txn_year': 'nunique',
            'txn_month': 'nunique',
            'txn_dayofweek': 'nunique',
            'txn_hour': 'nunique',
            'ProviderId': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
            'ProductCategory': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
            'ChannelId': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
            'PricingStrategy': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        })
        # Flatten multi-index columns
        agg_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in agg_df.columns]
        # Optional: Rename categorical lambda columns
        rename_map = {
            'ProviderId_<lambda>': 'ProviderId',
            'ProductCategory_<lambda>': 'ProductCategory',
            'ChannelId_<lambda>': 'ChannelId',
            'PricingStrategy_<lambda>': 'PricingStrategy'
        }
        agg_df.rename(columns=rename_map, inplace=True)
        agg_df.reset_index(inplace=True)
        return agg_df
# Main pipeline builder
def build_feature_pipeline():
    numeric_features = [
        'Amount_sum', 'Amount_mean', 'Amount_std', 'Amount_count',
        'txn_year_nunique', 'txn_month_nunique', 'txn_dayofweek_nunique', 'txn_hour_nunique'
    ]
    categorical_features = [
        'ProviderId', 'ProductCategory', 'ChannelId', 'PricingStrategy'
    ]
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    full_pipeline = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    return full_pipeline
# Feature engineering entry point
def engineer_features(df):
    # Ensure correct datetime parsing
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    # Extract datetime parts
    df['txn_year'] = df['TransactionStartTime'].dt.year
    df['txn_month'] = df['TransactionStartTime'].dt.month
    df['txn_dayofweek'] = df['TransactionStartTime'].dt.dayofweek
    df['txn_hour'] = df['TransactionStartTime'].dt.hour
    # Aggregate features
    transformer = CustomerAggregationTransformer()
    df_agg = transformer.fit_transform(df)
    return df_agg
