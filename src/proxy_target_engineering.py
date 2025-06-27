import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def compute_rfm_features(df, snapshot_date=None):
    #Compute RFM (Recency, Frequency, Monetary) features per CustomerId.
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'CustomerId': 'count',
        'Amount': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm.reset_index(inplace=True)
    return rfm
def cluster_rfm_customers(rfm_df, n_clusters=3, random_state=42):
    #Scale and cluster customers based on RFM values using KMeans.
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df['rfm_cluster'] = kmeans.fit_predict(rfm_scaled)
    return rfm_df

def assign_high_risk_cluster(rfm_df):
        #Assign high-risk label based on the cluster with low Frequency and low Monetary.
    
    cluster_summary = rfm_df.groupby('rfm_cluster')[['Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_summary.sort_values(by=['Frequency', 'Monetary']).index[0]
    rfm_df['is_high_risk'] = (rfm_df['rfm_cluster'] == high_risk_cluster).astype(int)
    return rfm_df[['CustomerId', 'is_high_risk']]

def create_proxy_target(df):
        #Full pipeline: computes RFM, clusters, assigns high-risk label.    
    rfm_df = compute_rfm_features(df)
    rfm_df = cluster_rfm_customers(rfm_df)
    rfm_df = assign_high_risk_cluster(rfm_df)
    return rfm_df
