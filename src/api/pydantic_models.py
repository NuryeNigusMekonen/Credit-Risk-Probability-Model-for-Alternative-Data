from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    Amount_sum: float
    Amount_mean: float
    Amount_std: float
    Amount_count: int
    txn_year_nunique: int
    txn_month_nunique: int
    txn_dayofweek_nunique: int
    txn_hour_nunique: int
    ProviderId: str
    ProductCategory: str
    ChannelId: str
    PricingStrategy: str

class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: int  
