from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    Amount_sum: float = Field(..., example=2500.5)
    Amount_mean: float = Field(..., example=125.0)
    Amount_std: float = Field(..., example=15.5)
    Amount_count: float = Field(..., example=20.0)
    txn_year_nunique: int = Field(..., example=1)
    txn_month_nunique: int = Field(..., example=2)
    txn_dayofweek_nunique: int = Field(..., example=5)
    txn_hour_nunique: int = Field(..., example=10)
    ProviderId: str = Field(..., example="ProviderId_4")
    ProductCategory: str = Field(..., example="airtime")
    ChannelId: str = Field(..., example="ChannelId_2")
    PricingStrategy: str = Field(..., example="2")

class PredictionResponse(BaseModel):
    risk_probability: float = Field(..., example=0.85)
