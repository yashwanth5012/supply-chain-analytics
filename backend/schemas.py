# backend/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class SingleSKURequest(BaseModel):
    product_category: str = Field(default="Electronic accessories", description="Product category for seasonality tuning")
    lag_1: float = Field(default=18.0, ge=0.1, description="Most recent recorded daily demand")
    unit_price: float = Field(default=55.0, ge=0.5, description="Unit sales price in USD")
    current_stock: float = Field(default=120.0, ge=0.0, description="Current on-hand inventory count")
    lead_time_days: int = Field(default=7, ge=1, le=90, description="Supplier replenishment lead time in days")
    forecast_horizon: int = Field(default=14, ge=7, le=60, description="Forward forecasting horizon in days")
    service_level: float = Field(default=0.95, ge=0.80, le=0.999, description="Target cycle service level (e.g. 0.95 for 95%)")

class ScenarioSimulationRequest(BaseModel):
    base_demand: float = Field(default=18.0, ge=0.5, description="Baseline average daily demand")
    current_stock: float = Field(default=120.0, ge=0.0, description="Current on-hand stock")
    unit_price: float = Field(default=55.0, ge=1.0, description="Unit price in USD")
    lead_time_days: float = Field(default=7.0, ge=1.0, description="Baseline lead time in days")
    demand_shock_pct: float = Field(default=25.0, description="Simulated demand surge/drop percentage (e.g., +25% or -20%)")
    lead_time_shift_days: float = Field(default=3.0, description="Simulated supplier delay in days (e.g., +3 days)")
    service_level: float = Field(default=0.95, ge=0.80, le=0.999, description="Target service level")

class SupplierRiskRequest(BaseModel):
    supplier_name: str = Field(default="Apex Semiconductor Supply", description="Vendor / Supplier identifier")
    stated_lead_time: float = Field(default=7.0, ge=1.0, description="Contractual lead time in days")
    actual_lead_times: Optional[List[float]] = Field(default=None, description="Recent historical shipment lead times")
    on_time_in_full_rate: float = Field(default=0.88, ge=0.0, le=1.0, description="On-Time In-Full (OTIF) fulfillment rate")
    single_source: bool = Field(default=True, description="Whether this supplier is a single point of failure")

class SKUItem(BaseModel):
    sku: str
    name: Optional[str] = None
    sales_history: List[float]
    unit_price: float = 25.0

class ABCXYZRequest(BaseModel):
    items: List[SKUItem]

class SystemHealthResponse(BaseModel):
    status: str
    engine: str
    version: str
    active_models: List[str]
    capabilities: List[str]
