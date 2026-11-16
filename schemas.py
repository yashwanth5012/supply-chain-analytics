from pydantic import BaseModel
from typing import List, Dict, Any

class ModelMetricSchema(BaseModel):
    MAE: float
    RMSE: float

class InventoryKPIsSchema(BaseModel):
    mean_daily_demand: float
    demand_std_dev: float
    service_level_target: str
    safety_stock: int
    reorder_point: int
    economic_order_quantity: int

class AnalysisResponseSchema(BaseModel):
    status: str
    total_historical_records: int
    model_evaluation: Dict[str, Any]
    inventory_recommendations: Dict[str, Any]
    abc_xyz_matrix: List[Dict[str, Any]]
    historical_data: List[Dict[str, Any]]
    forecast_data: List[Dict[str, Any]]