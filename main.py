# backend/main.py
import io
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from inventory_engine import compute_inventory_policy, compute_abc_xyz_matrix

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
MODEL_PATH = BASE_DIR / "model.joblib"
METADATA_PATH = BASE_DIR / "metadata.json"

# Global Variables for Artifacts
MODEL_PIPELINE = None
METADATA = None

def load_artifacts():
    global MODEL_PIPELINE, METADATA
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError("model.joblib or metadata.json not found. Train and save model first!")
    
    MODEL_PIPELINE = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        METADATA = json.load(f)

# Modern Lifespan Event Handler (Replaces deprecated @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

app = FastAPI(title="NexusSCM Production Architecture API", version="5.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    # Check multiple relative paths so it works both locally and on Render
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html"),
        os.path.join(os.getcwd(), "frontend", "index.html"),
        os.path.join(os.getcwd(), "..", "frontend", "index.html")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return FileResponse(path)
            
    return HTMLResponse("<h2>Frontend index.html not found.</h2>", status_code=404)


# Request Schema for Single Analysis
class SinglePredictRequest(BaseModel):
    product_line: str = "Electronic accessories"
    lag_1: float = 18.0
    lag_2: float = 16.0
    lag_7: float = 14.0
    rolling_mean_3: float = 17.0
    rolling_mean_7: float = 15.5
    rolling_std_7: float = 2.1
    ema_7: float = 16.2
    price_per_unit: float = 55.0
    avg_rating: float = 7.2
    cogs_ratio: float = 0.95
    horizon: int = 14
    lead_time_days: float = 7.0
    lead_time_std: float = 1.5
    service_level: float = 0.95
    holding_cost: float = 2.5
    order_cost: float = 50.0
    current_stock: float = 120.0


# --- 1. SINGLE ANALYSIS ENDPOINT (Using Saved joblib Model) ---
@app.post("/api/predict/single")
def predict_single(req: SinglePredictRequest):
    if MODEL_PIPELINE is None:
        load_artifacts()

    input_data = pd.DataFrame([{
        'day_of_week': 2,
        'month': 10,
        'day_of_month': 15,
        'is_weekend': 0,
        'lag_1': req.lag_1,
        'lag_2': req.lag_2,
        'lag_7': req.lag_7,
        'rolling_mean_3': req.rolling_mean_3,
        'rolling_mean_7': req.rolling_mean_7,
        'rolling_std_7': req.rolling_std_7,
        'ema_7': req.ema_7,
        'price_per_unit': req.price_per_unit,
        'avg_rating': req.avg_rating,
        'cogs_ratio': req.cogs_ratio,
        'demand_momentum': req.lag_1 / (req.rolling_mean_7 + 1e-5),
        'product_line': req.product_line
    }])

    predictions = []
    curr_lag1 = req.lag_1
    curr_roll7 = req.rolling_mean_7
    
    for i in range(1, req.horizon + 1):
        input_data['lag_1'] = curr_lag1
        input_data['rolling_mean_7'] = curr_roll7
        
        pred_val = float(MODEL_PIPELINE.predict(input_data)[0])
        pred_val = round(max(0.0, pred_val), 2)
        
        predictions.append({"date": f"Day +{i}", "predicted_sales": pred_val})
        curr_lag1 = pred_val
        curr_roll7 = (curr_roll7 * 6 + pred_val) / 7.0

    synthetic_history = [req.lag_7, req.lag_2, req.lag_1, req.rolling_mean_3, req.rolling_mean_7]
    inv_kpis = compute_inventory_policy(
        demand_history=synthetic_history,
        lead_time_days=req.lead_time_days,
        lead_time_std=req.lead_time_std,
        service_level=req.service_level,
        order_cost=req.order_cost,
        holding_cost_unit_year=req.holding_cost
    )

    stockout_alert = req.current_stock <= inv_kpis['reorder_point']
    inv_payload = {
        **inv_kpis,
        "capital_at_risk": round(inv_kpis['safety_stock'] * req.price_per_unit, 2),
        "current_stock": req.current_stock,
        "stockout_alert": stockout_alert,
        "recommended_order_quantity": inv_kpis['economic_order_quantity'] if stockout_alert else 0
    }

    return {
        "status": "success",
        "analysis_type": "Single Item Prediction",
        "model_used": METADATA["winning_model"],
        "accuracy_score": METADATA["leaderboard"][METADATA["winning_model"]]["Accuracy"],
        "inventory_kpis": inv_payload,
        "forecast_data": predictions
    }


# --- 2. BATCH ANALYSIS ENDPOINT (Using Saved joblib Model) ---
@app.post("/api/predict/batch")
async def predict_batch(
    file: UploadFile = File(...),
    horizon: int = 14,
    service_level: float = 0.95,
    lead_time_days: float = 7.0,
    lead_time_std: float = 1.5,
    order_cost: float = 50.0,
    holding_cost: float = 2.5,
    current_stock: float = 500.0
):
    if MODEL_PIPELINE is None:
        load_artifacts()

    contents = await file.read()
    try:
        df = pd.read_excel(io.BytesIO(contents)) if file.filename.endswith('.xlsx') else pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File parsing error: {str(e)}")

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce') if 'Date' in df.columns else pd.to_datetime(df.iloc[:, 0], errors='coerce')
    target_col = 'Quantity' if 'Quantity' in df.columns else ('Total' if 'Total' in df.columns else df.columns[1])
    df['sales'] = pd.to_numeric(df[target_col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False), errors='coerce').fillna(0)
    
    clean_df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    sales_history = clean_df['sales'].tail(30).tolist()
    date_history = clean_df['Date'].tail(30).dt.strftime("%Y-%m-%d").tolist()

    last_val = sales_history[-1]
    mean_val = float(np.mean(sales_history))
    std_val = float(np.std(sales_history)) if len(sales_history) > 1 else 1.0

    batch_input = pd.DataFrame([{
        'day_of_week': 1,
        'month': 10,
        'day_of_month': 15,
        'is_weekend': 0,
        'lag_1': last_val,
        'lag_2': sales_history[-2] if len(sales_history) > 1 else last_val,
        'lag_7': sales_history[-7] if len(sales_history) >= 7 else last_val,
        'rolling_mean_3': float(np.mean(sales_history[-3:])),
        'rolling_mean_7': float(np.mean(sales_history[-7:])),
        'rolling_std_7': std_val,
        'ema_7': mean_val,
        'price_per_unit': 25.0,
        'avg_rating': 7.0,
        'cogs_ratio': 0.95,
        'demand_momentum': last_val / (mean_val + 1e-5),
        'product_line': 'Electronic accessories'
    }])

    predictions = []
    last_date = pd.to_datetime(date_history[-1])
    curr_lag = last_val

    for i in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=i)
        batch_input['lag_1'] = curr_lag
        pred_val = float(MODEL_PIPELINE.predict(batch_input)[0])
        pred_val = round(max(0.0, pred_val), 2)

        predictions.append({"date": next_date.strftime("%Y-%m-%d"), "predicted_sales": pred_val})
        curr_lag = pred_val

    inv_kpis = compute_inventory_policy(
        demand_history=sales_history,
        lead_time_days=lead_time_days,
        lead_time_std=lead_time_std,
        service_level=service_level,
        order_cost=order_cost,
        holding_cost_unit_year=holding_cost
    )

    stockout_alert = current_stock <= inv_kpis['reorder_point']
    inv_payload = {
        **inv_kpis,
        "capital_at_risk": round(inv_kpis['safety_stock'] * 25.0, 2),
        "current_stock": current_stock,
        "stockout_alert": stockout_alert,
        "recommended_order_quantity": inv_kpis['economic_order_quantity'] if stockout_alert else 0
    }

    abc_xyz = compute_abc_xyz_matrix([{'item_id': 'BATCH-SKU-01', 'sales_history': sales_history}])
    historical_series = [{"date": d, "sales": s} for d, s in zip(date_history, sales_history)]

    return {
        "status": "success",
        "analysis_type": "Batch File Analysis",
        "model_used": METADATA["winning_model"],
        "accuracy_score": METADATA["leaderboard"][METADATA["winning_model"]]["Accuracy"],
        "model_performance": {
            "winning_model": METADATA["winning_model"],
            "leaderboard": METADATA["leaderboard"]
        },
        "inventory_kpis": inv_payload,
        "abc_xyz_matrix": abc_xyz,
        "historical_data": historical_series,
        "forecast_data": predictions
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)