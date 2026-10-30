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
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Live ML Demand Engine with Explainable AI & Alerts</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body class="bg-gray-900 text-gray-100 p-8">
        <div class="max-w-6xl mx-auto space-y-6">
            <header class="border-b border-gray-700 pb-4">
                <h1 class="text-2xl font-bold text-indigo-400">⚡ Live ML Demand Engine with Explainable AI</h1>
                <p class="text-sm text-gray-400">Real-time inventory integration, dynamic stockout alerts, and XGBoost feature attribution.</p>
            </header>

            <div class="bg-gray-800 p-6 rounded-lg border border-gray-700">
                <h2 class="text-lg font-semibold mb-3">1. Real-Time Data Integration (Sales CSV)</h2>
                <div class="flex items-center space-x-4">
                    <input type="file" id="csvFile" accept=".csv" class="text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:bg-indigo-600 file:text-white hover:file:bg-indigo-500 cursor-pointer"/>
                    <button onclick="uploadCSV()" class="bg-indigo-600 hover:bg-indigo-500 px-5 py-2 rounded-md font-semibold text-sm">Upload & Sync Sales</button>
                </div>
                <p id="uploadStatus" class="mt-2 text-sm text-gray-400"></p>
            </div>

            <div id="controlPanel" class="bg-gray-800 p-6 rounded-lg border border-gray-700 hidden space-y-4">
                <h2 class="text-lg font-semibold">2. Live Model Training & Inventory Controls</h2>
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div>
                        <label class="block text-xs font-semibold text-gray-400 mb-1">Select SKU</label>
                        <select id="skuSelect" onchange="trainModel()" class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"></select>
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-gray-400 mb-1">Forecast Days</label>
                        <input type="number" id="forecastDays" value="30" min="7" max="90" class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"/>
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-gray-400 mb-1">Lead Time (Days)</label>
                        <input type="number" id="leadTime" value="7" min="1" class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm"/>
                    </div>
                    <div class="flex items-end">
                        <button onclick="trainModel()" class="w-full bg-emerald-600 hover:bg-emerald-500 py-2 rounded font-semibold text-sm">Train & Evaluate</button>
                    </div>
                </div>

                <div class="pt-3 border-t border-gray-700 flex items-center justify-between">
                    <span class="text-xs font-semibold text-indigo-300">⚡ Live Warehouse Stock Sync:</span>
                    <div class="flex items-center space-x-2">
                        <input type="number" id="liveStockInput" placeholder="New Stock Qty" class="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-sm w-36"/>
                        <button onclick="updateLiveStock()" class="bg-purple-600 hover:bg-purple-500 px-3 py-1 rounded text-xs font-semibold">Update Live Stock</button>
                    </div>
                </div>
            </div>

            <div id="alertBanner" class="p-4 rounded-lg border hidden font-semibold text-sm"></div>

            <div id="resultsPanel" class="space-y-6 hidden">
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div class="bg-gray-800 p-4 rounded border border-gray-700">
                        <p class="text-xs text-gray-400 font-semibold">WMAPE (Model Accuracy)</p>
                        <h3 id="mWMAPE" class="text-xl font-bold text-indigo-400">--</h3>
                    </div>
                    <div class="bg-gray-800 p-4 rounded border border-gray-700">
                        <p class="text-xs text-gray-400 font-semibold">Current Stock / Coverage</p>
                        <h3 id="mCurrentStock" class="text-xl font-bold text-gray-200">--</h3>
                    </div>
                    <div class="bg-gray-800 p-4 rounded border border-gray-700">
                        <p class="text-xs text-gray-400 font-semibold">Calculated Safety Stock</p>
                        <h3 id="mSS" class="text-xl font-bold text-emerald-400">--</h3>
                    </div>
                    <div class="bg-gray-800 p-4 rounded border border-gray-700">
                        <p class="text-xs text-gray-400 font-semibold">Reorder Point (ROP)</p>
                        <h3 id="mROP" class="text-xl font-bold text-amber-400">--</h3>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div class="md:col-span-2 bg-gray-800 p-6 rounded-lg border border-gray-700">
                        <h3 class="text-lg font-semibold mb-4">Historical Demand vs Live XGBoost Forecast</h3>
                        <div class="h-80">
                            <canvas id="forecastChart"></canvas>
                        </div>
                    </div>

                    <div class="bg-gray-800 p-6 rounded-lg border border-gray-700">
                        <h3 class="text-lg font-semibold mb-2 text-indigo-300">💡 Explainable AI (XAI)</h3>
                        <p class="text-xs text-gray-400 mb-4">Feature importance ranking explaining why the model predicted these values:</p>
                        <div id="xaiList" class="space-y-3"></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let chartInstance = null;

            async function uploadCSV() {
                const fileInput = document.getElementById('csvFile');
                if (!fileInput.files[0]) return alert("Select a CSV file first.");
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                document.getElementById('uploadStatus').innerText = "Uploading & processing data...";

                const res = await fetch('/api/upload-csv', { method: 'POST', body: formData });
                const data = await res.json();

                if (res.ok) {
                    document.getElementById('uploadStatus').innerText = `Synced ${data.total_rows} records across ${data.skus.length} SKUs.`;
                    document.getElementById('skuSelect').innerHTML = data.skus.map(s => `<option value="${s}">${s}</option>`).join('');
                    document.getElementById('controlPanel').classList.remove('hidden');
                    trainModel();
                } else {
                    document.getElementById('uploadStatus').innerText = `Error: ${data.detail}`;
                }
            }

            async function updateLiveStock() {
                const sku = document.getElementById('skuSelect').value;
                const newStock = document.getElementById('liveStockInput').value;
                if (!newStock) return alert("Enter a stock quantity.");

                await fetch(`/api/update-stock?product_id=${sku}&new_stock=${newStock}`, { method: 'POST' });
                trainModel();
            }

            async function trainModel() {
                const sku = document.getElementById('skuSelect').value;
                const days = document.getElementById('forecastDays').value;
                const leadTime = document.getElementById('leadTime').value;

                const res = await fetch(`/api/train-sku/${sku}?forecast_days=${days}&lead_time_days=${leadTime}`);
                const data = await res.json();
                if (!res.ok) return alert(data.detail);

                document.getElementById('resultsPanel').classList.remove('hidden');

                const alertBanner = document.getElementById('alertBanner');
                alertBanner.classList.remove('hidden', 'bg-red-900/50', 'border-red-600', 'text-red-200', 'bg-amber-900/50', 'border-amber-600', 'text-amber-200', 'bg-emerald-900/50', 'border-emerald-600', 'text-emerald-200');
                
                if (data.inventory.alert_level === 'CRITICAL') {
                    alertBanner.classList.add('bg-red-900/50', 'border-red-600', 'text-red-200');
                    alertBanner.innerHTML = `🚨 <strong>CRITICAL STOCKOUT RISK:</strong> ${data.inventory.alert_message}`;
                } else if (data.inventory.alert_level === 'WARNING') {
                    alertBanner.classList.add('bg-amber-900/50', 'border-amber-600', 'text-amber-200');
                    alertBanner.innerHTML = `⚠️ <strong>LOW STOCK WARNING:</strong> ${data.inventory.alert_message}`;
                } else {
                    alertBanner.classList.add('bg-emerald-900/50', 'border-emerald-600', 'text-emerald-200');
                    alertBanner.innerHTML = `✅ <strong>OPTIMAL INVENTORY:</strong> ${data.inventory.alert_message}`;
                }

                document.getElementById('mWMAPE').innerText = `${data.evaluation.wmape_percent}%`;
                document.getElementById('mCurrentStock').innerText = `${data.inventory.current_stock} units (${data.inventory.days_of_supply}d)`;
                document.getElementById('mSS').innerText = `${data.inventory.safety_stock} units`;
                document.getElementById('mROP').innerText = `${data.inventory.reorder_point} units`;
                document.getElementById('liveStockInput').value = data.inventory.current_stock;

                const xaiList = document.getElementById('xaiList');
                xaiList.innerHTML = data.explainable_ai.map(item => `
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span class="text-gray-300 font-medium">${item.feature}</span>
                            <span class="text-indigo-400 font-semibold">${item.importance_pct}%</span>
                        </div>
                        <div class="w-full bg-gray-700 rounded-full h-2">
                            <div class="bg-indigo-500 h-2 rounded-full" style="width: ${item.importance_pct}%"></div>
                        </div>
                    </div>
                `).join('');

                const histDates = data.history.map(d => d.date);
                const histSales = data.history.map(d => d.sales);
                const fcDates = data.forecast.map(d => d.date);
                const fcSales = data.forecast.map(d => d.forecast);

                const labels = [...histDates, ...fcDates];
                const histDataset = [...histSales, ...Array(fcDates.length).fill(null)];
                const fcDataset = [...Array(histDates.length - 1).fill(null), histSales[histSales.length - 1], ...fcSales];

                const ctx = document.getElementById('forecastChart').getContext('2d');
                if (chartInstance) chartInstance.destroy();

                chartInstance = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [
                            { label: 'Historical Sales', data: histDataset, borderColor: '#6366f1', backgroundColor: 'rgba(99, 102, 241, 0.1)', fill: true },
                            { label: 'XGBoost Forecast', data: fcDataset, borderColor: '#10b981', borderDash: [5, 5], backgroundColor: 'rgba(16, 185, 129, 0.1)', fill: true }
                        ]
                    },
                    options: { responsive: true, maintainAspectRatio: false }
                });
            }
        </script>
    </body>
    </html>
    """


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