import io
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from xgboost import XGBRegressor

app = FastAPI(title="Live ML Supply Chain Engine with Explainable AI & Automated Alerts")

# Global state for Real-Time Data Integration
DATA_STORE = {
    "df": None,
    "current_stock": {}  # Product ID -> Current Live Inventory
}

# Pure Python / NumPy metrics
def calc_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def calc_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# ==========================================
# 1. MACHINE LEARNING & EXPLAINABLE AI ENGINE
# ==========================================

def train_and_forecast_sku(product_id, df_sku, current_stock_qty=None, forecast_days=30, lead_time_days=7, lead_time_std=2, service_level_z=1.65):
    df = df_sku.copy().sort_values('date')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Feature Engineering
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['lag_1'] = df['quantity_sold'].shift(1)
    df['lag_7'] = df['quantity_sold'].shift(7)
    df['rolling_7_mean'] = df['quantity_sold'].shift(1).rolling(7).mean()
    df['rolling_30_mean'] = df['quantity_sold'].shift(1).rolling(30).mean()

    df_clean = df.dropna()
    if len(df_clean) < 14:
        raise ValueError("Insufficient data history (at least 14 days required).")

    # Train/Validation Split (80/20)
    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx]
    val_df = df_clean.iloc[split_idx:]

    feature_cols = ['dayofweek', 'month', 'day', 'lag_1', 'lag_7', 'rolling_7_mean', 'rolling_30_mean']
    feature_labels = {
        'dayofweek': 'Day of Week Seasonality',
        'month': 'Monthly Trend',
        'day': 'Day of Month',
        'lag_1': 'Yesterday Sales (Lag 1)',
        'lag_7': 'Last Week Sales (Lag 7)',
        'rolling_7_mean': '7-Day Rolling Average',
        'rolling_30_mean': '30-Day Rolling Average'
    }

    X_train, y_train = train_df[feature_cols], train_df['quantity_sold']
    X_val, y_val = val_df[feature_cols], val_df['quantity_sold']

    # Train XGBoost Model
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # Explainable AI (XAI)
    importance_scores = model.feature_importances_
    total_importance = float(np.sum(importance_scores)) if np.sum(importance_scores) > 0 else 1.0
    xai_explanations = [
        {
            "feature": feature_labels[col],
            "importance_pct": round(float(score / total_importance) * 100, 1)
        }
        for col, score in zip(feature_cols, importance_scores)
    ]
    xai_explanations = sorted(xai_explanations, key=lambda x: x['importance_pct'], reverse=True)

    # Evaluate Accuracy
    val_preds = model.predict(X_val)
    mae = calc_mae(y_val, val_preds)
    rmse = calc_rmse(y_val, val_preds)
    actual_sum = y_val.sum()
    wmape = (np.sum(np.abs(y_val - val_preds)) / actual_sum * 100) if actual_sum > 0 else 0.0

    # Recursive Forecast Generation
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecasts = []
    recent_history = list(df['quantity_sold'].values)

    for new_date in future_dates:
        lag_1 = recent_history[-1]
        lag_7 = recent_history[-7] if len(recent_history) >= 7 else lag_1
        r7 = np.mean(recent_history[-7:])
        r30 = np.mean(recent_history[-30:]) if len(recent_history) >= 30 else r7

        feat = np.array([[new_date.dayofweek, new_date.month, new_date.day, lag_1, lag_7, r7, r30]])
        pred = max(0, float(model.predict(feat)[0]))
        forecasts.append({"date": new_date.strftime("%Y-%m-%d"), "forecast": round(pred, 2)})
        recent_history.append(pred)

    # Inventory Control
    avg_daily_demand = float(df['quantity_sold'].mean())
    std_daily_demand = float(df['quantity_sold'].std()) if len(df) > 1 else 0.0

    safety_stock = int(np.ceil(service_level_z * np.sqrt(
        (lead_time_days * (std_daily_demand ** 2)) +
        ((avg_daily_demand ** 2) * (lead_time_std ** 2))
    )))
    reorder_point = int(np.ceil((avg_daily_demand * lead_time_days) + safety_stock))

    stock = current_stock_qty if current_stock_qty is not None else int(avg_daily_demand * (lead_time_days + 2))
    days_of_supply = round(stock / avg_daily_demand, 1) if avg_daily_demand > 0 else 999.0

    if stock <= safety_stock:
        alert_level = "CRITICAL"
        alert_msg = f"Stock ({stock} units) is below Safety Stock ({safety_stock} units)! Immediate stockout risk."
    elif stock <= reorder_point:
        alert_level = "WARNING"
        alert_msg = f"Stock ({stock} units) hit Reorder Point ({reorder_point} units). Trigger purchase order."
    else:
        alert_level = "OPTIMAL"
        alert_msg = f"Stock ({stock} units) is optimal. Covers ~{days_of_supply} days of demand."

    return {
        "evaluation": {"mae": round(mae, 2), "rmse": round(rmse, 2), "wmape_percent": round(wmape, 2)},
        "inventory": {
            "current_stock": stock,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "days_of_supply": days_of_supply,
            "alert_level": alert_level,
            "alert_message": alert_msg
        },
        "explainable_ai": xai_explanations,
        "history": [{"date": d.strftime("%Y-%m-%d"), "sales": float(v)} for d, v in zip(df.index, df['quantity_sold'])],
        "forecast": forecasts
    }

# ==========================================
# 2. FASTAPI ENDPOINTS & ROUTING
# ==========================================

@app.post("/upload-csv")
@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        required_cols = {'date', 'product_id', 'quantity_sold'}
        if not required_cols.issubset(set(df.columns)):
            raise HTTPException(status_code=400, detail=f"CSV must contain columns: {list(required_cols)}")

        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df['quantity_sold'] = pd.to_numeric(df['quantity_sold'], errors='coerce').fillna(0)
        DATA_STORE["df"] = df
        
        skus = list(df['product_id'].unique())
        for sku in skus:
            if sku not in DATA_STORE["current_stock"]:
                avg_sales = df[df['product_id'] == sku]['quantity_sold'].mean()
                DATA_STORE["current_stock"][sku] = int(avg_sales * 10)

        return {"message": "CSV loaded", "skus": skus, "total_rows": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-stock")
@app.post("/api/update-stock")
def update_stock(product_id: str, new_stock: int):
    DATA_STORE["current_stock"][product_id] = new_stock
    return {"message": f"Updated live stock for {product_id} to {new_stock} units"}

@app.get("/train-sku/{product_id}")
@app.get("/api/train-sku/{product_id}")
def train_sku(product_id: str, forecast_days: int = 30, lead_time_days: int = 7):
    if DATA_STORE["df"] is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet.")
    df_sku = DATA_STORE["df"][DATA_STORE["df"]['product_id'] == product_id]
    if len(df_sku) == 0:
        raise HTTPException(status_code=404, detail="SKU not found.")
    
    current_qty = DATA_STORE["current_stock"].get(product_id, None)
    return train_and_forecast_sku(
        product_id, 
        df_sku, 
        current_stock_qty=current_qty, 
        forecast_days=forecast_days, 
        lead_time_days=lead_time_days
    )

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

                const res = await fetch('upload-csv', { method: 'POST', body: formData });
                const data = await res.json();

                if (res.ok) {
                    document.getElementById('uploadStatus').innerText = `Synced ${data.total_rows} records across ${data.skus.length} SKUs.`;
                    document.getElementById('skuSelect').innerHTML = data.skus.map(s => `<option value="${s}">${s}</option>`).join('');
                    document.getElementById('controlPanel').classList.remove('hidden');
                    trainModel();
                } else {
                    document.getElementById('uploadStatus').innerText = `Error: ${data.detail || 'Failed to process file'}`;
                }
            }

            async function updateLiveStock() {
                const sku = document.getElementById('skuSelect').value;
                const newStock = document.getElementById('liveStockInput').value;
                if (!newStock) return alert("Enter a stock quantity.");

                await fetch(`update-stock?product_id=${sku}&new_stock=${newStock}`, { method: 'POST' });
                trainModel();
            }

            async function trainModel() {
                const sku = document.getElementById('skuSelect').value;
                const days = document.getElementById('forecastDays').value;
                const leadTime = document.getElementById('leadTime').value;

                const res = await fetch(`train-sku/${sku}?forecast_days=${days}&lead_time_days=${leadTime}`);
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