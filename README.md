# NexusSCM Enterprise | Next-Gen AI Demand & Inventory Control Tower

![NexusSCM](https://img.shields.io/badge/NexusSCM-v6.0%20Enterprise-06b6d4?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-emerald?style=for-the-badge)

NexusSCM Enterprise is an enterprise-grade AI supply chain analytics and inventory optimization platform. It combines machine learning multi-horizon demand forecasting with stochastic operations research to calculate exact Safety Stock ($SS$), dynamic Reorder Points ($ROP$), Economic Order Quantities ($EOQ$), 9-Box ABC-XYZ segmentation, and real-time supplier disruption risk scoring.

---

## 🌟 Key Capabilities

### 1. ⚡ Probabilistic Multi-Horizon Demand Forecasting
- Recursive multi-step forward demand projections across 7, 14, 30, and 60-day horizons.
- **P10 / P50 / P90 Confidence Envelopes**: Captures downside, median expected, and upside demand uncertainty.
- Vectorized pure NumPy decision tree and ensemble architectures (Random Forest & Gradient Boosting Regressors) engineered for high throughput and zero external C-extension dependencies.

### 2. 📦 Stochastic Operations Research Inventory Policy
- Dual-uncertainty stochastic Safety Stock formulation:
  $$\sigma_{DLT} = \sqrt{L \cdot \sigma_D^2 + D^2 \cdot \sigma_L^2}$$
  $$SS = Z \cdot \sigma_{DLT}$$
  $$ROP = (D \cdot L) + SS$$
- **Wilson EOQ Formulation**: Balances annual ordering costs against inventory holding carrying costs.
- **Runway & Stockout Detection**: Real-time Days of Inventory (DOI) tracking and emergency reorder alert triggers (`CRITICAL_STOCKOUT_RISK`, `REORDER_TRIGGERED`, `OPTIMAL_HEALTH`, `OVERSTOCKED`).

### 3. 🎲 What-If Supply Chain Stress Simulator
- Interactive simulation of geopolitical shocks, port bottlenecks, and supplier lead-time slippage.
- Live recalculation of time-to-stockout and adjusted safety buffer requirements under custom demand shocks (-50% to +100%).

### 4. 📊 9-Box Strategic ABC-XYZ Matrix
- **ABC Analysis**: Pareto revenue contribution (Class A: top 70%, Class B: next 20%, Class C: bottom 10%).
- **XYZ Analysis**: Demand predictability via Coefficient of Variation ($CV \le 0.5$, $CV \le 1.0$, $CV > 1.0$).
- Actionable operational playbooks (e.g., Just-In-Time replenishment for AX vs make-to-order / buffering for CZ).

### 5. 🛡️ Supplier Reliability & Disruption Risk Index
- Quantitative risk scoring (0–100) factoring in delivery lead-time variance, On-Time In-Full (OTIF) rate, and single-source dependency penalties.

---

## 🏗️ Architecture

```
supply_chain_project/
├── api/
│   └── index.py             # Vercel serverless function entrypoint
├── backend/
│   ├── config.py            # Platform configuration & settings
│   ├── inventory_engine.py  # Stochastic OR, ROP, EOQ, ABC-XYZ, & Risk engine
│   ├── ml_engine.py         # Vectorized ML Regressors & Time-Series pipelines
│   ├── schemas.py           # Pydantic v2 data models & validation schemas
│   └── main.py              # FastAPI application & executive control tower UI
├── frontend/
│   └── index.html           # Standalone modern dark-mode executive dashboard
├── test_data.csv            # Representative time-series sales dataset
├── vercel.json              # Vercel serverless deployment routing
├── requirements.txt         # Production serverless dependencies
└── README.md                # Technical platform documentation
```

---

## 🚀 Quickstart & Local Execution

### 1. Clone & Setup
```bash
git clone https://github.com/yashwanth5012/supply-chain-analytics.git
cd supply-chain-analytics
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Local Server
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
Visit `http://localhost:8000` to open the executive dashboard, or `http://localhost:8000/docs` for the interactive Swagger OpenAPI docs.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health status and active ML model registry |
| `POST` | `/api/single-predict` | Single-SKU forecast with P10/P50/P90 bands & inventory policy |
| `POST` | `/api/batch-predict` | Batch CSV ingestion, multi-model tournament, & ABC-XYZ matrix |
| `POST` | `/api/scenario-simulate` | What-If supply chain stress testing and disruption simulation |
| `POST` | `/api/supplier-risk` | Vendor reliability and disruption vulnerability index (0-100) |
| `POST` | `/api/abc-xyz-matrix` | 9-Box inventory classification matrix |
| `GET` | `/api/sample-data` | Demo dataset for 1-click testing |

---

## 🚢 Cloud Deployment on Vercel

NexusSCM Enterprise is configured out-of-the-box for serverless deployment on Vercel:

```bash
npx vercel --prod
```

---

## 👤 Author
- **Yashwanth A** ([@yashwanth5012](https://github.com/yashwanth5012))
- Project: [supply-chain-analytics](https://github.com/yashwanth5012/supply-chain-analytics)

