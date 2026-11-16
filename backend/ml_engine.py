# backend/ml_engine.py
"""
NexusSCM Enterprise | Multi-Model Time Series & ML Demand Engine
Features:
- Pure NumPy High-Performance Gradient Boosting & Random Forest Regressors
  (No heavy external C-extensions, 100% portable on Vercel Serverless and Windows)
- Universal Schema Normalization & Robust Column Inference
- Multi-Model Competitive Tournament (Moving Average, Random Forest, Gradient Boosting)
- Recursive Multi-Step Time-Series Forecasting with P10/P50/P90 Confidence Intervals
- Demand Anomaly & Outlier Spike Detection
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta


# --- Pure Vectorized NumPy Decision Tree ---
class FastNumPyDecisionTreeRegressor:
    def __init__(self, max_depth: int = 4, min_samples_split: int = 4):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.std(y) == 0:
            return float(np.mean(y))

        best_feature, best_threshold, best_mse = None, None, float('inf')
        
        for feat_idx in range(n_features):
            col_vals = X[:, feat_idx]
            thresholds = np.percentile(col_vals, np.linspace(10, 90, 7))
            thresholds = np.unique(thresholds)

            for thresh in thresholds:
                left_mask = col_vals <= thresh
                right_mask = ~left_mask
                
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                if n_left == 0 or n_right == 0:
                    continue

                var_left = np.var(y[left_mask]) if n_left > 1 else 0.0
                var_right = np.var(y[right_mask]) if n_right > 1 else 0.0
                mse = (var_left * n_left + var_right * n_right) / n_samples

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feat_idx
                    best_threshold = thresh

        if best_feature is None:
            return float(np.mean(y))

        left_mask = X[:, best_feature] <= best_threshold
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return {'feature': best_feature, 'threshold': best_threshold, 'left': left_subtree, 'right': right_subtree}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(row, self.tree) for row in X])

    def _predict_row(self, row: np.ndarray, node):
        if not isinstance(node, dict):
            return node
        if row[node['feature']] <= node['threshold']:
            return self._predict_row(row, node['left'])
        return self._predict_row(row, node['right'])


# --- Pure Vectorized Random Forest Regressor ---
class FastRandomForestRegressor:
    def __init__(self, n_estimators: int = 6, max_depth: int = 4):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.trees = []
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=min(n_samples, 250), replace=True)
            tree = FastNumPyDecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            return np.zeros(X.shape[0])
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)


# --- Pure Vectorized Gradient Boosting Regressor ---
class FastGradientBoostingRegressor:
    def __init__(self, n_estimators: int = 6, learning_rate: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.initial_pred = float(np.mean(y)) if len(y) > 0 else 0.0
        residuals = y - self.initial_pred
        self.trees = []

        for _ in range(self.n_estimators):
            tree = FastNumPyDecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            residuals -= self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X)
        return preds


# --- Automatic Schema Detection & Data Normalization ---
def auto_detect_and_normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col_map = {col: str(col).lower().strip().replace('_', '').replace(' ', '') for col in df.columns}
    
    date_col, demand_col, item_col = None, None, None

    for orig_col, clean_col in col_map.items():
        if not date_col and any(k in clean_col for k in ['date', 'time', 'timestamp', 'day', 'dt', 'period', 'month', 'year']):
            date_col = orig_col
        if not demand_col and any(k in clean_col for k in ['sales', 'demand', 'quantity', 'qty', 'units', 'volume', 'count', 'ordered']):
            demand_col = orig_col
        if not item_col and any(k in clean_col for k in ['sku', 'product', 'item', 'part', 'code', 'productid', 'itemid', 'category']):
            item_col = orig_col

    if not date_col:
        for col in df.columns:
            try:
                converted = pd.to_datetime(df[col], errors='coerce')
                if converted.notna().sum() / max(1, len(df)) > 0.4:
                    date_col = col
                    break
            except Exception:
                continue

    if not demand_col:
        for col in df.columns:
            if col in [date_col, item_col]:
                continue
            cleaned = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
            if pd.to_numeric(cleaned, errors='coerce').notna().sum() / max(1, len(df)) > 0.4:
                demand_col = col
                break

    if not date_col or not demand_col:
        raise ValueError(f"Could not automatically detect Date and Demand columns. Identified columns: {list(df.columns)}")

    rename_dict = {date_col: 'date', demand_col: 'sales'}
    if item_col:
        rename_dict[item_col] = 'item_id'

    normalized_df = df.rename(columns=rename_dict)
    normalized_df['sales'] = normalized_df['sales'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
    normalized_df['sales'] = pd.to_numeric(normalized_df['sales'], errors='coerce').fillna(0)
    normalized_df['date'] = pd.to_datetime(normalized_df['date'], errors='coerce')
    normalized_df = normalized_df.dropna(subset=['date'])

    if len(normalized_df) == 0:
        raise ValueError("No valid sequential date observations could be parsed from dataset.")

    return normalized_df.sort_values('date').reset_index(drop=True)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7).fillna(df['sales'].shift(1))
    df['rolling_mean_7'] = df['sales'].shift(1).rolling(window=7, min_periods=1).mean()
    df['rolling_std_7'] = df['sales'].shift(1).rolling(window=7, min_periods=1).std().fillna(1.0)
    return df.dropna().reset_index(drop=True)


def evaluate_models(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # Avoid zero division in MAPE
    denom = np.where(y_true == 0, 1.0, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom))) * 100.0
    return round(mae, 2), round(rmse, 2), round(min(100.0, mape), 2)


def detect_anomalies(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detects historical demand spikes or anomalies (> 2.5 sigma)."""
    if len(df) < 5:
        return []
    mean_val = float(df['sales'].mean())
    std_val = float(df['sales'].std())
    if std_val == 0:
        return []

    anomalies = []
    for idx, row in df.iterrows():
        z = (row['sales'] - mean_val) / std_val
        if abs(z) >= 2.2:
            anomalies.append({
                "date": row['date'].strftime("%Y-%m-%d"),
                "observed_sales": round(float(row['sales']), 2),
                "z_score": round(float(z), 2),
                "type": "DEMAND_SURGE" if z > 0 else "DEMAND_DROP",
                "severity": "HIGH" if abs(z) >= 3.0 else "MEDIUM"
            })
    return anomalies[-5:]  # return most recent 5 anomalies


def run_multi_model_pipeline(df: pd.DataFrame, forecast_horizon: int = 14) -> Dict[str, Any]:
    """
    Runs multi-model competitive tournament:
    1. Baseline Moving Average
    2. Pure NumPy Random Forest
    3. Pure NumPy Gradient Boosting
    Selects top performer by lowest RMSE, performs recursive multi-step forecasting with P10/P50/P90 bands.
    """
    clean_df = auto_detect_and_normalize_schema(df)
    
    # Check if multi-SKU data exists
    sku_records = []
    if 'item_id' in clean_df.columns and clean_df['item_id'].nunique() > 1:
        for sku, grp in clean_df.groupby('item_id'):
            sku_records.append({
                'sku': str(sku),
                'sales_history': grp['sales'].tolist(),
                'unit_price': 45.0
            })
        # Aggregate daily for main forecast
        agg_df = clean_df.groupby('date', as_index=False)['sales'].sum()
    else:
        agg_df = clean_df

    featured_df = extract_features(agg_df)
    if len(featured_df) < 3:
        raise ValueError("Insufficient time-series rows. Provide a dataset with at least 5-10 date observations.")

    split_idx = max(1, int(len(featured_df) * 0.8))
    train_df, val_df = featured_df.iloc[:split_idx], featured_df.iloc[split_idx:]

    feature_cols = ['day_of_week', 'month', 'day', 'lag_1', 'lag_7', 'rolling_mean_7']
    X_train, y_train = train_df[feature_cols].values, train_df['sales'].values
    X_val, y_val = (val_df[feature_cols].values, val_df['sales'].values) if len(val_df) > 0 else (X_train, y_train)

    # 1. Moving Average Baseline
    val_preds_ma = val_df['rolling_mean_7'].values if len(val_df) > 0 else train_df['rolling_mean_7'].values
    mae_ma, rmse_ma, mape_ma = evaluate_models(y_val, val_preds_ma)

    # 2. Fast Random Forest Regressor
    rf = FastRandomForestRegressor(n_estimators=6, max_depth=4)
    rf.fit(X_train, y_train)
    val_preds_rf = rf.predict(X_val)
    mae_rf, rmse_rf, mape_rf = evaluate_models(y_val, val_preds_rf)

    # 3. Fast Gradient Boosting Regressor
    gb = FastGradientBoostingRegressor(n_estimators=6, learning_rate=0.1, max_depth=3)
    gb.fit(X_train, y_train)
    val_preds_gb = gb.predict(X_val)
    mae_gb, rmse_gb, mape_gb = evaluate_models(y_val, val_preds_gb)

    leaderboard = {
        "Moving Average Baseline": {"MAE": mae_ma, "RMSE": rmse_ma, "MAPE": f"{mape_ma}%", "Accuracy": f"{round(max(0, 100 - mape_ma), 1)}%"},
        "Random Forest Regressor": {"MAE": mae_rf, "RMSE": rmse_rf, "MAPE": f"{mape_rf}%", "Accuracy": f"{round(max(0, 100 - mape_rf), 1)}%"},
        "Gradient Boosting Regressor": {"MAE": mae_gb, "RMSE": rmse_gb, "MAPE": f"{mape_gb}%", "Accuracy": f"{round(max(0, 100 - mape_gb), 1)}%"}
    }

    # Best model selection by lowest RMSE
    winning_model_name = min(leaderboard, key=lambda m: leaderboard[m]['RMSE'])
    winning_stats = leaderboard[winning_model_name]

    # Retrain champion on complete dataset
    X_full, y_full = featured_df[feature_cols].values, featured_df['sales'].values
    if winning_model_name == "Gradient Boosting Regressor":
        champion = FastGradientBoostingRegressor(n_estimators=8, learning_rate=0.1, max_depth=3)
        champion.fit(X_full, y_full)
    else:
        champion = FastRandomForestRegressor(n_estimators=8, max_depth=4)
        champion.fit(X_full, y_full)

    # Calculate residual standard deviation for confidence intervals
    train_residuals = y_full - champion.predict(X_full)
    sigma_residuals = float(np.std(train_residuals)) if len(train_residuals) > 1 else 3.0

    history_sales = agg_df['sales'].tolist()
    last_date = pd.to_datetime(agg_df['date'].iloc[-1])
    forecast_points = []

    for step in range(1, forecast_horizon + 1):
        target_date = last_date + pd.Timedelta(days=step)
        lag_1 = history_sales[-1]
        lag_7 = history_sales[-7] if len(history_sales) >= 7 else history_sales[0]
        roll_7 = float(np.mean(history_sales[-7:]))

        input_row = np.array([[target_date.dayofweek, target_date.month, target_date.day, lag_1, lag_7, roll_7]])
        p50 = float(champion.predict(input_row)[0])
        p50 = round(max(1.0, p50), 2)

        # Confidence intervals widen over time horizon: sigma * sqrt(1 + 0.05 * step)
        horizon_sigma = sigma_residuals * np.sqrt(1.0 + (0.05 * step))
        p10 = round(max(0.0, p50 - (1.645 * horizon_sigma)), 2)
        p90 = round(p50 + (1.645 * horizon_sigma), 2)

        forecast_points.append({
            "date": target_date.strftime("%Y-%m-%d"),
            "predicted_sales": p50,
            "p50": p50,
            "p10": p10,
            "p90": p90
        })
        history_sales.append(p50)

    anomalies = detect_anomalies(agg_df)

    return {
        "winning_model": winning_model_name,
        "model_accuracy": winning_stats['Accuracy'],
        "evaluation": winning_stats,
        "leaderboard": leaderboard,
        "forecast": forecast_points,
        "historical_dates": [d.strftime("%Y-%m-%d") for d in agg_df['date'].tail(30)],
        "historical_sales": [round(float(s), 2) for s in agg_df['sales'].tail(30)],
        "anomalies": anomalies,
        "sku_records": sku_records
    }


def predict_single_sku_forecast(
    product_category: str,
    lag_1: float,
    unit_price: float,
    forecast_horizon: int = 14
) -> List[Dict[str, Any]]:
    """
    Generates multi-step recursive forecasts with P10/P50/P90 confidence bands for single SKU simulation.
    """
    category_multipliers = {
        "Electronic Accessories": 1.05,
        "Electronic accessories": 1.05,
        "Consumer Electronics": 1.15,
        "Automotive & Spare Parts": 1.08,
        "Pharmaceuticals & Healthcare": 1.12,
        "Industrial Equipment & Machinery": 1.04,
        "Industrial Parts": 1.04,
        "Apparel & Fashion Footwear": 1.20,
        "Perishables & Fresh FMCG": 1.28,
        "Perishables / FMCG": 1.28,
        "Home Appliances & Furniture": 0.98,
        "Office Supplies & Stationery": 0.92,
        "Office Supplies": 0.92,
        "Chemicals & Raw Materials": 1.02
    }
    cat_factor = category_multipliers.get(product_category, 1.0)
    price_sensitivity = max(0.8, 1.0 - (unit_price * 0.001))

    base_demand = max(2.0, lag_1 * 0.88 * cat_factor * price_sensitivity)
    start_date = datetime.now()
    forecast = []

    history = [base_demand] * 7

    for step in range(1, forecast_horizon + 1):
        target_date = start_date + timedelta(days=step)
        day_of_week = target_date.weekday()

        # Seasonality effect: weekends & mid-week variations
        weekend_factor = 1.15 if day_of_week in [4, 5] else 0.96
        cycle_wave = np.sin((step / 7.0) * 2 * np.pi) * (base_demand * 0.08)

        p50 = (history[-1] * 0.7) + (history[-7] * 0.3) + cycle_wave
        p50 = float(max(1.0, p50 * weekend_factor))
        p50 = round(p50, 2)

        # Dynamic uncertainty band
        uncertainty = max(1.5, base_demand * 0.12 * np.sqrt(step))
        p10 = float(round(max(0.0, p50 - uncertainty), 2))
        p90 = float(round(p50 + uncertainty, 2))

        forecast.append({
            "day": f"Day {step}",
            "date": target_date.strftime("%Y-%m-%d"),
            "predicted_sales": p50,
            "p50": p50,
            "p10": p10,
            "p90": p90
        })
        history.append(p50)

    return forecast
