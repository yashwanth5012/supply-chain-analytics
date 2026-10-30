# backend/train.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def auto_detect_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Detects date, quantity/sales, and SKU fields from GitHub or custom datasets."""
    df = df.copy()
    col_map = {col: str(col).lower().strip().replace('_', '').replace(' ', '') for col in df.columns}
    
    date_col, demand_col, item_col = None, None, None

    # Flexible matching for GitHub datasets
    for orig_col, clean_col in col_map.items():
        if not date_col and any(k in clean_col for k in ['date', 'time', 'timestamp', 'day', 'dt', 'period', 'postdate']):
            date_col = orig_col
        if not demand_col and any(k in clean_col for k in ['sales', 'demand', 'quantity', 'qty', 'units', 'volume', 'amount', 'totalinvoiced']):
            demand_col = orig_col
        if not item_col and any(k in clean_col for k in ['sku', 'product', 'item', 'part', 'code', 'itemcode', 'category']):
            item_col = orig_col

    # Fallback date detection
    if not date_col:
        for col in df.columns:
            try:
                if pd.to_datetime(df[col], errors='coerce').notna().sum() / len(df) > 0.4:
                    date_col = col
                    break
            except Exception:
                continue

    # Fallback sales detection
    if not demand_col:
        for col in df.columns:
            if col in [date_col, item_col]:
                continue
            cleaned = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
            if pd.to_numeric(cleaned, errors='coerce').notna().sum() / len(df) > 0.4:
                demand_col = col
                break

    if not date_col or not demand_col:
        raise ValueError(f"Could not parse Date and Demand columns from headers: {list(df.columns)}")

    rename_dict = {date_col: 'date', demand_col: 'sales'}
    if item_col:
        rename_dict[item_col] = 'item_id'

    norm_df = df.rename(columns=rename_dict)
    norm_df['sales'] = norm_df['sales'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
    norm_df['sales'] = pd.to_numeric(norm_df['sales'], errors='coerce').fillna(0)
    norm_df['date'] = pd.to_datetime(norm_df['date'], errors='coerce')
    norm_df = norm_df.dropna(subset=['date'])

    # Aggregate daily observations
    if 'item_id' in norm_df.columns:
        norm_df = norm_df.groupby(['item_id', 'date'], as_index=False)['sales'].sum()
    else:
        norm_df = norm_df.groupby('date', as_index=False)['sales'].sum()

    return norm_df.sort_values('date').reset_index(drop=True)


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates time-series lag and rolling window features."""
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    df['rolling_mean_7'] = df['sales'].shift(1).rolling(window=7, min_periods=1).mean()
    return df.dropna().reset_index(drop=True)


def train_and_export_model(data_path: str, artifact_dir: str = "."):
    """Trains regressors on real dataset and serializes winning model to joblib."""
    raw_df = pd.read_excel(data_path) if data_path.endswith('.xlsx') else pd.read_csv(data_path)

    clean_df = auto_detect_schema(raw_df)
    featured_df = build_time_features(clean_df)

    if len(featured_df) < 5:
        raise ValueError("Dataset has insufficient time observations for model training.")

    feature_cols = ['day_of_week', 'month', 'day', 'lag_1', 'lag_7', 'rolling_mean_7']
    X = featured_df[feature_cols].values
    y = featured_df['sales'].values

    # Holdout validation split (80% train / 20% test)
    split_idx = max(1, int(len(featured_df) * 0.8))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    }

    leaderboard = {}
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val if len(X_val) > 0 else X_train)
        target = y_val if len(y_val) > 0 else y_train
        
        mae = float(mean_absolute_error(target, preds))
        rmse = float(np.sqrt(mean_squared_error(target, preds)))
        
        leaderboard[name] = {"MAE": round(mae, 2), "RMSE": round(rmse, 2)}
        fitted_models[name] = model

    # Winning algorithm selection
    best_name = min(leaderboard, key=lambda m: leaderboard[m]['RMSE'])
    best_model = fitted_models[best_name]

    # Retrain winner on full dataset
    best_model.fit(X, y)

    # Save .joblib artifact
    model_filepath = os.path.join(artifact_dir, "model.joblib")
    joblib.dump(best_model, model_filepath)

    # Extract multi-SKU list for ABC-XYZ Matrix
    sku_list = []
    if 'item_id' in clean_df.columns:
        for item_id, group in clean_df.groupby('item_id'):
            sku_list.append({"item_id": str(item_id), "sales_history": group['sales'].tail(30).tolist()})
    else:
        sku_list.append({"item_id": "PRIMARY-GITHUB-SKU", "sales_history": clean_df['sales'].tail(30).tolist()})

    metadata = {
        "winning_model": best_name,
        "leaderboard": leaderboard,
        "feature_columns": feature_cols,
        "last_historical_date": clean_df['date'].iloc[-1].strftime("%Y-%m-%d"),
        "recent_sales_history": clean_df['sales'].tail(30).tolist(),
        "recent_dates": clean_df['date'].tail(30).dt.strftime("%Y-%m-%d").tolist(),
        "sku_records": sku_list
    }

    metadata_filepath = os.path.join(artifact_dir, "metadata.json")
    with open(metadata_filepath, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata