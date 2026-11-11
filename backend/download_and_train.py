# backend/train_and_save.py
import os
import json
import joblib
import urllib.request
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

GITHUB_URL = "https://raw.githubusercontent.com/selva86/datasets/master/supermarket_sales.csv"
LOCAL_CSV = os.path.join(os.path.dirname(__file__), "github_sales.csv")

def download_dataset():
    """Downloads real supermarket sales dataset from GitHub."""
    print(f"📥 Downloading real sales dataset from GitHub:\n   {GITHUB_URL}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(GITHUB_URL, headers=headers)
    with urllib.request.urlopen(req) as resp, open(LOCAL_CSV, 'wb') as f:
        f.write(resp.read())
    print(f"✅ Saved raw CSV to: {LOCAL_CSV}\n")

def perform_advanced_feature_engineering(csv_path: str):
    """
    Advanced Feature Engineering Strategy:
    1. Group by Date AND Product Line (Category-level granularity).
    2. Engineer Category Lags, Category EMAs, and Moving Averages.
    3. Include Price Elasticity, Average Rating, and COGS ratios.
    4. One-Hot Encode Product Line categories.
    """
    df = pd.read_csv(csv_path)

    # Standardize Column Names
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Total'] = pd.to_numeric(df['Total'], errors='coerce').fillna(0)
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
    df['Unit price'] = pd.to_numeric(df['Unit price'], errors='coerce').fillna(0)
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(0)
    df['cogs'] = pd.to_numeric(df['cogs'], errors='coerce').fillna(0)

    # Group by Date and Product line to preserve demand signal specificity
    cat_daily = df.groupby(['Date', 'Product line'], as_index=False).agg({
        'Quantity': 'sum',
        'Total': 'sum',
        'Unit price': 'mean',
        'Rating': 'mean',
        'cogs': 'sum'
    }).sort_values(['Product line', 'Date']).reset_index(drop=True)

    # Rename target variable
    cat_daily.rename(columns={'Date': 'date', 'Quantity': 'sales', 'Product line': 'product_line'}, inplace=True)

    # --- ADVANCED TIME-SERIES FEATURE GENERATION PER CATEGORY ---
    featured_groups = []

    for category, group in cat_daily.groupby('product_line'):
        g = group.copy().sort_values('date').reset_index(drop=True)

        # Calendar Signals
        g['day_of_week'] = g['date'].dt.dayofweek
        g['month'] = g['date'].dt.month
        g['day_of_month'] = g['date'].dt.day
        g['is_weekend'] = g['day_of_week'].isin([5, 6]).astype(int)

        # Category-Specific Lags (Strictly shifted by 1 to prevent data leakage)
        g['lag_1'] = g['sales'].shift(1)
        g['lag_2'] = g['sales'].shift(2)
        g['lag_7'] = g['sales'].shift(7)

        # Category Moving Statistics
        g['rolling_mean_3'] = g['sales'].shift(1).rolling(window=3, min_periods=1).mean()
        g['rolling_mean_7'] = g['sales'].shift(1).rolling(window=7, min_periods=1).mean()
        g['rolling_std_7'] = g['sales'].shift(1).rolling(window=7, min_periods=1).std().fillna(0)
        
        # Exponential Moving Average
        g['ema_7'] = g['sales'].shift(1).ewm(span=7, adjust=False).mean()

        # Domain Signals: Price & Margin Elasticity
        g['price_per_unit'] = g['Unit price']
        g['avg_rating'] = g['Rating']
        g['cogs_ratio'] = g['cogs'] / (g['Total'] + 1e-5)
        g['demand_momentum'] = g['lag_1'] / (g['rolling_mean_7'] + 1e-5)

        featured_groups.append(g)

    full_df = pd.concat(featured_groups, axis=0).dropna().sort_values('date').reset_index(drop=True)

    numeric_features = [
        'day_of_week', 'month', 'day_of_month', 'is_weekend',
        'lag_1', 'lag_2', 'lag_7',
        'rolling_mean_3', 'rolling_mean_7', 'rolling_std_7',
        'ema_7', 'price_per_unit', 'avg_rating', 'cogs_ratio', 'demand_momentum'
    ]
    categorical_features = ['product_line']

    return full_df, numeric_features, categorical_features

def main():
    if not os.path.exists(LOCAL_CSV):
        download_dataset()

    full_df, numeric_features, categorical_features = perform_advanced_feature_engineering(LOCAL_CSV)

    feature_cols = numeric_features + categorical_features
    X = full_df[feature_cols]
    y = full_df['sales']

    # 80/20 Chronological Validation Split
    split_idx = int(len(full_df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # Preprocessor: Standardize Numerics + One-Hot Encode Categories
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    # Candidate Models tuned for Category-Level Demand Forecasting
    candidates = {
        "Category-Aware Random Forest": Pipeline([
            ('preprocessor', preprocessor),
            ('reg', RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=3, min_samples_leaf=2, random_state=42))
        ]),
        "Category-Aware Gradient Boosting": Pipeline([
            ('preprocessor', preprocessor),
            ('reg', GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=5, subsample=0.85, random_state=42))
        ]),
        "Category-Aware Hist Gradient Boosting": Pipeline([
            ('preprocessor', preprocessor),
            ('reg', HistGradientBoostingRegressor(max_iter=250, learning_rate=0.03, max_depth=5, random_state=42))
        ])
    }

    eval_results = {}
    fitted_pipelines = {}

    print("==================================================")
    print("  CATEGORY-LEVEL FEATURE ENGINEERING BACKTEST METRICS  ")
    print("==================================================")

    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        
        mae = float(mean_absolute_error(y_val, preds))
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mape = float(np.mean(np.abs((y_val - preds) / np.maximum(1.0, y_val))) * 100)
        accuracy_score = max(0.0, 100.0 - mape)

        eval_results[name] = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE": f"{round(mape, 2)}%",
            "Accuracy": f"{round(accuracy_score, 2)}%"
        }
        fitted_pipelines[name] = pipe

        print(f"📊 Model: {name}")
        print(f"   - MAE  (Mean Absolute Error):  {round(mae, 2)} units")
        print(f"   - RMSE (Root Mean Sq Error):  {round(rmse, 2)} units")
        print(f"   - MAPE (Error Percentage):    {round(mape, 2)}%")
        print(f"   - Overall Accuracy Score:     {round(accuracy_score, 2)}%\n")

    winning_name = min(eval_results, key=lambda m: eval_results[m]['RMSE'])
    winning_pipeline = fitted_pipelines[winning_name]

    print("--------------------------------------------------")
    print(f"🏆 Winning Model: {winning_name}")
    print(f"🎯 Best Accuracy Achieved: {eval_results[winning_name]['Accuracy']}")
    print("--------------------------------------------------")

    # HUMAN-IN-THE-LOOP CONFIRMATION
    user_choice = input("\nAre you satisfied with this accuracy? Save model as 'model.joblib'? (y/n): ").strip().lower()

    if user_choice in ['y', 'yes']:
        # Retrain winning pipeline on 100% of category data
        winning_pipeline.fit(X, y)

        artifact_dir = os.path.dirname(__file__)
        model_path = os.path.join(artifact_dir, "model.joblib")
        meta_path = os.path.join(artifact_dir, "metadata.json")

        joblib.dump(winning_pipeline, model_path)

        metadata = {
            "winning_model": winning_name,
            "leaderboard": eval_results,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "recent_sales_history": full_df['sales'].tail(30).tolist(),
            "recent_dates": full_df['date'].tail(30).dt.strftime("%Y-%m-%d").tolist()
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ SUCCESS: Serialized winning pipeline to '{model_path}'")
        print(f"📄 Metadata saved to '{meta_path}'")
    else:
        print("\n❌ ABORTED: Model was NOT saved to disk.")

if __name__ == "__main__":
    main()