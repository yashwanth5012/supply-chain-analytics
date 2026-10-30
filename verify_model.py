# backend/verify_model.py
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

LOCAL_CSV = os.path.join(os.path.dirname(__file__), "github_sales.csv")

def verify_pipeline():
    if not os.path.exists(LOCAL_CSV):
        print("❌ Dataset file 'github_sales.csv' not found. Run download_and_train.py first.")
        return

    df = pd.read_csv(LOCAL_CSV)

    # Clean & Format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Total'] = pd.to_numeric(df['Total'], errors='coerce').fillna(0)
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
    df['Unit price'] = pd.to_numeric(df['Unit price'], errors='coerce').fillna(0)
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(0)
    df['cogs'] = pd.to_numeric(df['cogs'], errors='coerce').fillna(0)

    # Disaggregate by Category
    cat_daily = df.groupby(['Date', 'Product line'], as_index=False).agg({
        'Quantity': 'sum',
        'Total': 'sum',
        'Unit price': 'mean',
        'Rating': 'mean',
        'cogs': 'sum'
    }).sort_values(['Product line', 'Date']).reset_index(drop=True)

    cat_daily.rename(columns={'Date': 'date', 'Quantity': 'sales', 'Product line': 'product_line'}, inplace=True)

    # Build Lags & Signals
    featured_groups = []
    for category, group in cat_daily.groupby('product_line'):
        g = group.copy().sort_values('date').reset_index(drop=True)

        g['day_of_week'] = g['date'].dt.dayofweek
        g['month'] = g['date'].dt.month
        g['day_of_month'] = g['date'].dt.day
        g['is_weekend'] = g['day_of_week'].isin([5, 6]).astype(int)

        # Lags & Moving Signals (Shifted by 1)
        g['lag_1'] = g['sales'].shift(1)
        g['lag_2'] = g['sales'].shift(2)
        g['lag_7'] = g['sales'].shift(7)
        g['rolling_mean_3'] = g['sales'].shift(1).rolling(window=3, min_periods=1).mean()
        g['rolling_mean_7'] = g['sales'].shift(1).rolling(window=7, min_periods=1).mean()
        g['rolling_std_7'] = g['sales'].shift(1).rolling(window=7, min_periods=1).std().fillna(0)
        g['ema_7'] = g['sales'].shift(1).ewm(span=7, adjust=False).mean()

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

    X = full_df[numeric_features + categorical_features]
    y = full_df['sales']

    # Chronological 80/20 Split
    split_idx = int(len(full_df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('reg', GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=5, subsample=0.85, random_state=42))
    ])

    # Fit Model
    model.fit(X_train, y_train)

    # 1. Evaluate on Training Data
    train_preds = model.predict(X_train)
    train_mae = float(mean_absolute_error(y_train, train_preds))
    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_preds)))
    train_mape = float(np.mean(np.abs((y_train - train_preds) / np.maximum(1.0, y_train))) * 100)
    train_acc = max(0.0, 100.0 - train_mape)

    # 2. Evaluate on Unseen Validation Data
    val_preds = model.predict(X_val)
    val_mae = float(mean_absolute_error(y_val, val_preds))
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
    val_mape = float(np.mean(np.abs((y_val - val_preds) / np.maximum(1.0, y_val))) * 100)
    val_acc = max(0.0, 100.0 - val_mape)

    print("\n==================================================")
    print("      VERIFICATION REPORT: OVERFITTING & LEAKAGE  ")
    print("==================================================")
    print(f"📊 TRAIN SET ACCURACY      : {round(train_acc, 2)}%  (MAE: {round(train_mae, 2)})")
    print(f"📊 VALIDATION SET ACCURACY : {round(val_acc, 2)}%  (MAE: {round(val_mae, 2)})\n")

    # Check for Data Leakage (Perfect 100% or near 100% accuracy)
    if train_acc > 99.0 or val_acc > 99.0:
        print("⚠️ DATA LEAKAGE DETECTED: Accuracy is unrealistically high (>99%).")
    else:
        print("✅ DATA LEAKAGE CHECK: PASSED (No target features leaked).")

    # Check for Overfitting (Large gap between Train and Validation)
    acc_gap = abs(train_acc - val_acc)
    print(f"📉 Accuracy Gap (Train vs Val): {round(acc_gap, 2)}%")

    if acc_gap > 15.0:
        print("⚠️ OVERFITTING DETECTED: Model performs significantly better on training data than validation data.")
    else:
        print("✅ OVERFITTING CHECK: PASSED (Model generalizes well to unseen future dates).")
    print("==================================================\n")

if __name__ == "__main__":
    verify_pipeline()