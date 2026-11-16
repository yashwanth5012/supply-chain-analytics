import numpy as np
import pandas as pd

# --- Fast Vectorized NumPy Decision Tree ---
class FastNumPyDecisionTreeRegressor:
    def __init__(self, max_depth=4, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.std(y) == 0:
            return float(np.mean(y))

        best_feature, best_threshold, best_mse = None, None, float('inf')
        
        for feat_idx in range(n_features):
            col_vals = X[:, feat_idx]
            thresholds = np.percentile(col_vals, np.linspace(10, 90, 9))
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

    def predict(self, X):
        return np.array([self._predict_row(row, self.tree) for row in X])

    def _predict_row(self, row, node):
        if not isinstance(node, dict):
            return node
        if row[node['feature']] <= node['threshold']:
            return self._predict_row(row, node['left'])
        return self._predict_row(row, node['right'])


# --- Fast Random Forest Regressor ---
class FastRandomForestRegressor:
    def __init__(self, n_estimators=5, max_depth=4):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=min(n_samples, 200), replace=True)
            tree = FastNumPyDecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)


# --- Fast Gradient Boosting Regressor ---
class FastGradientBoostingRegressor:
    def __init__(self, n_estimators=5, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_pred = 0

    def fit(self, X, y):
        self.initial_pred = float(np.mean(y))
        residuals = y - self.initial_pred
        self.trees = []

        for _ in range(self.n_estimators):
            tree = FastNumPyDecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            residuals -= self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        preds = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X)
        return preds


# --- Universal Dataset Ingestion & Column Inference Engine ---
def auto_detect_and_normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    col_map = {col: str(col).lower().strip().replace('_', '').replace(' ', '') for col in df.columns}
    
    date_col, demand_col, item_col = None, None, None

    # Step 1: Broad Keyword Identification
    for orig_col, clean_col in col_map.items():
        if not date_col and any(k in clean_col for k in ['date', 'time', 'timestamp', 'day', 'dt', 'period', 'month', 'year', 'created']):
            date_col = orig_col
        if not demand_col and any(k in clean_col for k in ['sales', 'demand', 'quantity', 'qty', 'units', 'volume', 'amount', 'count', 'ordered', 'target']):
            demand_col = orig_col
        if not item_col and any(k in clean_col for k in ['sku', 'product', 'item', 'part', 'code', 'productid', 'itemid']):
            item_col = orig_col

    # Step 2: Datetime Fallback Inspection
    if not date_col:
        for col in df.columns:
            try:
                converted = pd.to_datetime(df[col], errors='coerce')
                if converted.notna().sum() / max(1, len(df)) > 0.4:
                    date_col = col
                    break
            except Exception:
                continue

    # Step 3: Numeric Fallback Inspection for Sales
    if not demand_col:
        for col in df.columns:
            if col in [date_col, item_col]:
                continue
            cleaned = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
            if pd.to_numeric(cleaned, errors='coerce').notna().sum() / max(1, len(df)) > 0.4:
                demand_col = col
                break

    if not date_col or not demand_col:
        raise ValueError(f"Could not automatically detect Date and Sales/Demand columns. Columns found: {list(df.columns)}")

    rename_dict = {date_col: 'date', demand_col: 'sales'}
    if item_col:
        rename_dict[item_col] = 'item_id'

    normalized_df = df.rename(columns=rename_dict)

    # Clean numeric sales values
    normalized_df['sales'] = normalized_df['sales'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
    normalized_df['sales'] = pd.to_numeric(normalized_df['sales'], errors='coerce').fillna(0)
    
    # Coerce Dates
    normalized_df['date'] = pd.to_datetime(normalized_df['date'], errors='coerce')
    normalized_df = normalized_df.dropna(subset=['date'])

    if len(normalized_df) == 0:
        raise ValueError("No valid sequential date observations could be parsed from your dataset.")

    # Aggregate daily observations
    if 'item_id' in normalized_df.columns:
        normalized_df = normalized_df.groupby(['item_id', 'date'], as_index=False)['sales'].sum()
    else:
        normalized_df = normalized_df.groupby('date', as_index=False)['sales'].sum()

    return normalized_df.sort_values('date').reset_index(drop=True)


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    df['rolling_mean_7'] = df['sales'].shift(1).rolling(window=7, min_periods=1).mean()

    return df.dropna().reset_index(drop=True)


def calculate_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return round(mae, 2), round(rmse, 2)


def run_multi_model_pipeline(df: pd.DataFrame, forecast_horizon: int = 14):
    clean_df = auto_detect_and_normalize_schema(df)
    featured_df = extract_time_features(clean_df)

    if len(featured_df) < 3:
        raise ValueError("Insufficient time-series rows found in dataset. Provide a dataset with at least 5-10 date rows.")

    split_idx = max(1, int(len(featured_df) * 0.8))
    train_df, val_df = featured_df.iloc[:split_idx], featured_df.iloc[split_idx:]

    feature_cols = ['day_of_week', 'month', 'day', 'lag_1', 'lag_7', 'rolling_mean_7']
    X_train, y_train = train_df[feature_cols].values, train_df['sales'].values
    X_val, y_val = (val_df[feature_cols].values, val_df['sales'].values) if len(val_df) > 0 else (X_train, y_train)

    # Model 1: Baseline Moving Average
    val_preds_ma = val_df['rolling_mean_7'].values if len(val_df) > 0 else train_df['rolling_mean_7'].values
    mae_ma, rmse_ma = calculate_metrics(y_val, val_preds_ma)

    # Model 2: Pure NumPy Random Forest
    rf_model = FastRandomForestRegressor(n_estimators=5, max_depth=4)
    rf_model.fit(X_train, y_train)
    val_preds_rf = rf_model.predict(X_val)
    mae_rf, rmse_rf = calculate_metrics(y_val, val_preds_rf)

    # Model 3: Pure NumPy Gradient Boosting
    gb_model = FastGradientBoostingRegressor(n_estimators=5, learning_rate=0.1, max_depth=3)
    gb_model.fit(X_train, y_train)
    val_preds_gb = gb_model.predict(X_val)
    mae_gb, rmse_gb = calculate_metrics(y_val, val_preds_gb)

    leaderboard = {
        "Moving Average Baseline": {"MAE": mae_ma, "RMSE": rmse_ma},
        "Random Forest Regressor": {"MAE": mae_rf, "RMSE": rmse_rf},
        "Gradient Boosting Regressor": {"MAE": mae_gb, "RMSE": rmse_gb}
    }

    winning_model_name = min(leaderboard, key=lambda m: leaderboard[m]['RMSE'])

    # Retrain selected algorithm on entire historical dataset
    X_full, y_full = featured_df[feature_cols].values, featured_df['sales'].values
    if winning_model_name == "Random Forest Regressor":
        final_model = FastRandomForestRegressor(n_estimators=5, max_depth=4)
        final_model.fit(X_full, y_full)
    elif winning_model_name == "Gradient Boosting Regressor":
        final_model = FastGradientBoostingRegressor(n_estimators=5, learning_rate=0.1, max_depth=3)
        final_model.fit(X_full, y_full)
    else:
        final_model = None

    history_sales = clean_df['sales'].tolist()
    last_date = pd.to_datetime(clean_df['date'].iloc[-1])
    predictions = []

    for i in range(1, forecast_horizon + 1):
        next_date = last_date + pd.Timedelta(days=i)
        lag_1 = history_sales[-1]
        lag_7 = history_sales[-7] if len(history_sales) >= 7 else history_sales[0]
        rolling_7 = float(np.mean(history_sales[-7:]))

        if final_model is not None:
            input_feat = np.array([[next_date.dayofweek, next_date.month, next_date.day, lag_1, lag_7, rolling_7]])
            pred_val = float(final_model.predict(input_feat)[0])
        else:
            pred_val = rolling_7

        pred_val = round(max(0.0, pred_val), 2)
        predictions.append({"date": next_date.strftime("%Y-%m-%d"), "predicted_sales": pred_val})
        history_sales.append(pred_val)

    return {
        "winning_model": winning_model_name,
        "evaluation": leaderboard[winning_model_name],
        "leaderboard": leaderboard,
        "forecast": predictions,
        "cleaned_data": clean_df
    }