import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# STEP 1: CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "RELIANCE_features.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Daily_Return", "MA5", "MA10", "MA20", "MA50",
    "RSI", "MACD", "Signal_Line", "BB_Upper",
    "BB_Lower", "Previous_Close",
    "Previous_Return", "Previous_Volume"
]

FORECAST_DAYS = 5


# ==========================================
# STEP 2: LOAD DATASET
# ==========================================

print("Loading Reliance dataset...")

data = pd.read_csv(
    DATA_PATH,
    index_col=0,
    parse_dates=True
)

data = data.sort_index()

if data.empty:
    raise ValueError("Dataset is empty.")

missing = [
    col for col in FEATURES + ["Close"]
    if col not in data.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# ==========================================
# STEP 3: PREPARE FEATURES
# ==========================================

X_all = data[FEATURES].replace(
    [np.inf, -np.inf],
    np.nan
)

# Chronological 80/20 split
split_position = int(len(data) * 0.80)

if split_position <= 0 or split_position >= len(data):
    raise ValueError("Not enough records for training and testing.")

split_date = data.index[split_position]
train_end_date = data.index[split_position - 1]

results = []


# ==========================================
# STEP 4: TRAIN FIVE FORECASTING MODELS
# ==========================================

for day in range(1, FORECAST_DAYS + 1):

    print("\n" + "=" * 45)
    print(f"TRAINING MODEL FOR DAY {day}")
    print("=" * 45)

    # Target: closing price after the selected number of sessions
    target = data["Close"].shift(-day)

    # Date on which each target price occurs
    target_dates = pd.Series(
        data.index,
        index=data.index
    ).shift(-day)

    valid = (
        X_all.notna().all(axis=1)
        & target.notna()
        & target_dates.notna()
    )

    # Purge training samples whose target falls
    # on or after the test period
    train_mask = (
        valid
        & (data.index < split_date)
        & (target_dates <= train_end_date)
    )

    test_mask = (
        valid
        & (data.index >= split_date)
    )

    X_train = X_all.loc[train_mask]
    y_train = target.loc[train_mask]

    X_test = X_all.loc[test_mask]
    y_test = target.loc[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            f"Insufficient training/testing data for Day {day}."
        )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # Create Random Forest model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train, y_train)

    # Test model
    predictions = model.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(f"Day {day} MAE: {mae:.4f}")
    print(f"Day {day} RMSE: {rmse:.4f}")
    print(f"Day {day} R2 Score: {r2:.4f}")

    # Save model
    model_path = MODEL_DIR / f"forecast_day_{day}.pkl"

    joblib.dump(model, model_path)

    print("Model saved:", model_path.name)

    results.append({
        "Forecast_Day": day,
        "Training_Records": len(X_train),
        "Testing_Records": len(X_test),
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })


# ==========================================
# STEP 5: SAVE MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame(results)

comparison_path = MODEL_DIR / "multi_day_model_comparison.csv"

comparison.to_csv(
    comparison_path,
    index=False
)

print("\n" + "=" * 45)
print("5-DAY MODEL TRAINING COMPLETED!")
print("=" * 45)

print(comparison.to_string(index=False))

print("\nComparison saved:", comparison_path)
print("\nAll five forecasting models saved successfully!")