import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# STEP 1: LOAD FEATURE DATASET
# ==========================================

data = pd.read_csv(
    "dataset/RELIANCE_features.csv",
    index_col=0,
    parse_dates=True
)

data = data.sort_index()

print("DATASET LOADED SUCCESSFULLY!")
print("Dataset Shape:", data.shape)


# ==========================================
# STEP 2: SELECT FEATURES
# ==========================================

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Daily_Return",
    "MA5",
    "MA10",
    "MA20",
    "MA50",
    "RSI",
    "MACD",
    "Signal_Line",
    "BB_Upper",
    "BB_Lower",
    "Previous_Close",
    "Previous_Return",
    "Previous_Volume"
]

target = "Target"

X = data[features]
y = data[target]

print("\nFeatures Selected!")
print("Number of Features:", len(features))


# ==========================================
# STEP 3: SPLIT TRAINING AND TESTING DATA
# ==========================================

split_index = int(len(data) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nDATA SPLITTING COMPLETED!")

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))


# ==========================================
# STEP 4: CREATE MODELS
# ==========================================

models = {

    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
}


# ==========================================
# STEP 5: TRAIN AND EVALUATE MODELS
# ==========================================

os.makedirs("models", exist_ok=True)

results = []

predictions = pd.DataFrame(index=y_test.index)

for name, model in models.items():

    print("\nTraining:", name)

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate errors
    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    predictions[name] = y_pred

    # Save model
    filename = name.lower().replace(" ", "_")

    joblib.dump(
        model,
        f"models/{filename}.pkl"
    )

    print(name, "training completed!")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))


# ==========================================
# STEP 6: COMPARE MODEL PERFORMANCE
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE"
)

print("\n================================")
print("MODEL PERFORMANCE COMPARISON")
print("================================")

print(results_df.to_string(index=False))


# ==========================================
# STEP 7: SAVE RESULTS
# ==========================================

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

predictions.insert(
    0,
    "Actual_Price",
    y_test
)

predictions.to_csv(
    "models/predictions.csv"
)


# ==========================================
# STEP 8: SAVE BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

best_model = models[best_model_name]

filename = best_model_name.lower().replace(" ", "_")

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\n================================")
print("MACHINE LEARNING COMPLETED!")
print("================================")

print("Best Model:", best_model_name)

print("All models saved successfully!")

print("Model comparison saved!")
print("Predictions saved!")