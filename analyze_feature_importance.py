import pandas as pd
import joblib
import plotly.graph_objects as go

from pathlib import Path

# ==========================================
# STEP 1: PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

FEATURES = [
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


# ==========================================
# STEP 2: LOAD FIVE TRAINED MODELS
# ==========================================

feature_importance = {}

print("\n======================================")
print("     FEATURE IMPORTANCE ANALYSIS")
print("======================================\n")

for day in range(1, 6):

    model_path = MODEL_DIR / f"forecast_day_{day}.pkl"

    if not model_path.exists():

        print(f"Model not found: {model_path}")
        continue

    model = joblib.load(model_path)

    print(f"Loaded Day {day} model successfully.")

    # Check feature importance availability
    if not hasattr(model, "feature_importances_"):

        print(f"Day {day} model does not support feature importance.")
        continue

    # Get feature names from trained model when available
    if hasattr(model, "feature_names_in_"):

        feature_names = list(model.feature_names_in_)

    else:

        feature_names = FEATURES

    importances = model.feature_importances_

    if len(feature_names) != len(importances):

        print(f"Feature mismatch in Day {day} model.")
        continue

    # Store feature importance
    feature_importance[f"Day_{day}"] = importances


# ==========================================
# STEP 3: CREATE FEATURE IMPORTANCE TABLE
# ==========================================

if not feature_importance:

    print("\nNo valid feature importance data found.")
    raise SystemExit

importance_df = pd.DataFrame(
    feature_importance,
    index=feature_names
)

importance_df.index.name = "Feature"

importance_df = importance_df.reset_index()

# Calculate average importance across five models
day_columns = [
    col for col in importance_df.columns
    if col.startswith("Day_")
]

importance_df["Average_Importance"] = (
    importance_df[day_columns].mean(axis=1)
)

# Sort by average importance
importance_df = importance_df.sort_values(
    by="Average_Importance",
    ascending=False
)

# ==========================================
# STEP 4: SAVE RESULTS
# ==========================================

output_path = MODEL_DIR / "feature_importance.csv"

importance_df.to_csv(
    output_path,
    index=False
)

print("\nFeature importance table:")
print(
    importance_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nResults saved to:")
print(output_path)


# ==========================================
# STEP 5: VISUALIZE FEATURE IMPORTANCE
# ==========================================

top_features = importance_df.head(15).sort_values(
    by="Average_Importance",
    ascending=True
)

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=top_features["Average_Importance"],
        y=top_features["Feature"],
        orientation="h",
        name="Average Importance"
    )
)

fig.update_layout(
    title="Top 15 Important Features - Random Forest Models",
    xaxis_title="Average Feature Importance",
    yaxis_title="Features",
    template="plotly_dark",
    height=650
)

# Save interactive chart
chart_path = MODEL_DIR / "feature_importance.html"

fig.write_html(str(chart_path))

print("\nInteractive chart saved to:")
print(chart_path)

print("\n======================================")
print(" FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("======================================")