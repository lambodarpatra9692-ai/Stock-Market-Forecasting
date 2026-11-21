import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ==========================================
# STEP 1: PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

INPUT_PATH = MODEL_DIR / "feature_importance.csv"

# ==========================================
# STEP 2: LOAD FEATURE IMPORTANCE DATA
# ==========================================

print("\n======================================")
print("  FIVE MODEL FEATURE COMPARISON")
print("======================================\n")

if not INPUT_PATH.exists():

    print("Feature importance CSV not found.")
    print("Run analyze_feature_importance.py first.")

    raise SystemExit

importance_df = pd.read_csv(INPUT_PATH)

required_columns = [
    "Feature",
    "Day_1",
    "Day_2",
    "Day_3",
    "Day_4",
    "Day_5"
]

missing_columns = [
    col for col in required_columns
    if col not in importance_df.columns
]

if missing_columns:

    print("Missing columns:", missing_columns)
    raise SystemExit

# Keep only required columns
importance_df = importance_df[required_columns].copy()

# Convert importance values to numeric
day_columns = [
    "Day_1",
    "Day_2",
    "Day_3",
    "Day_4",
    "Day_5"
]

for col in day_columns:

    importance_df[col] = pd.to_numeric(
        importance_df[col],
        errors="coerce"
    )

importance_df = importance_df.dropna(
    subset=day_columns
)

# ==========================================
# STEP 3: TOP 5 FEATURES FOR EACH MODEL
# ==========================================

print("\nTOP 5 FEATURES BY FORECAST MODEL\n")

top5_results = []

for day in day_columns:

    top_features = (
        importance_df[
            ["Feature", day]
        ]
        .sort_values(
            by=day,
            ascending=False
        )
        .head(5)
        .reset_index(drop=True)
    )

    top_features.insert(
        0,
        "Forecast_Model",
        day
    )

    top_features.insert(
        1,
        "Rank",
        range(1, len(top_features) + 1)
    )

    top_features = top_features.rename(
        columns={
            day: "Importance"
        }
    )

    top5_results.append(top_features)

    print(f"\n{day} MODEL")

    print(
        top_features[
            ["Rank", "Feature", "Importance"]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

top5_df = pd.concat(
    top5_results,
    ignore_index=True
)

top5_path = MODEL_DIR / "top5_feature_comparison.csv"

top5_df.to_csv(
    top5_path,
    index=False
)

# ==========================================
# STEP 4: GROUPED BAR CHART
# ==========================================

print("\nCreating grouped feature comparison chart...")

# Select top 10 features by overall average importance
importance_df["Average_Importance"] = (
    importance_df[day_columns].mean(axis=1)
)

top10 = importance_df.nlargest(
    10,
    "Average_Importance"
)

fig = go.Figure()

for day in day_columns:

    fig.add_trace(
        go.Bar(
            x=top10["Feature"],
            y=top10[day],
            name=day.replace("_", " ")
        )
    )

fig.update_layout(
    title="Feature Importance Comparison Across Five Models",
    xaxis_title="Features",
    yaxis_title="Feature Importance",
    barmode="group",
    template="plotly_dark",
    height=600,
    xaxis_tickangle=-35
)

grouped_chart_path = (
    MODEL_DIR / "feature_importance_comparison.html"
)

fig.write_html(
    str(grouped_chart_path)
)

# ==========================================
# STEP 5: FEATURE IMPORTANCE HEATMAP
# ==========================================

print("Creating feature importance heatmap...")

heatmap_df = importance_df.sort_values(
    by="Average_Importance",
    ascending=False
)

heatmap_fig = go.Figure(
    data=go.Heatmap(
        z=heatmap_df[day_columns].values,
        x=["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"],
        y=heatmap_df["Feature"],
        colorscale="Viridis",
        colorbar=dict(
            title="Importance"
        )
    )
)

heatmap_fig.update_layout(
    title="Feature Importance Heatmap - Five Forecast Models",
    xaxis_title="Forecast Horizon",
    yaxis_title="Features",
    template="plotly_dark",
    height=700
)

heatmap_path = (
    MODEL_DIR / "feature_importance_heatmap.html"
)

heatmap_fig.write_html(
    str(heatmap_path)
)

# ==========================================
# STEP 6: FEATURE IMPORTANCE CHANGE ANALYSIS
# ==========================================

print("\nFEATURE IMPORTANCE CHANGE ANALYSIS\n")

for day in day_columns:

    highest_feature = importance_df.loc[
        importance_df[day].idxmax()
    ]

    print(
        f"{day}: {highest_feature['Feature']} "
        f"({highest_feature[day]:.4f})"
    )

# ==========================================
# STEP 7: SAVE COMPARISON SUMMARY
# ==========================================

comparison_path = (
    MODEL_DIR / "feature_importance_comparison.csv"
)

importance_df.to_csv(
    comparison_path,
    index=False
)

print("\n======================================")
print("      ANALYSIS COMPLETED")
print("======================================")

print("\nGenerated files:")

print(top5_path)
print(grouped_chart_path)
print(heatmap_path)
print(comparison_path)

print("\n======================================")