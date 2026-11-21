import pandas as pd

data = pd.read_csv("models/multi_day_model_comparison.csv")

print("Column names:")
print(data.columns.tolist())

print("\nForecast performance:")
print(data)