import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==================================
# STEP 1: LOAD DATASET
# ==================================

data = pd.read_csv(
    "dataset/RELIANCE_cleaned.csv",
    index_col=0,
    parse_dates=True
)

data = data.sort_index()

print("DATASET LOADED SUCCESSFULLY!")
print(data.head())


# ==================================
# STEP 2: BASIC STATISTICS
# ==================================

print("\nSTOCK PRICE ANALYSIS")

print("Highest Closing Price:", data["Close"].max())
print("Lowest Closing Price:", data["Close"].min())
print("Average Closing Price:", data["Close"].mean())

print("Highest Price Date:", data["Close"].idxmax())
print("Lowest Price Date:", data["Close"].idxmin())


# ==================================
# STEP 3: DAILY RETURNS
# ==================================

data["Daily_Return"] = (
    data["Close"].pct_change() * 100
)

print("\nDAILY RETURNS")
print(data[["Close", "Daily_Return"]].head(10))


# ==================================
# STEP 4: MOVING AVERAGES
# ==================================

data["MA20"] = data["Close"].rolling(
    window=20
).mean()

data["MA50"] = data["Close"].rolling(
    window=50
).mean()

print("\nMOVING AVERAGES")
print(data[["Close", "MA20", "MA50"]].tail(10))


# ==================================
# STEP 5: CREATE PLOTS FOLDER
# ==================================

os.makedirs("plots", exist_ok=True)


# ==================================
# STEP 6: CLOSING PRICE GRAPH
# ==================================

plt.figure(figsize=(12, 6))

plt.plot(data.index, data["Close"],
         label="Closing Price")

plt.title("Reliance Stock Closing Price")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("plots/closing_price.png")
plt.show()


# ==================================
# STEP 7: MOVING AVERAGE GRAPH
# ==================================

plt.figure(figsize=(12, 6))

plt.plot(data.index, data["Close"],
         label="Closing Price")

plt.plot(data.index, data["MA20"],
         label="MA20")

plt.plot(data.index, data["MA50"],
         label="MA50")

plt.title("Stock Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("plots/moving_average.png")
plt.show()


# ==================================
# STEP 8: TRADING VOLUME GRAPH
# ==================================

plt.figure(figsize=(12, 6))

plt.bar(data.index, data["Volume"])

plt.title("Reliance Trading Volume")
plt.xlabel("Date")
plt.ylabel("Number of Shares")
plt.tight_layout()

plt.savefig("plots/trading_volume.png")
plt.show()


# ==================================
# STEP 9: CORRELATION HEATMAP
# ==================================

correlation = data[
    ["Open", "High", "Low", "Close", "Volume"]
].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Stock Market Correlation Heatmap")
plt.tight_layout()

plt.savefig("plots/correlation_heatmap.png")
plt.show()


# ==================================
# STEP 10: SAVE ANALYSIS DATA
# ==================================

data.to_csv("dataset/RELIANCE_eda.csv")

print("\n================================")
print("EDA COMPLETED SUCCESSFULLY!")
print("All graphs saved in plots folder.")
print("Analysis dataset saved successfully!")
print("================================")