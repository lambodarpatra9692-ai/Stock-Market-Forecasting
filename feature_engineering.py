import pandas as pd
import numpy as np

# ==========================================
# STEP 1: LOAD CLEANED DATASET
# ==========================================

data = pd.read_csv(
    "dataset/RELIANCE_cleaned.csv",
    index_col=0,
    parse_dates=True
)

data = data.sort_index()

print("DATASET LOADED SUCCESSFULLY!")
print(data.head())


# ==========================================
# STEP 2: DAILY RETURNS
# ==========================================

data["Daily_Return"] = data["Close"].pct_change()

print("\nDaily Returns Created!")


# ==========================================
# STEP 3: MOVING AVERAGES
# ==========================================

data["MA5"] = data["Close"].rolling(window=5).mean()

data["MA10"] = data["Close"].rolling(window=10).mean()

data["MA20"] = data["Close"].rolling(window=20).mean()

data["MA50"] = data["Close"].rolling(window=50).mean()

print("Moving Averages Created!")


# ==========================================
# STEP 4: RSI (RELATIVE STRENGTH INDEX)
# ==========================================

delta = data["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss

data["RSI"] = 100 - (100 / (1 + rs))

# Handle periods with no price losses
data.loc[avg_loss == 0, "RSI"] = 100

print("RSI Indicator Created!")


# ==========================================
# STEP 5: MACD
# ==========================================

ema12 = data["Close"].ewm(
    span=12,
    adjust=False
).mean()

ema26 = data["Close"].ewm(
    span=26,
    adjust=False
).mean()

data["MACD"] = ema12 - ema26

data["Signal_Line"] = data["MACD"].ewm(
    span=9,
    adjust=False
).mean()

print("MACD Created!")


# ==========================================
# STEP 6: BOLLINGER BANDS
# ==========================================

data["BB_Middle"] = data["Close"].rolling(
    window=20
).mean()

std = data["Close"].rolling(window=20).std()

data["BB_Upper"] = data["BB_Middle"] + (2 * std)

data["BB_Lower"] = data["BB_Middle"] - (2 * std)

print("Bollinger Bands Created!")


# ==========================================
# STEP 7: LAG FEATURES
# ==========================================

data["Previous_Close"] = data["Close"].shift(1)

data["Previous_Return"] = data["Daily_Return"].shift(1)

data["Previous_Volume"] = data["Volume"].shift(1)

print("Lag Features Created!")


# ==========================================
# STEP 8: TARGET VARIABLE
# ==========================================

# Predict the next trading day's closing price

data["Target"] = data["Close"].shift(-1)

print("Target Variable Created!")


# ==========================================
# STEP 9: HANDLE MISSING VALUES
# ==========================================

data.replace([np.inf, -np.inf], np.nan, inplace=True)

data.dropna(inplace=True)


# ==========================================
# STEP 10: SAVE FEATURE DATASET
# ==========================================

data.to_csv("dataset/RELIANCE_features.csv")

print("\n====================================")
print("FEATURE ENGINEERING COMPLETED!")
print("====================================")

print("\nFinal Dataset Shape:")
print(data.shape)

print("\nFinal Columns:")
print(data.columns.tolist())

print("\nFirst 5 Records:")
print(data.head())

print("\nFeature dataset saved successfully!")