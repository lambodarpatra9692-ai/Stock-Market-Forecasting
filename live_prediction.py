import pandas as pd
import numpy as np
import yfinance as yf


# ==========================================
# STEP 1: CONFIGURATION
# ==========================================

TICKER = "RELIANCE.NS"

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
# STEP 2: DOWNLOAD MARKET DATA
# ==========================================

def download_market_data():

    print("Downloading Reliance market data...")

    data = yf.download(
        TICKER,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(
            "Unable to download stock market data."
        )

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):

        data.columns = data.columns.get_level_values(0)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing market columns: {missing}"
        )

    data = data[required].copy()

    # Convert columns into numeric values
    data = data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Remove incomplete records
    data = data.dropna(
        subset=required
    )

    # Sort dates
    data = data.sort_index()

    # Remove duplicate dates
    data = data[
        ~data.index.duplicated(keep="last")
    ]

    return data


# ==========================================
# STEP 3: FEATURE ENGINEERING
# ==========================================

def create_live_features(data):

    data = data.copy()

    # Daily Return
    data["Daily_Return"] = (
        data["Close"].pct_change()
    )

    # Moving Averages
    data["MA5"] = (
        data["Close"].rolling(5).mean()
    )

    data["MA10"] = (
        data["Close"].rolling(10).mean()
    )

    data["MA20"] = (
        data["Close"].rolling(20).mean()
    )

    data["MA50"] = (
        data["Close"].rolling(50).mean()
    )

    # RSI - 14 Periods
    delta = data["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    data["RSI"] = (
        100 - (100 / (1 + rs))
    )

    data.loc[avg_loss == 0, "RSI"] = 100

    # MACD
    ema12 = data["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = data["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = ema12 - ema26

    # Signal Line
    data["Signal_Line"] = (
        data["MACD"].ewm(
            span=9,
            adjust=False
        ).mean()
    )

    # Bollinger Bands
    rolling_mean = (
        data["Close"].rolling(20).mean()
    )

    rolling_std = (
        data["Close"].rolling(20).std()
    )

    data["BB_Upper"] = (
        rolling_mean + 2 * rolling_std
    )

    data["BB_Lower"] = (
        rolling_mean - 2 * rolling_std
    )

    # Previous Day Features
    data["Previous_Close"] = (
        data["Close"].shift(1)
    )

    data["Previous_Return"] = (
        data["Daily_Return"].shift(1)
    )

    data["Previous_Volume"] = (
        data["Volume"].shift(1)
    )

    # Replace infinite values
    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove incomplete feature records
    data = data.dropna(
        subset=FEATURES
    )

    return data


# ==========================================
# STEP 4: GET LATEST LIVE FEATURES
# ==========================================

def get_latest_live_features():

    # Download market data
    raw_data = download_market_data()

    # Generate features
    feature_data = create_live_features(
        raw_data
    )

    if feature_data.empty:
        raise ValueError(
            "Unable to create complete market features."
        )

    # Latest available trading record
    latest_row = feature_data.iloc[-1]

    # Select features in training order
    latest_features = feature_data[FEATURES]

    print("\nLATEST MARKET DATA:")

    print(
        latest_row[[
            "Close",
            "MA5",
            "MA20",
            "MA50",
            "RSI",
            "MACD"
        ]]
    )

    print("\nLATEST TRADING DATE:")
    print(latest_row.name)

    print("\nFEATURE COUNT:")
    print(len(FEATURES))

    return latest_row, latest_features


# ==========================================
# STEP 5: TEST THE MODULE
# ==========================================

if __name__ == "__main__":

    try:

        latest_row, feature_data = get_latest_live_features()

        print("\nLIVE DATA FETCHED SUCCESSFULLY!")

        print("\nLATEST FEATURES:")

        print(
            feature_data[FEATURES].tail(1)
        )

    except Exception as error:

        print("\nERROR:")
        print(error)