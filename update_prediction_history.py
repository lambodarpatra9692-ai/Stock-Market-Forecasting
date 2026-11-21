import pandas as pd
import yfinance as yf
from pathlib import Path

# ==========================================
# STEP 1: FILE CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

HISTORY_PATH = (
    BASE_DIR / "history" / "forecast_history.csv"
)


# ==========================================
# STEP 2: UPDATE PREDICTION HISTORY
# ==========================================

def update_prediction_history():

    print("\n===================================")
    print("   PREDICTION HISTORY UPDATER")
    print("===================================\n")

    # Check whether history file exists
    if not HISTORY_PATH.exists():
        print("Prediction history file not found.")
        return

    # Load prediction history
    history_df = pd.read_csv(HISTORY_PATH)

    if history_df.empty:
        print("Prediction history is empty.")
        return

    # Check required columns
    required_columns = [
        "Origin_Date",
        "Forecast_Day",
        "Predicted_Close",
        "Prediction_Status"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in history_df.columns
    ]

    if missing_columns:
        print("Missing columns:", missing_columns)
        return

    # Convert dates safely
    history_df["Origin_Date"] = pd.to_datetime(
        history_df["Origin_Date"],
        format="mixed",
        errors="coerce"
    )

    # Convert numeric columns safely
    history_df["Predicted_Close"] = pd.to_numeric(
        history_df["Predicted_Close"],
        errors="coerce"
    )

    # Ensure output columns exist
    if "Actual_Close" not in history_df.columns:
        history_df["Actual_Close"] = float("nan")

    if "Actual_Date" not in history_df.columns:
        history_df["Actual_Date"] = pd.NA

    if "Prediction_Error" not in history_df.columns:
        history_df["Prediction_Error"] = float("nan")

    # ==========================================
    # STEP 3: FETCH ACTUAL MARKET DATA
    # ==========================================

    print("Fetching Reliance market data...")

    try:
        stock = yf.Ticker("RELIANCE.NS")

        actual_data = stock.history(period="1y")

    except Exception as error:
        print("Market data fetching failed:", error)
        return

    if actual_data.empty:
        print("Unable to fetch actual market data.")
        return

    # Preserve Indian market dates (IST)
    if actual_data.index.tz is not None:
        actual_data.index = (
            actual_data.index
            .tz_localize(None)
            .normalize()
        )
    else:
        actual_data.index = actual_data.index.normalize()

    # Remove invalid closing prices
    actual_data = actual_data.dropna(subset=["Close"])

    # Sort trading dates
    actual_data = actual_data.sort_index()

    print("\nLatest available trading dates:")
    print(actual_data.index[-5:])

    latest_market_date = actual_data.index.max()

    print("\nLatest actual market date:")
    print(latest_market_date.strftime("%Y-%m-%d"))

    # ==========================================
    # STEP 4: UPDATE PENDING PREDICTIONS
    # ==========================================

    completed_count = 0
    pending_count = 0

    for index, row in history_df.iterrows():

        # Skip completed predictions
        if str(row["Prediction_Status"]).strip().lower() == "completed":
            continue

        # Skip invalid records
        if pd.isna(row["Origin_Date"]):
            print(f"Skipping row {index}: Invalid origin date.")
            pending_count += 1
            continue

        if pd.isna(row["Predicted_Close"]):
            print(f"Skipping row {index}: Invalid predicted price.")
            pending_count += 1
            continue

        try:
            forecast_day = int(
                str(row["Forecast_Day"])
                .replace("Day ", "")
                .strip()
            )

            if forecast_day < 1 or forecast_day > 5:
                print(f"Skipping row {index}: Invalid forecast day.")
                pending_count += 1
                continue

        except (ValueError, TypeError):
            print(f"Skipping row {index}: Invalid forecast day.")
            pending_count += 1
            continue

        # Get forecast origin date
        origin_date = row["Origin_Date"].normalize()

        # Find future trading sessions
        future_dates = actual_data.index[
            actual_data.index > origin_date
        ].unique().sort_values()

        # Check whether actual trading session is available
        if len(future_dates) < forecast_day:

            pending_count += 1
            continue

        # Select the correct future trading session
        actual_date = future_dates[forecast_day - 1]

        # Get actual closing price
        actual_close = float(
            actual_data.loc[actual_date, "Close"]
        )

        predicted_close = float(row["Predicted_Close"])

        # Calculate prediction error
        prediction_error = (
            predicted_close - actual_close
        )

        # Update history
        history_df.loc[index, "Actual_Close"] = actual_close

        history_df.loc[index, "Actual_Date"] = (
            actual_date.strftime("%Y-%m-%d")
        )

        history_df.loc[index, "Prediction_Error"] = (
            prediction_error
        )

        history_df.loc[index, "Prediction_Status"] = "Completed"

        completed_count += 1

        print(
            f"Updated: {row['Forecast_Day']} | "
            f"Actual Date: {actual_date.strftime('%Y-%m-%d')} | "
            f"Actual Close: ₹{actual_close:.2f}"
        )

    # ==========================================
    # STEP 5: SAVE UPDATED HISTORY
    # ==========================================

    history_df.to_csv(
        HISTORY_PATH,
        index=False
    )

    # ==========================================
    # STEP 6: DISPLAY UPDATE SUMMARY
    # ==========================================

    total_completed = (
        history_df["Prediction_Status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("completed")
        .sum()
    )

    total_pending = (
        history_df["Prediction_Status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("pending")
        .sum()
    )

    print("\n===================================")
    print("       UPDATE SUMMARY")
    print("===================================")

    print("Newly completed:", completed_count)
    print("Total completed:", total_completed)
    print("Total pending:", total_pending)

    print("\nUpdated history saved successfully!")

    print("===================================\n")


# ==========================================
# STEP 7: RUN PROGRAM
# ==========================================

if __name__ == "__main__":
    update_prediction_history()