import yfinance as yf
import pandas as pd

# Select company
stock_symbol = "RELIANCE.NS"

# Download historical data
data = yf.download(
    stock_symbol,
    period="5y",
    interval="1d",
    auto_adjust=True
)

# Display first 10 records
print(data.head(10))

# Save data into CSV
data.to_csv("dataset/RELIANCE.csv")

print("Stock data downloaded successfully!")