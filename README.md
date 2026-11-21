📈 Reliance Stock Market Intelligence & Forecasting

An interactive stock market analysis and machine learning forecasting application built using Python, Streamlit, Scikit-learn, and Yahoo Finance data.

This project analyzes historical stock market data of Reliance Industries Limited (NSE: RELIANCE), generates live price predictions, forecasts stock prices for the next five trading sessions, and evaluates forecasting performance using machine learning models.

🚀 Project Overview

The main objective of this project is to develop a data-driven stock market intelligence system that combines technical analysis, machine learning, interactive visualizations, and multi-day price forecasting in a single dashboard.

The application uses historical stock market data and technical indicators to train predictive models and generate future price estimates.

✨ Key Features

1. Stock Market Overview

- Historical stock price visualization.
- Open, High, Low, Close and Volume analysis.
- Interactive dashboard.

2. Technical Analysis

- Moving Averages (MA5, MA10, MA20, MA50).
- Relative Strength Index (RSI).
- Moving Average Convergence Divergence (MACD).
- Bollinger Bands.
- Daily returns analysis.

3. Machine Learning Model Evaluation

- Model performance comparison.
- Mean Absolute Error (MAE).
- Root Mean Squared Error (RMSE).
- R² Score.

4. Price Prediction

- Machine learning-based stock price estimation.
- Actual versus predicted price comparison.
- Interactive prediction visualizations.

5. Live Prediction

- Retrieves the latest available stock market data.
- Generates predictions using trained machine learning models.
- Uses updated technical indicators.

6. Five-Day Stock Forecasting

- Separate forecasting models for five future trading sessions.
- Day-wise predicted closing prices.
- Interactive five-day forecast visualization.

7. Forecast Performance

- Evaluation of individual forecasting horizons.
- Comparison of prediction errors across five days.

8. Prediction History

- Stores previous forecasts.
- Tracks predicted and actual closing prices.
- Updates forecast status when actual market data becomes available.
- Analyzes historical forecasting errors.

9. Feature Importance Analysis

- Identifies important input features.
- Compares feature importance across five forecasting models.
- Interactive feature importance visualizations.

🛠️ Technologies Used

Technology| Purpose
Python| Programming language
Pandas| Data manipulation
NumPy| Numerical calculations
Scikit-learn| Machine learning
Random Forest Regressor| Stock price forecasting
Yahoo Finance (yfinance)| Stock market data
Streamlit| Interactive dashboard
Plotly| Data visualization
Joblib| Model serialization

📂 Project Structure

Stock-Market-Forecasting/
│
├── app.py
├── live_prediction.py
├── feature_engineering.py
├── train_multi_day_models.py
├── update_prediction_history.py
├── analyze_feature_importance.py
├── compare_feature_importance.py
│
├── dataset/
│   ├── RELIANCE_cleaned.csv
│   └── RELIANCE_features.csv
│
├── models/
│   ├── best_model.pkl
│   ├── forecast_day_1.pkl
│   ├── forecast_day_2.pkl
│   ├── forecast_day_3.pkl
│   ├── forecast_day_4.pkl
│   └── forecast_day_5.pkl
│
├── history/
│   └── forecast_history.csv
│
├── requirements.txt
└── README.md

The structure above is representative. Update it if your actual repository contains additional or differently named files.

