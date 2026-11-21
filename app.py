import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from pathlib import Path

from live_prediction import (
    get_latest_live_features
)

# ==========================================
# STEP 1: PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Reliance Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "RELIANCE_features.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
PREDICTION_PATH = BASE_DIR / "models" / "predictions.csv"
COMPARISON_PATH = BASE_DIR / "models" / "model_comparison.csv"

FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Daily_Return", "MA5", "MA10", "MA20", "MA50",
    "RSI", "MACD", "Signal_Line", "BB_Upper",
    "BB_Lower", "Previous_Close",
    "Previous_Return", "Previous_Volume"
]


# ==========================================
# STEP 2: CUSTOM PROFESSIONAL DESIGN
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
    color: #E5E7EB;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263247;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #F8FAFC !important;
}

.metric-card {
    background: linear-gradient(135deg, #17243A, #111827);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #293B55;
}

.metric-label {
    color: #94A3B8;
    font-size: 14px;
}

.metric-value {
    color: #F8FAFC;
    font-size: 27px;
    font-weight: bold;
}

.metric-delta {
    color: #22C55E;
    font-size: 13px;
}

div[data-testid="stMetric"] {
    background: #17243A;
    border: 1px solid #293B55;
    padding: 15px;
    border-radius: 12px;
}

.stButton button {
    background-color: #2563EB;
    color: white;
    border-radius: 8px;
}

hr {
    border-color: #293B55;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# STEP 3: LOAD DATA AND MODEL
# ==========================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True
    )

    predictions = pd.read_csv(
        PREDICTION_PATH,
        index_col=0,
        parse_dates=True
    )

    comparison = pd.read_csv(COMPARISON_PATH)

    data = data.sort_index()
    predictions = predictions.sort_index()

    return data, predictions, comparison


@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


# ==========================================
# STEP 4: CHECK REQUIRED FILES
# ==========================================

required_files = [
    DATA_PATH,
    MODEL_PATH,
    PREDICTION_PATH,
    COMPARISON_PATH
]

missing_files = [
    file.name
    for file in required_files
    if not file.exists()
]

if missing_files:

    st.error("Some required project files are missing.")

    for file in missing_files:
        st.write("❌", file)

    st.stop()


try:

    data, predictions, comparison = load_data()
    model = load_model()

except Exception as error:

    st.error(f"Unable to load project data: {error}")
    st.stop()


# Validate columns

missing_features = [
    column for column in FEATURES
    if column not in data.columns
]

if missing_features:

    st.error(f"Missing features: {missing_features}")
    st.stop()

if "Close" not in data.columns:

    st.error("Closing price column is missing.")
    st.stop()

if data.empty:

    st.error("Dataset is empty.")
    st.stop()


# ==========================================
# STEP 5: SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("📈 RELIANCE")
st.sidebar.caption("STOCK INTELLIGENCE PLATFORM")

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Technical Analysis",
        "Model Evaluation",
        "Price Prediction",
        "Live Prediction",
        "5-Day Forecast",
        "Forecast Performance",
        "Prediction History"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Company Information")

st.sidebar.write("**Company:** Reliance Industries")
st.sidebar.write("**Symbol:** RELIANCE.NS")
st.sidebar.write("**Exchange:** NSE India")
st.sidebar.write("**Sector:** Diversified")

st.sidebar.divider()

st.sidebar.caption(
    "Powered by Python | Pandas | Scikit-learn | Streamlit"
)


# ==========================================
# STEP 6: GLOBAL HEADER
# ==========================================

st.title("📊 Reliance Stock Intelligence")

st.markdown(
    "Advanced stock market analysis and "
    "machine learning forecasting platform."
)

st.caption(
    f"Dataset: {data.index.min().strftime('%d %b %Y')} "
    f"to {data.index.max().strftime('%d %b %Y')}"
)

st.divider()


# ==========================================
# STEP 7: OVERVIEW DASHBOARD
# ==========================================

if page == "Overview":

    st.header("Market Overview")

    valid_close = data["Close"].dropna()

    latest_close = valid_close.iloc[-1]

    previous_close = (
        valid_close.iloc[-2]
        if len(valid_close) > 1
        else latest_close
    )

    daily_change = latest_close - previous_close

    daily_change_percent = (
        daily_change / previous_close * 100
        if previous_close != 0
        else 0
    )

    highest_price = valid_close.max()
    lowest_price = valid_close.min()

    total_records = len(data)

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Latest Closing Price",
        f"₹{latest_close:,.2f}",
        f"{daily_change_percent:+.2f}%"
    )

    col2.metric(
        "Previous Close",
        f"₹{previous_close:,.2f}"
    )

    col3.metric(
        "Highest Recorded Price",
        f"₹{highest_price:,.2f}"
    )

    col4.metric(
        "Total Trading Records",
        f"{total_records:,}"
    )

    st.divider()

    # STOCK PRICE CHART

    st.subheader("Historical Stock Performance")

    period = st.selectbox(
        "Select Time Period",
        ["All", "1 Year", "6 Months", "3 Months"]
    )

    chart_data = data.copy()

    if period == "1 Year":
        chart_data = chart_data.tail(252)

    elif period == "6 Months":
        chart_data = chart_data.tail(126)

    elif period == "3 Months":
        chart_data = chart_data.tail(63)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["Close"],
            name="Closing Price",
            mode="lines",
            line=dict(
                color="#22D3EE",
                width=2
            ),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,0.07)"
        )
    )

    fig.update_layout(
        title="RELIANCE Historical Stock Price",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#111827",
        height=480,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # TRADING VOLUME

    st.subheader("Trading Volume Analysis")

    volume_fig = go.Figure()

    volume_fig.add_trace(
        go.Bar(
            x=chart_data.index,
            y=chart_data["Volume"],
            name="Trading Volume",
            marker_color="#3B82F6"
        )
    )

    volume_fig.update_layout(
        title="Daily Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#111827",
        height=350
    )

    st.plotly_chart(
        volume_fig,
        use_container_width=True
    )


# ==========================================
# STEP 8: TECHNICAL ANALYSIS
# ==========================================

elif page == "Technical Analysis":

    st.header("Technical Indicator Analysis")

    # MOVING AVERAGES

    st.subheader("Moving Average Analysis")

    selected_ma = st.multiselect(
        "Select Moving Averages",
        ["MA5", "MA10", "MA20", "MA50"],
        default=["MA5", "MA20", "MA50"]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            name="Closing Price",
            line=dict(color="#FFFFFF", width=2)
        )
    )

    colors = [
        "#22D3EE",
        "#F59E0B",
        "#A78BFA",
        "#22C55E"
    ]

    for i, column in enumerate(selected_ma):

        if column in data.columns:

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[column],
                    name=column,
                    line=dict(
                        color=colors[i % len(colors)],
                        width=1.5
                    )
                )
            )

    fig.update_layout(
        title="Closing Price vs Moving Averages",
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#111827",
        height=500,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # RSI INDICATOR

    st.subheader("Relative Strength Index (RSI)")

    if "RSI" in data.columns:

        rsi_fig = go.Figure()

        rsi_fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["RSI"],
                name="RSI",
                line=dict(
                    color="#A78BFA",
                    width=2
                )
            )
        )

        rsi_fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text="Overbought"
        )

        rsi_fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="#22C55E",
            annotation_text="Oversold"
        )

        rsi_fig.update_layout(
            title="RSI Indicator",
            yaxis_title="RSI",
            yaxis_range=[0, 100],
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            plot_bgcolor="#111827",
            height=350
        )

        st.plotly_chart(
            rsi_fig,
            use_container_width=True
        )

    st.divider()

    # MACD INDICATOR

    st.subheader("MACD Analysis")

    if all(
        column in data.columns
        for column in ["MACD", "Signal_Line"]
    ):

        macd_fig = go.Figure()

        macd_fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MACD"],
                name="MACD",
                line=dict(color="#22D3EE")
            )
        )

        macd_fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Signal_Line"],
                name="Signal Line",
                line=dict(color="#F59E0B")
            )
        )

        macd_fig.update_layout(
            title="MACD vs Signal Line",
            template="plotly_dark",
            paper_bgcolor="#0B1120",
            plot_bgcolor="#111827",
            height=350
        )

        st.plotly_chart(
            macd_fig,
            use_container_width=True
        )


# ==========================================
# STEP 9: MODEL EVALUATION
# ==========================================

elif page == "Model Evaluation":

    st.header("Machine Learning Performance")

    st.write(
        "Compare machine learning models "
        "using different evaluation metrics."
    )

    st.subheader("Model Comparison")

    st.dataframe(
        comparison.style.format({
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "R2 Score": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # MODEL SELECTION

    available_models = [
        column for column in predictions.columns
        if column not in ["Actual_Price"]
    ]

    if not available_models:

        st.error("No model predictions available.")
        st.stop()

    selected_model = st.selectbox(
        "Select Machine Learning Model",
        available_models
    )

    # ACTUAL VS PREDICTED

    st.subheader("Actual vs Predicted Price")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=predictions["Actual_Price"],
            name="Actual Price",
            line=dict(
                color="#FFFFFF",
                width=2
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=predictions[selected_model],
            name="Predicted Price",
            line=dict(
                color="#22D3EE",
                width=2
            )
        )
    )

    fig.update_layout(
        title=f"Model Prediction: {selected_model}",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#111827",
        height=450,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # MODEL METRICS

    result = comparison[
        comparison["Model"] == selected_model
    ]

    if not result.empty:

        metrics = result.iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Mean Absolute Error",
            f"{metrics['MAE']:.4f}"
        )

        col2.metric(
            "Root Mean Squared Error",
            f"{metrics['RMSE']:.4f}"
        )

        col3.metric(
            "R² Score",
            f"{metrics['R2 Score']:.4f}"
        )

    st.info(
        "Lower MAE and RMSE generally indicate "
        "smaller prediction errors. R² should be "
        "interpreted together with the other metrics."
    )


# ==========================================
# STEP 10: NEXT SESSION PRICE PREDICTION
# ==========================================

elif page == "Price Prediction":

    st.header("🔮 Next Session Price Prediction")

    st.write(
        "Estimate the next trading session's "
        "closing price using the trained machine learning model."
    )

    available_data = data.dropna(
        subset=FEATURES
    )

    if available_data.empty:

        st.error("No complete feature records available.")
        st.stop()

    latest_row = available_data.iloc[[-1]]

    latest_features = latest_row[FEATURES]

    # CHECK MODEL INPUT

    if (
        hasattr(model, "n_features_in_")
        and model.n_features_in_ != len(FEATURES)
    ):

        st.error(
            "The model feature count does not match "
            "the dashboard feature count."
        )

        st.stop()

    # PREDICTION

    predicted_price = model.predict(
        latest_features
    )[0]

    latest_close = float(
        latest_row["Close"].iloc[0]
    )

    latest_date = latest_row.index[0]

    price_difference = (
        predicted_price - latest_close
    )

    percentage_change = (
        price_difference / latest_close * 100
        if latest_close != 0
        else 0
    )

    # PREDICTION RESULT

    st.divider()

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Latest Closing Price",
        f"₹{latest_close:,.2f}"
    )

    col2.metric(
        "Predicted Next Price",
        f"₹{predicted_price:,.2f}",
        f"{price_difference:+.2f}"
    )

    col3.metric(
        "Estimated Change",
        f"{percentage_change:+.2f}%"
    )

    st.divider()

    # VISUAL COMPARISON

    st.subheader("Price Comparison")

    prediction_fig = go.Figure()

    prediction_fig.add_trace(
        go.Bar(
            x=[
                "Latest Closing Price",
                "Predicted Next Price"
            ],
            y=[
                latest_close,
                predicted_price
            ],
            marker_color=[
                "#3B82F6",
                "#22D3EE"
            ],
            text=[
                f"₹{latest_close:.2f}",
                f"₹{predicted_price:.2f}"
            ],
            textposition="outside"
        )
    )

    prediction_fig.update_layout(
        title="Actual vs Estimated Price",
        yaxis_title="Price (₹)",
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        plot_bgcolor="#111827",
        height=400
    )

    st.plotly_chart(
        prediction_fig,
        use_container_width=True
    )

    st.info(
        f"Prediction generated using feature data "
        f"dated {latest_date.strftime('%d-%m-%Y')}."
    )

    st.warning(
        "This is an experimental machine learning estimate. "
        "It is not a guaranteed future market price "
        "or investment recommendation."
    )

# ==========================================
# STEP 11: LIVE STOCK MARKET PREDICTION
# ==========================================

elif page == "Live Prediction":

    st.header("📡 Live Stock Market Prediction")

    st.write(
        "Fetch the latest available market data "
        "and predict the next trading session's closing price."
    )

    if st.button("Fetch Latest Market Data"):

        try:

            with st.spinner("Fetching latest market data..."):

                # Fetch latest available market features
                latest_row, live_features = get_latest_live_features()

                # Select features in the exact training order
                latest_features = live_features[FEATURES].iloc[[-1]]

                # Generate prediction
                predicted_price = float(
                    model.predict(latest_features)[0]
                )

                latest_close = float(latest_row["Close"])

                latest_date = latest_row.name

                price_difference = (
                    predicted_price - latest_close
                )

                percentage_change = (
                    price_difference / latest_close * 100
                    if latest_close != 0
                    else 0
                )

            st.success("Prediction generated successfully!")

            st.divider()

            st.subheader("Latest Market Prediction")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Latest Available Close",
                f"₹{latest_close:,.2f}"
            )

            col2.metric(
                "Predicted Next Close",
                f"₹{predicted_price:,.2f}"
            )

            col3.metric(
                "Expected Change",
                f"{percentage_change:+.2f}%",
                f"₹{price_difference:+.2f}"
            )

            st.divider()

            st.subheader("Latest Technical Indicators")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "RSI",
                f"{float(latest_row['RSI']):.2f}"
            )

            col2.metric(
                "MACD",
                f"{float(latest_row['MACD']):.2f}"
            )

            col3.metric(
                "MA50",
                f"₹{float(latest_row['MA50']):,.2f}"
            )

            st.write(
                f"**Latest trading data date:** "
                f"{latest_date.strftime('%d-%m-%Y')}"
            )

            st.divider()

            st.subheader("Price Comparison")

            live_fig = go.Figure()

            live_fig.add_trace(
                go.Bar(
                    x=[
                        "Latest Closing Price",
                        "Predicted Next Price"
                    ],
                    y=[
                        latest_close,
                        predicted_price
                    ],
                    marker_color=[
                        "#3B82F6",
                        "#22D3EE"
                    ],
                    text=[
                        f"₹{latest_close:.2f}",
                        f"₹{predicted_price:.2f}"
                    ],
                    textposition="outside"
                )
            )

            live_fig.update_layout(
                title="Latest Price vs Predicted Price",
                yaxis_title="Price (₹)",
                template="plotly_dark",
                paper_bgcolor="#0B1120",
                plot_bgcolor="#111827",
                height=400
            )

            st.plotly_chart(
                live_fig,
                use_container_width=True
            )

            st.warning(
                "This is an experimental machine learning prediction, "
                "not a guaranteed future stock price or investment advice."
            )

        except Exception as error:

            st.error(
                f"Unable to generate live prediction: {error}"
            )

# ==========================================
# STEP 12: 5-DAY STOCK PRICE FORECAST
# ==========================================

elif page == "5-Day Forecast":

    st.header("📅 5-Day Stock Price Forecast")

    st.write(
        "Forecast Reliance Industries closing prices "
        "for the next five trading sessions."
    )

    st.info(
        "Each forecast is generated using a separately "
        "trained Random Forest model."
    )

    if st.button("Generate 5-Day Forecast"):

        try:

            with st.spinner("Generating five-day forecasts..."):

                # Fetch latest available market features
                latest_row, live_features = get_latest_live_features()

                # Select latest feature record
                latest_features = live_features[FEATURES].iloc[[-1]]

                latest_close = float(latest_row["Close"])

                forecast_results = []

                # Load and predict using five models
                for day in range(1, 6):

                    model_path = (
                        BASE_DIR
                        / "models"
                        / f"forecast_day_{day}.pkl"
                    )

                    if not model_path.exists():
                        raise FileNotFoundError(
                            f"Missing model: {model_path.name}"
                        )

                    forecast_model = joblib.load(model_path)

                    predicted_price = float(
                        forecast_model.predict(latest_features)[0]
                    )

                    change = predicted_price - latest_close

                    percentage = (
                        change / latest_close * 100
                        if latest_close != 0
                        else 0
                    )

                    forecast_results.append({
                        "Trading Session": f"Day {day}",
                        "Predicted Closing Price": predicted_price,
                        "Change from Latest Close": change,
                        "Change (%)": percentage
                    })

            # Convert results into DataFrame
            forecast_df = pd.DataFrame(forecast_results)

            # ==========================================
            # SAVE FORECAST HISTORY - PREVENT DUPLICATES
            # ==========================================

            history_dir = BASE_DIR / "history"
            history_dir.mkdir(parents=True, exist_ok=True)

            history_path = history_dir / "forecast_history.csv"

            # Get forecast origin date
            origin_date = pd.Timestamp(latest_row.name).normalize()

            # Generate timestamp    
            generated_at = pd.Timestamp.now()

            # Create new history records
            history_records = []

            for _, row in forecast_df.iterrows():

                history_records.append({
                    "Generated_At": generated_at,
                    "Origin_Date": origin_date,
                    "Forecast_Day": row["Trading Session"],
                    "Latest_Close": latest_close,
                    "Predicted_Close": row["Predicted Closing Price"],
                    "Actual_Close": np.nan,
                    "Prediction_Status": "Pending"
                })

                new_history = pd.DataFrame(history_records)

                # Load existing history
                if history_path.exists():

                    old_history = pd.read_csv(history_path)

                # Normalize existing dates
                old_history["Origin_Date"] = pd.to_datetime(
                    old_history["Origin_Date"],
                    format="mixed",
                    errors="coerce"
                ).dt.normalize()

                # Create existing forecast keys
                existing_keys = set(
                    zip(
                        old_history["Origin_Date"],
                        old_history["Forecast_Day"].astype(str)
                    )
                )

            else:

                old_history = pd.DataFrame()
                existing_keys = set()

            # Find only new forecast records
            new_records = []

            for _, row in new_history.iterrows():

                forecast_key = (
                    row["Origin_Date"],
                    str(row["Forecast_Day"])
                )   

                if forecast_key not in existing_keys:

                    new_records.append(row)

                    # Prevent duplicates within this same batch
                    existing_keys.add(forecast_key)

                # Save only new records
                if new_records:

                    new_history = pd.DataFrame(new_records)

                    updated_history = pd.concat(
                        [old_history, new_history],
                        ignore_index=True
                    )

                updated_history.to_csv(
                    history_path,
                    index=False
                )

                st.success(
                    f"{len(new_records)} new forecast records saved successfully!"
                )

            else:

                st.info(
                    "Forecasts for this origin date already exist. "
                    "Duplicate records were not saved."
                )

            # ==========================================
            # DISPLAY FORECAST RESULTS
            # ==========================================

            st.success("Five-day forecasting completed!")

            st.divider()

            # Latest available price
            st.subheader("Latest Market Information")

            col1, col2 = st.columns(2)

            col1.metric(
                "Latest Available Closing Price",
                f"₹{latest_close:,.2f}"
            )

            col2.write(
                f"**Latest trading date:** "
                f"{latest_row.name.strftime('%d-%m-%Y')}"
            )

            st.divider()

            # Forecast table
            st.subheader("Five-Day Forecast Results")

            display_df = forecast_df.copy()

            display_df["Predicted Closing Price"] = (
                display_df["Predicted Closing Price"]
                .map(lambda x: f"₹{x:,.2f}")
            )

            display_df["Change from Latest Close"] = (
                display_df["Change from Latest Close"]
                .map(lambda x: f"₹{x:+,.2f}")
            )

            display_df["Change (%)"] = (
                display_df["Change (%)"]
                .map(lambda x: f"{x:+.2f}%")
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # Forecast chart
            st.subheader("Five-Day Forecast Chart")

            chart_fig = go.Figure()

            chart_fig.add_trace(
                go.Scatter(
                    x=["Latest Close"] + forecast_df[
                        "Trading Session"
                    ].tolist(),

                    y=[latest_close] + forecast_df[
                        "Predicted Closing Price"
                    ].tolist(),

                    mode="lines+markers+text",

                    text=[
                        f"₹{latest_close:.2f}"
                    ] + [
                        f"₹{price:.2f}"
                        for price in forecast_df[
                            "Predicted Closing Price"
                        ]
                    ],

                    textposition="top center",

                    name="Forecasted Price",

                    line=dict(
                        color="#22D3EE",
                        width=3
                    ),

                    marker=dict(size=10)
                )
            )

            chart_fig.update_layout(
                title="Reliance Five-Day Price Forecast",
                xaxis_title="Trading Session",
                yaxis_title="Predicted Price (₹)",
                template="plotly_dark",
                paper_bgcolor="#0B1120",
                plot_bgcolor="#111827",
                height=500
            )

            st.plotly_chart(
                chart_fig,
                use_container_width=True
            )

            st.divider()

            # Model information
            st.subheader("Forecasting Model Information")

            st.write("Algorithm: Random Forest Regressor")
            st.write("Number of models: 5")
            st.write("Forecast horizon: 5 trading sessions")

            st.warning(
                "These are experimental model estimates, not "
                "guaranteed future prices or investment advice."
            )

        except Exception as error:

            st.error(
                f"Unable to generate five-day forecast: {error}"
            )



# ==========================================
# STEP 13: FORECAST PERFORMANCE ANALYSIS
# ==========================================

elif page == "Forecast Performance":

    st.header("📊 Forecast Performance Analysis")

    st.write(
        "Evaluate the performance of all five "
        "Random Forest forecasting models."
    )

    # Load performance data
    performance_path = (
        BASE_DIR
        / "models"
        / "multi_day_model_comparison.csv"
    )

    if not performance_path.exists():

        st.error(
            "Performance CSV file not found. "
            "Please run train_multi_day_models.py first."
        )

    else:

        performance_df = pd.read_csv(performance_path)

        # Validate required columns
        required_columns = [
            "Forecast_Day",
            "Training_Records",
            "Testing_Records",
            "MAE",
            "RMSE",
            "R2 Score"
        ]

        if not all(
            col in performance_df.columns
            for col in required_columns
        ):

            st.error("CSV columns do not match expected format.")

        else:

            performance_df = performance_df.sort_values(
                "Forecast_Day"
            )

            performance_df["Forecast_Label"] = (
                "Day "
                + performance_df["Forecast_Day"].astype(str)
            )

            st.success(
                "Forecast performance data loaded successfully!"
            )

            st.divider()

            # ==================================
            # PERFORMANCE SUMMARY
            # ==================================

            st.subheader("Performance Summary")

            best_model = performance_df.loc[
                performance_df["MAE"].idxmin()
            ]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Best Forecast Horizon",
                f"Day {int(best_model['Forecast_Day'])}"
            )

            col2.metric(
                "Lowest MAE",
                f"₹{best_model['MAE']:.2f}"
            )

            col3.metric(
                "Best R² Score",
                f"{performance_df['R2 Score'].max():.4f}"
            )

            st.divider()

            # ==================================
            # PERFORMANCE TABLE
            # ==================================

            st.subheader("Detailed Model Performance")

            display_df = performance_df[
                [
                    "Forecast_Label",
                    "Training_Records",
                    "Testing_Records",
                    "MAE",
                    "RMSE",
                    "R2 Score"
                ]
            ].copy()

            display_df = display_df.rename(
                columns={
                    "Forecast_Label": "Forecast Day",
                    "Training_Records": "Training Records",
                    "Testing_Records": "Testing Records",
                    "R2 Score": "R² Score"
                }
            )

            st.dataframe(
                display_df.style.format({
                    "MAE": "₹{:.2f}",
                    "RMSE": "₹{:.2f}",
                    "R² Score": "{:.4f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # ==================================
            # MAE CHART
            # ==================================

            st.subheader("Mean Absolute Error (MAE)")

            mae_fig = go.Figure()

            mae_fig.add_trace(
                go.Bar(
                    x=performance_df["Forecast_Label"],
                    y=performance_df["MAE"],
                    text=[
                        f"₹{value:.2f}"
                        for value in performance_df["MAE"]
                    ],
                    textposition="outside",
                    name="MAE"
                )
            )

            mae_fig.update_layout(
                title="MAE Comparison Across Forecast Days",
                xaxis_title="Forecast Horizon",
                yaxis_title="MAE (₹)",
                template="plotly_dark",
                height=400
            )

            st.plotly_chart(
                mae_fig,
                use_container_width=True
            )

            st.caption(
                "Lower MAE indicates smaller average prediction errors."
            )

            st.divider()

            # ==================================
            # RMSE CHART
            # ==================================

            st.subheader("Root Mean Squared Error (RMSE)")

            rmse_fig = go.Figure()

            rmse_fig.add_trace(
                go.Bar(
                    x=performance_df["Forecast_Label"],
                    y=performance_df["RMSE"],
                    text=[
                        f"₹{value:.2f}"
                        for value in performance_df["RMSE"]
                    ],
                    textposition="outside",
                    name="RMSE"
                )
            )

            rmse_fig.update_layout(
                title="RMSE Comparison Across Forecast Days",
                xaxis_title="Forecast Horizon",
                yaxis_title="RMSE (₹)",
                template="plotly_dark",
                height=400
            )

            st.plotly_chart(
                rmse_fig,
                use_container_width=True
            )

            st.caption(
                "Lower RMSE indicates fewer or smaller large prediction errors."
            )

            st.divider()

            # ==================================
            # R2 SCORE CHART
            # ==================================

            st.subheader("R² Score Comparison")

            r2_fig = go.Figure()

            r2_fig.add_trace(
                go.Scatter(
                    x=performance_df["Forecast_Label"],
                    y=performance_df["R2 Score"],
                    mode="lines+markers+text",
                    text=[
                        f"{value:.4f}"
                        for value in performance_df["R2 Score"]
                    ],
                    textposition="top center",
                    name="R² Score"
                )
            )

            r2_fig.update_layout(
                title="R² Score Across Forecast Days",
                xaxis_title="Forecast Horizon",
                yaxis_title="R² Score",
                template="plotly_dark",
                height=400
            )

            st.plotly_chart(
                r2_fig,
                use_container_width=True
            )

            st.caption(
                "Higher R² generally indicates better fit on the test data."
            )

            st.divider()

            # ==================================
            # FINAL INTERPRETATION
            # ==================================

            st.subheader("Model Performance Interpretation")

            st.write(
                f"""
                - Best forecast horizon: Day {int(best_model['Forecast_Day'])}
                - Lowest MAE: ₹{best_model['MAE']:.2f}
                - Lowest RMSE: ₹{best_model['RMSE']:.2f}
                - Highest R² Score: {performance_df['R2 Score'].max():.4f}
                """
            )

            st.info(
                "The forecasting models generally show decreasing "
                "performance as the prediction horizon increases. "
                "These metrics are based on historical test data "
                "and do not guarantee future performance."
            )

# ==========================================
# STEP 14: PREDICTION HISTORY & ACCURACY
# ==========================================

elif page == "Prediction History":

    st.header("📜 Prediction History & Accuracy")

    st.write(
        "Track historical forecasts and evaluate "
        "predicted prices against actual market prices."
    )

    history_path = (
        BASE_DIR / "history" / "forecast_history.csv"
    )

    if not history_path.exists():

        st.warning(
            "Prediction history not found. "
            "Generate a five-day forecast first."
        )

    else:

        history_df = pd.read_csv(history_path)

        if history_df.empty:

            st.info("Prediction history is empty.")

        else:

            # ==================================
            # STEP 1: DATA PREPARATION
            # ==================================

            required_columns = [
                "Origin_Date",
                "Forecast_Day",
                "Predicted_Close",
                "Actual_Close",
                "Prediction_Status",
                "Actual_Date"
            ]

            for col in required_columns:
                if col not in history_df.columns:
                    history_df[col] = np.nan

            history_df["Predicted_Close"] = pd.to_numeric(
                history_df["Predicted_Close"],
                errors="coerce"
            )

            history_df["Actual_Close"] = pd.to_numeric(
                history_df["Actual_Close"],
                errors="coerce"
            )

            history_df["Origin_Date"] = pd.to_datetime(
                history_df["Origin_Date"],
                format="mixed",
                errors="coerce"
            )

            history_df["Actual_Date"] = pd.to_datetime(
                history_df["Actual_Date"],
                errors="coerce"
            )

            history_df["Prediction_Status"] = (
                history_df["Prediction_Status"]
                .fillna("Pending")
                .astype(str)
                .str.strip()
            )

            # ==================================
            # STEP 2: HISTORY SUMMARY
            # ==================================

            completed_df = history_df[
                (
                    history_df["Prediction_Status"]
                    .str.lower() == "completed"
                )
                & history_df["Actual_Close"].notna()
                & history_df["Predicted_Close"].notna()
            ].copy()

            pending_count = (
                history_df["Prediction_Status"]
                .str.lower() == "pending"
            ).sum()

            completed_count = len(completed_df)

            st.subheader("History Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Predictions",
                len(history_df)
            )

            col2.metric(
                "Completed",
                completed_count
            )

            col3.metric(
                "Pending",
                pending_count
            )

            st.divider()

            # ==================================
            # STEP 3: COMPLETE HISTORY TABLE
            # ==================================

            st.subheader("All Prediction Records")

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # ==================================
            # STEP 4: ACCURACY ANALYSIS
            # ==================================

            st.subheader("Prediction Accuracy Analysis")

            if not completed_df.empty:

                # Sort records
                if "Generated_At" in completed_df.columns:

                    completed_df["Generated_At"] = (
                        pd.to_datetime(
                            completed_df["Generated_At"],
                            errors="coerce"
                        )
                    )

                    completed_df = completed_df.sort_values(
                        "Generated_At"
                    )

                # Remove duplicate forecasts from analysis.
                # Keep the latest record for each origin date/day.

                completed_df = completed_df.drop_duplicates(
                    subset=["Origin_Date", "Forecast_Day"],
                    keep="last"
                )

                # Calculate prediction errors

                completed_df["Prediction_Error"] = (
                    completed_df["Predicted_Close"]
                    - completed_df["Actual_Close"]
                )

                completed_df["Absolute_Error"] = (
                    completed_df["Prediction_Error"].abs()
                )

                completed_df["Squared_Error"] = (
                    completed_df["Prediction_Error"] ** 2
                )

                completed_df["Percentage_Error"] = (
                    completed_df["Absolute_Error"]
                    / completed_df["Actual_Close"].replace(
                        0, np.nan
                    ).abs()
                ) * 100

                # Overall metrics

                mae = completed_df["Absolute_Error"].mean()

                rmse = np.sqrt(
                    completed_df["Squared_Error"].mean()
                )

                mape = completed_df[
                    "Percentage_Error"
                ].mean()

                accuracy = (
                    100 - mape
                    if pd.notna(mape)
                    else np.nan
                )

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "MAE",
                    f"₹{mae:,.2f}"
                )

                col2.metric(
                    "RMSE",
                    f"₹{rmse:,.2f}"
                )

                col3.metric(
                    "MAPE",
                    f"{mape:.2f}%"
                    if pd.notna(mape)
                    else "N/A"
                )

                col4.metric(
                    "100 - MAPE",
                    f"{accuracy:.2f}%"
                    if pd.notna(accuracy)
                    else "N/A"
                )

                st.caption(
                    "100 - MAPE is an error-derived indicator, "
                    "not a formal prediction accuracy score. "
                    "It can also be negative."
                )

                st.divider()

                # ==================================
                # STEP 5: PERFORMANCE BY FORECAST DAY
                # ==================================

                st.subheader(
                    "Forecast Day Performance Comparison"
                )

                day_performance = (
                    completed_df
                    .groupby("Forecast_Day")
                    .agg(
                        MAE=("Absolute_Error", "mean"),
                        RMSE=("Squared_Error",
                              lambda x: np.sqrt(x.mean())),
                        MAPE=("Percentage_Error", "mean"),
                        Predictions=("Actual_Close", "count")
                    )
                    .reset_index()
                )

                day_performance = day_performance.sort_values(
                    "Forecast_Day"
                )

                st.dataframe(
                    day_performance.style.format({
                        "MAE": "₹{:.2f}",
                        "RMSE": "₹{:.2f}",
                        "MAPE": "{:.2f}%",
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                # Identify best-performing forecast day

                valid_days = day_performance.dropna(
                    subset=["MAE"]
                )

                if not valid_days.empty:

                    best_day = valid_days.loc[
                        valid_days["MAE"].idxmin()
                    ]

                    st.success(
                        f"Best-performing forecast horizon: "
                        f"{best_day['Forecast_Day']} "
                        f"(MAE: ₹{best_day['MAE']:.2f})"
                    )

                st.divider()

                # ==================================
                # STEP 6: MAE COMPARISON CHART
                # ==================================

                st.subheader("MAE by Forecast Day")

                mae_fig = go.Figure()

                mae_fig.add_trace(
                    go.Bar(
                        x=day_performance["Forecast_Day"],
                        y=day_performance["MAE"],
                        text=day_performance["MAE"].round(2),
                        textposition="auto",
                        name="MAE"
                    )
                )

                mae_fig.update_layout(
                    title="Mean Absolute Error by Forecast Horizon",
                    xaxis_title="Forecast Day",
                    yaxis_title="MAE (₹)",
                    template="plotly_dark",
                    height=400
                )

                st.plotly_chart(
                    mae_fig,
                    use_container_width=True
                )

                st.divider()

                # ==================================
                # STEP 7: ACTUAL VS PREDICTED CHART
                # ==================================

                st.subheader(
                    "Actual vs Predicted Closing Prices"
                )

                chart_df = completed_df.copy()

                if chart_df["Actual_Date"].notna().any():

                    chart_df = chart_df.sort_values(
                        "Actual_Date"
                    )

                    x_values = chart_df["Actual_Date"]

                else:

                    x_values = chart_df.index

                comparison_fig = go.Figure()

                comparison_fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=chart_df["Predicted_Close"],
                        mode="lines+markers",
                        name="Predicted Price"
                    )
                )

                comparison_fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=chart_df["Actual_Close"],
                        mode="lines+markers",
                        name="Actual Price"
                    )
                )

                comparison_fig.update_layout(
                    title="Actual vs Predicted Stock Prices",
                    xaxis_title="Actual Trading Date",
                    yaxis_title="Closing Price (₹)",
                    template="plotly_dark",
                    height=450
                )

                st.plotly_chart(
                    comparison_fig,
                    use_container_width=True
                )

                st.divider()

                # ==================================
                # STEP 8: PREDICTION ERROR CHART
                # ==================================

                st.subheader("Prediction Error Analysis")

                error_fig = go.Figure()

                error_fig.add_trace(
                    go.Bar(
                        x=chart_df["Forecast_Day"],
                        y=chart_df["Prediction_Error"],
                        name="Prediction Error"
                    )
                )

                error_fig.add_hline(
                    y=0,
                    line_dash="dash"
                )

                error_fig.update_layout(
                    title="Predicted Price - Actual Price",
                    xaxis_title="Forecast Day",
                    yaxis_title="Prediction Error (₹)",
                    template="plotly_dark",
                    height=400
                )

                st.plotly_chart(
                    error_fig,
                    use_container_width=True
                )

            else:

                st.info(
                    "No completed predictions available yet. "
                    "Metrics and charts will appear when actual "
                    "market closing prices become available."
                )
# ==========================================
# STEP 15: FOOTER
# ==========================================

st.divider()

col1, col2 = st.columns(2)

with col1:

    st.caption(
        "RELIANCE STOCK INTELLIGENCE PLATFORM"
    )

with col2:

    st.caption(
        "Python | Pandas | NumPy | Scikit-learn | Plotly | Streamlit"
    )