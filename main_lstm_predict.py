from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
import csv
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/stocks")
def get_stocks():
    try:
        with open("nse_stocks.csv", newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            return list(reader)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load stock list: {str(e)}")

@app.get("/stocks/{symbol}/current")
def get_current_price(symbol: str):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period="1d")
    if todays_data.empty:
        raise HTTPException(status_code=404, detail="No current data available.")
    return {"symbol": symbol, "price": todays_data["Close"].iloc[-1]}

@app.get("/stocks/{symbol}/history")
def get_stock_history(symbol: str, period: str = "1mo", interval: str = "1d"):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        if hist.empty:
            raise HTTPException(status_code=404, detail="No historical data found.")
        hist.reset_index(inplace=True)
        return [
            {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "close": row["Close"]
            } for _, row in hist.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stocks/{symbol}/predict")
def predict_price(symbol: str):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="6mo")
    if df.empty or len(df) < 60:
        raise HTTPException(status_code=404, detail="Not enough data for prediction.")

    df = df[["Close"]]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    seq_len = 30
    X = []
    y = []

    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

    # Predict next 7 trading days
    last_seq = scaled_data[-seq_len:].reshape(1, seq_len, 1)
    predictions = []

    count = 0
    start_date = datetime.today()
    days_added = 0

    while count < 7:
        pred = model.predict(last_seq, verbose=0)[0][0]
        predictions.append(pred)
        # roll the sequence
        last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)
        count += 1

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    future_dates = []
    count = 0
    days_added = 0

    while len(future_dates) < 7:
        date = start_date + timedelta(days=days_added + 1)
        if date.weekday() < 5:
            future_dates.append(date.strftime("%Y-%m-%d"))
            count += 1
        days_added += 1

    return {
        "symbol": symbol,
        "prediction": [
            {"date": future_dates[i], "price": float(predictions[i])}
            for i in range(7)
        ]
    }