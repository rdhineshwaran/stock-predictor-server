from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import csv

app = FastAPI()

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
    hist = ticker.history(period="6mo")
    if hist.empty or len(hist) < 30:
        raise HTTPException(status_code=404, detail="Not enough data for prediction.")

    hist = hist.reset_index()
    hist["MA7"] = hist["Close"].rolling(window=7).mean()
    hist["Lag1"] = hist["Close"].shift(1)
    hist["Lag2"] = hist["Close"].shift(2)
    hist["Volume"] = hist["Volume"]
    hist.dropna(inplace=True)

    features = ["Lag1", "Lag2", "MA7", "Volume"]
    X = hist[features]
    y = hist["Close"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Forecast next 7 trading days
    last_known = hist.iloc[-1]
    future_data = []

    days_added = 0
    count = 0
    start_date = datetime.today()
    lag1 = last_known["Close"]
    lag2 = last_known["Lag1"]
    ma7 = last_known["MA7"]
    vol = last_known["Volume"]

    while count < 7:
        next_date = start_date + timedelta(days=days_added + 1)
        if next_date.weekday() < 5:
            X_pred = pd.DataFrame([{
                "Lag1": lag1,
                "Lag2": lag2,
                "MA7": ma7,
                "Volume": vol
            }])
            pred = model.predict(X_pred)[0]
            future_data.append({
                "price": float(pred),
                "date": next_date.strftime("%Y-%m-%d")
            })
            # update lags
            lag2 = lag1
            lag1 = pred
            count += 1
        days_added += 1

    return {
        "symbol": symbol,
        "prediction": future_data
    }