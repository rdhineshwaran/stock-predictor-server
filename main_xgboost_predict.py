from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
import csv
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

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
        return {
            "symbol": symbol,
            "history": [
                {
                    "x": row["Date"].strftime("%Y-%m-%d"),
                    "y": row["Close"]
                } for _, row in hist.iterrows()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stocks/{symbol}/predict")
def predict_price(symbol: str):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="6mo")
    if hist.empty or len(hist) < 60:
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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    last_row = hist.iloc[-1]
    lag1 = last_row["Close"]
    lag2 = last_row["Lag1"]
    ma7 = last_row["MA7"]
    vol = last_row["Volume"]

    future_data = []
    start_date = datetime.today()
    days_added = 0
    count = 0

    while count < 7:
        date = start_date + timedelta(days=days_added + 1)
        if date.weekday() < 5:
            input_data = pd.DataFrame([{
                "Lag1": lag1,
                "Lag2": lag2,
                "MA7": ma7,
                "Volume": vol
            }])
            input_scaled = scaler.transform(input_data)
            pred = model.predict(input_scaled)[0]
            future_data.append({
                "x": date.strftime("%Y-%m-%d"),
                "y": float(pred)
            })
            lag2 = lag1
            lag1 = pred
            count += 1
        days_added += 1

    return {
        "symbol": symbol,
        "prediction": future_data
    }

@app.get("/index-history")
def get_index_history(index: str, period: str = "1mo", interval: str = "1d"):
    alias_map = {
        "nifty50": "^NSEI",
        "nifty_bank": "^NSEBANK",
        "sensex": "^BSESN"
    }

    symbol = alias_map.get(index.lower())
    if not symbol:
        raise HTTPException(status_code=400, detail="Invalid index alias")

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        if hist.empty:
            raise HTTPException(status_code=404, detail="No historical data found.")
        hist.reset_index(inplace=True)
        return {
            "index": index.lower(),
            "symbol": symbol,
            "history": [
                {
                    "x": row["Date"].strftime("%Y-%m-%d"),
                    "y": row["Close"]
                } for _, row in hist.iterrows()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))