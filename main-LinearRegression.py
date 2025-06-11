from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta
import csv

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
    hist = ticker.history(period="6mo")
    if hist.empty:
        raise HTTPException(status_code=404, detail="Not enough data for prediction.")

    hist = hist.reset_index()
    hist["Days"] = np.arange(len(hist))
    X = hist[["Days"]]
    y = hist["Close"]

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.arange(len(hist), len(hist) + 7).reshape(-1, 1)
    predictions = model.predict(future_days)

    start_date = datetime.today()
    dated_prediction = []
    days_added = 0
    i = 0

    while len(dated_prediction) < len(predictions):
        future_date = start_date + timedelta(days=days_added + 1)
        if future_date.weekday() < 5:  # Weekdays only
            dated_prediction.append({
                "price": float(predictions[i]),
                "date": future_date.strftime("%Y-%m-%d")
            })
            i += 1
        days_added += 1

    return {
        "symbol": symbol,
        "prediction": dated_prediction
    }