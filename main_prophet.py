from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/stocks/{symbol}/history")
def get_stock_history(symbol: str, period: str = "6mo"):
    try:
        df = yf.download(symbol, period=period)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for symbol")
        df.reset_index(inplace=True)

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=500, detail=f"Missing column in data: {col}")

        df = df.astype({col: float for col in required_cols})
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

        return {
            "symbol": symbol.upper(),
            "history": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stocks/{symbol}/predict")
def predict_stock(symbol: str):
    try:
        df = yf.download(symbol, period="6mo")
        if df.empty or len(df) < 30:
            raise HTTPException(status_code=400, detail="Not enough data for prediction")

        df.reset_index(inplace=True)
        prophet_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)

        prediction = forecast[["ds", "yhat"]].tail(7)
        return {
            "symbol": symbol.upper(),
            "prediction": [
                {
                    "x": pd.to_datetime(row["ds"]).strftime("%Y-%m-%d"),
                    "y": float(round(row["yhat"], 2))
                }
                for _, row in prediction.iterrows()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
