from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
import xgboost as xgb
import pandas as pd

app = FastAPI(title="Golden Box AI Filter")
model = xgb.XGBClassifier()

@app.on_event("startup")
def load_model():
    model.load_model("sweep_model.json")
    print("✅ XGBoost Model loaded into memory")

class FakeoutPayload(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    avg_vol_10: float
    box_high: float
    box_low: float
    hour: int
    is_upside: int

@app.post("/predict_sweep")
def predict_sweep(data: FakeoutPayload):
    try:
        total_size = data.high - data.low
        if total_size == 0: total_size = 0.0001
            
        box_width = data.box_high - data.box_low
        vol_spike_ratio = data.volume / data.avg_vol_10 if data.avg_vol_10 > 0 else 1.0
        
        upper_wick = data.high - max(data.open, data.close)
        lower_wick = min(data.open, data.close) - data.low
        
        rejection_intensity = (upper_wick / total_size) if data.is_upside == 1 else (lower_wick / total_size)

        df_features = pd.DataFrame([{
            'rejection_intensity': rejection_intensity,
            'vol_spike_ratio': vol_spike_ratio,
            'box_width': box_width,
            'hour': data.hour,
            'is_upside': data.is_upside
        }])

        probability = model.predict_proba(df_features)[0][1]
        
        # 0.60 means the AI needs to be at least 60% confident to take the trade. Adjust as needed.
        CONFIDENCE_THRESHOLD = 0.60 
        is_armed = bool(probability >= CONFIDENCE_THRESHOLD)

        return {
            "armed": is_armed,
            "confidence": round(float(probability), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))