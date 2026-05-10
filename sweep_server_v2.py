"""
Golden AI Server — Fixed & Hardened
=====================================
Fixes:
  1. Added GET /predict_sweep to catch MT5 hitting wrong method (shows clear error)
  2. Fixed candle targeting — uses the payload's OHLCV to find the exact candle
     instead of blindly using iloc[-2] which may not be the fakeout candle
  3. Added /debug endpoint so you can test from browser
  4. Better error messages that print to console for easy diagnosis
"""

import json
import traceback

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import MetaTrader5 as mt5

try:
    from train_sweep_v2 import engineer_all_features, build_fakeout_features
except ModuleNotFoundError:
    engineer_all_features = None
    build_fakeout_features = None

# ── Load model & threshold ────────────────────────────────────────────────────
with open("sweep_threshold.json", "r") as f:
    meta_info = json.load(f)

FEATURES  = meta_info["features"]
THRESHOLD = meta_info["threshold"]

model = xgb.XGBClassifier()
model.load_model("sweep_model.json")

app = FastAPI(title="Golden AI - V2")


def _engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    if engineer_all_features is not None:
        return engineer_all_features(df)

    h, l, c, o, v = df["high"], df["low"], df["close"], df["open"], df["volume"]

    df["box_high"] = h.rolling(10).max().shift(1)
    df["box_low"] = l.rolling(10).min().shift(1)
    df["avg_vol_10"] = v.rolling(10).mean().shift(1)
    df["box_width"] = df["box_high"] - df["box_low"]

    prev_close = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_close).abs(),
        (l - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean().shift(1)
    df["atr_5"] = tr.rolling(5).mean().shift(1)

    df["sma_20"] = c.rolling(20).mean().shift(1)
    df["sma_50"] = c.rolling(50).mean().shift(1)
    df["sma_200"] = c.rolling(200).mean().shift(1)
    df["trend_dist_20"] = (c - df["sma_20"]) / df["sma_20"]
    df["trend_dist_50"] = (c - df["sma_50"]) / df["sma_50"]
    df["trend_alignment"] = np.sign(df["sma_20"] - df["sma_50"]) * np.sign(df["sma_50"] - df["sma_200"])

    df["roc_5"] = c.pct_change(5).shift(1)

    delta = c.diff().shift(1)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    bb_mid = c.rolling(20).mean().shift(1)
    bb_std = c.rolling(20).std().shift(1)
    df["bb_width"] = (2 * bb_std) / bb_mid
    df["bb_position"] = (c - bb_mid) / (2 * bb_std)

    df["vol_spike_ratio"] = v / df["avg_vol_10"].replace(0, 1e-9)
    df["vol_trend"] = v.rolling(5).mean().shift(1) / v.rolling(20).mean().shift(1)

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["london_session"] = df["hour"].between(8, 16).astype(int)
    df["ny_session"] = df["hour"].between(13, 21).astype(int)
    df["overlap_session"] = df["hour"].between(13, 16).astype(int)
    df["bull_candle"] = (c > o).astype(int)
    df["consec_bull"] = df["bull_candle"].rolling(3).sum().shift(1)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _build_fakeout_features(df: pd.DataFrame) -> pd.DataFrame:
    if build_fakeout_features is not None:
        return build_fakeout_features(df)

    total_size = (df["high"] - df["low"]).replace(0, 1e-6)
    df["total_size"] = total_size
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["is_upside"] = df["upside_fakeout"]
    df["rejection_intensity"] = np.where(
        df["is_upside"] == 1,
        df["upper_wick"] / total_size,
        df["lower_wick"] / total_size,
    )
    df["penetration_depth"] = np.where(
        df["is_upside"] == 1,
        (df["high"] - df["box_high"]) / df["atr_14"].replace(0, 1e-6),
        (df["box_low"] - df["low"]) / df["atr_14"].replace(0, 1e-6),
    )
    df["box_position"] = (df["close"] - df["box_low"]) / df["box_width"].replace(0, 1e-6)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / total_size
    df["close_pullback"] = np.where(
        df["is_upside"] == 1,
        (df["box_high"] - df["close"]) / df["atr_14"].replace(0, 1e-6),
        (df["close"] - df["box_low"]) / df["atr_14"].replace(0, 1e-6),
    )
    df["vol_expansion"] = total_size / df["atr_5"].replace(0, 1e-6)
    df["rsi_extreme"] = np.where(
        df["is_upside"] == 1,
        df["rsi_14"] / 100,
        1 - df["rsi_14"] / 100,
    )
    df["counter_trend"] = np.where(
        df["is_upside"] == 1,
        (df["trend_alignment"] > 0).astype(int),
        (df["trend_alignment"] < 0).astype(int),
    )
    return df


def _predict_fakeout_from_payload(data: "FakeoutPayload") -> pd.DataFrame:
    total_size = data.high - data.low
    if total_size == 0:
        total_size = 1e-6

    box_width = data.box_high - data.box_low
    vol_spike_ratio = data.volume / data.avg_vol_10 if data.avg_vol_10 > 0 else 1.0
    upper_wick = data.high - max(data.open, data.close)
    lower_wick = min(data.open, data.close) - data.low

    return pd.DataFrame([{
        "rejection_intensity": (upper_wick / total_size) if data.is_upside == 1 else (lower_wick / total_size),
        "penetration_depth": ((data.high - data.box_high) if data.is_upside == 1 else (data.box_low - data.low)) / max(total_size, 1e-6),
        "body_ratio": abs(data.close - data.open) / total_size,
        "close_pullback": (data.box_high - data.close) if data.is_upside == 1 else (data.close - data.box_low),
        "vol_expansion": total_size,
        "vol_spike_ratio": vol_spike_ratio,
        "vol_trend": 1.0,
        "box_width": box_width,
        "box_position": (data.close - data.box_low) / max(box_width, 1e-6),
        "trend_dist_20": 0.0,
        "trend_dist_50": 0.0,
        "trend_alignment": 0.0,
        "roc_5": 0.0,
        "rsi_extreme": 0.5,
        "counter_trend": 0,
        "atr_14": total_size,
        "bb_width": 0.0,
        "bb_position": 0.0,
        "hour": data.hour,
        "day_of_week": 0,
        "london_session": int(8 <= data.hour <= 16),
        "ny_session": int(13 <= data.hour <= 21),
        "overlap_session": int(13 <= data.hour <= 16),
        "is_upside": data.is_upside,
    }])


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    if not mt5.initialize():
        print("❌ MT5 Init failed — keep MT5 open before starting server")
    else:
        info = mt5.terminal_info()
        print(f"✅ MT5 connected (build {info.build})")
        print(f"✅ Server LIVE | Threshold: {THRESHOLD:.4f} | Features: {len(FEATURES)}")
        print(f"✅ Endpoints:")
        print(f"     POST http://127.0.0.1:8000/predict_sweep  ← MT5 EA calls this")
        print(f"     GET  http://127.0.0.1:8000/health         ← sanity check")
        print(f"     GET  http://127.0.0.1:8000/debug          ← browser test")


# ── Payload schema ────────────────────────────────────────────────────────────
class FakeoutPayload(BaseModel):
    open:       float
    high:       float
    low:        float
    close:      float
    volume:     float
    avg_vol_10: float
    box_high:   float
    box_low:    float
    hour:       int
    is_upside:  int


# ── Main prediction endpoint ──────────────────────────────────────────────────
@app.post("/predict_sweep")
def predict_sweep(data: FakeoutPayload):
    print(f"\n[PREDICT] Received → is_upside={data.is_upside} | "
          f"H={data.high:.2f} L={data.low:.2f} C={data.close:.2f} | "
          f"box_high={data.box_high:.2f} box_low={data.box_low:.2f}")

    try:
        # ── 1. Fetch recent history ──────────────────────────────────────────
        # 250 bars covers SMA-200 + ATR-14 + buffer
        rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H1, 0, 250)
        if rates is None:
            print(f"[WARN] MT5 data fetch failed: {mt5.last_error()}. Using payload-only features.")
            row = _predict_fakeout_from_payload(data)[FEATURES]
        else:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")

            if "real_volume" in df.columns and df["real_volume"].sum() > 0:
                df.rename(columns={"real_volume": "volume"}, inplace=True)
            else:
                df.rename(columns={"tick_volume": "volume"}, inplace=True)

            df = df[["time", "open", "high", "low", "close", "volume"]].copy()
            df = _engineer_all_features(df)

        if rates is not None:
            # ── 4. Find the fakeout candle by matching OHLCV from payload ────────
            tolerance = 0.01
            mask = (df["close"] - data.close).abs() < tolerance

            if mask.sum() == 0:
                print("[WARN] Could not match candle by close price. Using the most recent closed candle.")
                target_idx = df.index[-2]
            elif mask.sum() > 1:
                candidates = df[mask]
                target_idx = (candidates["high"] - data.high).abs().idxmin()
                print(f"[WARN] Multiple close-price matches ({mask.sum()}). Picked idx={target_idx} by high proximity.")
            else:
                target_idx = df[mask].index[0]
                print(f"[OK] Matched fakeout candle at index {target_idx} / time={df.loc[target_idx, 'time']}")

            # ── 5. Inject fakeout context from the EA payload ────────────────────
            df.loc[target_idx, "upside_fakeout"] = 1 if data.is_upside == 1 else 0
            df.loc[target_idx, "downside_fakeout"] = 1 if data.is_upside == 0 else 0
            df.loc[target_idx, "box_high"] = data.box_high
            df.loc[target_idx, "box_low"] = data.box_low
            df.loc[target_idx, "box_width"] = data.box_high - data.box_low
            df.loc[target_idx, "avg_vol_10"] = data.avg_vol_10

            # ── 6. Build fakeout-specific features ───────────────────────────────
            df = _build_fakeout_features(df)

            # ── 7. Extract feature vector for this specific candle ────────────────
            row = df.loc[[target_idx], FEATURES]
        else:
            row = _predict_fakeout_from_payload(data)[FEATURES]

        # Sanity check — catch NaN features before predicting
        nan_cols = row.columns[row.isnull().any()].tolist()
        if nan_cols:
            print(f"[WARN] NaN features: {nan_cols} — filling with 0")
            row = row.fillna(0)

        print(f"[FEATURES] {row.iloc[0].to_dict()}")

        # ── 8. Predict ───────────────────────────────────────────────────────
        prob  = float(model.predict_proba(row)[0][1])
        armed = prob >= THRESHOLD

        print(f"[RESULT] prob={prob:.4f} | threshold={THRESHOLD:.4f} | armed={armed}")

        return {
            "armed":      armed,
            "confidence": round(prob, 4),
            "threshold":  round(THRESHOLD, 4),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[EXCEPTION] {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_sweep_payload")
def predict_sweep_payload(data: FakeoutPayload):
    """Payload-only prediction path for historical replay/backtests.

    This bypasses MT5 history reads so offline replays can call the same model
    with candle data from a CSV or other historical source.
    """
    try:
        row = _predict_fakeout_from_payload(data)[FEATURES]

        nan_cols = row.columns[row.isnull().any()].tolist()
        if nan_cols:
            print(f"[WARN] NaN features: {nan_cols} — filling with 0")
            row = row.fillna(0)

        prob = float(model.predict_proba(row)[0][1])
        armed = prob >= THRESHOLD

        return {
            "armed": armed,
            "confidence": round(prob, 4),
            "threshold": round(THRESHOLD, 4),
            "mode": "payload_only",
        }
    except Exception as e:
        print(f"[EXCEPTION] {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Health check (GET — browser friendly) ────────────────────────────────────
@app.get("/health")
def health():
    mt5_ok = mt5.terminal_info() is not None
    return {
        "status":    "ok",
        "mt5":       mt5_ok,
        "threshold": THRESHOLD,
        "features":  len(FEATURES),
    }


# ── Debug endpoint — simulates a fake fakeout so you can test from browser ───
@app.get("/debug")
def debug():
    """
    Hit http://127.0.0.1:8000/debug in your browser to confirm the
    prediction pipeline works end-to-end without needing the MT5 EA.
    """
    fake = FakeoutPayload(
        open=3300.0, high=3320.0, low=3295.0, close=3302.0,
        volume=1200, avg_vol_10=900, box_high=3315.0, box_low=3280.0,
        hour=14, is_upside=1
    )
    row = _predict_fakeout_from_payload(fake)[FEATURES]
    prob = float(model.predict_proba(row)[0][1])
    return {
        "debug_input": fake.dict(),
        "prediction": {
            "armed": prob >= THRESHOLD,
            "confidence": round(prob, 4),
            "threshold": round(THRESHOLD, 4),
        },
    }


# ── Catch-all for wrong methods (helps diagnose MT5 hitting wrong route) ──────
@app.get("/predict_sweep")
def predict_sweep_wrong_method():
    return JSONResponse(
        status_code=405,
        content={
            "error": "Method Not Allowed — use POST not GET",
            "fix":   "Your MT5 EA WebRequest must use POST method",
            "example_payload": {
                "open": 3300.0, "high": 3320.0, "low": 3295.0, "close": 3302.0,
                "volume": 1200, "avg_vol_10": 900,
                "box_high": 3315.0, "box_low": 3280.0,
                "hour": 14, "is_upside": 1
            }
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)