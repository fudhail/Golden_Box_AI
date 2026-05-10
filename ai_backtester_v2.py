"""
Golden AI v2 - Ensemble Backtester Engine
Handles Date Ranges, Fixed Lot Sizes, and Stacking ML Predictions.
"""

import json
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import pytz

# Try importing ML libs
import xgboost as xgb
try: import lightgbm as lgb
except ImportError: pass
try: from catboost import CatBoostClassifier
except ImportError: pass

def load_ensemble_models():
    """Loads the base models and the meta-learner config."""
    with open("sweep_threshold.json", "r") as f:
        meta_info = json.load(f)
    
    models = {}
    if "xgb" in meta_info["base_models"]:
        m = xgb.XGBClassifier()
        m.load_model("sweep_model.json")
        models["xgb"] = m
        
    if meta_info.get("has_lgb", False):
        m = lgb.Booster(model_file="sweep_lgb.txt")
        models["lgb"] = m
        
    if meta_info.get("has_cat", False):
        m = CatBoostClassifier()
        m.load_model("sweep_cat.cbm")
        models["cat"] = m
        
    return models, meta_info

def predict_ensemble(X, models, meta_info):
    """Generates base predictions and passes them through the meta-learner."""
    preds = []
    
    if "xgb" in models:
        preds.append(models["xgb"].predict_proba(X)[:, 1])
    if "lgb" in models:
        preds.append(models["lgb"].predict(X)) # Booster.predict returns raw proba
    if "cat" in models:
        preds.append(models["cat"].predict_proba(X)[:, 1])
        
    base_preds = np.column_stack(preds)
    
    # 1. Scale base predictions
    mean = np.array(meta_info["meta_scaler_mean"])
    scale = np.array(meta_info["meta_scaler_std"])
    scaled_preds = (base_preds - mean) / scale
    
    # 2. Logistic Regression Meta-Learner (Sigmoid: 1 / (1 + exp(-(Xw + b))))
    w = np.array(meta_info["meta_weights"])
    b = meta_info["meta_intercept"]
    z = np.dot(scaled_preds, w) + b
    final_proba = 1 / (1 + np.exp(-z))
    
    return final_proba

def pull_data_by_date(symbol, timeframe, date_from, date_to):
    """Pulls MT5 data using a specific date range."""
    if not mt5.initialize():
        raise Exception("MT5 Init Failed. Make sure MT5 is open.")
    
    tz = pytz.timezone("Etc/UTC")
    d_from = datetime(date_from.year, date_from.month, date_from.day, tzinfo=tz)
    d_to = datetime(date_to.year, date_to.month, date_to.day, 23, 59, 59, tzinfo=tz)
    
    rates = mt5.copy_rates_range(symbol, timeframe, d_from, d_to)
    if rates is None or len(rates) == 0:
        raise Exception("No data found for this date range.")
        
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    
    # === THE FIX: Explicitly assign volume and drop the duplicate ===
    if "real_volume" in df.columns and df["real_volume"].sum() > 0:
        df["volume"] = df["real_volume"]
    else:
        df["volume"] = df["tick_volume"]
        
    # Strip out the extra MT5 columns to prevent matrix collisions
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    
    return df

def engineer_features(df, rolling_bars=10, atr_period=14):
    """Applies your exact feature engineering logic, imported from training for 1:1 match."""
    from train_sweep_v2 import engineer_all_features, build_fakeout_features
    
    # 1. Base features (respects time properly, uses shift(1) for box/ma/atr)
    df = engineer_all_features(df)
    
    # 2. Detect fakeouts (do not drop non-fakeouts so we can simulate over time)
    df["upside_fakeout"]   = (
        (df["high"]  > df["box_high"]) &
        (df["close"] < df["box_high"]) &
        (df["open"]  < df["box_high"])
    ).astype(int)
    df["downside_fakeout"] = (
        (df["low"]   < df["box_low"]) &
        (df["close"] > df["box_low"]) &
        (df["open"]  > df["box_low"])
    ).astype(int)
    df["is_fakeout"] = df["upside_fakeout"] | df["downside_fakeout"]
    
    # 3. Add fakeout specific properties (wicks, body ratios, etc)
    df = build_fakeout_features(df)
    
    # Do not dropna here because we need continuous time for the dashboard loop.
    # We will just fillna(0) for the first few rows.
    df.fillna(0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df