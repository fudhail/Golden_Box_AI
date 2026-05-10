import time
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# Import your exact backtest logic
from ai_backtester_v2 import engineer_features, load_ensemble_models, predict_ensemble

# --- CONFIGURATION ---
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
LOT_SIZE = 0.05
MAGIC_NUMBER = 101010
SL_MULTIPLIER = 0.3
RR_RATIO = 3.0

def execute_trade(symbol, lot, order_type, price, sl, tp):
    """Sends the actual trade command to MT5"""
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot),
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20, # Max slippage allowed
        "magic": MAGIC_NUMBER,
        "comment": "Golden AI Engine",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Order Failed: {result.comment}")
    else:
        print(f"✅ Trade Executed! Ticket: {result.order}")

def run_live_bot():
    if not mt5.initialize():
        print("MT5 Init Failed. Ensure MT5 is open.")
        return

    print("🚀 Golden AI Live Engine Started...")
    models, meta_info = load_ensemble_models()
    last_candle_time = None

    while True:
        # 1. Get the current active H1 candle time
        current_candle = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)[0]
        current_time = current_candle['time']

        # 2. Only run the AI if a NEW hour just started
        if last_candle_time is None or current_time != last_candle_time:
            last_candle_time = current_time
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] New H1 Candle. Scanning market...")

            # Check if we already have an open trade (don't open a new one if we do)
            positions = mt5.positions_get(symbol=SYMBOL)
            if positions and len(positions) > 0:
                print("⏳ Trade already active. Waiting...")
                time.sleep(60) # Sleep for a minute and check again
                continue

            # 3. Pull the last 250 bars to build the indicators
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 250)
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df["volume"] = df.get("real_volume", df["tick_volume"])
            
            # 4. Engineer Features (Using the exact backtest logic)
            df = engineer_features(df, rolling_bars=6, atr_period=14)

            # 5. Check the newly closed candle (index -1 in our engineered df)
            last_closed = df.iloc[-1]
            
            if last_closed["is_fakeout"] == 1:
                print("⚡ Fakeout Detected! Consulting AI Ensemble...")
                
                features = meta_info["features"]
                X = df.loc[[df.index[-1]], features] # Extract features for this specific candle
                
                for col in features:
                    if col not in X.columns: X[col] = 0.0

                prob = predict_ensemble(X, models, meta_info)[0]
                print(f"🧠 AI Confidence: {prob:.4f} (Threshold: {meta_info['threshold']:.4f})")

                if prob >= meta_info['threshold']:
                    print("🔥 AI ARMED. Executing Trade...")
                    
                    ask = mt5.symbol_info_tick(SYMBOL).ask
                    bid = mt5.symbol_info_tick(SYMBOL).bid
                    atr = last_closed["atr_14"]
                    c1_h = df.iloc[-2]["high"]
                    c1_l = df.iloc[-2]["low"]

                    if last_closed["upside_fakeout"] == 1:
                        sl = c1_h + (atr * SL_MULTIPLIER)
                        tp = bid - (abs(bid - sl) * RR_RATIO)
                        execute_trade(SYMBOL, LOT_SIZE, mt5.ORDER_TYPE_SELL, bid, sl, tp)

                    elif last_closed["downside_fakeout"] == 1:
                        sl = c1_l - (atr * SL_MULTIPLIER)
                        tp = ask + (abs(ask - sl) * RR_RATIO)
                        execute_trade(SYMBOL, LOT_SIZE, mt5.ORDER_TYPE_BUY, ask, sl, tp)
                else:
                    print("🛡️ AI Rejected Setup (Low Edge).")
            else:
                print("💤 No Fakeout. Waiting for next hour...")

        # Sleep for 5 seconds to prevent maxing out your CPU, then check the time again
        time.sleep(5)

if __name__ == "__main__":
    run_live_bot()