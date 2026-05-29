import time
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# Import your exact backtest logic
from ai_backtester_v2 import engineer_features, load_ensemble_models, predict_ensemble

# --- CONFIGURATION ---
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
LOT_SIZE = 0.1
MAGIC_NUMBER = 101010
SL_MULTIPLIER = 0.60
RR_RATIO = 3.0

def execute_trade(symbol, lot, order_type, price, sl, tp):
    """Sends the actual trade command to MT5. Returns True on success."""
    direction = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
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
    
    print(f"[Golden 10H AI] [ORDER SEND] {direction} | Price:{price:.3f} | SL:{sl:.3f} | TP:{tp:.3f}")
    
    try:
        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            print(f"[Golden 10H AI] ❌ MT5 RETURNED None! Last error: {err}")
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[Golden 10H AI] ❌ MT5 ORDER FAILED | RetCode: {result.retcode} | Comment: {result.comment}")
            return False
        else:
            print(f"[Golden 10H AI] 💰 {direction} EXECUTED | Ticket: {result.order} | Vol: {result.volume}")
            return True
    except Exception as e:
        print(f"[Golden 10H AI] 💥 EXCEPTION in execute_trade: {e}")
        return False

def run_live_bot():
    if not mt5.initialize():
        print("MT5 Init Failed. Ensure MT5 is open.")
        return

    print("🚀 Golden AI Live Engine Started...")
    models, meta_info = load_ensemble_models()
    last_candle_time = None

    while True:
        # --- PARTIAL PROFIT & BREAK-EVEN MONITORING (DISABLED) ---
        positions = mt5.positions_get(symbol=SYMBOL)
        if False: # positions and len(positions) > 0:
            for pos in positions:
                if pos.magic == MAGIC_NUMBER and pos.volume == LOT_SIZE:
                    entry = pos.price_open
                    tp = pos.tp
                    
                    if pos.type == mt5.ORDER_TYPE_BUY:
                        r_dist = (tp - entry) / RR_RATIO
                        partial_tp = entry + (r_dist * (RR_RATIO / 2.0))
                        tick = mt5.symbol_info_tick(SYMBOL)
                        if tick.bid >= partial_tp:
                            print(f"🎯 Target 1 Reached! Partial Out & Break Even for {SYMBOL} (BUY)")
                            close_req = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "position": pos.ticket,
                                "symbol": pos.symbol,
                                "volume": pos.volume / 2.0,
                                "type": mt5.ORDER_TYPE_SELL,
                                "price": tick.bid,
                                "magic": MAGIC_NUMBER,
                                "comment": "Partial Close",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": mt5.ORDER_FILLING_IOC,
                            }
                            res = mt5.order_send(close_req)
                            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                                sl_req = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "position": pos.ticket,
                                    "sl": entry,
                                    "tp": tp
                                }
                                mt5.order_send(sl_req)

                    elif pos.type == mt5.ORDER_TYPE_SELL:
                        r_dist = (entry - tp) / RR_RATIO
                        partial_tp = entry - (r_dist * (RR_RATIO / 2.0))
                        tick = mt5.symbol_info_tick(SYMBOL)
                        if tick.ask <= partial_tp:
                            print(f"🎯 Target 1 Reached! Partial Out & Break Even for {SYMBOL} (SELL)")
                            close_req = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "position": pos.ticket,
                                "symbol": pos.symbol,
                                "volume": pos.volume / 2.0,
                                "type": mt5.ORDER_TYPE_BUY,
                                "price": tick.ask,
                                "magic": MAGIC_NUMBER,
                                "comment": "Partial Close",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": mt5.ORDER_FILLING_IOC,
                            }
                            res = mt5.order_send(close_req)
                            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                                sl_req = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "position": pos.ticket,
                                    "sl": entry,
                                    "tp": tp
                                }
                                mt5.order_send(sl_req)
        # ----------------------------------------------

        # 1. Get the current active H1 candle time
        current_candle = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)[0]
        current_time = current_candle['time']

        # 2. Only run the AI if a NEW hour just started
        if last_candle_time is None or current_time != last_candle_time:
            last_candle_time = current_time
            print("-" * 57)
            print(f"[Golden 10H AI] [NEW HOUR] Scanning market at {datetime.now().strftime('%H:%M:%S')}...")

            # Check if we already have an open trade (don't open a new one if we do)
            if positions and len(positions) > 0:
                print("[Golden 10H AI] Status: Trade active — skipping fakeout scan.")
                time.sleep(60) # Sleep for a minute and check again
                continue

            # 3. Pull the last 250 bars to build the indicators
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 250)
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            
            if "real_volume" in df.columns and df["real_volume"].sum() > 0:
                df["volume"] = df["real_volume"]
            else:
                df["volume"] = df["tick_volume"]
            
            # 4. Engineer Features (Using the exact backtest logic)
            df = engineer_features(df, rolling_bars=10, atr_period=14)

            # 5. Check the newly closed candle (index -2 in our engineered df, as index -1 is the newly opened candle)
            last_closed = df.iloc[-2]
            
            print(f"[Golden 10H AI] [CANDLE 1 DATA] O:{last_closed['open']:.3f} | H:{last_closed['high']:.3f} | L:{last_closed['low']:.3f} | C:{last_closed['close']:.3f}")
            print(f"[Golden 10H AI] [BOX DATA] High:{last_closed['box_high']:.3f} | Low:{last_closed['box_low']:.3f}")

            if last_closed["is_fakeout"] == 1:
                if last_closed["upside_fakeout"] == 1:
                    print("[Golden 10H AI] ⚡ UPSIDE FAKEOUT DETECTED! Wick pierced box high.")
                else:
                    print("[Golden 10H AI] ⚡ DOWNSIDE FAKEOUT DETECTED! Wick pierced box low.")
                
                features = meta_info["features"]
                X = df.loc[[df.index[-2]], features] # Extract features for this specific closed candle
                
                for col in features:
                    if col not in X.columns: X[col] = 0.0

                prob = predict_ensemble(X, models, meta_info)[0]
                armed = prob >= meta_info['threshold']
                
                print(f"[Golden 10H AI] 🧠 AI Confidence: {prob:.4f} (Threshold: {meta_info['threshold']:.4f}) | Armed: {'YES ✅' if armed else 'NO ❌'}")

                if armed:
                    tick = mt5.symbol_info_tick(SYMBOL)
                    ask = tick.ask
                    bid = tick.bid
                    atr = last_closed["atr_14"]
                    c1_h = last_closed["high"]
                    c1_l = last_closed["low"]

                    if last_closed["upside_fakeout"] == 1:
                        # SELL: price falls. SL is ABOVE entry (bid). TP is BELOW entry.
                        sl = round(c1_h + (atr * SL_MULTIPLIER), 3)
                        risk = abs(bid - sl)  # risk = sl - bid (since sl > bid for a sell)
                        tp = round(bid - (risk * RR_RATIO), 3)
                        print(f"[Golden 10H AI] [TRADE DESK] Targeting SELL | Entry:{bid:.3f} | SL:{sl:.3f} | TP:{tp:.3f} | Risk:{risk:.3f}")
                        execute_trade(SYMBOL, LOT_SIZE, mt5.ORDER_TYPE_SELL, bid, sl, tp)

                    elif last_closed["downside_fakeout"] == 1:
                        # BUY: price rises. SL is BELOW entry (ask). TP is ABOVE entry.
                        sl = round(c1_l - (atr * SL_MULTIPLIER), 3)
                        risk = abs(ask - sl)  # risk = ask - sl (since sl < ask for a buy)
                        tp = round(ask + (risk * RR_RATIO), 3)
                        print(f"[Golden 10H AI] [TRADE DESK] Targeting BUY | Entry:{ask:.3f} | SL:{sl:.3f} | TP:{tp:.3f} | Risk:{risk:.3f}")
                        execute_trade(SYMBOL, LOT_SIZE, mt5.ORDER_TYPE_BUY, ask, sl, tp)
                else:
                    if last_closed["upside_fakeout"] == 1:
                        print("[Golden 10H AI] 🛡️ AI REJECTED UPSIDE SWEEP. Setup classified as low-edge trap.")
                    else:
                        print("[Golden 10H AI] 🛡️ AI REJECTED DOWNSIDE SWEEP. Setup classified as low-edge trap.")
            else:
                print("[Golden 10H AI] 💤 No Fakeout. Waiting for next hour...")

        # Sleep for 5 seconds to prevent maxing out your CPU, then check the time again
        time.sleep(5)

if __name__ == "__main__":
    run_live_bot()