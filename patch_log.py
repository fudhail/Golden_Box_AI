import re

with open("live_trader.py", "r", encoding="utf-8") as f:
    code = f.read()

# Update execute_trade
code = re.sub(
    r'    if result\.retcode != mt5\.TRADE_RETCODE_DONE:\s+print\(f"❌ Order Failed: \{result\.comment\}"\)\s+else:\s+print\(f"✅ Trade Executed! Ticket: \{result\.order\}"\)',
    '''    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[Golden 10H AI] ❌ MT5 ORDER FAILED | Error: {result.retcode} | {result.comment}")
    else:
        direction = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"
        print(f"[Golden 10H AI] 💰 {direction} EXECUTED | Ticket: {result.order}")''',
    code
)

# Update the while loop logic
old_block = """        if last_candle_time is None or current_time != last_candle_time:
            last_candle_time = current_time
            print(f"\\n[{datetime.now().strftime('%H:%M:%S')}] New H1 Candle. Scanning market...")

            # Check if we already have an open trade (don't open a new one if we do)
            if positions and len(positions) > 0:
                print("⏳ Trade already active. Waiting...")
                time.sleep(60) # Sleep for a minute and check again
                continue"""

new_block = """        if last_candle_time is None or current_time != last_candle_time:
            last_candle_time = current_time
            print("-" * 57)
            print(f"[Golden 10H AI] [NEW HOUR] Scanning market at {datetime.now().strftime('%H:%M:%S')}...")

            # Check if we already have an open trade (don't open a new one if we do)
            if positions and len(positions) > 0:
                print("[Golden 10H AI] Status: Trade active — skipping fakeout scan.")
                time.sleep(60) # Sleep for a minute and check again
                continue"""

code = code.replace(old_block, new_block)

old_block2 = """            # 5. Check the newly closed candle (index -1 in our engineered df)
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
                print("💤 No Fakeout. Waiting for next hour...")"""

new_block2 = """            # 5. Check the newly closed candle (index -1 in our engineered df)
            last_closed = df.iloc[-1]
            
            print(f"[Golden 10H AI] [CANDLE 1 DATA] O:{last_closed['open']:.3f} | H:{last_closed['high']:.3f} | L:{last_closed['low']:.3f} | C:{last_closed['close']:.3f}")
            print(f"[Golden 10H AI] [BOX DATA] High:{last_closed['box_high']:.3f} | Low:{last_closed['box_low']:.3f}")

            if last_closed["is_fakeout"] == 1:
                if last_closed["upside_fakeout"] == 1:
                    print("[Golden 10H AI] ⚡ UPSIDE FAKEOUT DETECTED! Wick pierced box high.")
                else:
                    print("[Golden 10H AI] ⚡ DOWNSIDE FAKEOUT DETECTED! Wick pierced box low.")
                
                features = meta_info["features"]
                X = df.loc[[df.index[-1]], features] # Extract features for this specific candle
                
                for col in features:
                    if col not in X.columns: X[col] = 0.0

                prob = predict_ensemble(X, models, meta_info)[0]
                armed = prob >= meta_info['threshold']
                
                print(f"[Golden 10H AI] 🧠 AI Confidence: {prob:.4f} (Threshold: {meta_info['threshold']:.4f}) | Armed: {'YES ✅' if armed else 'NO ❌'}")

                if armed:
                    ask = mt5.symbol_info_tick(SYMBOL).ask
                    bid = mt5.symbol_info_tick(SYMBOL).bid
                    atr = last_closed["atr_14"]
                    c1_h = df.iloc[-2]["high"]
                    c1_l = df.iloc[-2]["low"]

                    if last_closed["upside_fakeout"] == 1:
                        sl = c1_h + (atr * SL_MULTIPLIER)
                        tp = bid - (abs(bid - sl) * RR_RATIO)
                        print(f"[Golden 10H AI] [TRADE DESK] Targeting SELL | Entry:{bid:.3f} | SL:{sl:.3f} | TP:{tp:.3f}")
                        execute_trade(SYMBOL, LOT_SIZE, mt5.ORDER_TYPE_SELL, bid, sl, tp)

                    elif last_closed["downside_fakeout"] == 1:
                        sl = c1_l - (atr * SL_MULTIPLIER)
                        tp = ask + (abs(ask - sl) * RR_RATIO)
                        print(f"[Golden 10H AI] [TRADE DESK] Targeting BUY | Entry:{ask:.3f} | SL:{sl:.3f} | TP:{tp:.3f}")
                        execute_trade(SYMBOL, LOT_SIZE, mt5.ORDER_TYPE_BUY, ask, sl, tp)
                else:
                    if last_closed["upside_fakeout"] == 1:
                        print("[Golden 10H AI] 🛡️ AI REJECTED UPSIDE SWEEP. Setup classified as low-edge trap.")
                    else:
                        print("[Golden 10H AI] 🛡️ AI REJECTED DOWNSIDE SWEEP. Setup classified as low-edge trap.")
            else:
                print("[Golden 10H AI] 💤 No Fakeout. Waiting for next hour...")"""

code = code.replace(old_block2, new_block2)

with open("live_trader.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Updated successfully!")
