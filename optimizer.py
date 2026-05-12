import pandas as pd
import MetaTrader5 as mt5
from datetime import date, timedelta
import time
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from ai_backtester_v2 import pull_data_by_date, engineer_features, load_ensemble_models, predict_ensemble

# --- OPTIMIZATION CONFIGURATION ---
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
INITIAL_BALANCE = 10000.0
LOT_SIZE = 0.1
CONTRACT_SIZE = 100.0

# Define parameter grid for grid search
PARAM_GRID = {
    # We only sweep trade simulation parameters.
    # Feature parameters (rolling_bars, atr_period) are fixed to what the AI was trained on.
    "take_profit_rr": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    "atr_sl_mult": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
}

def simulate_trades(df, take_profit_rr, atr_sl_mult):
    """Runs the simulation loop extremely fast on a pre-computed dataframe."""
    balance = INITIAL_BALANCE
    trades = []
    active_trade = None
    pending_trade = None
    
    # Pre-extract columns for speed
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    times = df["time"].values
    is_fakeout = df["is_fakeout"].values
    armed = df["armed"].values
    upside = df["upside_fakeout"].values
    downside = df["downside_fakeout"].values
    atrs = df["atr_14"].values
    
    for i in range(1, len(df)):
        c_open = opens[i]
        c_high = highs[i]
        c_low = lows[i]
        c_time = times[i]

        if active_trade is None and pending_trade is not None:
            entry = c_open
            if pending_trade["type"] == "BUY":
                sl_dist = abs(entry - pending_trade["sl"])
                tp = entry + (sl_dist * take_profit_rr)
                partial_tp = entry + (sl_dist * (take_profit_rr / 2.0))
            else:
                sl_dist = abs(entry - pending_trade["sl"])
                tp = entry - (sl_dist * take_profit_rr)
                partial_tp = entry - (sl_dist * (take_profit_rr / 2.0))

            active_trade = {
                "type": pending_trade["type"],
                "entry": entry,
                "sl": pending_trade["sl"],
                "tp": tp,
                "partial_tp": partial_tp,
                "lot": LOT_SIZE,
                "partial_taken": False
            }
            pending_trade = None
        
        if active_trade:
            if active_trade['type'] == 'BUY':
                # --- PARTIAL PROFIT & BREAK-EVEN DISABLED ---
                # if not active_trade['partial_taken'] and c_high >= active_trade['partial_tp']:
                #     profit_dollars = (active_trade['partial_tp'] - active_trade['entry']) * (active_trade['lot'] / 2.0) * CONTRACT_SIZE
                #     balance += profit_dollars
                #     trades.append({'type': 'BUY', 'result': 'PARTIAL_WIN', 'pnl': profit_dollars})
                #     active_trade['partial_taken'] = True
                #     active_trade['lot'] /= 2.0
                #     active_trade['sl'] = active_trade['entry'] # Break-even

                # Check SL and TP
                if c_low <= active_trade['sl']:
                    if active_trade['sl'] == active_trade['entry']:
                        pnl = 0.0
                        res = 'BREAK_EVEN'
                    else:
                        pnl = -(active_trade['entry'] - active_trade['sl']) * active_trade['lot'] * CONTRACT_SIZE
                        res = 'LOSS'
                    balance += pnl
                    trades.append({'type': 'BUY', 'result': res, 'pnl': pnl})
                    active_trade = None
                elif c_high >= active_trade['tp']:
                    profit_dollars = (active_trade['tp'] - active_trade['entry']) * active_trade['lot'] * CONTRACT_SIZE
                    balance += profit_dollars
                    trades.append({'type': 'BUY', 'result': 'WIN', 'pnl': profit_dollars})
                    active_trade = None

            elif active_trade['type'] == 'SELL':
                # --- PARTIAL PROFIT & BREAK-EVEN DISABLED ---
                # if not active_trade['partial_taken'] and c_low <= active_trade['partial_tp']:
                #     profit_dollars = (active_trade['entry'] - active_trade['partial_tp']) * (active_trade['lot'] / 2.0) * CONTRACT_SIZE
                #     balance += profit_dollars
                #     trades.append({'type': 'SELL', 'result': 'PARTIAL_WIN', 'pnl': profit_dollars})
                #     active_trade['partial_taken'] = True
                #     active_trade['lot'] /= 2.0
                #     active_trade['sl'] = active_trade['entry'] # Break-even

                # Check SL and TP
                if c_high >= active_trade['sl']:
                    if active_trade['sl'] == active_trade['entry']:
                        pnl = 0.0
                        res = 'BREAK_EVEN'
                    else:
                        pnl = -(active_trade['sl'] - active_trade['entry']) * active_trade['lot'] * CONTRACT_SIZE
                        res = 'LOSS'
                    balance += pnl
                    trades.append({'type': 'SELL', 'result': res, 'pnl': pnl})
                    active_trade = None
                elif c_low <= active_trade['tp']:
                    profit_dollars = (active_trade['entry'] - active_trade['tp']) * active_trade['lot'] * CONTRACT_SIZE
                    balance += profit_dollars
                    trades.append({'type': 'SELL', 'result': 'WIN', 'pnl': profit_dollars})
                    active_trade = None
        else:
            if is_fakeout[i] == 1 and armed[i]:
                fakeout_h, fakeout_l = c_high, c_low
                atr = atrs[i]

                if upside[i] == 1:
                    sl = fakeout_h + (atr * atr_sl_mult)
                    pending_trade = {'type': 'SELL', 'sl': sl}
                elif downside[i] == 1:
                    sl = fakeout_l - (atr * atr_sl_mult)
                    pending_trade = {'type': 'BUY', 'sl': sl}

    # Process results
    net_profit = balance - INITIAL_BALANCE
    
    real_trades_pnl = []
    current_pnl = 0.0
    for t in trades:
        current_pnl += t['pnl']
        if t['result'] in ('WIN', 'LOSS', 'BREAK_EVEN'):
            real_trades_pnl.append(current_pnl)
            current_pnl = 0.0
            
    total_trades = len(real_trades_pnl)
    wins = len([p for p in real_trades_pnl if p > 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    # Ranking Formula: Penalize low win rates aggressively
    score = net_profit * (win_rate / 100.0)
    
    return {
        "take_profit_rr": take_profit_rr,
        "atr_sl_mult": atr_sl_mult,
        "net_profit": round(net_profit, 2),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "score": round(score, 2)
    }

def run_optimizer():
    print("=" * 60)
    print("🚀 GOLDEN AI - STRATEGY OPTIMIZER (Grid Search)")
    print("=" * 60)
    
    # 1. Load data
    date_to = date.today()
    date_from = date_to - timedelta(days=730) # ~2 years of data
    
    print(f"\n[1/4] 📥 Pulling MT5 Data from {date_from} to {date_to}...")
    try:
        df_raw = pull_data_by_date(SYMBOL, TIMEFRAME, date_from, date_to)
    except Exception as e:
        print(f"❌ Error pulling data: {e}")
        return
        
    print(f"      ✅ Loaded {len(df_raw)} candles.")
    
    # 2. Engineer Features
    print("\n[2/4] ⚙️ Engineering Features (Using pre-trained configs)...")
    df = engineer_features(df_raw.copy())
    
    # 3. Load Models & Predict
    print("\n[3/4] 🧠 Running AI Ensemble Predictions...")
    models, meta_info = load_ensemble_models()
    features = meta_info["features"]
    threshold = meta_info["threshold"]
    
    fakeout_idx = df[df["is_fakeout"] == 1].index
    for col in features:
        if col not in df.columns: df[col] = 0.0
        
    df["ai_prob"] = 0.0
    df["armed"] = False
    
    if len(fakeout_idx) > 0:
        X = df.loc[fakeout_idx, features]
        df.loc[fakeout_idx, "ai_prob"] = predict_ensemble(X, models, meta_info)
        df["armed"] = df["ai_prob"] >= threshold
        armed_count = df["armed"].sum()
        print(f"      ✅ Found {len(fakeout_idx)} fakeouts. {armed_count} armed by AI.")
    else:
        print("      ❌ No fakeouts found in the dataset.")
        return
        
    # 4. Simulation Loop Grid Search
    print(f"\n[4/4] ⚡ Running fast permutations ({len(PARAM_GRID['take_profit_rr']) * len(PARAM_GRID['atr_sl_mult'])} combinations)...")
    results = []
    start_time = time.time()
    
    for tp_rr in PARAM_GRID["take_profit_rr"]:
        for sl_mult in PARAM_GRID["atr_sl_mult"]:
            res = simulate_trades(df, tp_rr, sl_mult)
            results.append(res)
                    
    elapsed = time.time() - start_time
    print(f"      ✅ Optimization completed in {elapsed:.2f} seconds.")
    
    if not results:
        return
        
    # Sort and rank
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="score", ascending=False).reset_index(drop=True)
    
    # Save to CSV
    csv_file = "optimization_results.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"\n💾 Full results saved to {csv_file}")
    
    print("\n🏆 TOP 10 STRATEGY CONFIGURATIONS 🏆")
    print("-" * 80)
    print(f"{'Rank':<5} | {'TP RR':<6} | {'SL Mult':<8} | {'Trades':<6} | {'Win%':<6} | {'Net Pnl':<10} | {'Score':<10}")
    print("-" * 80)
    for i, row in results_df.head(10).iterrows():
        print(f"{(i+1):<5} | {row['take_profit_rr']:<6} | {row['atr_sl_mult']:<8} | {row['total_trades']:<6} | {row['win_rate']:<6.1f} | ${row['net_profit']:<9.2f} | {row['score']:<10.2f}")
    print("-" * 80)

if __name__ == "__main__":
    run_optimizer()
