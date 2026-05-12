import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import MetaTrader5 as mt5
from datetime import date

# Import from the ensemble-updated backend
from ai_backtester_v2 import pull_data_by_date, engineer_features, load_ensemble_models, predict_ensemble

st.set_page_config(page_title="Golden AI - V2 Backtester", layout="wide")

st.title("⚡ Golden 10H AI - Ensemble Backtest Engine")

# ==========================================
# SIDEBAR: SETTINGS & INPUTS
# ==========================================
st.sidebar.header("⚙️ Backtest Settings")
symbol = st.sidebar.text_input("Symbol", value="XAUUSD")

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Custom Period")
date_from = st.sidebar.date_input("Start Date", value=date(2026, 1, 1))
date_to = st.sidebar.date_input("End Date", value=date(2026, 5, 9))

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Strategy Parameters")
rolling_bars = st.sidebar.number_input("Rolling window (Box)", value=10, step=1)
take_profit_rr = st.sidebar.number_input("Risk:Reward ratio", value=3.0, step=0.5)
atr_period = st.sidebar.number_input("ATR period for SL", value=14, step=1)
atr_sl_mult = st.sidebar.number_input("ATR SL Multiplier", value=0.3, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Risk Management")
initial_balance = st.sidebar.number_input("Initial Balance ($)", value=10000.0)
lot_size = st.sidebar.number_input("Fixed Lot Size", value=0.1, step=0.01)

timeframe = mt5.TIMEFRAME_H1

# ==========================================
# BACKTEST EXECUTION
# ==========================================
if st.sidebar.button("🚀 RUN BACKTEST", use_container_width=True):
    with st.spinner("Downloading Data & Running Ensemble AI..."):
        try:
            # 1. Setup & Data
            models, meta_info = load_ensemble_models()
            features = meta_info["features"]
            threshold = meta_info["threshold"]
            
            df = pull_data_by_date(symbol, timeframe, date_from, date_to)
            df = engineer_features(df, rolling_bars=int(rolling_bars), atr_period=int(atr_period))
            
            # 2. Ensemble Predictions
            fakeout_idx = df[df["is_fakeout"] == 1].index
            for col in features:
                if col not in df.columns: df[col] = 0.0
                    
            if len(fakeout_idx) > 0:
                X = df.loc[fakeout_idx, features]
                df["ai_prob"] = 0.0
                df.loc[fakeout_idx, "ai_prob"] = predict_ensemble(X, models, meta_info)
                df["armed"] = df["ai_prob"] >= threshold
            else:
                st.warning("No fakeouts found in this date range.")
                st.stop()

            # 3. Simulation Loop (match live EA timing as closely as possible)
            balance = initial_balance
            equity_curve = [balance]
            trades = []
            active_trade = None
            pending_trade = None
            
            CONTRACT_SIZE = 100.0  # Standard XAUUSD contract size

            for i in range(1, len(df)):
                curr = df.iloc[i]

                if active_trade is None and pending_trade is not None:
                    entry = curr["open"]
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
                        "signal_time": pending_trade["signal_time"],
                        "lot": lot_size,
                        "partial_taken": False
                    }
                    pending_trade = None
                
                if active_trade:
                    if active_trade['type'] == 'BUY':
                        # --- PARTIAL PROFIT & BREAK-EVEN DISABLED ---
                        # if not active_trade['partial_taken'] and curr['high'] >= active_trade['partial_tp']:
                        #     profit_dollars = (active_trade['partial_tp'] - active_trade['entry']) * (active_trade['lot'] / 2.0) * CONTRACT_SIZE
                        #     balance += profit_dollars
                        #     trades.append({'time': curr['time'], 'type': 'BUY', 'result': 'PARTIAL_WIN', 'pnl': profit_dollars, 'entry': active_trade['entry']})
                        #     active_trade['partial_taken'] = True
                        #     active_trade['lot'] /= 2.0
                        #     active_trade['sl'] = active_trade['entry']

                        if curr['low'] <= active_trade['sl']:
                            if active_trade['sl'] == active_trade['entry']:
                                pnl = 0.0
                                res = 'BREAK_EVEN'
                            else:
                                pnl = -(active_trade['entry'] - active_trade['sl']) * active_trade['lot'] * CONTRACT_SIZE
                                res = 'LOSS'
                            balance += pnl
                            trades.append({'time': curr['time'], 'type': 'BUY', 'result': res, 'pnl': pnl, 'entry': active_trade['entry']})
                            active_trade = None
                        elif curr['high'] >= active_trade['tp']:
                            profit_dollars = (active_trade['tp'] - active_trade['entry']) * active_trade['lot'] * CONTRACT_SIZE
                            balance += profit_dollars
                            trades.append({'time': curr['time'], 'type': 'BUY', 'result': 'WIN', 'pnl': profit_dollars, 'entry': active_trade['entry']})
                            active_trade = None
                    elif active_trade['type'] == 'SELL':
                        # --- PARTIAL PROFIT & BREAK-EVEN DISABLED ---
                        # if not active_trade['partial_taken'] and curr['low'] <= active_trade['partial_tp']:
                        #     profit_dollars = (active_trade['entry'] - active_trade['partial_tp']) * (active_trade['lot'] / 2.0) * CONTRACT_SIZE
                        #     balance += profit_dollars
                        #     trades.append({'time': curr['time'], 'type': 'SELL', 'result': 'PARTIAL_WIN', 'pnl': profit_dollars, 'entry': active_trade['entry']})
                        #     active_trade['partial_taken'] = True
                        #     active_trade['lot'] /= 2.0
                        #     active_trade['sl'] = active_trade['entry']

                        if curr['high'] >= active_trade['sl']:
                            if active_trade['sl'] == active_trade['entry']:
                                pnl = 0.0
                                res = 'BREAK_EVEN'
                            else:
                                pnl = -(active_trade['sl'] - active_trade['entry']) * active_trade['lot'] * CONTRACT_SIZE
                                res = 'LOSS'
                            balance += pnl
                            trades.append({'time': curr['time'], 'type': 'SELL', 'result': res, 'pnl': pnl, 'entry': active_trade['entry']})
                            active_trade = None
                        elif curr['low'] <= active_trade['tp']:
                            profit_dollars = (active_trade['entry'] - active_trade['tp']) * active_trade['lot'] * CONTRACT_SIZE
                            balance += profit_dollars
                            trades.append({'time': curr['time'], 'type': 'SELL', 'result': 'WIN', 'pnl': profit_dollars, 'entry': active_trade['entry']})
                            active_trade = None
                else:
                    if curr["is_fakeout"] == 1 and curr["armed"]:
                        fakeout_h, fakeout_l = curr["high"], curr["low"]
                        atr = curr["atr_14"]

                        if curr["upside_fakeout"] == 1:
                            sl = fakeout_h + (atr * atr_sl_mult)
                            pending_trade = {
                                'type': 'SELL',
                                'sl': sl,
                                'signal_time': curr['time'],
                            }
                        elif curr["downside_fakeout"] == 1:
                            sl = fakeout_l - (atr * atr_sl_mult)
                            pending_trade = {
                                'type': 'BUY',
                                'sl': sl,
                                'signal_time': curr['time'],
                            }
                
                equity_curve.append(balance)

            # ==========================================
            # DASHBOARD RENDERING
            # ==========================================
            st.success("Backtest Complete!")
            
            real_trades_pnl = []
            current_pnl = 0.0
            for t in trades:
                current_pnl += t['pnl']
                if t['result'] in ('WIN', 'LOSS', 'BREAK_EVEN'):
                    real_trades_pnl.append(current_pnl)
                    current_pnl = 0.0
                    
            wins = len([p for p in real_trades_pnl if p > 0])
            total_trades = len(real_trades_pnl)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            net_profit = balance - initial_balance

            # KPI Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Net Profit", f"${net_profit:,.2f}", f"{(net_profit/initial_balance)*100:.2f}%")
            col2.metric("Win Rate", f"{win_rate:.1f}%")
            col3.metric("Total Trades", total_trades)
            col4.metric("Final Balance", f"${balance:,.2f}")

            # Plot 1: Equity Curve
            st.markdown("### 📈 Equity Curve")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(y=equity_curve, mode='lines', fill='tozeroy', line=dict(color='dodgerblue')))
            fig_eq.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0), template="plotly_dark")
            st.plotly_chart(fig_eq, use_container_width=True)

           # Plot 2: MT5-Style Advanced Candlestick Chart
            st.markdown("### 🕯️ Advanced Trade Execution Chart")
            
            # To keep the browser fast, we plot the last 1000 bars
            plot_df = df.tail(1000).copy()
            
            fig_chart = go.Figure()

            # 1. Base Candlesticks (MT5 Colors)
            fig_chart.add_trace(go.Candlestick(
                x=plot_df['time'], open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'],
                increasing_line_color='#00ff00', decreasing_line_color='#ff0000', 
                name='Price'
            ))

            # 2. Moving Averages
            fig_chart.add_trace(go.Scatter(
                x=plot_df['time'], y=plot_df['sma_20'], mode='lines', 
                line=dict(color='orange', width=1.5), name='SMA 20'
            ))
            fig_chart.add_trace(go.Scatter(
                x=plot_df['time'], y=plot_df['sma_50'], mode='lines', 
                line=dict(color='cyan', width=1.5), name='SMA 50'
            ))

            # 3. 10H Box Levels (Dashed lines like your MT5 EA)
            fig_chart.add_trace(go.Scatter(
                x=plot_df['time'], y=plot_df['box_high'], mode='lines', 
                line=dict(color='dodgerblue', width=1, dash='dash'), name='Box High'
            ))
            fig_chart.add_trace(go.Scatter(
                x=plot_df['time'], y=plot_df['box_low'], mode='lines', 
                line=dict(color='dodgerblue', width=1, dash='dash'), name='Box Low'
            ))

            # 4. Overlay Trades (Arrows)
            trade_df = pd.DataFrame(trades)
            if not trade_df.empty:
                visible_trades = trade_df[trade_df['time'] >= plot_df['time'].iloc[0]]
                
                buys = visible_trades[visible_trades['type'] == 'BUY']
                sells = visible_trades[visible_trades['type'] == 'SELL']

                # Buy Arrows (Below the candle)
                fig_chart.add_trace(go.Scatter(
                    x=buys['time'], y=buys['entry'] - (plot_df['atr_14'].mean() * 0.5), mode='markers',
                    marker=dict(symbol='triangle-up', size=14, color='lime', line=dict(color='black', width=1)), 
                    name='BUY Signal'
                ))
                
                # Sell Arrows (Above the candle)
                fig_chart.add_trace(go.Scatter(
                    x=sells['time'], y=sells['entry'] + (plot_df['atr_14'].mean() * 0.5), mode='markers',
                    marker=dict(symbol='triangle-down', size=14, color='magenta', line=dict(color='black', width=1)), 
                    name='SELL Signal'
                ))

            # 5. MT5 Layout Styling
            fig_chart.update_layout(
                height=750, 
                template="plotly_dark", 
                dragmode="pan",  # Allows clicking and dragging to scroll left/right
                xaxis_rangeslider_visible=True, # Adds the scrollbar at the bottom
                plot_bgcolor='#121212',  # Deep MT5 black/grey
                paper_bgcolor='#121212',
                yaxis=dict(side='right', title="Price", fixedrange=False), # MT5 puts price on the right
                xaxis=dict(title="Time", type='category', nticks=10, fixedrange=False), # Removes weekend gaps
                margin=dict(l=10, r=10, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_chart, use_container_width=True, config={'scrollZoom': True})

            # Display Trade Log below
            st.markdown("### 📝 Trade Log")
            st.dataframe(trade_df, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")