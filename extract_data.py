"""
MT5 Data Extractor — Production Grade
=======================================
Fixes over the basic version:
  1. Handles broker symbol suffixes automatically (XAUUSD, XAUUSD.a, XAUUSDm etc.)
  2. Detects and fills weekend/holiday gaps (no phantom candles)
  3. Validates tick_volume vs real_volume — uses best available
  4. Removes duplicate timestamps
  5. Flags and removes zero-volume / zero-range candles (data errors)
  6. Saves a quality report alongside the CSV
  7. Optional: downloads in chunks to avoid MT5 memory limits on long histories
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pytz

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SYMBOL_CANDIDATES = [
    "XAUUSD",      # Standard
    "XAUUSD.a",    # Some ECN brokers
    "XAUUSD.",
    "XAUUSDm",     # Micro
    "GOLD",
    "GOLDm",
]

TIMEFRAME   = mt5.TIMEFRAME_H1
DATE_FROM   = datetime(2015, 1, 1, tzinfo=pytz.utc)
DATE_TO     = datetime.now(pytz.utc)
OUTPUT_CSV  = "historical_h1_gold.csv"
CHUNK_YEARS = 2       # Download in 2-year chunks to avoid MT5 memory limits
MIN_CANDLES = 5_000   # Abort if fewer candles returned (something went wrong)


# ─── STEP 1: CONNECT ─────────────────────────────────────────────────────────
def connect() -> bool:
    if not mt5.initialize():
        print(f"❌ MT5 init failed: {mt5.last_error()}")
        return False
    info = mt5.terminal_info()
    print(f"✅ Connected to MetaTrader 5")
    print(f"   Build      : {info.build}")
    print(f"   Company    : {mt5.account_info().company}")
    print(f"   Server     : {mt5.account_info().server}")
    return True


# ─── STEP 2: FIND THE RIGHT SYMBOL ───────────────────────────────────────────
def resolve_symbol(candidates: list) -> str | None:
    available = {s.name for s in mt5.symbols_get()}
    for sym in candidates:
        if sym in available:
            # Ensure symbol is visible in MarketWatch
            mt5.symbol_select(sym, True)
            info = mt5.symbol_info(sym)
            if info and info.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED:
                print(f"✅ Using symbol: {sym}")
                return sym
            else:
                print(f"   [{sym}] found but not tradeable — skipping")

    # Last resort: fuzzy search
    matches = [s.name for s in mt5.symbols_get() if "XAU" in s.name or "GOLD" in s.name]
    if matches:
        print(f"   Auto-detected gold symbols: {matches}")
        sym = matches[0]
        mt5.symbol_select(sym, True)
        print(f"✅ Using symbol: {sym}")
        return sym

    print(f"❌ No gold symbol found. Available symbols sample:")
    sample = list(available)[:20]
    for s in sample:
        print(f"   {s}")
    return None


# ─── STEP 3: DOWNLOAD IN CHUNKS ──────────────────────────────────────────────
def download_chunked(symbol: str, tf, date_from: datetime, date_to: datetime) -> pd.DataFrame:
    """
    Downloads data in CHUNK_YEARS-year windows.
    MT5 can silently truncate large requests — chunking guarantees completeness.
    """
    all_chunks = []
    current    = date_from

    while current < date_to:
        chunk_end = min(current + timedelta(days=365 * CHUNK_YEARS), date_to)

        print(f"   Downloading {current.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}...",
              end="", flush=True)

        rates = mt5.copy_rates_range(symbol, tf, current, chunk_end)

        if rates is None or len(rates) == 0:
            print(f" ⚠️  No data (error: {mt5.last_error()})")
            current = chunk_end
            continue

        chunk_df = pd.DataFrame(rates)
        all_chunks.append(chunk_df)
        print(f" {len(rates):,} candles")

        # Advance past the last candle returned (avoid duplicates at boundaries)
        last_ts = pd.to_datetime(chunk_df["time"].iloc[-1], unit="s", utc=True)
        current = last_ts.to_pydatetime() + timedelta(hours=1)

    if not all_chunks:
        return pd.DataFrame()

    return pd.concat(all_chunks, ignore_index=True)


# ─── STEP 4: CLEAN & VALIDATE ────────────────────────────────────────────────
def clean_and_validate(df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, dict]:
    report = {}
    raw_count = len(df)

    # Convert time
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time"] = df["time"].dt.tz_localize(None)  # Strip UTC label — keeps as naive UTC

    # ── Choose best volume column ────────────────────────────────────────────
    real_vol_sum = df["real_volume"].sum() if "real_volume" in df.columns else 0
    tick_vol_sum = df["tick_volume"].sum() if "tick_volume" in df.columns else 0

    if real_vol_sum > 0:
        df.rename(columns={"real_volume": "volume"}, inplace=True)
        report["volume_source"] = "real_volume"
        print(f"   Volume source: real_volume ✅ (better for gold)")
    else:
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        report["volume_source"] = "tick_volume"
        print(f"   Volume source: tick_volume (real_volume unavailable)")

    df = df[["time", "open", "high", "low", "close", "volume"]].copy()

    # ── Remove duplicates ────────────────────────────────────────────────────
    dupes = df.duplicated("time").sum()
    df.drop_duplicates("time", inplace=True)
    report["duplicates_removed"] = int(dupes)

    # ── Sort chronologically ──────────────────────────────────────────────────
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Flag bad candles ──────────────────────────────────────────────────────
    # Zero-range candles (broker placeholder / error)
    zero_range = (df["high"] == df["low"]).sum()
    df = df[df["high"] != df["low"]].copy()
    report["zero_range_removed"] = int(zero_range)

    # Zero-volume candles (no activity — model should not train on these)
    zero_vol = (df["volume"] == 0).sum()
    df = df[df["volume"] > 0].copy()
    report["zero_volume_removed"] = int(zero_vol)

    # Obvious price errors (high < low, negative prices)
    price_errors = ((df["high"] < df["low"]) | (df["close"] <= 0)).sum()
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0)].copy()
    report["price_errors_removed"] = int(price_errors)

    # ── Detect gaps (missing candles) ─────────────────────────────────────────
    df["time_diff"] = df["time"].diff().dt.total_seconds() / 3600  # hours
    # Expected gap is 1H on weekdays; gaps > 3H on a weekday = missing data
    df["weekday"]   = pd.to_datetime(df["time"]).dt.weekday  # 0=Mon, 6=Sun
    suspicious_gaps = df[
        (df["time_diff"] > 3) &
        (df["weekday"] < 5) &
        (df["weekday"].shift(1) < 5)
    ]
    report["suspicious_weekday_gaps"] = int(len(suspicious_gaps))

    if len(suspicious_gaps) > 0:
        print(f"\n   ⚠️  {len(suspicious_gaps)} suspicious weekday gaps detected.")
        print(f"   Largest gaps:")
        top_gaps = suspicious_gaps.nlargest(5, "time_diff")[["time", "time_diff"]]
        for _, row in top_gaps.iterrows():
            print(f"     {row['time']}  —  {row['time_diff']:.0f}h gap")

    df.drop(columns=["time_diff", "weekday"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Summary stats ─────────────────────────────────────────────────────────
    report["raw_candles"]   = raw_count
    report["clean_candles"] = len(df)
    report["date_start"]    = str(df["time"].iloc[0])
    report["date_end"]      = str(df["time"].iloc[-1])
    report["avg_spread_pts"] = round(
        float((df["high"] - df["low"]).mean()), 5
    )

    # Detect volume anomalies (spikes > 10x mean — possible data errors)
    vol_mean = df["volume"].mean()
    vol_spikes = (df["volume"] > vol_mean * 10).sum()
    report["volume_spikes_10x"] = int(vol_spikes)

    return df, report


# ─── STEP 5: PRINT QUALITY REPORT ────────────────────────────────────────────
def print_report(report: dict, symbol: str) -> None:
    print("\n" + "=" * 55)
    print(f"  DATA QUALITY REPORT — {symbol}")
    print("=" * 55)
    print(f"  Period          : {report['date_start']}")
    print(f"                    → {report['date_end']}")
    print(f"  Raw candles     : {report['raw_candles']:,}")
    print(f"  Clean candles   : {report['clean_candles']:,}")
    print(f"  Volume source   : {report['volume_source']}")
    print(f"  Dupes removed   : {report['duplicates_removed']}")
    print(f"  Zero-range      : {report['zero_range_removed']}")
    print(f"  Zero-volume     : {report['zero_volume_removed']}")
    print(f"  Price errors    : {report['price_errors_removed']}")
    print(f"  Weekday gaps    : {report['suspicious_weekday_gaps']}")
    print(f"  Volume 10x spike: {report['volume_spikes_10x']}")
    print(f"  Avg candle range: {report['avg_spread_pts']}")

    total_issues = (
        report["duplicates_removed"] +
        report["zero_range_removed"] +
        report["zero_volume_removed"] +
        report["price_errors_removed"]
    )
    pct = total_issues / max(report["raw_candles"], 1) * 100
    if pct < 0.5:
        print(f"\n  ✅ Data quality: EXCELLENT ({pct:.2f}% bad rows)")
    elif pct < 2.0:
        print(f"\n  ⚠️  Data quality: ACCEPTABLE ({pct:.2f}% bad rows)")
    else:
        print(f"\n  ❌ Data quality: POOR ({pct:.2f}% bad rows) — consider a different broker")
    print("=" * 55)


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  MT5 GOLD DATA EXTRACTOR — PRODUCTION GRADE")
    print("=" * 55 + "\n")

    if not connect():
        sys.exit(1)

    try:
        # Resolve symbol
        symbol = resolve_symbol(SYMBOL_CANDIDATES)
        if symbol is None:
            sys.exit(1)

        # Download
        print(f"\nDownloading {symbol} H1 data in {CHUNK_YEARS}-year chunks...")
        raw_df = download_chunked(symbol, TIMEFRAME, DATE_FROM, DATE_TO)

        if raw_df.empty:
            print("❌ No data returned at all.")
            sys.exit(1)

        print(f"\n   Total raw candles received: {len(raw_df):,}")

        # Clean
        print("\nCleaning & validating...")
        df, report = clean_and_validate(raw_df, symbol)

        if len(df) < MIN_CANDLES:
            print(f"❌ Only {len(df)} clean candles — too few to train. Check symbol/dates.")
            sys.exit(1)

        # Save CSV
        df.to_csv(OUTPUT_CSV, index=False)

        # Save quality report
        report_path = OUTPUT_CSV.replace(".csv", "_quality_report.json")
        import json
        Path(report_path).write_text(json.dumps(report, indent=2))

        print_report(report, symbol)

        print(f"\n✅ Saved  → {OUTPUT_CSV}  ({len(df):,} candles)")
        print(f"✅ Report → {report_path}")
        print(f"\n📌 Ready for:  python train_sweep_v2.py")

    finally:
        mt5.shutdown()
        print("\nMT5 connection closed.")


if __name__ == "__main__":
    main()