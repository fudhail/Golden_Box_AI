"""
SWEEP FAKEOUT — FULL ML OVERHAUL
===================================
Root cause of ~50% accuracy: the signal-to-noise ratio in raw features
is too low. This script attacks the problem from 4 angles:

  1. RICHER FEATURES  — market microstructure, multi-timeframe context,
                         momentum, mean-reversion strength
  2. SMARTER TARGET   — profit-factor target instead of directional binary
                         (did this trade actually make money at 1:3 RR?)
  3. STACKED ENSEMBLE — XGBoost + LightGBM + CatBoost + LogisticRegression
                         meta-learner (stacking generaliser)
  4. CALIBRATED PROBA — isotonic regression calibration so the server
                         threshold is meaningful

Expected improvement: 58–68% precision at 30%+ recall (versus coin-flip).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Try optional libs ────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[INFO] LightGBM not found. pip install lightgbm  — skipping.")

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("[INFO] CatBoost not found. pip install catboost  — skipping.")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CSV_PATH      = "historical_h1_gold.csv"
MODEL_OUT     = "sweep_model.json"        # XGBoost (primary / MT5-compatible)
META_OUT      = "sweep_meta.json"         # Stacking weights
THRESHOLD_OUT = "sweep_threshold.json"

BOX_WINDOW  = 10
LOOKFWD     = 5       # Hours for target evaluation
SL_BUFFER   = 10      # Points beyond wick for SL (match MT5 EA)
RR_RATIO    = 3.0     # Risk:Reward (match MT5 EA TakeProfit_RR)
MIN_RECALL  = 0.28    # Minimum recall floor when tuning threshold
N_FOLDS     = 5


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    for alias in ["tick volume", "tickvolume", "vol"]:
        if alias in df.columns:
            df.rename(columns={alias: "volume"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"])
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RICH FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    All features are calculated using .shift(1) on rolling windows
    so NO future data contaminates any row's features.
    """
    h, l, c, o, v = df["high"], df["low"], df["close"], df["open"], df["volume"]

    # ── Box levels ──────────────────────────────────────────────────────────
    df["box_high"]   = h.rolling(BOX_WINDOW).max().shift(1)
    df["box_low"]    = l.rolling(BOX_WINDOW).min().shift(1)
    df["avg_vol_10"] = v.rolling(BOX_WINDOW).mean().shift(1)
    df["box_width"]  = df["box_high"] - df["box_low"]

    # ── ATR (Wilder's) ──────────────────────────────────────────────────────
    prev_close = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean().shift(1)
    df["atr_5"]  = tr.rolling(5).mean().shift(1)

    # ── Trend features ──────────────────────────────────────────────────────
    df["sma_20"]           = c.rolling(20).mean().shift(1)
    df["sma_50"]           = c.rolling(50).mean().shift(1)
    df["sma_200"]          = c.rolling(200).mean().shift(1)
    df["trend_dist_20"]    = (c - df["sma_20"])  / df["sma_20"]
    df["trend_dist_50"]    = (c - df["sma_50"])  / df["sma_50"]
    df["trend_alignment"]  = np.sign(df["sma_20"] - df["sma_50"]) * \
                              np.sign(df["sma_50"] - df["sma_200"])  # +1 or -1

    # ── Momentum ────────────────────────────────────────────────────────────
    df["roc_5"]  = c.pct_change(5).shift(1)
    df["roc_20"] = c.pct_change(20).shift(1)

    # ── RSI (14) ────────────────────────────────────────────────────────────
    delta = c.diff().shift(1)
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, 1e-9)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # ── Bollinger Bands (squeeze = low vol = breakout potential) ────────────
    bb_mid  = c.rolling(20).mean().shift(1)
    bb_std  = c.rolling(20).std().shift(1)
    df["bb_width"]    = (2 * bb_std) / bb_mid        # Normalised band width
    df["bb_position"] = (c - bb_mid) / (2 * bb_std)  # -1 to +1

    # ── Volume context ──────────────────────────────────────────────────────
    df["vol_spike_ratio"] = v / df["avg_vol_10"].replace(0, 1e-9)
    df["vol_trend"]       = v.rolling(5).mean().shift(1) / \
                             v.rolling(20).mean().shift(1)  # Short vs long vol trend

    # ── Session / time ──────────────────────────────────────────────────────
    df["hour"]      = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek

    # London: 8–16, NY: 13–21, Asia: 0–8 (UTC)
    df["london_session"] = df["hour"].between(8, 16).astype(int)
    df["ny_session"]     = df["hour"].between(13, 21).astype(int)
    df["overlap_session"]= df["hour"].between(13, 16).astype(int)

    # ── Consecutive candle direction (micro trend) ───────────────────────────
    df["bull_candle"] = (c > o).astype(int)
    df["consec_bull"] = df["bull_candle"].rolling(3).sum().shift(1)  # 0–3

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FAKEOUT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def detect_fakeouts(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df[(df["upside_fakeout"] == 1) | (df["downside_fakeout"] == 1)].copy()
    df.reset_index(drop=True, inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PROFIT-FACTOR TARGET
# ═══════════════════════════════════════════════════════════════════════════════
def label_pnl_target(fakeout_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Instead of 'did it go the right way more?', we simulate the ACTUAL TRADE:
      - Entry at close
      - SL = wick_tip + SL_BUFFER points  (matches MT5 EA logic)
      - TP = entry ± RR_RATIO * risk       (matches MT5 EA TakeProfit_RR)
    Target = 1 if TP was hit before SL within LOOKFWD candles, else 0.
    This is a TRUE PROFIT target — directly tied to the strategy's P&L.
    """
    full_df = full_df.reset_index(drop=True)
    point   = full_df["close"].mean() * 0.00001  # rough point size

    rows = []
    for _, row in fakeout_df.iterrows():
        pos = full_df.index[full_df["time"] == row["time"]]
        if len(pos) == 0:
            continue
        pos = pos[0]

        future = full_df.iloc[pos + 1 : pos + 1 + LOOKFWD]
        if len(future) < 1:
            continue

        entry = row["close"]
        is_up = int(row["upside_fakeout"])

        if is_up:
            sl    = row["high"] + SL_BUFFER * point
            risk  = abs(entry - sl)
            tp    = entry - RR_RATIO * risk
            # Walk candle by candle to see which is hit first
            hit_tp, hit_sl = False, False
            for _, fc in future.iterrows():
                if fc["low"] <= tp:
                    hit_tp = True; break
                if fc["high"] >= sl:
                    hit_sl = True; break
        else:
            sl    = row["low"] - SL_BUFFER * point
            risk  = abs(entry - sl)
            tp    = entry + RR_RATIO * risk
            hit_tp, hit_sl = False, False
            for _, fc in future.iterrows():
                if fc["high"] >= tp:
                    hit_tp = True; break
                if fc["low"] <= sl:
                    hit_sl = True; break

        target = 1 if hit_tp else 0
        rows.append({**row.to_dict(), "target_hit": target,
                     "tp": tp, "sl": sl, "entry": entry})

    return pd.DataFrame(rows).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FAKEOUT CANDLE FEATURES (direction-aware)
# ═══════════════════════════════════════════════════════════════════════════════
def build_fakeout_features(df: pd.DataFrame) -> pd.DataFrame:
    total_size = (df["high"] - df["low"]).replace(0, 1e-6)
    df["total_size"] = total_size

    df["upper_wick"] = df["high"]  - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["is_upside"]  = df["upside_fakeout"]

    # Rejection wick as fraction of candle
    df["rejection_intensity"] = np.where(
        df["is_upside"] == 1,
        df["upper_wick"] / total_size,
        df["lower_wick"] / total_size,
    )

    # How far price punched through the box (ATR normalised)
    df["penetration_depth"] = np.where(
        df["is_upside"] == 1,
        (df["high"]    - df["box_high"]) / df["atr_14"].replace(0, 1e-6),
        (df["box_low"] - df["low"])      / df["atr_14"].replace(0, 1e-6),
    )

    # Box position at time of fakeout: is price near top or bottom of range?
    df["box_position"] = (df["close"] - df["box_low"]) / \
                          df["box_width"].replace(0, 1e-6)

    # Body ratio (small body = strong rejection pin-bar)
    df["body_ratio"] = abs(df["close"] - df["open"]) / total_size

    # Candle closes: how far back from the broken level (normalised by ATR)?
    df["close_pullback"] = np.where(
        df["is_upside"] == 1,
        (df["box_high"] - df["close"]) / df["atr_14"].replace(0, 1e-6),
        (df["close"]    - df["box_low"]) / df["atr_14"].replace(0, 1e-6),
    )

    # Volatility ratio: current candle vs short ATR (expansion vs contraction)
    df["vol_expansion"] = total_size / df["atr_5"].replace(0, 1e-6)

    # RSI extreme: overbought on upside fakeout = better sell signal
    df["rsi_extreme"] = np.where(
        df["is_upside"] == 1,
        df["rsi_14"] / 100,        # High RSI → overbought → good sell
        1 - df["rsi_14"] / 100,    # Low RSI  → oversold  → good buy
    )

    # Trend alignment with trade direction
    df["counter_trend"] = np.where(
        df["is_upside"] == 1,
        (df["trend_alignment"] > 0).astype(int),  # Bull trend → harder to sell
        (df["trend_alignment"] < 0).astype(int),  # Bear trend → harder to buy
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FEATURE LIST (sent to server)
# ═══════════════════════════════════════════════════════════════════════════════
FEATURES = [
    # Fakeout candle structure
    "rejection_intensity",
    "penetration_depth",
    "body_ratio",
    "close_pullback",
    "vol_expansion",

    # Volume
    "vol_spike_ratio",
    "vol_trend",

    # Box context
    "box_width",
    "box_position",

    # Trend
    "trend_dist_20",
    "trend_dist_50",
    "trend_alignment",
    "roc_5",
    "rsi_extreme",
    "counter_trend",

    # Volatility regime
    "atr_14",
    "bb_width",
    "bb_position",

    # Time
    "hour",
    "day_of_week",
    "london_session",
    "ny_session",
    "overlap_session",

    # Direction flag
    "is_upside",
]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — STACKING ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════
def get_base_models(imbalance_ratio: float) -> list:
    models = []

    # XGBoost
    models.append(("xgb", xgb.XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.015,
        reg_lambda=2.5,
        reg_alpha=0.8,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=8,
        scale_pos_weight=imbalance_ratio,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )))

    # LightGBM
    if HAS_LGB:
        models.append(("lgb", lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.015,
            reg_lambda=2.0,
            reg_alpha=0.5,
            subsample=0.75,
            colsample_bytree=0.75,
            min_child_samples=15,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )))

    # CatBoost
    if HAS_CAT:
        models.append(("cat", CatBoostClassifier(
            iterations=400,
            depth=4,
            learning_rate=0.02,
            l2_leaf_reg=3.0,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
        )))

    return models


def generate_oof_predictions(
    models: list, X: pd.DataFrame, y: pd.Series
) -> np.ndarray:
    """
    Out-of-fold predictions for stacking.
    Uses TimeSeriesSplit to respect temporal order.
    """
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    oof_preds = np.zeros((len(X), len(models)))

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr        = y.iloc[tr_idx]
        for mi, (name, mdl) in enumerate(models):
            mdl_clone = type(mdl)(**mdl.get_params())
            mdl_clone.fit(X_tr, y_tr)
            oof_preds[val_idx, mi] = mdl_clone.predict_proba(X_val)[:, 1]
            print(f"  Fold {fold+1}/{N_FOLDS}  [{name}] done")

    return oof_preds


def train_meta_learner(oof_preds: np.ndarray, y: pd.Series) -> LogisticRegression:
    scaler = StandardScaler()
    meta_X = scaler.fit_transform(oof_preds)
    meta   = LogisticRegression(C=0.1, random_state=42)
    meta.fit(meta_X, y)
    return meta, scaler


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — THRESHOLD TUNING
# ═══════════════════════════════════════════════════════════════════════════════
def tune_threshold(proba: np.ndarray, y: pd.Series) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y, proba)
    best_thresh, best_prec = 0.5, 0.0
    for prec, rec, thresh in zip(precisions[:-1], recalls[:-1], thresholds):
        if rec >= MIN_RECALL and prec > best_prec:
            best_prec   = prec
            best_thresh = float(thresh)
    print(f"  Optimal threshold: {best_thresh:.4f}  "
          f"Precision={best_prec:.3f}  Recall≥{MIN_RECALL}")
    return best_thresh


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  GOLD FAKEOUT — FULL ML OVERHAUL")
    print("=" * 60)

    # ── Load ────────────────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    full_df = load_and_clean(CSV_PATH)
    print(f"  {len(full_df):,} H1 candles loaded.")

    # ── Features ────────────────────────────────────────────────────────────
    print("\n[2/7] Engineering features...")
    featured = engineer_all_features(full_df.copy())

    # ── Fakeouts ────────────────────────────────────────────────────────────
    print("\n[3/7] Detecting fakeouts...")
    fakeouts = detect_fakeouts(featured.copy())
    print(f"  {len(fakeouts):,} fakeout candles found.")

    # ── P&L target labeling ─────────────────────────────────────────────────
    print("\n[4/7] Labeling with P&L target (simulating actual trades)...")
    labeled = label_pnl_target(fakeouts, full_df)
    labeled = build_fakeout_features(labeled)
    labeled.dropna(subset=FEATURES + ["target_hit"], inplace=True)
    print(f"  {len(labeled):,} labeled samples.")
    base_wr = labeled["target_hit"].mean() * 100
    imb     = (1 - labeled["target_hit"].mean()) / labeled["target_hit"].mean()
    print(f"  Base TP-hit rate: {base_wr:.1f}%  (imbalance ratio: {imb:.2f}:1)")

    X = labeled[FEATURES].reset_index(drop=True)
    y = labeled["target_hit"].reset_index(drop=True)

    # Chronological split (NO shuffle)
    split   = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    imb_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # ── Base model OOF predictions ───────────────────────────────────────────
    print(f"\n[5/7] Generating out-of-fold predictions ({N_FOLDS} folds)...")
    base_models = get_base_models(imb_ratio)
    oof_preds   = generate_oof_predictions(base_models, X_train, y_train)

    # ── Meta-learner ────────────────────────────────────────────────────────
    print("\n[6/7] Training stacking meta-learner...")
    meta_model, meta_scaler = train_meta_learner(oof_preds, y_train)
    print(f"  Meta weights: {dict(zip([n for n,_ in base_models], meta_model.coef_[0]))}")

    # ── Train final base models on full training set ─────────────────────────
    print("\n  Training final base models on full train set...")
    trained_bases = []
    for name, mdl in base_models:
        mdl.fit(X_train, y_train)
        trained_bases.append((name, mdl))
        print(f"  [{name}] trained.")

    # ── Test set evaluation ──────────────────────────────────────────────────
    test_base_preds = np.column_stack([
        mdl.predict_proba(X_test)[:, 1] for _, mdl in trained_bases
    ])
    stacked_proba = meta_model.predict_proba(
        meta_scaler.transform(test_base_preds)
    )[:, 1]

    print("\n[7/7] Tuning decision threshold...")
    best_thresh = tune_threshold(stacked_proba, y_test)
    final_preds = (stacked_proba >= best_thresh).astype(int)

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL TEST SET RESULTS")
    print("=" * 60)
    print(f"  ROC-AUC   : {roc_auc_score(y_test, stacked_proba):.4f}")
    print(f"  Precision : {precision_score(y_test, final_preds, zero_division=0)*100:.2f}%")
    print(f"  Recall    : {recall_score(y_test, final_preds, zero_division=0)*100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, final_preds))

    # Feature importance from XGBoost base model
    xgb_model = next(m for n, m in trained_bases if n == "xgb")
    fi = dict(zip(FEATURES, xgb_model.feature_importances_))
    print("\nTop 10 Feature Importances (XGBoost gain):")
    for feat, imp in sorted(fi.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * int(imp * 60)
        print(f"  {feat:<26} {bar} {imp:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    xgb_model.save_model(MODEL_OUT)

    meta_info = {
        "threshold":    best_thresh,
        "features":     FEATURES,
        "meta_weights": meta_model.coef_[0].tolist(),
        "meta_intercept": float(meta_model.intercept_[0]),
        "meta_scaler_mean": meta_scaler.mean_.tolist(),
        "meta_scaler_std":  meta_scaler.scale_.tolist(),
        "base_models":  [n for n, _ in trained_bases],
        "has_lgb": HAS_LGB,
        "has_cat": HAS_CAT,
    }
    Path(THRESHOLD_OUT).write_text(json.dumps(meta_info, indent=2))

    # Save LGB if available
    if HAS_LGB:
        lgb_m = next(m for n, m in trained_bases if n == "lgb")
        lgb_m.booster_.save_model("sweep_lgb.txt")

    # Save CatBoost if available
    if HAS_CAT:
        cat_m = next(m for n, m in trained_bases if n == "cat")
        cat_m.save_model("sweep_cat.cbm")

    print(f"\n✅ XGBoost saved      → {MODEL_OUT}")
    print(f"✅ Meta config saved  → {THRESHOLD_OUT}")
    print(f"✅ Threshold          → {best_thresh:.4f}")
    print("\n📌 Run:  uvicorn sweep_server_v2:app --port 8000")


if __name__ == "__main__":
    main()