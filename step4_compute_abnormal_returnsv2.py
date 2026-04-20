"""
STEP 4: Compute abnormal returns + microstructure metrics from MBP-10 data
===========================================================================
Reads cached MBP-10 parquet windows from step 3 and computes:

  ── Price fix ──────────────────────────────────────────────────────────────
  Databento prices are stored as fixed-point int64 scaled by 1e9.
  Step 3 applies the /1e9 conversion when saving parquets.
  This file detects and corrects any double-conversion automatically.

  ── Abnormal return metrics ────────────────────────────────────────────────
  σ_baseline:   std of 1-second mid-price log returns in [-35min, -5min]
                converted to per-minute volatility (× sqrt(60))

  cumret(h):    cumulative log return from t=0 to t=h minutes
  SAR(h):       cumret(h) / (σ_baseline × √h)
                |SAR| > 2 = statistically notable at event level
  Horizons:     1, 2, 5, 10, 15 minutes

  ── Microstructure metrics ─────────────────────────────────────────────────
  spread_change_bps:   spread widening pre→post (uncertainty signal)
  imbalance_change:    ob_imbalance shift pre→post (directional pressure)
  depth_change_pct:    % change in total book depth (liquidity withdrawal)
  price_impact_bps:    5-min mid-price move in basis points

  ── Realized volatility ────────────────────────────────────────────────────
  rv_pre:   realized vol in [-5min, 0] window (1-second returns, annualised)
  rv_post:  realized vol in [0, +5min] window
  rv_ratio: rv_post / rv_pre  (> 1 = volatility increased after post)

  ── Depth depletion ────────────────────────────────────────────────────────
  Tracks total book depth (bid + ask) resampled to 5-second intervals
  across the full [-35, +15] window. Saved per-event for time-series plots.
  Summary metrics:
    depth_min_pct:      minimum depth reached as % of pre-event baseline
    depth_recovery_min: minutes after post until depth recovers to 80% baseline

"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
CLASSIFIED_PATH  = Path("data/classified_posts.csv")
WINDOWS_DIR      = Path("data/market_windows_ob")
RESULTS_PATH     = Path("data/event_study_results.csv")
SUMMARY_PATH     = Path("data/topic_summary.csv")
MICRO_PATH       = Path("data/microstructure_summary.csv")
DEPTH_DIR        = Path("data/depth_profiles")

HORIZONS         = [1, 2, 5, 10, 15]
BASELINE_START   = -35
BASELINE_END     = -5
MIN_BASELINE_OBS = 30   # minimum 1-second ticks for σ estimate

SPY_MIN_PRICE    = 50.0    # sanity bounds for double-conversion detection
SPY_MAX_PRICE    = 5000.0


# ── Price sanity check & fix ──────────────────────────────────────────────────

def fix_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and correct double-conversion of Databento fixed-point prices.
    Step 3 already divides by 1e9 when saving parquets.
    If mid_price mean is < 1.0 the conversion was applied twice — fix it.
    """
    df = df.copy()

    if "mid_price" not in df.columns:
        return df

    sample_mean = df["mid_price"].dropna().mean()

    if sample_mean < 1.0:
        print(f"    [price-fix] mid_price mean={sample_mean:.2e} — "
              f"applying ×1e9 correction")
        # Scale all price columns back up
        px_cols = [c for c in df.columns if "_px_" in c]
        for col in px_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce") * 1e9

        if "mid_price" in df.columns:
            df["mid_price"]    = df["mid_price"] * 1e9
        if "weighted_mid" in df.columns:
            df["weighted_mid"] = df["weighted_mid"] * 1e9
        if "spread" in df.columns:
            df["spread"]       = df["spread"] * 1e9

        # Recompute spread_bps with corrected prices
        if "bid_px_00" in df.columns and "ask_px_00" in df.columns:
            df["spread"]     = df["ask_px_00"] - df["bid_px_00"]
            df["spread_bps"] = np.where(
                df["mid_price"] > 0,
                (df["spread"] / df["mid_price"]) * 10_000,
                np.nan
            )

    elif sample_mean > SPY_MAX_PRICE:
        # Prices still in fixed-point — divide
        print(f"    [price-fix] mid_price mean={sample_mean:.2e} — "
              f"applying ÷1e9 correction")
        px_cols = [c for c in df.columns if "_px_" in c]
        for col in px_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 1e9
        if "mid_price" in df.columns:
            df["mid_price"]    = df["mid_price"] / 1e9
        if "weighted_mid" in df.columns:
            df["weighted_mid"] = df["weighted_mid"] / 1e9
        if "spread" in df.columns:
            df["spread"]       = df["spread"] / 1e9
        if "bid_px_00" in df.columns and "ask_px_00" in df.columns:
            df["spread"]     = df["ask_px_00"] - df["bid_px_00"]
            df["spread_bps"] = np.where(
                df["mid_price"] > 0,
                (df["spread"] / df["mid_price"]) * 10_000,
                np.nan
            )

    return df


# ── Resample to 1-second mid prices ──────────────────────────────────────────

def resample_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """
    MBP-10 has hundreds of ticks per minute.
    Resample to 1-second last mid_price for stable return calculations.
    """
    valid = df[df["mid_price"].notna() &
               (df["mid_price"] > SPY_MIN_PRICE) &
               (df["mid_price"] < SPY_MAX_PRICE)].copy()

    if valid.empty:
        return pd.DataFrame(columns=["mid_price", "minutes_from_post"])

    resampled = valid["mid_price"].resample("1s").last().ffill()
    result = resampled.to_frame()
    post_time = df["post_time"].iloc[0]
    result["minutes_from_post"] = (
        (result.index - post_time).total_seconds() / 60
    ).round(4)
    return result


# ── Baseline volatility ───────────────────────────────────────────────────────

def baseline_vol(df_1s: pd.DataFrame) -> float:
    """
    σ_baseline = std of 1-second log returns in [BASELINE_START, BASELINE_END].
    Scaled to per-minute by × sqrt(60).
    Returns NaN if insufficient observations.
    """
    pre = df_1s[
        (df_1s["minutes_from_post"] >= BASELINE_START) &
        (df_1s["minutes_from_post"] <= BASELINE_END)
    ]["mid_price"].dropna()

    if len(pre) < MIN_BASELINE_OBS:
        return np.nan

    log_rets = np.log(pre / pre.shift(1)).dropna()
    return log_rets.std() * np.sqrt(60)


# ── Cumulative return ─────────────────────────────────────────────────────────

def cumulative_return(df_1s: pd.DataFrame, horizon_min: int) -> float:
    """Log return from t=0 to t=horizon_min using 1-second mid prices."""
    at_zero = df_1s[df_1s["minutes_from_post"] >= 0]["mid_price"].dropna()
    at_h    = df_1s[df_1s["minutes_from_post"] <= horizon_min]["mid_price"].dropna()

    if at_zero.empty or at_h.empty:
        return np.nan

    p0 = at_zero.iloc[0]
    ph = at_h.iloc[-1]

    if p0 <= 0 or ph <= 0:
        return np.nan

    return np.log(ph / p0)


def sar(cum_ret: float, sigma: float, horizon: int) -> float:
    if any(np.isnan(x) for x in [cum_ret, sigma]) or sigma == 0:
        return np.nan
    return cum_ret / (sigma * np.sqrt(horizon))


# ── Realized volatility ───────────────────────────────────────────────────────

def realized_vol(df_1s: pd.DataFrame,
                 t_start: float, t_end: float) -> float:
    """
    Realized volatility (std of 1-second log returns) over a window.
    Returned as annualised-to-per-minute figure (× sqrt(60)).
    """
    window = df_1s[
        (df_1s["minutes_from_post"] >= t_start) &
        (df_1s["minutes_from_post"] <= t_end)
    ]["mid_price"].dropna()

    if len(window) < 5:
        return np.nan

    log_rets = np.log(window / window.shift(1)).dropna()
    return log_rets.std() * np.sqrt(60)


# ── Depth depletion ───────────────────────────────────────────────────────────

def depth_profile(df: pd.DataFrame, post_id: str) -> pd.DataFrame:
    """
    Resample total book depth (bid + ask) to 5-second intervals
    across the full event window. Used for time-series depth plots.
    """
    df = df.copy()

    if "bid_depth" not in df.columns or "ask_depth" not in df.columns:
        return pd.DataFrame()

    df["total_depth"] = df["bid_depth"] + df["ask_depth"]
    valid = df[df["total_depth"] > 0]["total_depth"]

    if valid.empty:
        return pd.DataFrame()

    resampled = valid.resample("5s").mean().ffill()
    profile = resampled.to_frame(name="total_depth")
    post_time = df["post_time"].iloc[0]
    profile["minutes_from_post"] = (
        (profile.index - post_time).total_seconds() / 60
    ).round(3)
    profile["post_id"] = post_id
    return profile


def depth_depletion_metrics(df: pd.DataFrame) -> dict:
    """
    Summary metrics for depth depletion around t=0.

    baseline_depth:     mean total depth in [-10min, -1min]
    depth_min_pct:      lowest depth reached in [0, +10min] as % of baseline
    depth_recovery_min: minutes after post until depth recovers to ≥80% baseline
                        NaN if depth never drops below 80% or never recovers
    depth_at_5m_pct:    depth at t=+5min as % of baseline
    """
    if "bid_depth" not in df.columns or "ask_depth" not in df.columns:
        return {
            "baseline_depth": np.nan, "depth_min_pct": np.nan,
            "depth_recovery_min": np.nan, "depth_at_5m_pct": np.nan,
        }

    df = df.copy()
    df["total_depth"] = df["bid_depth"] + df["ask_depth"]

    baseline = df[
        (df["minutes_from_post"] >= -10) &
        (df["minutes_from_post"] <= -1)
    ]["total_depth"].mean()

    if np.isnan(baseline) or baseline == 0:
        return {
            "baseline_depth": np.nan, "depth_min_pct": np.nan,
            "depth_recovery_min": np.nan, "depth_at_5m_pct": np.nan,
        }

    post_window = df[
        (df["minutes_from_post"] >= 0) &
        (df["minutes_from_post"] <= 10)
    ]

    if post_window.empty:
        return {
            "baseline_depth": baseline, "depth_min_pct": np.nan,
            "depth_recovery_min": np.nan, "depth_at_5m_pct": np.nan,
        }

    min_depth     = post_window["total_depth"].min()
    depth_min_pct = (min_depth / baseline) * 100

    # Recovery: first 5-second window where depth ≥ 80% of baseline
    profile = post_window["total_depth"].resample("5s").mean()
    recovery_threshold = baseline * 0.80
    recovered = profile[profile >= recovery_threshold]
    depth_recovery_min = (
        (recovered.index[0] - df["post_time"].iloc[0]).total_seconds() / 60
        if not recovered.empty else np.nan
    )

    # Depth at exactly +5 minutes
    at_5m = df[
        (df["minutes_from_post"] >= 4.9) &
        (df["minutes_from_post"] <= 5.1)
    ]["total_depth"].mean()
    depth_at_5m_pct = (at_5m / baseline * 100) if not np.isnan(at_5m) else np.nan

    return {
        "baseline_depth":      baseline,
        "depth_min_pct":       depth_min_pct,
        "depth_recovery_min":  depth_recovery_min,
        "depth_at_5m_pct":     depth_at_5m_pct,
    }


# ── Microstructure metrics ────────────────────────────────────────────────────

def microstructure_metrics(df: pd.DataFrame) -> dict:
    """
    Order book microstructure comparison pre vs post.
    PRE:  [-5min, 0]
    POST: [0, +2min]
    """
    pre  = df[(df["minutes_from_post"] >= -5)  & (df["minutes_from_post"] <= 0)]
    post = df[(df["minutes_from_post"] >=  0)  & (df["minutes_from_post"] <= 2)]

    def smean(series):
        v = series.dropna()
        return v.mean() if not v.empty else np.nan

    sp_pre  = smean(pre["spread_bps"])
    sp_post = smean(post["spread_bps"])
    sp_chg  = (sp_post - sp_pre) if not any(
        np.isnan(x) for x in [sp_pre, sp_post]) else np.nan

    imb_pre  = smean(pre["ob_imbalance"])
    imb_post = smean(post["ob_imbalance"])
    imb_chg  = (imb_post - imb_pre) if not any(
        np.isnan(x) for x in [imb_pre, imb_post]) else np.nan

    d_pre = smean((pre["bid_depth"] + pre["ask_depth"])
                  if "bid_depth" in pre.columns else pd.Series(dtype=float))
    d_post = smean((post["bid_depth"] + post["ask_depth"])
                   if "bid_depth" in post.columns else pd.Series(dtype=float))
    depth_chg_pct = ((d_post - d_pre) / d_pre * 100) if (
        not any(np.isnan(x) for x in [d_pre, d_post]) and d_pre > 0) else np.nan

    p0 = df[df["minutes_from_post"] >= 0]["mid_price"].dropna()
    p5 = df[df["minutes_from_post"] <= 5]["mid_price"].dropna()
    price_impact_bps = (np.log(p5.iloc[-1] / p0.iloc[0]) * 10_000
                        if not p0.empty and not p5.empty
                        and p0.iloc[0] > SPY_MIN_PRICE else np.nan)

    return {
        "spread_pre_bps":    sp_pre,
        "spread_post_bps":   sp_post,
        "spread_change_bps": sp_chg,
        "imbalance_pre":     imb_pre,
        "imbalance_post":    imb_post,
        "imbalance_change":  imb_chg,
        "depth_change_pct":  depth_chg_pct,
        "price_impact_bps":  price_impact_bps,
    }


# ── Per-event analysis ────────────────────────────────────────────────────────

def analyse_event(window_path: Path, post_meta: pd.Series) -> dict:
    df = pd.read_parquet(window_path)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    # Fix any price conversion issues
    df = fix_prices(df)

    # Resample to 1-second for return/vol calculations
    df_1s = resample_to_seconds(df)

    σ = baseline_vol(df_1s)

    result = {
        "post_id":          post_meta["id"],
        "created_at":       post_meta["created_at"],
        "topic":            post_meta["topic"],
        "sentiment":        post_meta["sentiment"],
        "confidence":       post_meta["confidence"],
        "sigma_baseline":   σ,
        "n_baseline_ticks": int(len(df[
            (df["minutes_from_post"] >= BASELINE_START) &
            (df["minutes_from_post"] <= BASELINE_END)
        ])),
    }

    # SAR at each horizon
    for h in HORIZONS:
        cr = cumulative_return(df_1s, h)
        result[f"cumret_{h}m"] = cr
        result[f"sar_{h}m"]    = sar(cr, σ, h)

    # Sentiment alignment at 5-min
    cr5 = result.get("cumret_5m", np.nan)
    if not np.isnan(cr5) if isinstance(cr5, float) else True:
        sent = post_meta["sentiment"]
        if sent == "bullish":
            result["sentiment_aligned"] = int(cr5 > 0)
        elif sent == "bearish":
            result["sentiment_aligned"] = int(cr5 < 0)
        else:
            result["sentiment_aligned"] = np.nan
    else:
        result["sentiment_aligned"] = np.nan

    # Realized volatility pre and post
    rv_pre  = realized_vol(df_1s, -5, 0)
    rv_post = realized_vol(df_1s,  0, 5)
    result["rv_pre"]   = rv_pre
    result["rv_post"]  = rv_post
    result["rv_ratio"] = (rv_post / rv_pre
                          if not any(np.isnan(x) for x in [rv_pre, rv_post])
                          and rv_pre > 0 else np.nan)

    # Depth depletion
    result.update(depth_depletion_metrics(df))

    # Microstructure
    result.update(microstructure_metrics(df))

    return result


def save_depth_profiles(window_path: Path, post_id: str):
    """Save 5-second depth profile for this event to its own parquet."""
    try:
        df = pd.read_parquet(window_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df = fix_prices(df)
        profile = depth_profile(df, post_id)
        if not profile.empty:
            profile.to_parquet(DEPTH_DIR / f"{post_id}.parquet")
    except Exception as e:
        print(f"  [depth-profile warn] {post_id}: {e}")


# ── Summary builders ──────────────────────────────────────────────────────────

def build_topic_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for topic in results["topic"].unique():
        grp = results[results["topic"] == topic]
        for h in HORIZONS:
            vals = grp[f"sar_{h}m"].dropna()
            if len(vals) < 3:
                continue
            t_stat, p_val = stats.ttest_1samp(vals, 0)
            rows.append({
                "topic":       topic,
                "horizon_min": h,
                "n_events":    len(vals),
                "mean_sar":    vals.mean(),
                "median_sar":  vals.median(),
                "std_sar":     vals.std(),
                "t_stat":      t_stat,
                "p_value":     p_val,
                "significant": p_val < 0.05,
                "hit_rate":    grp["sentiment_aligned"].mean(),
                "rv_ratio_mean": grp["rv_ratio"].mean(),
                "depth_min_pct_mean": grp["depth_min_pct"].mean(),
                "depth_recovery_mean": grp["depth_recovery_min"].mean(),
            })
    return (pd.DataFrame(rows)
            .sort_values(["topic", "horizon_min"])
            .reset_index(drop=True))


def build_microstructure_summary(results: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "spread_change_bps", "imbalance_change",
        "depth_change_pct",  "price_impact_bps",
        "rv_ratio",          "depth_min_pct",
        "depth_recovery_min",
    ]
    rows = []
    for topic in results["topic"].unique():
        grp = results[results["topic"] == topic]
        row = {"topic": topic, "n_events": len(grp)}
        for col in cols:
            if col not in grp.columns:
                continue
            vals = grp[col].dropna()
            row[f"{col}_mean"] = vals.mean() if len(vals) else np.nan
            if len(vals) >= 3:
                t, p = stats.ttest_1samp(vals, 0)
                row[f"{col}_pval"] = p
                row[f"{col}_sig"]  = p < 0.05
            else:
                row[f"{col}_pval"] = np.nan
                row[f"{col}_sig"]  = False
        rows.append(row)
    return (pd.DataFrame(rows)
            .sort_values("topic")
            .reset_index(drop=True))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)

    posts = pd.read_csv(CLASSIFIED_PATH, parse_dates=["created_at"])
    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True, format="ISO8601")    
    posts["id"] = posts["id"].astype(str)

    window_files = {p.stem: p for p in WINDOWS_DIR.glob("*.parquet")}
    print(f"[1/4] {len(window_files)} cached MBP-10 windows found")

    all_results, skipped = [], 0

    for i, (_, row) in enumerate(posts.iterrows()):
        pid = str(row["id"])
        if pid not in window_files:
            continue
        try:
            result = analyse_event(window_files[pid], row)
            all_results.append(result)
            save_depth_profiles(window_files[pid], pid)
            if (i + 1) % 50 == 0:
                print(f"  → {len(all_results)} events processed...")
        except Exception as e:
            print(f"  [warn] Skipped {pid}: {e}")
            skipped += 1

    print(f"[2/4] Analysed {len(all_results)} events ({skipped} skipped)")

    if not all_results:
        print("[!] No events found — run step3 first")
        return

    results_df = pd.DataFrame(all_results)

    # Verify prices look real before saving
    sample_price = results_df["sigma_baseline"].dropna().mean()
    print(f"[3/4] Price check — mean σ_baseline: {sample_price:.6f} "
          f"({'OK' if sample_price < 1 else 'check units'})")

    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"[✓] Event results          → {RESULTS_PATH}")

    summary = build_topic_summary(results_df)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"[✓] Topic summary          → {SUMMARY_PATH}")

    micro = build_microstructure_summary(results_df)
    micro.to_csv(MICRO_PATH, index=False)
    print(f"[✓] Microstructure summary → {MICRO_PATH}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  SAR by topic × horizon  (★ = p<0.05)")
    print("═" * 72)
    pivot     = summary.pivot_table(index="topic", columns="horizon_min",
                                    values="mean_sar",    aggfunc="first")
    sig_pivot = summary.pivot_table(index="topic", columns="horizon_min",
                                    values="significant", aggfunc="first")
    for topic in pivot.index:
        row_str = f"  {topic:<18}"
        for h in HORIZONS:
            if h in pivot.columns:
                val = pivot.loc[topic, h]
                sig = sig_pivot.loc[topic, h] if topic in sig_pivot.index else False
                marker = "★" if sig else " "
                row_str += f"  {h}m:{val:+.3f}{marker}"
        print(row_str)

    print("\n" + "═" * 72)
    print("  Volatility & Depth by topic")
    print("  rv_ratio > 1 = vol increased | depth_min_pct < 100 = liquidity pulled")
    print("═" * 72)
    for _, row in micro.iterrows():
        rv   = row.get("rv_ratio_mean", np.nan)
        dmin = row.get("depth_min_pct_mean", np.nan)
        drec = row.get("depth_recovery_min_mean", np.nan)
        pi   = row.get("price_impact_bps_mean", np.nan)
        print(f"  {row['topic']:<18}  "
              f"rv_ratio {rv:+.3f}  "
              f"depth_min {dmin:.1f}%  "
              f"recovery {drec:.1f}min  "
              f"impact {pi:+.1f}bps")

    print(f"\n[✓] Depth profiles → {DEPTH_DIR}/ "
          f"({len(list(DEPTH_DIR.glob('*.parquet')))} files)")


if __name__ == "__main__":
    main()
