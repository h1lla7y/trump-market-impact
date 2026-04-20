"""
STEP 3: Pull SPY MBP-10 order book data from Databento for each event window
=============================================================================
Fetches top-10 levels of the order book (both sides) for a window of:
  [-35 min, +15 min] around each market-relevant post timestamp

Why MBP-10 over OHLCV:
  - Captures bid-ask spread widening (first reaction, within seconds of post)
  - Order book imbalance (leading indicator before price moves)
  - Depth depletion (liquidity pulled from one side pre-move)
  - OHLCV can be derived from MBP-10 — strictly more informative

Schema:  mbp-10  (market-by-price, top 10 levels, both sides)
Dataset: XNAS.ITCH (Nasdaq TotalView — where SPY is listed)

Each MBP-10 message contains:
  bid_px_00..bid_px_09  — bid prices at levels 0 (best) to 9
  ask_px_00..ask_px_09  — ask prices at levels 0 (best) to 9
  bid_sz_00..bid_sz_09  — bid sizes at each level
  ask_sz_00..ask_sz_09  — ask sizes at each level
  action: A(dd) C(ancel) M(odify) T(rade) F(ill)
"""

from dotenv import load_dotenv
load_dotenv()
import os
import sys
import time
import argparse
import databento as db
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone

# ── Config ───────────────────────────────────────────────────────────────────
IN_PATH      = Path("data/classified_posts.csv")
WINDOWS_DIR  = Path("data/market_windows_ob")
LOG_PATH     = Path("data/fetch_log_ob.csv")

SYMBOL         = "SPY"
DATASET        = "XNAS.ITCH"
SCHEMA         = "mbp-10"
PRE_MINUTES    = 35
POST_MINUTES   = 15
MIN_CONFIDENCE = 0.4

# Column name helpers for 10 price levels
BID_PX = [f"bid_px_0{i}" for i in range(10)]
ASK_PX = [f"ask_px_0{i}" for i in range(10)]
BID_SZ = [f"bid_sz_0{i}" for i in range(10)]
ASK_SZ = [f"ask_sz_0{i}" for i in range(10)]


# ── Fetch log helpers ─────────────────────────────────────────────────────────

def load_fetch_log() -> set:
    if LOG_PATH.exists():
        return set(pd.read_csv(LOG_PATH)["post_id"].astype(str).tolist())
    return set()


def append_fetch_log(post_id: str, status: str, rows: int):
    entry = pd.DataFrame([{
        "post_id":     post_id,
        "status":      status,
        "rows_fetched": rows,
        "fetched_at":  datetime.now(timezone.utc).isoformat(),
    }])
    entry.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)


# ── Order book enrichment ─────────────────────────────────────────────────────

def enrich_ob(df: pd.DataFrame, post_time: pd.Timestamp,
              post_id: str) -> pd.DataFrame:
    """
    Add derived microstructure metrics to raw MBP-10 ticks.

    New columns
    -----------
    mid_price       (best_bid + best_ask) / 2
    spread_bps      bid-ask spread in basis points
    bid_depth       total size summed across all 10 bid levels
    ask_depth       total size summed across all 10 ask levels
    ob_imbalance    (bid_depth - ask_depth) / total_depth
                    +1 = all bids (price pressure up)
                    -1 = all asks (price pressure down)
                     0 = balanced book
    weighted_mid    size-weighted mid — accounts for depth shape,
                    more robust than simple mid for large-tick instruments
    seconds_from_post
    minutes_from_post
    """
    df = df.copy()

    # Databento stores prices as fixed-point int64 scaled by 1e9 → convert
    price_cols = [c for c in df.columns if "_px_" in c]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 1e9

    # Mid price and spread
    b0 = "bid_px_00"
    a0 = "ask_px_00"
    if b0 in df.columns and a0 in df.columns:
        df["mid_price"]  = (df[b0] + df[a0]) / 2
        df["spread"]     = df[a0] - df[b0]
        df["spread_bps"] = (df["spread"] / df["mid_price"]) * 10_000
    else:
        df["mid_price"] = df["spread_bps"] = np.nan

    # Book depth (sum across levels present in data)
    eb_sz = [c for c in BID_SZ if c in df.columns]
    ea_sz = [c for c in ASK_SZ if c in df.columns]
    df["bid_depth"] = df[eb_sz].sum(axis=1).astype(float) if eb_sz else np.nan
    df["ask_depth"] = df[ea_sz].sum(axis=1).astype(float) if ea_sz else np.nan

    total = df["bid_depth"] + df["ask_depth"]
    df["ob_imbalance"] = np.where(
        total > 0,
        (df["bid_depth"] - df["ask_depth"]) / total,
        np.nan
    )

    # Volume-weighted mid price
    eb_px = [c for c in BID_PX if c in df.columns]
    ea_px = [c for c in ASK_PX if c in df.columns]
    if eb_px and ea_px and eb_sz and ea_sz:
        bid_wt = (df[eb_px].values * df[eb_sz].values).sum(axis=1)
        ask_wt = (df[ea_px].values * df[ea_sz].values).sum(axis=1)
        tot_sz = df[eb_sz].values.sum(axis=1) + df[ea_sz].values.sum(axis=1)
        df["weighted_mid"] = np.where(
            tot_sz > 0,
            (bid_wt + ask_wt) / tot_sz,
            df["mid_price"]
        )
    else:
        df["weighted_mid"] = df["mid_price"]

    # Time reference
    df["post_id"]           = post_id
    df["post_time"]         = post_time
    elapsed                 = (df.index - post_time).total_seconds()
    df["seconds_from_post"] = elapsed.round(1)
    df["minutes_from_post"] = (elapsed / 60).round(4)

    return df


# ── Databento fetch ───────────────────────────────────────────────────────────

def fetch_window(client: db.Historical, post_id: str,
                 post_time: pd.Timestamp) -> pd.DataFrame | None:
    start = post_time - timedelta(minutes=PRE_MINUTES)
    end   = post_time + timedelta(minutes=POST_MINUTES)

    try:
        raw = client.timeseries.get_range(
            dataset=DATASET,
            symbols=[SYMBOL],
            schema=SCHEMA,
            start=start.isoformat(),
            end=end.isoformat(),
            stype_in="raw_symbol",
        )
        df = raw.to_df()
        if df.empty:
            print(f"  [warn] No ticks for post {post_id} at {post_time}")
            return None

        df.index = pd.to_datetime(df.index, utc=True)
        df = enrich_ob(df, post_time, post_id)
        return df

    except Exception as e:
        print(f"  [error] {post_id}: {e}")
        return None


def check_cost(client: db.Historical, relevant: pd.DataFrame):
    """Print Databento cost estimate without pulling any data."""
    start = (relevant["created_at"].min()
             - timedelta(minutes=PRE_MINUTES)).isoformat()
    end   = (relevant["created_at"].max()
             + timedelta(minutes=POST_MINUTES)).isoformat()

    print("\n── Cost estimate (no data pulled) ───────────────────────────")
    try:
        worst = client.metadata.get_cost(
            dataset=DATASET, symbols=[SYMBOL],
            schema=SCHEMA, start=start, end=end,
        )
        # Approximate actual cost: events × window_minutes / total_minutes
        total_min = (relevant["created_at"].max()
                     - relevant["created_at"].min()
                     ).total_seconds() / 60
        window_min = (PRE_MINUTES + POST_MINUTES) * len(relevant)
        approx = worst * (window_min / total_min) if total_min > 0 else worst

        print(f"  Full range (worst case):        ${worst:.4f}")
        print(f"  Event-windowed estimate ({len(relevant)} events): ${approx:.4f}")
        print(f"  Tip: run --limit 3 first to verify data shape before full pull")
    except Exception as e:
        print(f"  Could not retrieve estimate: {e}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch MBP-10 event windows")
    parser.add_argument("--check-cost", action="store_true",
                        help="Show cost estimate and exit")
    parser.add_argument("--limit", type=int, default=None,
                        help="Fetch only first N events (testing)")
    args = parser.parse_args()

    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)

    posts = pd.read_csv(IN_PATH, parse_dates=["created_at"])
    posts["created_at"] = pd.to_datetime(posts["created_at"], format="ISO8601", utc=True)

    relevant = posts[
        (posts["confidence"] >= MIN_CONFIDENCE) &
        (posts["during_market_hours"] == True) &
        (posts["topic"] != "irrelevant") &
        (posts["created_at"] < pd.Timestamp("2026-04-04", tz="UTC")) # <-- ADD THIS LINE

    ].copy()

    print(f"[1/3] {len(relevant)} market-relevant in-hours events")
    print(f"      {relevant['topic'].value_counts().to_dict()}")

    api_key = os.environ.get("DATABENTO_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "Set your key: export DATABENTO_KEY='db-xxxxxxxx'"
        )
    client = db.Historical(key=api_key)

    if args.check_cost:
        check_cost(client, relevant)
        sys.exit(0)

    already = load_fetch_log()
    to_fetch = relevant[~relevant["id"].astype(str).isin(already)]
    if args.limit:
        to_fetch = to_fetch.head(args.limit)
        print(f"[info] --limit {args.limit} active")

    print(f"[2/3] {len(to_fetch)} to fetch | {len(already)} cached")

    if to_fetch.empty:
        print("[✓] All windows already cached")
        return

    fetched = skipped = 0

    for _, row in to_fetch.iterrows():
        post_id   = str(row["id"])
        post_time = row["created_at"]
        out_file  = WINDOWS_DIR / f"{post_id}.parquet"

        df = fetch_window(client, post_id, post_time)

        if df is not None:
            df.to_parquet(out_file)
            append_fetch_log(post_id, "ok", len(df))
            fetched += 1
            avg_spread = df["spread_bps"].mean()
            avg_imb    = df["ob_imbalance"].mean()
            print(f"  [{fetched}] {post_id} → {len(df):,} ticks | "
                  f"spread {avg_spread:.2f}bps | imbalance {avg_imb:+.3f}")
        else:
            append_fetch_log(post_id, "no_data", 0)
            skipped += 1

        time.sleep(0.5)

    print(f"\n[✓] Done — {fetched} fetched, {skipped} skipped")
    print(f"    Saved → {WINDOWS_DIR}/")


if __name__ == "__main__":
    main()
