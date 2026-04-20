"""
STEP 1: Fetch Trump's Truth Social posts
========================================
Pulls from CNN's live archive (updated every 5 min, free, no auth).
Filters to the last 6 months and saves to disk.

Run: python step1_fetch_posts.py
Output: data/raw_posts.csv
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
CNN_ARCHIVE_URL = "https://ix.cnn.io/data/truth-social/truth_archive.json"
LOOKBACK_DAYS   = 180          # 6 months
OUT_PATH        = Path("data/raw_posts.csv")


def fetch_posts(lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    print(f"[1/3] Fetching Truth Social archive from CNN...")
    r = requests.get(CNN_ARCHIVE_URL, timeout=30)
    r.raise_for_status()
    data = r.json()
    print(f"      → {len(data):,} total posts retrieved")

    df = pd.DataFrame(data)

    # ── Normalise timestamp ──────────────────────────────────────────────────
    # CNN archive uses 'created_at' in ISO-8601 UTC
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)

    # ── Filter to lookback window ────────────────────────────────────────────
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    df = df[df["created_at"] >= cutoff].copy()
    print(f"      → {len(df):,} posts in last {lookback_days} days")

    # ── Clean content field (strip HTML tags) ────────────────────────────────
    df["text"] = (
        df["content"]
        .str.replace(r"<[^>]+>", " ", regex=True)   # remove HTML
        .str.replace(r"\s+", " ", regex=True)         # collapse whitespace
        .str.strip()
    )

    # ── Select and sort ──────────────────────────────────────────────────────
    keep = ["id", "created_at", "text", "url",
            "replies_count", "reblogs_count", "favourites_count"]
    # only keep columns that actually exist in the response
    keep = [c for c in keep if c in df.columns]
    df = df[keep].sort_values("created_at").reset_index(drop=True)

    # ── Market hours flag ────────────────────────────────────────────────────
    # NYSE: 09:30–16:00 ET (UTC-4 in summer, UTC-5 in winter)
    # We use a simple UTC offset approximation; pytz not required
    df["hour_et"] = df["created_at"].dt.tz_convert("America/New_York").dt.hour
    df["minute_et"] = df["created_at"].dt.tz_convert("America/New_York").dt.minute
    df["during_market_hours"] = (
        (df["hour_et"] > 9) | ((df["hour_et"] == 9) & (df["minute_et"] >= 30))
    ) & (df["hour_et"] < 16)

    in_hours  = df["during_market_hours"].sum()
    out_hours = (~df["during_market_hours"]).sum()
    print(f"      → {in_hours} during market hours | {out_hours} outside market hours")

    return df


def main():
    OUT_PATH.parent.mkdir(exist_ok=True)
    df = fetch_posts()
    df.to_csv(OUT_PATH, index=False)
    print(f"\n[✓] Saved {len(df):,} posts → {OUT_PATH}")
    print(f"\nSample:\n{df[['created_at','during_market_hours','text']].head(3).to_string()}")


if __name__ == "__main__":
    main()
