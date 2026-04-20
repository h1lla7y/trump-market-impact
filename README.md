# Trump Truth Social → SPY Market Impact Analysis

An event study examining whether Donald Trump's Truth Social posts produce measurable impact (abnormal returns and microstructure distortions) in SPY (S&P 500 ETF).

## Overview

This project combines NLP-based post classification with high-frequency market microstructure data to quantify the price impact, liquidity withdrawal, and volatility response triggered by Trump's posts across different market-relevant topics.

**Data window**: 6-month rolling lookback  
**Posts analyzed**: 3,215 total | ~500 market-relevant | 286 events studied  
**Market data**: Nasdaq ITCH MBP-10 (top 10 bid/ask levels), [-35 min, +15 min] event window

---

## Methodology

### Step 1 — Fetch Posts (`step1_fetch_posts.py`)
Pulls Trump's Truth Social archive from CNN's public endpoint (updated every 5 min). Filters to market hours (9:30 AM – 4:00 PM ET), strips HTML, and outputs structured CSV.

### Step 2 — Classify by Topic (`step2_classify_posts.py`)
Three interchangeable backends:
- **Keyword** — rule-based, fast, ~80% accuracy
- **Ollama** — local LLM (llama3), free, no API calls
- **Claude** — Anthropic Claude API, highest accuracy (~$0.50–1.00 total)

Topics detected: `tariffs`, `fed_rates`, `geopolitics`, `specific_equity`, `crypto`, `energy`, `irrelevant`  
Each post also gets a sentiment label: `bullish`, `bearish`, `neutral`, `mixed`

### Step 3 — Fetch Market Data (`step3_fetch_market_data.py`)
Fetches XNAS.ITCH MBP-10 order book data for SPY around each post timestamp via Databento. Computes mid-price, bid-ask spread (bps), order book imbalance, and bid/ask depth per snapshot.

### Step 4 — Event Study (`step4_compute_abnormal_returnsv2.py`)
**Standardized Abnormal Return (SAR)**:
- Baseline volatility: σ estimated from [-35, -5] min pre-event window
- `SAR(h) = cumret(h) / (σ × √h)` at horizons h = 1, 2, 5, 10, 15 min
- |SAR| > 2 → statistically significant at ~95% confidence

**Microstructure metrics per event**:
| Metric | Definition |
|---|---|
| `spread_change_bps` | Bid-ask spread widening post-event |
| `ob_imbalance` | Order book imbalance shift (directional pressure) |
| `depth_change_pct` | Liquidity withdrawal (%) |
| `price_impact_bps` | 5-min mid-price move in basis points |
| `rv_ratio` | Realized volatility post/pre (> 1 = elevated volatility) |
| `depth_min_pct` | Minimum depth as % of baseline |
| `depth_recovery_min` | Minutes to recover to 80% baseline depth |

### Step 5 — Static Visualization (`step5_visualise.py`)
Matplotlib/seaborn charts: SAR heatmap by topic × horizon, average return paths, sentiment hit rates, event volume by topic.

### Step 6 — Interactive Dashboard (`step6_dashboard.py`)
Streamlit + Plotly dashboard with filtering, drill-down, and real-time updates.

---

## Key Findings

### Post Volume by Topic
| Topic | Events |
|---|---|
| Geopolitics | 211 |
| Tariffs | 28 |
| Energy | 13 |
| Specific equity | 14 |
| Crypto | 11 |
| Fed rates | 9 |

### Abnormal Returns (5-min horizon)
- **Geopolitics**: SAR = −0.24 (statistically significant bearish drift)
- **Specific equity**: SAR = −0.69 (strongest negative abnormal return)
- **Tariffs**: Positive directional alignment 71.4% of the time (highest hit rate)
- **Specific equity**: 0% sentiment hit rate — bullish stock mentions produce *negative* SPY returns

### Microstructure Response (Geopolitics posts, p < 0.05)
- Spread widening: **+0.68 bps**
- Order book imbalance: **+0.008** (net selling pressure)
- Depth depletion: **−1.3%** immediately post-event
- Realized volatility ratio: **1.12** (12% increase in volatility)
- Depth falls to **~46% of baseline** at trough; recovers to 80% in ~5 seconds

### Cross-topic Patterns
- All topics show `rv_ratio > 1` (volatility universally elevated post-post)
- Tariff posts trigger **extreme depth depletion** (~4.4% drop)
- Crypto posts have the **lowest SPY impact** of all market-relevant topics
- Geopolitics dominates statistically due to high event count (211 events)

---
---

## Data & Costs

| Source | Cost | Notes |
|---|---|---|
| Truth Social posts | Free | Via CNN public archive endpoint |
| Keyword classifier | Free | No external API needed |
| Ollama classifier | Free | Requires local llama3 model |
| Claude classifier | ~$0.50–1.00 | Per full 6-month run |
| Databento market data | ~$0.50–2.00 | Per full 6-month run |

---

---

## Limitations

- Post timestamps from CNN archive may have minor latency vs. actual post time
- SPY only — single ETF does not capture sector-level rotation
- Event window contamination possible when posts cluster within 35 minutes
- Classification accuracy varies by backend; keyword method may miss nuanced posts
- Small sample sizes for some topics (fed_rates: 9 events) reduce statistical power

---

## License

MIT
