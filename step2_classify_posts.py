"""
STEP 2: Classify posts by market-relevant topic
================================================
Three backend options — choose based on your setup:

  --backend keyword   Zero cost, zero setup. Rule-based keyword matching.
                      Best starting point. Catches ~80% of cases correctly.

  --backend ollama    Free local LLM inference via Ollama.
                      Requires: ollama.com → install → `ollama pull llama3`
                      No API costs, runs on your machine, good quality.

  --backend claude    Claude API (most accurate, small cost ~$0.50-1 total).
                      Requires: ANTHROPIC_API_KEY env var
                      console.anthropic.com → API Keys → Create Key

Recommendation: run --backend keyword first to see volume/topic breakdown,
then --backend ollama or --backend claude to improve ambiguous cases.

All backends write to the same output file. The cache is shared — switching
backends won't re-classify posts already in the cache.

Run:
  python step2_classify_posts.py --backend keyword   # fastest, free
  python step2_classify_posts.py --backend ollama    # free, better quality
  python step2_classify_posts.py --backend claude    # best quality, small cost

Output: data/classified_posts.csv
        data/classification_cache.json
"""

import re
import json
import time
import argparse
import pandas as pd
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
IN_PATH    = Path("data/raw_posts.csv")
OUT_PATH   = Path("data/classified_posts.csv")
CACHE_PATH = Path("data/classification_cache.json")

VALID_TOPICS = {
    "tariffs", "fed_rates", "specific_equity",
    "geopolitics", "crypto", "energy", "irrelevant",
}
VALID_SENTIMENTS = {"bullish", "bearish", "neutral", "mixed"}


# ── BACKEND 1: Keyword rules ──────────────────────────────────────────────────

KEYWORD_RULES = {
    "tariffs": {
        "keywords": [
            "tariff", "tariffs", "trade war", "trade deal", "import duty",
            "import tax", "export", "wto", "trade deficit", "trade surplus",
            "china trade", "mexico trade", "canada trade", "reciprocal",
            "section 301", "section 232", "steel tariff", "aluminum tariff",
            "trade representative", "ustr",
        ],
        "bearish_words":  ["tariff", "trade war", "import duty", "impose"],
        "bullish_words":  ["trade deal", "agreement", "resolved", "lifted"],
    },
    "fed_rates": {
        "keywords": [
            "federal reserve", "fed ", "interest rate", "interest rates",
            "jerome powell", "powell", "fomc", "rate cut", "rate hike",
            "monetary policy", "inflation", "cpi", "pce", "quantitative",
            "treasury yield", "basis point", "tightening", "pivot",
        ],
        "bearish_words":  ["rate hike", "tightening", "hawkish", "raise rates"],
        "bullish_words":  ["rate cut", "dovish", "pivot", "lower rates"],
    },
    "geopolitics": {
        "keywords": [
            "war", "ukraine", "russia", "nato", "china", "taiwan", "iran",
            "north korea", "sanctions", "military", "troops", "invasion",
            "missile", "nuclear", "pentagon", "defense", "foreign policy",
            "middle east", "israel", "hamas", "ceasefire", "diplomat",
        ],
        "bearish_words":  ["war", "invasion", "sanctions", "conflict", "attack"],
        "bullish_words":  ["ceasefire", "peace", "deal", "agreement", "resolved"],
    },
    "specific_equity": {
        "keywords": [
            "apple", "tesla", "nvidia", "microsoft", "google", "amazon",
            "meta ", "facebook", "boeing", "ford", "gm ", "general motors",
            "bank of america", "jpmorgan", "goldman", "elon musk", "aapl",
            "tsla", "nvda", "msft", "stock market", "wall street", "ipo",
            "sec ", "earnings", "short seller",
        ],
        "bearish_words":  ["investigate", "fine", "sanction", "bankrupt", "fraud"],
        "bullish_words":  ["deal", "contract", "great company", "doing well"],
    },
    "crypto": {
        "keywords": [
            "bitcoin", "crypto", "cryptocurrency", "ethereum", "blockchain",
            "digital currency", "btc", "eth", "coinbase", "binance",
            "defi", "nft", "token", "stablecoin",
        ],
        "bearish_words":  ["ban", "illegal", "crackdown", "regulate harshly"],
        "bullish_words":  ["bitcoin reserve", "strategic reserve", "support crypto"],
    },
    "energy": {
        "keywords": [
            "oil", "gas", "opec", "energy", "gasoline", "pipeline",
            "lng", "natural gas", "spr", "strategic petroleum",
            "drill", "fossil fuel", "renewable", "solar", "wind power",
            "keystone", "crude", "barrel", "energy policy",
        ],
        "bearish_words":  ["cut production", "opec cut", "restrict drilling"],
        "bullish_words":  ["drill baby drill", "increase production", "spr release"],
    },
}

IRRELEVANT_SIGNALS = [
    "happy birthday", "congratulations", "merry christmas", "happy new year",
    "great speech", "rally", "fake news media", "witch hunt", "the radical left",
    "sleepy", "crooked", "lamestream", "mainstream media",
    "subscribe to", "watch my", "tune in", "join me",
    "god bless", "prayers", "prayer",
]


def keyword_classify(text: str) -> dict:
    text_lower = text.lower()

    # Check irrelevant first
    if any(sig in text_lower for sig in IRRELEVANT_SIGNALS):
        irrelevant_score = sum(1 for s in IRRELEVANT_SIGNALS if s in text_lower)
        if irrelevant_score >= 2:
            return {
                "topic": "irrelevant", "sentiment": "neutral",
                "confidence": 0.85, "tickers": [],
                "reasoning": "keyword: irrelevant signals detected",
            }

    # Score each topic
    scores = {}
    for topic, rules in KEYWORD_RULES.items():
        hits = sum(1 for kw in rules["keywords"] if kw in text_lower)
        scores[topic] = hits

    best_topic = max(scores, key=scores.get)
    best_score = scores[best_topic]

    if best_score == 0:
        return {
            "topic": "irrelevant", "sentiment": "neutral",
            "confidence": 0.6, "tickers": [],
            "reasoning": "keyword: no market keywords found",
        }

    # Confidence scales with keyword hits, capped at 0.85
    confidence = min(0.4 + (best_score * 0.1), 0.85)

    # Sentiment from directional keywords
    rules = KEYWORD_RULES[best_topic]
    bearish_hits = sum(1 for w in rules.get("bearish_words", []) if w in text_lower)
    bullish_hits = sum(1 for w in rules.get("bullish_words", []) if w in text_lower)

    if bullish_hits > bearish_hits:
        sentiment = "bullish"
    elif bearish_hits > bullish_hits:
        sentiment = "bearish"
    elif bearish_hits > 0 and bullish_hits > 0:
        sentiment = "mixed"
    else:
        sentiment = "neutral"

    # Simple ticker extraction
    ticker_pattern = r'\b([A-Z]{2,5})\b'
    common_non_tickers = {
        "I", "A", "US", "UK", "EU", "UN", "FBI", "CIA", "DOJ", "SEC",
        "GOP", "DNC", "NYC", "DC", "FL", "TX", "CA", "NY", "COVID",
        "NATO", "MAGA", "FAKE", "NEWS", "TRUE", "TRUTH",
    }
    tickers = [
        m for m in re.findall(ticker_pattern, text)
        if m not in common_non_tickers
    ][:5]

    return {
        "topic":      best_topic,
        "sentiment":  sentiment,
        "confidence": confidence,
        "tickers":    tickers,
        "reasoning":  f"keyword: {best_score} hits for {best_topic}",
    }

def _validate(result: dict) -> dict:
    if result.get("topic") not in VALID_TOPICS:
        result["topic"] = "irrelevant"
    if result.get("sentiment") not in VALID_SENTIMENTS:
        result["sentiment"] = "neutral"
    result["confidence"] = float(result.get("confidence", 0.0))
    result["tickers"]    = result.get("tickers", [])
    result["reasoning"]  = result.get("reasoning", "")
    return result
 
 
# ── Cache helpers ─────────────────────────────────────────────────────────────
 
def load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}
 
 
def save_cache(cache: dict):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["keyword", "ollama", "claude"],
        default="keyword",
        help="Classification backend (default: keyword — free, no setup)",
    )
    parser.add_argument(
        "--ollama-model", default="llama3",
        help="Ollama model name (default: llama3)",
    )
    args = parser.parse_args()
 
    print(f"[info] Backend: {args.backend}")
    if args.backend == "keyword":
        print("       Free rule-based classification — no API needed")
    elif args.backend == "ollama":
        print(f"       Local LLM: {args.ollama_model}")
        print("       Requires: ollama running at localhost:11434")
        print("       Install:  https://ollama.com  then  ollama pull llama3")
    elif args.backend == "claude":
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise EnvironmentError(
                "Set ANTHROPIC_API_KEY: export ANTHROPIC_API_KEY='sk-ant-xxx'\n"
                "Get key: console.anthropic.com → API Keys → Create Key"
            )
        import anthropic
        claude_client = anthropic.Anthropic()
        print("       Claude API — small cost, best quality")
 
    df = pd.read_csv(IN_PATH)
    print(f"\n[1/3] Loaded {len(df):,} posts")
 
    cache = load_cache()
    already = sum(1 for pid in df["id"].astype(str) if pid in cache)
    print(f"[2/3] Cache: {already} done, {len(df)-already} to classify")
 
    results = []
    for i, row in df.iterrows():
        post_id = str(row["id"])
        text    = str(row["text"])
 
        if post_id in cache:
            results.append(cache[post_id])
            continue
 
        if args.backend == "keyword":
            result = keyword_classify(text)
        elif args.backend == "ollama":
            result = ollama_classify(text, model=args.ollama_model)
            time.sleep(0.1)
        else:  # claude
            result = claude_classify(claude_client, text)
            time.sleep(0.35)
 
        cache[post_id] = result
        results.append(result)
 
        if (i + 1) % 100 == 0:
            save_cache(cache)
            print(f"  → {i+1}/{len(df)} classified")
 
    save_cache(cache)
 
    results_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    df.to_csv(OUT_PATH, index=False)
 
    print(f"\n── Topic breakdown ─────────────────────────────────────────")
    print(df["topic"].value_counts().to_string())
    relevant = df[df["confidence"] >= 0.4]
    print(f"\n── Market-relevant (confidence ≥ 0.4): {len(relevant)} posts")
    print(f"[✓] Saved → {OUT_PATH}")
 
 
if __name__ == "__main__":
    main()