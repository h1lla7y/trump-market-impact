"""
STEP 5: Visualisation dashboard
================================
Reads the outputs from steps 1-4 and produces a set of charts:

  1. SAR heatmap — topic × horizon
  2. SAR distribution per topic (violin plots)
  3. Cumulative return path — average per topic over 15 min
  4. Hit rate (sentiment alignment) per topic
  5. Post volume by topic over time

"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
RESULTS_PATH  = Path("data/event_study_results.csv")
SUMMARY_PATH  = Path("data/topic_summary.csv")
WINDOWS_DIR   = Path("data/market_windows")
CLASSIFIED    = Path("data/classified_posts.csv")
OUT_DIR       = Path("output")
HORIZONS      = [1, 2, 5, 10, 15]

TOPIC_COLORS = {
    "tariffs":         "#E63946",
    "fed_rates":       "#457B9D",
    "specific_equity": "#2A9D8F",
    "geopolitics":     "#E9C46A",
    "crypto":          "#F4A261",
    "energy":          "#A8DADC",
    "irrelevant":      "#CCCCCC",
}

plt.rcParams.update({
    "font.family":     "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "#0F0F0F",
    "axes.facecolor":   "#0F0F0F",
    "axes.labelcolor":  "#DDDDDD",
    "xtick.color":      "#AAAAAA",
    "ytick.color":      "#AAAAAA",
    "text.color":       "#DDDDDD",
    "grid.color":       "#222222",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
})


def load_data():
    results  = pd.read_csv(RESULTS_PATH, parse_dates=["created_at"])
    summary  = pd.read_csv(SUMMARY_PATH)
    posts    = pd.read_csv(CLASSIFIED, parse_dates=["created_at"])
    return results, summary, posts


def plot_sar_heatmap(ax, summary: pd.DataFrame):
    """Heatmap of mean SAR by topic × horizon."""
    topics = [t for t in TOPIC_COLORS if t != "irrelevant"
              and t in summary["topic"].values]

    matrix = pd.DataFrame(index=topics, columns=HORIZONS, dtype=float)
    sig_matrix = pd.DataFrame(index=topics, columns=HORIZONS, dtype=bool)

    for _, row in summary.iterrows():
        if row["topic"] in topics and row["horizon_min"] in HORIZONS:
            matrix.loc[row["topic"], row["horizon_min"]]     = row["mean_sar"]
            sig_matrix.loc[row["topic"], row["horizon_min"]] = row["significant"]

    matrix = matrix.astype(float)
    vmax = max(np.nanmax(np.abs(matrix.values)), 0.5) \
           if not matrix.empty else 1.0

    im = ax.imshow(
        matrix.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto"
    )

    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"{h}m" for h in HORIZONS], fontsize=9)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics, fontsize=9)
    ax.set_title("Mean Standardised Abnormal Return (SAR)\nGreen = bullish | Red = bearish",
                 fontsize=10, pad=10, color="#DDDDDD")

    # Annotate cells
    for i in range(len(topics)):
        for j in range(len(HORIZONS)):
            val = matrix.iloc[i, j]
            sig = sig_matrix.iloc[i, j]
            if not np.isnan(val):
                marker = "★" if sig else ""
                ax.text(j, i, f"{val:.2f}{marker}", ha="center", va="center",
                        fontsize=8, color="black" if abs(val) > vmax * 0.5 else "#DDDDDD")

    plt.colorbar(im, ax=ax, fraction=0.03, label="SAR")


def plot_cumret_paths(ax, results: pd.DataFrame):
    """Average cumulative return path by topic."""
    topics = [t for t in TOPIC_COLORS if t != "irrelevant"
              and t in results["topic"].values]

    for topic in topics:
        grp = results[results["topic"] == topic]
        means = []
        for h in HORIZONS:
            col = f"cumret_{h}m"
            if col in grp.columns:
                means.append(grp[col].mean() * 100)  # to percentage
            else:
                means.append(np.nan)

        color = TOPIC_COLORS.get(topic, "#FFFFFF")
        ax.plot([0] + HORIZONS, [0] + means, marker="o", markersize=4,
                label=topic, color=color, linewidth=1.8)

    ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Minutes after post")
    ax.set_ylabel("Avg cumulative return (%)")
    ax.set_title("Average Return Path by Topic\n(0 = post timestamp)",
                 fontsize=10, pad=10)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.2)
    ax.grid(True, axis="y")


def plot_hit_rate(ax, summary: pd.DataFrame):
    """Bar chart of sentiment alignment hit rate at 5m horizon."""
    s5 = summary[summary["horizon_min"] == 5].copy()
    s5 = s5[s5["topic"] != "irrelevant"].dropna(subset=["hit_rate"])
    s5 = s5.sort_values("hit_rate", ascending=True)

    colors = [TOPIC_COLORS.get(t, "#888888") for t in s5["topic"]]
    bars = ax.barh(s5["topic"], s5["hit_rate"], color=colors, alpha=0.85)

    ax.axvline(0.5, color="#FFFFFF", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Sentiment Alignment Hit Rate @ 5m\n(>50% = post direction predicts return)",
                 fontsize=10, pad=10)
    ax.set_xlabel("% events where return matches sentiment")

    for bar, val in zip(bars, s5["hit_rate"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", fontsize=8)


def plot_event_count(ax, posts: pd.DataFrame):
    """Bar chart: number of market-relevant posts per topic."""
    relevant = posts[
        (posts["confidence"] >= 0.4) & (posts["topic"] != "irrelevant")
    ]
    counts = relevant["topic"].value_counts()
    colors = [TOPIC_COLORS.get(t, "#888888") for t in counts.index]

    ax.barh(counts.index, counts.values, color=colors, alpha=0.85)
    ax.set_title("Market-Relevant Posts by Topic\n(confidence ≥ 0.4)",
                 fontsize=10, pad=10)
    ax.set_xlabel("Post count")

    for i, (topic, count) in enumerate(counts.items()):
        ax.text(count + 0.5, i, str(count), va="center", fontsize=8)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    if not RESULTS_PATH.exists() or not SUMMARY_PATH.exists():
        print("[!] Run steps 1–4 first to generate data.")
        return

    results, summary, posts = load_data()
    print(f"[✓] Loaded {len(results)} events, {len(summary)} summary rows")

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0F0F0F")
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_heatmap = fig.add_subplot(gs[0, :2])
    ax_paths   = fig.add_subplot(gs[1, :2])
    ax_hitrate = fig.add_subplot(gs[0, 2])
    ax_count   = fig.add_subplot(gs[1, 2])

    plot_sar_heatmap(ax_heatmap, summary)
    plot_cumret_paths(ax_paths, results)
    plot_hit_rate(ax_hitrate, summary)
    plot_event_count(ax_count, posts)

    fig.suptitle(
        "Trump Truth Social → SPY Market Impact Analysis",
        fontsize=14, fontweight="bold", color="#FFFFFF", y=0.98
    )

    out_path = OUT_DIR / "dashboard.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#0F0F0F")
    print(f"[✓] Dashboard saved → {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
