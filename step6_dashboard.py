"""
Trump Truth Social → SPY Market Impact Dashboard
=================================================
Interactive Streamlit dashboard with Plotly charts.

Install:
  pip install streamlit plotly pandas numpy scipy

Run:
  streamlit run step6_dashboard.py

  Then open http://localhost:8501 in your browser.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
from scipy import stats

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trump → SPY Impact",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #080c10;
    color: #c9d1d9;
  }
  .main { background-color: #080c10; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.5px;
  }
  h1 { color: #e6edf3; font-size: 1.6rem; font-weight: 600; }
  h2 { color: #c9d1d9; font-size: 1.1rem; font-weight: 500;
       border-bottom: 1px solid #21262d; padding-bottom: 6px; }
  h3 { color: #8b949e; font-size: 0.85rem; font-weight: 400;
       text-transform: uppercase; letter-spacing: 1px; }

  .metric-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 8px;
  }
  .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 4px;
  }
  .metric-label {
    font-size: 0.72rem;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-family: 'IBM Plex Mono', monospace;
  }
  .metric-sub {
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
  }
  .bullish  { color: #3fb950; }
  .bearish  { color: #f85149; }
  .neutral  { color: #8b949e; }
  .warning  { color: #d29922; }

  .stSelectbox label, .stMultiSelect label, .stSlider label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }
  [data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #8b949e;
  }
  .stTabs [aria-selected="true"] { color: #e6edf3; }
  div[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px;
  }
  .sig-badge {
    display: inline-block;
    background: #1f6feb22;
    border: 1px solid #1f6feb66;
    color: #58a6ff;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.68rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-left: 6px;
  }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
TOPIC_COLORS = {
    "tariffs":         "#f85149",
    "fed_rates":       "#58a6ff",
    "specific_equity": "#3fb950",
    "geopolitics":     "#d29922",
    "crypto":          "#f0883e",
    "energy":          "#a5d6ff",
}
HORIZONS = [1, 2, 5, 10, 15]
PLOTLY_TEMPLATE = "plotly_dark"
PAPER_BG  = "#080c10"
PLOT_BG   = "#0d1117"
GRID_COL  = "#21262d"
FONT_MONO = "IBM Plex Mono"


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    base = Path("data")
    results  = pd.read_csv(base / "event_study_results.csv",
                           parse_dates=["created_at"])
    summary  = pd.read_csv(base / "topic_summary.csv")
    micro    = pd.read_csv(base / "microstructure_summary.csv")
    posts    = pd.read_csv(base / "classified_posts.csv",
                           parse_dates=["created_at"])
    return results, summary, micro, posts


@st.cache_data
def load_depth_profiles(topic_filter: tuple):
    depth_dir = Path("data/depth_profiles")
    results   = pd.read_csv("data/event_study_results.csv")
    topic_ids = set(
        results[results["topic"].isin(topic_filter)]["post_id"].astype(str)
    )
    frames = []
    for f in depth_dir.glob("*.parquet"):
        if f.stem in topic_ids:
            try:
                df = pd.read_parquet(f)
                topic = results.loc[
                    results["post_id"].astype(str) == f.stem, "topic"
                ].values
                if len(topic):
                    df["topic"] = topic[0]
                    frames.append(df)
            except Exception:
                pass
    return pd.concat(frames) if frames else pd.DataFrame()


def fmt_sar(val):
    if pd.isna(val):
        return "—"
    cls = "bearish" if val < -0.1 else "bullish" if val > 0.1 else "neutral"
    return f'<span class="{cls}">{val:+.3f}</span>'


# ── Chart builders ────────────────────────────────────────────────────────────

def chart_sar_heatmap(summary, topics):
    sub = summary[summary["topic"].isin(topics)]
    pivot     = sub.pivot_table(index="topic", columns="horizon_min",
                                values="mean_sar",    aggfunc="first")
    sig_pivot = sub.pivot_table(index="topic", columns="horizon_min",
                                values="significant", aggfunc="first")

    z     = pivot.values.astype(float)
    xlabs = [f"{h}m" for h in pivot.columns]
    ylabs = list(pivot.index)

    text = []
    for i, topic in enumerate(ylabs):
        row = []
        for j, h in enumerate(pivot.columns):
            val = pivot.loc[topic, h] if topic in pivot.index else np.nan
            sig = sig_pivot.loc[topic, h] if topic in sig_pivot.index else False
            row.append(f"{'★ ' if sig else ''}{val:+.3f}" if not np.isnan(val) else "")
        text.append(row)

    lim = max(abs(np.nanmax(z)), abs(np.nanmin(z)), 0.3)

    fig = go.Figure(go.Heatmap(
        z=z, x=xlabs, y=ylabs, text=text,
        texttemplate="%{text}",
        textfont=dict(family=FONT_MONO, size=11),
        colorscale=[
            [0.0,  "#f85149"], [0.35, "#6e3630"],
            [0.5,  "#1c2128"],
            [0.65, "#1a3a2a"], [1.0,  "#3fb950"],
        ],
        zmin=-lim, zmax=lim,
        colorbar=dict(
            title=dict(text="SAR", font=dict(family=FONT_MONO, size=10)),
            tickfont=dict(family=FONT_MONO, size=9),
            thickness=12, len=0.8,
        ),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> @ %{x}<br>SAR: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Mean Standardised Abnormal Return (SAR) — ★ p<0.05",
                   font=dict(family=FONT_MONO, size=12)),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_MONO, color="#c9d1d9"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig


def chart_return_paths(results, topics):
    fig = go.Figure()
    for topic in topics:
        grp = results[results["topic"] == topic]
        means = [0.0]
        for h in HORIZONS:
            col = f"cumret_{h}m"
            means.append(grp[col].mean() * 100 if col in grp.columns else np.nan)

        fig.add_trace(go.Scatter(
            x=[0] + HORIZONS, y=means,
            mode="lines+markers",
            name=topic,
            line=dict(color=TOPIC_COLORS.get(topic, "#888"), width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{topic}</b><br>%{{x}}m: %{{y:.4f}}%<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color="#21262d", width=1, dash="dash"))
    fig.update_layout(
        title=dict(text="Average Cumulative Return Path (t=0 = post timestamp)",
                   font=dict(family=FONT_MONO, size=12)),
        xaxis=dict(title="Minutes after post", gridcolor=GRID_COL,
                   tickfont=dict(family=FONT_MONO, size=9)),
        yaxis=dict(title="Avg cumulative return (%)", gridcolor=GRID_COL,
                   tickfont=dict(family=FONT_MONO, size=9),
                   tickformat=".3f"),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_MONO, color="#c9d1d9"),
        legend=dict(bgcolor="#0d1117", bordercolor="#21262d", borderwidth=1,
                    font=dict(family=FONT_MONO, size=10)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=340,
    )
    return fig


def chart_volatility(results, topics):
    fig = go.Figure()
    topic_list, rv_pre_vals, rv_post_vals, rv_ratio_vals = [], [], [], []

    for topic in topics:
        grp = results[results["topic"] == topic]
        topic_list.append(topic)
        rv_pre_vals.append(grp["rv_pre"].mean() * 100)
        rv_post_vals.append(grp["rv_post"].mean() * 100)
        rv_ratio_vals.append(grp["rv_ratio"].mean())

    fig.add_trace(go.Bar(
        name="Pre-post vol",
        x=topic_list, y=rv_pre_vals,
        marker_color=[f"rgba({int(TOPIC_COLORS.get(t,'#888888')[1:3],16)},"
              f"{int(TOPIC_COLORS.get(t,'#888888')[3:5],16)},"
              f"{int(TOPIC_COLORS.get(t,'#888888')[5:7],16)},0.33)"
              for t in topic_list],
        marker_line_color=[TOPIC_COLORS.get(t, "#888") for t in topic_list],
        marker_line_width=1.5,
        hovertemplate="<b>%{x}</b><br>Pre vol: %{y:.5f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Post-post vol",
        x=topic_list, y=rv_post_vals,
        marker_color=[TOPIC_COLORS.get(t, "#888") for t in topic_list],
        hovertemplate="<b>%{x}</b><br>Post vol: %{y:.5f}%<extra></extra>",
    ))

    # rv_ratio line on secondary axis
    fig.add_trace(go.Scatter(
        name="Vol ratio (post/pre)",
        x=topic_list, y=rv_ratio_vals,
        mode="lines+markers",
        marker=dict(size=8, symbol="diamond",
                    color=[TOPIC_COLORS.get(t, "#888") for t in topic_list]),
        line=dict(color="#e6edf3", width=1.5, dash="dot"),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>rv_ratio: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=1.0, line=dict(color="#8b949e", width=1, dash="dash"),
                  yref="y2", annotation_text="ratio=1",
                  annotation_font=dict(family=FONT_MONO, size=9, color="#8b949e"))

    fig.update_layout(
        title=dict(text="Realized Volatility Pre vs Post — ratio > 1 = vol spike",
                   font=dict(family=FONT_MONO, size=12)),
        barmode="group",
        xaxis=dict(gridcolor=GRID_COL, tickfont=dict(family=FONT_MONO, size=9)),
        yaxis=dict(title="Vol (per-min, %)", gridcolor=GRID_COL,
                   tickfont=dict(family=FONT_MONO, size=9)),
        yaxis2=dict(title="rv_ratio", overlaying="y", side="right",
                    showgrid=False, tickfont=dict(family=FONT_MONO, size=9),
                    range=[0.5, max(rv_ratio_vals) * 1.3 if rv_ratio_vals else 2]),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_MONO, color="#c9d1d9"),
        legend=dict(bgcolor="#0d1117", bordercolor="#21262d", borderwidth=1,
                    font=dict(family=FONT_MONO, size=10)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=340,
    )
    return fig


def chart_depth_depletion(results, topics):
    """Average depth % of baseline over time, per topic."""
    fig = go.Figure()

    depth_dir = Path("data/depth_profiles")
    if not depth_dir.exists():
        fig.add_annotation(text="No depth profiles found — re-run step 4",
                           showarrow=False, font=dict(color="#8b949e"))
        return fig

    for topic in topics:
        topic_ids = set(
            results[results["topic"] == topic]["post_id"].astype(str)
        )
        all_profiles = []

        for pid in topic_ids:
            f = depth_dir / f"{pid}.parquet"
            if not f.exists():
                continue
            try:
                df = pd.read_parquet(f)
                # Normalise depth to % of pre-event baseline [-10, -1 min]
                baseline = df[
                    (df["minutes_from_post"] >= -10) &
                    (df["minutes_from_post"] <= -1)
                ]["total_depth"].mean()
                if baseline > 0:
                    df["depth_pct"] = df["total_depth"] / baseline * 100
                    all_profiles.append(df[["minutes_from_post", "depth_pct"]])
            except Exception:
                continue

        if not all_profiles:
            continue

        combined = pd.concat(all_profiles)
        # Bin to 15-second intervals and average across events
        combined["min_bin"] = (combined["minutes_from_post"] * 4).round() / 4
        avg = combined.groupby("min_bin")["depth_pct"].mean().reset_index()
        avg = avg[(avg["min_bin"] >= -10) & (avg["min_bin"] <= 15)]

        fig.add_trace(go.Scatter(
            x=avg["min_bin"], y=avg["depth_pct"],
            mode="lines",
            name=topic,
            line=dict(color=TOPIC_COLORS.get(topic, "#888"), width=2),
            hovertemplate=f"<b>{topic}</b><br>%{{x:.2f}}m: %{{y:.1f}}% of baseline<extra></extra>",
        ))

    fig.add_vline(x=0, line=dict(color="#8b949e", width=1.5, dash="dash"),
                  annotation_text="post time",
                  annotation_font=dict(family=FONT_MONO, size=9, color="#8b949e"))
    fig.add_hline(y=100, line=dict(color="#21262d", width=1))
    fig.add_hline(y=80, line=dict(color="#d29922", width=1, dash="dot"),
                  annotation_text="80% recovery threshold",
                  annotation_font=dict(family=FONT_MONO, size=9, color="#d29922"))

    fig.update_layout(
        title=dict(text="Order Book Depth — % of Pre-Event Baseline (15s bins)",
                   font=dict(family=FONT_MONO, size=12)),
        xaxis=dict(title="Minutes from post", gridcolor=GRID_COL,
                   tickfont=dict(family=FONT_MONO, size=9)),
        yaxis=dict(title="Depth (% of baseline)", gridcolor=GRID_COL,
                   tickfont=dict(family=FONT_MONO, size=9)),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_MONO, color="#c9d1d9"),
        legend=dict(bgcolor="#0d1117", bordercolor="#21262d", borderwidth=1,
                    font=dict(family=FONT_MONO, size=10)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=360,
    )
    return fig


def chart_hit_rate(summary, topics):
    s5 = summary[
        (summary["horizon_min"] == 5) &
        (summary["topic"].isin(topics))
    ].dropna(subset=["hit_rate"]).sort_values("hit_rate")

    fig = go.Figure(go.Bar(
        x=s5["hit_rate"] * 100,
        y=s5["topic"],
        orientation="h",
        marker_color=[TOPIC_COLORS.get(t, "#888") for t in s5["topic"]],
        text=[f"{v:.0f}%" for v in s5["hit_rate"] * 100],
        textposition="outside",
        textfont=dict(family=FONT_MONO, size=10),
        hovertemplate="<b>%{y}</b><br>Hit rate: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=50, line=dict(color="#8b949e", width=1.5, dash="dash"),
                  annotation_text="50% (random)",
                  annotation_font=dict(family=FONT_MONO, size=9, color="#8b949e"))

    fig.update_layout(
        title=dict(text="Sentiment Alignment @ 5m  (>50% = post direction predicts return)",
                   font=dict(family=FONT_MONO, size=12)),
        xaxis=dict(title="% events where return matches sentiment",
                   range=[0, 110], gridcolor=GRID_COL,
                   tickfont=dict(family=FONT_MONO, size=9)),
        yaxis=dict(gridcolor=GRID_COL, tickfont=dict(family=FONT_MONO, size=9)),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_MONO, color="#c9d1d9"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=280,
    )
    return fig


def chart_event_count(results, topics):
    counts = (results[results["topic"].isin(topics)]
              ["topic"].value_counts().reset_index())
    counts.columns = ["topic", "n_events_analysed"]
    counts = counts.sort_values("n_events_analysed")

    fig = go.Figure(go.Bar(
        x=counts["n_events_analysed"],
        y=counts["topic"],
        orientation="h",
        marker_color=[TOPIC_COLORS.get(t, "#888") for t in counts["topic"]],
        text=counts["n_events_analysed"],
        textposition="outside",
        textfont=dict(family=FONT_MONO, size=10),
        hovertemplate="<b>%{y}</b><br>Events analysed: %{x}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Events Analysed per Topic  (market hours + Databento window)",
                   font=dict(family=FONT_MONO, size=12)),
        xaxis=dict(gridcolor=GRID_COL, tickfont=dict(family=FONT_MONO, size=9)),
        yaxis=dict(gridcolor=GRID_COL, tickfont=dict(family=FONT_MONO, size=9)),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_MONO, color="#c9d1d9"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=280,
    )
    return fig


def chart_spread_change(micro, topics):
    sub = micro[micro["topic"].isin(topics)].copy()
    sub = sub.sort_values("spread_change_bps_mean")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sub["topic"],
        y=sub["spread_change_bps_mean"],
        marker_color=[
            "#f85149" if v > 0 else "#3fb950"
            for v in sub["spread_change_bps_mean"]
        ],
        text=[f"{v:+.2f}bps" for v in sub["spread_change_bps_mean"]],
        textposition="outside",
        textfont=dict(family=FONT_MONO, size=10),
        hovertemplate="<b>%{x}</b><br>Spread Δ: %{y:+.3f}bps<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#21262d", width=1))
    fig.update_layout(
        title=dict(text="Bid-Ask Spread Change Post-Post (bps)  +ve = widening = uncertainty",
                   font=dict(family=FONT_MONO, size=12)),
        xaxis=dict(gridcolor=GRID_COL, tickfont=dict(family=FONT_MONO, size=9)),
        yaxis=dict(title="Spread change (bps)", gridcolor=GRID_COL,
                   tickfont=dict(family=FONT_MONO, size=9)),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_MONO, color="#c9d1d9"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
    )
    return fig


def chart_event_scatter(results, topics, horizon=5):
    sub = results[results["topic"].isin(topics)].copy()
    col = f"sar_{horizon}m"
    sub = sub[sub[col].notna()].copy()

    fig = go.Figure()
    for topic in topics:
        grp = sub[sub["topic"] == topic]
        fig.add_trace(go.Scatter(
            x=grp["created_at"],
            y=grp[col],
            mode="markers",
            name=topic,
            marker=dict(
                color=TOPIC_COLORS.get(topic, "#888"),
                size=7, opacity=0.75,
                line=dict(width=0.5, color="#0d1117"),
            ),
            hovertemplate=(
                f"<b>{topic}</b><br>"
                "Date: %{x|%Y-%m-%d %H:%M}<br>"
                f"SAR @{horizon}m: %{{y:+.3f}}<extra></extra>"
            ),
        ))

    fig.add_hline(y=0,  line=dict(color="#21262d", width=1))
    fig.add_hline(y=2,  line=dict(color="#3fb950", width=0.8, dash="dot"))
    fig.add_hline(y=-2, line=dict(color="#f85149", width=0.8, dash="dot"))

    fig.update_layout(
        title=dict(text=f"Individual Event SAR @ {horizon}m  (dashed lines = ±2σ threshold)",
                   font=dict(family=FONT_MONO, size=12)),
        xaxis=dict(title="Post date", gridcolor=GRID_COL,
                   tickfont=dict(family=FONT_MONO, size=9)),
        yaxis=dict(title=f"SAR @ {horizon}m", gridcolor=GRID_COL,
                   tickfont=dict(family=FONT_MONO, size=9)),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_MONO, color="#c9d1d9"),
        legend=dict(bgcolor="#0d1117", bordercolor="#21262d", borderwidth=1,
                    font=dict(family=FONT_MONO, size=10)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=360,
    )
    return fig


# ── App layout ────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <h1>📊 Trump Truth Social → SPY Market Impact</h1>
    <p style="font-family:'IBM Plex Mono',monospace; color:#6e7681; font-size:0.8rem;
              margin-top:-8px; margin-bottom:20px;">
      MBP-10 order book · standardised abnormal returns · realized volatility · depth depletion
    </p>
    """, unsafe_allow_html=True)

    # Load data
    try:
        results, summary, micro, posts = load_data()
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}\nRun steps 1–4 first.")
        st.stop()

    all_topics = sorted(results["topic"].unique().tolist())

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Filters")

        selected_topics = st.multiselect(
            "Topics",
            options=all_topics,
            default=all_topics,
        )

        min_conf = st.slider(
            "Min confidence threshold",
            min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        )

        scatter_horizon = st.select_slider(
            "Scatter plot horizon",
            options=HORIZONS, value=5,
        )

        st.markdown("---")
        st.markdown("### Dataset")
        total_events = len(results[results["topic"].isin(selected_topics)])
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{total_events}</div>
          <div class="metric-label">Events analysed</div>
          <div class="metric-sub">market hours · Databento matched</div>
        </div>
        """, unsafe_allow_html=True)

        date_range = results["created_at"]
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="font-size:1rem">
            {pd.to_datetime(date_range.min()).strftime('%b %d %Y')}<br>
            → {pd.to_datetime(date_range.max()).strftime('%b %d %Y')}
          </div>
          <div class="metric-label">Date range</div>
        </div>
        """, unsafe_allow_html=True)

    if not selected_topics:
        st.warning("Select at least one topic in the sidebar.")
        st.stop()

    # Filter results
    filtered = results[
        (results["topic"].isin(selected_topics)) &
        (results["confidence"] >= min_conf)
    ].copy()
    filtered_summary = summary[summary["topic"].isin(selected_topics)]
    filtered_micro   = micro[micro["topic"].isin(selected_topics)]

    # ── Top metric cards ──────────────────────────────────────────────────────
    cols = st.columns(5)
    metric_defs = [
        ("Strongest signal",
         filtered_summary.loc[filtered_summary["mean_sar"].abs().idxmax(), "topic"]
         if not filtered_summary.empty else "—",
         "topic by mean |SAR|", None),
        ("Best hit rate",
         f"{filtered_summary[filtered_summary['horizon_min']==5]['hit_rate'].max()*100:.0f}%"
         if not filtered_summary.empty else "—",
         "sentiment alignment @ 5m", None),
        ("Tariff SAR @5m",
         f"{filtered_summary[(filtered_summary['topic']=='tariffs') & (filtered_summary['horizon_min']==5)]['mean_sar'].values[0]:+.3f}"
         if 'tariffs' in selected_topics and not filtered_summary.empty else "—",
         "bearish drift signal", "bearish"),
        ("Energy rv_ratio",
         f"{filtered[filtered['topic']=='energy']['rv_ratio'].mean():+.3f}"
         if 'energy' in selected_topics and not filtered.empty else "—",
         "highest vol spike topic", "warning"),
        ("Depth min (tariffs)",
         f"{filtered[filtered['topic']=='tariffs']['depth_min_pct'].mean():.1f}%"
         if 'tariffs' in selected_topics and not filtered.empty else "—",
         "of baseline — most depletion", "bearish"),
    ]
    for col, (label, value, sub, cls) in zip(cols, metric_defs):
        cls_str = f' class="{cls}"' if cls else ""
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value"{cls_str}>{value}</div>
          <div class="metric-label">{label}</div>
          <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈  Returns & SAR",
        "🌊  Volatility",
        "📚  Depth Depletion",
        "🔍  Event Explorer",
    ])

    with tab1:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(chart_sar_heatmap(filtered_summary, selected_topics),
                            use_container_width=True)
        with c2:
            st.plotly_chart(chart_hit_rate(filtered_summary, selected_topics),
                            use_container_width=True)

        st.plotly_chart(chart_return_paths(filtered, selected_topics),
                        use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(chart_event_count(filtered, selected_topics),
                            use_container_width=True)
        with c4:
            st.plotly_chart(chart_spread_change(filtered_micro, selected_topics),
                            use_container_width=True)

    with tab2:
        st.plotly_chart(chart_volatility(filtered, selected_topics),
                        use_container_width=True)

        st.markdown("#### Realized Volatility by Topic — Summary Table")
        vol_tbl = (
            filtered.groupby("topic")
            .agg(
                n_events=("rv_ratio", "count"),
                rv_pre_mean=("rv_pre", "mean"),
                rv_post_mean=("rv_post", "mean"),
                rv_ratio_mean=("rv_ratio", "mean"),
                rv_ratio_median=("rv_ratio", "median"),
            )
            .reset_index()
        )
        vol_tbl = vol_tbl[vol_tbl["topic"].isin(selected_topics)]
        for col in ["rv_pre_mean", "rv_post_mean"]:
            vol_tbl[col] = (vol_tbl[col] * 100).map("{:.6f}%".format)
        vol_tbl["rv_ratio_mean"]   = vol_tbl["rv_ratio_mean"].map("{:+.3f}".format)
        vol_tbl["rv_ratio_median"] = vol_tbl["rv_ratio_median"].map("{:+.3f}".format)
        st.dataframe(
            vol_tbl.set_index("topic"),
            use_container_width=True,
        )

    with tab3:
        st.plotly_chart(chart_depth_depletion(filtered, selected_topics),
                        use_container_width=True)

        st.markdown("#### Depth Depletion Summary")
        depth_tbl = (
            filtered[filtered["topic"].isin(selected_topics)]
            .groupby("topic")
            .agg(
                n_events=("depth_min_pct", "count"),
                depth_min_pct=("depth_min_pct", "mean"),
                depth_at_5m_pct=("depth_at_5m_pct", "mean"),
                depth_recovery_min=("depth_recovery_min", "mean"),
            )
            .reset_index()
        )
        for col in ["depth_min_pct", "depth_at_5m_pct"]:
            depth_tbl[col] = depth_tbl[col].map("{:.1f}%".format)
        depth_tbl["depth_recovery_min"] = depth_tbl["depth_recovery_min"].map("{:.2f} min".format)
        st.dataframe(depth_tbl.set_index("topic"), use_container_width=True)

        st.caption(
            "depth_min_pct: lowest depth reached as % of pre-event baseline  ·  "
            "recovery threshold: 80% of baseline  ·  "
            "profiles resampled to 15-second bins"
        )

    with tab4:
        st.plotly_chart(
            chart_event_scatter(filtered, selected_topics, scatter_horizon),
            use_container_width=True,
        )

        st.markdown("#### Individual Events")
        cols_to_show = [
            "created_at", "topic", "sentiment", "confidence",
            f"sar_{scatter_horizon}m", f"cumret_{scatter_horizon}m",
            "rv_ratio", "depth_min_pct", "price_impact_bps",
        ]
        cols_to_show = [c for c in cols_to_show if c in filtered.columns]
        display_df = filtered[cols_to_show].copy()
        display_df["created_at"] = pd.to_datetime(
            display_df["created_at"]
        ).dt.strftime("%Y-%m-%d %H:%M")

        for col in [f"sar_{scatter_horizon}m", f"cumret_{scatter_horizon}m",
                    "rv_ratio", "depth_min_pct", "price_impact_bps"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(
                    lambda x: f"{x:+.4f}" if pd.notna(x) else "—"
                )

        st.dataframe(
            display_df.sort_values("created_at", ascending=False)
                      .reset_index(drop=True),
            use_container_width=True,
            height=400,
        )


if __name__ == "__main__":
    main()
