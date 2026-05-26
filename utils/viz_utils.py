"""
Visualization Utility Functions
All Plotly / Matplotlib / Seaborn chart generators used across the Streamlit pages.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — required for Streamlit
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import streamlit as st


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic Theme Fetcher
# ──────────────────────────────────────────────────────────────────────────────
def _get_theme_config() -> dict:
    """Retrieve theme-specific plotting variables if running inside Streamlit."""
    try:
        # Check if we are running in Streamlit and have a theme_config in session state
        if hasattr(st, "session_state") and st.session_state and "theme_config" in st.session_state:
            return st.session_state["theme_config"]
    except Exception:
        pass
    
    # Default fallback: Royal Vineyard (Burgundy Dark)
    return {
        "plotly_template": "plotly_dark",
        "plotly_colors": ["#8B0000", "#C0392B", "#E74C3C", "#F39C12", "#F1C40F"],
        "text_color": "#E8D5B7",
        "bg_color": "rgba(0,0,0,0)",
        "heatmap_cmap": "RdYlGn"
    }


def _base_layout(fig, title: str = ""):
    """Apply a shared theme-specific layout to any Plotly figure."""
    config = _get_theme_config()
    text_color = config["text_color"]
    bg_color = config["bg_color"]
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=text_color)),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 1. Correlation Heatmap
# ──────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    """Seaborn correlation heatmap. Returns a Matplotlib figure."""
    config = _get_theme_config()
    text_color = config["text_color"]
    is_light = (text_color == "#2C2C2C")
    fig_bg = "#FAF7F0" if is_light else "#0E1117"
    
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(fig_bg)
    ax.set_facecolor(fig_bg)

    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))   # hide upper triangle

    cmap = config.get("heatmap_cmap", "RdYlGn")

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        linewidths=0.5,
        linecolor="#EAE3D2" if is_light else "#1F2937",
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7, "color": "white" if not is_light else "black"},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, color=text_color, pad=15)
    ax.tick_params(colors=text_color, labelsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 2. Feature Importance Bar Chart
# ──────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(feature_importance: dict):
    """Horizontal bar chart for Random Forest feature importances. Returns Plotly fig."""
    fi_df = (
        pd.DataFrame.from_dict(feature_importance, orient="index", columns=["importance"])
        .sort_values("importance", ascending=True)
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    config = _get_theme_config()
    palette = config["plotly_colors"]
    template = config["plotly_template"]

    fig = px.bar(
        fi_df,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=palette,
        title="🌲 Feature Importance — Random Forest Classifier",
        labels={"importance": "Importance Score", "feature": "Feature"},
        template=template,
    )
    fig.update_coloraxes(showscale=False)
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>")
    _base_layout(fig)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3. Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray):
    """Annotated confusion matrix as a Plotly heatmap."""
    labels = ["Bad Wine", "Good Wine"]
    return ff_confusion_matrix(cm, labels)


def ff_confusion_matrix(cm: np.ndarray, labels: list):
    """Build a nice Plotly confusion matrix figure."""
    z      = cm[::-1]          # flip so (0,0) is top-left = True Negative
    x_labs = labels
    y_labs  = labels[::-1]

    config = _get_theme_config()
    text_color = config["text_color"]
    bg_color = config["bg_color"]
    
    is_light = (text_color == "#2C2C2C")
    if is_light:
        colorscale = [[0, "#FFF5F5"], [0.5, "#E6B8B8"], [1, "#8B0000"]]
        text_font_color = "black"
    else:
        colorscale = [[0, "#1A0000"], [0.5, "#8B0000"], [1, "#E74C3C"]]
        text_font_color = "white"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labs,
            y=y_labs,
            colorscale=colorscale,
            showscale=False,
            text=[[str(v) for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 20, "color": text_font_color},
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted Label", color=text_color),
        yaxis=dict(title="True Label",      color=text_color),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        margin=dict(l=40, r=40, t=60, b=40),
        height=380,
    )
    # Annotate quadrants
    ann_color = "#555" if is_light else "#aaa"
    annotations = [
        dict(x=0, y=1, text="TN (True Negative)", showarrow=False, font=dict(color=ann_color, size=9), yshift=25),
        dict(x=1, y=1, text="FP (False Positive)", showarrow=False, font=dict(color=ann_color, size=9), yshift=25),
        dict(x=0, y=0, text="FN (False Negative)", showarrow=False, font=dict(color=ann_color, size=9), yshift=25),
        dict(x=1, y=0, text="TP (True Positive)", showarrow=False, font=dict(color=ann_color, size=9), yshift=25),
    ]
    fig.update_layout(annotations=annotations)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4. Quality Distribution Bar Chart
# ──────────────────────────────────────────────────────────────────────────────
def plot_quality_distribution(df: pd.DataFrame):
    """Bar chart of wine quality score distribution."""
    counts = df["quality"].value_counts().sort_index().reset_index()
    counts.columns = ["Quality Score", "Count"]

    config = _get_theme_config()
    palette = config["plotly_colors"]
    template = config["plotly_template"]
    text_color = config["text_color"]

    fig = px.bar(
        counts,
        x="Quality Score",
        y="Count",
        color="Count",
        color_continuous_scale=palette,
        title="🍷 Wine Quality Score Distribution",
        template=template,
        text="Count",
    )
    fig.update_traces(textposition="outside", textfont_color=text_color)
    fig.update_coloraxes(showscale=False)
    _base_layout(fig)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5. Distribution Plots (histograms + KDE) for each feature
# ──────────────────────────────────────────────────────────────────────────────
def plot_feature_distributions(df: pd.DataFrame, features: list):
    """Grid of histogram + KDE overlay for each feature. Returns Matplotlib figure."""
    config = _get_theme_config()
    text_color = config["text_color"]
    is_light = (text_color == "#2C2C2C")
    fig_bg = "#FAF7F0" if is_light else "#0E1117"
    plot_bg = "#FFFBF2" if is_light else "#111827"
    border_color = "#D4CFC9" if is_light else "#374151"

    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
    fig.patch.set_facecolor(fig_bg)
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        ax.set_facecolor(plot_bg)
        # separate by label if column exists
        if "quality_label" in df.columns:
            bad_color = "#E74C3C"
            good_color = "#D4AF37" if is_light else "#F1C40F"
            for label, color, name in [(0, bad_color, "Bad/Avg"), (1, good_color, "Good")]:
                subset = df[df["quality_label"] == label][feat].dropna()
                ax.hist(subset, bins=25, alpha=0.5, color=color, label=name, density=True)
                subset.plot.kde(ax=ax, color=color, linewidth=1.5)
        else:
            ax.hist(df[feat].dropna(), bins=25, color="#8B0000", alpha=0.7, density=True)

        ax.set_title(feat.title(), color=text_color, fontsize=10)
        ax.tick_params(colors=text_color, labelsize=8)
        ax.spines["bottom"].set_color(border_color)
        ax.spines["left"].set_color(border_color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if "quality_label" in df.columns and i == 0:
            ax.legend(fontsize=8, labelcolor=text_color, facecolor=plot_bg)

    # Hide any unused axes
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Wine Quality", color=text_color, fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6. Prediction Confidence Gauge
# ──────────────────────────────────────────────────────────────────────────────
def plot_confidence_gauge(confidence: float, label: str):
    """Plotly gauge chart for prediction confidence."""
    config = _get_theme_config()
    text_color = config["text_color"]
    bg_color = config["bg_color"]
    is_light = (text_color == "#2C2C2C")
    
    good_color = "#B8860B" if is_light else "#F1C40F"
    bad_color = "#C0392B" if is_light else "#E74C3C"
    color = good_color if label == "Good Wine" else bad_color

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={"suffix": "%", "font": {"size": 28, "color": color}},
            title={"text": f"Confidence: {label}", "font": {"size": 16, "color": text_color}},
            gauge={
                "axis":       {"range": [0, 100], "tickcolor": text_color},
                "bar":        {"color": color},
                "bgcolor":    "#E6E1D8" if is_light else "#1F2937",
                "bordercolor": "#D4CFC9" if is_light else "#374151",
                "steps": [
                    {"range": [0,  50], "color": "#FADBD8" if is_light else "#3D0000"},
                    {"range": [50, 75], "color": "#F5B7B1" if is_light else "#6B0000"},
                    {"range": [75,100], "color": "#E6B8B8" if is_light else "#8B1A1A"},
                ],
                "threshold": {
                    "line": {"color": good_color, "width": 4},
                    "thickness": 0.75,
                    "value": 75,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 7. Accuracy Metrics Bar Chart
# ──────────────────────────────────────────────────────────────────────────────
def plot_metrics_bar(metrics: dict):
    """Bar chart comparing key model metrics."""
    report = metrics["classification_report"]
    data = {
        "Metric":    ["Accuracy", "ROC-AUC", "Precision (Good)", "Recall (Good)", "F1 (Good)"],
        "Score (%)": [
            metrics["accuracy"],
            metrics["roc_auc"],
            round(report["Good Wine (1)"]["precision"] * 100, 2),
            round(report["Good Wine (1)"]["recall"]    * 100, 2),
            round(report["Good Wine (1)"]["f1-score"]  * 100, 2),
        ],
    }
    
    config = _get_theme_config()
    palette = config["plotly_colors"]
    template = config["plotly_template"]
    text_color = config["text_color"]
    
    fig = px.bar(
        data,
        x="Metric",
        y="Score (%)",
        color="Score (%)",
        color_continuous_scale=palette,
        title="📊 Model Performance Metrics",
        template=template,
        text="Score (%)",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont_color=text_color)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(yaxis=dict(range=[0, 110]))
    _base_layout(fig)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 8. Box Plots: Feature vs Quality Label
# ──────────────────────────────────────────────────────────────────────────────
def plot_boxplots(df: pd.DataFrame, features: list):
    """Grid of box plots (Good vs Bad) for top features. Returns Plotly figure."""
    top_feats = features[:6]  # show up to 6
    n_cols    = 3
    n_rows    = int(np.ceil(len(top_feats) / n_cols))

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=top_feats)
    df_plot = df.copy()
    if "quality_label" not in df_plot.columns:
        df_plot["quality_label"] = (df_plot["quality"] >= 7).astype(int)

    config = _get_theme_config()
    text_color = config["text_color"]
    bg_color = config["bg_color"]
    is_light = (text_color == "#2C2C2C")
    
    good_color = "#B8860B" if is_light else "#F1C40F"
    bad_color = "#C0392B" if is_light else "#E74C3C"

    for idx, feat in enumerate(top_feats):
        row = idx // n_cols + 1
        col = idx %  n_cols + 1
        for label, name, color in [(0, "Bad Wine", bad_color), (1, "Good Wine", good_color)]:
            vals = df_plot[df_plot["quality_label"] == label][feat]
            fig.add_trace(
                go.Box(y=vals, name=name, marker_color=color, showlegend=(idx == 0)),
                row=row, col=col,
            )
    fig.update_layout(
        title_text="Feature Distribution: Good vs Bad Wine Comparison",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        height=480,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig
