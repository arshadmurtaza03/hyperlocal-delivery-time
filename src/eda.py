# src/eda.py
"""
EDA script for Hyperlocal Delivery Time Predictor.
Generates summary statistics, several plots, saves them to reports/figures,
and writes a short Markdown report at reports/EDA_report.md.
Run: python src/eda.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import argparse
import logging
# Use Agg backend for servers without display
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

DATA_PATH = "data/delivery_data.csv"
FIG_DIR = "reports/figures"
REPORT_MD = "reports/EDA_report.md"
SUMMARY_CSV = "reports/summary_stats.csv"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)

# basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def _safe_save(fig, path):
    try:
        save_fig(fig, path)
        logging.info(f"Saved: {path}")
    except Exception:
        logging.exception(f"Failed to save figure {path}")

def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run data generation first.")

    df = pd.read_csv(DATA_PATH)
    # Basic cleaning: dropna for EDA
    df = df.dropna().reset_index(drop=True)

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 1) Summary stats
    summary = df.describe(include="all")
    summary.to_csv(SUMMARY_CSV)
    print(f"Saved summary to {SUMMARY_CSV}")

    # 2) Histogram: delivery_time_min
    if "delivery_time_min" in df.columns:
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(1,1,1)
        df["delivery_time_min"].plot.hist(bins=50, ax=ax)
        ax.set_title("Distribution: delivery_time_min")
        ax.set_xlabel("Delivery time (minutes)")
        _safe_save(fig, os.path.join(FIG_DIR, "hist_delivery_time.png"))
    else:
        logging.warning("Column delivery_time_min not found; skipping its histogram.")

    # 3) Histogram: distance_km
    if "distance_km" in df.columns:
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(1,1,1)
        df["distance_km"].plot.hist(bins=40, ax=ax)
        ax.set_title("Distribution: distance_km")
        ax.set_xlabel("Distance (km)")
        _safe_save(fig, os.path.join(FIG_DIR, "hist_distance.png"))
    else:
        logging.warning("Column distance_km not found; skipping its histogram.")

    # 4) Scatter: distance vs delivery_time
    if all(c in df.columns for c in ("distance_km", "delivery_time_min")):
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(1,1,1)
        ax.scatter(df["distance_km"], df["delivery_time_min"], alpha=0.5)
        ax.set_title("Distance vs Delivery Time")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Delivery time (min)")
        _safe_save(fig, os.path.join(FIG_DIR, "scatter_distance_vs_time.png"))
    else:
        logging.warning("Columns distance_km or delivery_time_min missing; skipping scatter plot.")

    # 5) Boxplot: delivery_time by traffic level
    if all(c in df.columns for c in ("delivery_time_min", "traffic")):
        fig = plt.figure(figsize=(7,4))
        ax = fig.add_subplot(1,1,1)
        # Ensure traffic is categorical with a consistent order
        order = ["low", "moderate", "heavy", "very_heavy"]
        df["traffic"] = pd.Categorical(df["traffic"], categories=order, ordered=True)
        df.boxplot(column="delivery_time_min", by="traffic", ax=ax)
        ax.set_title("Delivery time by Traffic level")
        ax.set_xlabel("Traffic")
        ax.set_ylabel("Delivery time (min)")
        # pandas' boxplot adds a super title we remove:
        plt.suptitle("")
        _safe_save(fig, os.path.join(FIG_DIR, "boxplot_time_by_traffic.png"))
    else:
        logging.warning("Columns delivery_time_min or traffic missing; skipping boxplot.")

    # 6) Correlation matrix of numeric features
    corr = df[numeric_cols].corr() if numeric_cols else pd.DataFrame()
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(1,1,1)
    cax = ax.imshow(corr, interpolation="nearest")
    fig.colorbar(cax)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    ax.set_title("Correlation matrix (numeric features)")
    # annotate correlation values
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    _safe_save(fig, os.path.join(FIG_DIR, "corr_matrix.png"))

    # 7) Scatter matrix (pairplot-like) for selected numeric features
    selected = ["distance_km", "num_stops", "delivery_experience_years", "delivery_time_min"]
    # Guard in case dataset is small
    sel = [c for c in selected if c in df.columns]
    if sel:
        fig = scatter_matrix(df[sel], figsize=(9,9), diagonal="hist")
        # scatter_matrix returns array of axes; save the figure that contains them
        _safe_save(plt.gcf(), os.path.join(FIG_DIR, "scatter_matrix.png"))
    else:
        logging.warning("Not enough numeric columns for scatter matrix; skipping.")

    # 8) Save a short Markdown report summarizing key points (auto)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# EDA Report — Hyperlocal Delivery Time Predictor\n\n")
        f.write("Auto-generated summary and figures.\n\n")
        f.write("## Summary statistics\n\n")
        f.write(f"- Summary CSV: `{SUMMARY_CSV}`\n\n")
        f.write("## Figures\n\n")
        f.write("1. Delivery time distribution: `reports/figures/hist_delivery_time.png`\n")
        f.write("2. Distance distribution: `reports/figures/hist_distance.png`\n")
        f.write("3. Distance vs delivery time scatter: `reports/figures/scatter_distance_vs_time.png`\n")
        f.write("4. Delivery time by traffic boxplot: `reports/figures/boxplot_time_by_traffic.png`\n")
        f.write("5. Correlation matrix: `reports/figures/corr_matrix.png`\n")
        f.write("6. Scatter matrix (pairwise): `reports/figures/scatter_matrix.png`\n\n")
        # Add a tiny automatic observation based on correlation
        if "delivery_time_min" in corr.columns:
            # show top correlated feature (abs corr except self)
            s = corr["delivery_time_min"].abs().drop("delivery_time_min", errors="ignore")
            if not s.empty:
                top = s.idxmax()
                val = s.max()
                f.write(f"## Quick note\n\n- The feature most correlated (absolute) with delivery_time_min is `{top}` (|corr|={val:.2f}).\n")
        f.write("\n---\n")
        f.write("Generated by `src/eda.py`.\n")
    print(f"Wrote EDA report to {REPORT_MD}")

if __name__ == "__main__":
    main()
