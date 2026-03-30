#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
DERIVED_DIR = ROOT / "data" / "derived"
FIG_DIR = ROOT / "figures"

PALETTE = {
    "ink": "#173042",
    "muted": "#6b7b88",
    "grid": "#dbe2ea",
    "paper": "#ffffff",
    "panel": "#ffffff",
    "accept": "#17766f",
    "reject": "#c45b2d",
    "accent": "#2e5fd0",
    "accent_soft": "#8ea8e6",
    "sand": "#e4eefc",
    "before": "#9aa7bb",
    "after": "#17766f",
    "highlight": "#b8860b",
}


def blend_colors(color_a: str, color_b: str, weight: float) -> str:
    weight = float(np.clip(weight, 0.0, 1.0))
    rgb_a = np.asarray(to_rgb(color_a))
    rgb_b = np.asarray(to_rgb(color_b))
    blended = (1.0 - weight) * rgb_a + weight * rgb_b
    return to_hex(blended)


def score_bin_colors(bin_orders: pd.Series | np.ndarray) -> list[str]:
    colors = []
    for value in np.asarray(bin_orders, dtype=int):
        if value <= 2:
            colors.append(PALETTE["reject"])
        elif value <= 6:
            colors.append(PALETTE["accent_soft"])
        else:
            colors.append(PALETTE["accept"])
    return colors


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.titlesize": 10.5,
            "axes.labelsize": 8.7,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 8.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": PALETTE["panel"],
            "axes.edgecolor": PALETTE["muted"],
            "axes.labelcolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "grid.color": PALETTE["grid"],
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "figure.facecolor": PALETTE["paper"],
            "savefig.facecolor": PALETTE["paper"],
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DERIVED_DIR / f"{name}.csv")


def mean_difference_ci(df: pd.DataFrame, feature: str) -> tuple[float, float, float]:
    accepted = df.loc[df["accept"] == 1, feature].dropna().to_numpy()
    rejected = df.loc[df["accept"] == 0, feature].dropna().to_numpy()
    effect = float(accepted.mean() - rejected.mean())
    accept_var = float(np.var(accepted, ddof=1)) if len(accepted) > 1 else 0.0
    reject_var = float(np.var(rejected, ddof=1)) if len(rejected) > 1 else 0.0
    se = np.sqrt((accept_var / max(len(accepted), 1)) + (reject_var / max(len(rejected), 1)))
    return effect, effect - 1.96 * se, effect + 1.96 * se


def disagreement_gap_df(paper_df: pd.DataFrame) -> pd.DataFrame:
    work = paper_df.copy()
    work["disagreement_group"] = pd.qcut(
        work["score_std"].rank(method="first"),
        3,
        labels=["Low", "Medium", "High"],
    )
    rows: list[dict[str, object]] = []
    for feature, label in [
        ("mean_sentiment", "Sentiment"),
        ("mean_politeness", "Politeness"),
    ]:
        for group in ["Low", "Medium", "High"]:
            subset = work[work["disagreement_group"] == group]
            effect, ci_low, ci_high = mean_difference_ci(subset, feature)
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "disagreement_group": group,
                    "effect": effect,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def pooled_effect(data: pd.DataFrame) -> tuple[float, float, float]:
    std_err = (data["ci_high"] - data["effect"]) / 1.96
    weights = 1.0 / np.square(std_err)
    pooled = float(np.sum(weights * data["effect"]) / np.sum(weights))
    pooled_se = float(np.sqrt(1.0 / np.sum(weights)))
    return pooled, pooled - 1.96 * pooled_se, pooled + 1.96 * pooled_se


def plot_figure1_overview(paper_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.35, 3.9), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    by_year = (
        paper_df.groupby(["year", "accept"]).size().rename("n").reset_index()
        .pivot(index="year", columns="accept", values="n")
        .fillna(0)
        .rename(columns={0: "Rejected", 1: "Accepted"})
        .reset_index()
    )
    years = by_year["year"].astype(int).tolist()
    x = np.arange(len(years))
    ax_left.bar(
        x,
        by_year["Rejected"],
        width=0.74,
        color=PALETTE["reject"],
        alpha=0.86,
        label="Rejected papers",
    )
    ax_left.bar(
        x,
        by_year["Accepted"],
        width=0.74,
        bottom=by_year["Rejected"],
        color=PALETTE["accept"],
        alpha=0.88,
        label="Accepted papers",
    )
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(years)
    ax_left.set_ylabel("Papers")
    ax_left.set_xlabel("Conference year")
    ax_left.set_title("ICLR cohorts and acceptance volume", loc="left", pad=8, fontweight="bold")
    ax_left.grid(axis="y", alpha=0.45)
    ax_left.set_ylim(0, 1320)
    ax_left.set_yticks([0, 300, 600, 900, 1200])

    accept_rate = paper_df.groupby("year")["accept"].mean().reindex(years).to_numpy()
    ax_rate = ax_left.twinx()
    ax_rate.plot(
        x,
        accept_rate,
        color=PALETTE["accent"],
        marker="o",
        linewidth=2.2,
        label="Acceptance rate",
    )
    ax_rate.set_ylabel("Acceptance rate")
    ax_rate.set_ylim(0.30, 0.42)
    ax_rate.set_yticks([0.32, 0.36, 0.40])
    ax_rate.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:.0f}%"))

    legend_items = [
        Patch(facecolor=PALETTE["reject"], edgecolor="none", label="Rejected papers"),
        Patch(facecolor=PALETTE["accept"], edgecolor="none", label="Accepted papers"),
        Line2D([0], [0], color=PALETTE["accent"], marker="o", lw=2.2, label="Acceptance rate"),
    ]
    ax_left.legend(
        handles=legend_items,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.99),
        ncol=1,
        columnspacing=0.8,
        handletextpad=0.5,
        borderaxespad=0.0,
    )
    add_panel_label(ax_left, "A")

    feature_rows = []
    for feature, label in [
        ("mean_sentiment", "Sentiment"),
        ("mean_politeness", "Politeness"),
        ("mean_constructiveness", "Constructiveness"),
        ("mean_toxicity", "Toxicity"),
    ]:
        effect, ci_low, ci_high = mean_difference_ci(paper_df, feature)
        feature_rows.append({"label": label, "effect": effect, "ci_low": ci_low, "ci_high": ci_high})
    diff_df = pd.DataFrame(feature_rows)
    y = np.arange(len(diff_df))[::-1]
    ax_right.axvline(0, color=PALETTE["muted"], linestyle="--", linewidth=1.0)
    ax_right.hlines(y, diff_df["ci_low"], diff_df["ci_high"], color=PALETTE["ink"], linewidth=1.6)
    colors = [PALETTE["accent"], PALETTE["accept"], PALETTE["highlight"], PALETTE["before"]]
    ax_right.scatter(diff_df["effect"], y, s=40, color=colors, zorder=3)
    ax_right.set_yticks(y)
    ax_right.set_yticklabels(diff_df["label"])
    ax_right.set_xlabel("Accepted - rejected mean difference")
    ax_right.set_title("Paper-level language gaps", loc="left", pad=8, fontweight="bold")
    ax_right.grid(axis="x", alpha=0.45)
    x_upper = max(0.22, float(diff_df["ci_high"].max()) + 0.03)
    x_lower = min(-0.02, float(diff_df["ci_low"].min()) - 0.01)
    ax_right.set_xlim(x_lower, x_upper)
    ax_right.text(
        -0.26,
        1.04,
        "B",
        transform=ax_right.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=PALETTE["ink"],
    )

    save_figure(fig, "figure1_overview")


def plot_figure1_design(paper_df: pd.DataFrame, bridge_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.35, 3.6), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.05])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    bins = np.linspace(0, 1, 21)
    for decision, color, label in [
        (1, PALETTE["accept"], "Accepted"),
        (0, PALETTE["reject"], "Rejected"),
    ]:
        values = paper_df.loc[paper_df["accept"] == decision, "mean_score_percentile"]
        ax_left.hist(
            values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.18,
            linewidth=0.0,
            color=color,
        )
        ax_left.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=color,
            label=label,
        )
    ax_left.axvspan(0.35, 0.65, color=PALETTE["sand"], alpha=0.55, zorder=0)
    ax_left.set_xlabel("Within-year percentile of mean score")
    ax_left.set_ylabel("Density")
    ax_left.set_title("Decision overlap in score space", loc="left", pad=8, fontweight="bold")
    ax_left.legend(frameon=False, loc="upper left")
    ax_left.grid(axis="y", alpha=0.45)
    add_panel_label(ax_left, "A")

    deciles = bridge_df.copy().sort_values("bin_order")
    x = np.arange(len(deciles))
    zone_colors = score_bin_colors(deciles["bin_order"])
    ax_right.bar(
        x,
        deciles["n_papers"],
        width=0.78,
        color=zone_colors,
        edgecolor="#ffffff",
        linewidth=1.0,
    )
    ax_right.set_ylabel("Papers per decile")
    ax_right.set_xlabel("Within-year mean-score decile")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(deciles["score_bin"], rotation=35, ha="right")
    ax_right.set_title("Where the matched layer operates", loc="left", pad=8, fontweight="bold")
    ax_right.axvspan(3 - 0.5, 6 - 0.5, color=PALETTE["sand"], alpha=0.60, zorder=0)
    ax_right.set_ylim(0, float(deciles["n_papers"].max()) * 1.24)
    ax_right.grid(axis="y", alpha=0.35)

    ax_rate = ax_right.twinx()
    ax_rate.plot(x, deciles["acceptance_rate"], color=PALETTE["accent"], marker="o", linewidth=2.2)
    ax_rate.set_ylabel("Acceptance rate")
    ax_rate.set_ylim(0, 1.02)
    ax_right.legend(
        handles=[
            Patch(facecolor=PALETTE["accent_soft"], edgecolor="none", label="Papers"),
            Line2D([0], [0], color=PALETTE["accent"], marker="o", lw=2.2, label="Acceptance rate"),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.99),
        borderaxespad=0.0,
    )
    add_panel_label(ax_right, "B")

    save_figure(fig, "figure1_design_motivation")


def plot_figure2_main(paper_df: pd.DataFrame, paper_margins: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.35, 3.65), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 0.92])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    ax_left.plot(
        paper_margins["mean_sentiment"],
        paper_margins["predicted_probability"],
        color=PALETTE["accent"],
        linewidth=2.4,
    )
    q10 = float(paper_df["mean_sentiment"].quantile(0.10))
    q90 = float(paper_df["mean_sentiment"].quantile(0.90))
    low_row = paper_margins.iloc[(paper_margins["mean_sentiment"] - q10).abs().argmin()]
    high_row = paper_margins.iloc[(paper_margins["mean_sentiment"] - q90).abs().argmin()]
    x_pad = 0.05 * (q90 - q10 if q90 > q10 else 1.0)
    x_min = q10 - x_pad
    x_max = q90 + x_pad
    rug_y0 = float(paper_margins["predicted_probability"].min()) - 0.00015
    rug_y1 = rug_y0 + 0.00020
    support = np.quantile(paper_df["mean_sentiment"].dropna(), np.linspace(0.05, 0.95, 26))
    ax_left.vlines(support, rug_y0, rug_y1, color=PALETTE["grid"], linewidth=1.0, alpha=0.9)
    ax_left.axvline(q10, color=PALETTE["grid"], linestyle=":", linewidth=1.0)
    ax_left.axvline(q90, color=PALETTE["grid"], linestyle=":", linewidth=1.0)
    ax_left.scatter(
        [low_row["mean_sentiment"], high_row["mean_sentiment"]],
        [low_row["predicted_probability"], high_row["predicted_probability"]],
        color=[PALETTE["reject"], PALETTE["accept"]],
        s=34,
        zorder=3,
    )
    ax_left.set_title("Conditional acceptance is nearly flat", loc="left", pad=8, fontweight="bold")
    ax_left.set_xlabel("Paper-level mean sentiment")
    ax_left.set_ylabel("Predicted probability of acceptance")
    ax_left.set_ylim(
        float(paper_margins["predicted_probability"].min()) - 0.004,
        float(paper_margins["predicted_probability"].max()) + 0.004,
    )
    ax_left.set_xlim(x_min, x_max)
    ax_left.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:.1f}%"))
    ax_left.grid(axis="y", alpha=0.45)
    add_panel_label(ax_left, "A")

    rejected = paper_df.loc[paper_df["accept"] == 0, "mean_sentiment"].dropna().to_numpy()
    accepted = paper_df.loc[paper_df["accept"] == 1, "mean_sentiment"].dropna().to_numpy()
    violin = ax_right.violinplot(
        [rejected, accepted],
        positions=[1, 2],
        widths=0.8,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for idx, body in enumerate(violin["bodies"]):
        body.set_facecolor([PALETTE["reject"], PALETTE["accept"]][idx])
        body.set_edgecolor([PALETTE["reject"], PALETTE["accept"]][idx])
        body.set_alpha(0.34)
    violin["cmedians"].set_color(PALETTE["ink"])
    violin["cmedians"].set_linewidth(1.2)
    ax_right.scatter(
        [1, 2],
        [rejected.mean(), accepted.mean()],
        color=[PALETTE["reject"], PALETTE["accept"]],
        s=28,
        zorder=3,
    )
    ax_right.set_xticks([1, 2])
    ax_right.set_xticklabels(["Rejected", "Accepted"])
    ax_right.set_ylabel("Paper-level mean sentiment")
    ax_right.set_title("Raw sentiment still differs by decision", loc="left", pad=8, fontweight="bold")
    y_min = float(min(rejected.min(), accepted.min()))
    y_max = float(max(rejected.max(), accepted.max()))
    ax_right.set_ylim(y_min - 0.08, y_max + 0.16)
    ax_right.grid(axis="y", alpha=0.4)
    add_panel_label(ax_right, "B")

    save_figure(fig, "figure2_paper_fe_margins")


def plot_figure2_model_summary(paper_ame: pd.DataFrame, review_ame: pd.DataFrame) -> None:
    feature_map = {
        "sentiment": "Sentiment",
        "mean_sentiment": "Sentiment",
        "politeness": "Politeness",
        "mean_politeness": "Politeness",
        "constructiveness": "Constructiveness",
        "mean_constructiveness": "Constructiveness",
        "toxicity": "Toxicity",
        "mean_toxicity": "Toxicity",
    }
    paper_plot = (
        paper_ame[paper_ame["term"].isin(["mean_sentiment", "mean_politeness", "mean_constructiveness"])]
        .copy()
        .assign(model_label="Paper-level acceptance")
    )
    review_plot = (
        review_ame[review_ame["term"].isin(["sentiment", "politeness", "constructiveness"])]
        .copy()
        .assign(model_label="Review-level recommendation")
    )
    plot_df = pd.concat([paper_plot, review_plot], ignore_index=True)
    order = ["Sentiment", "Politeness", "Constructiveness"]
    plot_df["feature_label"] = plot_df["term"].map(feature_map)
    base_y = {label: val for label, val in zip(order, np.arange(len(order))[::-1])}
    offsets = {"Review-level recommendation": 0.13, "Paper-level acceptance": -0.13}
    colors = {
        "Review-level recommendation": PALETTE["accent"],
        "Paper-level acceptance": PALETTE["accept"],
    }

    fig, ax = plt.subplots(figsize=(3.8, 4.15), layout="constrained")
    ax.axvline(0, color=PALETTE["muted"], linestyle="--", linewidth=1.0)
    for _, row in plot_df.iterrows():
        y = base_y[row["feature_label"]] + offsets[row["model_label"]]
        ax.hlines(y, row["ci_low"], row["ci_high"], color=colors[row["model_label"]], linewidth=1.5)
        ax.scatter(row["ame"], y, color=colors[row["model_label"]], s=34, zorder=3)
    ax.set_yticks([base_y[label] for label in order])
    ax.set_yticklabels(order)
    ax.set_xlabel("Average marginal effect")
    ax.set_title("Language effects by model layer", loc="left", pad=8, fontweight="bold")
    ax.grid(axis="x", alpha=0.45)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=colors["Review-level recommendation"], lw=1.6, label="Review-level recommendation"),
            Line2D([0], [0], marker="o", color=colors["Paper-level acceptance"], lw=1.6, label="Paper-level acceptance"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.10),
    )
    save_figure(fig, "figure2_main_model")


def plot_figure3_matched(psm_effects: pd.DataFrame, balance_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.35, 3.6), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    label_map = {
        "Primary window 35th-65th percentile": "Primary 35-65 window",
        "Window sensitivity 40th-60th percentile": "Narrow 40-60 window",
        "Window sensitivity 30th-70th percentile": "Wide 30-70 window",
        "Treatment sensitivity: top vs bottom sentiment tercile": "Top vs bottom terciles",
    }
    effects = psm_effects.copy()
    effects["short_label"] = effects["specification"].map(label_map)
    order = [
        "Primary 35-65 window",
        "Narrow 40-60 window",
        "Wide 30-70 window",
        "Top vs bottom terciles",
    ]
    effects["short_label"] = pd.Categorical(effects["short_label"], categories=order, ordered=True)
    effects = effects.sort_values("short_label", ascending=False)
    y = np.arange(len(effects))
    ax_left.axvline(0, color=PALETTE["muted"], linewidth=1.0)
    ax_left.hlines(y, effects["ci_low"], effects["ci_high"], color=PALETTE["ink"], linewidth=1.5)
    colors = [PALETTE["highlight"] if "Primary" in label else PALETTE["accent"] for label in effects["short_label"]]
    ax_left.scatter(effects["att"], y, s=46, color=colors, zorder=3)
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(effects["short_label"])
    ax_left.set_xlabel("Matched acceptance contrast (ATT estimand)")
    ax_left.set_title("Matched borderline contrasts", loc="left", pad=8, fontweight="bold")
    ax_left.grid(axis="x", alpha=0.45)
    x_max = float(max(effects["ci_high"].max(), 0.06))
    x_min = float(min(effects["ci_low"].min(), -0.12))
    ax_left.set_xlim(x_min - 0.01, x_max + 0.01)
    add_panel_label(ax_left, "A")

    balance = balance_df.copy()
    balance["before_abs"] = balance["smd_before"].abs()
    balance["after_abs"] = balance["smd_after"].abs()
    balance = balance.sort_values("before_abs", ascending=True)
    y2 = np.arange(len(balance))
    for idx, row in enumerate(balance.itertuples(index=False)):
        ax_right.hlines(idx, row.after_abs, row.before_abs, color=PALETTE["grid"], linewidth=2.0)
        ax_right.scatter(row.before_abs, idx, color=PALETTE["before"], s=30, zorder=3)
        ax_right.scatter(row.after_abs, idx, color=PALETTE["after"], s=30, zorder=3)
    ax_right.axvline(0.10, color=PALETTE["reject"], linestyle="--", linewidth=1.2)
    ax_right.set_yticks(y2)
    ax_right.set_yticklabels(balance["label"])
    ax_right.set_xlabel("Absolute SMD")
    ax_right.set_title("Balance after matching", loc="center", pad=8, fontweight="bold")
    ax_right.grid(axis="x", alpha=0.45)
    ax_right.legend(
        handles=[
            Line2D([0], [0], marker="o", color=PALETTE["before"], lw=0, label="Before"),
            Line2D([0], [0], marker="o", color=PALETTE["after"], lw=0, label="After"),
        ],
        frameon=False,
        loc="lower right",
    )
    add_panel_label(ax_right, "B")

    save_figure(fig, "figure3_matched_effect_balance")


def draw_summary_diamond(ax: plt.Axes, center: float, y: float, ci_low: float, ci_high: float, color: str) -> None:
    height = 0.22
    diamond = Polygon(
        [(ci_low, y), (center, y + height), (ci_high, y), (center, y - height)],
        closed=True,
        facecolor=color,
        edgecolor=color,
        alpha=0.75,
    )
    ax.add_patch(diamond)


def plot_figure3_temporal(year_difference_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.35, 3.55), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    feature_specs = [
        ("mean_sentiment", "Sentiment gap"),
        ("mean_politeness", "Politeness gap"),
    ]

    for idx, (feature, title) in enumerate(feature_specs):
        ax = fig.add_subplot(gs[0, idx])
        data = year_difference_df[year_difference_df["feature"] == feature].copy().sort_values("year", ascending=False)
        pooled, ci_low, ci_high = pooled_effect(data)
        plot_years = data["year"].astype(int).astype(str).tolist() + ["Pooled"]
        y = np.arange(len(plot_years))[::-1]
        ax.axvline(0, color=PALETTE["muted"], linestyle="--", linewidth=1.0)
        ax.hlines(y[:-1], data["ci_low"], data["ci_high"], color=PALETTE["ink"], linewidth=1.5)
        ax.scatter(data["effect"], y[:-1], color=PALETTE["accent"], s=34, zorder=3)
        draw_summary_diamond(ax, pooled, y[-1], ci_low, ci_high, PALETTE["accept"])
        ax.set_yticks(y)
        ax.set_yticklabels(plot_years)
        ax.set_xlabel("Accepted - rejected mean difference")
        ax.set_title(title, loc="left", pad=8, fontweight="bold")
        ax.grid(axis="x", alpha=0.45)
        x_min = min(0.0, float(data["ci_low"].min()) - 0.01, ci_low - 0.01)
        x_max = max(float(data["ci_high"].max()) + 0.015, ci_high + 0.015)
        ax.set_xlim(x_min, x_max)
        add_panel_label(ax, "A" if idx == 0 else "B")

    save_figure(fig, "figure3_temporal_stability")


def plot_figure4_heterogeneity(hetero_year: pd.DataFrame, hetero_disagreement: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.35, 3.45), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    year_df = hetero_year.copy().sort_values("year", ascending=False)
    y = np.arange(len(year_df))[::-1]
    ax_left.axvline(0, color=PALETTE["muted"], linestyle="--", linewidth=1.0)
    ax_left.errorbar(
        year_df["ame"],
        y,
        xerr=[year_df["ame"] - year_df["ci_low"], year_df["ci_high"] - year_df["ame"]],
        fmt="o",
        color=PALETTE["accent"],
        ecolor=PALETTE["ink"],
        elinewidth=1.6,
        capsize=2.6,
        markersize=5.2,
        zorder=3,
    )
    ax_left.set_yticks(y)
    ax_left.set_yticklabels(year_df["year"].astype(int).astype(str))
    ax_left.set_xlabel("Average marginal effect of sentiment")
    ax_left.set_title("Year-specific conditional effects", loc="left", pad=8, fontweight="bold")
    ax_left.grid(axis="x", alpha=0.45)
    add_panel_label(ax_left, "A")

    disagreement_order = ["Low disagreement", "Medium disagreement", "High disagreement"]
    diss_df = hetero_disagreement.copy()
    diss_df["disagreement_group"] = pd.Categorical(
        diss_df["disagreement_group"],
        categories=disagreement_order,
        ordered=True,
    )
    diss_df = diss_df.sort_values("disagreement_group", ascending=False)
    y2 = np.arange(len(diss_df))[::-1]
    ax_right.axvline(0, color=PALETTE["muted"], linestyle="--", linewidth=1.0)
    ax_right.errorbar(
        diss_df["ame"],
        y2,
        xerr=[diss_df["ame"] - diss_df["ci_low"], diss_df["ci_high"] - diss_df["ame"]],
        fmt="o",
        color=PALETTE["accent"],
        ecolor=PALETTE["ink"],
        elinewidth=1.6,
        capsize=2.6,
        markersize=5.2,
        zorder=3,
    )
    ax_right.set_yticks(y2)
    ax_right.set_yticklabels(["Low", "Medium", "High"])
    ax_right.set_xlabel("Average marginal effect of sentiment")
    ax_right.set_title("Effects by reviewer disagreement", loc="left", pad=8, fontweight="bold")
    ax_right.grid(axis="x", alpha=0.45)
    add_panel_label(ax_right, "B")

    span = max(
        abs(float(year_df["ci_low"].min())),
        abs(float(year_df["ci_high"].max())),
        abs(float(diss_df["ci_low"].min())),
        abs(float(diss_df["ci_high"].max())),
        0.04,
    )
    span += 0.01
    ax_left.set_xlim(-span, span)
    ax_right.set_xlim(-span, span)

    save_figure(fig, "figure4_heterogeneity_ame")


def plot_figure4_practical_limits(paper_df: pd.DataFrame, prediction_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.35, 3.55), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    gap_df = disagreement_gap_df(paper_df)
    x = np.arange(3)
    for feature, color, marker in [
        ("Sentiment", PALETTE["accent"], "o"),
        ("Politeness", PALETTE["accept"], "s"),
    ]:
        data = gap_df[gap_df["label"] == feature].set_index("disagreement_group").loc[["Low", "Medium", "High"]].reset_index()
        ax_left.errorbar(
            x,
            data["effect"],
            yerr=[data["effect"] - data["ci_low"], data["ci_high"] - data["effect"]],
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=5.2,
            capsize=3,
            label=feature,
        )
    ax_left.axhline(0, color=PALETTE["muted"], linestyle="--", linewidth=1.0)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(["Low", "Medium", "High"])
    ax_left.set_xlabel("Reviewer disagreement tier")
    ax_left.set_ylabel("Accepted - rejected mean")
    ax_left.set_title("Descriptive gaps by disagreement", loc="left", pad=8, fontweight="bold")
    ax_left.legend(frameon=False, loc="upper right")
    ax_left.grid(axis="y", alpha=0.45)
    add_panel_label(ax_left, "A")

    pred = prediction_df.copy().sort_values("heldout_year")
    pred["delta_auc_x1k"] = pred["delta_auc"] * 1000
    pred["delta_brier_x1k"] = pred["delta_brier"] * 1000
    years = pred["heldout_year"].astype(int).astype(str).tolist()
    x2 = np.arange(len(pred))
    width = 0.34
    ax_right.axhline(0, color=PALETTE["muted"], linestyle="--", linewidth=1.0)
    ax_right.bar(x2 - width / 2, pred["delta_auc_x1k"], width=width, color=PALETTE["accent"], alpha=0.9, label=r"$\Delta$AUC")
    ax_right.bar(x2 + width / 2, pred["delta_brier_x1k"], width=width, color=PALETTE["reject"], alpha=0.78, label=r"$\Delta$Brier")
    ax_right.set_xticks(x2)
    ax_right.set_xticklabels(years)
    ax_right.set_xlabel("Held-out year")
    ax_right.set_ylabel("Change after adding language ($\\times 10^{-3}$)")
    ax_right.set_title("Held-out predictive gain is negligible", loc="left", pad=8, fontweight="bold")
    ax_right.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.02),
        ncol=2,
        borderaxespad=0.0,
    )
    ax_right.grid(axis="y", alpha=0.45)
    add_panel_label(ax_right, "B")

    save_figure(fig, "figure4_practical_limits")


def main() -> None:
    setup_style()
    paper_df = read_csv("paper_level_canonical")
    paper_ame = read_csv("paper_ame")
    review_ame = read_csv("review_ame")
    paper_margins = read_csv("paper_margins")
    year_difference_df = read_csv("year_difference_effects")
    hetero_year = read_csv("heterogeneity_year")
    hetero_disagreement = read_csv("heterogeneity_disagreement")
    prediction_df = read_csv("cross_year_prediction_diagnostics")
    bridge_df = read_csv("score_bin_bridge")
    psm_effects = read_csv("psm_effects")
    psm_balance = read_csv("psm_primary_balance")

    plot_figure1_overview(paper_df)
    plot_figure1_design(paper_df, bridge_df)
    plot_figure2_main(paper_df, paper_margins)
    plot_figure2_model_summary(paper_ame, review_ame)
    plot_figure3_matched(psm_effects, psm_balance)
    plot_figure3_temporal(year_difference_df)
    plot_figure4_heterogeneity(hetero_year, hetero_disagreement)
    plot_figure4_practical_limits(paper_df, prediction_df)


if __name__ == "__main__":
    main()
