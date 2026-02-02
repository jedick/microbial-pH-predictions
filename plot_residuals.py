#!/usr/bin/env python3
"""
Plot residual pH (y-axis) vs actual pH (x-axis) for HGB and HyenaDNA models.

Creates a Seaborn figure with side-by-side panels for traditional ML (HGB) and
deep learning (HyenaDNA). Points are colored by environment from the Hugging
Face dataset.

Usage:
  python plot_residuals.py
  python plot_residuals.py --hgb-csv results/sklearn/hgb_test_predictions.csv \\
      --hyenadna-csv results/hyenadna/test_predictions.csv --output-dir results/figures

Requires: matplotlib, seaborn, pandas, datasets (Hugging Face)
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Legend order and colors for environments (publication style)
ENV_ORDER = [
    "River & Seawater",
    "Lake & Pond",
    "Groundwater",
    "Geothermal",
    "Hyperalkaline",
    "Sediment",
    "Soil",
]
ENV_COLORS = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
]
ENV_COLOR_MAP = {e: ENV_COLORS[i] for i, e in enumerate(ENV_ORDER)}


def load_environment_from_hf(dataset_repo: str) -> pd.DataFrame:
    """Load study_name, sample_id, environment from Hugging Face dataset."""
    dataset = load_dataset(dataset_repo, split="train")
    rows = []
    for item in dataset:
        if item.get("pH") is not None:
            rows.append(
                {
                    "study_name": item["study_name"],
                    "sample_id": item["sample_id"],
                    "environment": item["environment"],
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["sample_id"])


def load_hgb_data(path: Path) -> pd.DataFrame:
    """Load HGB predictions CSV. Columns: sample_id, study_name, true_pH, predicted_pH, residual."""
    df = pd.read_csv(path)
    df = df.rename(columns={"true_pH": "actual_pH"})
    df["model"] = "HGB"
    return df


def load_hyenadna_data(path: Path) -> pd.DataFrame:
    """Load HyenaDNA predictions CSV. Uses true_ph, predicted_ph_mean, residual_mean."""
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "true_ph": "actual_pH",
            "residual_mean": "residual",
        }
    )
    df["model"] = "HyenaDNA"
    return df


def plot_seaborn(
    hgb_df: pd.DataFrame, hyenadna_df: pd.DataFrame, out_dir: Path
) -> None:
    """Side-by-side residual vs actual pH with Seaborn, colored by environment."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    all_envs = set(hgb_df["environment"].tolist()) | set(
        hyenadna_df["environment"].tolist()
    )
    hue_order = [e for e in ENV_ORDER if e in all_envs] + sorted(
        all_envs - set(ENV_ORDER)
    )
    palette = {e: ENV_COLOR_MAP.get(e, "#7F7F7F") for e in all_envs}

    for ax, df, title in [
        (axes[0], hgb_df, "HGB (traditional ML)"),
        (axes[1], hyenadna_df, "HyenaDNA (deep learning)"),
    ]:
        mae = df["residual"].abs().mean()
        sns.scatterplot(
            data=df,
            x="actual_pH",
            y="residual",
            hue="environment",
            hue_order=hue_order,
            palette=palette,
            ax=ax,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.3,
            s=60,
            legend=True,
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=1, zorder=0)
        ax.set_xlabel("Actual pH")
        ax.set_ylabel("Residual pH (actual − predicted)")
        ax.set_title(title)
        ax.text(
            0.98,
            0.98,
            f"MAE = {mae:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none"),
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Residual pH (actual − predicted)")
    fig.suptitle("Residual vs actual pH by environment", fontsize=12, y=1.02)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        out_path = out_dir / f"residual_vs_actual.{ext}"
        fig.savefig(out_path, dpi=150 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot residual vs actual pH for HGB and HyenaDNA (Seaborn)."
    )
    parser.add_argument(
        "--hgb-csv",
        type=str,
        default="results/sklearn/hgb_test_predictions.csv",
        help="Path to HGB test predictions CSV (default: results/sklearn/hgb_test_predictions.csv)",
    )
    parser.add_argument(
        "--hyenadna-csv",
        type=str,
        default="results/hyenadna/test_predictions.csv",
        help="Path to HyenaDNA test predictions CSV (default: results/hyenadna/test_predictions.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
        help="Directory for output figures (default: results/figures)",
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default="jedick/microbial-DNA-pH",
        help="Hugging Face dataset for environment labels (default: jedick/microbial-DNA-pH)",
    )
    args = parser.parse_args()

    hgb_path = Path(args.hgb_csv)
    hyenadna_path = Path(args.hyenadna_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not hgb_path.exists():
        raise FileNotFoundError(f"HGB predictions not found: {hgb_path}")
    if not hyenadna_path.exists():
        raise FileNotFoundError(
            f"HyenaDNA predictions not found: {hyenadna_path}. "
            "Run: python test_hyenadna.py --checkpoint <path> --output results/hyenadna/test_predictions.csv"
        )

    print(f"Loading environment labels from {args.dataset_repo}...")
    env_df = load_environment_from_hf(args.dataset_repo)

    hgb_df = load_hgb_data(hgb_path)
    hgb_df = hgb_df.merge(
        env_df[["sample_id", "environment"]], on="sample_id", how="left"
    ).fillna({"environment": "unknown"})
    hgb_df = hgb_df[["environment", "actual_pH", "residual", "model"]]

    hyenadna_df = load_hyenadna_data(hyenadna_path)
    hyenadna_df = hyenadna_df.merge(
        env_df[["sample_id", "environment"]], on="sample_id", how="left"
    ).fillna({"environment": "unknown"})
    hyenadna_df = hyenadna_df[["environment", "actual_pH", "residual", "model"]]

    plot_seaborn(hgb_df, hyenadna_df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
