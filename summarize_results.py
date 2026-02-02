#!/usr/bin/env python3
"""
Summarize training results: HyenaDNA experiments and comparison with traditional ML.

Reads config.json, training_log.json, and test_predictions.csv from each
results/hyenadna/expt* directory and builds a markdown table of hyperparameters
and metrics. For the best HyenaDNA model (lowest test MAE), runs plot_residuals.py
to compare with HGB and can add a link to the figure in the README Results section.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def find_expt_dirs(results_dir: Path) -> List[Path]:
    """Return sorted list of expt* subdirectories in results/hyenadna."""
    hyenadna_dir = results_dir / "hyenadna"
    if not hyenadna_dir.is_dir():
        return []
    dirs = [
        d for d in hyenadna_dir.iterdir() if d.is_dir() and d.name.startswith("expt")
    ]
    return sorted(dirs, key=lambda p: p.name)


def load_config(expt_dir: Path) -> Optional[Dict[str, Any]]:
    """Load config.json; return None if missing or invalid."""
    path = expt_dir / "config.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_best_val_loss(expt_dir: Path) -> Optional[float]:
    """Return best (minimum) validation loss from training_log.json."""
    path = expt_dir / "training_log.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            log = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(log, list) or not log:
        return None
    losses = [e.get("loss") for e in log if isinstance(e, dict) and "loss" in e]
    return min(losses) if losses else None


def compute_test_mae(expt_dir: Path) -> Optional[float]:
    """Compute test MAE from test_predictions.csv (true vs predicted)."""
    path = expt_dir / "test_predictions.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return None
    # HyenaDNA test_hyenadna.py writes true_ph and predicted_ph_mean
    true_col = "true_ph" if "true_ph" in df.columns else "true_pH"
    pred_col = (
        "predicted_ph_mean" if "predicted_ph_mean" in df.columns else "predicted_pH"
    )
    if true_col not in df.columns or pred_col not in df.columns:
        return None
    valid = df[true_col].notna() & df[pred_col].notna()
    if valid.sum() == 0:
        return None
    return float((df.loc[valid, true_col] - df.loc[valid, pred_col]).abs().mean())


def collect_row(expt_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Build one table row: hyperparameters and metrics."""
    best_val = get_best_val_loss(expt_dir)
    test_mae = compute_test_mae(expt_dir)
    return {
        "expt": expt_dir.name,
        "expt_dir": expt_dir,
        "batch_size": config.get("batch_size", ""),
        "num_epochs": config.get("num_epochs", ""),
        "head_architecture": config.get("head_architecture", ""),
        "pooling_mode": config.get("pooling_mode", ""),
        "best_val_loss": best_val,
        "test_mae": test_mae,
    }


def format_cell(value: Any) -> str:
    """Format a table cell for markdown (empty string for None)."""
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def table_to_markdown(rows: List[Dict[str, Any]]) -> str:
    """Convert list of row dicts to a markdown table. Lowest test MAE is bold."""
    if not rows:
        return (
            "*No HyenaDNA experiment directories found under `results/hyenadna/expt*`.*"
        )
    headers = [
        "Experiment",
        "Batch size",
        "Epochs",
        "Head",
        "Pooling",
        "Best val loss",
        "Test MAE",
    ]
    keys = [
        "expt",
        "batch_size",
        "num_epochs",
        "head_architecture",
        "pooling_mode",
        "best_val_loss",
        "test_mae",
    ]
    valid_maes = [r.get("test_mae") for r in rows if r.get("test_mae") is not None]
    min_mae = min(valid_maes) if valid_maes else None
    row_cells: List[List[str]] = []
    for r in rows:
        cells = []
        for k in keys:
            cell = format_cell(r.get(k))
            if k == "test_mae" and r.get("test_mae") == min_mae and min_mae is not None:
                cell = f"**{cell}**"
            cells.append(cell)
        row_cells.append(cells)
    col_widths = [len(h) for h in headers]
    for cells in row_cells:
        for i, c in enumerate(cells):
            col_widths[i] = max(col_widths[i], len(c))
    lines = []
    sep = (
        "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    )
    lines.append(sep)
    lines.append("| " + " | ".join("-" * max(3, w) for w in col_widths) + " |")
    for cells in row_cells:
        lines.append(
            "| "
            + " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cells))
            + " |"
        )
    return "\n".join(lines)


def find_best_expt_dir(rows: List[Dict[str, Any]]) -> Optional[Path]:
    """Return expt_dir for the row with lowest test MAE, or None."""
    valid = [r for r in rows if r.get("test_mae") is not None]
    if not valid:
        return None
    best = min(valid, key=lambda r: r["test_mae"])
    return best.get("expt_dir")


def run_plot_residuals(
    hyenadna_csv: Path,
    hgb_csv: Path,
    figures_dir: Path,
    project_root: Path,
) -> bool:
    """Run plot_residuals.py; return True on success."""
    script = project_root / "plot_residuals.py"
    if not script.exists():
        print(f"Warning: {script} not found, skipping residual plot.", file=sys.stderr)
        return False
    if not hgb_csv.exists():
        print(
            f"Warning: HGB CSV not found at {hgb_csv}, skipping residual plot.",
            file=sys.stderr,
        )
        return False
    if not hyenadna_csv.exists():
        print(
            f"Warning: HyenaDNA CSV not found at {hyenadna_csv}, skipping residual plot.",
            file=sys.stderr,
        )
        return False
    figures_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "--hgb-csv",
        str(hgb_csv),
        "--hyenadna-csv",
        str(hyenadna_csv),
        "--output-dir",
        str(figures_dir),
    ]
    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode == 0


def build_markdown_section(
    table: str,
    figure_relative_path: Optional[str] = None,
) -> str:
    """Build the full Results section markdown."""
    regenerate = "Regenerate with `python summarize_results.py --update-readme`."
    body = (
        "### HyenaDNA training\n\n"
        "Summary of experiments in `results/hyenadna/expt*`: hyperparameters from "
        "`config.json`, best validation loss from `training_log.json`, and test MAE "
        "computed from `test_predictions.csv`. " + regenerate + "\n\n"
        f"{table}\n"
    )
    if figure_relative_path:
        body += (
            "\nComparison of best HyenaDNA model (lowest test MAE) with HGB (traditional ML):\n\n"
            f"![Residual vs actual pH]({figure_relative_path})\n"
        )
    return "## Results\n\n" + body


def update_readme(readme_path: Path, section_md: str) -> None:
    """Insert or replace the Results section in the README."""
    text = readme_path.read_text()
    marker = "## Results"
    if marker in text:
        # Replace from "## Results" to the next "## " or end of file
        start = text.index(marker)
        rest = text[start + len(marker) :]
        next_hr = rest.find("\n## ")
        if next_hr != -1:
            end = start + len(marker) + next_hr
        else:
            end = len(text)
        new_text = (
            text[:start]
            + section_md.rstrip()
            + ("\n" if end < len(text) else "")
            + text[end:]
        )
    else:
        # Append before end of file
        new_text = text.rstrip() + "\n\n" + section_md.rstrip() + "\n"
    readme_path.write_text(new_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize results: HyenaDNA experiments table and optional residual plot + README update."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results root directory (default: results)",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Update README with Results section (table and, if generated, figure link)",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help="Path to README (default: README.md)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not run plot_residuals.py for the best model (skip figure generation)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    results_dir = (
        args.results_dir
        if args.results_dir.is_absolute()
        else (project_root / args.results_dir)
    )
    results_dir = results_dir.resolve()
    hgb_csv = results_dir / "sklearn" / "hgb_test_predictions.csv"
    figures_dir = results_dir / "figures"

    expt_dirs = find_expt_dirs(results_dir)
    rows: List[Dict[str, Any]] = []
    for expt_dir in expt_dirs:
        config = load_config(expt_dir)
        if config is None:
            continue
        rows.append(collect_row(expt_dir, config))

    table = table_to_markdown(rows)

    figure_relative_path: Optional[str] = None
    if args.update_readme and not args.no_plot and rows:
        best_dir = find_best_expt_dir(rows)
        if best_dir is not None:
            hyenadna_csv = best_dir / "test_predictions.csv"
            if run_plot_residuals(hyenadna_csv, hgb_csv, figures_dir, project_root):
                # Path for README: relative to repo root
                figure_relative_path = (
                    str(args.results_dir).replace("\\", "/")
                    + "/figures/residual_vs_actual.png"
                )

    section_md = build_markdown_section(
        table, figure_relative_path=figure_relative_path
    )

    print(section_md)

    if args.update_readme:
        if not args.readme.exists():
            raise SystemExit(f"README not found: {args.readme}")
        readme_path = (
            args.readme if args.readme.is_absolute() else (project_root / args.readme)
        )
        update_readme(readme_path, section_md)
        print(f"Updated {readme_path} with Results section.")


if __name__ == "__main__":
    main()
