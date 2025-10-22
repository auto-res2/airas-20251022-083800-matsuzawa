import argparse
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import yaml
from scipy.stats import mannwhitneyu

# --------------------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_learning_curve(history_df: pd.DataFrame, run_id: str, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    x = history_df.get("cumulative_time", history_df.index)
    for col in ["best_so_far", "val_acc", "epoch_train_acc"]:
        if col in history_df.columns:
            plt.plot(x, history_df[col], label=col)
    plt.xlabel("Cumulative Time (s)" if "cumulative_time" in history_df.columns else "Event Index")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curves – {run_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, run_id: str, out_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix – {run_id}")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def plot_box(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="method", y=metric, data=df, palette="Set2")
    plt.ylabel(metric)
    plt.title(f"Distribution of {metric} per method")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def plot_improvement_bar(improv_dict: Dict[str, float], metric: str, out_path: Path) -> None:
    methods = list(improv_dict.keys())
    values = [improv_dict[m] * 100 for m in methods]  # percent
    plt.figure(figsize=(6, 4))
    sns.barplot(x=methods, y=values, palette="Set3")
    plt.ylabel(f"Improvement over baseline (%) – {metric}")
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


# --------------------------------------------------------------------------------------
# Statistics & Aggregation helpers
# --------------------------------------------------------------------------------------

def run_significance_tests(df: pd.DataFrame, out_dir: Path) -> Path:
    methods = df["method"].unique().tolist()
    metrics = [c for c in df.columns if c not in {"run_id", "method"}]
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m1, m2 in combinations(methods, 2):
        key = f"{m1}_vs_{m2}"
        results[key] = {}
        g1, g2 = df[df["method"] == m1], df[df["method"] == m2]
        for metric in metrics:
            x, y = g1[metric].dropna().values, g2[metric].dropna().values
            if len(x) == 0 or len(y) == 0:
                statistic, p_val = float("nan"), float("nan")
            else:
                statistic, p_val = mannwhitneyu(x, y, alternative="two-sided")
            results[key][metric] = {"statistic": float(statistic), "p_value": float(p_val)}
    out_path = ensure_dir(out_dir) / "aggregated_significance_tests.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(out_path)
    return out_path


def compute_improvement_rates(df: pd.DataFrame, maximise_metrics: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute improvement rates of each method over baseline (first method in alphabetical order)."""
    baseline_method = None
    for name in ["baseline", "comparative-1", "boil", "BOIL"]:
        if name in df["method"].unique():
            baseline_method = name
            break
    if baseline_method is None:
        baseline_method = sorted(df["method"].unique())[0]
    baseline_values = df[df["method"] == baseline_method]
    improvs: Dict[str, Dict[str, float]] = {}
    for method in df["method"].unique():
        if method == baseline_method:
            continue
        improvs[method] = {}
        target_values = df[df["method"] == method]
        for metric in maximise_metrics:
            base_mean = baseline_values[metric].mean()
            tgt_mean = target_values[metric].mean()
            if base_mean == 0:
                rate = float("nan")
            else:
                rate = (tgt_mean - base_mean) / base_mean
            improvs[method][metric] = rate
        # time_to_85 is a minimisation metric
        if "time_to_85" in df.columns:
            base_mean = baseline_values["time_to_85"].mean()
            tgt_mean = target_values["time_to_85"].mean()
            rate = (base_mean - tgt_mean) / base_mean if base_mean != 0 else float("nan")
            improvs[method]["time_to_85"] = rate
    return improvs


def aggregate_and_compare(all_results: Dict[str, Dict], out_dir: Path) -> None:
    records: List[Dict] = []
    for run_id, data in all_results.items():
        summary = data["summary"]
        config = data["config"]
        records.append({
            "run_id": run_id,
            "method": config.get("method", "unknown"),
            "best_val_acc": summary.get("final_best_val_acc", np.nan),
            "auc": summary.get("auc_accuracy", np.nan),
            "time_to_85": summary.get("time_to_85", np.nan),
        })

    df = pd.DataFrame.from_records(records)
    out_dir = ensure_dir(out_dir)
    df.to_json(out_dir / "aggregated_metrics.json", orient="records", indent=2)

    # ------------------- box-plots for each metric -------------------
    for metric in ["best_val_acc", "auc", "time_to_85"]:
        if metric not in df.columns:
            continue
        box_path = out_dir / f"comparison_{metric}_boxplot.pdf"
        plot_box(df, metric, box_path)
        print(box_path)

    # ------------------- improvement rates ---------------------------
    maximize_metrics = ["best_val_acc", "auc"]
    improv_rates = compute_improvement_rates(df, maximize_metrics)
    improv_json = out_dir / "aggregated_improvement_rates.json"
    with open(improv_json, "w", encoding="utf-8") as fp:
        json.dump(improv_rates, fp, indent=2)
    print(improv_json)

    for metric in maximize_metrics + ["time_to_85"]:
        bar_path = out_dir / f"comparison_improvement_{metric}.pdf"
        flat = {m: vals.get(metric, np.nan) for m, vals in improv_rates.items()}
        if len(flat) == 0:
            continue
        plot_improvement_bar(flat, metric, bar_path)
        print(bar_path)

    # ------------------- simple mean bar for best acc ----------------
    plt.figure(figsize=(8, 4))
    sns.barplot(x="run_id", y="best_val_acc", hue="method", data=df, dodge=False)
    plt.xticks(rotation=45, ha="right")
    for i, v in enumerate(df["best_val_acc"].values):
        plt.text(i, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    plt.ylabel("Best Validation Accuracy")
    plt.tight_layout()
    acc_bar = out_dir / "comparison_best_accuracy_bar_chart.pdf"
    plt.savefig(acc_bar, format="pdf")
    plt.close()
    print(acc_bar)

    # ------------------- statistical tests ---------------------------
    run_significance_tests(df, out_dir)


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str, help="JSON string list of run IDs to evaluate")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    run_ids: List[str] = json.loads(args.run_ids)

    with open(results_dir / "config.yaml", "r", encoding="utf-8") as fp:
        wandb_cfg = yaml.safe_load(fp)
    entity, project = wandb_cfg.get("entity"), wandb_cfg.get("project")

    api = wandb.Api()
    all_results: Dict[str, Dict] = {}

    for run_id in run_ids:
        run_save_dir = ensure_dir(results_dir / run_id)
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history()
        summary = run.summary._json_dict
        config = dict(run.config)

        # ------------------- export history --------------------------
        hist_path = run_save_dir / "metrics.json"
        history.to_json(hist_path, orient="records", indent=2)
        print(hist_path)

        # ------------------- figures --------------------------------
        lc_path = run_save_dir / f"{run_id}_learning_curve.pdf"
        plot_learning_curve(history, run_id, lc_path)
        print(lc_path)

        if "confusion_matrix" in summary:
            cm_arr = np.asarray(summary["confusion_matrix"])
            cm_path = run_save_dir / f"{run_id}_confusion_matrix.pdf"
            plot_confusion_matrix(cm_arr, run_id, cm_path)
            print(cm_path)

        all_results[run_id] = {"history": history, "summary": summary, "config": config}

    aggregate_and_compare(all_results, results_dir / "comparison")


if __name__ == "__main__":
    main()