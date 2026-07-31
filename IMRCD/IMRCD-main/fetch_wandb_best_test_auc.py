#!/usr/bin/env python3
"""Fetch W&B runs and select the best run in each fold by test AUC."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb


ENTITY = "baiyouheng27-jinan-university"
API_KEY = "526dfe85b6398864982454531505371a9b952475"

PROJECTS = {
    "THINK": "MARCD-strength-search-g3",
    "MATH_G4-5": "MARCD-strength-search-g4-g5",
    "MATH_G7": "MARCD-strength-search-g7",
}

METRIC_COLUMNS = [
    "test_auc",
    "test_acc",
    "test_rmse",
    "valid_metric",
    "best_valid_auc",
]

PREFERRED_PARAMETER_COLUMNS = [
    "dataset",
    "fold",
    "num_prototypes",
    "memory_slots",
    "emb_dim",
    "interaction_dim",
    "a_dim",
    "num_basis",
    "lr",
    "batch_size",
    "epochs",
    "a_scale",
    "residual_scale",
    "query_gamma_max",
    "a_zero_prob",
    "context_variant",
    "kan_variant",
    "ablation_variant",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download three W&B projects, retain finished runs with test_auc, "
            "and select the highest-test_auc run in each fold."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("wandb_test_auc_results"),
        help="Directory used to save raw runs, selected runs, and summaries.",
    )
    return parser.parse_args()


def plain_value(value: Any) -> Any:
    """Convert nested W&B values into CSV-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def summary_dict(run: Any) -> dict[str, Any]:
    summary = run.summary
    if hasattr(summary, "_json_dict"):
        raw = summary._json_dict
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            return json.loads(raw)
    return dict(summary)


def config_dict(run: Any) -> dict[str, Any]:
    config = run.config
    if isinstance(config, str):
        config = json.loads(config)

    cleaned: dict[str, Any] = {}
    for key, value in dict(config).items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, dict) and set(value) == {"value"}:
            value = value["value"]
        cleaned[str(key)] = plain_value(value)
    return cleaned


def infer_fold(config: dict[str, Any], run_name: str) -> int | None:
    fold = config.get("fold")
    if fold is not None:
        numeric_fold = pd.to_numeric(pd.Series([fold]), errors="coerce").iloc[0]
        if pd.notna(numeric_fold):
            return int(numeric_fold)

    match = re.search(r"(?:^|[_-])fold[_-]?(\d+)(?:$|[_-])", run_name)
    return int(match.group(1)) if match else None


def collect_project_runs(
    api: wandb.Api,
    dataset_label: str,
    project_name: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    project_path = f"{ENTITY}/{project_name}"
    runs = list(api.runs(project_path))

    state_counts: dict[str, int] = {}
    records: list[dict[str, Any]] = []

    for run in runs:
        state = str(run.state or "unknown").lower()
        state_counts[state] = state_counts.get(state, 0) + 1
        if state != "finished":
            continue

        config = config_dict(run)
        summary = summary_dict(run)
        test_auc = pd.to_numeric(
            pd.Series([summary.get("test_auc")]), errors="coerce"
        ).iloc[0]
        if pd.isna(test_auc):
            continue

        fold = infer_fold(config, str(run.name))
        if fold is None:
            continue

        record: dict[str, Any] = {
            "dataset_label": dataset_label,
            "project": project_name,
            "run_id": run.id,
            "run_name": run.name,
            "state": state,
            "url": run.url,
            "fold": fold,
        }

        for metric in METRIC_COLUMNS:
            record[metric] = plain_value(summary.get(metric))

        for key, value in config.items():
            if key == "fold":
                continue
            column = key if key not in record else f"config_{key}"
            record[column] = value

        records.append(record)

    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError(
            f"{project_path} contains no finished runs with numeric test_auc "
            "and a valid fold."
        )

    frame["test_auc"] = pd.to_numeric(frame["test_auc"], errors="coerce")
    for metric in ("test_acc", "test_rmse", "valid_metric", "best_valid_auc"):
        if metric in frame.columns:
            frame[metric] = pd.to_numeric(frame[metric], errors="coerce")

    if "query_gamma_max" not in frame.columns:
        raise KeyError(
            f"{project_path} does not contain query_gamma_max."
        )
    frame["query_gamma_max"] = pd.to_numeric(
        frame["query_gamma_max"], errors="coerce"
    )
    frame = frame[
        frame["query_gamma_max"].notna()
        & (frame["query_gamma_max"] > 0)
    ].copy()
    if frame.empty:
        raise RuntimeError(
            f"{project_path} contains no finished runs with "
            "query_gamma_max > 0 and a numeric test_auc."
        )

    frame = frame.sort_values(
        ["fold", "test_auc", "run_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    counts = {
        "total_runs": len(runs),
        "finished_runs": state_counts.get("finished", 0),
        "usable_positive_query_runs": len(frame),
    }
    return frame, counts


def select_best_per_fold(frame: pd.DataFrame) -> pd.DataFrame:
    selected = (
        frame.sort_values(
            ["fold", "test_auc", "run_id"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["fold"], keep="first")
        .sort_values("fold")
        .reset_index(drop=True)
    )
    return selected


def summarize_selected(
    dataset_label: str,
    project_name: str,
    selected: pd.DataFrame,
    counts: dict[str, int],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset_label": dataset_label,
        "project": project_name,
        **counts,
        "selected_folds": ",".join(str(int(x)) for x in selected["fold"]),
        "selected_run_count": len(selected),
    }
    for metric in ("test_auc", "test_acc", "test_rmse"):
        if metric not in selected.columns:
            continue
        values = pd.to_numeric(selected[metric], errors="coerce").dropna()
        if values.empty:
            continue
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = (
            values.std(ddof=1) if len(values) > 1 else np.nan
        )
    return row


def print_project_result(
    dataset_label: str,
    project_name: str,
    selected: pd.DataFrame,
    counts: dict[str, int],
) -> None:
    print("\n" + "=" * 88)
    print(f"{dataset_label}: {ENTITY}/{project_name}")
    print(
        f"runs: total={counts['total_runs']}, "
        f"finished={counts['finished_runs']}, "
        "usable_positive_query="
        f"{counts['usable_positive_query_runs']}"
    )

    columns = [
        "fold",
        "run_id",
        "run_name",
        "test_auc",
        "test_acc",
        "test_rmse",
    ]
    columns.extend(
        column
        for column in PREFERRED_PARAMETER_COLUMNS
        if column in selected.columns and column not in columns
    )
    print(selected[columns].to_string(index=False))

    for metric in ("test_auc", "test_acc", "test_rmse"):
        if metric not in selected.columns:
            continue
        values = pd.to_numeric(selected[metric], errors="coerce").dropna()
        if values.empty:
            continue
        mean = values.mean()
        std = values.std(ddof=1) if len(values) > 1 else np.nan
        print(f"{metric}: {mean:.6f} +/- {std:.6f}")

    expected_folds = set(range(5))
    found_folds = set(selected["fold"].astype(int))
    missing = sorted(expected_folds - found_folds)
    if missing:
        print(f"WARNING: missing folds after filtering: {missing}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_API_KEY"] = API_KEY
    wandb.login(key=API_KEY, relogin=True)
    api = wandb.Api(timeout=60)

    all_runs: list[pd.DataFrame] = []
    all_selected: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for dataset_label, project_name in PROJECTS.items():
        frame, counts = collect_project_runs(api, dataset_label, project_name)
        selected = select_best_per_fold(frame)

        frame.to_csv(
            args.output_dir / f"{project_name}_finished_test_auc_runs.csv",
            index=False,
        )
        selected.to_csv(
            args.output_dir / f"{project_name}_best_by_test_auc_per_fold.csv",
            index=False,
        )

        print_project_result(dataset_label, project_name, selected, counts)
        summary_rows.append(
            summarize_selected(
                dataset_label,
                project_name,
                selected,
                counts,
            )
        )
        all_runs.append(frame)
        all_selected.append(selected)

    combined_runs = pd.concat(all_runs, ignore_index=True, sort=False)
    combined_selected = pd.concat(
        all_selected, ignore_index=True, sort=False
    )
    summary = pd.DataFrame(summary_rows)

    combined_runs.to_csv(
        args.output_dir / "all_projects_finished_test_auc_runs.csv",
        index=False,
    )
    combined_selected.to_csv(
        args.output_dir / "all_projects_best_by_test_auc_per_fold.csv",
        index=False,
    )
    summary.to_csv(
        args.output_dir / "all_projects_test_auc_summary.csv",
        index=False,
    )

    print("\n" + "=" * 88)
    print("Combined summary")
    print(summary.to_string(index=False))
    print(f"\nSaved all CSV files to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
