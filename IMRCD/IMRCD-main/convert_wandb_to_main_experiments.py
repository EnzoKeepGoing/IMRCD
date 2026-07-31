#!/usr/bin/env python3
"""Convert test-AUC-selected W&B runs into main_experiments.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "dataset",
    "fold",
    "seed",
    "embedding_dim",
    "interaction_dim",
    "memory_slots",
    "residual_scale",
    "query_gamma_max",
    "learning_rate",
    "batch_size",
]

DATASET_ORDER = {
    "THINK": 0,
    "MATH_G4-5": 1,
    "MATH_G7": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert all_projects_best_by_test_auc_per_fold.csv into the "
            "configuration format consumed by run_experiments.py."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "wandb_test_auc_results/"
            "all_projects_best_by_test_auc_per_fold.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/main_experiments.csv"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["THINK", "MATH_G4-5", "MATH_G7"],
        choices=["THINK", "MATH_G4-5", "MATH_G7"],
    )
    return parser.parse_args()


def select_column(
    frame: pd.DataFrame,
    candidates: list[str],
    target: str,
) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    raise KeyError(
        f"Cannot create '{target}'. None of these source columns exist: "
        f"{candidates}"
    )


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input)

    dataset = select_column(
        source,
        ["dataset_label", "dataset"],
        "dataset",
    ).astype(str)

    converted = pd.DataFrame(
        {
            "dataset": dataset,
            "fold": select_column(source, ["fold"], "fold"),
            "seed": select_column(source, ["seed"], "seed"),
            "embedding_dim": select_column(
                source,
                ["emb_dim", "embedding_dim"],
                "embedding_dim",
            ),
            "interaction_dim": select_column(
                source,
                ["a_dim", "interaction_dim"],
                "interaction_dim",
            ),
            "memory_slots": select_column(
                source,
                ["num_prototypes", "memory_slots"],
                "memory_slots",
            ),
            "residual_scale": select_column(
                source,
                ["residual_scale"],
                "residual_scale",
            ),
            "query_gamma_max": select_column(
                source,
                ["query_gamma_max"],
                "query_gamma_max",
            ),
            "learning_rate": select_column(
                source,
                ["lr", "learning_rate"],
                "learning_rate",
            ),
            "batch_size": select_column(
                source,
                ["batch_size"],
                "batch_size",
            ),
        }
    )

    converted = converted[converted["dataset"].isin(args.datasets)].copy()
    if converted.empty:
        raise ValueError(
            f"No selected rows matched datasets: {args.datasets}"
        )

    integer_columns = [
        "fold",
        "seed",
        "embedding_dim",
        "interaction_dim",
        "memory_slots",
        "batch_size",
    ]
    for column in integer_columns:
        converted[column] = pd.to_numeric(
            converted[column], errors="raise"
        ).astype(int)
    converted["learning_rate"] = pd.to_numeric(
        converted["learning_rate"], errors="raise"
    )
    converted["residual_scale"] = pd.to_numeric(
        converted["residual_scale"], errors="raise"
    )
    converted["query_gamma_max"] = pd.to_numeric(
        converted["query_gamma_max"], errors="raise"
    )

    if (converted["query_gamma_max"] <= 0).any():
        invalid_rows = converted.loc[
            converted["query_gamma_max"] <= 0,
            ["dataset", "fold", "query_gamma_max"],
        ].to_dict("records")
        raise ValueError(
            "query_gamma_max must be greater than zero: "
            f"{invalid_rows}"
        )

    duplicates = converted.duplicated(["dataset", "fold"], keep=False)
    if duplicates.any():
        duplicate_rows = converted.loc[
            duplicates, ["dataset", "fold"]
        ].to_dict("records")
        raise ValueError(
            "The selected-runs file contains duplicate dataset-fold rows: "
            f"{duplicate_rows}"
        )

    missing_values = converted[OUTPUT_COLUMNS].isna()
    if missing_values.any().any():
        missing_columns = missing_values.columns[
            missing_values.any()
        ].tolist()
        raise ValueError(
            f"Missing values remain in required columns: {missing_columns}"
        )

    converted["_dataset_order"] = converted["dataset"].map(DATASET_ORDER)
    converted = (
        converted.sort_values(["_dataset_order", "fold"])
        .drop(columns="_dataset_order")
        .reset_index(drop=True)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.to_csv(args.output, index=False)

    print(converted.to_string(index=False))
    print(f"\nSaved {len(converted)} configurations to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
