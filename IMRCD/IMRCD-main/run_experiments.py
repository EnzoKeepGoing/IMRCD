import argparse
from argparse import Namespace
from pathlib import Path

import pandas as pd

from train_imrcd import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run IMRCD configurations from a CSV file"
    )
    parser.add_argument(
        "--config", default="configs/main_experiments.csv"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["THINK"],
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    arguments = parser.parse_args()

    configurations = pd.read_csv(arguments.config)
    required_columns = {
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
    }
    missing_columns = required_columns - set(configurations.columns)
    if missing_columns:
        raise ValueError(
            "The experiment configuration is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    configurations = configurations[
        configurations["dataset"].isin(arguments.datasets)
    ]
    if configurations.empty:
        raise ValueError("No experiment configuration matched --datasets")

    results = []
    for row in configurations.itertuples(index=False):
        experiment = Namespace(
            dataset=row.dataset,
            data_root=arguments.data_root,
            output_dir=arguments.output_dir,
            fold=int(row.fold),
            seed=int(row.seed),
            device=arguments.device,
            num_workers=arguments.num_workers,
            embedding_dim=int(row.embedding_dim),
            interaction_dim=int(row.interaction_dim),
            interaction_hidden_dim=64,
            interaction_heads=4,
            num_basis=3,
            memory_dim=32,
            memory_groups=4,
            memory_slots=int(row.memory_slots),
            residual_scale=float(row.residual_scale),
            query_gamma_max=float(row.query_gamma_max),
            learning_rate=float(row.learning_rate),
            batch_size=int(row.batch_size),
            epochs=arguments.epochs,
            patience=arguments.patience,
        )
        print(f"\nRunning {row.dataset}, fold {int(row.fold)}")
        results.append(run_experiment(experiment))

    summary_rows = [
        {
            "dataset": result["dataset"],
            "fold": result["fold"],
            "seed": result["seed"],
            "test_auc": result["test_auc"],
            "test_acc": result["test_acc"],
            "test_rmse": result["test_rmse"],
        }
        for result in results
    ]
    output_directory = Path(arguments.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    fold_results = pd.DataFrame(summary_rows)
    fold_output_path = output_directory / "fold_results.csv"
    fold_results.to_csv(fold_output_path, index=False)

    dataset_summary = (
        fold_results.groupby("dataset")
        .agg(
            folds=("fold", "count"),
            auc_mean=("test_auc", "mean"),
            auc_std=("test_auc", "std"),
            acc_mean=("test_acc", "mean"),
            acc_std=("test_acc", "std"),
            rmse_mean=("test_rmse", "mean"),
            rmse_std=("test_rmse", "std"),
        )
        .reset_index()
    )
    summary_output_path = output_directory / "dataset_summary.csv"
    dataset_summary.to_csv(summary_output_path, index=False)
    print(f"\nSaved fold results to {fold_output_path}")
    print(f"Saved dataset summary to {summary_output_path}")


if __name__ == "__main__":
    main()
